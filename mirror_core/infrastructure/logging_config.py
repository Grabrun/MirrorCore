"""
结构化日志配置 (D-T02)

集成 structlog，配置全系统 JSON 日志格式、trace_id 绑定及敏感信息过滤器。

设计文档 §4.1:
- 所有日志以 JSON 格式输出到 stdout
- trace_id（会话ID）贯穿调用链
- 用户原文默认不记录，记录时需哈希脱敏
"""

from __future__ import annotations

import hashlib
import logging
import sys
from typing import Any, Dict

import structlog


def hash_preview(text: str, length: int = 8) -> str:
    """
    返回输入文本的 MD5 摘要（日志脱敏用途）。

    Args:
        text: 原始文本
        length: 摘要保留长度 (默认 8)

    Returns:
        哈希摘要字符串
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


def _sensitive_field_processor(
    logger: logging.Logger,
    method_name: str,
    event_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    structlog 处理器：自动脱敏敏感字段。

    对包含 key/secret/password/token 等后缀的字段值进行哈希处理。
    """
    sensitive_suffixes = ("key", "secret", "password", "token", "api_key")
    for key in list(event_dict.keys()):
        if isinstance(event_dict[key], str):
            key_lower = key.lower()
            if any(suffix in key_lower for suffix in sensitive_suffixes):
                event_dict[key] = hash_preview(event_dict[key])
            if key_lower == "user_text" or key_lower == "message":
                event_dict[key] = hash_preview(event_dict[key])
    return event_dict


def setup_logging(
    log_level: str = "INFO",
    json_format: bool = True,
) -> None:
    """
    配置全系统日志。

    Args:
        log_level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        json_format: 是否使用 JSON 格式 (False 时用彩色控制台)
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso")

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        _sensitive_field_processor,
    ]

    if json_format:
        # JSON 格式输出 (生产环境)
        processors = shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ]
        formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.processors.JSONRenderer(),
            ],
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
    else:
        # 彩色控制台输出 (开发环境)
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(),
        ]
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(),
        ))

    # 配置 root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 配置 structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # 第三方库日志级别
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("watchdog").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


def bind_trace_id(trace_id: str) -> None:
    """
    绑定 trace_id 到当前上下文的 structlog 中。

    Args:
        trace_id: 会话/请求追踪 ID
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(trace_id=trace_id)
