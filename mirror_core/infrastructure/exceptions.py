"""
全局异常处理 + 降级策略 (D-T03 / D-T04)

D-T03: 全局异常捕获 + 会话中断降级回复
D-T04: 数据库操作失败指数退避重试装饰器 + AI 调用降级熔断
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time as _time
from typing import Any, Callable, Optional, Tuple, Type, Union

logger = logging.getLogger("mirror_core.infrastructure.exceptions")

# ===== 降级回复模板 (D-T03) =====
FALLBACK_REPLY = "刚才好像走神了…我们继续吧。"
FALLBACK_DEGRADED = "（系统有点忙，请稍等一下）"


# ===== D-T04: 指数退避重试装饰器 =====

def retry(
    max_retries: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 10.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """
    指数退避重试装饰器。

    用法:
        @retry(max_retries=3, base_delay=0.1)
        async def query_db(): ...

    Args:
        max_retries: 最大重试次数
        base_delay: 初始等待秒数
        max_delay: 最大等待秒数
        exceptions: 可重试的异常类型元组
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.debug(
                            "%s 失败 (尝试 %d/%d), %.2f秒后重试: %s",
                            func.__name__, attempt + 1, max_retries + 1,
                            delay, exc,
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.warning(
                            "%s 全部重试失败 (%d 次): %s",
                            func.__name__, max_retries + 1, exc,
                        )
            raise last_exc  # type: ignore
        return wrapper
    return decorator


# ===== D-T04: AI 调用降级熔断 =====

class CircuitBreaker:
    """
    熔断器 — 防止连续失败对下游造成压力。

    状态: CLOSED (正常) → OPEN (熔断) → HALF_OPEN (半开) → CLOSED

    用法:
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        async with cb:
            result = await call_ai_api()
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        fallback: str = "",
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._fallback = fallback
        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0

    @property
    def state(self) -> str:
        return self._state

    async def __aenter__(self) -> "CircuitBreaker":
        if self._state == self.OPEN:
            if _time.time() - self._last_failure_time >= self._recovery_timeout:
                self._state = self.HALF_OPEN
                logger.info("熔断器半开: 允许一次试探请求")
            else:
                raise CircuitBreakerOpenError(self._fallback)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Optional[bool]:
        if exc_type is not None:
            self._failure_count += 1
            self._last_failure_time = _time.time()
            if self._failure_count >= self._failure_threshold:
                self._state = self.OPEN
                logger.warning(
                    "熔断器打开: %d 次连续失败, 冷却 %ds",
                    self._failure_count, self._recovery_timeout,
                )
            if self._state == self.HALF_OPEN:
                self._state = self.OPEN
        else:
            self._failure_count = 0
            if self._state == self.HALF_OPEN:
                self._state = self.CLOSED
                logger.info("熔断器关闭: 试探请求成功")
            self._state = self.CLOSED
        return None

    def reset(self) -> None:
        """手动重置熔断器。"""
        self._state = self.CLOSED
        self._failure_count = 0


class CircuitBreakerOpenError(Exception):
    """熔断器打开时抛出的异常。"""

    def __init__(self, fallback: str = ""):
        self.fallback = fallback
        super().__init__(fallback or "服务暂时不可用")


# ===== D-T03: 通用降级回复 =====

def get_fallback_reply() -> str:
    """返回标准降级回复。"""
    return FALLBACK_REPLY


def get_degraded_reply() -> str:
    """返回降级繁忙提示。"""
    return FALLBACK_DEGRADED
