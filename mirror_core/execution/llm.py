"""
文本生成器 (TextGenerator)

B-T32: TextGenerator，包含 LLM 调用、超时重试与降级逻辑
B-T33: post_process 方法，集成 Kaomoji 映射表与防火墙锚点注入

设计文档 §3.4.1:

LLM 调用流程:
    1. 调用 AI Provider chat()
    2. 超时 15s → 重试 (最多 2 次)
    3. 全部失败 → 降级返回"系统繁忙"
    4. API 429 → 读取 Retry-After 等待

post_process 流程:
    1. 根据情感状态选择颜文字 (Kaomoji)
    2. 调用 SafetyEngine.inject_reality_anchor() 注入锚点
"""

from __future__ import annotations

import asyncio
import logging
import random
import time as _time
from typing import Any, Dict, List, Optional

from mirror_core.core.safety import SafetyEngine
from mirror_core.emotion.engine import (
    EmotionalState,
    KAOMOJI_MAP,
    _meets_condition,
)
from mirror_core.execution.ai_providers import AIProvider, ChatResponse

logger = logging.getLogger("mirror_core.execution.llm")

# 降级回复
FALLBACK_REPLIES = [
    "（系统好像有点忙，请稍等一下...）",
    "我刚才好像走神了…我们换个话题好吗？",
    "（信号不太好，你能再说一遍吗？）",
]

# 重试配置
MAX_RETRIES = 2
TIMEOUT_SECONDS = 15
RETRY_DELAY_BASE = 1.0  # 指数退避基数（秒）


class TextGenerator:
    """
    文本生成器。

    封装 LLM 调用、超时重试、降级回复、后处理（颜文字+锚点注入）。
    """

    def __init__(
        self,
        provider: AIProvider,
        safety_engine: Optional[SafetyEngine] = None,
    ):
        """
        Args:
            provider: AI Provider 实例
            safety_engine: 安全引擎（用于锚点注入，可选）
        """
        self._provider = provider
        self._safety_engine = safety_engine

    @property
    def provider(self) -> AIProvider:
        return self._provider

    # ---- B-T32: LLM 调用 + 重试 + 降级 ----

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """
        生成回复，包含超时重试与降级逻辑。

        Args:
            messages: OpenAI Chat Completions 格式的消息列表
            **kwargs: 传递给 AI Provider 的额外参数

        Returns:
            回复文本（可能为降级回复）
        """
        last_error = ""

        for attempt in range(1, MAX_RETRIES + 2):  # 首次 + MAX_RETRIES 次重试
            try:
                response = await asyncio.wait_for(
                    self._provider.chat(messages, **kwargs),
                    timeout=TIMEOUT_SECONDS,
                )
                return response.content

            except asyncio.TimeoutError:
                last_error = f"超时 ({TIMEOUT_SECONDS}s)"
                logger.warning(
                    "LLM 调用超时 (尝试 %d/%d)",
                    attempt, MAX_RETRIES + 1,
                )
                if attempt <= MAX_RETRIES:
                    await self._backoff(attempt)

            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "LLM 调用失败 (尝试 %d/%d): %s",
                    attempt, MAX_RETRIES + 1, exc,
                )
                # 检查 429（速率限制）
                if self._is_rate_limit(exc):
                    retry_after = self._extract_retry_after(exc)
                    if retry_after:
                        await asyncio.sleep(min(retry_after, 30))
                elif attempt <= MAX_RETRIES:
                    await self._backoff(attempt)

        # 全部失败，降级
        logger.error("LLM 调用全部失败 (%d 次尝试), 使用降级回复: %s", MAX_RETRIES + 1, last_error)
        return random.choice(FALLBACK_REPLIES)

    async def _backoff(self, attempt: int) -> None:
        """指数退避等待。"""
        delay = RETRY_DELAY_BASE * (2 ** (attempt - 1))
        await asyncio.sleep(delay)

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        """判断是否为 HTTP 429 速率限制错误。"""
        exc_str = str(exc).lower()
        return "429" in exc_str or "rate limit" in exc_str or "too many requests" in exc_str

    @staticmethod
    def _extract_retry_after(exc: Exception) -> Optional[float]:
        """从异常中提取 Retry-After 值。"""
        exc_str = str(exc)
        # httpx 错误可能包含 Retry-After 头
        if "retry-after" in exc_str.lower():
            import re
            match = re.search(r"retry-after[=:]?\s*(\d+)", exc_str, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

    # ---- B-T33: 后处理 ----

    async def post_process(
        self,
        raw_text: str,
        emotion: Optional[EmotionalState] = None,
        dependency_score: float = 0.0,
    ) -> str:
        """
        对原始回复进行后处理。

        流程:
        1. 根据情感状态选择颜文字并追加
        2. 调用 SafetyEngine 注入现实锚点（高依赖度时）

        Args:
            raw_text: 原始回复文本
            emotion: 当前情感状态（可选）
            dependency_score: 当前依赖度评分（可选）

        Returns:
            后处理后的回复文本
        """
        result = raw_text

        # 1. 颜文字插入 (§3.4.1.2)
        if emotion:
            kaomoji = self._select_kaomoji(emotion)
            if kaomoji:
                result = f"{result} {kaomoji}"

        # 2. 现实锚点注入 (§3.4.1.3)
        if self._safety_engine:
            result = await self._safety_engine.inject_reality_anchor(
                response=result,
                dependency_score=dependency_score,
            )

        return result

    def _select_kaomoji(self, emotion: EmotionalState) -> str:
        """
        根据情感状态选择颜文字。

        使用 emotion.engine 中定义的 KAOMOJI_MAP 和 _meets_condition。

        Args:
            emotion: 当前情感状态

        Returns:
            匹配的颜文字（无匹配时返回空字符串）
        """
        for condition, emojis in KAOMOJI_MAP.items():
            if _meets_condition(condition, emotion):
                return random.choice(emojis)
        return ""
