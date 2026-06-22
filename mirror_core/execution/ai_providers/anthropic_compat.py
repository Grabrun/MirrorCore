"""
Anthropic Messages API 适配器

B-T31: Anthropic 兼容适配器

使用 Anthropic Messages API 格式。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from mirror_core.execution.ai_providers import AIProvider, ChatResponse

logger = logging.getLogger("mirror_core.execution.ai_providers.anthropic_compat")

ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_DEFAULT_MAX_TOKENS = 8192
ANTHROPIC_DEFAULT_EMBEDDING_DIM = 1024
ANTHROPIC_TIMEOUT = 30
ANTHROPIC_API_VERSION = "2023-06-01"


class AnthropicCompatProvider(AIProvider):
    """
    Anthropic Messages API 适配器。

    Anthropic 不使用 OpenAI 格式，需单独实现消息转换。
    注意: Anthropic 没有官方 Embedding API，会回退到 mock 降级。
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = ANTHROPIC_DEFAULT_MODEL,
        max_tokens: int = ANTHROPIC_DEFAULT_MAX_TOKENS,
        embedding_dim: int = ANTHROPIC_DEFAULT_EMBEDDING_DIM,
    ):
        self._api_key = api_key
        self._model = model
        self._max_tokens_val = max_tokens
        self._embedding_dim = embedding_dim
        self._client = httpx.AsyncClient(
            base_url="https://api.anthropic.com/v1",
            timeout=httpx.Timeout(ANTHROPIC_TIMEOUT),
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_API_VERSION,
                "content-type": "application/json",
            },
        )

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def max_tokens(self) -> int:
        return self._max_tokens_val

    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        system, converted = self._convert_messages(messages)
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": converted,
            "max_tokens": kwargs.pop("max_tokens", self._max_tokens_val),
        }
        if tools:
            payload["tools"] = tools
        if system:
            payload["system"] = system
        payload.update(kwargs)

        resp = await self._client.post("/messages", json=payload)
        resp.raise_for_status()
        data = resp.json()

        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block["text"]

        usage = data.get("usage", {})
        return ChatResponse(
            content=content,
            model=data.get("model", self._model),
            usage={"prompt_tokens": usage.get("input_tokens", 0),
                   "completion_tokens": usage.get("output_tokens", 0)},
        )

    async def embed(self, text: str) -> List[float]:
        raise NotImplementedError(
            "Anthropic 无官方 Embedding API，请使用 OpenAI 兼容的 Embedding 服务"
        )

    async def close(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _convert_messages(
        messages: List[Dict[str, str]],
    ) -> tuple[Optional[str], list]:
        """将 OpenAI 格式转换为 Anthropic 格式。"""
        system_parts = []
        converted = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            elif msg["role"] == "user":
                converted.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                converted.append({"role": "assistant", "content": msg["content"]})
        return "\n\n".join(system_parts) if system_parts else None, converted
