"""
OpenAI 兼容适配器

B-T30: 实现 Provider 工厂与 OpenAI 兼容适配器

支持 OpenAI API 协议的服务（vLLM, Ollama, LocalAI 等）。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from mirror_core.execution.ai_providers import AIProvider, ChatResponse

logger = logging.getLogger("mirror_core.execution.ai_providers.openai_compat")

OPENAI_DEFAULT_MODEL = "gpt-4o"
OPENAI_DEFAULT_MAX_TOKENS = 8192
OPENAI_DEFAULT_EMBEDDING_DIM = 1536
OPENAI_TIMEOUT = 30


class OpenAICompatProvider(AIProvider):
    """
    OpenAI API 兼容适配器。

    可用于 OpenAI、DeepSeek (兼容模式)、vLLM、Ollama、LocalAI 等。
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = OPENAI_DEFAULT_MODEL,
        embed_model: str = "text-embedding-3-small",
        embedding_dim: int = OPENAI_DEFAULT_EMBEDDING_DIM,
        max_tokens: int = OPENAI_DEFAULT_MAX_TOKENS,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._embed_model = embed_model
        self._embedding_dim = embedding_dim
        self._max_tokens = max_tokens
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(OPENAI_TIMEOUT),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
        payload.update(kwargs)

        resp = await self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        content = choice.get("message", {}).get("content", "") or ""
        usage = data.get("usage", {})

        return ChatResponse(
            content=content,
            model=data.get("model", self._model),
            usage=usage,
        )

    async def embed(self, text: str) -> List[float]:
        resp = await self._client.post(
            "/embeddings",
            json={"model": self._embed_model, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]

    async def close(self) -> None:
        await self._client.aclose()


class DeepSeekCompatProvider(OpenAICompatProvider):
    """
    DeepSeek 官方 API（兼容 OpenAI 协议）。

    仅调整默认模型和 base_url。
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "deepseek-chat",
    ):
        super().__init__(
            base_url="https://api.deepseek.com/v1",
            api_key=api_key,
            model=model,
            embed_model="deepseek-embedding",
            embedding_dim=2048,
            max_tokens=8192,
        )
