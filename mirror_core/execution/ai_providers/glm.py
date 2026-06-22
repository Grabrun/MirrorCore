"""
智谱 GLM 适配器

B-T31: GLM 预集成适配器

智谱 ChatGLM API 格式与 OpenAI 略有不同。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from mirror_core.execution.ai_providers import AIProvider, ChatResponse

logger = logging.getLogger("mirror_core.execution.ai_providers.glm")

GLM_DEFAULT_MODEL = "glm-4-plus"
GLM_DEFAULT_MAX_TOKENS = 8192
GLM_DEFAULT_EMBEDDING_DIM = 1024
GLM_TIMEOUT = 30


class GLMProvider(AIProvider):
    """
    智谱 GLM API 适配器。

    兼容 OpenAI 协议但端点格式不同，
    使用 /api/paas/v4/ 前缀。
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = GLM_DEFAULT_MODEL,
        max_tokens: int = GLM_DEFAULT_MAX_TOKENS,
        embedding_dim: int = GLM_DEFAULT_EMBEDDING_DIM,
    ):
        self._api_key = api_key
        self._model = model
        self._max_tokens_val = max_tokens
        self._embedding_dim = embedding_dim
        self._client = httpx.AsyncClient(
            base_url="https://open.bigmodel.cn/api/paas/v4",
            timeout=httpx.Timeout(GLM_TIMEOUT),
            headers={
                "Authorization": f"Bearer {api_key}",
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
            json={"model": "embedding-2", "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]

    async def close(self) -> None:
        await self._client.aclose()
