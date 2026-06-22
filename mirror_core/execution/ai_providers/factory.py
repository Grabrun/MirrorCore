"""
AI 提供商工厂

B-T30: Provider 工厂方法

根据配置动态创建不同的 AI Provider 实例。
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from mirror_core.execution.ai_providers import AIProvider

logger = logging.getLogger("mirror_core.execution.ai_providers.factory")


def create_provider(config: Dict[str, Any]) -> AIProvider:
    """
    工厂方法：根据配置字典动态创建 AI Provider 实例。

    Args:
        config: 配置字典，必须包含 type 字段
            type: "openai-compat" | "deepseek" | "anthropic" | "glm"
            其余字段为对应提供商的参数

    Returns:
        AIProvider 实例

    Raises:
        ValueError: 未知 provider 类型
    """
    provider_type = config.get("type", "").lower().strip()

    if provider_type == "openai-compat":
        from mirror_core.execution.ai_providers.openai_compat import (
            OpenAICompatProvider,
        )
        return OpenAICompatProvider(
            base_url=config.get("base_url", "https://api.openai.com/v1"),
            api_key=config.get("api_key", ""),
            model=config.get("model", "gpt-4o"),
            embed_model=config.get("embed_model", "text-embedding-3-small"),
            embedding_dim=config.get("embedding_dim", 1536),
            max_tokens=config.get("max_tokens", 8192),
        )

    if provider_type == "deepseek":
        from mirror_core.execution.ai_providers.openai_compat import (
            DeepSeekCompatProvider,
        )
        return DeepSeekCompatProvider(
            api_key=config.get("api_key", ""),
            model=config.get("model", "deepseek-chat"),
        )

    if provider_type == "anthropic":
        from mirror_core.execution.ai_providers.anthropic_compat import (
            AnthropicCompatProvider,
        )
        return AnthropicCompatProvider(
            api_key=config.get("api_key", ""),
            model=config.get("model", "claude-sonnet-4-20250514"),
            max_tokens=config.get("max_tokens", 8192),
            embedding_dim=config.get("embedding_dim", 1024),
        )

    if provider_type == "glm":
        from mirror_core.execution.ai_providers.glm import GLMProvider
        return GLMProvider(
            api_key=config.get("api_key", ""),
            model=config.get("model", "glm-4-plus"),
            max_tokens=config.get("max_tokens", 8192),
            embedding_dim=config.get("embedding_dim", 1024),
        )

    raise ValueError(
        f"未知的 AI Provider 类型: '{provider_type}'。"
        f"可用类型: openai-compat, deepseek, anthropic, glm"
    )
