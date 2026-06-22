"""
AI 提供商抽象接口

B-T29: AIProvider 抽象接口 (chat / embed / embedding_dim / max_tokens)

设计文档 §3.4.2:
- chat(): 发送对话请求，返回文本回复
- embed(): 将文本向量化，返回 float 列表
- embedding_dim: 向量维度
- max_tokens: 模型最大 Token 数
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChatResponse:
    """AI 聊天响应"""
    content: str
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)


class AIProvider(ABC):
    """AI 提供商抽象基类"""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """向量维度"""
        ...

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """模型最大 Token 数"""
        ...

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """
        发送对话请求。

        Args:
            messages: OpenAI Chat Completions 格式的消息列表
            tools: 工具定义（可选）
            **kwargs: 额外参数（temperature, max_tokens 等）

        Returns:
            ChatResponse
        """
        ...

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        将文本向量化。

        Args:
            text: 输入文本

        Returns:
            浮点数向量列表
        """
        ...
