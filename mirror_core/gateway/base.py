"""
渠道适配器抽象基类与标准化数据模型

B-T03: 定义 ChannelAdapter 抽象基类、MessageContent 与 UserMessage 数据模型
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class MessageContent:
    """
    标准化的消息内容载荷

    由渠道适配器负责将不同格式的消息转换为统一结构。
    非文本渠道（TUI）下行时自动降级为标签文本。
    """
    text: Optional[str] = None
    image_path: Optional[str] = None
    mime_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_text_only(self) -> bool:
        """是否仅为纯文本消息（无图片附件）"""
        return self.image_path is None

    @property
    def display_text(self) -> str:
        """获取显示文本：有图片时附加标签"""
        if self.image_path:
            return f"{self.text or ''} [图片: {self.mime_type or 'unknown'}]"
        return self.text or ""


@dataclass
class UserMessage:
    """
    标准化后的用户消息数据模型

    由 Gateway 将各渠道原始消息统一转换为此格式后发布到事件总线。
    """
    platform: str
    platform_user_id: str
    internal_user_id: str
    session_id: str
    text: str
    timestamp: float


@dataclass
class RawMessage:
    """
    原始消息数据模型

    适配器接收渠道原始消息后构造此对象，传递给 Gateway.ingress()。
    raw_data 字段保留渠道特定的原始数据，供未来扩展。
    """
    platform: str
    platform_user_id: str
    text: str
    timestamp: float
    raw_data: Dict[str, Any] = field(default_factory=dict)


class ChannelAdapter(ABC):
    """
    渠道适配器抽象基类

    所有渠道适配器必须实现此接口。Gateway 通过此接口统一管理适配器生命周期。
    """

    @abstractmethod
    async def start(self) -> None:
        """
        启动适配器，建立与渠道的连接。

        启动失败应抛出异常并记录日志，Gateway 将尝试按配置重连。
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """
        停止适配器，断开与渠道的连接。

        应在关闭前刷新所有缓冲消息。
        """
        ...

    @abstractmethod
    async def send_message(self, target_id: str, content: MessageContent) -> bool:
        """
        发送消息到指定用户。

        Args:
            target_id: 渠道内的用户标识（如微信 openid、QQ user_id）
            content: 标准化消息内容（文本 + 可选图片）

        Returns:
            True 表示发送成功，False 表示发送失败（Gateway 将缓冲重试）
        """
        ...

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """返回渠道平台名称（小写英文，如 'wechat', 'qq', 'webchat', 'tui'）"""
        ...

    @property
    def status(self) -> str:
        """
        返回适配器当前状态。

        返回值: 'connected', 'connecting', 'disconnected', 'error'
        子类可覆盖以提供更精确的状态信息。
        """
        return "unknown"
