"""
统一网关核心

B-T04: 实现 Gateway 核心逻辑：消息标准化与会话解析、egress回复分发、适配器注册管理
B-T09: REST API: GET /api/v1/gateway/adapters 与 POST /api/v1/gateway/reload
"""

from __future__ import annotations

import logging
import time as _time
from typing import Dict, List, Optional

from mirror_core.bus import BaseEvent, EventBus, EventType
from mirror_core.gateway.base import (
    ChannelAdapter,
    MessageContent,
    RawMessage,
    UserMessage,
)
from mirror_core.gateway.session import SessionManager

logger = logging.getLogger("mirror_core.gateway.gateway")


class Gateway:
    """
    统一消息网关

    职责：
    - 管理渠道适配器的注册与生命周期
    - 将适配器传入的原始消息标准化为 UserMessage 事件发布到总线
    - 监听 REPLY_READY 事件，将回复路由回正确的适配器
    - 适配器断开时缓冲消息（最多 100 条/目标），重连后自动冲刷
    """

    def __init__(self, bus: EventBus, session_manager: SessionManager):
        self._bus = bus
        self._session_manager = session_manager
        self._adapters: Dict[str, ChannelAdapter] = {}
        self._message_buffer: Dict[str, List[MessageContent]] = {}
        self._started = False

    # ---- 生命周期 ----

    async def start(self) -> None:
        """启动网关：注册 REPLY_READY 订阅者，启动所有已注册的适配器"""
        if self._started:
            logger.warning("网关已启动，忽略重复调用")
            return

        self._bus.subscribe(EventType.REPLY_READY, self._on_reply_ready)
        self._bus.subscribe(EventType.SHUTDOWN, self._on_shutdown)

        for name, adapter in self._adapters.items():
            try:
                await adapter.start()
                logger.info("适配器已启动", extra={"platform": name})
            except Exception:
                logger.exception("适配器启动失败", extra={"platform": name})

        self._started = True
        logger.info("网关已启动，适配器数量: %d", len(self._adapters))

    async def stop(self) -> None:
        """停止网关：停止所有适配器"""
        for name, adapter in self._adapters.items():
            try:
                await adapter.stop()
                logger.info("适配器已停止", extra={"platform": name})
            except Exception:
                logger.exception("适配器停止异常", extra={"platform": name})

        self._started = False
        logger.info("网关已停止")

    # ---- 适配器管理 ----

    async def register_adapter(self, adapter: ChannelAdapter) -> None:
        """
        注册渠道适配器。

        如果网关已启动，自动调用 adapter.start()。
        """
        name = adapter.platform_name
        if name in self._adapters:
            logger.warning("适配器 %s 已被覆盖注册", name)

        self._adapters[name] = adapter
        logger.info("适配器已注册", extra={"platform": name})

        if self._started:
            try:
                await adapter.start()
            except Exception:
                logger.exception("适配器启动失败", extra={"platform": name})

    def unregister_adapter(self, platform_name: str) -> Optional[ChannelAdapter]:
        """注销渠道适配器"""
        adapter = self._adapters.pop(platform_name, None)
        if adapter:
            logger.info("适配器已注销", extra={"platform": platform_name})
        return adapter

    def get_adapter(self, platform_name: str) -> Optional[ChannelAdapter]:
        """获取指定平台的适配器"""
        return self._adapters.get(platform_name)

    def list_adapters(self) -> List[dict]:
        """返回所有已注册适配器的状态信息（用于 REST API）"""
        return [
            {
                "platform": name,
                "status": adapter.status,
                "type": type(adapter).__name__,
            }
            for name, adapter in self._adapters.items()
        ]

    async def reload_adapters(self) -> int:
        """
        重新加载所有适配器（先停止再启动）。

        Returns:
            成功启动的适配器数量
        """
        for name, adapter in self._adapters.items():
            try:
                await adapter.stop()
            except Exception:
                logger.exception("适配器停止异常", extra={"platform": name})

        success_count = 0
        for name, adapter in self._adapters.items():
            try:
                await adapter.start()
                success_count += 1
                logger.info("适配器已重新启动", extra={"platform": name})
            except Exception:
                logger.exception("适配器重新启动失败", extra={"platform": name})

        return success_count

    # ---- 消息入口 ----

    async def ingress(self, raw_msg: RawMessage) -> BaseEvent:
        """
        消息入口：接收适配器传入的 RawMessage，标准化后发布到事件总线。

        流程：
        1. 通过 SessionManager 解析或创建会话
        2. 构建 UserMessage 数据模型
        3. 发布 USER_MESSAGE 事件到事件总线
        4. 返回已发布的事件（含 event_id，用于幂等追踪）

        Args:
            raw_msg: 适配器构造的原始消息对象

        Returns:
            已发布到事件总线的 BaseEvent（含 event_id）
        """
        internal_user_id, session_id = await self._session_manager.resolve(
            raw_msg.platform, raw_msg.platform_user_id
        )

        message = UserMessage(
            platform=raw_msg.platform,
            platform_user_id=raw_msg.platform_user_id,
            internal_user_id=internal_user_id,
            session_id=session_id,
            text=raw_msg.text,
            timestamp=raw_msg.timestamp,
        )

        event = BaseEvent(
            type=EventType.USER_MESSAGE,
            source=f"gateway.{raw_msg.platform}",
            payload={"message": message},
        )
        await self._bus.publish(event)
        return event

    # ---- 消息出口 ----

    async def _on_reply_ready(self, event: BaseEvent) -> None:
        """
        处理 REPLY_READY 事件：将回复路由到对应渠道适配器发送。

        期望 event.payload 结构：
        {
            "platform": str,
            "target_id": str,
            "content": { "text": str, "image_path": str|None, "mime_type": str|None }
        }
        """
        payload = event.payload
        platform = payload.get("platform", "")
        target_id = payload.get("target_id", "")
        content_dict = payload.get("content", {})

        if not platform or not target_id:
            logger.error("REPLY_READY 事件缺少必要字段")
            return

        content = MessageContent(
            text=content_dict.get("text"),
            image_path=content_dict.get("image_path"),
            mime_type=content_dict.get("mime_type"),
            metadata=content_dict.get("metadata", {}),
        )

        adapter = self._adapters.get(platform)
        if not adapter:
            logger.error(
                "找不到适配器",
                extra={"platform": platform, "target_id": target_id},
            )
            self._buffer_message(platform, target_id, content)
            return

        try:
            success = await adapter.send_message(target_id, content)
            if not success:
                logger.warning(
                    "消息发送失败，已缓冲",
                    extra={"platform": platform, "target_id": target_id},
                )
                self._buffer_message(platform, target_id, content)
        except Exception:
            logger.exception(
                "消息发送异常，已缓冲",
                extra={"platform": platform, "target_id": target_id},
            )
            self._buffer_message(platform, target_id, content)

    # ---- 消息缓冲 ----

    def _buffer_message(
        self, platform: str, target_id: str, content: MessageContent
    ) -> None:
        """缓冲发送失败的消息，适配器重连后自动冲刷"""
        key = f"{platform}:{target_id}"
        if key not in self._message_buffer:
            self._message_buffer[key] = []
        if len(self._message_buffer[key]) < 100:
            self._message_buffer[key].append(content)
            logger.debug(
                "消息已缓冲",
                extra={
                    "key": key,
                    "buffer_size": len(self._message_buffer[key]),
                },
            )
        else:
            logger.warning(
                "缓冲已达上限（100条），丢弃消息",
                extra={"key": key},
            )

    async def flush_buffer(self, platform: str, target_id: str) -> int:
        """
        冲刷指定目标的缓冲消息。

        适配器重连成功后调用此方法恢复未发送的消息。

        Returns:
            成功发送的消息数量
        """
        key = f"{platform}:{target_id}"
        buffered = self._message_buffer.pop(key, [])
        if not buffered:
            return 0

        adapter = self._adapters.get(platform)
        if not adapter:
            for content in buffered:
                self._buffer_message(platform, target_id, content)
            return 0

        sent_count = 0
        for content in buffered:
            try:
                success = await adapter.send_message(target_id, content)
                if success:
                    sent_count += 1
                else:
                    self._buffer_message(platform, target_id, content)
            except Exception:
                self._buffer_message(platform, target_id, content)

        logger.info(
            "缓冲消息冲刷完成",
            extra={
                "platform": platform,
                "target_id": target_id,
                "sent": sent_count,
                "total": len(buffered),
            },
        )
        return sent_count

    # ---- 事件处理 ----

    async def _on_shutdown(self, event: BaseEvent) -> None:
        """处理 SHUTDOWN 事件：停止所有适配器"""
        await self.stop()

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def adapter_count(self) -> int:
        return len(self._adapters)

    @property
    def buffered_message_count(self) -> int:
        return sum(len(msgs) for msgs in self._message_buffer.values())


# ---- REST API 路由 ----

def create_gateway_router(gateway: Gateway):
    """
    创建网关管理 REST API 路由。

    B-T09: GET /api/v1/gateway/adapters 与 POST /api/v1/gateway/reload

    使用方式：
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(create_gateway_router(gateway))
    """
    from fastapi import APIRouter

    router = APIRouter(prefix="/api/v1/gateway")

    @router.get("/adapters")
    async def get_adapters():
        """返回所有已注册适配器的状态"""
        return {"adapters": gateway.list_adapters()}

    @router.post("/reload")
    async def reload_adapters():
        """重新加载所有适配器"""
        count = await gateway.reload_adapters()
        return {"success": True, "restarted": count}

    @router.get("/status")
    async def gateway_status():
        """返回网关整体状态"""
        return {
            "started": gateway.is_started,
            "adapter_count": gateway.adapter_count,
            "buffered_messages": gateway.buffered_message_count,
        }

    return router
