"""
微信渠道适配器

B-T08: 封装 iLink SDK 的微信适配器

微信渠道通过微信官方 iLink 插件集成。
适配器封装 iLink 协议，将同步回调转换为异步。
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
from typing import Callable, Optional

from mirror_core.gateway.base import ChannelAdapter, MessageContent, RawMessage

logger = logging.getLogger("mirror_core.gateway.adapters.wechat")


class WechatAdapter(ChannelAdapter):
    """
    微信渠道适配器

    封装 iLink SDK，使用 asyncio.to_thread 将同步回调转为异步。
    注意：此适配器需要 iLink SDK 环境支持，当前为桩实现。

    Args:
        ingress_callback: Gateway.ingress 回调
        app_id: iLink 应用 ID（占位）
    """

    def __init__(
        self,
        ingress_callback: Callable,
        app_id: str = "",
    ):
        self._ingress = ingress_callback
        self._app_id = app_id
        self._running = False
        self._connected = False

    async def start(self) -> None:
        """
        启动微信适配器。

        实际部署时需要集成 iLink SDK 的初始化逻辑。
        当前为桩实现，仅标记为已连接。
        """
        self._running = True
        self._connected = True

        # TODO: 实际接入 iLink SDK 时的初始化代码示例：
        # def on_message(openid: str, content: str):
        #     raw_msg = RawMessage(
        #         platform="wechat",
        #         platform_user_id=openid,
        #         text=content,
        #         timestamp=_time.time(),
        #     )
        #     asyncio.run_coroutine_threadsafe(
        #         self._ingress(raw_msg), self._loop
        #     )
        # self._sdk = iLinkClient(app_id=self._app_id)
        # self._sdk.set_message_callback(on_message)
        # await asyncio.to_thread(self._sdk.start)

        logger.info(
            "微信适配器已启动（桩模式）",
            extra={"app_id": self._app_id or "未配置"},
        )

    async def stop(self) -> None:
        """停止微信适配器"""
        self._running = False
        self._connected = False
        # TODO: await asyncio.to_thread(self._sdk.stop)
        logger.info("微信适配器已停止")

    async def send_message(self, target_id: str, content: MessageContent) -> bool:
        """
        发送微信消息。

        实际部署时调用 iLink SDK 的发送接口。
        当前为桩实现，始终返回成功。

        Args:
            target_id: 微信用户的 openid
            content: 消息内容
        """
        if not self._connected:
            return False

        # TODO: await asyncio.to_thread(
        #     self._sdk.send_text, target_id, content.text
        # )
        logger.debug(
            "微信消息发送（桩）",
            extra={"target_id": target_id, "text": content.text},
        )
        return True

    @property
    def platform_name(self) -> str:
        return "wechat"

    @property
    def status(self) -> str:
        return "connected" if self._connected else "disconnected"
