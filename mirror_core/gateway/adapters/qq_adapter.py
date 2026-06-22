"""
QQ 渠道适配器

B-T06: 基于反向 WebSocket 连接 NapCat 实例的 QQ 适配器

QQ 协议通过 NapCat（无头 QQ 客户端）提供 WebSocket API。
适配器以客户端身份连接 NapCat 的反向 WebSocket，订阅消息并发送回复。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time as _time
from typing import Callable, Optional

import aiohttp
from yarl import URL as _URL

from mirror_core.gateway.base import ChannelAdapter, MessageContent, RawMessage

logger = logging.getLogger("mirror_core.gateway.adapters.qq")


class QQAdapter(ChannelAdapter):
    """
    QQ 渠道适配器

    通过反向 WebSocket 连接 NapCat 实例，接收和发送 QQ 消息。
    自动重连：连接断开后以指数退避策略重连。

    Args:
        ingress_callback: Gateway.ingress 回调
        napcat_ws_url: NapCat 反向 WebSocket 地址（如 ws://127.0.0.1:6700）
        reconnect_interval: 重连间隔（秒），默认 5
    """

    def __init__(
        self,
        ingress_callback: Callable,
        napcat_ws_url: str = "ws://127.0.0.1:6700",
        reconnect_interval: float = 5.0,
    ):
        self._ingress = ingress_callback
        self._ws_url = _URL(napcat_ws_url)
        self._reconnect_interval = reconnect_interval
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._running = False
        self._connected = False

    async def start(self) -> None:
        """启动适配器，连接 NapCat"""
        self._running = True
        self._session = aiohttp.ClientSession()
        await self._connect()

    async def _connect(self) -> None:
        """连接到 NapCat 反向 WebSocket（含重连逻辑）"""
        retry = 0
        while self._running and not self._connected:
            try:
                ws = await self._session.ws_connect(
                    str(self._ws_url), timeout=10.0
                )
                self._ws = ws
                self._connected = True
                logger.info("已连接到 NapCat", extra={"url": str(self._ws_url)})

                # 阻塞直到断开连接
                await self._message_loop(ws)

            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                retry += 1
                wait = min(self._reconnect_interval * (1.5 ** (retry - 1)), 60)
                logger.warning(
                    "NapCat 连接失败，%.1f 秒后重试 (第 %d 次)",
                    wait,
                    retry,
                    extra={"url": str(self._ws_url), "error": str(exc)},
                )
                await asyncio.sleep(wait)
            except asyncio.CancelledError:
                break
            finally:
                self._connected = False
                self._ws = None

    async def _message_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """接收 NapCat 消息循环"""
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_message(msg.data)
            elif msg.type == aiohttp.WSMsgType.CLOSED:
                break
            elif msg.type == aiohttp.WSMsgType.ERROR:
                break

    async def _handle_message(self, raw: str) -> None:
        """处理 NapCat 推送的消息"""
        try:
            data = json.loads(raw)
            post_type = data.get("post_type")
            if post_type != "message":
                return

            user_id = str(data.get("user_id", ""))
            text = data.get("message", "")
            if isinstance(text, list):
                # NapCat 富文本消息为数组，提取纯文本
                text = " ".join(
                    seg.get("data", {}).get("text", "")
                    for seg in text
                    if isinstance(seg, dict)
                )

            if user_id and text.strip():
                raw_msg = RawMessage(
                    platform="qq",
                    platform_user_id=user_id,
                    text=text.strip(),
                    timestamp=_time.time(),
                    raw_data=data,
                )
                await self._ingress(raw_msg)

        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("NapCat 消息解析失败", extra={"error": str(exc)})

    async def stop(self) -> None:
        """停止适配器，断开连接"""
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._session:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None
        self._connected = False
        logger.info("QQ 适配器已停止")

    async def send_message(self, target_id: str, content: MessageContent) -> bool:
        """通过 NapCat HTTP API 发送 QQ 消息"""
        if not self._connected or not self._session:
            return False

        try:
            # 用 yarl.URL 安全地转换协议并拼接路径
            http_url = self._ws_url.with_scheme(
                "https" if self._ws_url.scheme == "wss" else "http"
            )
            send_msg_url = http_url / "send_msg"

            async with self._session.post(
                str(send_msg_url),
                json={
                    "message_type": "private",
                    "user_id": int(target_id),
                    "message": content.text or "",
                },
            ) as resp:
                return resp.status == 200
        except Exception:
            logger.exception("QQ 消息发送失败", extra={"target_id": target_id})
            return False

    @property
    def platform_name(self) -> str:
        return "qq"

    @property
    def status(self) -> str:
        return "connected" if self._connected else "disconnected"
