"""
WebChat 渠道适配器

B-T05: 基于 FastAPI WebSocket 实现 WebChat 适配器
B-T42/B-T43: JWT 匿名认证支持

提供 WebSocket 端点 /ws?token=***，浏览器通过 JWT 令牌认证后连接。
支持文本消息收发，图片表情包以 base64 或 URL 形式传递。
"""

from __future__ import annotations

import json
import logging
import time as _time
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from mirror_core.gateway.auth import verify_token
from mirror_core.gateway.base import ChannelAdapter, MessageContent, RawMessage

logger = logging.getLogger("mirror_core.gateway.adapters.webchat")


class WebChatAdapter(ChannelAdapter):
    """
    WebChat 渠道适配器

    基于 FastAPI WebSocket，提供 JWT 认证的实时聊天接入。

    Args:
        ingress_callback: Gateway.ingress 回调
        app: FastAPI 应用实例（用于注册 WebSocket 路由）
        jwt_secret: JWT 签名密钥（用于验证 WebSocket 令牌）
    """

    def __init__(
        self,
        app: FastAPI,
        ingress_callback,
        jwt_secret: str = "",
    ):
        self._app = app
        self._ingress = ingress_callback
        self._jwt_secret = jwt_secret
        self._connections: Dict[str, WebSocket] = {}
        self._running = False
        self._ws_registered = False

    async def start(self) -> None:
        """注册 JWT 认证的 WebSocket 端点 + 静态文件服务（F-T01）"""
        if self._ws_registered:
            logger.warning("WebSocket 端点已注册，忽略重复调用")
            return

        # 挂载 WebChat 前端静态文件
        import os as _os
        webchat_dir = _os.path.join(_os.path.dirname(__file__), "..", "..", "webchat")
        webchat_dir = _os.path.abspath(webchat_dir)
        if _os.path.isdir(webchat_dir):
            self._app.mount("/webchat", StaticFiles(directory=webchat_dir, html=True), name="webchat")

            @self._app.get("/")
            async def root_redirect():
                from fastapi.responses import RedirectResponse
                return RedirectResponse(url="/webchat/index.html")

            logger.info("WebChat 前端已挂载: %s", webchat_dir)

        jwt_secret = self._jwt_secret

        @self._app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            # 从 query 参数获取 JWT 令牌
            token = websocket.query_params.get("token", "")
            if not token or not jwt_secret:
                await websocket.close(code=4001, reason="缺少认证令牌")
                logger.warning("WebSocket 连接被拒绝：缺少令牌")
                return

            user_id = verify_token(token, jwt_secret)
            if not user_id:
                await websocket.close(code=4001, reason="令牌无效或已过期")
                logger.warning("WebSocket 连接被拒绝：令牌无效")
                return

            await websocket.accept()
            self._connections[user_id] = websocket
            logger.info("WebSocket 连接已建立", extra={"user_id": user_id})

            try:
                while True:
                    raw = await websocket.receive_text()
                    data = json.loads(raw)
                    text = data.get("text", "")

                    if text.strip():
                        raw_msg = RawMessage(
                            platform="webchat",
                            platform_user_id=user_id,
                            text=text.strip(),
                            timestamp=_time.time(),
                        )
                        await self._ingress(raw_msg)

            except WebSocketDisconnect:
                logger.info("WebSocket 已断开", extra={"user_id": user_id})
            except Exception:
                logger.exception("WebSocket 处理异常", extra={"user_id": user_id})
            finally:
                self._connections.pop(user_id, None)

        self._ws_registered = True
        self._running = True
        logger.info("WebChat 适配器已启动")

    async def stop(self) -> None:
        """关闭所有 WebSocket 连接"""
        self._running = False
        for user_id, ws in list(self._connections.items()):
            try:
                await ws.close()
            except Exception:
                pass
        self._connections.clear()
        logger.info("WebChat 适配器已停止")

    async def send_message(self, target_id: str, content: MessageContent) -> bool:
        """通过 WebSocket 发送消息给指定用户"""
        ws = self._connections.get(target_id)
        if not ws:
            logger.warning("用户不在线", extra={"user_id": target_id})
            return False

        try:
            payload = {"text": content.text or ""}
            if content.image_path:
                payload["image_path"] = content.image_path
                payload["mime_type"] = content.mime_type

            await ws.send_text(json.dumps(payload, ensure_ascii=False))
            return True
        except Exception:
            logger.exception("WebSocket 发送失败", extra={"user_id": target_id})
            return False

    @property
    def platform_name(self) -> str:
        return "webchat"

    @property
    def status(self) -> str:
        return "connected" if self._running else "disconnected"
