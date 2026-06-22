"""
渠道适配器测试

覆盖：
- ChannelAdapter 抽象基类接口验证
- WebChatAdapter WebSocket 消息收发（含 JWT 认证）
- TuiAdapter 消息显示与降级
- WechatAdapter 桩模式
"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mirror_core.gateway.adapters.tui_adapter import TuiAdapter
from mirror_core.gateway.adapters.webchat_adapter import WebChatAdapter
from mirror_core.gateway.adapters.wechat_adapter import WechatAdapter
from mirror_core.gateway.auth import generate_secret, generate_token
from mirror_core.gateway.base import ChannelAdapter, MessageContent, RawMessage


class TestChannelAdapterBase:
    """ChannelAdapter 抽象基类接口验证"""

    def test_cannot_instantiate_abstract(self):
        """抽象基类不能直接实例化"""
        with pytest.raises(TypeError):
            ChannelAdapter()

    def test_message_content_defaults(self):
        """MessageContent 默认值"""
        mc = MessageContent()
        assert mc.text is None
        assert mc.image_path is None
        assert mc.is_text_only is True

    def test_message_content_with_image(self):
        """带图片的消息"""
        mc = MessageContent(text="看这个", image_path="/img/cat.png", mime_type="image/png")
        assert mc.is_text_only is False
        assert "[图片:" in mc.display_text

    def test_message_content_is_text_only(self):
        """纯文本消息"""
        mc = MessageContent(text="你好")
        assert mc.is_text_only is True
        assert mc.display_text == "你好"


class TestWebChatAdapter:
    """WebChat 适配器测试"""

    @pytest.fixture
    def app(self):
        return FastAPI()

    @pytest.fixture
    def jwt_secret(self):
        return generate_secret()

    @pytest.fixture
    def test_token(self, jwt_secret):
        return generate_token("test_user", jwt_secret)

    @pytest.mark.asyncio
    async def test_start_stop(self, app, jwt_secret):
        """启动和停止"""
        ingress = AsyncMock()
        adapter = WebChatAdapter(app, ingress, jwt_secret=jwt_secret)
        assert adapter.status == "disconnected"

        await adapter.start()
        assert adapter.status == "connected"

        await adapter.stop()
        assert adapter.status == "disconnected"

    @pytest.mark.asyncio
    async def test_send_message_to_offline_user(self, app, jwt_secret):
        """发送给不在线的用户应返回 False"""
        ingress = AsyncMock()
        adapter = WebChatAdapter(app, ingress, jwt_secret=jwt_secret)
        await adapter.start()

        result = await adapter.send_message("nonexistent", MessageContent(text="hi"))
        assert result is False

        await adapter.stop()

    @pytest.mark.asyncio
    async def test_websocket_no_token_rejected(self, app, jwt_secret):
        """无 JWT 令牌的连接应被拒绝（ingress 不被调用）"""
        ingress = AsyncMock()
        adapter = WebChatAdapter(app, ingress, jwt_secret=jwt_secret)
        await adapter.start()

        # 缺少令牌时 ingress 不应被调用
        ingress.assert_not_awaited()
        await adapter.stop()

    @pytest.mark.asyncio
    async def test_websocket_lifecycle(self, app, jwt_secret, test_token):
        """WebSocket 完整生命周期测试（含 JWT 认证）"""
        ingress = AsyncMock()
        adapter = WebChatAdapter(app, ingress, jwt_secret=jwt_secret)
        await adapter.start()

        with TestClient(app) as client:
            with client.websocket_connect(f"/ws?token={test_token}") as ws:
                ws.send_json({"text": "你好世界"})
                await asyncio.sleep(0.05)

                ingress.assert_awaited_once()
                args = ingress.call_args.args
                raw_msg = args[0]
                assert raw_msg.platform == "webchat"
                assert raw_msg.platform_user_id == "test_user"
                assert raw_msg.text == "你好世界"

        await adapter.stop()

    @pytest.mark.asyncio
    async def test_send_message_online(self, app, jwt_secret, test_token):
        """在线用户应能收到消息"""
        ingress = AsyncMock()
        adapter = WebChatAdapter(app, ingress, jwt_secret=jwt_secret)
        await adapter.start()

        with TestClient(app) as client:
            with client.websocket_connect(f"/ws?token={test_token}") as ws:
                result = await adapter.send_message("test_user", MessageContent(text="回复"))
                assert result is True

        await adapter.stop()


class TestTuiAdapter:
    """TUI 适配器测试"""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """启动和停止"""
        ingress = AsyncMock()
        adapter = TuiAdapter(ingress)

        assert adapter.status == "disconnected"
        await adapter.start()
        assert adapter.status == "connected"
        await adapter.stop()
        assert adapter.status == "disconnected"

    @pytest.mark.asyncio
    async def test_send_message(self):
        """发送消息"""
        ingress = AsyncMock()
        adapter = TuiAdapter(ingress)
        await adapter.start()

        result = await adapter.send_message("tui_user", MessageContent(text="你好"))
        assert result is True

        await adapter.stop()

    @pytest.mark.asyncio
    async def test_image_downgrade(self):
        """图片应降级为文本标签"""
        ingress = AsyncMock()
        adapter = TuiAdapter(ingress)
        await adapter.start()

        content = MessageContent(
            text="看这个猫", image_path="/img/cat.png", mime_type="image/png"
        )
        result = await adapter.send_message("tui_user", content)
        assert result is True

        await adapter.stop()

    @pytest.mark.asyncio
    async def test_platform_name(self):
        """平台名称应为 tui"""
        ingress = AsyncMock()
        adapter = TuiAdapter(ingress)
        assert adapter.platform_name == "tui"


class TestWechatAdapter:
    """微信适配器测试（桩模式）"""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """启动和停止（桩模式）"""
        ingress = AsyncMock()
        adapter = WechatAdapter(ingress, app_id="test_app")

        assert adapter.status == "disconnected"
        await adapter.start()
        assert adapter.status == "connected"
        await adapter.stop()
        assert adapter.status == "disconnected"

    @pytest.mark.asyncio
    async def test_send_message(self):
        """发送消息（桩模式应返回 True）"""
        ingress = AsyncMock()
        adapter = WechatAdapter(ingress)
        await adapter.start()

        result = await adapter.send_message("openid_xxx", MessageContent(text="你好"))
        assert result is True

        await adapter.stop()

    @pytest.mark.asyncio
    async def test_send_when_disconnected(self):
        """断开时发送应返回 False"""
        ingress = AsyncMock()
        adapter = WechatAdapter(ingress)

        result = await adapter.send_message("openid_xxx", MessageContent(text="你好"))
        assert result is False

    @pytest.mark.asyncio
    async def test_platform_name(self):
        """平台名称应为 wechat"""
        ingress = AsyncMock()
        adapter = WechatAdapter(ingress)
        assert adapter.platform_name == "wechat"
