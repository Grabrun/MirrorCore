"""
Gateway 核心逻辑测试

覆盖：
- 适配器注册/注销/列表
- 消息入口 (ingress)：事件发布到总线
- 消息出口 (egress)：REPLY_READY 事件路由到适配器
- 消息缓冲：发送失败时自动缓冲
- 适配器重载
"""

import asyncio
import os
import tempfile
import time as _time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirror_core.bus import BaseEvent, EventBus, EventType
from mirror_core.gateway.base import ChannelAdapter, MessageContent, RawMessage
from mirror_core.gateway.gateway import Gateway
from mirror_core.gateway.session import SessionManager
from mirror_core.infrastructure.database import Database


@pytest.fixture
async def bus():
    """事件总线"""
    eb = EventBus()
    yield eb
    await eb.shutdown()


class MockAdapter(ChannelAdapter):
    """模拟适配器，用于测试"""

    def __init__(self, name: str = "mock", fail_send: bool = False):
        self._name = name
        self._fail_send = fail_send
        self._started = False
        self._stopped = False
        self.sent_messages = []

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._stopped = True

    async def send_message(self, target_id: str, content: MessageContent) -> bool:
        if self._fail_send:
            return False
        self.sent_messages.append((target_id, content))
        return True

    @property
    def platform_name(self) -> str:
        return self._name

    @property
    def status(self) -> str:
        return "connected" if self._started else "disconnected"


@pytest.fixture
async def gateway(bus):
    """创建 Gateway 实例"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db = Database(db_path)
    await db.initialize()
    sm = SessionManager(db)
    await sm.initialize()

    gw = Gateway(bus, sm)

    yield gw, db

    await db.close()
    os.unlink(db_path)


class TestGatewayAdapters:
    """适配器注册管理测试"""

    @pytest.mark.asyncio
    async def test_register_adapter(self, gateway):
        """注册适配器"""
        gw, _ = gateway
        adapter = MockAdapter("test")
        await gw.register_adapter(adapter)

        assert gw.adapter_count == 1
        assert gw.get_adapter("test") is adapter

    @pytest.mark.asyncio
    async def test_register_and_start(self, gateway):
        """注册后自动启动适配器"""
        gw, _ = gateway
        adapter = MockAdapter("test")
        await gw.start()
        await gw.register_adapter(adapter)

        assert adapter._started is True

    @pytest.mark.asyncio
    async def test_unregister_adapter(self, gateway):
        """注销适配器"""
        gw, _ = gateway
        adapter = MockAdapter("test")
        await gw.register_adapter(adapter)

        removed = gw.unregister_adapter("test")
        assert removed is adapter
        assert gw.adapter_count == 0

    @pytest.mark.asyncio
    async def test_unregister_nonexistent(self, gateway):
        """注销不存在的适配器返回 None"""
        gw, _ = gateway
        result = gw.unregister_adapter("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_adapters(self, gateway):
        """列出适配器"""
        gw, _ = gateway
        await gw.register_adapter(MockAdapter("alpha"))
        await gw.register_adapter(MockAdapter("beta"))

        adapters = gw.list_adapters()
        assert len(adapters) == 2
        platforms = {a["platform"] for a in adapters}
        assert platforms == {"alpha", "beta"}


class TestGatewayIngress:
    """消息入口测试"""

    @pytest.mark.asyncio
    async def test_ingress_publishes_event(self, bus, gateway):
        """ingress 应发布 USER_MESSAGE 事件到总线"""
        gw, _ = gateway

        handler = AsyncMock()
        bus.subscribe(EventType.USER_MESSAGE, handler)

        raw_msg = RawMessage(platform="test", platform_user_id="user001", text="你好", timestamp=_time.time())
        await gw.ingress(raw_msg)

        # 验证事件发布
        handler.assert_awaited_once()
        published_event = handler.call_args[0][0]
        assert published_event.type == EventType.USER_MESSAGE
        assert published_event.source == "gateway.test"
        assert published_event.payload["message"].platform == "test"
        assert published_event.payload["message"].text == "你好"

    @pytest.mark.asyncio
    async def test_ingress_creates_session(self, bus, gateway):
        """ingress 应为新用户创建会话"""
        gw, db = gateway

        await gw.ingress(RawMessage(platform="test", platform_user_id="new_user", text="hello", timestamp=_time.time()))

        row = await db.fetch_one(
            "SELECT * FROM user_sessions WHERE platform=? AND platform_user_id=?",
            ("test", "new_user"),
        )
        assert row is not None

    @pytest.mark.asyncio
    async def test_ingress_reuses_session(self, bus, gateway):
        """同一用户多次 ingress 应复用同一 session"""
        gw, _ = gateway

        await gw.ingress(RawMessage(platform="test", platform_user_id="user001", text="第一条", timestamp=_time.time()))
        await gw.ingress(RawMessage(platform="test", platform_user_id="user001", text="第二条", timestamp=_time.time()))

        # 两次发消息，验证 event 中的用户 ID 一致
        # 通过 SessionManager 直接查
        internal_id, _ = await gw._session_manager.resolve("test", "user001")
        assert internal_id.startswith("u_")


class TestGatewayEgress:
    """消息出口测试"""

    @pytest.mark.asyncio
    async def test_reply_ready_routes_to_adapter(self, bus, gateway):
        """REPLY_READY 事件应路由到正确的适配器"""
        gw, _ = gateway
        adapter = MockAdapter("test")
        await gw.register_adapter(adapter)
        await gw.start()

        # 发布 REPLY_READY 事件
        reply_event = BaseEvent(
            type=EventType.REPLY_READY,
            source="test",
            payload={
                "platform": "test",
                "target_id": "user001",
                "content": {"text": "回复消息"},
            },
        )
        await bus.publish(reply_event)
        await asyncio.sleep(0.05)

        # 验证适配器收到消息
        assert len(adapter.sent_messages) == 1
        assert adapter.sent_messages[0][0] == "user001"
        assert adapter.sent_messages[0][1].text == "回复消息"

    @pytest.mark.asyncio
    async def test_reply_to_unknown_adapter_buffers(self, bus, gateway):
        """找不到适配器时应缓冲消息"""
        gw, _ = gateway
        await gw.start()

        # 回复到不存在的适配器
        reply_event = BaseEvent(
            type=EventType.REPLY_READY,
            source="test",
            payload={
                "platform": "unknown",
                "target_id": "user001",
                "content": {"text": "丢失的消息"},
            },
        )
        await bus.publish(reply_event)
        await asyncio.sleep(0.05)

        assert gw.buffered_message_count == 1

    @pytest.mark.asyncio
    async def test_send_failure_buffers_message(self, bus, gateway):
        """发送失败的消息应被缓冲"""
        gw, _ = gateway
        adapter = MockAdapter("test", fail_send=True)
        await gw.register_adapter(adapter)
        await gw.start()

        reply_event = BaseEvent(
            type=EventType.REPLY_READY,
            source="test",
            payload={
                "platform": "test",
                "target_id": "user001",
                "content": {"text": "发送失败的消息"},
            },
        )
        await bus.publish(reply_event)
        await asyncio.sleep(0.05)

        assert gw.buffered_message_count == 1

    @pytest.mark.asyncio
    async def test_flush_buffer(self, bus, gateway):
        """冲刷缓冲应重发消息"""
        gw, _ = gateway

        # 先发一条到不存在的适配器（缓冲）
        await gw.start()
        reply_event = BaseEvent(
            type=EventType.REPLY_READY,
            source="test",
            payload={
                "platform": "test",
                "target_id": "user001",
                "content": {"text": "缓冲的消息"},
            },
        )
        await bus.publish(reply_event)
        await asyncio.sleep(0.05)

        # 注册适配器并冲刷
        adapter = MockAdapter("test")
        await gw.register_adapter(adapter)
        sent = await gw.flush_buffer("test", "user001")

        assert sent == 1
        assert len(adapter.sent_messages) == 1

    @pytest.mark.asyncio
    async def test_flush_empty_buffer(self, bus, gateway):
        """冲刷空缓冲应返回 0"""
        gw, _ = gateway
        sent = await gw.flush_buffer("test", "user001")
        assert sent == 0


class TestGatewayEventHandlers:
    """Gateway 事件处理测试"""

    @pytest.mark.asyncio
    async def test_shutdown_stops_adapters(self, bus, gateway):
        """SHUTDOWN 事件应停止所有适配器"""
        gw, _ = gateway
        adapter = MockAdapter("test")
        await gw.register_adapter(adapter)
        await gw.start()

        # 发布 SHUTDOWN
        await bus.shutdown()
        await asyncio.sleep(0.05)

        assert adapter._stopped is True

    @pytest.mark.asyncio
    async def test_reload_adapters(self, bus, gateway):
        """重载适配器应先停止再启动"""
        gw, _ = gateway
        adapter = MockAdapter("test")
        await gw.register_adapter(adapter)

        await gw.start()
        assert adapter._started is True

        # 停止然后手动验证
        adapter._started = False
        count = await gw.reload_adapters()
        assert count == 1
        assert adapter._started is True



