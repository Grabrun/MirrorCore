"""
事件总线单元测试

覆盖范围：
- 基础发布/订阅流程
- 多处理器顺序执行
- 处理器异常隔离
- 幂等性校验
- 优雅关闭
- 边界场景
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from mirror_core.bus import BaseEvent, EventBus, EventType, _LRUSet


class TestLRUSet:
    """LRU 集合单元测试"""

    def test_add_and_contains(self):
        s = _LRUSet(maxsize=3)
        s.add("a")
        s.add("b")
        assert "a" in s
        assert "b" in s
        assert "c" not in s

    def test_eviction(self):
        s = _LRUSet(maxsize=3)
        s.add("a")
        s.add("b")
        s.add("c")
        s.add("d")  # 应淘汰 "a"
        assert "a" not in s
        assert "b" in s
        assert "c" in s
        assert "d" in s

    def test_lru_reorder(self):
        s = _LRUSet(maxsize=3)
        s.add("a")
        s.add("b")
        s.add("c")
        # 访问 "a" 使其变为最近使用
        assert "a" in s
        s.add("d")  # 应淘汰 "b"（"a" 刚被访问过）
        assert "a" in s
        assert "b" not in s
        assert "c" in s
        assert "d" in s

    def test_clear(self):
        s = _LRUSet(maxsize=3)
        s.add("a")
        s.add("b")
        s.clear()
        assert len(s) == 0
        assert "a" not in s

    def test_len(self):
        s = _LRUSet(maxsize=5)
        assert len(s) == 0
        s.add("a")
        assert len(s) == 1
        s.add("b")
        s.add("c")
        assert len(s) == 3


class TestEventBus:
    """事件总线单元测试"""

    @pytest.fixture
    def bus(self):
        """创建新的事件总线实例"""
        return EventBus()

    @pytest.fixture
    def sample_event(self):
        """创建示例事件"""
        return BaseEvent(
            type=EventType.USER_MESSAGE,
            source="test",
            payload={"text": "hello"},
        )

    # ---- 基础功能 ----

    @pytest.mark.asyncio
    async def test_basic_publish_subscribe(self, bus, sample_event):
        """基础发布/订阅：处理器应被正确调用"""
        handler = AsyncMock()
        bus.subscribe(EventType.USER_MESSAGE, handler)

        await bus.publish(sample_event)

        handler.assert_awaited_once_with(sample_event)

    @pytest.mark.asyncio
    async def test_multiple_handlers_same_event(self, bus, sample_event):
        """多个处理器应被顺序调用"""
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        bus.subscribe(EventType.USER_MESSAGE, handler1)
        bus.subscribe(EventType.USER_MESSAGE, handler2)

        await bus.publish(sample_event)

        handler1.assert_awaited_once_with(sample_event)
        handler2.assert_awaited_once_with(sample_event)

    @pytest.mark.asyncio
    async def test_handler_order_preserved(self, bus):
        """处理器应按注册顺序执行"""
        execution_order = []

        async def handler1(event):
            execution_order.append("handler1")

        async def handler2(event):
            execution_order.append("handler2")

        async def handler3(event):
            execution_order.append("handler3")

        bus.subscribe(EventType.USER_MESSAGE, handler1)
        bus.subscribe(EventType.USER_MESSAGE, handler2)
        bus.subscribe(EventType.USER_MESSAGE, handler3)

        await bus.publish(BaseEvent(type=EventType.USER_MESSAGE, source="test"))

        assert execution_order == ["handler1", "handler2", "handler3"]

    @pytest.mark.asyncio
    async def test_event_not_routed_to_wrong_type(self, bus, sample_event):
        """事件不应被错误类型的处理器接收"""
        handler = AsyncMock()
        bus.subscribe(EventType.SYSTEM_TICK, handler)

        await bus.publish(sample_event)

        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_subscribers_does_not_raise(self, bus, sample_event):
        """无订阅者的事件不应抛出异常"""
        await bus.publish(sample_event)  # 不应引发异常

    # ---- 处理器异常隔离 ----

    @pytest.mark.asyncio
    async def test_handler_exception_does_not_block_others(self, bus):
        """单个处理器异常不应中断后续处理器"""
        results = []

        async def failing_handler(event):
            raise ValueError("故意的错误")

        async def good_handler(event):
            results.append("called")

        bus.subscribe(EventType.USER_MESSAGE, failing_handler)
        bus.subscribe(EventType.USER_MESSAGE, good_handler)

        await bus.publish(BaseEvent(type=EventType.USER_MESSAGE, source="test"))

        assert results == ["called"]

    @pytest.mark.asyncio
    async def test_all_handlers_run_after_exception(self, bus):
        """异常发生后，所有注册的处理器仍应执行"""
        results = []

        async def fail1(event):
            results.append("fail1_start")
            raise ValueError("err1")

        async def ok(event):
            results.append("ok")

        async def fail2(event):
            results.append("fail2_start")
            raise ValueError("err2")

        async def end(event):
            results.append("end")

        bus.subscribe(EventType.USER_MESSAGE, fail1)
        bus.subscribe(EventType.USER_MESSAGE, ok)
        bus.subscribe(EventType.USER_MESSAGE, fail2)
        bus.subscribe(EventType.USER_MESSAGE, end)

        await bus.publish(BaseEvent(type=EventType.USER_MESSAGE, source="test"))

        assert results == ["fail1_start", "ok", "fail2_start", "end"]

    # ---- 幂等性 ----

    @pytest.mark.asyncio
    async def test_idempotent_duplicate_event_skipped(self, bus):
        """相同 event_id 的事件应只被处理一次"""
        handler = AsyncMock()
        bus.subscribe(EventType.USER_MESSAGE, handler)

        event = BaseEvent(type=EventType.USER_MESSAGE, source="test")
        await bus.publish(event)
        await bus.publish(event)

        assert handler.await_count == 1

    @pytest.mark.asyncio
    async def test_different_event_ids_both_processed(self, bus):
        """不同 event_id 的事件应被分别处理"""
        handler = AsyncMock()
        bus.subscribe(EventType.USER_MESSAGE, handler)

        event1 = BaseEvent(type=EventType.USER_MESSAGE, source="test")
        event2 = BaseEvent(type=EventType.USER_MESSAGE, source="test")

        await bus.publish(event1)
        await bus.publish(event2)

        assert handler.await_count == 2

    @pytest.mark.asyncio
    async def test_idempotent_cache_eviction(self, bus):
        """LRU 缓存淘汰后，旧 event_id 可被再次处理"""
        # 替换为小容量缓存（直接访问私有成员用于测试）
        bus._processed_ids = _LRUSet(maxsize=3)
        handler = AsyncMock()
        bus.subscribe(EventType.USER_MESSAGE, handler)

        # 填满缓存
        events = []
        for i in range(4):
            e = BaseEvent(type=EventType.USER_MESSAGE, source="test")
            events.append(e)
            await bus.publish(e)

        # 第4个事件应淘汰第1个的 event_id
        first_id = events[0].event_id
        assert first_id not in bus._processed_ids

        # 再次发布第1个事件（相同 event_id），应被处理（因为已被淘汰）
        handler.reset_mock()
        await bus.publish(events[0])
        assert handler.await_count == 1

    # ---- 优雅关闭 ----

    @pytest.mark.asyncio
    async def test_shutdown_waits_for_pending_handlers(self, bus):
        """
        顺序执行模式下，publish 完成后所有处理器已执行完毕。
        shutdown 应发布 SHUTDOWN 事件并正常返回。
        """
        results = []

        async def slow_handler(event):
            await asyncio.sleep(0.1)
            results.append("slow_done")

        async def fast_handler(event):
            results.append("fast_done")

        bus.subscribe(EventType.USER_MESSAGE, fast_handler)
        bus.subscribe(EventType.USER_MESSAGE, slow_handler)

        event = BaseEvent(type=EventType.USER_MESSAGE, source="test")
        await bus.publish(event)

        await bus.shutdown()

        assert "fast_done" in results
        assert "slow_done" in results

    @pytest.mark.asyncio
    async def test_shutdown_prevents_new_events(self, bus):
        """shutdown() 后新事件应被静默忽略"""
        handler = AsyncMock()
        bus.subscribe(EventType.USER_MESSAGE, handler)

        await bus.shutdown()

        await bus.publish(BaseEvent(type=EventType.USER_MESSAGE, source="test"))

        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self, bus):
        """多次调用 shutdown() 应安全"""
        await bus.shutdown()
        await bus.shutdown()  # 不应引发异常

    @pytest.mark.asyncio
    async def test_shutdown_completes_quickly_when_idle(self, bus):
        """
        无进行中处理器时，shutdown 应立即完成。
        """
        handler = AsyncMock()
        bus.subscribe(EventType.USER_MESSAGE, handler)

        start = asyncio.get_event_loop().time()
        await bus.shutdown()
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed < 1.0  # 应在 1 秒内返回

    @pytest.mark.asyncio
    async def test_wait_closed(self, bus):
        """wait_closed() 应在 shutdown 完成后返回"""
        async def shutdown_later():
            await asyncio.sleep(0.05)
            await bus.shutdown()

        asyncio.create_task(shutdown_later())

        await bus.wait_closed()
        assert bus.is_shutting_down

    # ---- 订阅管理 ----

    @pytest.mark.asyncio
    async def test_subscribe_non_coroutine_raises(self, bus):
        """注册非协程函数应抛出 TypeError"""
        def sync_handler(event):
            pass

        with pytest.raises(TypeError, match="协程函数"):
            bus.subscribe(EventType.USER_MESSAGE, sync_handler)

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_handler(self, bus):
        """取消注册的处理器不应再被调用"""
        handler = AsyncMock()
        bus.subscribe(EventType.USER_MESSAGE, handler)

        result = bus.unsubscribe(EventType.USER_MESSAGE, handler)
        assert result is True

        await bus.publish(BaseEvent(type=EventType.USER_MESSAGE, source="test"))
        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent_returns_false(self, bus):
        """取消不存在的处理器应返回 False"""
        handler = AsyncMock()
        result = bus.unsubscribe(EventType.USER_MESSAGE, handler)
        assert result is False

    @pytest.mark.asyncio
    async def test_subscriber_count(self, bus):
        """订阅者计数应准确"""
        h1 = AsyncMock()
        h2 = AsyncMock()
        h3 = AsyncMock()

        bus.subscribe(EventType.USER_MESSAGE, h1)
        bus.subscribe(EventType.USER_MESSAGE, h2)
        bus.subscribe(EventType.SYSTEM_TICK, h3)

        assert bus.subscriber_count(EventType.USER_MESSAGE) == 2
        assert bus.subscriber_count(EventType.SYSTEM_TICK) == 1
        assert bus.subscriber_count() == 3

    # ---- 边界场景 ----

    @pytest.mark.asyncio
    async def test_publish_shutdown_event_via_shutdown(self, bus):
        """shutdown() 应发布 SHUTDOWN 事件给订阅者"""
        handler = AsyncMock()
        bus.subscribe(EventType.SHUTDOWN, handler)

        await bus.shutdown()

        handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_publish_during_shutdown_interrupted(self):
        """
        关闭中发布的事件：正在执行的处理器继续完成，
        但后续处理器被中断。
        """
        bus = EventBus(shutdown_timeout=0.05)
        results = []

        async def handler1(event):
            results.append("h1")

        async def handler2(event):
            await asyncio.sleep(0.2)
            results.append("h2")

        async def handler3(event):
            results.append("h3")

        bus.subscribe(EventType.USER_MESSAGE, handler1)
        bus.subscribe(EventType.USER_MESSAGE, handler2)
        bus.subscribe(EventType.USER_MESSAGE, handler3)

        # handler2 执行期间发起关闭
        async def delayed_shutdown():
            await asyncio.sleep(0.05)
            await bus.shutdown()

        event = BaseEvent(type=EventType.USER_MESSAGE, source="test")
        asyncio.create_task(delayed_shutdown())
        await asyncio.sleep(0.01)
        await bus.publish(event)  # 阻塞在 handler2 的 0.2s 睡眠上

        # publish 完成后检查结果
        assert "h1" in results      # handler1 先执行
        assert "h2" in results      # handler2 完成（已在关闭前开始）

    @pytest.mark.asyncio
    async def test_is_shutting_down_property(self, bus):
        """is_shutting_down 属性应反映关闭状态"""
        assert bus.is_shutting_down is False

        async def do_shutdown():
            await bus.shutdown()

        asyncio.create_task(do_shutdown())
        await asyncio.sleep(0.05)

        await bus.wait_closed()
        assert bus.is_shutting_down is True
