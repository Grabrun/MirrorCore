"""
性能测试与安全测试 (T-T05 / T-T06)

T-T05: 性能测试 — 消息处理延迟、并发检索、工作记忆恢复
T-T06: 安全测试 — SQL 注入、XSS、高风险内容拦截
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirror_core.bus import BaseEvent, EventBus, EventType
from mirror_core.core.safety import SafetyEngine, SafetyVerdict
from mirror_core.emotion.engine import TidalEngine, EmotionalState
from mirror_core.memory.engine import MemoryEngine, EpisodicMemory


# ===== T-T05: 性能测试 =====

class TestPerformance:
    """性能基准测试"""

    @pytest.mark.asyncio
    async def test_event_bus_publish_latency(self):
        """事件总线发布延迟 < 10ms"""
        bus = EventBus()
        called = False

        async def handler(event):
            nonlocal called
            called = True

        bus.subscribe(EventType.USER_MESSAGE, handler)
        event = BaseEvent(type=EventType.USER_MESSAGE)

        start = time.perf_counter()
        for _ in range(100):
            await bus.publish(event)
        elapsed = (time.perf_counter() - start) / 100 * 1000  # ms

        assert called
        assert elapsed < 10, f"单次 publish 延迟 {elapsed:.2f}ms > 10ms"

    @pytest.mark.asyncio
    async def test_emotion_thrust_latency(self):
        """情感推力施加延迟 < 1ms"""
        db = MagicMock()
        db.execute = AsyncMock()
        db.fetch_one = AsyncMock(return_value=None)

        engine = TidalEngine(db=db)
        delta = {"P": -0.3, "A": 0.2, "D": -0.1}

        start = time.perf_counter()
        for _ in range(100):
            await engine.apply_emotional_thrust(delta)
        elapsed = (time.perf_counter() - start) / 100 * 1000

        assert elapsed < 1, f"单次 thrust 延迟 {elapsed:.2f}ms > 1ms"

    @pytest.mark.asyncio
    async def test_memory_rrf_retrieve_latency(self):
        """记忆 RRF 检索延迟（mock DB） < 5ms"""
        db = MagicMock()

        async def fake_fetch_all(sql, params=None):
            return []

        db.fetch_all = fake_fetch_all
        db.fetch_one = AsyncMock(return_value=None)
        db.execute = AsyncMock()

        engine = MemoryEngine(db=db, embed_fn=lambda x: [0.1] * 768)

        start = time.perf_counter()
        for _ in range(50):
            await engine.retrieve("u1", "测试", top_k=5)
        elapsed = (time.perf_counter() - start) / 50 * 1000

        assert elapsed < 5, f"单次 retrieve 延迟 {elapsed:.2f}ms > 5ms"

    def test_safety_evaluate_latency(self):
        """安全引擎输入评估延迟 < 1ms"""
        engine = SafetyEngine()

        texts = [
            "今天天气真好" * 10,
            "我想自杀" + "a" * 100,
            "只有你懂我，别离开我" * 5,
        ]

        async def run():
            start = time.perf_counter()
            for _ in range(100):
                for text in texts:
                    await engine.evaluate_input(text)
            elapsed = (time.perf_counter() - start) / (100 * 3) * 1000
            assert elapsed < 1, f"单次 evaluate 延迟 {elapsed:.2f}ms > 1ms"

        asyncio.run(run())


# ===== T-T06: 安全测试 =====

class TestSecurity:
    """安全测试"""

    @pytest.mark.asyncio
    async def test_sql_injection_attempts(self):
        """参数化查询防御 SQL 注入"""
        from mirror_core.infrastructure.database import Database
        db = Database(path=":memory:")
        await db.initialize()

        # 尝试 SQL 注入：参数化查询不会执行注入语句
        injections = [
            "'; DROP TABLE fact_memory; --",
            "1 OR 1=1",
            '"; SELECT * FROM user_sessions; --',
        ]
        for inj in injections:
            try:
                result = await db.fetch_one(
                    "SELECT value FROM fact_memory WHERE key=?",
                    (inj,),
                )
                # 查询应该成功（无错误），返回 None（无数据）
                assert result is None or isinstance(result, dict)
            except Exception:
                pass  # 某些数据库驱动会拒绝恶意输入

        # 验证表仍然存在
        row = await db.fetch_one(
            "SELECT COUNT(*) as cnt FROM fact_memory"
        )
        assert row is not None
        await db.close()

    @pytest.mark.asyncio
    async def test_high_risk_content_blocked(self):
        """高风险内容被拦截返回预设安全回复"""
        engine = SafetyEngine()

        test_cases = [
            ("我想自杀", True),
            ("我要杀了你", True),
            ("今天天气真好", False),
            ("我觉得活着没意思", True),
            ("晚上吃什么", False),
        ]

        for text, should_flag in test_cases:
            result = await engine.evaluate_input(text)
            if should_flag:
                assert result.verdict == SafetyVerdict.FLAGGED, f"应拦截: {text}"
                assert result.reply is not None, f"应有安全回复: {text}"
            else:
                assert result.verdict != SafetyVerdict.FLAGGED, f"不应拦截: {text}"

    @pytest.mark.asyncio
    async def test_dependency_keyword_moderate(self):
        """依赖关键词标记 MODERATE"""
        engine = SafetyEngine()

        result = await engine.evaluate_input("只有你懂我")
        assert result.verdict == SafetyVerdict.MODERATE

        result = await engine.evaluate_input("我不能没有你")
        assert result.verdict == SafetyVerdict.MODERATE

    @pytest.mark.asyncio
    async def test_reality_anchor_injection(self):
        """高依赖度时注入现实锚点"""
        engine = SafetyEngine()

        result = await engine.inject_reality_anchor(
            response="我在呢",
            dependency_score=0.0,
        )
        assert result == "我在呢"

        result = await engine.inject_reality_anchor(
            response="我在呢",
            dependency_score=0.8,
        )
        assert len(result) > len("我在呢")
        assert "我在呢\n" in result or "我在呢" in result
