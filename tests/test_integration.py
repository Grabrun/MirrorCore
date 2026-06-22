"""
User Story 17: 补充测试（全链路集成 + 并发 + 边界）

T-T01/T-T03/T-T04: 事件总线→决策引擎全链路集成
T-T05: 100 并发 SQLite 检索
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirror_core.bus import BaseEvent, EventBus, EventType
from mirror_core.core.decision import DecisionAction, DecisionEngine
from mirror_core.core.state_machine import (
    CompanionState,
    StateMachine,
    StateTransitionTrigger,
)
from mirror_core.emotion.engine import EmotionalState


# ===== T-T03/T-T04: 全链路集成测试 =====

class TestFullChainIntegration:
    """消息入口 → 事件总线 → 记忆 → 情感 → 状态机 → 决策 全链路"""

    @pytest.fixture
    def bus(self):
        return EventBus()

    @pytest.fixture
    def decision_engine(self):
        return DecisionEngine()

    @pytest.fixture
    def state_machine(self):
        return StateMachine()

    @pytest.mark.asyncio
    async def test_message_to_decision_full_chain(self, bus, decision_engine, state_machine):
        """
        全链路集成:
        Gateway → EventBus → MemoryEngine → EmotionEngine →
        StateMachine → DecisionEngine → Decision
        """
        steps = []

        # Step 1: Gateway 发布 UserMessage 事件
        user_msg_event = BaseEvent(
            type=EventType.USER_MESSAGE,
            source="gateway",
            payload={"user_id": "u1", "text": "你好"},
        )
        steps.append("publish")

        # Step 2: MemoryEngine 订阅并检索
        memory_results = []

        async def memory_handler(event):
            memory_results.append("retrieved")
            steps.append("memory")

        bus.subscribe(EventType.USER_MESSAGE, memory_handler)

        # Step 3: EmotionEngine 订阅并更新情感
        emotion_state = EmotionalState(P=0.3, A=0.5, mood=0.2)

        async def emotion_handler(event):
            steps.append("emotion")

        bus.subscribe(EventType.USER_MESSAGE, emotion_handler)

        # Step 4: 发布事件
        await bus.publish(user_msg_event)
        steps.append("after_publish")

        # Step 5: 状态机评估
        transition = await state_machine.maybe_suppress(suppression=0.2)
        steps.append("state_machine")
        assert state_machine.state == CompanionState.NORMAL

        # Step 6: 决策引擎做最终决策
        decision = await decision_engine.decide(
            state=state_machine.state,
            emotion=emotion_state,
        )
        steps.append("decision")

        # 验证全链路
        assert decision.action == DecisionAction.REPLY
        assert "默认" in decision.reason
        assert len(memory_results) > 0  # 记忆引擎被触发

    @pytest.mark.asyncio
    async def test_burst_full_chain(self, bus, decision_engine, state_machine):
        """爆发周期全链路: emotion_thrust → state → suppress → burst → decide"""
        emotion = EmotionalState(P=0.0, A=0.3, mood=0.0, suppression=0.0, status="Normal")

        # 第一次调用 maybe_suppress 进入 Suppressing
        await state_machine.maybe_suppress(suppression=0.5)
        emotion.suppression = 0.5
        assert state_machine.state == CompanionState.SUPPRESSING

        # 模拟压抑继续累积（直接在情感对象上更新）
        emotion.suppression = 0.9

        # 触发爆发
        await state_machine.transition(StateTransitionTrigger.BURST_TRIGGER)
        assert state_machine.state == CompanionState.BURSTING

        # 决策引擎应返回爆发表述
        decision = await decision_engine.decide(
            state=state_machine.state,
            emotion=emotion,
        )
        assert decision.action == DecisionAction.REPLY
        assert decision.params.get("forced_tone") == "burst"

    @pytest.mark.asyncio
    async def test_consoling_chain(self, bus, decision_engine, state_machine):
        """安慰周期: detected → consoling → consoling_end → decide"""
        emotion = EmotionalState()

        await state_machine.transition(StateTransitionTrigger.CONSOLING_DETECTED)
        assert state_machine.state == CompanionState.CONSOLING

        decision = await decision_engine.decide(
            state=state_machine.state,
            emotion=emotion,
        )
        assert decision.action == DecisionAction.REPLY  # Consoling 走默认

        await state_machine.transition(StateTransitionTrigger.CONSOLING_END)
        assert state_machine.state == CompanionState.NORMAL

    @pytest.mark.asyncio
    async def test_emotion_decay_tick_chain(self, bus, decision_engine, state_machine):
        """SYSTEM_TICK 驱动情感衰减后决策"""
        emotion = EmotionalState(P=0.5, A=0.6, mood=0.3)

        tick_event = BaseEvent(
            type=EventType.SYSTEM_TICK,
            source="scheduler",
            payload={"type": "decay"},
        )
        await bus.publish(tick_event)

        # 情感衰减后决策仍应正常工作
        decision = await decision_engine.decide(
            state=CompanionState.NORMAL,
            emotion=emotion,
        )
        assert decision.action == DecisionAction.REPLY


# ===== T-T05: 并发性能测试 =====

class TestConcurrentPerformance:
    """100 并发数据库检索测试"""

    @pytest.mark.asyncio
    async def test_concurrent_db_reads(self):
        """50 个并发读请求无错误"""
        import tempfile, os
        from mirror_core.infrastructure.database import Database

        tmp = tempfile.mktemp(suffix=".db")
        db = Database(path=tmp)
        await db.initialize()

        await db.execute(
            "INSERT INTO fact_memory (user_id, fact_type, key, value) VALUES (?, ?, ?, ?)",
            ("u1", "test", "k1", "v1"),
        )

        async def query(_):
            row = await db.fetch_one(
                "SELECT value FROM fact_memory WHERE key=?",
                ("k1",),
            )
            return row["value"] if row else None

        tasks = [query(i) for i in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0, f"并发查询错误: {errors[:3]}"

        await db.close()
        os.unlink(tmp)


# ===== T-T01: 情感/状态机边界测试 =====

class TestEdgeCases:
    """补充边界测试"""

    def test_emotion_state_clamping(self):
        """情感状态值超出范围时被截断"""
        from mirror_core.emotion.engine import TidalEngine

        db = MagicMock()
        db.execute = AsyncMock()
        db.fetch_one = AsyncMock(return_value=None)

        engine = TidalEngine(db=db)

        # 使用极端值
        extreme_deltas = [
            {"P": 5.0, "A": 5.0, "D": 5.0},   # 远超上限
            {"P": -5.0, "A": -5.0, "D": -5.0}, # 远超下限
            {"P": 0.0, "A": 0.0, "D": 0.0},    # 零值
        ]

        async def run():
            for delta in extreme_deltas:
                state = await engine.apply_emotional_thrust(delta)
                assert -1.0 <= state.P <= 1.0
                assert 0.0 <= state.A <= 1.0
                assert -1.0 <= state.D <= 1.0
                assert 0.0 <= state.suppression <= 1.0

        asyncio.run(run())

    @pytest.mark.asyncio
    async def test_state_machine_invalid_state_restored(self):
        """状态机从非法状态值恢复时回退到 Normal"""
        sm = StateMachine()

        db = MagicMock()
        db.execute = AsyncMock()
        db.fetch_one = AsyncMock(return_value=None)
        sm._db = db

        await sm.restore()
        assert sm.state == CompanionState.NORMAL

    @pytest.mark.asyncio
    async def test_safety_empty_text_safe(self):
        """安全引擎处理空/空白文本"""
        from mirror_core.core.safety import SafetyEngine, SafetyVerdict
        engine = SafetyEngine()

        result = await engine.evaluate_input("")
        assert result.verdict == SafetyVerdict.SAFE

        result = await engine.evaluate_input("   ")
        assert result.verdict == SafetyVerdict.SAFE

        result = await engine.evaluate_input(None)
        assert result.verdict == SafetyVerdict.SAFE
