"""
状态机单元测试

覆盖范围：
- 所有合法转换 + 非法转换拒绝
- maybe_suppress 动态评估 (B-T24)
- 状态持久化与恢复
- STATE_TRANSITION 事件发布
- 边界场景
"""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from mirror_core.bus import EventType
from mirror_core.core.state_machine import (
    CompanionState,
    StateMachine,
    StateTransition,
    StateTransitionTrigger,
)


class TestCompanionState:
    """状态枚举测试"""

    def test_values(self):
        assert CompanionState.NORMAL.value == "Normal"
        assert CompanionState.SUPPRESSING.value == "Suppressing"
        assert CompanionState.BURSTING.value == "Bursting"
        assert CompanionState.REFLECTING.value == "Reflecting"
        assert CompanionState.CONSOLING.value == "Consoling"

    def test_from_string(self):
        assert CompanionState("Normal") == CompanionState.NORMAL
        assert CompanionState("Bursting") == CompanionState.BURSTING


class TestStateMachineTransitions:
    """核心转换逻辑测试 (B-T23)"""

    @pytest.fixture
    def sm(self):
        """干净的状态机（无总线、无数据库）"""
        return StateMachine()

    @pytest.mark.asyncio
    async def test_initial_state(self, sm):
        """初始状态为 Normal"""
        assert sm.state == CompanionState.NORMAL
        assert sm.is_normal
        assert not sm.is_suppressing
        assert not sm.is_bursting
        assert not sm.is_reflecting
        assert not sm.is_consoling

    @pytest.mark.asyncio
    async def test_normal_to_suppressing(self, sm):
        """Normal → Suppressing (EMOTION_CHANGED, suppression > 0.3)"""
        await sm.transition(
            StateTransitionTrigger.EMOTION_CHANGED,
            suppression=0.5,
        )
        assert sm.state == CompanionState.SUPPRESSING
        assert sm.is_suppressing

    @pytest.mark.asyncio
    async def test_normal_to_suppressing_boundary(self, sm):
        """阈值边界: suppression=0.3 不触发，0.32 触发"""
        # 边界下不移
        await sm.transition(
            StateTransitionTrigger.EMOTION_CHANGED,
            suppression=0.3,
        )
        assert sm.state == CompanionState.NORMAL

        # 边界上触发
        await sm.transition(
            StateTransitionTrigger.EMOTION_CHANGED,
            suppression=0.3001,
        )
        assert sm.state == CompanionState.SUPPRESSING

    @pytest.mark.asyncio
    async def test_suppressing_to_bursting(self, sm):
        """Suppressing → Bursting (BURST_TRIGGER)"""
        await sm.transition(StateTransitionTrigger.EMOTION_CHANGED, suppression=0.5)
        assert sm.state == CompanionState.SUPPRESSING

        await sm.transition(StateTransitionTrigger.BURST_TRIGGER)
        assert sm.state == CompanionState.BURSTING
        assert sm.is_bursting

    @pytest.mark.asyncio
    async def test_bursting_to_reflecting(self, sm):
        """Bursting → Reflecting (BURST_END)"""
        await sm.transition(StateTransitionTrigger.EMOTION_CHANGED, suppression=0.5)
        await sm.transition(StateTransitionTrigger.BURST_TRIGGER)
        await sm.transition(StateTransitionTrigger.BURST_END)

        assert sm.state == CompanionState.REFLECTING
        assert sm.is_reflecting

    @pytest.mark.asyncio
    async def test_reflecting_to_normal(self, sm):
        """Reflecting → Normal (REFLECT_TIMEOUT)"""
        await sm.transition(StateTransitionTrigger.EMOTION_CHANGED, suppression=0.5)
        await sm.transition(StateTransitionTrigger.BURST_TRIGGER)
        await sm.transition(StateTransitionTrigger.BURST_END)
        await sm.transition(StateTransitionTrigger.REFLECT_TIMEOUT)

        assert sm.state == CompanionState.NORMAL

    @pytest.mark.asyncio
    async def test_normal_to_consoling(self, sm):
        """Normal → Consoling (CONSOLING_DETECTED)"""
        await sm.transition(StateTransitionTrigger.CONSOLING_DETECTED)
        assert sm.state == CompanionState.CONSOLING
        assert sm.is_consoling

    @pytest.mark.asyncio
    async def test_consoling_to_normal(self, sm):
        """Consoling → Normal (CONSOLING_END)"""
        await sm.transition(StateTransitionTrigger.CONSOLING_DETECTED)
        await sm.transition(StateTransitionTrigger.CONSOLING_END)

        assert sm.state == CompanionState.NORMAL

    @pytest.mark.asyncio
    async def test_full_cycle_burst(self, sm):
        """完整爆发周期: Normal → Suppressing → Bursting → Reflecting → Normal"""
        # Normal
        assert sm.state == CompanionState.NORMAL

        # → Suppressing
        await sm.transition(StateTransitionTrigger.EMOTION_CHANGED, suppression=0.5)
        assert sm.state == CompanionState.SUPPRESSING

        # → Bursting
        await sm.transition(StateTransitionTrigger.BURST_TRIGGER)
        assert sm.state == CompanionState.BURSTING

        # → Reflecting
        await sm.transition(StateTransitionTrigger.BURST_END)
        assert sm.state == CompanionState.REFLECTING

        # → Normal
        await sm.transition(StateTransitionTrigger.REFLECT_TIMEOUT)
        assert sm.state == CompanionState.NORMAL

    @pytest.mark.asyncio
    async def test_full_cycle_consoling(self, sm):
        """安慰周期: Normal → Consoling → Normal"""
        assert sm.state == CompanionState.NORMAL

        await sm.transition(StateTransitionTrigger.CONSOLING_DETECTED)
        assert sm.state == CompanionState.CONSOLING

        await sm.transition(StateTransitionTrigger.CONSOLING_END)
        assert sm.state == CompanionState.NORMAL


class TestInvalidTransitions:
    """非法转换测试"""

    @pytest.fixture
    def sm(self):
        return StateMachine()

    @pytest.mark.asyncio
    async def test_invalid_suppressing_to_consoling(self, sm):
        """Suppressing → Consoling 是非法转换"""
        await sm.transition(StateTransitionTrigger.EMOTION_CHANGED, suppression=0.5)
        with pytest.raises(ValueError, match="非法状态转换"):
            await sm.transition(StateTransitionTrigger.CONSOLING_DETECTED)

    @pytest.mark.asyncio
    async def test_invalid_bursting_to_suppressing(self, sm):
        """Bursting → Suppressing 非法"""
        await sm.transition(StateTransitionTrigger.EMOTION_CHANGED, suppression=0.5)
        await sm.transition(StateTransitionTrigger.BURST_TRIGGER)
        with pytest.raises(ValueError, match="非法状态转换"):
            await sm.transition(StateTransitionTrigger.EMOTION_CHANGED, suppression=0.1)

    @pytest.mark.asyncio
    async def test_invalid_normal_to_bursting(self, sm):
        """Normal → Bursting 非法（必须先经过 Suppressing）"""
        with pytest.raises(ValueError, match="非法状态转换"):
            await sm.transition(StateTransitionTrigger.BURST_TRIGGER)

    @pytest.mark.asyncio
    async def test_invalid_reflecting_to_consoling(self, sm):
        """Reflecting → Consoling 非法"""
        await sm.transition(StateTransitionTrigger.EMOTION_CHANGED, suppression=0.5)
        await sm.transition(StateTransitionTrigger.BURST_TRIGGER)
        await sm.transition(StateTransitionTrigger.BURST_END)
        with pytest.raises(ValueError, match="非法状态转换"):
            await sm.transition(StateTransitionTrigger.CONSOLING_DETECTED)


class TestMaybeSuppress:
    """动态转换条件评估测试 (B-T24)"""

    @pytest.fixture
    def sm(self):
        return StateMachine()

    @pytest.mark.asyncio
    async def test_maybe_suppress_triggered(self, sm):
        """suppression > 0.3 触发 Suppressing"""
        result = await sm.maybe_suppress(suppression=0.5)
        assert result.to_state == CompanionState.SUPPRESSING
        assert sm.state == CompanionState.SUPPRESSING
        assert result.is_change

    @pytest.mark.asyncio
    async def test_maybe_suppress_not_triggered(self, sm):
        """suppression ≤ 0.3 保持 Normal"""
        result = await sm.maybe_suppress(suppression=0.2)
        assert result.to_state == CompanionState.NORMAL
        assert sm.state == CompanionState.NORMAL
        assert not result.is_change  # 没有实际变化

    @pytest.mark.asyncio
    async def test_maybe_suppress_boundary(self, sm):
        """边界值精确测试"""
        # 0.3 不移
        await sm.maybe_suppress(suppression=0.3)
        assert sm.state == CompanionState.NORMAL

        # 略高于 0.3 移
        await sm.maybe_suppress(suppression=0.31)
        assert sm.state == CompanionState.SUPPRESSING


class TestCanTransitionTo:
    """状态可达性测试"""

    @pytest.fixture
    def sm(self):
        return StateMachine()

    def test_normal_can_reach_suppressing(self, sm):
        """从 Normal 可达 Suppressing"""
        assert sm.can_transition_to(CompanionState.SUPPRESSING)

    def test_normal_can_reach_consoling(self, sm):
        """从 Normal 可达 Consoling"""
        assert sm.can_transition_to(CompanionState.CONSOLING)

    def test_normal_cannot_reach_bursting_directly(self, sm):
        """从 Normal 不可达 Bursting（需经过 Suppressing）"""
        assert not sm.can_transition_to(CompanionState.BURSTING)

    def test_normal_cannot_reach_reflecting_directly(self, sm):
        """从 Normal 不可达 Reflecting"""
        assert not sm.can_transition_to(CompanionState.REFLECTING)

    def test_normal_can_reach_normal_via_maybe_suppress(self, sm):
        """
        从 Normal 可达 Normal（maybe_suppress 评估结果为 Normal 时）
        修复 F-004: 验证 can_transition_to 正确处理动态评估的两种可能结果
        """
        assert sm.can_transition_to(CompanionState.NORMAL)

    @pytest.mark.asyncio
    async def test_suppressing_reaches_bursting(self, sm):
        """从 Suppressing 可达 Bursting"""
        await sm.transition(StateTransitionTrigger.EMOTION_CHANGED, suppression=0.5)
        assert sm.can_transition_to(CompanionState.BURSTING)


class TestTransitionRecord:
    """转换记录测试"""

    @pytest.fixture
    def sm(self):
        return StateMachine()

    @pytest.mark.asyncio
    async def test_transition_record_content(self, sm):
        """转换记录包含 from/to/trigger/timestamp/metadata"""
        result = await sm.transition(
            StateTransitionTrigger.EMOTION_CHANGED,
            suppression=0.5,
        )
        assert result.from_state == CompanionState.NORMAL
        assert result.to_state == CompanionState.SUPPRESSING
        assert result.trigger == StateTransitionTrigger.EMOTION_CHANGED
        assert result.timestamp > 0
        assert result.metadata.get("suppression") == 0.5

    @pytest.mark.asyncio
    async def test_last_transition_property(self, sm):
        """last_transition 属性记录最近一次转换"""
        assert sm.last_transition is None

        await sm.transition(
            StateTransitionTrigger.EMOTION_CHANGED,
            suppression=0.5,
        )
        assert sm.last_transition is not None
        assert sm.last_transition.to_state == CompanionState.SUPPRESSING

    @pytest.mark.asyncio
    async def test_same_state_transition(self, sm):
        """相同状态转换不应修改 last_transition"""
        result = await sm.maybe_suppress(suppression=0.0)
        assert not result.is_change
        # last_transition 仍为 None，因为没有有效的状态变更
        assert sm.last_transition is None


class TestEventPublishing:
    """STATE_TRANSITION 事件发布测试 (AC #2)"""

    @pytest.fixture
    def bus(self):
        bus = MagicMock()
        bus.publish = AsyncMock()
        return bus

    @pytest.fixture
    def sm(self, bus):
        return StateMachine(bus=bus)

    @pytest.mark.asyncio
    async def test_publishes_on_state_change(self, sm, bus):
        """状态变更时发布 STATE_TRANSITION 事件"""
        await sm.transition(StateTransitionTrigger.EMOTION_CHANGED, suppression=0.5)

        # 验证 publish 被调用
        bus.publish.assert_awaited_once()

        # 验证事件内容
        event = bus.publish.await_args.args[0]
        assert event.type == EventType.STATE_TRANSITION
        assert event.source == "state_machine"
        assert event.payload["from_state"] == "Normal"
        assert event.payload["to_state"] == "Suppressing"
        assert event.payload["trigger"] == "emotion_changed"

    @pytest.mark.asyncio
    async def test_no_publish_on_same_state(self, sm, bus):
        """未改变状态时不应发布事件"""
        await sm.maybe_suppress(suppression=0.0)
        bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_publish_without_bus(self):
        """没有总线时不应崩溃"""
        sm = StateMachine()  # bus=None
        await sm.transition(StateTransitionTrigger.EMOTION_CHANGED, suppression=0.5)
        assert sm.state == CompanionState.SUPPRESSING


class TestPersistence:
    """状态持久化测试 (AC #3)"""

    @pytest.fixture
    def db(self):
        db = MagicMock()
        db.execute = AsyncMock()
        db.fetch_one = AsyncMock()
        return db

    @pytest.fixture
    def sm(self, db):
        return StateMachine(bus=None, db=db)

    @pytest.mark.asyncio
    async def test_persist_on_transition(self, sm, db):
        """状态变更后自动持久化"""
        await sm.transition(StateTransitionTrigger.EMOTION_CHANGED, suppression=0.5)

        # 验证持久化写入（检查 SQL 中包含 INSERT INTO semantic_memory）
        call_args_list = [(c.args, c.kwargs) for c in db.execute.call_args_list]
        sql_args = [args for args, _ in call_args_list]
        found = False
        for sql, params in sql_args:
            if "INSERT INTO semantic_memory" in sql:
                assert "Suppressing" in params
                found = True
                break
        assert found, "未找到持久化调用 (semantic_memory)"

    @pytest.mark.asyncio
    async def test_restore_normal(self, sm, db):
        """数据库中没有状态时恢复为 Normal"""
        db.fetch_one.return_value = None
        await sm.restore()
        assert sm.state == CompanionState.NORMAL

    @pytest.mark.asyncio
    async def test_restore_saved_state(self, sm, db):
        """从数据库恢复持久化状态"""
        mock_row = MagicMock()
        mock_row.__getitem__.return_value = "Bursting"
        db.fetch_one.return_value = mock_row

        await sm.restore()
        assert sm.state == CompanionState.BURSTING

    @pytest.mark.asyncio
    async def test_restore_invalid_state(self, sm, db):
        """恢复非法状态值时回退到 Normal"""
        mock_row = MagicMock()
        mock_row.__getitem__.return_value = "NonExistentState"
        db.fetch_one.return_value = mock_row

        await sm.restore()
        assert sm.state == CompanionState.NORMAL

    @pytest.mark.asyncio
    async def test_force_persist(self, sm, db):
        """强制持久化"""
        await sm.force_persist()
        db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_without_db(self):
        """没有数据库时不应崩溃"""
        sm = StateMachine()  # db=None
        await sm.transition(StateTransitionTrigger.EMOTION_CHANGED, suppression=0.5)
        assert sm.state == CompanionState.SUPPRESSING


class TestRealDBPersistence:
    """真实 SQLite 集成测试 (F-008)"""

    @pytest.fixture
    async def real_db(self):
        """创建临时数据库并运行所有迁移"""
        from mirror_core.infrastructure.database import Database
        tmp = tempfile.mktemp(suffix=".db")
        db = Database(path=tmp)
        await db.initialize()
        yield db
        await db.close()
        os.unlink(tmp)

    @pytest.fixture
    async def real_sm(self, real_db):
        sm = StateMachine(bus=None, db=real_db)
        await sm.restore()
        return sm

    @pytest.mark.asyncio
    async def test_persist_and_restore_roundtrip(self, real_sm, real_db):
        """持久化 → 恢复的完整往返"""
        # 初始状态
        assert real_sm.state == CompanionState.NORMAL

        # 执行状态转换
        await real_sm.transition(
            StateTransitionTrigger.EMOTION_CHANGED,
            suppression=0.5,
        )
        assert real_sm.state == CompanionState.SUPPRESSING

        # 验证数据库中有数据
        row = await real_db.fetch_one(
            "SELECT companion_state FROM semantic_memory WHERE user_id=?",
            ("default",),
        )
        assert row is not None
        assert row["companion_state"] == "Suppressing"

        # 创建新的状态机实例并恢复
        sm2 = StateMachine(bus=None, db=real_db)
        await sm2.restore()
        assert sm2.state == CompanionState.SUPPRESSING

    @pytest.mark.asyncio
    async def test_multiple_transitions_persist(self, real_sm, real_db):
        """多次转换后数据库持续更新"""
        # Normal → Suppressing
        await real_sm.transition(
            StateTransitionTrigger.EMOTION_CHANGED, suppression=0.5
        )
        row1 = await real_db.fetch_one(
            "SELECT companion_state FROM semantic_memory WHERE user_id=?",
            ("default",),
        )
        assert row1["companion_state"] == "Suppressing"

        # Suppressing → Bursting
        await real_sm.transition(StateTransitionTrigger.BURST_TRIGGER)
        row2 = await real_db.fetch_one(
            "SELECT companion_state FROM semantic_memory WHERE user_id=?",
            ("default",),
        )
        assert row2["companion_state"] == "Bursting"

    @pytest.mark.asyncio
    async def test_no_db_no_crash(self):
        """没有数据库时完整操作不崩溃"""
        sm = StateMachine(bus=None, db=None)
        await sm.restore()
        assert sm.state == CompanionState.NORMAL

        await sm.transition(
            StateTransitionTrigger.EMOTION_CHANGED, suppression=0.5
        )
        assert sm.state == CompanionState.SUPPRESSING

        await sm.force_persist()  # 不应崩溃
