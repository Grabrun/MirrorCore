"""
主动陪伴模块单元测试

覆盖范围：
- 触发条件评估 (B-T40)
- 静音时段逻辑 (B-T41)
- 频率限制 (B-T41)
- 模板选择
- 事件发布
- 边界场景
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirror_core.core.proactive import (
    GOOD_MORNING_TEMPLATES,
    GOOD_NIGHT_TEMPLATES,
    ProactiveConfig,
    ProactiveManager,
    SILENCE_CONCERN_TEMPLATES,
    TRIGGER_TYPES,
)
from mirror_core.emotion.engine import EmotionalState


@pytest.fixture
def bus():
    b = MagicMock()
    b.publish = AsyncMock()
    return b


@pytest.fixture
def memory():
    m = MagicMock()
    m.retrieve = AsyncMock(return_value=[])
    m.get_fact = AsyncMock(return_value=None)
    return m


@pytest.fixture
def manager(bus, memory):
    return ProactiveManager(bus=bus, memory_engine=memory)


class TestTriggerTypes:
    """触发类型常量测试"""

    def test_all_triggers_defined(self):
        assert len(TRIGGER_TYPES) == 6
        assert "good_morning" in TRIGGER_TYPES
        assert "good_night" in TRIGGER_TYPES
        assert "anniversary" in TRIGGER_TYPES
        assert "mood_improved" in TRIGGER_TYPES
        assert "silence_concern" in TRIGGER_TYPES
        assert "night_metaphor" in TRIGGER_TYPES


class TestQuietHours:
    """静音时段测试 (B-T41)"""

    @pytest.fixture
    def mgr(self):
        return ProactiveManager(config=ProactiveConfig(quiet_hours=(22, 7)))

    def test_in_quiet_hours_midnight(self, mgr):
        assert mgr.is_in_quiet_hours(0)

    def test_in_quiet_hours_late_night(self, mgr):
        assert mgr.is_in_quiet_hours(23)

    def test_not_in_quiet_hours_morning(self, mgr):
        assert not mgr.is_in_quiet_hours(8)

    def test_not_in_quiet_hours_afternoon(self, mgr):
        assert not mgr.is_in_quiet_hours(14)

    def test_edge_6am_is_quiet(self, mgr):
        """6 点在静音区 [22,7) 内"""
        assert mgr.is_in_quiet_hours(6)

    def test_edge_7am_not_quiet(self, mgr):
        """7 点不在静音区 [22,7)"""
        assert not mgr.is_in_quiet_hours(7)

    def test_edge_22pm_is_quiet(self, mgr):
        """22 点在静音区"""
        assert mgr.is_in_quiet_hours(22)

    def test_custom_quiet_hours_same_day(self):
        """同一天不跨日的静音区间 (13, 14)"""
        mgr = ProactiveManager(config=ProactiveConfig(quiet_hours=(13, 14)))
        assert mgr.is_in_quiet_hours(13)
        assert not mgr.is_in_quiet_hours(12)
        assert not mgr.is_in_quiet_hours(14)

    def test_filter_quiet_blocks_regular(self, manager):
        """静音时段阻断非例外触发"""
        result = manager._filter_quiet_hours(
            ["good_morning", "good_night"], current_hour=23,
        )
        assert result == []

    def test_filter_quiet_allows_anniversary(self, manager):
        """静音时段允许纪念日穿透"""
        result = manager._filter_quiet_hours(
            ["anniversary", "good_morning"], current_hour=23,
        )
        assert result == ["anniversary"]

    def test_filter_not_quiet_passes_all(self, manager):
        """非静音时段全部通过"""
        result = manager._filter_quiet_hours(
            ["good_morning", "good_night"], current_hour=14,
        )
        assert len(result) == 2


class TestEvaluate:
    """触发条件评估测试 (B-T40)"""

    @pytest.fixture
    def mgr(self, bus, memory):
        return ProactiveManager(bus=bus, memory_engine=memory)

    @pytest.mark.asyncio
    async def test_evaluate_no_triggers(self, mgr):
        """无触发条件满足时返回空列表"""
        with patch("mirror_core.core.proactive._time.localtime") as mock_time:
            mock_tm = MagicMock()
            mock_tm.tm_hour = 14
            mock_tm.tm_yday = 100
            mock_time.return_value = mock_tm

            triggers = await mgr.evaluate(user_id="u1")
            assert triggers == []

    @pytest.mark.asyncio
    async def test_evaluate_max_per_day(self, mgr):
        """超过每日上限后不再触发"""
        mgr._daily_count = 100  # 超过 max_per_day=5
        triggers = await mgr.evaluate(user_id="u1")
        assert triggers == []

    @pytest.mark.asyncio
    async def test_night_metaphor_triggered(self, mgr):
        """深夜时段触发隐喻"""
        with patch("mirror_core.core.proactive._time.localtime") as mock_time:
            mock_tm = MagicMock()
            mock_tm.tm_hour = 23
            mock_tm.tm_yday = 100
            mock_time.return_value = mock_tm

            mgr._config.physiological_metaphor = True
            mgr._config.metaphor_base_probability = 1.0  # 确保触发

            triggers = await mgr.evaluate(user_id="u1")
            # night_metaphor 在静音时段应穿透
            assert "night_metaphor" in triggers

    @pytest.mark.asyncio
    async def test_night_metaphor_disabled(self, mgr):
        """禁用后不触发"""
        with patch("mirror_core.core.proactive._time.localtime") as mock_time:
            mock_tm = MagicMock()
            mock_tm.tm_hour = 23
            mock_tm.tm_yday = 100
            mock_time.return_value = mock_tm

            mgr._config.physiological_metaphor = False
            triggers = await mgr.evaluate(user_id="u1")
            assert "night_metaphor" not in triggers

    @pytest.mark.asyncio
    async def test_silence_concern_detected(self, mgr, memory):
        """长期静默检测"""
        with patch("mirror_core.core.proactive._time.localtime") as mock_time:
            mock_tm = MagicMock()
            mock_tm.tm_hour = 14
            mock_tm.tm_yday = 100
            mock_time.return_value = mock_tm

        # 模拟最后活跃时间在很久以前
        old_time = time.time() - 86400 * 10  # 10 天前

        # 模拟 retrieve 返回旧数据
        from mirror_core.memory.engine import EpisodicMemory
        memory.retrieve.return_value = [
            EpisodicMemory(
                id="old", user_id="u1", summary="old",
                timestamp=old_time,
            )
        ]

        triggers = await mgr.evaluate(user_id="u1")
        assert "silence_concern" in triggers


class TestDailyLimit:
    """每日频率限制测试 (B-T41)"""

    def test_daily_reset(self):
        """跨天重置计数"""
        mgr = ProactiveManager()
        mgr._daily_count = 5

        # 模拟不同天
        with patch("mirror_core.core.proactive._time.localtime") as mock_time:
            mock_tm = MagicMock()
            mock_tm.tm_yday = 200
            mock_time.return_value = mock_tm

            mgr._check_daily_reset()
            assert mgr._daily_count == 0

    def test_same_day_no_reset(self):
        """同一天不重置"""
        mgr = ProactiveManager()
        mgr._daily_count = 5
        mgr._last_reset_day = 100

        with patch("mirror_core.core.proactive._time.localtime") as mock_time:
            mock_tm = MagicMock()
            mock_tm.tm_yday = 100
            mock_time.return_value = mock_tm

            mgr._check_daily_reset()
            assert mgr._daily_count == 5

    def test_record_trigger_increments(self):
        """record_trigger 增加计数"""
        mgr = ProactiveManager()
        mgr.record_trigger("good_morning")
        assert mgr._daily_count == 1


class TestTemplates:
    """模板消息测试"""

    def test_good_morning_templates(self):
        assert len(GOOD_MORNING_TEMPLATES) > 0
        assert all(isinstance(t, str) for t in GOOD_MORNING_TEMPLATES)

    def test_good_night_templates(self):
        assert len(GOOD_NIGHT_TEMPLATES) > 0

    def test_silence_concern_templates(self):
        assert len(SILENCE_CONCERN_TEMPLATES) > 0

    def test_get_random_template(self):
        for trigger in TRIGGER_TYPES:
            template = ProactiveManager._get_random_template(trigger)
            if trigger != "mood_improved" and trigger != "anniversary":
                assert template is not None, f"{trigger} 应有模板"
                assert isinstance(template, str)

    def test_build_template_payload(self, manager):
        """构建模板载荷"""
        payload = manager._build_template_payload(["good_morning", "good_night"])
        assert "good_morning" in payload
        assert "good_night" in payload
        assert isinstance(payload["good_morning"], str)


class TestEventPublishing:
    """事件发布测试"""

    @pytest.mark.asyncio
    async def test_publish_called(self, manager, bus):
        """发布 PROACTIVE_CHANCE 事件"""
        await manager.publish_proactive_event(["good_morning"])
        bus.publish.assert_awaited_once()
        event = bus.publish.await_args.args[0]
        assert event.type.value == "proactive_chance"
        assert event.source == "proactive_manager"
        assert "good_morning" in event.payload["triggers"]

    @pytest.mark.asyncio
    async def test_publish_empty_no_op(self, manager, bus):
        """空触发列表不发布"""
        await manager.publish_proactive_event([])
        bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_publish_without_bus(self):
        """无总线时不崩溃"""
        mgr = ProactiveManager(bus=None)
        await mgr.publish_proactive_event(["good_morning"])


class TestEmotionEvaluation:
    """情感状态评估测试"""

    @pytest.mark.asyncio
    async def test_mood_improved_detected(self, manager):
        """心境从负面转正时触发"""
        # 设置上一轮心境为负
        manager._last_mood = -0.6

        emotion = EmotionalState(P=0.3, A=0.4, mood=0.1)
        triggers = manager.evaluate_with_emotion(user_id="u1", emotion=emotion)
        assert "mood_improved" in triggers

    @pytest.mark.asyncio
    async def test_mood_not_improved(self, manager):
        """心境没有显著改善时不触发"""
        manager._last_mood = -0.3  # > -0.5，不算"大幅回升"
        emotion = EmotionalState(P=0.0, mood=-0.1)
        triggers = manager.evaluate_with_emotion(user_id="u1", emotion=emotion)
        assert "mood_improved" not in triggers

    @pytest.mark.asyncio
    async def test_mood_over_limit(self, manager):
        """超过每日上限后不触发心境好转"""
        from unittest.mock import patch, MagicMock
        with patch("mirror_core.core.proactive._time.localtime") as mock_time:
            mock_tm = MagicMock()
            mock_tm.tm_yday = 100
            mock_time.return_value = mock_tm
            manager._daily_count = 100
            manager._last_reset_day = 100  # 阻止重置
            manager._last_mood = -0.6
            emotion = EmotionalState(P=0.3, mood=0.1)
            triggers = manager.evaluate_with_emotion(user_id="u1", emotion=emotion)
            assert triggers == []


class TestConfig:
    """配置测试"""

    def test_default_config(self):
        config = ProactiveConfig()
        assert config.enabled is True
        assert config.max_per_day == 5
        assert config.quiet_hours == (22, 7)
        assert config.silence_threshold_days == 3
        assert config.physiological_metaphor is True
        assert config.metaphor_base_probability == 0.25
        assert config.metaphor_max_per_night == 2

    def test_custom_config(self):
        config = ProactiveConfig(max_per_day=3, quiet_hours=(23, 7))
        assert config.max_per_day == 3
        assert config.quiet_hours == (23, 7)
