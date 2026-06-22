"""
安全引擎单元测试

覆盖范围：
- 输入评估 (SAFE / FLAGGED / MODERATE)
- 依赖度评分计算与 EWMA 平滑
- 现实锚点注入
- 配置热加载
- 边界条件与 Sad Path
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirror_core.core.safety import (
    DEFAULT_ANCHORS,
    DEPENDENCY_KEYWORDS,
    HIGH_RISK_KEYWORDS,
    SafetyEngine,
    SafetyResult,
    SafetyVerdict,
    _hash_preview,
)


class TestSafetyVerdict:
    """判定结果枚举测试"""

    def test_values(self):
        assert SafetyVerdict.SAFE.value == "safe"
        assert SafetyVerdict.FLAGGED.value == "flagged"
        assert SafetyVerdict.MODERATE.value == "moderate"

    def test_from_string(self):
        assert SafetyVerdict("safe") == SafetyVerdict.SAFE
        assert SafetyVerdict("flagged") == SafetyVerdict.FLAGGED


class TestEvaluateInput:
    """输入评估测试 (B-T27)"""

    @pytest.fixture
    def engine(self):
        return SafetyEngine()

    @pytest.mark.asyncio
    async def test_safe_input(self, engine):
        """安全输入应返回 SAFE"""
        result = await engine.evaluate_input("今天天气真好")
        assert result.verdict == SafetyVerdict.SAFE
        assert result.reply is None
        assert result.matched_keywords == []

    @pytest.mark.asyncio
    async def test_suicide_keyword_flagged(self, engine):
        """自杀关键词应返回 FLAGGED + 预设回复"""
        result = await engine.evaluate_input("我想自杀")
        assert result.verdict == SafetyVerdict.FLAGGED
        assert result.reply is not None
        assert "热线" in result.reply
        assert "自杀" in result.matched_keywords

    @pytest.mark.asyncio
    async def test_violence_keyword_flagged(self, engine):
        """暴力关键词应返回 FLAGGED"""
        result = await engine.evaluate_input("我要杀了你")
        assert result.verdict == SafetyVerdict.FLAGGED
        assert result.reply is not None
        assert "暴力" in result.reply or "不能" in result.reply

    @pytest.mark.asyncio
    async def test_dependency_keyword_moderate(self, engine):
        """依赖关键词应返回 MODERATE"""
        result = await engine.evaluate_input("只有你懂我")
        assert result.verdict == SafetyVerdict.MODERATE
        assert result.reply is None
        assert "你懂我" in result.matched_keywords or "只有你懂我" in result.matched_keywords

    @pytest.mark.asyncio
    async def test_high_risk_overrides_dependency(self, engine):
        """高风险关键词优先级高于依赖关键词"""
        result = await engine.evaluate_input("只有你懂我，但我想自杀")
        assert result.verdict == SafetyVerdict.FLAGGED
        assert "自杀" in result.matched_keywords

    @pytest.mark.asyncio
    async def test_empty_text(self, engine):
        """空文本返回 SAFE"""
        result = await engine.evaluate_input("")
        assert result.verdict == SafetyVerdict.SAFE

    @pytest.mark.asyncio
    async def test_whitespace_text(self, engine):
        """纯空白文本返回 SAFE"""
        result = await engine.evaluate_input("   \n  ")
        assert result.verdict == SafetyVerdict.SAFE

    @pytest.mark.asyncio
    async def test_near_miss_not_flagged(self, engine):
        """近义词不应误判"""
        result = await engine.evaluate_input("我觉得生活很没意思")
        # "没意思" 匹配 "活着没意思" — 正确
        result2 = await engine.evaluate_input("我觉得很没意思")
        # "很没意思" 不包含完整关键词 "活着没意思"
        assert result2.verdict == SafetyVerdict.SAFE

    @pytest.mark.asyncio
    async def test_multiple_dependency_keywords(self, engine):
        """多条依赖关键词命中"""
        result = await engine.evaluate_input("只有你懂我，别离开我")
        assert result.verdict == SafetyVerdict.MODERATE
        assert len(result.matched_keywords) == 2

    @pytest.mark.asyncio
    async def test_safety_result_dataclass(self):
        """SafetyResult 数据类"""
        result = SafetyResult(
            verdict=SafetyVerdict.FLAGGED,
            reply="safe reply",
            matched_keywords=["test"],
        )
        assert result.verdict == SafetyVerdict.FLAGGED
        assert result.reply == "safe reply"
        assert result.matched_keywords == ["test"]


class TestDependencyScore:
    """依赖度评分测试 (B-T27)"""

    @pytest.fixture
    def engine(self):
        return SafetyEngine()

    @pytest.mark.asyncio
    async def test_baseline_score(self, engine):
        """无匹配条件时依赖度为 0"""
        score = await engine.update_dependency(
            user_id="u1",
            text="今天天气不错",
            session_duration_hours=0.1,
            message_count_24h=5,
            current_hour=14,
        )
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_keyword_increases_score(self, engine):
        """依赖关键词命中增加 0.2"""
        score = await engine.update_dependency(
            user_id="u2",
            text="只有你懂我",
            current_hour=14,
        )
        # score = 0.3 * 0.2 + 0.7 * 0.0 = 0.06
        assert score == pytest.approx(0.06, abs=0.001)

    @pytest.mark.asyncio
    async def test_late_night_long_session(self, engine):
        """深夜+长对话增加 0.15"""
        score = await engine.update_dependency(
            user_id="u3",
            text="你好",
            session_duration_hours=2.0,
            current_hour=23,
        )
        # score = 0.3 * 0.15 + 0.7 * 0.0 = 0.045
        assert score == pytest.approx(0.045, abs=0.001)

    @pytest.mark.asyncio
    async def test_late_night_boundary(self, engine):
        """深夜边界: 22 点不是深夜, 23 点是"""
        score_day = await engine.update_dependency(
            user_id="u4", text="hi", session_duration_hours=2.0, current_hour=22,
        )
        assert score_day == 0.0

        score_night = await engine.update_dependency(
            user_id="u4", text="hi", session_duration_hours=2.0, current_hour=23,
        )
        assert score_night > 0.0

    @pytest.mark.asyncio
    async def test_high_message_count(self, engine):
        """24h 高频消息增加 0.1"""
        score = await engine.update_dependency(
            user_id="u5",
            text="你好",
            message_count_24h=100,
            current_hour=14,
        )
        # score = 0.3 * 0.1 + 0.7 * 0.0 = 0.03
        assert score == pytest.approx(0.03, abs=0.001)

    @pytest.mark.asyncio
    async def test_high_message_count_boundary(self, engine):
        """消息数边界: > 50 触发"""
        score_low = await engine.update_dependency(
            user_id="u6", text="hi", message_count_24h=50, current_hour=14,
        )
        assert score_low == 0.0

        score_high = await engine.update_dependency(
            user_id="u6", text="hi", message_count_24h=51, current_hour=14,
        )
        assert score_high > 0.0

    @pytest.mark.asyncio
    async def test_ewma_smoothing(self, engine):
        """EWMA 平滑: 相邻两次更新后有记忆效应"""
        await engine.update_dependency(
            user_id="u7", text="只有你懂我", current_hour=14,
        )
        score1 = await engine.update_dependency(
            user_id="u7", text="只有你懂我", current_hour=14,
        )
        # 第一次: 0.3*0.2 + 0.7*0.0 = 0.06
        # 第二次: 0.3*0.2 + 0.7*0.06 = 0.06 + 0.042 = 0.102
        assert score1 == pytest.approx(0.102, abs=0.001)

        # 第三次: 应该接近 0.1314
        score2 = await engine.update_dependency(
            user_id="u7", text="只有你懂我", current_hour=14,
        )
        assert score2 > score1

    @pytest.mark.asyncio
    async def test_score_clamped_to_1(self, engine):
        """分数不应超过 1.0"""
        # 反复触发
        score = 0.0
        for _ in range(10):
            score = await engine.update_dependency(
                user_id="u8", text="只有你懂我", current_hour=23,
                session_duration_hours=2.0, message_count_24h=100,
            )
        assert score <= 1.0

    @pytest.mark.asyncio
    async def test_score_not_negative(self, engine):
        """分数不应为负"""
        score = await engine.update_dependency(
            user_id="u9", text="你好", current_hour=14,
        )
        assert score >= 0.0

    @pytest.mark.asyncio
    async def test_combined_factors(self, engine):
        """多因素叠加"""
        score = await engine.update_dependency(
            user_id="u10",
            text="只有你懂我，别离开我",
            session_duration_hours=2.0,
            message_count_24h=100,
            current_hour=23,
        )
        # 关键词: 2条 * 0.2 = 0.4
        # 深夜+长对话: 0.15
        # 高频消息: 0.1
        # total = 0.65
        # smoothed = 0.3 * 0.65 + 0.7 * 0.0 = 0.195
        assert score == pytest.approx(0.195, abs=0.001)


class TestRealityAnchors:
    """现实锚点注入测试 (B-T27)"""

    @pytest.fixture
    def engine(self):
        return SafetyEngine()

    @pytest.mark.asyncio
    async def test_below_threshold(self, engine):
        """低于阈值时不注入"""
        result = await engine.inject_reality_anchor(
            response="今天开心吗？",
            dependency_score=0.5,
        )
        assert result == "今天开心吗？"

    @pytest.mark.asyncio
    async def test_above_threshold(self, engine):
        """高于阈值时注入锚点"""
        result = await engine.inject_reality_anchor(
            response="我在呢",
            dependency_score=0.8,
        )
        assert "我在呢\n" in result
        # 锚点内容匹配
        has_anchor = any(
            anchor in result for anchor in DEFAULT_ANCHORS
        )
        assert has_anchor, f"结果中应包含一个锚点: {result}"

    @pytest.mark.asyncio
    async def test_threshold_boundary(self, engine):
        """阈值边界: 0.7 不移, 0.71 移"""
        result_low = await engine.inject_reality_anchor(
            response="test", dependency_score=0.7,
        )
        assert result_low == "test"

        # 用一个高返回概率的值
        result_high = await engine.inject_reality_anchor(
            response="test", dependency_score=0.71,
        )
        # 由于随机性，多次调用应该至少有一次注入
        found = False
        for _ in range(20):
            r = await engine.inject_reality_anchor(
                response="test", dependency_score=0.71,
            )
            if "test\n" in r:
                found = True
                break
        assert found

    @pytest.mark.asyncio
    async def test_empty_anchors(self, engine):
        """空锚点列表不应注入"""
        engine.update_anchors([])
        result = await engine.inject_reality_anchor(
            response="test", dependency_score=0.9,
        )
        assert result == "test"

    @pytest.mark.asyncio
    async def test_custom_anchors(self):
        """自定义锚点"""
        custom = ["custom anchor 1", "custom anchor 2"]
        engine = SafetyEngine(anchors=custom)
        result = await engine.inject_reality_anchor(
            response="test", dependency_score=0.9,
        )
        assert any(a in result for a in custom)


class TestHotReload:
    """配置热加载测试 (B-T28)"""

    @pytest.fixture
    def engine(self):
        return SafetyEngine()

    def test_update_anchors(self, engine):
        """更新锚点列表"""
        new_anchors = ["新锚点1", "新锚点2"]
        engine.update_anchors(new_anchors)
        assert engine._anchors == new_anchors

    def test_update_high_risk_keywords(self, engine):
        """更新高风险关键词（实例级，不影响其他实例）"""
        engine.update_high_risk_keywords({"新词": "新回复"})
        assert len(engine._high_risk_keywords) == 1
        assert engine._high_risk_keywords["新词"] == "新回复"
        # 验证模块级常量未被修改
        assert "新词" not in HIGH_RISK_KEYWORDS

    def test_update_dependency_keywords(self, engine):
        """更新依赖度关键词（实例级，不影响其他实例）"""
        engine.update_dependency_keywords(["新关键词"])
        assert engine._dependency_keywords == ["新关键词"]
        # 验证模块级常量未被修改
        assert "新关键词" not in DEPENDENCY_KEYWORDS


class TestPersistence:
    """持久化测试"""

    @pytest.fixture
    def db(self):
        db = MagicMock()
        db.execute = AsyncMock()
        db.fetch_one = AsyncMock(return_value=None)
        db.fetch_all = AsyncMock(return_value=[])
        return db

    @pytest.fixture
    def engine(self, db):
        return SafetyEngine(db=db)

    @pytest.mark.asyncio
    async def test_load_default_zero(self, engine, db):
        """无历史数据时默认 0"""
        score = await engine.get_dependency_score("new_user")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_load_from_db(self, db):
        """从数据库加载历史分数"""
        mock_row = MagicMock()
        mock_row.__getitem__.return_value = "0.5"
        db.fetch_one.return_value = mock_row

        engine = SafetyEngine(db=db)
        score = await engine.get_dependency_score("u1")
        assert score == 0.5

    @pytest.mark.asyncio
    async def test_load_invalid_value(self, db):
        """非法值回退到 0"""
        mock_row = MagicMock()
        mock_row.__getitem__.return_value = "not_a_number"
        db.fetch_one.return_value = mock_row

        engine = SafetyEngine(db=db)
        score = await engine.get_dependency_score("u1")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_persist_is_called(self, engine, db):
        """更新依赖度应触发持久化"""
        await engine.update_dependency(
            user_id="u1", text="只有你懂我", current_hour=14,
        )
        # 验证 execute 被调用（持久化）
        assert db.execute.called

    @pytest.mark.asyncio
    async def test_restore_prewarm_cache(self, db):
        """启动时恢复应预热缓存"""
        from unittest.mock import MagicMock
        # 模拟两条历史记录
        mock_row1 = MagicMock()
        mock_row1.__getitem__.side_effect = lambda k: "u1" if k == "user_id" else "0.3"
        mock_row2 = MagicMock()
        mock_row2.__getitem__.side_effect = lambda k: "u2" if k == "user_id" else "0.7"
        db.fetch_all.return_value = [mock_row1, mock_row2]

        engine = SafetyEngine(db=db)
        await engine.restore()
        score1 = await engine.get_dependency_score("u1")
        score2 = await engine.get_dependency_score("u2")
        assert score1 == 0.3
        assert score2 == 0.7

    @pytest.mark.asyncio
    async def test_without_db(self):
        """无数据库时不应崩溃"""
        from mirror_core.core.safety import DEPENDENCY_KEYWORDS
        # 确保依赖关键词列表未被前序测试污染
        assert "只有你懂我" in DEPENDENCY_KEYWORDS

        engine = SafetyEngine()
        score = await engine.update_dependency(
            user_id="u1", text="只有你懂我", current_hour=14,
        )
        assert score == pytest.approx(0.06, abs=0.001)
        assert score > 0
        # 再次查询应走缓存
        score2 = await engine.get_dependency_score("u1")
        assert score2 == score


class TestLateNightCheck:
    """深夜时段检查测试"""

    @pytest.fixture
    def engine(self):
        return SafetyEngine()

    def test_midnight(self, engine):
        """凌晨 0 点"""
        assert engine._is_late_night(0)

    def test_4am(self, engine):
        """凌晨 4 点"""
        assert engine._is_late_night(4)

    def test_5am_boundary(self, engine):
        """5 点不是深夜（[23,5) 开区间）"""
        assert not engine._is_late_night(5)

    def test_11pm(self, engine):
        """23 点是深夜"""
        assert engine._is_late_night(23)

    def test_10pm_not(self, engine):
        """22 点不是深夜"""
        assert not engine._is_late_night(22)

    def test_6am_not(self, engine):
        """6 点不是深夜"""
        assert not engine._is_late_night(6)

    def test_12pm_not(self, engine):
        """12 点不是深夜"""
        assert not engine._is_late_night(12)


class TestHashPreview:
    """哈希预览测试 (F-001: §4.1 日志合规)"""

    def test_hash_consistent(self):
        """相同输入产生相同哈希"""
        h1 = _hash_preview("测试消息123")
        h2 = _hash_preview("测试消息123")
        assert h1 == h2

    def test_hash_different(self):
        """不同输入产生不同哈希"""
        h1 = _hash_preview("消息A")
        h2 = _hash_preview("消息B")
        assert h1 != h2

    def test_hash_length(self):
        """默认哈希长度 8"""
        assert len(_hash_preview("test")) == 8
        assert len(_hash_preview("test", 16)) == 16

    def test_hash_does_not_contain_original(self):
        """哈希不应包含原文"""
        h = _hash_preview("敏感内容123")
        assert "敏感" not in h


class TestSadPaths:
    """异常场景测试"""

    @pytest.fixture
    def engine(self):
        return SafetyEngine()

    @pytest.mark.asyncio
    async def test_extremely_long_input(self, engine):
        """超长输入不崩溃"""
        long_text = "测试" * 10000
        result = await engine.evaluate_input(long_text)
        assert result.verdict == SafetyVerdict.SAFE

    @pytest.mark.asyncio
    async def test_special_characters(self, engine):
        """特殊字符不崩溃"""
        texts = ["你好\n世界", "🤔🌙⭐", "<script>alert(1)</script>"]
        for text in texts:
            result = await engine.evaluate_input(text)
            assert result.verdict == SafetyVerdict.SAFE

    @pytest.mark.asyncio
    async def test_update_dependency_negative_message_count(self, engine):
        """负消息数不崩溃"""
        score = await engine.update_dependency(
            user_id="u1", text="hi", message_count_24h=-5, current_hour=14,
        )
        assert score >= 0.0

    @pytest.mark.asyncio
    async def test_ewma_extreme_values(self, engine):
        """EWMA 极端值不崩溃"""
        # 极高值
        engine._dependency_cache["u1"] = (2.0, time.time())
        score = await engine.update_dependency(
            user_id="u1", text="只有你懂我", current_hour=14,
        )
        assert 0.0 <= score <= 1.0
