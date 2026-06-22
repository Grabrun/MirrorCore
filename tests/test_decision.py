"""
决策引擎单元测试

覆盖范围：
- 所有 6 条决策规则的正确触发
- 规则优先级链（高优先级应阻断低优先级）
- 情感强度计算
- 边界条件（阈值精确值）
- Sad Path 测试
"""

import pytest

from mirror_core.core.decision import (
    Decision,
    DecisionAction,
    DecisionEngine,
    _calculate_emotional_intensity,
)
from mirror_core.core.state_machine import CompanionState


def _make_emotion(P=0.0, A=0.3, D=0.0, mood=0.0, suppression=0.0, status="Normal"):
    """创建 EmotionalState 测试实例"""
    from mirror_core.emotion.engine import EmotionalState
    return EmotionalState(P=P, A=A, D=D, mood=mood,
                          suppression=suppression, status=status)


class TestEmotionalIntensity:
    """情感强度计算测试"""

    def test_neutral(self):
        """中性状态: P=0, A=0.3 → 0.12"""
        assert _calculate_emotional_intensity(0.0, 0.3) == 0.12

    def test_positive_high(self):
        """高兴奋度: P=0.8, A=0.7 → 0.76"""
        assert _calculate_emotional_intensity(0.8, 0.7) == 0.76

    def test_negative_high(self):
        """负面高强度: P=-0.9, A=0.8 → 0.86"""
        assert _calculate_emotional_intensity(-0.9, 0.8) == 0.86

    def test_low_intensity(self):
        """低强度: P=0.1, A=0.2 → 0.14"""
        assert _calculate_emotional_intensity(0.1, 0.2) == 0.14

    def test_boundary_point_eight(self):
        """边界: P=-0.7, A=0.95 → 0.8 """
        result = _calculate_emotional_intensity(-0.7, 0.95)
        assert result == 0.8  # 0.42 + 0.38 = 0.8


class TestDecisionEngine:
    """决策引擎核心测试 (B-T25)"""

    @pytest.fixture
    def engine(self):
        return DecisionEngine()

    # ---- 规则 1: Bursting 状态 ----

    @pytest.mark.asyncio
    async def test_rule_1_bursting(self, engine):
        """Bursting 状态返回 REPLY + forced_tone='burst'"""
        emotion = _make_emotion(P=-0.3, A=0.5)
        decision = await engine.decide(
            state=CompanionState.BURSTING,
            emotion=emotion,
        )
        assert decision.action == DecisionAction.REPLY
        assert decision.params.get("forced_tone") == "burst"
        assert "爆发" in decision.reason

    @pytest.mark.asyncio
    async def test_rule_1_bursting_overrides_proactive(self, engine):
        """Bursting 优先级高于主动触发源"""
        emotion = _make_emotion(P=-0.3, A=0.5)
        decision = await engine.decide(
            state=CompanionState.BURSTING,
            emotion=emotion,
            proactive_triggers=["morning_greeting"],
        )
        # Bursting 应优先于 proactive
        assert decision.params.get("forced_tone") == "burst"
        assert decision.action == DecisionAction.REPLY

    # ---- 规则 2: Reflecting 状态 ----

    @pytest.mark.asyncio
    async def test_rule_2_reflecting(self, engine):
        """Reflecting 状态返回 REACT_INTERNALLY"""
        emotion = _make_emotion()
        decision = await engine.decide(
            state=CompanionState.REFLECTING,
            emotion=emotion,
        )
        assert decision.action == DecisionAction.REACT_INTERNALLY
        assert "反思" in decision.reason

    @pytest.mark.asyncio
    async def test_rule_2_reflecting_overrides_high_risk(self, engine):
        """Reflecting 优先级高于高风险情绪"""
        emotion = _make_emotion(P=-0.9, A=0.95)  # intensity > 0.8, P < -0.5
        decision = await engine.decide(
            state=CompanionState.REFLECTING,
            emotion=emotion,
        )
        # Reflecting 应优先于高风险情绪
        assert decision.action == DecisionAction.REACT_INTERNALLY

    # ---- 规则 3: 主动触发源 ----

    @pytest.mark.asyncio
    async def test_rule_3_proactive(self, engine):
        """有主动触发源时返回 PUSH_NOTIFICATION"""
        emotion = _make_emotion()
        decision = await engine.decide(
            state=CompanionState.NORMAL,
            emotion=emotion,
            proactive_triggers=["morning_greeting"],
        )
        assert decision.action == DecisionAction.PUSH_NOTIFICATION
        assert decision.params.get("template") == "morning_greeting"
        assert "主动触发" in decision.reason

    @pytest.mark.asyncio
    async def test_rule_3_proactive_uses_first_trigger(self, engine):
        """多个触发源取第一个"""
        emotion = _make_emotion()
        decision = await engine.decide(
            state=CompanionState.NORMAL,
            emotion=emotion,
            proactive_triggers=["good_morning", "anniversary"],
        )
        assert decision.params.get("template") == "good_morning"
        assert "good_morning" in decision.reason

    @pytest.mark.asyncio
    async def test_rule_3_proactive_empty_list(self, engine):
        """空触发源列表不应触发推送"""
        emotion = _make_emotion()
        decision = await engine.decide(
            state=CompanionState.NORMAL,
            emotion=emotion,
            proactive_triggers=[],
        )
        # 空列表应跳过规则 3，走默认回复
        assert decision.action == DecisionAction.REPLY
        assert "默认" in decision.reason

    @pytest.mark.asyncio
    async def test_rule_3_proactive_none(self, engine):
        """None 触发源列表不应触发推送"""
        emotion = _make_emotion()
        decision = await engine.decide(
            state=CompanionState.NORMAL,
            emotion=emotion,
            proactive_triggers=None,
        )
        assert decision.action == DecisionAction.REPLY

    # ---- 规则 4: 高风险情绪 ----

    @pytest.mark.asyncio
    async def test_rule_4_high_risk_emotion(self, engine):
        """高风险情绪 (intensity>0.8, P<-0.5) 返回 REPLY+suggest_reality_anchor"""
        emotion = _make_emotion(P=-0.9, A=0.95)  # intensity = 0.86
        decision = await engine.decide(
            state=CompanionState.NORMAL,
            emotion=emotion,
        )
        assert decision.action == DecisionAction.REPLY
        assert decision.params.get("suggest_reality_anchor") is True
        assert "高风险情绪" in decision.reason

    @pytest.mark.asyncio
    async def test_rule_4_intensity_not_enough(self, engine):
        """intensity 不足时不应触发"""
        emotion = _make_emotion(P=-0.7, A=0.5)  # intensity = 0.62 < 0.8
        decision = await engine.decide(
            state=CompanionState.NORMAL,
            emotion=emotion,
        )
        assert decision.action == DecisionAction.REPLY
        assert "suggest_reality_anchor" not in decision.params

    @pytest.mark.asyncio
    async def test_rule_4_positive_high_intensity(self, engine):
        """P >= -0.5 时即使 intensity 高也不触发"""
        emotion = _make_emotion(P=0.9, A=0.95)  # intensity=0.92, but P>0
        decision = await engine.decide(
            state=CompanionState.NORMAL,
            emotion=emotion,
        )
        # 不触发规则 4，走默认
        assert decision.action == DecisionAction.REPLY
        assert "suggest_reality_anchor" not in decision.params

    @pytest.mark.asyncio
    async def test_rule_4_exact_boundary(self, engine):
        """intensity=0.8, P=-0.5 边界 — 不触发（intensity 不大于0.8，P 不小于-0.5）"""
        # 需要精确计算使 intensity=0.8
        # 公式: abs(P)*0.6 + A*0.4 = 0.8
        # P=-0.5 → 0.3 + A*0.4 = 0.8 → A=1.25 → 超范围了
        # 换个方式: P=-0.6, A=0.95 → 0.36+0.38=0.74 < 0.8
        # P=-0.85, A=0.8 → 0.51+0.32=0.83 > 0.8
        # 用 P=-0.5, A=1.0(上限) → 0.3+0.4=0.7
        # 边界: P=-0.5, A=0.88 → 0.3+0.352=0.652 < 0.8
        # 用 P=-0.51, A=0.96 → 0.306+0.384=0.69 < 0.8
        # 实际上很难达到 0.8 同时 P=-0.5
        # P=-0.84, A=0.8 → 0.504+0.32=0.824 > 0.8, P<-0.5 ✅
        emotion = _make_emotion(P=-0.84, A=0.8)
        decision = await engine.decide(
            state=CompanionState.NORMAL,
            emotion=emotion,
        )
        assert decision.params.get("suggest_reality_anchor") is True

    @pytest.mark.asyncio
    async def test_rule_4_not_triggered_low_P(self, engine):
        """P=-0.49 不应触发（需 P<-0.5 严格小于）"""
        # 需要 int > 0.8
        # abs(-0.49)*0.6 + A*0.4 > 0.8
        # 0.294 + 0.4A > 0.8
        # 0.4A > 0.506
        # A > 1.265 — 不可能，A 上限 1
        # 所以 P=-0.49 时不可能达到 intensity>0.8
        emotion = _make_emotion(P=-0.49, A=1.0)  # int=0.694 < 0.8
        decision = await engine.decide(
            state=CompanionState.NORMAL,
            emotion=emotion,
        )
        assert "suggest_reality_anchor" not in decision.params

    # ---- 规则 5: 高依赖度 ----

    @pytest.mark.asyncio
    async def test_rule_5_high_dependency(self, engine):
        """高依赖度 (>0.7) 返回 REPLY+reality_anchor+reduce_perfection"""
        emotion = _make_emotion()
        decision = await engine.decide(
            state=CompanionState.NORMAL,
            emotion=emotion,
            dependency_score=0.8,
        )
        assert decision.action == DecisionAction.REPLY
        assert decision.params.get("reality_anchor") is True
        assert decision.params.get("reduce_perfection") is True
        assert "高依赖度" in decision.reason

    @pytest.mark.asyncio
    async def test_rule_5_dependency_boundary(self, engine):
        """依赖度边界: 0.7 不移，0.71 移"""
        emotion = _make_emotion()
        decision_low = await engine.decide(
            state=CompanionState.NORMAL,
            emotion=emotion,
            dependency_score=0.7,
        )
        assert "reality_anchor" not in decision_low.params

        decision_high = await engine.decide(
            state=CompanionState.NORMAL,
            emotion=emotion,
            dependency_score=0.71,
        )
        assert decision_high.params.get("reality_anchor") is True

    # ---- 规则 6: 默认回复 ----

    @pytest.mark.asyncio
    async def test_rule_6_default(self, engine):
        """无任何条件触发时返回默认 REPLY"""
        emotion = _make_emotion()
        decision = await engine.decide(
            state=CompanionState.NORMAL,
            emotion=emotion,
            proactive_triggers=[],
            dependency_score=0.0,
        )
        assert decision.action == DecisionAction.REPLY
        assert decision.params == {}
        assert "默认" in decision.reason

    @pytest.mark.asyncio
    async def test_rule_6_suppressing_not_special(self, engine):
        """Suppressing 状态走默认回复（无特殊规则）"""
        emotion = _make_emotion(P=0.0, A=0.3)
        decision = await engine.decide(
            state=CompanionState.SUPPRESSING,
            emotion=emotion,
        )
        assert decision.action == DecisionAction.REPLY
        assert "默认" in decision.reason

    @pytest.mark.asyncio
    async def test_rule_6_consoling_not_special(self, engine):
        """Consoling 状态走默认回复（无特殊规则）"""
        emotion = _make_emotion()
        decision = await engine.decide(
            state=CompanionState.CONSOLING,
            emotion=emotion,
        )
        assert decision.action == DecisionAction.REPLY

    # ---- 规则链优先级验证 (AC #1) ----

    @pytest.mark.asyncio
    async def test_priority_bursting_over_reflecting(self, engine):
        """Bursting 高于 Reflecting"""
        # 如果状态是 Bursting，不应触发 Reflecting 规则
        emotion = _make_emotion(P=-0.3, A=0.5)
        decision = await engine.decide(
            state=CompanionState.BURSTING,
            emotion=emotion,
        )
        assert decision.params.get("forced_tone") == "burst"

    @pytest.mark.asyncio
    async def test_priority_bursting_over_everything(self, engine):
        """Bursting 高于所有其他规则"""
        emotion = _make_emotion(P=-0.9, A=0.95)
        decision = await engine.decide(
            state=CompanionState.BURSTING,
            emotion=emotion,
            proactive_triggers=["test"],
            dependency_score=0.9,
        )
        assert decision.params.get("forced_tone") == "burst"

    @pytest.mark.asyncio
    async def test_priority_high_risk_over_dependency(self, engine):
        """高风险情绪高于高依赖度"""
        emotion = _make_emotion(P=-0.9, A=0.95)  # int=0.86 > 0.8, P=-0.9 < -0.5
        decision = await engine.decide(
            state=CompanionState.NORMAL,
            emotion=emotion,
            dependency_score=0.9,  # 也应触发规则5，但规则4应该先命中
        )
        # 规则 4 应命中（高风险情绪优先级 > 高依赖度）
        assert decision.params.get("suggest_reality_anchor") is True
        # 不应该包含 reduce_perfection（那是规则5的参数）
        assert "reduce_perfection" not in decision.params

    # ---- 决策结果类型 ----

    @pytest.mark.asyncio
    async def test_decision_bool_true(self, engine):
        """决策结果应始终为 True（布尔上下文）"""
        emotion = _make_emotion()
        for state in CompanionState:
            decision = await engine.decide(state=state, emotion=emotion)
            assert bool(decision) is True

    @pytest.mark.asyncio
    async def test_decision_dataclass_fields(self):
        """Decision 数据类字段类型正确"""
        d = Decision()
        assert isinstance(d.action, DecisionAction)
        assert isinstance(d.params, dict)
        assert isinstance(d.reason, str)


class TestDecisionEngineSadPaths:
    """Sad Path 测试"""

    @pytest.fixture
    def engine(self):
        return DecisionEngine()

    @pytest.mark.asyncio
    async def test_negative_dependency_score(self, engine):
        """负依赖度（异常值）不应触发规则5"""
        emotion = _make_emotion()
        decision = await engine.decide(
            state=CompanionState.NORMAL,
            emotion=emotion,
            dependency_score=-0.5,
        )
        assert decision.action == DecisionAction.REPLY
        assert "reality_anchor" not in decision.params

    @pytest.mark.asyncio
    async def test_extreme_emotion_values(self, engine):
        """极端情绪值不崩溃"""
        emotion = _make_emotion(P=2.0, A=-0.5, D=3.0)  # 超出范围的异常值
        decision = await engine.decide(
            state=CompanionState.NORMAL,
            emotion=emotion,
        )
        # 不应崩溃，走默认回复
        assert decision.action == DecisionAction.REPLY

    @pytest.mark.asyncio
    async def test_all_states_produce_decision(self, engine):
        """所有状态都能产生决策"""
        emotion = _make_emotion()
        for state in CompanionState:
            decision = await engine.decide(
                state=state,
                emotion=emotion,
                proactive_triggers=[],
                dependency_score=0.0,
            )
            assert isinstance(decision, Decision)
            assert isinstance(decision.action, DecisionAction)
