"""
决策引擎 (DecisionEngine)

User Story 9：作为系统，我需要决策引擎，以根据当前内外部条件
做出最终的动作决策 (回复/推送/内部反应)。

B-T25: 决策规则链与优先级判断逻辑
B-T26: 集成主动触发源的评估结果到决策流程

设计文档 §3.3.5:

决策动作:
    REPLY             — 生成回复（可附加爆发语气、现实锚点）
    REACT_INTERNALLY  — 仅内部更新情绪，不输出回复
    PUSH_NOTIFICATION — 主动向用户发送消息
    LOG_MEMORY        — 将当前交互摘要写入长期记忆

决策规则优先级 (从高到低):
    1. Bursting 状态 → REPLY (forced_tone='burst')
    2. Reflecting 状态 → REACT_INTERNALLY (沉默反思)
    3. 有主动触发源 → PUSH_NOTIFICATION
    4. 高风险情绪 (intensity>0.8 且 P<-0.5) → REPLY (suggest_reality_anchor)
    5. 高依赖度 (>0.7) → REPLY (reality_anchor, reduce_perfection)
    6. 默认 → REPLY
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from mirror_core.core.context import AssembledContext
from mirror_core.core.state_machine import CompanionState

logger = logging.getLogger("mirror_core.core.decision")


class DecisionAction(str, Enum):
    """决策动作类型"""
    REPLY = "REPLY"                         # 生成回复
    REACT_INTERNALLY = "REACT_INTERNALLY"   # 仅更新内部状态
    PUSH_NOTIFICATION = "PUSH_NOTIFICATION" # 主动推送通知
    LOG_MEMORY = "LOG_MEMORY"               # 写入长期记忆（暂未使用，预留未来功能）


@dataclass
class Decision:
    """
    决策结果。

    Attributes:
        action: 决策动作
        params: 动作参数
            REPLY:       forced_tone, suggest_reality_anchor, reduce_perfection
            PUSH_NOTIFICATION: template (主动消息模板)
            REACT_INTERNALLY / LOG_MEMORY:  (无特定参数)
        reason: 决策原因（用于调试/日志）
    """
    action: DecisionAction = DecisionAction.REPLY
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def __bool__(self) -> bool:
        """方便测试中断言是否有决策产生"""
        return True


def _calculate_emotional_intensity(P: float, A: float) -> float:
    """
    计算情感强度（从 PAD 状态推算）。

    设计文档 §3.3.5 决策规则中使用的 emotion.intensity。
    由于 EmotionalState 不保存当前强度，从 P(愉悦度)和 A(激活度)推算：
        intensity = |P| * 0.6 + A * 0.4
    范围 [0, 1]，高强度信号（|P|大 + A高）得分高。

    Returns:
        情感强度值 [0, 1]
    """
    return round(abs(P) * 0.6 + A * 0.4, 3)


class DecisionEngine:
    """
    决策引擎。

    根据当前状态、情绪、外部触发等条件，按优先级链生成最终动作决策。

    规则优先级 (§3.3.5.3):
        BURSTING > REFLECTING > 主动触发 > 高风险情绪 >
        高依赖度 > 默认回复
    """

    # 决策原因描述
    _REASONS = {
        "bursting": "爆发状态: 需生成爆发语气回复",
        "reflecting": "反思状态: 静默内部处理",
        "proactive": "主动触发源: 需要推送通知",
        "high_risk_emotion": "高风险情绪: 高强度负面情绪, 需注入现实锚点",
        "high_dependency": "高依赖度: 超过阈值, 需注入现实锚点并降低回复完美度",
        "default": "默认回复",
    }

    async def decide(
        self,
        context: Optional[AssembledContext] = None,
        state: CompanionState = CompanionState.NORMAL,
        emotion: "EmotionalState" = None,  # type: ignore[name-defined]
        proactive_triggers: Optional[List[str]] = None,
        dependency_score: float = 0.0,
    ) -> Decision:
        """
        根据当前条件执行决策规则链。

        规则按优先级从高到低逐一评估，命中第一条即返回。

        Args:
            context: 组装上下文（设计文档 §3.3.5 接口要求，当前规则链未使用）
            state: 当前伴侣状态 (CompanionState，默认 Normal)
            emotion: 当前情感状态 (EmotionalState)
            proactive_triggers: 主动触发源列表（来自 ProactiveManager，可选）
            dependency_score: 依赖度评分 [0, 1]（来自 SafetyEngine，默认 0）

        Returns:
            决策结果 (Decision)
        """
        if emotion is None:
            from mirror_core.emotion.engine import EmotionalState
            emotion = EmotionalState()
        proactive_triggers = proactive_triggers or []
        intensity = _calculate_emotional_intensity(emotion.P, emotion.A)

        # 规则 1: BURSTING 状态 → REPLY (爆发语气)
        if state == CompanionState.BURSTING:
            return Decision(
                action=DecisionAction.REPLY,
                params={"forced_tone": "burst"},
                reason=self._REASONS["bursting"],
            )

        # 规则 2: REFLECTING 状态 → REACT_INTERNALLY (沉默反思)
        if state == CompanionState.REFLECTING:
            return Decision(
                action=DecisionAction.REACT_INTERNALLY,
                reason=self._REASONS["reflecting"],
            )

        # 规则 3: 有主动触发源 → PUSH_NOTIFICATION (B-T26)
        if proactive_triggers:
            return Decision(
                action=DecisionAction.PUSH_NOTIFICATION,
                params={"template": proactive_triggers[0]},
                reason=f"{self._REASONS['proactive']}: {proactive_triggers[0]}",
            )

        # 规则 4: 高风险情绪 (intensity > 0.8 且 P < -0.5)
        if intensity > 0.8 and emotion.P < -0.5:
            return Decision(
                action=DecisionAction.REPLY,
                params={"suggest_reality_anchor": True},
                reason=self._REASONS["high_risk_emotion"],
            )

        # 规则 5: 高依赖度 (> 0.7)
        if dependency_score > 0.7:
            return Decision(
                action=DecisionAction.REPLY,
                params={
                    "reality_anchor": True,
                    "reduce_perfection": True,
                },
                reason=self._REASONS["high_dependency"],
            )

        # 规则 6 (默认): REPLY
        return Decision(
            action=DecisionAction.REPLY,
            reason=self._REASONS["default"],
        )
