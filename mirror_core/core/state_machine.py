"""
状态机 (StateMachine)

User Story 8：作为系统，我需要状态机来管理伴侣的高层交互状态，
确保行为模式的有序切换。

B-T23: StateMachine 类和全部状态转换逻辑
B-T24: maybe_suppress 等动态转换条件评估

设计文档 §3.3.4：

状态定义与转换:
    Normal ←→ Suppressing ←→ Bursting → Reflecting → Normal
    Normal ←→ Consoling → Normal

转换规则:
    (Normal, EMOTION_CHANGED)    → maybe_suppress (动态评估)
    (Suppressing, BURST_TRIGGER)  → Bursting
    (Bursting, BURST_END)         → Reflecting
    (Reflecting, REFLECT_TIMEOUT)  → Normal
    (Normal, CONSOLING_DETECTED)   → Consoling
    (Consoling, CONSOLING_END)     → Normal

非功能设计:
    - 非法跃迁被拒绝并抛出 ValueError
    - 状态变更通过事件总线发布 STATE_TRANSITION 事件
    - 状态持久化到 fact_memory 表 (fact_type='companion_state')
    - 待 DB 迁移 v6 后改为 semantic_memory.companion_state 字段

设计偏差记录 (D-001):
    设计文档 §3.3.4 使用全局 EventType 作为转换触发器键，
    但 BURST_TRIGGER / BURST_END 等是状态机内部触发器，
    不应污染全局事件枚举。因此实现使用独立的 StateTransitionTrigger 枚举。
    外部模块通过调用 transition() / maybe_suppress() 方法与状态机交互，
    无需直接引用 StateTransitionTrigger。
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from mirror_core.bus import BaseEvent, EventBus, EventType

logger = logging.getLogger("mirror_core.core.state_machine")


class CompanionState(str, Enum):
    """伴侣高层交互状态"""
    NORMAL = "Normal"
    SUPPRESSING = "Suppressing"
    BURSTING = "Bursting"
    REFLECTING = "Reflecting"
    CONSOLING = "Consoling"


class StateTransitionTrigger(str, Enum):
    """状态转换触发器（状态机内部事件，非总线 EventType）"""
    EMOTION_CHANGED = "emotion_changed"          # 情感引擎更新后触发的评估
    BURST_TRIGGER = "burst_trigger"              # 爆发触发（阈值/催化）
    BURST_END = "burst_end"                      # 爆发结束，进入恢复期
    REFLECT_TIMEOUT = "reflect_timeout"           # 恢复期超时
    CONSOLING_DETECTED = "consoling_detected"     # 检测到安慰话语
    CONSOLING_END = "consoling_end"               # 安慰结束


@dataclass
class StateTransition:
    """一次完整的状态转换记录"""
    from_state: CompanionState
    to_state: CompanionState
    trigger: StateTransitionTrigger
    timestamp: float = field(default_factory=_time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_change(self) -> bool:
        """是否发生了实际的状态改变"""
        return self.from_state != self.to_state


class StateMachine:
    """
    伴侣高层交互状态机。

    管理 Normal / Suppressing / Bursting / Reflecting / Consoling
    五个状态之间的合法转换，确保行为模式的有序切换。

    用法:
        sm = StateMachine(bus, db)
        await sm.restore()                  # 启动时恢复持久化状态
        await sm.transition(BURST_TRIGGER)  # 触发转换
        print(sm.state)                     # CompanionState.BURSTING
    """

    # ---- 转换表 ----
    # key: (from_state, trigger)
    # value: target_state | "maybe_suppress" (动态评估)
    _TRANSITIONS: Dict[tuple, object] = {
        (CompanionState.NORMAL, StateTransitionTrigger.EMOTION_CHANGED): "maybe_suppress",
        (CompanionState.SUPPRESSING, StateTransitionTrigger.BURST_TRIGGER): CompanionState.BURSTING,
        (CompanionState.BURSTING, StateTransitionTrigger.BURST_END): CompanionState.REFLECTING,
        (CompanionState.REFLECTING, StateTransitionTrigger.REFLECT_TIMEOUT): CompanionState.NORMAL,
        (CompanionState.NORMAL, StateTransitionTrigger.CONSOLING_DETECTED): CompanionState.CONSOLING,
        (CompanionState.CONSOLING, StateTransitionTrigger.CONSOLING_END): CompanionState.NORMAL,
    }

    def __init__(self, bus: Optional[EventBus] = None, db=None):
        """
        Args:
            bus: 事件总线（用于发布 STATE_TRANSITION 事件）
            db: Database 实例（用于状态持久化）
        """
        self._bus = bus
        self._db = db
        self._state = CompanionState.NORMAL
        self._last_transition: Optional[StateTransition] = None

    # ---- 属性 ----

    @property
    def state(self) -> CompanionState:
        """当前伴侣状态。"""
        return self._state

    @property
    def last_transition(self) -> Optional[StateTransition]:
        """最近一次状态转换记录。"""
        return self._last_transition

    @property
    def is_suppressing(self) -> bool:
        return self._state == CompanionState.SUPPRESSING

    @property
    def is_bursting(self) -> bool:
        return self._state == CompanionState.BURSTING

    @property
    def is_reflecting(self) -> bool:
        return self._state == CompanionState.REFLECTING

    @property
    def is_consoling(self) -> bool:
        return self._state == CompanionState.CONSOLING

    @property
    def is_normal(self) -> bool:
        return self._state == CompanionState.NORMAL

    # ---- B-T23: 核心转换逻辑 ----

    async def transition(
        self,
        trigger: StateTransitionTrigger,
        **kwargs: Any,
    ) -> StateTransition:
        """
        尝试根据触发器进行状态转换。

        流程:
        1. 查表获取目标状态（可能为 "maybe_suppress" 动态评估）
        2. 动态评估时，读取 suppression 参数决定目标
        3. 目标与当前不同则更新状态、发布事件、持久化

        Args:
            trigger: 转换触发器
            **kwargs: 动态评估所需参数，如 suppression=0.5

        Returns:
            转换记录（含 from/to/trigger）

        Raises:
            ValueError: 非法转换（表中无此(状态,触发器)组合）
        """
        key = (self._state, trigger)
        if key not in self._TRANSITIONS:
            raise ValueError(
                f"非法状态转换: {self._state.value} ← {trigger.value} — "
                f"该触发器在当前状态下未定义"
            )

        target = self._TRANSITIONS[key]

        # 动态评估 (B-T24)
        if target == "maybe_suppress":
            target_state = self._evaluate_maybe_suppress(
                suppression=kwargs.get("suppression", 0.0),
            )
        else:
            target_state = target

        # 构造转换记录
        transition = StateTransition(
            from_state=self._state,
            to_state=target_state,
            trigger=trigger,
            metadata=kwargs,
        )

        if transition.is_change:
            self._state = target_state
            self._last_transition = transition

            logger.info(
                "状态转换: %s → %s (触发: %s)",
                transition.from_state.value,
                transition.to_state.value,
                transition.trigger.value,
            )

            # 发布 STATE_TRANSITION 事件 (AC #2)
            await self._publish_state_change(transition)

            # 持久化 (AC #3)
            await self._persist()

        return transition

    def can_transition_to(self, target: CompanionState) -> bool:
        """
        检查是否存在从当前状态到目标状态的合法路径。

        对于 "maybe_suppress" 这种动态评估，收集所有可能的结果状态
        （Normal 和 Suppressing）并逐一检查。

        Args:
            target: 目标状态

        Returns:
            是否存在合法转换
        """
        for (from_state, _trigger), to in self._TRANSITIONS.items():
            if from_state != self._state:
                continue

            if to == "maybe_suppress":
                # 动态评估的两个可能结果
                if target in (CompanionState.SUPPRESSING, CompanionState.NORMAL):
                    return True
            elif to == target:
                return True

        return False

    # ---- B-T24: 动态转换条件评估 ----

    def _evaluate_maybe_suppress(self, suppression: float) -> CompanionState:
        """
        动态评估是否进入 Suppressing 状态。

        设计文档 §3.3.4.1:
        若 suppression > 0.3 则进入 SUPPRESSING，否则保持 Normal

        Args:
            suppression: 当前压抑值 [0, 1]

        Returns:
            评估后的目标状态
        """
        if suppression > 0.3:
            return CompanionState.SUPPRESSING
        return CompanionState.NORMAL

    async def maybe_suppress(self, suppression: float) -> StateTransition:
        """
        外部便捷接口：基于压抑值评估是否需要进入 Suppressing。

        Args:
            suppression: 当前压抑值 [0, 1]

        Returns:
            转换记录
        """
        return await self.transition(
            StateTransitionTrigger.EMOTION_CHANGED,
            suppression=suppression,
        )

    # ---- 持久化 (AC #3) ----

    async def _persist(self) -> None:
        """
        将当前状态持久化到 semantic_memory 表。

        设计文档 §3.3.4.3: 状态持久化到 semantic_memory 表。
        使用 companion_state 字段（DB 迁移 v6 添加）。
        兼容性：旧版库无该字段时静默降级（由迁移保证）。
        """
        if not self._db:
            return
        try:
            # 尝试写入 semantic_memory（优先，设计文档要求）
            await self._db.execute(
                """
                INSERT INTO semantic_memory (user_id, companion_state, last_updated)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    companion_state = excluded.companion_state,
                    last_updated = excluded.last_updated
                """,
                ("default", self._state.value, _time.time()),
            )
        except (RuntimeError, AttributeError) as exc:
            logger.warning("状态机持久化失败 (数据库/迁移问题): %s", exc)

    async def restore(self) -> None:
        """
        从 semantic_memory 表恢复状态。

        设计文档 §3.3.4.3: 重启后以内存状态为准恢复。
        若数据库中有持久化状态则加载，否则使用默认 Normal。
        """
        if not self._db:
            return
        try:
            row = await self._db.fetch_one(
                "SELECT companion_state FROM semantic_memory WHERE user_id=?",
                ("default",),
            )
            if row and row["companion_state"]:
                restored = CompanionState(row["companion_state"])
                self._state = restored
                logger.info("状态机状态已恢复: %s", restored.value)
        except (ValueError, KeyError, AttributeError) as exc:
            logger.warning("状态机恢复失败: %s，使用默认 Normal", exc)
        except Exception:
            logger.info("无持久化状态机状态，使用默认 Normal")

    async def force_persist(self) -> None:
        """强制立即持久化当前状态。"""
        await self._persist()

    # ---- 事件发布 ----

    async def _publish_state_change(self, transition: StateTransition) -> None:
        """
        通过事件总线发布 STATE_TRANSITION 事件。

        Args:
            transition: 状态转换记录
        """
        if not self._bus:
            return
        event = BaseEvent(
            type=EventType.STATE_TRANSITION,
            source="state_machine",
            payload={
                "from_state": transition.from_state.value,
                "to_state": transition.to_state.value,
                "trigger": transition.trigger.value,
                "timestamp": transition.timestamp,
            },
        )
        await self._bus.publish(event)
        logger.debug(
            "STATE_TRANSITION 事件已发布: %s → %s",
            transition.from_state.value,
            transition.to_state.value,
        )
