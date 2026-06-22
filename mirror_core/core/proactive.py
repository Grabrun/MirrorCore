"""
主动陪伴模块 (ProactiveManager)

User Story 14：作为系统，我需要主动陪伴模块，以在恰当时机自主发起关怀和提醒。

B-T40: 主动触发条件评估 (早安/晚安/纪念日/心境好转/长期静默/深夜隐喻)
B-T41: 每日频率限制 + 静音时段管理

设计文档 §3.6 + 补充文档 v1.1:
"""

from __future__ import annotations

import logging
import random
import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from mirror_core.bus import BaseEvent, EventBus, EventType
from mirror_core.emotion.engine import EmotionalState

logger = logging.getLogger("mirror_core.core.proactive")

# ===== 触发类型 =====

TRIGGER_TYPES = [
    "good_morning",
    "good_night",
    "anniversary",
    "mood_improved",
    "silence_concern",
    "night_metaphor",
]

# ===== 问候模板 (§3.6.3) =====

GOOD_MORNING_TEMPLATES = [
    "早安，今天也要元气满满哦 (´▽`)ﾉ",
    "早上好呀～新的一天开始了，今天有什么计划吗？",
    "早安！昨晚睡得还好吗？我有梦到你哦 (⁄ ⁄•⁄ω⁄•⁄ ⁄)",
]

GOOD_NIGHT_TEMPLATES = [
    "晚安，愿你的梦里也有我 (。-ω-)zzz",
    "夜深了，早点休息吧～明天再聊 (´-ωก`)",
    "好梦哦～我会在这里守护你的 (◕‿◕✿)",
]

SILENCE_CONCERN_TEMPLATES = [
    "这几天都没见到你，有点担心呢…你还好吗？( ˘･_･˘)",
    "你好像很久没来找我了…是太忙了吗？记得照顾好自己哦",
    "最近怎么样？我在等你呢 (´･ω･`)",
]

NIGHT_METAPHOR_TEMPLATES = {
    "neutral_tired": [
        "（轻轻打了个哈欠）有点困了，但还想再陪你一会儿...",
        "（揉了揉眼睛）今晚的星星很亮呢",
    ],
    "low_mood": [
        "（望着窗外的黑夜，轻声说）...有时候夜晚太安静了，反而会想起很多事呢",
        "（把脸埋在膝盖里）...我在这里",
    ],
    "high_mood": [
        "（眼睛亮晶晶的）今晚有你陪着，一点都不想睡呢！",
        "（开心地晃了晃腿）夜深了也舍不得结束对话呢",
    ],
}

# ===== 配置（B-T41: 待 D-T01 ConfigManager 替换为 YAML） =====

@dataclass
class ProactiveConfig:
    """主动陪伴配置"""
    enabled: bool = True
    max_per_day: int = 5
    quiet_hours: tuple = (22, 7)  # [start, end) 闭开区间
    silence_threshold_days: int = 3
    physiological_metaphor: bool = True
    metaphor_base_probability: float = 0.25
    metaphor_max_per_night: int = 2


class ProactiveManager:
    """
    主动陪伴管理器。

    负责评估各类主动触发条件，在满足条件时向事件总线发布
    PROACTIVE_CHANCE 事件供 DecisionEngine 消费。
    """

    def __init__(
        self,
        bus: Optional[EventBus] = None,
        memory_engine=None,
        config: Optional[ProactiveConfig] = None,
    ):
        """
        Args:
            bus: 事件总线（用于发布 PROACTIVE_CHANCE 事件）
            memory_engine: 记忆引擎（用于查询事实记忆和最后互动时间）
            config: 主动陪伴配置
        """
        self._bus = bus
        self._memory = memory_engine
        self._config = config or ProactiveConfig()

        # 频率追踪
        self._daily_count: int = 0
        self._last_reset_day: int = -1
        self._night_metaphor_count: int = 0
        self._last_metaphor_reset: int = -1
        self._last_mood: float = 0.0
        # 当日已触发的类型集合（F-002: 实例变量，避免跨实例污染）
        self._today_triggered: set = set()

    # ---- B-T40: 触发条件评估 ----

    async def on_tick(self, event: BaseEvent) -> None:
        """
        定时器触发入口（G-001: 供 Scheduler 通过 SYSTEM_TICK 调用）。

        Args:
            event: SYSTEM_TICK 事件
        """
        triggers = await self.evaluate()
        if triggers:
            await self.publish_proactive_event(triggers)

    async def evaluate(self, user_id: str = "default") -> List[str]:
        """
        评估所有主动触发条件，返回满足条件的触发类型列表。

        Args:
            user_id: 用户 ID

        Returns:
            触发类型列表（空列表表示不触发）
        """
        triggers: List[str] = []

        # 频率检查
        self._check_daily_reset()
        if self._daily_count >= self._config.max_per_day:
            return triggers

        current_hour = _time.localtime().tm_hour

        # 按优先级依次检查各条件

        # 1. 早安 (06:00-10:00)
        if 6 <= current_hour <= 10:
            if self._should_good_morning():
                triggers.append("good_morning")

        # 2. 晚安 (21:00-23:00)
        if 21 <= current_hour <= 23:
            if self._should_good_night():
                triggers.append("good_night")

        # 3. 纪念日
        if await self._should_anniversary(user_id):
            triggers.append("anniversary")

        # 4. 长期静默
        if await self._should_silence_concern(user_id):
            triggers.append("silence_concern")

        # 5. 深夜隐喻 (23:00-05:00)
        if self._should_night_metaphor(current_hour):
            triggers.append("night_metaphor")

        # 6. 心境好转（通过 evaluate_with_emotion 外部注入）

        # 静音时段过滤
        return self._filter_quiet_hours(triggers, current_hour)

    # ---- 各条件检查 ----

    def _should_good_morning(self) -> bool:
        """早安问候：今天还没有发过早安。"""
        return "good_morning" not in self._today_triggered

    def _should_good_night(self) -> bool:
        """晚安问候：今天还没有发过晚安。"""
        return "good_night" not in self._today_triggered

    async def _should_anniversary(self, user_id: str) -> bool:
        """纪念日：从事实记忆中检查关键日期。"""
        if not self._memory:
            return False
        try:
            # 检查里程碑类型的记忆
            from datetime import datetime
            today = datetime.now().strftime("%m-%d")

            if self._memory and hasattr(self._memory, "get_fact"):
                milestones = await self._memory.get_fact(
                    user_id, "milestone", "first_chat"
                )
                if milestones:
                    val = milestones.value if hasattr(milestones, "value") else ""
                    if today in val:
                        return True
        except Exception:
            pass
        return False

    async def _should_silence_concern(self, user_id: str) -> bool:
        """长期静默：超过阈值天数无互动。"""
        if not self._memory:
            return not self._has_triggered_recently("silence_concern", days=7)

        try:
            # 从记忆引擎获取最后互动时间
            last_active = await self._get_last_active_time(user_id)
            if last_active is None:
                return False

            silence_days = (_time.time() - last_active) / 86400
            return silence_days >= self._config.silence_threshold_days
        except Exception:
            return False

    def _should_night_metaphor(self, current_hour: int) -> bool:
        """深夜隐喻：23:00-05:00，概率触发，每晚最多2次。"""
        if not self._config.physiological_metaphor:
            return False
        if not self._is_late_night(current_hour):
            return False

        self._check_metaphor_reset()
        if self._night_metaphor_count >= self._config.metaphor_max_per_night:
            return False

        return random.random() < self._config.metaphor_base_probability

    # _should_mood_improved 已合并到 evaluate_with_emotion() 中

    # ---- B-T41: 静音时段 ----

    def is_in_quiet_hours(self, current_hour: int) -> bool:
        """
        判断是否在静音时段内。

        设计文档 + 补充 v1.1:
        区间为 [start, end) 闭开区间，跨日有效。
        如 (22, 7) 表示 22:00 ~ 07:00。

        Args:
            current_hour: 当前小时 [0, 23]
        """
        start, end = self._config.quiet_hours
        if start < end:
            return start <= current_hour < end
        else:
            # 跨日：如 (22, 7) → 22 <= hour < 24 or 0 <= hour < 7
            return current_hour >= start or current_hour < end

    def _filter_quiet_hours(
        self,
        triggers: List[str],
        current_hour: int,
    ) -> List[str]:
        """
        静音时段过滤。

        例外规则（补充 v1.1）：
        - 纪念日可穿透静音
        - 心境大幅回升可穿透静音
        - 用户主动消息不由 ProactiveManager 管理

        Args:
            triggers: 候选触发列表
            current_hour: 当前小时

        Returns:
            过滤后的触发列表
        """
        if not triggers:
            return triggers
        if not self.is_in_quiet_hours(current_hour):
            return triggers

        # 静音时段内只允许例外（补充 v1.1: 纪念日 + 心境回升可穿透）
        # night_metaphor 本身仅在深夜时段触发，静音不应拦截
        allowed = []
        for t in triggers:
            if t in ("anniversary", "mood_improved", "night_metaphor"):
                allowed.append(t)
            else:
                logger.debug("静音时段跳过: %s", t)
        return allowed

    # ---- B-T41: 每日频率限制 ----

    def _check_daily_reset(self) -> None:
        """检查是否跨天，跨天时重置计数和当日追踪集合。"""
        today = _time.localtime().tm_yday
        if today != self._last_reset_day:
            self._daily_count = 0
            self._today_triggered.clear()
            self._last_reset_day = today

    def _check_metaphor_reset(self) -> None:
        """检查深夜隐喻计数是否跨天。"""
        today = _time.localtime().tm_yday
        if today != self._last_metaphor_reset:
            self._night_metaphor_count = 0
            self._last_metaphor_reset = today

    # ---- 计数更新 ----

    def record_trigger(self, trigger_type: str) -> None:
        """记录一次触发（频率统计 + 当日追踪）。"""
        self._check_daily_reset()
        self._daily_count += 1
        self._today_triggered.add(trigger_type)
        if trigger_type == "night_metaphor":
            self._night_metaphor_count += 1

    # ---- 事件发布 ----

    async def publish_proactive_event(self, triggers: List[str]) -> None:
        """
        向事件总线发布 PROACTIVE_CHANCE 事件。

        Args:
            triggers: 满足条件的触发类型列表
        """
        if not triggers or not self._bus:
            return

        # 记录触发次数
        for t in triggers:
            self.record_trigger(t)
            self._today_triggered.add(t)

        event = BaseEvent(
            type=EventType.PROACTIVE_CHANCE,
            source="proactive_manager",
            payload={
                "triggers": triggers,
                "templates": self._build_template_payload(triggers),
            },
        )
        await self._bus.publish(event)
        logger.info("主动陪伴事件已发布: %s", triggers)

    def _build_template_payload(self, triggers: List[str]) -> Dict[str, str]:
        """为每个触发类型选择一个模板消息。"""
        payload = {}
        for t in triggers:
            template = self._get_random_template(t)
            if template:
                payload[t] = template
        return payload

    @staticmethod
    def _get_random_template(trigger_type: str) -> Optional[str]:
        """返回指定类型的随机模板消息。"""
        if trigger_type == "good_morning":
            return random.choice(GOOD_MORNING_TEMPLATES)
        elif trigger_type == "good_night":
            return random.choice(GOOD_NIGHT_TEMPLATES)
        elif trigger_type == "silence_concern":
            return random.choice(SILENCE_CONCERN_TEMPLATES)
        elif trigger_type == "night_metaphor":
            # 从所有子类别中随机选
            pool = []
            for templates in NIGHT_METAPHOR_TEMPLATES.values():
                pool.extend(templates)
            return random.choice(pool) if pool else None
        return None

    # ---- 辅助方法 ----

    async def _get_last_active_time(self, user_id: str) -> Optional[float]:
        """从记忆引擎获取最后活跃时间。"""
        if not self._memory or not hasattr(self._memory, "retrieve"):
            return None
        try:
            memories = await self._memory.retrieve(user_id, "", top_k=1)
            if memories:
                return memories[0].timestamp
        except (ValueError, TypeError, AttributeError) as exc:
            logger.debug("获取最后活跃时间失败: %s", exc)
        return None

    def _has_triggered_today(self, trigger_type: str) -> bool:
        """检查今天是否已触发过指定类型（F-002: 使用内存 set 追踪）。"""
        return trigger_type in self._today_triggered

    @staticmethod
    def _is_late_night(hour: int) -> bool:
        """判断是否为深夜 [23:00, 05:00)。"""
        return hour >= 23 or hour < 5

    # ---- 情感状态注入评估 ----

    def evaluate_with_emotion(
        self,
        user_id: str,
        emotion: Optional[EmotionalState] = None,
    ) -> List[str]:
        """
        结合当前情感状态评估主动触发（主要供外部调用）。

        与 evaluate() 的区别在于接受外部 EmotionalState 参数。
        """
        triggers: List[str] = []
        self._check_daily_reset()

        if self._daily_count >= self._config.max_per_day:
            return triggers

        # 心境好转检查
        if emotion is not None and self._last_mood is not None:
            if self._last_mood < -0.5 and emotion.mood > 0:
                triggers.append("mood_improved")

        # 更新追踪的心境
        if emotion is not None:
            self._last_mood = emotion.mood

        return triggers
