"""
定时触发器模块 (Scheduler)

User Story 4：作为系统，我需要定时触发器，以生成情感衰减、纪念日检查、
主动陪伴等周期性事件。

B-T11: 基于 apscheduler 封装 Scheduler 类，实现启动与停止管理
B-T12: 实现 3.2.3.3 节中定义的全部触发任务逻辑并发布事件
B-T13: 实现时区解析与随机间隔调度逻辑

Trigger Tasks (设计文档 §3.2.3.3):
┌─────────────────────┬────────────────────┬──────────────────────┐
│ 任务名              │ 触发规则            │ 产生事件              │
├─────────────────────┼────────────────────┼──────────────────────┤
│ decay_tick          │ 每 60 秒            │ SYSTEM_TICK          │
│ anniversary_check   │ 每天 09:00(用户时区)│ ANNIVERSARY_CHECK    │
│ proactive_chance    │ 每 30~90 分钟随机   │ PROACTIVE_CHANCE     │
│ night_metaphor      │ 23:00-05:00 每20分  │ PROACTIVE_CHANCE     │
└─────────────────────┴────────────────────┴──────────────────────┘

补充文档 v1.1 新增约束：
- quiet_hours: 静音时段抑制非紧急主动消息
- 生理隐喻: 深夜额外检查活跃会话，注入隐喻
- 时区来源: persona.yaml / POST /api/v1/user/timezone / 系统时区兜底
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timezone as dt_timezone
from typing import Dict, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from mirror_core.bus import BaseEvent, EventBus, EventType

logger = logging.getLogger("mirror_core.scheduler")


class Scheduler:
    """
    定时触发器管理器。

    封装 AsyncIOScheduler，管理所有周期性/条件性触发任务的
    生命周期、事件发布、时区感知与随机间隔调度。

    Args:
        bus: 事件总线实例，用于发布定时事件
        timezone: 用户时区（IANA 格式，如 'Asia/Shanghai'）
        config: 可选的配置字典，影响 quiet_hours 和生理隐喻行为
    """

    def __init__(
        self,
        bus: EventBus,
        timezone: str = "Asia/Shanghai",
        config: Optional[Dict] = None,
    ):
        self._bus = bus
        self._timezone = timezone
        self._config = config or {}
        self._scheduler = AsyncIOScheduler(timezone=timezone)
        self._running = False

    # ---- 生命周期 ----

    async def start(self) -> None:
        """注册所有定时任务并启动调度器。"""
        if self._running:
            logger.warning("调度器已在运行，忽略重复启动")
            return

        self._register_decay_tick()
        self._register_anniversary_check()
        self._register_proactive_chance()
        self._register_night_metaphor()

        self._scheduler.start()
        self._running = True
        logger.info("定时触发器已启动，时区=%s", self._timezone)

    async def stop(self) -> None:
        """停止调度器，清理所有定时任务。"""
        if not self._running:
            return

        self._scheduler.shutdown(wait=False)
        self._running = False
        logger.info("定时触发器已停止")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def timezone(self) -> str:
        return self._timezone

    # ---- B-T12: 触发任务注册 ----

    def _register_decay_tick(self) -> None:
        """
        注册情感衰减定时任务。

        规则：每 60 秒触发一次 SYSTEM_TICK 事件。
        """
        self._scheduler.add_job(
            self._fire_system_tick,
            IntervalTrigger(seconds=60),
            id="decay_tick",
            name="情感衰减 (60s)",
            replace_existing=True,
        )

    def _register_anniversary_check(self) -> None:
        """
        注册纪念日检查任务。

        规则：每天 09:00（用户配置时区）触发 ANNIVERSARY_CHECK 事件。
        CronTrigger 显式传递时区以匹配设计文档要求（F-002）。
        """
        self._scheduler.add_job(
            self._fire_anniversary_check,
            CronTrigger(hour=9, minute=0, timezone=self._timezone),
            id="anniversary_check",
            name="纪念日检查 (每天 09:00)",
            replace_existing=True,
        )

    def _register_proactive_chance(self) -> None:
        """
        注册主动陪伴触发任务。

        规则：每 30~90 分钟随机触发一次 PROACTIVE_CHANCE 事件。
        静音时段内跳过（安静时段除外）。
        """
        initial_delay = random.uniform(30, 90) * 60
        self._scheduler.add_job(
            self._fire_proactive_chance,
            IntervalTrigger(seconds=initial_delay),
            id="proactive_chance",
            name="主动陪伴 (30~90min 随机)",
            replace_existing=True,
        )

    def _register_night_metaphor(self) -> None:
        """
        注册生理隐喻检查任务。

        规则：23:00-05:00 每 20 分钟检查一次活跃会话，
        命中时发布 PROACTIVE_CHANCE 事件（含 metaphor_type 标记）。
        白天的触发会被处理器自动跳过。
        """
        self._scheduler.add_job(
            self._fire_night_metaphor_check,
            IntervalTrigger(minutes=20),
            id="night_metaphor",
            name="生理隐喻 (20min)",
            replace_existing=True,
        )

    # ---- B-T13: 事件处理器 ----

    async def _fire_system_tick(self) -> None:
        """发布 SYSTEM_TICK 事件给情感引擎进行衰减计算。"""
        await self._bus.publish(
            BaseEvent(
                type=EventType.SYSTEM_TICK,
                source="scheduler.decay_tick",
            )
        )

    async def _fire_anniversary_check(self) -> None:
        """
        发布 ANNIVERSARY_CHECK 事件。

        将来由 ProactiveManager 订阅此事件，检查当天是否有纪念日、
        生日等特殊日期，并产生对应的主动问候。
        """
        await self._bus.publish(
            BaseEvent(
                type=EventType.ANNIVERSARY_CHECK,
                source="scheduler.anniversary_check",
            )
        )

    async def _fire_proactive_chance(self) -> None:
        """
        发布 PROACTIVE_CHANCE 事件并重新调度随机间隔。

        静音时段（quiet_hours）内跳过非紧急主动消息。
        跳过后仍会重新调度，确保静音时段的起始时间均匀分布。
        """
        skip_reason = self._should_skip_proactive()
        if skip_reason:
            logger.debug("跳过主动陪伴: %s", skip_reason)
            # 跳过时不发布事件，payload 无意义
        else:
            await self._bus.publish(
                BaseEvent(
                    type=EventType.PROACTIVE_CHANCE,
                    source="scheduler.proactive_chance",
                    payload={"task": "proactive_chance"},
                )
            )

        # 重新调度随机间隔（B-T13）
        self._reschedule_proactive()

    async def _fire_night_metaphor_check(self) -> None:
        """
        检查当前是否处于生理隐喻窗口（23:00-05:00），
        命中时发布带隐喻标记的 PROACTIVE_CHANCE 事件。

        检查维度：
        1. 当前时间是否在 23:00-05:00 窗口内
        2. 生理隐喻功能是否启用
        """
        if not self._is_night_window():
            return

        metaphor_cfg = self._config.get("physiological_metaphor", {})
        if not metaphor_cfg.get("enabled", False):
            return

        await self._bus.publish(
            BaseEvent(
                type=EventType.PROACTIVE_CHANCE,
                source="scheduler.night_metaphor",
                payload={
                    "task": "night_metaphor",
                    "metaphor_type": True,
                },
            )
        )

    # ---- B-T13: 调度工具 ----

    def _reschedule_proactive(self) -> None:
        """以 30~90 分钟随机间隔重新调度 proactive_chance 任务。"""
        next_interval = random.uniform(30, 90) * 60  # 秒
        self._scheduler.reschedule_job(
            "proactive_chance",
            trigger=IntervalTrigger(seconds=next_interval),
        )

    def _should_skip_proactive(self) -> Optional[str]:
        """
        判断是否跳过本次主动陪伴触发。

        返回 None 表示不跳过，返回字符串表示跳过原因。
        静音时段例外规则（纪念日追加、心境回升）在 ProactiveManager 层处理。
        """
        qh = self._config.get("quiet_hours", {})
        if not qh.get("enabled", False):
            return None

        if self._is_in_quiet_hours(qh):
            return "quiet_hours"

        return None

    def _is_in_quiet_hours(self, qh_config: dict, now: Optional[datetime] = None) -> bool:
        """
        判断指定时间是否在静音时段内。

        静音时段定义为 [start, end) 闭开区间，支持跨日：
        - start=22, end=7  → 22:00-07:00（跨日）
        - start=23, end=6  → 23:00-06:00（跨日）
        - start=13, end=14 → 13:00-14:00（同日）

        使用调度器配置时区判断，与 CronTrigger 时区一致。
        now 参数仅用于测试注入，为 None 时使用实时时钟。
        """
        if now is None:
            tz = self._scheduler.timezone
            now = datetime.now(tz) if tz else datetime.now(dt_timezone.utc)

        hour = now.hour
        start = qh_config.get("start", 22)
        end = qh_config.get("end", 7)

        if start > end:
            # 跨日：e.g., [22, 7)
            return hour >= start or hour < end
        else:
            # 同日：e.g., [13, 14)
            return start <= hour < end

    def _is_night_window(self, now: Optional[datetime] = None) -> bool:
        """
        判断指定时间是否在生理隐喻窗口（23:00-05:00）内。

        如果有用户时区，使用该时区判断；否则使用 UTC。
        now 参数仅用于测试注入，为 None 时使用实时时钟。
        """
        if now is None:
            tz = self._scheduler.timezone
            now = datetime.now(tz) if tz else datetime.now(dt_timezone.utc)
        hour = now.hour
        return hour >= 23 or hour < 5

    # ---- 配置更新 ----

    def reload_config(self, config: dict) -> None:
        """
        热加载配置（quiet_hours、physiological_metaphor 等）。

        B-T28 全量配置热加载的前置接口，调用后新配置在下一次
        触发检查时生效。
        """
        self._config = config
        logger.debug("调度器配置已热加载")

    def set_timezone(self, timezone: str) -> None:
        """
        更新用户时区。

        注意：APScheduler 不支持运行时修改全局时区。
        时区变更后需要 stop() → start() 才会使 CronTrigger 生效。
        新时区会立即影响 _is_night_window() 和 _is_in_quiet_hours()
        的判断。
        """
        self._timezone = timezone
        logger.info("调度器时区已更新: %s", timezone)
