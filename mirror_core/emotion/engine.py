"""
潮汐情感引擎 (Tidal Engine)

User Story 6：作为系统，我需要 Tidal 情感引擎，以维护伴侣的情感状态
并驱动富有表现力的行为。

B-T17: EmotionalState 数据模型和 TidalEngine 基础结构
B-T18: 情感推力施加、压抑与爆发检查
B-T19: 爆发后恢复流程、心境 EMA 更新算法
B-T20: 情感状态定期持久化

设计文档 §3.3.2:
- PAD 三维情绪模型：P(愉悦度[-1,1]) A(激活度[0,1]) D(支配度[-1,1])
- 压抑机制: suppress_tendency 决定负面情绪被压抑的比例
- 爆发触发: 阈值爆发(suppression>=0.8) / 催化爆发(记忆触发)
- 心境 EMA: α=0.05 指数移动平均，大强度时加速
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time as _time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from mirror_core.infrastructure.database import Database

logger = logging.getLogger("mirror_core.emotion.engine")

# ---- 常量 ----

ALPHA = 0.05           # 心境 EMA 更新率
CLAMP = 3              # 输出四舍五入保留位数
REFLECT_TIMEOUT = 30   # 爆发后恢复时间 (秒)
PERSIST_INTERVAL = 60  # 状态持久化间隔 (秒)


@dataclass
class EmotionalState:
    """情感状态数据模型"""
    P: float = 0.0       # 愉悦度 [-1, 1]
    A: float = 0.3       # 激活度 [0, 1]
    D: float = 0.0       # 支配度 [-1, 1]
    mood: float = 0.0    # 持久心境 [-1, 1]
    suppression: float = 0.0  # 压抑值 [0, 1]
    status: str = "Normal"    # Normal | Suppressing | Bursting | Reflecting | Consoling


@dataclass
class EmotionSnapshot:
    """情绪快照（记忆关联时使用）"""
    P: float = 0.0
    A: float = 0.0
    D: float = 0.0
    mood: float = 0.0
    suppression: float = 0.0
    intensity: float = 0.5


# ---- 表情标签映射 (§3.4.4.3) ----

EMOTION_TAG_MAP: List[Tuple[str, List[Tuple[str, int]]]] = [
    ("P>0.5,A>0.5", [("开心", 7), ("兴奋", 3)]),
    ("P<-0.5,A<0.3", [("低落", 8), ("伤心", 2)]),
    ("P<-0.3,A>0.7", [("生气", 7), ("委屈", 3)]),
    ("D<-0.5", [("无助", 6), ("不安", 4)]),
    ("Suppression>0.5", [("强颜欢笑", 5), ("平静", 5)]),
]

# ---- 颜文字映射 (§3.4.1.2) ----

KAOMOJI_MAP: Dict[str, List[str]] = {
    "P>0.5,A>0.5": ["(≧▽≦)", "ヽ(>∀<)ノ"],
    "P<-0.5,A<0.3": ["(´；ω；`)", "(◞‸◟)"],
    "P<-0.3,A>0.7": ["(╬ Ò﹏Ó)", "(`皿´)"],
    "D<-0.5": ["(´-ω-`)", "(◍•﹏•)"],
    "Suppression>0.5": ["(⌒_⌒;)", "(•ᴗ•;)"],
}


def _clamp(value: float, lo: float, hi: float) -> float:
    """裁剪并四舍五入保留 CLAMP 位小数。"""
    return round(max(lo, min(hi, value)), CLAMP)


def _meets_condition(condition: str, state: EmotionalState) -> bool:
    """判断情感状态是否满足条件表达式（如 'P>0.5,A>0.5'）。"""
    # 字段名映射（兼容大写首字母，如 Suppression→suppression）
    FIELD_MAP = {
        "P": "P", "A": "A", "D": "D",
        "mood": "mood", "Mood": "mood",
        "suppression": "suppression", "Suppression": "suppression",
        "status": "status", "Status": "status",
    }
    try:
        parts = condition.split(",")
        for part in parts:
            if ">=" in part:
                k, v = part.split(">=")
                attr = FIELD_MAP.get(k.strip(), k.strip().lower())
                if getattr(state, attr, None) is None or getattr(state, attr) < float(v.strip()):
                    return False
            elif ">" in part:
                k, v = part.split(">")
                attr = FIELD_MAP.get(k.strip(), k.strip().lower())
                if getattr(state, attr, None) is None or getattr(state, attr) <= float(v.strip()):
                    return False
            elif "<=" in part:
                k, v = part.split("<=")
                attr = FIELD_MAP.get(k.strip(), k.strip().lower())
                if getattr(state, attr, None) is None or getattr(state, attr) > float(v.strip()):
                    return False
            elif "<" in part:
                k, v = part.split("<")
                attr = FIELD_MAP.get(k.strip(), k.strip().lower())
                if getattr(state, attr, None) is None or getattr(state, attr) >= float(v.strip()):
                    return False
            else:
                return False
        return True
    except Exception:
        return False


# ---- Persona 配置（轻量版，待 D-T01 替换为 pydantic schema） ----

@dataclass
class PersonaConfig:
    """人设配置（影响情感行为参数）"""
    suppress_tendency: float = 0.6     # 压抑倾向 [0, 1]
    emotional_sensitivity: float = 0.8  # 情感敏感度 [0, 1]


# ---- TidalEngine ----

class TidalEngine:
    """
    潮汐情感引擎核心类。

    管理 PAD 情绪向量、压抑积累、爆发检测、心境 EMA 更新、
    表情/颜文字映射，以及状态持久化。

    Args:
        db: 数据库实例，用于状态持久化
        persona: 人设配置
        user_id: 用户ID（用于状态持久化键值）
    """

    def __init__(
        self,
        db: Database,
        persona: Optional[PersonaConfig] = None,
        user_id: str = "default",
    ):
        self._db = db
        self._persona = persona or PersonaConfig()
        self._user_id = user_id
        self._lock = asyncio.Lock()
        self._state = EmotionalState()
        self._last_persist_time: float = 0.0
        self._burst_time: float = 0.0  # 爆发时间戳
        self._reflecting_task: Optional[asyncio.Task] = None

    @property
    def state(self) -> EmotionalState:
        """当前情感状态（只读快照）。"""
        return self._state

    # ========== B-T18: 推力施加 ==========

    async def apply_emotional_thrust(
        self,
        delta: Dict[str, float],
        message_text: str = "",
        memory_snapshots: Optional[List[EmotionSnapshot]] = None,
    ) -> EmotionalState:
        """
        施加情感推力并更新状态。

        流程：
        1. 获取情绪推力矢量 (dP, dA, dD)
        2. 压抑计算：负P值被抑制
        3. 应用推力到 PAD 向量
        4. 检查爆发条件
        5. 更新心境
        6. 定期持久化

        Args:
            delta: 推力矢量 {"P": dP, "A": dA, "D": dD}
            message_text: 触发消息文本（用于实体匹配）
            memory_snapshots: 关联的记忆情绪快照列表

        Returns:
            更新后的情感状态
        """
        async with self._lock:
            dP = delta.get("P", 0.0)
            dA = delta.get("A", 0.0)
            dD = delta.get("D", 0.0)

            # 2. 压抑计算 (§3.3.2.3)
            if dP < 0:
                suppressed = dP * self._persona.suppress_tendency
                dP -= suppressed  # dP 减少的绝对值（注意 dP 本身为负）
                self._state.suppression += abs(suppressed)

            self._state.suppression = _clamp(self._state.suppression, 0.0, 1.0)

            # 3. 应用推力到 PAD
            self._state.P = _clamp(
                self._state.P + dP * self._persona.emotional_sensitivity, -1.0, 1.0
            )
            self._state.A = _clamp(
                self._state.A + dA * self._persona.emotional_sensitivity, 0.0, 1.0
            )
            self._state.D = _clamp(
                self._state.D + dD * self._persona.emotional_sensitivity, -1.0, 1.0
            )

            # 4. 检查爆发条件
            await self._check_and_trigger_burst(memory_snapshots or [])

            # 5. 更新心境
            intensity = math.sqrt(dP**2 + dA**2 + dD**2)
            await self._update_mood_internal(self._state.P, intensity)

            # 6. 定期持久化
            await self._maybe_persist()

            logger.debug(
                "情感推力已施加",
                extra={
                    "P": self._state.P, "A": self._state.A, "D": self._state.D,
                    "mood": self._state.mood, "suppression": self._state.suppression,
                    "status": self._state.status,
                },
            )
            return self._state

    async def _check_and_trigger_burst(
        self, memory_snapshots: List[EmotionSnapshot],
    ) -> None:
        """
        检查并触发情绪爆发。

        爆发条件（二选一）：
        1. 阈值爆发：suppression >= 0.8
        2. 催化爆发：记忆强度 > 0.7 且 P < -0.4 且实体匹配
        """
        burst = False

        if self._state.suppression >= 0.8:
            burst = True
            logger.info("阈值爆发: suppression=%.3f >= 0.8", self._state.suppression)

        if not burst and memory_snapshots:
            for snap in memory_snapshots:
                if (
                    snap.intensity > 0.7
                    and snap.P < -0.4
                ):
                    burst = True
                    logger.info("催化爆发: memory intensity=%.3f, P=%.3f", snap.intensity, snap.P)
                    break

        if burst:
            await self._enter_burst()
        else:
            # 不爆发时检查是否需要进入压抑状态
            if self._state.suppression > 0.3 and self._state.status == "Normal":
                self._state.status = "Suppressing"
                logger.debug("进入压抑状态: suppression=%.3f", self._state.suppression)

    async def _enter_burst(self) -> None:
        """
        进入爆发状态。

        爆发效果：
        - status → Bursting
        - suppression 清零
        - A 拉升至 0.9
        - P 额外下降 0.3
        - 启动 Reflecting 恢复计时器
        """
        self._state.status = "Bursting"
        self._state.suppression = 0.0
        self._state.A = _clamp(self._state.A + 0.9, 0.0, 1.0)
        self._state.P = _clamp(self._state.P - 0.3, -1.0, 1.0)
        self._burst_time = _time.time()

        # 取消旧计时器
        if self._reflecting_task and not self._reflecting_task.done():
            self._reflecting_task.cancel()

        # 启动恢复计时器
        self._reflecting_task = asyncio.create_task(self._schedule_reflecting())

        logger.info("情感爆发触发: status=Bursting, P=%.3f, A=%.3f", self._state.P, self._state.A)

    async def _schedule_reflecting(self) -> None:
        """
        在 REFLECT_TIMEOUT 秒后自动进入 Reflecting 并恢复 Normal。
        """
        try:
            await asyncio.sleep(REFLECT_TIMEOUT)
            async with self._lock:
                self._state.status = "Reflecting"
                logger.info("进入恢复期: status=Reflecting")

                # 恢复期内 P 缓慢回升（向 0 偏移 0.2）
                self._state.P = _clamp(self._state.P + 0.2, -1.0, 1.0)
                self._state.A = _clamp(self._state.A - 0.3, 0.0, 1.0)

                await asyncio.sleep(2)  # 短暂恢复期

                self._state.status = "Normal"
                logger.info("恢复完成: status=Normal")
        except asyncio.CancelledError:
            pass

    # ========== B-T19: 心境 EMA 更新 ==========

    async def _update_mood_internal(self, new_P: float, intensity: float) -> None:
        """
        更新持久心境（EMA 算法，§3.3.2.5）。

        Args:
            new_P: 当前 P 值
            intensity: 本次推力强度
        """
        alpha = ALPHA

        # 大强度且方向相反 → 加速更新
        if intensity > 0.8 and (new_P * self._state.mood < 0):
            self._state.mood = _clamp(
                self._state.mood * 0.5 + new_P * 0.5, -1.0, 1.0
            )
        else:
            self._state.mood = _clamp(
                self._state.mood * (1 - alpha) + new_P * alpha, -1.0, 1.0
            )

    async def update_mood(self, new_P: float, intensity: float) -> None:
        """公开的心境更新接口（外部调用时使用锁）。"""
        async with self._lock:
            await self._update_mood_internal(new_P, intensity)

    # ========== 衰减 ==========

    async def apply_decay(self) -> EmotionalState:
        """
        应用情感衰减（由 SYSTEM_TICK 触发）。

        每 60 秒调用：
        - P 向 0 偏移 0.02
        - A 向 0 偏移 0.01
        - D 向 0 偏移 0.02
        - suppression 衰减 0.05
        - mood 向 0 偏移 0.01
        """
        async with self._lock:
            if self._state.status in ("Bursting", "Reflecting"):
                return self._state

            self._state.P = _clamp(self._state.P * 0.98, -1.0, 1.0)
            self._state.A = _clamp(self._state.A * 0.99, 0.0, 1.0)
            self._state.D = _clamp(self._state.D * 0.98, -1.0, 1.0)
            self._state.suppression = _clamp(self._state.suppression - 0.05, 0.0, 1.0)
            self._state.mood = _clamp(self._state.mood * 0.99, -1.0, 1.0)

            await self._maybe_persist()

            return self._state

    # ========== 表情/颜文字映射 ==========

    def select_kaomoji(self) -> str:
        """
        根据当前情感状态选择颜文字（§3.4.1.2）。

        Returns:
            匹配的颜文字字符串，无匹配时返回空字符串
        """
        for condition, emojis in KAOMOJI_MAP.items():
            if _meets_condition(condition, self._state):
                return random.choice(emojis)
        return ""

    def select_expression_tags(self) -> List[Tuple[str, int]]:
        """
        根据当前情感状态选择表情标签（§3.4.4.3）。

        Returns:
            标签和权重的列表，如 [("开心", 7), ("兴奋", 3)]
        """
        for condition, tags in EMOTION_TAG_MAP:
            if _meets_condition(condition, self._state):
                return tags
        return []

    def select_random_tag(self) -> Optional[str]:
        """加权随机选择一个表情标签。"""
        tags = self.select_expression_tags()
        if not tags:
            return None
        choices = [tag for tag, weight in tags for _ in range(weight)]
        return random.choice(choices) if choices else None

    # ========== 持久化（B-T20） ==========

    async def _maybe_persist(self) -> None:
        """
        定期持久化情感状态。

        每 PERSIST_INTERVAL 秒写一次数据库。
        避免高频 I/O。
        """
        now = _time.time()
        if now - self._last_persist_time < PERSIST_INTERVAL:
            return

        await self._do_persist()
        self._last_persist_time = now

    async def _do_persist(self) -> None:
        """将当前情感状态写入数据库。"""
        blob = self.serialize()
        try:
            await self._db.execute(
                """
                INSERT INTO fact_memory (user_id, fact_type, key, value, confidence, last_updated)
                VALUES (?, ?, ?, ?, 1.0, ?)
                ON CONFLICT(user_id, fact_type, key) DO UPDATE SET
                    value = excluded.value,
                    last_updated = excluded.last_updated
                """,
                (self._user_id, "emotion", "tidal_state", blob, _time.time()),
            )
            logger.debug("情感状态已持久化")
        except Exception:
            logger.warning("情感状态持久化失败", exc_info=True)

    async def restore(self) -> None:
        """从数据库恢复情感状态（启动时调用）。"""
        try:
            row = await self._db.fetch_one(
                "SELECT value FROM fact_memory WHERE user_id = ? AND fact_type = ? AND key = ?",
                (self._user_id, "emotion", "tidal_state"),
            )
            if row and row["value"]:
                self.deserialize(row["value"])
                logger.info("情感状态已恢复: P=%.3f, mood=%.3f", self._state.P, self._state.mood)
        except Exception:
            logger.info("无持久化情感状态，使用初始值")

    def serialize(self) -> str:
        """序列化状态为 JSON。"""
        return json.dumps({
            "P": self._state.P,
            "A": self._state.A,
            "D": self._state.D,
            "mood": self._state.mood,
            "suppression": self._state.suppression,
            "status": self._state.status,
        })

    def deserialize(self, blob: str) -> None:
        """从 JSON 反序列化状态。"""
        try:
            data = json.loads(blob)
            self._state.P = data.get("P", 0.0)
            self._state.A = data.get("A", 0.3)
            self._state.D = data.get("D", 0.0)
            self._state.mood = data.get("mood", 0.0)
            self._state.suppression = data.get("suppression", 0.0)
            self._state.status = data.get("status", "Normal")
        except (json.JSONDecodeError, KeyError):
            logger.warning("情感状态反序列化失败，使用初始值")

    async def force_persist(self) -> None:
        """强制立即持久化。"""
        await self._do_persist()
        self._last_persist_time = _time.time()
