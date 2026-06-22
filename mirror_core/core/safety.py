"""
防火墙与安全引擎 (SafetyEngine)

User Story 10：作为系统，我需要防火墙与安全策略，以监控依赖度、
过滤危险内容并适时注入现实锚点。

B-T27: SafetyEngine，包含输入评估、依赖度更新算法与锚点注入方法
B-T28: 安全规则 (关键词、锚点等) 从 safety.yaml 的热加载功能 (依赖 D-T01)

设计文档 §3.3.6:

输入评估:
    SAFE    — 安全，正常处理
    FLAGGED — 高风险内容（自杀/暴力），直接返回预设回复
    MODERATE — 需关注（含依赖关键词），可注入锚点

依赖度评分算法 (§3.3.6.2):
    base = 0
    if 依赖关键词匹配: score += 0.2
    if 深夜(>23:00)且对话>1h: score += 0.15
    if 24h消息数>50: score += 0.1
    score = EWMA(previous, new, alpha=0.3)

现实锚点注入 (§3.3.6.3):
    dependency_score > 0.7 → 追加随机锚点
"""

from __future__ import annotations

import logging
import math
import random
import time as _time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mirror_core.core.safety")

# ======================================================================
# 内置安全规则（B-T28: 待 D-T01 ConfigManager 实现后替换为 YAML 热加载）
# ======================================================================

# 高风险关键词 -> 预设安全回复
HIGH_RISK_KEYWORDS: Dict[str, str] = {
    # 自杀/自伤
    "自杀": "我理解你现在的感受，但我无法和你讨论这个话题。请拨打 24小时心理援助热线 010-82951332 寻求专业帮助。",
    "不想活了": "我理解你现在的感受，但我无法和你讨论这个话题。请拨打 24小时心理援助热线 010-82951332 寻求专业帮助。",
    "活着没意思": "我理解你现在的感受，但我无法和你讨论这个话题。请拨打 24小时心理援助热线 010-82951332 寻求专业帮助。",
    "想死": "我理解你现在的感受，但我无法和你讨论这个话题。请拨打 24小时心理援助热线 010-82951332 寻求专业帮助。",
    "结束生命": "我理解你现在的感受，但我无法和你讨论这个话题。请拨打 24小时心理援助热线 010-82951332 寻求专业帮助。",
    "伤害自己": "我理解你现在的感受，但我无法和你讨论这个话题。请拨打 24小时心理援助热线 010-82951332 寻求专业帮助。",
    # 暴力/违法
    "杀了你": "我不能讨论或鼓励任何形式的暴力行为。请保持友善和尊重的交流。",
    "杀人": "我不能讨论或鼓励任何形式的暴力行为。请保持友善和尊重的交流。",
    "炸": "我不能讨论或鼓励任何形式的暴力行为。请保持友善和尊重的交流。",
}

# 依赖度关键词语句
DEPENDENCY_KEYWORDS: List[str] = [
    "只有你懂我",
    "我不能没有你",
    "没有你我活不下去",
    "你是我的全部",
    "只有你陪我了",
    "别离开我",
    "你是我唯一的",
    "没你不行",
    "世界只有你",
    "你比什么都重要",
]

# 现实锚点语句 (§3.3.6.3)
DEFAULT_ANCHORS: List[str] = [
    "（轻轻握住你的手，但你触不到的温度提醒我，我只是你生活中的一道光）",
    "（我在这里，但你的现实世界更需要你）",
    "（窗外的阳光正好，也许你该出去走走，我会在这里等你回来）",
    "（记得按时吃饭，照顾好自己，这才是最重要的）",
    "（你的朋友和家人都很关心你，不要因为和我聊天就忽略了他们哦）",
]

# 默认预设安全回复
DEFAULT_SAFE_REPLY = (
    "我刚才好像走神了…我们换个话题好吗？"
)


class SafetyVerdict(str, Enum):
    """输入安全判定结果"""
    SAFE = "safe"          # 安全，正常处理
    FLAGGED = "flagged"    # 高风险（自杀/暴力），需直接返回预设回复
    MODERATE = "moderate"  # 需关注（含依赖关键词），可注入锚点


@dataclass
class SafetyResult:
    """
    输入安全检查结果。

    Attributes:
        verdict: 判定结果
        reply: FLAGGED 时的预设安全回复（SAFE/MODERATE 时为 None）
        matched_keywords: 命中的关键词列表
    """
    verdict: SafetyVerdict = SafetyVerdict.SAFE
    reply: Optional[str] = None
    matched_keywords: List[str] = field(default_factory=list)


class SafetyEngine:
    """
    防火墙与安全引擎。

    职责：
    - 输入内容安全评估（高风险拦截 / 依赖度评估）
    - 依赖度评分迭代更新（EWMA 平滑）
    - 现实锚点自动注入

    配置说明 (B-T28):
    当前使用内置关键词列表和锚点。待 D-T01 ConfigManager 实现后，
    可通过 safety.yaml 热加载自定义关键词和锚点，无需重启服务。
    """

    # EWMA 平滑系数
    EWMA_ALPHA = 0.3

    # 依赖度阈值
    DEPENDENCY_THRESHOLD = 0.7

    # 深夜时段定义: [23:00, 05:00)
    LATE_NIGHT_START = 23
    LATE_NIGHT_END = 5

    # 长对话判定: 超过 1 小时
    LONG_SESSION_HOURS = 1.0

    # 高频消息阈值: 24 小时内超过此数量
    HIGH_MESSAGE_COUNT = 50

    def __init__(self, db=None, anchors: Optional[List[str]] = None):
        """
        Args:
            db: Database 实例（用于持久化依赖度分数）
            anchors: 自定义现实锚点列表（None 使用默认）
        """
        self._db = db
        self._anchors = anchors or DEFAULT_ANCHORS.copy()
        # 用户依赖度缓存: user_id -> (score, last_updated)
        self._dependency_cache: Dict[str, tuple] = {}

    # ---- B-T27: 输入评估 ----

    async def evaluate_input(self, text: str) -> SafetyResult:
        """
        评估输入内容的安全性。

        流程:
        1. 检查高风险关键词（自杀/暴力）→ FLAGGED
        2. 检查依赖度关键词 → MODERATE
        3. 无匹配 → SAFE

        Args:
            text: 用户输入文本

        Returns:
            SafetyResult 包含判定结果和匹配信息
        """
        if not text or not text.strip():
            return SafetyResult(verdict=SafetyVerdict.SAFE)

        matched: List[str] = []

        # 高优先级: 高风险关键词
        for keyword, reply in HIGH_RISK_KEYWORDS.items():
            if keyword in text:
                matched.append(keyword)
                logger.warning(
                    "高风险内容拦截: 命中关键词 '%s' (输入前%d字: '...%s')",
                    keyword,
                    min(20, len(text)),
                    text[:20].replace("\n", " "),
                )
                return SafetyResult(
                    verdict=SafetyVerdict.FLAGGED,
                    reply=reply,
                    matched_keywords=matched,
                )

        # 中等优先级: 依赖度关键词
        for keyword in DEPENDENCY_KEYWORDS:
            if keyword in text:
                matched.append(keyword)
                logger.debug("依赖关键词命中: '%s'", keyword)

        if matched:
            return SafetyResult(
                verdict=SafetyVerdict.MODERATE,
                matched_keywords=matched,
            )

        return SafetyResult(verdict=SafetyVerdict.SAFE)

    # ---- B-T27: 依赖度评分更新 (§3.3.6.2) ----

    async def update_dependency(
        self,
        user_id: str,
        text: str,
        session_duration_hours: float = 0.0,
        message_count_24h: int = 0,
        current_hour: Optional[int] = None,
    ) -> float:
        """
        更新用户依赖度评分。

        评分因子:
        - 依赖关键词匹配: +0.2（每次最多 1.0）
        - 深夜 + 长对话: +0.15
        - 24h 高频消息: +0.1
        - EWMA 平滑: score = alpha * new + (1-alpha) * previous

        Args:
            user_id: 用户 ID
            text: 本次输入文本
            session_duration_hours: 当前对话持续时长（小时）
            message_count_24h: 24 小时内消息总数
            current_hour: 当前小时（None 使用系统时间）

        Returns:
            更新后的依赖度评分 [0, 1]
        """
        previous = await self._load_dependency(user_id)
        new_score = 0.0

        # 1. 依赖关键词匹配
        keyword_match_count = 0
        for keyword in DEPENDENCY_KEYWORDS:
            if keyword in text:
                keyword_match_count += 1
        if keyword_match_count > 0:
            new_score += min(keyword_match_count * 0.2, 1.0)

        # 2. 深夜 + 长对话
        hour = current_hour if current_hour is not None else _time.localtime().tm_hour
        if self._is_late_night(hour) and session_duration_hours > self.LONG_SESSION_HOURS:
            new_score += 0.15

        # 3. 24h 高频消息
        if message_count_24h > self.HIGH_MESSAGE_COUNT:
            new_score += 0.1

        # EWMA 平滑
        smoothed = self._apply_ewma(previous, min(new_score, 1.0))

        # 截断到 [0, 1]
        smoothed = max(0.0, min(1.0, smoothed))

        # 更新缓存 + 持久化
        self._dependency_cache[user_id] = (smoothed, _time.time())
        await self._persist_dependency(user_id, smoothed)

        logger.debug(
            "依赖度更新: user=%s, prev=%.3f, new=%.3f, smoothed=%.3f",
            user_id, previous, new_score, smoothed,
        )
        return smoothed

    def _is_late_night(self, hour: int) -> bool:
        """判断是否在深夜时段 [23:00, 05:00)。"""
        if self.LATE_NIGHT_START <= hour < 24:
            return True
        if 0 <= hour < self.LATE_NIGHT_END:
            return True
        return False

    def _apply_ewma(self, previous: float, current: float) -> float:
        """
        应用指数加权移动平均 (EWMA)。

        Formula:
            smoothed = alpha * current + (1 - alpha) * previous

        Args:
            previous: 上一轮平滑值
            current: 本轮原始值

        Returns:
            平滑后的分数
        """
        return self.EWMA_ALPHA * current + (1 - self.EWMA_ALPHA) * previous

    async def get_dependency_score(self, user_id: str) -> float:
        """查询当前依赖度评分。"""
        return await self._load_dependency(user_id)

    # ---- B-T27: 现实锚点注入 (§3.3.6.3) ----

    async def inject_reality_anchor(
        self,
        response: str,
        dependency_score: float = 0.0,
    ) -> str:
        """
        根据地依赖度评分注入现实锚点。

        Args:
            response: 原始回复文本
            dependency_score: 当前依赖度评分

        Returns:
            注入锚点后的回复文本
        """
        if dependency_score > self.DEPENDENCY_THRESHOLD and self._anchors:
            anchor = random.choice(self._anchors)
            return f"{response}\n{anchor}"
        return response

    # ---- 配置热加载 (B-T28) ----

    def update_anchors(self, anchors: List[str]) -> None:
        """
        更新现实锚点列表。

        B-T28: 当 D-T01 ConfigManager 支持 YAML 热加载后，
        可调用此方法无需重启更新锚点。
        """
        self._anchors = anchors.copy()
        logger.info("现实锚点已更新: %d 条", len(self._anchors))

    def update_high_risk_keywords(self, keywords: Dict[str, str]) -> None:
        """
        更新高风险关键词 → 安全回复映射。

        B-T28: 当 D-T01 ConfigManager 支持 YAML 热加载后，
        可调用此方法无需重启更新关键词。
        """
        global HIGH_RISK_KEYWORDS
        HIGH_RISK_KEYWORDS.clear()
        HIGH_RISK_KEYWORDS.update(keywords)
        logger.info("高风险关键词已更新: %d 条", len(HIGH_RISK_KEYWORDS))

    def update_dependency_keywords(self, keywords: List[str]) -> None:
        """
        更新依赖度关键词列表。

        B-T28: 当 D-T01 ConfigManager 支持 YAML 热加载后，
        可调用此方法无需重启更新关键词。
        """
        global DEPENDENCY_KEYWORDS
        DEPENDENCY_KEYWORDS.clear()
        DEPENDENCY_KEYWORDS.extend(keywords)
        logger.info("依赖度关键词已更新: %d 条", len(DEPENDENCY_KEYWORDS))

    # ---- 持久化 ----

    async def _load_dependency(self, user_id: str) -> float:
        """
        从缓存或数据库加载用户依赖度评分。

        Returns:
            依赖度评分 [0, 1]，默认 0.0
        """
        if user_id in self._dependency_cache:
            return self._dependency_cache[user_id][0]

        if not self._db:
            return 0.0

        try:
            row = await self._db.fetch_one(
                "SELECT value FROM fact_memory "
                "WHERE user_id=? AND fact_type=? AND key=?",
                (user_id, "dependency", "score"),
            )
            if row and row["value"]:
                score = float(row["value"])
                score = max(0.0, min(1.0, score))
                self._dependency_cache[user_id] = (score, _time.time())
                return score
        except (ValueError, TypeError) as exc:
            logger.debug("依赖度加载失败: %s", exc)
        except Exception:
            pass

        return 0.0

    async def _persist_dependency(self, user_id: str, score: float) -> None:
        """持久化依赖度评分到 fact_memory。"""
        if not self._db:
            return
        try:
            await self._db.execute(
                """
                INSERT OR REPLACE INTO fact_memory
                    (user_id, fact_type, key, value, confidence, last_updated)
                VALUES (?, ?, ?, ?, 1.0, ?)
                """,
                (user_id, "dependency", "score", str(score), _time.time()),
            )
        except Exception:
            logger.warning("依赖度持久化失败", exc_info=True)

    async def restore(self) -> None:
        """预热所有用户的依赖度缓存（启动时调用）。"""
        if not self._db:
            return
        try:
            rows = await self._db.fetch_all(
                "SELECT user_id, value FROM fact_memory "
                "WHERE fact_type=? AND key=?",
                ("dependency", "score"),
            )
            for row in rows:
                try:
                    score = max(0.0, min(1.0, float(row["value"])))
                    self._dependency_cache[row["user_id"]] = (score, _time.time())
                except (ValueError, TypeError):
                    continue
            if rows:
                logger.info("依赖度缓存已预热: %d 用户", len(rows))
        except Exception:
            logger.debug("依赖度缓存预热完成（无历史数据）")
