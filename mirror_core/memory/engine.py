"""
MemoryEngine — 记忆引擎

User Story 5：作为系统，我需要一个健壮的记忆引擎，以管理用户的四层记忆
并提供混合检索能力。

B-T14: MemoryEngine 核心类的基础 CRUD 与 RRF 混合检索方法
B-T15: 工作记忆的快照存储与恢复逻辑
B-T16: 事实记忆遗忘策略 (置信度衰减与归档)

设计文档 §3.3.1:
- 情景记忆 (episodic): FTS5 + 可选 vec0 向量混合检索 (RRF)
- 事实记忆 (fact): 结构化 key-value, UNIQUE(user_id, fact_type, key)
- 语义记忆 (semantic): 关系评分 (trust, intimacy, relationship_stage)
- 工作记忆: 内存 deque + conversation_turns 表快照
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Deque, Dict, List, Optional

from mirror_core.infrastructure.database import Database

logger = logging.getLogger("mirror_core.memory.engine")

# ---- 数据模型 ----

RRF_K = 60  # RRF 融合常数


@dataclass
class EpisodicMemory:
    """情景记忆"""
    id: str = ""
    user_id: str = ""
    session_id: str = ""
    timestamp: float = 0.0
    summary: str = ""
    emotion_json: str = "{}"
    intensity: float = 0.5
    tags: str = ""
    fts_content: str = ""
    created_at: float = 0.0


@dataclass
class FactMemory:
    """事实记忆"""
    user_id: str = ""
    fact_type: str = ""
    key: str = ""
    value: str = ""
    confidence: float = 1.0
    last_updated: float = 0.0


@dataclass
class SemanticMemory:
    """语义记忆（关系评分）"""
    user_id: str = ""
    trust_score: float = 0.5
    intimacy_score: float = 0.3
    relationship_stage: str = "acquaintance"
    last_updated: float = 0.0


@dataclass
class ConversationTurn:
    """对话轮次"""
    session_id: str = ""
    user_id: str = ""
    role: str = ""  # 'user' | 'assistant'
    content: str = ""
    timestamp: float = 0.0
    emotion_json: str = ""


# ---- MemoryEngine ----

class MemoryEngine:
    """
    记忆引擎核心类。

    管理四层记忆（情景/事实/语义/工作记忆）的 CRUD 与混合检索。

    Args:
        db: 数据库实例
        embed_fn: 可选的向量化函数 (text) → List[float]，提供时启用 RRF 向量融合
    """

    def __init__(
        self,
        db: Database,
        embed_fn: Optional[Callable[[str], Coroutine[Any, Any, List[float]]]] = None,
    ):
        self._db = db
        self._embed_fn = embed_fn
        self._vec0_available: Optional[bool] = None  # 惰性检测
        self._working_memory: Dict[str, Deque[ConversationTurn]] = {}

    # ========== B-T14: 情景记忆 CRUD ==========

    async def store_episodic(self, memory: EpisodicMemory) -> str:
        """
        存储一条情景记忆。

        Args:
            memory: 情景记忆对象

        Returns:
            记忆的 ID
        """
        mem_id = memory.id or uuid.uuid4().hex
        now = time.time()
        fts_content = f"{memory.summary} {memory.tags}"

        await self._db.execute(
            """
            INSERT INTO episodic_memory
                (id, user_id, session_id, timestamp, summary, emotion_json,
                 intensity, tags, fts_content, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mem_id, memory.user_id, memory.session_id,
                memory.timestamp or now, memory.summary,
                memory.emotion_json, memory.intensity,
                memory.tags, fts_content, now,
            ),
        )

        # 同步写入 FTS5 独立表
        await self._sync_fts(mem_id, memory.user_id, fts_content)

        # 如果有向量化函数，生成 embedding 并写入 vec0
        if self._embed_fn and await self._has_vec0():
            try:
                embedding = await self._embed_fn(fts_content)
                await self._store_embedding(mem_id, embedding)
            except Exception:
                logger.warning("向量化失败，跳过 vec0 写入")

        logger.debug("情景记忆已存储", extra={"id": mem_id, "user_id": memory.user_id})
        return mem_id

    async def _sync_fts(self, mem_id: str, user_id: str, fts_content: str) -> None:
        """同步写入 FTS5 索引（独立表，非外连）。"""
        # 获取 rowid 用于 FTS5 表关联
        rowid = await self._get_rowid(mem_id)
        if rowid is None:
            return
        try:
            await self._db.execute(
                "INSERT INTO episodic_fts (rowid, user_id, fts_content) VALUES (?, ?, ?)",
                (rowid, user_id, fts_content),
            )
        except Exception:
            logger.warning("FTS5 同步失败", exc_info=True)

    async def _store_embedding(self, mem_id: str, embedding: List[float]) -> None:
        """将向量写入 vec0 虚拟表。"""
        rowid = await self._get_rowid(mem_id)
        if rowid is None:
            return
        await self._db.execute(
            "INSERT INTO episodic_vec (rowid, embedding) VALUES (?, ?)",
            (rowid, json.dumps(embedding)),
        )

    async def _get_rowid(self, mem_id: str) -> Optional[int]:
        """获取情景记忆的内部 rowid。"""
        row = await self._db.fetch_one(
            "SELECT rowid FROM episodic_memory WHERE id = ?",
            (mem_id,),
        )
        return row["rowid"] if row else None

    async def get_episodic(self, mem_id: str) -> Optional[EpisodicMemory]:
        """按 ID 获取单条情景记忆。"""
        row = await self._db.fetch_one(
            "SELECT * FROM episodic_memory WHERE id = ?",
            (mem_id,),
        )
        if not row:
            return None
        return EpisodicMemory(
            id=row["id"], user_id=row["user_id"],
            session_id=row["session_id"], timestamp=row["timestamp"],
            summary=row["summary"], emotion_json=row["emotion_json"],
            intensity=row["intensity"], tags=row["tags"],
            created_at=row["created_at"],
        )

    async def delete_episodic(self, mem_id: str) -> bool:
        """删除一条情景记忆。"""
        rowid = await self._get_rowid(mem_id)
        if rowid is not None:
            # 删除后无法获取 rowid，所以先删 FTS5
            await self._db.execute(
                "DELETE FROM episodic_fts WHERE rowid = ?",
                (rowid,),
            )
        row = await self._db.execute(
            "DELETE FROM episodic_memory WHERE id = ?",
            (mem_id,),
        )
        return row.rowcount > 0

    # ========== B-T14: RRF 混合检索 ==========

    async def retrieve(
        self, user_id: str, query: str, top_k: int = 5
    ) -> List[EpisodicMemory]:
        """
        混合检索情景记忆。

        策略（按可用性降级）：
        1. FTS5 + vec0 双路 RRF 融合（当 embed_fn 和 vec0 均可用）
        2. FTS5 纯文本检索（vec0 不可用时的降级）

        Args:
            user_id: 用户 ID
            query: 搜索关键词
            top_k: 返回 Top-K 条结果

        Returns:
            按相关性降序排列的情景记忆列表
        """
        if not query.strip():
            return []

        has_vec0 = self._embed_fn is not None and await self._has_vec0()

        if has_vec0:
            return await self._retrieve_rrf(user_id, query, top_k)
        else:
            return await self._retrieve_fts_only(user_id, query, top_k)

    async def _retrieve_fts_only(
        self, user_id: str, query: str, top_k: int = 5
    ) -> List[EpisodicMemory]:
        """
        FTS5 + LIKE 联合检索。

        FTS5 默认 unicode61 tokenizer 不切分中文单词（"海边" 无法匹配 "海边散步"），
        因此先用 FTS5 精确匹配，再用 LIKE 子串匹配兜底。
        """
        if not query.strip():
            return []
        try:
            seen_ids: set[int] = set()
            memories: list = []

            # 第 1 步：FTS5 精确 token 匹配，批量获取 rowid
            fts_rows = await self._db.fetch_all(
                "SELECT rowid FROM episodic_fts WHERE episodic_fts MATCH ? ORDER BY rank LIMIT ?",
                (query, top_k * 2),
            )
            if fts_rows:
                fts_ids = [r["rowid"] for r in fts_rows]
                # 批量查询，避免 N+1
                batch = await self._batch_fetch_by_rowids(fts_ids, user_id, top_k)
                memories.extend(batch)
                seen_ids.update(r["rowid"] for r in batch)

            # 第 2 步：FTS5 不足时，用 LIKE 子串检索兜底
            if len(memories) < top_k:
                remaining = top_k - len(memories)
                terms = query.strip().split()
                like_params = [f"%{t}%" for t in terms]
                placeholders = " AND ".join("fts_content LIKE ?" for _ in terms)

                like_rows = await self._db.fetch_all(
                    f"""
                    SELECT rowid, * FROM episodic_memory
                    WHERE user_id = ? AND {placeholders}
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    [user_id] + like_params + [remaining],
                )
                for row in like_rows:
                    if row["rowid"] in seen_ids:
                        continue
                    seen_ids.add(row["rowid"])
                    memories.append(self._row_to_episodic(row))
                    if len(memories) >= top_k:
                        break

            return memories

        except Exception:
            logger.warning("FTS5 检索失败，返回空结果")
            return []

    async def _batch_fetch_by_rowids(
        self, rowids: List[int], user_id: str, limit: int,
    ) -> List[EpisodicMemory]:
        """批量按 rowid + user_id 获取情景记忆，避免 N+1 查询。"""
        if not rowids:
            return []
        # SQLite 不支持动态 ? 列表长度，手动构建
        placeholders = ",".join("?" for _ in rowids)
        rows = await self._db.fetch_all(
            f"""
            SELECT rowid, * FROM episodic_memory
            WHERE rowid IN ({placeholders}) AND user_id = ?
            LIMIT ?
            """,
            (*rowids, user_id, limit),
        )
        # 按原始 rowids 顺序排序（FTS5 rank 顺序）
        rowid_order = {rid: i for i, rid in enumerate(rowids)}
        rows.sort(key=lambda r: rowid_order.get(r["rowid"], 999))
        return [self._row_to_episodic(r) for r in rows]

    async def _retrieve_rrf(
        self, user_id: str, query: str, top_k: int = 5
    ) -> List[EpisodicMemory]:
        """FTS5 + vec0 双路 RRF 融合检索。"""
        fts_top_n = top_k * 3
        vec_top_n = top_k * 3

        # 1. FTS5 搜索（批量获取 rowid）
        fts_results: List[Dict[str, Any]] = []
        if query:
            try:
                fts_rows = await self._db.fetch_all(
                    "SELECT rowid FROM episodic_fts WHERE episodic_fts MATCH ? ORDER BY rank LIMIT ?",
                    (query, fts_top_n),
                )
                if fts_rows:
                    fts_ids = [r["rowid"] for r in fts_rows]
                    # 批量过滤 user_id
                    user_rows = await self._db.fetch_all(
                        f"SELECT rowid, user_id FROM episodic_memory WHERE rowid IN ({','.join('?' for _ in fts_ids)})",
                        fts_ids,
                    )
                    user_by_rowid = {r["rowid"]: r["user_id"] for r in user_rows}
                    for rid, uid in user_by_rowid.items():
                        if uid == user_id:
                            fts_results.append({"rowid": rid})
            except Exception:
                logger.debug("FTS5 检索无结果")

        # 2. 向量搜索
        vec_results: List[Dict[str, Any]] = []
        try:
            embedding = await self._embed_fn(query)
            vec_rows = await self._db.fetch_all(
                """
                SELECT v.rowid, v.distance
                FROM episodic_vec v
                JOIN episodic_memory m ON m.rowid = v.rowid
                WHERE m.user_id = ?
                ORDER BY v.distance
                LIMIT ?
                """,
                (user_id, vec_top_n),
            )
            # 注：实际向量搜索需要高效的 match 操作符，
            # 但 sqlite-vec 的 MATCH 语法不同，此处简单实现为 distance 排序
            for r in vec_rows:
                vec_results.append({"rowid": r["rowid"], "distance": r["distance"]})
        except Exception:
            logger.debug("向量检索无结果或不可用，降级为纯 FTS5")

        # 如果没有向量结果，回退到纯 FTS5
        if not vec_results:
            return await self._retrieve_fts_only(user_id, query, top_k)

        # 3. RRF 融合
        scores: Dict[int, float] = {}
        for rank, r in enumerate(fts_results, start=1):
            rid = r["rowid"]
            scores[rid] = scores.get(rid, 0) + 1.0 / (RRF_K + rank)
        for rank, r in enumerate(vec_results, start=1):
            rid = r["rowid"]
            scores[rid] = scores.get(rid, 0) + 1.0 / (RRF_K + rank)

        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]

        if not sorted_ids:
            return []

        # 4. 批量获取完整记忆（一次查询）
        memories = await self._batch_fetch_by_rowids(sorted_ids, user_id, top_k)
        return memories

    async def retrieve_by_user(
        self, user_id: str, limit: int = 20, offset: int = 0
    ) -> List[EpisodicMemory]:
        """按时间倒序获取用户的情景记忆。"""
        rows = await self._db.fetch_all(
            """
            SELECT * FROM episodic_memory
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            (user_id, limit, offset),
        )
        return [self._row_to_episodic(r) for r in rows]

    def _row_to_episodic(self, row) -> EpisodicMemory:
        return EpisodicMemory(
            id=row["id"], user_id=row["user_id"],
            session_id=row["session_id"], timestamp=row["timestamp"],
            summary=row["summary"], emotion_json=row["emotion_json"],
            intensity=row["intensity"], tags=row["tags"],
            created_at=row["created_at"],
        )

    # ========== B-T14: 事实记忆 CRUD ==========

    async def update_fact(
        self, user_id: str, fact_type: str, key: str, value: Any,
        confidence: float = 1.0,
    ) -> None:
        """
        更新或创建一条事实记忆（幂等，INSERT OR REPLACE）。

        Args:
            user_id: 用户 ID
            fact_type: 事实类型 ('preference','birthday','milestone','sensitive','custom')
            key: 事实键名
            value: 事实值（自动序列化）
            confidence: 置信度 [0, 1]
        """
        str_value = json.dumps(value) if not isinstance(value, str) else value
        await self._db.execute(
            """
            INSERT INTO fact_memory (user_id, fact_type, key, value, confidence, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, fact_type, key)
            DO UPDATE SET value = excluded.value,
                          confidence = excluded.confidence,
                          last_updated = excluded.last_updated
            """,
            (user_id, fact_type, key, str_value, confidence, time.time()),
        )

    async def get_fact(
        self, user_id: str, fact_type: str, key: str
    ) -> Optional[FactMemory]:
        """获取一条事实记忆。"""
        row = await self._db.fetch_one(
            "SELECT * FROM fact_memory WHERE user_id = ? AND fact_type = ? AND key = ?",
            (user_id, fact_type, key),
        )
        if not row:
            return None
        return FactMemory(
            user_id=row["user_id"], fact_type=row["fact_type"],
            key=row["key"], value=row["value"],
            confidence=row["confidence"], last_updated=row["last_updated"],
        )

    async def list_facts(
        self, user_id: str, fact_type: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[FactMemory]:
        """列出用户的事实记忆，可选按类型和置信度筛选。"""
        if fact_type:
            rows = await self._db.fetch_all(
                "SELECT * FROM fact_memory WHERE user_id = ? AND fact_type = ? AND confidence >= ?",
                (user_id, fact_type, min_confidence),
            )
        else:
            rows = await self._db.fetch_all(
                "SELECT * FROM fact_memory WHERE user_id = ? AND confidence >= ?",
                (user_id, min_confidence),
            )
        return [
            FactMemory(
                user_id=r["user_id"], fact_type=r["fact_type"],
                key=r["key"], value=r["value"],
                confidence=r["confidence"], last_updated=r["last_updated"],
            )
            for r in rows
        ]

    async def delete_fact(self, user_id: str, fact_type: str, key: str) -> bool:
        """删除一条事实记忆。"""
        cur = await self._db.execute(
            "DELETE FROM fact_memory WHERE user_id = ? AND fact_type = ? AND key = ?",
            (user_id, fact_type, key),
        )
        return cur.rowcount > 0

    # ========== B-T14: 语义记忆 CRUD ==========

    async def get_semantic(self, user_id: str) -> Dict[str, float]:
        """
        获取用户的语义记忆（关系评分）为字典。

        设计文档定义返回 Dict[str, float]。
        """
        row = await self._db.fetch_one(
            "SELECT * FROM semantic_memory WHERE user_id = ?",
            (user_id,),
        )
        if not row:
            return {}
        return {
            "trust_score": row["trust_score"],
            "intimacy_score": row["intimacy_score"],
            "relationship_stage": row["relationship_stage"],
            "last_updated": row["last_updated"],
        }

    async def update_semantic(
        self, user_id: str, relation: str, delta: float = 0.0,
    ) -> Dict[str, Any]:
        """
        更新用户的语义记忆（单字段增量更新）。

        设计文档定义：(user_id, relation, delta)
        relation 取值: 'trust_score' | 'intimacy_score'
        delta 为增量（可正可负），结果裁剪到 [0, 1]。
        关系阶段根据亲密度自动推断。

        Args:
            user_id: 用户 ID
            relation: 关系字段名，'trust_score' 或 'intimacy_score'
            delta: 增量值

        Returns:
            更新后的语义记忆字典
        """
        current = await self.get_semantic(user_id)
        if not current:
            current = {"trust_score": 0.5, "intimacy_score": 0.3}

        trust = current.get("trust_score", 0.5)
        intimacy = current.get("intimacy_score", 0.3)

        if relation == "trust_score":
            trust = max(0.0, min(1.0, trust + delta))
        elif relation == "intimacy_score":
            intimacy = max(0.0, min(1.0, intimacy + delta))

        stage = self._infer_stage(intimacy)

        await self._db.execute(
            """
            INSERT INTO semantic_memory (user_id, trust_score, intimacy_score, relationship_stage, last_updated)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                trust_score = excluded.trust_score,
                intimacy_score = excluded.intimacy_score,
                relationship_stage = excluded.relationship_stage,
                last_updated = excluded.last_updated
            """,
            (user_id, trust, intimacy, stage, time.time()),
        )

        return {
            "trust_score": trust,
            "intimacy_score": intimacy,
            "relationship_stage": stage,
        }

    @staticmethod
    def _infer_stage(intimacy: float) -> str:
        """根据亲密度评分推断关系阶段。"""
        if intimacy >= 0.8:
            return "soulmate"
        elif intimacy >= 0.6:
            return "close"
        elif intimacy >= 0.4:
            return "friend"
        else:
            return "acquaintance"

    # ========== B-T15: 工作记忆快照与恢复 ==========

    def add_to_working_memory(self, turn: ConversationTurn) -> None:
        """
        将一条对话轮次加入工作记忆（内存 deque）。

        在 deque 中标记为 'unsaved'，等待下次快照写入。
        """
        sid = turn.session_id
        if sid not in self._working_memory:
            self._working_memory[sid] = deque(maxlen=100)
        self._working_memory[sid].append(turn)

    async def snapshot_working_memory(
        self, session_id: str, turns: List[ConversationTurn],
        max_window: int = 20,
    ) -> int:
        """
        快照工作记忆中新轮次到 SQLite。

        每 5 分钟或会话结束时调用。已存在的轮次（按 timestamp 去重）不再写入。

        Args:
            session_id: 会话 ID
            turns: 要快照的对话轮次列表
            max_window: 保留的最大轮次数（超出时截断）

        Returns:
            本次写入的新轮次数
        """
        if not turns:
            return 0

        # 读取已有轮次的时间戳，去重
        rows = await self._db.fetch_all(
            "SELECT timestamp FROM conversation_turns WHERE session_id = ?",
            (session_id,),
        )
        saved_ts = {r["timestamp"] for r in rows}

        new_turns = [t for t in turns if t.timestamp not in saved_ts]
        if not new_turns:
            return 0

        await self._db.execute_many(
            """
            INSERT INTO conversation_turns (session_id, user_id, role, content, timestamp, emotion_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (t.session_id, t.user_id, t.role, t.content, t.timestamp, t.emotion_json)
                for t in new_turns
            ],
        )

        # 更新内部 deque（如果存在）
        if session_id in self._working_memory:
            dq = self._working_memory[session_id]
            for t in new_turns:
                if t not in dq:
                    dq.append(t)
            if len(dq) > max_window:
                self._working_memory[session_id] = deque(
                    list(dq)[-max_window:], maxlen=100
                )

        logger.debug(
            "工作记忆快照完成",
            extra={"session_id": session_id, "saved": len(new_turns)},
        )
        return len(new_turns)

    async def restore_working_memory(
        self, session_id: str, limit: int = 20,
    ) -> Deque[ConversationTurn]:
        """
        从 SQLite 恢复工作记忆到内存。

        在系统重启或会话恢复时调用。

        Args:
            session_id: 会话 ID
            limit: 恢复的最近轮次数

        Returns:
            对话轮次的 deque
        """
        rows = await self._db.fetch_all(
            """
            SELECT * FROM conversation_turns
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (session_id, limit),
        )

        turns = deque(maxlen=100)
        for r in reversed(rows):
            turn = ConversationTurn(
                session_id=r["session_id"], user_id=r["user_id"],
                role=r["role"], content=r["content"],
                timestamp=r["timestamp"], emotion_json=r["emotion_json"] or "",
            )
            turns.append(turn)

        self._working_memory[session_id] = turns
        logger.debug(
            "工作记忆已恢复",
            extra={"session_id": session_id, "count": len(turns)},
        )
        return turns

    # ========== B-T16: 遗忘策略 ==========

    async def apply_forgetting(
        self, decay_factor: float = 0.05,
        threshold: float = 0.1,
    ) -> int:
        """
        对所有事实记忆应用遗忘策略。

        定期（如每小时）调用：降低所有事实记忆的 confidence，
        低于阈值的记录将被删除。

        Args:
            decay_factor: 每次衰减的比例 (0-1)
            threshold: 置信度阈值，低于此值则删除

        Returns:
            被删除的事实记忆数量
        """
        # 1. 衰减所有事实记忆的置信度
        await self._db.execute(
            "UPDATE fact_memory SET confidence = MAX(0.0, confidence - ?)",
            (decay_factor,),
        )

        # 2. 删除低于阈值的记录
        cur = await self._db.execute(
            "DELETE FROM fact_memory WHERE confidence < ?",
            (threshold,),
        )
        deleted = cur.rowcount

        if deleted > 0:
            logger.info(
                "遗忘策略执行完成",
                extra={"decay_factor": decay_factor, "deleted": deleted},
            )

        return deleted

    # ========== 工具方法 ==========

    async def _has_vec0(self) -> bool:
        """检查 vec0 虚拟表是否可用。"""
        if self._vec0_available is not None:
            return self._vec0_available
        try:
            row = await self._db.fetch_one(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='episodic_vec'"
            )
            self._vec0_available = row is not None
        except Exception:
            self._vec0_available = False
        return self._vec0_available

    async def get_stats(self, user_id: str) -> Dict[str, Any]:
        """获取用户的记忆统计信息。"""
        episodic_count = 0
        fact_count = 0
        try:
            r1 = await self._db.fetch_one(
                "SELECT COUNT(*) as c FROM episodic_memory WHERE user_id = ?",
                (user_id,),
            )
            episodic_count = r1["c"] if r1 else 0
            r2 = await self._db.fetch_one(
                "SELECT COUNT(*) as c FROM fact_memory WHERE user_id = ?",
                (user_id,),
            )
            fact_count = r2["c"] if r2 else 0
        except Exception:
            pass

        semantic = await self.get_semantic(user_id)
        return {
            "user_id": user_id,
            "episodic_count": episodic_count,
            "fact_count": fact_count,
            "semantic": semantic if semantic else None,
        }
