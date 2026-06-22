"""
数据库基础设施层

基于 aiosqlite 的 SQLite 连接管理，支持：
- WAL 模式并发读写
- 版本化迁移系统
- 写队列串行化（防止 SQLITE_BUSY）
- 定期 WAL checkpoint
"""

from __future__ import annotations

import logging
import os
import time as _time
from typing import Any, List, Optional, Tuple

import aiosqlite

logger = logging.getLogger("mirror_core.infrastructure.database")


class Database:
    """SQLite 数据库管理器"""

    def __init__(self, path: str = "./data/mirror.db"):
        self._path = os.path.abspath(path)
        self._conn: Optional[aiosqlite.Connection] = None

    @property
    def path(self) -> str:
        return self._path

    async def initialize(self) -> None:
        """初始化数据库：创建目录、连接、WAL 模式、运行迁移"""
        os.makedirs(os.path.dirname(self._path), exist_ok=True)

        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA foreign_keys=ON;")

        await self._run_migrations()
        await self._try_create_vec0()

        logger.info("数据库已初始化", extra={"path": self._path})

    async def close(self) -> None:
        """关闭数据库连接"""
        if self._conn:
            await self._conn.close()
            self._conn = None
        logger.info("数据库连接已关闭")

    async def execute(self, sql: str, params: Tuple = ()) -> aiosqlite.Cursor:
        """执行写操作（INSERT/UPDATE/DELETE/CREATE）"""
        if not self._conn:
            raise RuntimeError("数据库未初始化，请先调用 initialize()")
        cursor = await self._conn.execute(sql, params)
        await self._conn.commit()
        return cursor

    async def execute_many(self, sql: str, params_list: List[Tuple]) -> None:
        """批量执行写操作"""
        if not self._conn:
            raise RuntimeError("数据库未初始化，请先调用 initialize()")
        await self._conn.executemany(sql, params_list)
        await self._conn.commit()

    async def fetch_one(self, sql: str, params: Tuple = ()) -> Optional[aiosqlite.Row]:
        """查询单行"""
        if not self._conn:
            raise RuntimeError("数据库未初始化")
        cursor = await self._conn.execute(sql, params)
        return await cursor.fetchone()

    async def fetch_all(self, sql: str, params: Tuple = ()) -> List[aiosqlite.Row]:
        """查询多行"""
        if not self._conn:
            raise RuntimeError("数据库未初始化")
        cursor = await self._conn.execute(sql, params)
        return await cursor.fetchall()

    async def checkpoint(self) -> None:
        """执行 WAL checkpoint（回收 WAL 文件空间）"""
        if self._conn:
            await self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

    async def _try_create_vec0(self) -> None:
        """尝试创建 sqlite-vec 向量虚拟表（可选，失败时静默降级为 FTS5 纯文本检索）。"""
        if not self._conn:
            return
        try:
            await self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS episodic_vec USING vec0(embedding FLOAT[768])"
            )
            await self._conn.commit()
            logger.info("sqlite-vec 向量表已创建")
        except Exception:
            logger.info("sqlite-vec 不可用，降级为 FTS5 纯文本检索")

    # ---- 迁移系统 ----

    async def _get_schema_version(self) -> int:
        """获取当前数据库版本号"""
        try:
            row = await self.fetch_one(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            )
            return row["version"] if row else 0
        except Exception:
            return 0

    async def _run_migrations(self) -> None:
        """按版本顺序运行所有待执行的迁移"""
        # 确保 schema_version 表存在
        await self.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at REAL DEFAULT (unixepoch())
            )
        """)

        current_version = await self._get_schema_version()
        pending = [m for m in MIGRATIONS if m[0] > current_version]
        pending.sort(key=lambda m: m[0])

        for version, description, sql_list in pending:
            logger.info("执行迁移 v%s: %s", version, description)
            for sql in sql_list:
                await self.execute(sql)
            await self.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (version,)
            )
            logger.info("迁移 v%s 完成", version)


# ===== 迁移定义 =====
# 格式: (version, description, [sql_statements])

MIGRATIONS: List[Tuple[int, str, List[str]]] = [
    (
        1,
        "创建会话绑定表 user_sessions",
        [
            """
            CREATE TABLE IF NOT EXISTS user_sessions (
                internal_user_id TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                platform_user_id TEXT NOT NULL,
                created_at REAL DEFAULT (unixepoch()),
                last_active REAL DEFAULT (unixepoch()),
                UNIQUE(platform, platform_user_id)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_sessions_platform
            ON user_sessions(platform, platform_user_id)
            """,
        ],
    ),
    (
        2,
        "创建事件幂等记录表 processed_events",
        [
            """
            CREATE TABLE IF NOT EXISTS processed_events (
                event_id TEXT PRIMARY KEY,
                processed_at REAL DEFAULT (unixepoch())
            )
            """,
        ],
    ),
    (
        3,
        "创建聊天历史记录表 conversation_turns",
        [
            """
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                emotion_json TEXT
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_turns_session_ts
            ON conversation_turns(session_id, timestamp DESC)
            """,
        ],
    ),
    (
        4,
        "DB-T02: 创建四层记忆表 (episodic_memory / fact_memory / semantic_memory)",
        [
            # 情景记忆表
            """
            CREATE TABLE IF NOT EXISTS episodic_memory (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT,
                timestamp REAL NOT NULL,
                summary TEXT NOT NULL,
                emotion_json TEXT NOT NULL DEFAULT '{}',
                intensity REAL DEFAULT 0.5,
                tags TEXT DEFAULT '',
                embedding BLOB,
                fts_content TEXT,
                created_at REAL DEFAULT (unixepoch())
            )
            """,
            # FTS5 全文搜索虚拟表（独立表，无需 content= 外联）
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts USING fts5(
                user_id UNINDEXED,
                fts_content
            )
            """,
            # 事实记忆表
            """
            CREATE TABLE IF NOT EXISTS fact_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                fact_type TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                last_updated REAL DEFAULT (unixepoch()),
                UNIQUE(user_id, fact_type, key)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_fact_user ON fact_memory(user_id)
            """,
            # 语义记忆表
            """
            CREATE TABLE IF NOT EXISTS semantic_memory (
                user_id TEXT PRIMARY KEY,
                trust_score REAL DEFAULT 0.5,
                intimacy_score REAL DEFAULT 0.3,
                relationship_stage TEXT DEFAULT 'acquaintance',
                last_updated REAL DEFAULT (unixepoch())
            )
            """,
        ],
    ),
    (
        5,
        "添加情景记忆用户时间线复合索引",
        [
            """
            CREATE INDEX IF NOT EXISTS idx_episodic_user_ts
            ON episodic_memory(user_id, timestamp DESC)
            """,
        ],
    ),
    (
        6,
        "DB-T02 修正: 添加 companion_state 到 semantic_memory 表",
        [
            """
            ALTER TABLE semantic_memory ADD COLUMN companion_state TEXT DEFAULT 'Normal'
            """,
        ],
    ),
]
