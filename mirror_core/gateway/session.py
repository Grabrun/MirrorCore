"""
会话绑定管理

DB-T01: 设计并创建会话绑定表 user_sessions
实现内存 + SQLite 两级缓存，重启后从 SQLite 恢复会话绑定关系。
"""

from __future__ import annotations

import logging
import time as _time
import uuid
from typing import Dict, Optional, Tuple

from mirror_core.infrastructure.database import Database

logger = logging.getLogger("mirror_core.gateway.session")


class SessionManager:
    """
    会话绑定管理器

    维护 (platform, platform_user_id) → internal_user_id 的映射关系。
    内存缓存提供 O(1) 查找，SQLite 提供持久化存储。

    使用方式：
        sm = SessionManager(db)
        await sm.initialize()
        internal_id, session_id = await sm.resolve("wechat", "o_xxxx")
    """

    def __init__(self, db: Database):
        self._db = db
        # 缓存: (platform, platform_user_id) → internal_user_id
        self._cache: Dict[Tuple[str, str], str] = {}
        # 缓存: internal_user_id → (platform, platform_user_id)
        self._reverse_cache: Dict[str, Tuple[str, str]] = {}
        # 最后活跃时间更新记录（用于节流）
        self._last_active_timestamps: Dict[str, float] = {}

    async def initialize(self) -> None:
        """从 SQLite 加载所有已有会话到内存缓存"""
        rows = await self._db.fetch_all(
            "SELECT internal_user_id, platform, platform_user_id FROM user_sessions"
        )
        for row in rows:
            key = (row["platform"], row["platform_user_id"])
            self._cache[key] = row["internal_user_id"]
            self._reverse_cache[row["internal_user_id"]] = key

        logger.info("会话管理器已初始化，已加载 %d 个会话", len(self._cache))

    async def resolve(
        self, platform: str, platform_user_id: str
    ) -> Tuple[str, str]:
        """
        解析 (platform, platform_user_id) 为 internal_user_id 和 session_id。

        如果已存在绑定关系，直接返回缓存的 internal_user_id；
        否则创建新的 internal_user_id，持久化并返回。

        Returns:
            (internal_user_id, session_id)
            session_id 当前与 internal_user_id 相同（单用户单会话模型）
        """
        key = (platform, platform_user_id)

        # 查内存缓存
        if key in self._cache:
            internal_id = self._cache[key]
            # 更新活跃时间（异步，不阻塞）
            await self._update_last_active(internal_id)
            return internal_id, internal_id

        # 查 SQLite（防御性：防止缓存 miss）
        row = await self._db.fetch_one(
            "SELECT internal_user_id FROM user_sessions "
            "WHERE platform = ? AND platform_user_id = ?",
            (platform, platform_user_id),
        )
        if row:
            internal_id = row["internal_user_id"]
            self._cache[key] = internal_id
            self._reverse_cache[internal_id] = key
            await self._update_last_active(internal_id)
            return internal_id, internal_id

        # 创建新会话
        internal_id = _generate_user_id()
        await self._db.execute(
            "INSERT INTO user_sessions (internal_user_id, platform, platform_user_id, "
            "created_at, last_active) VALUES (?, ?, ?, ?, ?)",
            (internal_id, platform, platform_user_id, _time.time(), _time.time()),
        )
        self._cache[key] = internal_id
        self._reverse_cache[internal_id] = key

        logger.info(
            "创建新会话",
            extra={
                "internal_user_id": internal_id,
                "platform": platform,
                "platform_user_id": platform_user_id,
            },
        )
        return internal_id, internal_id

    async def get_platform_info(
        self, internal_user_id: str
    ) -> Optional[Tuple[str, str]]:
        """
        根据 internal_user_id 获取平台信息。

        Returns:
            (platform, platform_user_id) 或 None（未找到）
        """
        # 查内存缓存
        if internal_user_id in self._reverse_cache:
            return self._reverse_cache[internal_user_id]

        # 查 SQLite
        row = await self._db.fetch_one(
            "SELECT platform, platform_user_id FROM user_sessions "
            "WHERE internal_user_id = ?",
            (internal_user_id,),
        )
        if row:
            key = (row["platform"], row["platform_user_id"])
            self._cache[key] = internal_user_id
            self._reverse_cache[internal_user_id] = key
            return key
        return None

    async def remove_session(
        self, platform: str, platform_user_id: str
    ) -> bool:
        """移除会话绑定关系"""
        key = (platform, platform_user_id)
        internal_id = self._cache.pop(key, None)
        if internal_id:
            self._reverse_cache.pop(internal_id, None)
        await self._db.execute(
            "DELETE FROM user_sessions WHERE platform = ? AND platform_user_id = ?",
            (platform, platform_user_id),
        )
        return internal_id is not None

    async def _update_last_active(self, internal_user_id: str) -> None:
        """
        异步更新最后活跃时间（带节流：每分钟最多更新一次）

        高频场景下（如深夜密集对话），避免每条消息都触发 SQL 写入。
        """
        now = _time.time()
        last = self._last_active_timestamps.get(internal_user_id, 0.0)
        if now - last < 60.0:
            return
        self._last_active_timestamps[internal_user_id] = now
        try:
            await self._db.execute(
                "UPDATE user_sessions SET last_active = ? WHERE internal_user_id = ?",
                (now, internal_user_id),
            )
        except Exception:
            logger.warning("更新活跃时间失败", extra={"internal_user_id": internal_user_id})

    @property
    def active_session_count(self) -> int:
        """返回内存中缓存的活跃会话数"""
        return len(self._cache)


def _generate_user_id() -> str:
    """生成全局唯一的 internal_user_id"""
    return f"u_{uuid.uuid4().hex[:16]}"
