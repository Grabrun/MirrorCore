"""
会话管理器测试

覆盖：
- 新会话创建
- 已有会话解析（含内存缓存命中）
- 重启恢复（从 SQLite 重建缓存）
- 平台信息查询
- 会话移除
"""

import os
import tempfile

import pytest

from mirror_core.gateway.session import SessionManager
from mirror_core.infrastructure.database import Database


@pytest.fixture
async def db_session():
    """创建带数据库的 SessionManager"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db = Database(db_path)
    await db.initialize()
    sm = SessionManager(db)
    await sm.initialize()

    yield db, sm

    await db.close()
    os.unlink(db_path)


class TestSessionManager:
    """SessionManager 测试"""

    @pytest.mark.asyncio
    async def test_new_session_creates_entry(self, db_session):
        """新会话应创建绑定记录并返回 internal_user_id"""
        _, sm = db_session

        internal_id, session_id = await sm.resolve("test", "user001")

        assert internal_id.startswith("u_")
        assert session_id == internal_id

    @pytest.mark.asyncio
    async def test_existing_session_returns_same_id(self, db_session):
        """相同 platform + platform_user_id 应返回相同的 internal_user_id"""
        _, sm = db_session

        id1, _ = await sm.resolve("test", "user001")
        id2, _ = await sm.resolve("test", "user001")

        assert id1 == id2

    @pytest.mark.asyncio
    async def test_different_platforms_return_different_ids(self, db_session):
        """不同平台的相同 user_id 应返回不同的 internal_user_id"""
        _, sm = db_session

        id_web, _ = await sm.resolve("webchat", "user001")
        id_qq, _ = await sm.resolve("qq", "user001")

        assert id_web != id_qq

    @pytest.mark.asyncio
    async def test_cache_hit_returns_without_db_query(self, db_session):
        """缓存命中不应查数据库"""
        db, sm = db_session

        # 第一次 resolve：写入缓存 + DB
        internal_id, _ = await sm.resolve("test", "user001")
        assert ("test", "user001") in sm._cache

    @pytest.mark.asyncio
    async def test_get_platform_info(self, db_session):
        """get_platform_info 应返回正确的平台信息"""
        _, sm = db_session

        internal_id, _ = await sm.resolve("test", "user001")
        info = await sm.get_platform_info(internal_id)

        assert info is not None
        assert info == ("test", "user001")

    @pytest.mark.asyncio
    async def test_get_platform_info_nonexistent(self, db_session):
        """不存在的 internal_user_id 应返回 None"""
        _, sm = db_session

        info = await sm.get_platform_info("u_nonexistent")
        assert info is None

    @pytest.mark.asyncio
    async def test_remove_session(self, db_session):
        """移除会话后应无法再解析"""
        _, sm = db_session

        await sm.resolve("test", "user001")
        result = await sm.remove_session("test", "user001")
        assert result is True

        # 再次 resolve 应创建新 ID
        new_id, _ = await sm.resolve("test", "user001")
        assert new_id != "u_test001"  # 全新的 ID

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self, db_session):
        """移除不存在的会话应返回 False"""
        _, sm = db_session

        result = await sm.remove_session("nonexistent", "user")
        assert result is False

    @pytest.mark.asyncio
    async def test_active_session_count(self, db_session):
        """active_session_count 应反映缓存中的会话数"""
        _, sm = db_session

        assert sm.active_session_count == 0
        await sm.resolve("test", "user001")
        assert sm.active_session_count == 1
        await sm.resolve("test", "user002")
        assert sm.active_session_count == 2
        await sm.resolve("test", "user001")  # 重复
        assert sm.active_session_count == 2  # 不增加


class TestSessionPersistence:
    """会话持久化测试（模拟重启场景）"""

    @pytest.mark.asyncio
    async def test_reload_from_db(self):
        """重启后应从 SQLite 恢复会话绑定"""
        # 第一次：创建数据库和会话
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        db1 = Database(db_path)
        await db1.initialize()
        sm1 = SessionManager(db1)
        await sm1.initialize()

        internal_id, _ = await sm1.resolve("test", "persist_user")
        await db1.close()

        # 第二次：模拟重启，使用同一个数据库文件
        db2 = Database(db_path)
        await db2.initialize()
        sm2 = SessionManager(db2)
        await sm2.initialize()

        restored_id, _ = await sm2.resolve("test", "persist_user")
        assert restored_id == internal_id
        assert sm2.active_session_count == 1

        await db2.close()
        os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_restored_session_matches(self):
        """恢复的会话应能正确查询平台信息"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        # 第一次：创建
        db1 = Database(db_path)
        await db1.initialize()
        sm1 = SessionManager(db1)
        await sm1.initialize()
        internal_id, _ = await sm1.resolve("wechat", "wx_openid_123")
        await db1.close()

        # 第二次：恢复
        db2 = Database(db_path)
        await db2.initialize()
        sm2 = SessionManager(db2)
        await sm2.initialize()

        info = await sm2.get_platform_info(internal_id)
        assert info == ("wechat", "wx_openid_123")

        await db2.close()
        os.unlink(db_path)
