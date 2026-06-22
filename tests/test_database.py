"""
数据库基础设施测试

覆盖：
- 数据库初始化与 WAL 模式
- 迁移系统（schema_version 表自动创建、版本追踪）
- 基本 CRUD 操作
- 会话绑定表的迁移
"""

import os
import tempfile

import pytest

from mirror_core.infrastructure.database import Database


@pytest.fixture
async def db():
    """创建临时数据库"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db = Database(db_path)
    await db.initialize()

    yield db

    await db.close()
    os.unlink(db_path)


class TestDatabaseInitialization:
    """数据库初始化测试"""

    @pytest.mark.asyncio
    async def test_initialize_creates_db_file(self):
        """初始化应创建数据库文件"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        os.unlink(db_path)  # 删除后让 Database 重新创建
        assert not os.path.exists(db_path)

        db = Database(db_path)
        await db.initialize()

        assert os.path.exists(db_path)
        await db.close()
        os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_initialize_runs_migrations(self, db):
        """初始化应自动创建 schema_version 表"""
        row = await db.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        assert row is not None

    @pytest.mark.asyncio
    async def test_migration_tracks_version(self, db):
        """迁移版本号应被正确记录"""
        row = await db.fetch_one(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        )
        assert row is not None
        assert row["version"] >= 2  # 应有至少 2 个迁移


class TestDatabaseOperations:
    """数据库基本操作测试"""

    @pytest.mark.asyncio
    async def test_execute_and_fetch(self, db):
        """基本写入和查询"""
        await db.execute(
            "INSERT INTO user_sessions (internal_user_id, platform, platform_user_id) "
            "VALUES (?, ?, ?)",
            ("u_test001", "test", "user001"),
        )

        row = await db.fetch_one(
            "SELECT * FROM user_sessions WHERE internal_user_id = ?",
            ("u_test001",),
        )
        assert row is not None
        assert row["platform"] == "test"
        assert row["platform_user_id"] == "user001"

    @pytest.mark.asyncio
    async def test_fetch_one_returns_none_for_missing(self, db):
        """查询不存在的记录应返回 None"""
        row = await db.fetch_one(
            "SELECT * FROM user_sessions WHERE internal_user_id = ?",
            ("u_nonexistent",),
        )
        assert row is None

    @pytest.mark.asyncio
    async def test_fetch_all(self, db):
        """批量查询"""
        # 插入 3 条记录
        for i in range(3):
            await db.execute(
                "INSERT INTO user_sessions (internal_user_id, platform, platform_user_id) "
                "VALUES (?, ?, ?)",
                (f"u_test{i:03d}", "test", f"user{i:03d}"),
            )

        rows = await db.fetch_all(
            "SELECT * FROM user_sessions WHERE platform = ? ORDER BY internal_user_id",
            ("test",),
        )
        assert len(rows) == 3

    @pytest.mark.asyncio
    async def test_unique_constraint(self, db):
        """UNIQUE 约束应生效"""
        await db.execute(
            "INSERT INTO user_sessions (internal_user_id, platform, platform_user_id) "
            "VALUES (?, ?, ?)",
            ("u_test001", "test", "user001"),
        )

        with pytest.raises(Exception):
            await db.execute(
                "INSERT INTO user_sessions (internal_user_id, platform, platform_user_id) "
                "VALUES (?, ?, ?)",
                ("u_test002", "test", "user001"),  # 相同 platform + platform_user_id
            )

    @pytest.mark.asyncio
    async def test_execute_many(self, db):
        """批量执行"""
        records = [
            (f"u_batch{i:03d}", "batch", f"user{i:03d}")
            for i in range(10)
        ]
        await db.execute_many(
            "INSERT INTO user_sessions (internal_user_id, platform, platform_user_id) "
            "VALUES (?, ?, ?)",
            records,
        )

        rows = await db.fetch_all(
            "SELECT * FROM user_sessions WHERE platform = ?", ("batch",)
        )
        assert len(rows) == 10

    @pytest.mark.asyncio
    async def test_update(self, db):
        """更新操作"""
        await db.execute(
            "INSERT INTO user_sessions (internal_user_id, platform, platform_user_id) "
            "VALUES (?, ?, ?)",
            ("u_upd001", "upd", "user001"),
        )

        await db.execute(
            "UPDATE user_sessions SET last_active = ? WHERE internal_user_id = ?",
            (1234567890.0, "u_upd001"),
        )

        row = await db.fetch_one(
            "SELECT last_active FROM user_sessions WHERE internal_user_id = ?",
            ("u_upd001",),
        )
        assert row["last_active"] == 1234567890.0

    @pytest.mark.asyncio
    async def test_delete(self, db):
        """删除操作"""
        await db.execute(
            "INSERT INTO user_sessions (internal_user_id, platform, platform_user_id) "
            "VALUES (?, ?, ?)",
            ("u_del001", "del", "user001"),
        )

        await db.execute(
            "DELETE FROM user_sessions WHERE internal_user_id = ?",
            ("u_del001",),
        )

        row = await db.fetch_one(
            "SELECT * FROM user_sessions WHERE internal_user_id = ?",
            ("u_del001",),
        )
        assert row is None


class TestDatabaseMigrations:
    """迁移系统测试"""

    @pytest.mark.asyncio
    async def test_schema_version_tracking(self, db):
        """schema_version 应追踪所有已应用的迁移"""
        versions = await db.fetch_all(
            "SELECT version FROM schema_version ORDER BY version"
        )
        applied = [row["version"] for row in versions]
        assert 1 in applied  # v1: user_sessions
        assert 2 in applied  # v2: processed_events

    @pytest.mark.asyncio
    async def test_idempotent_migration(self, db):
        """重复初始化不应导致迁移失败"""
        # 第二次初始化应安全（迁移是幂等的）
        await db._run_migrations()
        # 不应抛出异常

    @pytest.mark.asyncio
    async def test_idempotent_table_creation(self, db):
        """表已存在时 CREATE IF NOT EXISTS 应安全"""
        # 迁移 SQL 已使用 IF NOT EXISTS，再执行应 OK
        await db.execute(
            "CREATE TABLE IF NOT EXISTS user_sessions ("
            "    internal_user_id TEXT PRIMARY KEY,"
            "    platform TEXT NOT NULL,"
            "    platform_user_id TEXT NOT NULL"
            ")"
        )
