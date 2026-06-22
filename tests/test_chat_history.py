"""
聊天历史 API 测试

覆盖 B-T10: 聊天历史查询 API 的验收标准：
1. GET /api/v1/chat/history?session_id=xxx&limit=20 接口工作正常
2. 按 timestamp 降序返回结构正确的消息
3. 分页查询 (before 参数)
4. 空结果、参数边界、异常场景
5. 鉴权机制验证
"""

import os
import tempfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mirror_core.api.chat import create_chat_router
from mirror_core.infrastructure.database import Database


# ---- Fixtures ----

@pytest.fixture
def db_path():
    """生成临时数据库文件路径"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
async def db(db_path):
    """创建并初始化临时数据库，插入测试数据后返回"""
    db = Database(db_path)
    await db.initialize()

    test_turns = [
        ("s001", "u001", "user", "你好", 1000.0, None),
        ("s001", "u001", "assistant", "你好！有什么可以帮助你的？", 1001.0, '{"mood":"happy"}'),
        ("s001", "u001", "user", "今天天气怎么样？", 1002.0, None),
        ("s001", "u001", "assistant", "今天天气晴朗，适合出门散步。", 1003.0, '{"mood":"calm"}'),
        ("s001", "u001", "user", "帮我查一下明天的日程", 1004.0, None),
        ("s002", "u002", "user", "另一个会话的消息", 2000.0, None),
    ]

    await db.execute_many(
        "INSERT INTO conversation_turns "
        "(session_id, user_id, role, content, timestamp, emotion_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        test_turns,
    )

    yield db
    await db.close()


@pytest.fixture
def client(db):
    """创建 FastAPI 测试客户端（本地开发模式，无 Token 鉴权）"""
    app = FastAPI()
    app.include_router(create_chat_router(db))
    return TestClient(app)


@pytest.fixture
def base_params():
    """基础请求参数"""
    return {"session_id": "s001", "user_id": "u001"}


# ---- API 功能测试（本地开发模式 = 无 Token） ----

class TestChatHistoryAPI:
    """聊天历史查询 API 功能测试"""

    def test_get_history_returns_messages(self, client, base_params):
        """应返回指定 session 的历史消息列表"""
        response = client.get("/api/v1/chat/history", params=base_params)
        assert response.status_code == 200
        assert len(response.json()["messages"]) == 5

    def test_history_sorted_by_timestamp_desc(self, client, base_params):
        """消息应按 timestamp 降序排列"""
        response = client.get("/api/v1/chat/history", params=base_params)
        timestamps = [m["timestamp"] for m in response.json()["messages"]]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_history_message_structure(self, client, base_params):
        """每条消息应包含 role, content, timestamp 字段"""
        response = client.get("/api/v1/chat/history", params=base_params)
        msg = response.json()["messages"][0]

        assert "role" in msg and msg["role"] in ("user", "assistant")
        assert "content" in msg and isinstance(msg["content"], str)
        assert "timestamp" in msg and isinstance(msg["timestamp"], (int, float))

    def test_limit_parameter(self, client, base_params):
        """limit 参数应正确限制返回条数"""
        response = client.get("/api/v1/chat/history", params={**base_params, "limit": 2})
        assert len(response.json()["messages"]) == 2

    def test_limit_upper_boundary(self, client, base_params):
        """limit=200 应正常工作"""
        response = client.get("/api/v1/chat/history", params={**base_params, "limit": 200})
        assert response.status_code == 200

    def test_before_parameter_pagination(self, client, base_params):
        """before 参数应实现基于时间戳的滚动分页"""
        # 获取前 2 条（最新）
        r = client.get("/api/v1/chat/history", params={**base_params, "limit": 2})
        first_batch = r.json()["messages"]
        earliest_ts = first_batch[-1]["timestamp"]

        # 以 earliest_ts 为 before，获取更早的消息
        r2 = client.get("/api/v1/chat/history", params={**base_params, "limit": 10, "before": earliest_ts})
        data = r2.json()

        assert len(data["messages"]) > 0
        for msg in data["messages"]:
            assert msg["timestamp"] < earliest_ts

    def test_empty_session(self, client):
        """不存在的 session 应返回空列表"""
        r = client.get("/api/v1/chat/history", params={"session_id": "nonexistent", "user_id": "u001"})
        assert r.status_code == 200
        assert r.json()["messages"] == []

    def test_data_isolation_by_user_id(self, client):
        """user_id 不匹配时不应返回数据"""
        # s002 属于 u002，用 u001 查询应返回空
        r = client.get("/api/v1/chat/history", params={"session_id": "s002", "user_id": "u001"})
        assert r.status_code == 200
        assert r.json()["messages"] == []

        # 正确的 user_id 应返回数据
        r2 = client.get("/api/v1/chat/history", params={"session_id": "s002", "user_id": "u002"})
        assert r2.status_code == 200
        assert len(r2.json()["messages"]) == 1

    def test_different_sessions_isolated(self, client):
        """不同 session 的数据应隔离"""
        r1 = client.get("/api/v1/chat/history", params={"session_id": "s001", "user_id": "u001"})
        r2 = client.get("/api/v1/chat/history", params={"session_id": "s002", "user_id": "u002"})
        assert len(r1.json()["messages"]) == 5
        assert len(r2.json()["messages"]) == 1

    def test_limit_out_of_range(self, client, base_params):
        """limit=0 或 >200 应返回 422"""
        assert client.get("/api/v1/chat/history", params={**base_params, "limit": 0}).status_code == 422
        assert client.get("/api/v1/chat/history", params={**base_params, "limit": 201}).status_code == 422

    def test_missing_session_id(self, client):
        """缺少 session_id 应返回 422"""
        assert client.get("/api/v1/chat/history", params={"user_id": "u001"}).status_code == 422

    def test_missing_user_id(self, client):
        """缺少 user_id 应返回 422"""
        assert client.get("/api/v1/chat/history", params={"session_id": "s001"}).status_code == 422


# ---- 鉴权测试 ----

class TestChatHistoryAuth:
    """Bearer Token 鉴权测试"""

    @pytest.fixture(autouse=True)
    def _setup(self, db_path):
        """每个测试使用独立 app 实例"""
        self.db = Database(db_path)

    @pytest.mark.asyncio
    async def _init_db(self):
        await self.db.initialize()
        await self.db.execute(
            "INSERT INTO conversation_turns (session_id, user_id, role, content, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            ("s001", "u001", "user", "test", 1000.0),
        )

    def _make_client(self):
        app = FastAPI()
        app.include_router(create_chat_router(self.db))
        return TestClient(app)

    @pytest.mark.asyncio
    async def test_no_token_no_auth_local_dev(self):
        """未设置 MIRROR_API_TOKEN 时跳过鉴权（本地开发模式）"""
        # 确保 env 未设置
        old = os.environ.pop("MIRROR_API_TOKEN", None)
        try:
            await self._init_db()
            client = self._make_client()
            r = client.get("/api/v1/chat/history", params={"session_id": "s001", "user_id": "u001"})
            assert r.status_code == 200
        finally:
            if old is not None:
                os.environ["MIRROR_API_TOKEN"] = old
        await self.db.close()

    @pytest.mark.asyncio
    async def test_valid_token_allowed(self):
        """正确 Token 应通过鉴权"""
        os.environ["MIRROR_API_TOKEN"] = "test-token-123"
        try:
            await self._init_db()
            client = self._make_client()
            r = client.get(
                "/api/v1/chat/history",
                params={"session_id": "s001", "user_id": "u001"},
                headers={"Authorization": "Bearer test-token-123"},
            )
            assert r.status_code == 200
        finally:
            del os.environ["MIRROR_API_TOKEN"]
        await self.db.close()

    @pytest.mark.asyncio
    async def test_invalid_token_rejected(self):
        """错误 Token 应返回 403"""
        os.environ["MIRROR_API_TOKEN"] = "correct-token"
        try:
            await self._init_db()
            client = self._make_client()
            r = client.get(
                "/api/v1/chat/history",
                params={"session_id": "s001", "user_id": "u001"},
                headers={"Authorization": "Bearer wrong-token"},
            )
            assert r.status_code == 403
        finally:
            del os.environ["MIRROR_API_TOKEN"]
        await self.db.close()

    @pytest.mark.asyncio
    async def test_missing_auth_header(self):
        """Token 模式下缺少 Authorization 头应返回 401"""
        os.environ["MIRROR_API_TOKEN"] = "some-token"
        try:
            await self._init_db()
            client = self._make_client()
            r = client.get("/api/v1/chat/history", params={"session_id": "s001", "user_id": "u001"})
            assert r.status_code == 401
        finally:
            del os.environ["MIRROR_API_TOKEN"]
        await self.db.close()

    @pytest.mark.asyncio
    async def test_wrong_auth_scheme(self):
        """非 Bearer 格式应返回 401"""
        os.environ["MIRROR_API_TOKEN"] = "some-token"
        try:
            await self._init_db()
            client = self._make_client()
            r = client.get(
                "/api/v1/chat/history",
                params={"session_id": "s001", "user_id": "u001"},
                headers={"Authorization": "Basic dXNlcjpwYXNz"},
            )
            assert r.status_code == 401
        finally:
            del os.environ["MIRROR_API_TOKEN"]
        await self.db.close()

    @pytest.mark.asyncio
    async def test_empty_token_rejected(self):
        """空 Token 应返回 401"""
        os.environ["MIRROR_API_TOKEN"] = "some-token"
        try:
            await self._init_db()
            client = self._make_client()
            r = client.get(
                "/api/v1/chat/history",
                params={"session_id": "s001", "user_id": "u001"},
                headers={"Authorization": "Bearer "},
            )
            assert r.status_code == 401
        finally:
            del os.environ["MIRROR_API_TOKEN"]
        await self.db.close()


# ---- 异常场景测试 ----

class TestErrorScenarios:

    @pytest.mark.asyncio
    async def test_db_not_initialized(self):
        """数据库未初始化时应返回 503"""
        db = Database("/tmp/nonexistent/not_initialized.db")
        app = FastAPI()
        app.include_router(create_chat_router(db))
        client = TestClient(app)

        r = client.get("/api/v1/chat/history", params={"session_id": "s001", "user_id": "u001"})
        assert r.status_code == 503


# ---- 数据库迁移测试 ----

class TestConversationTurnsMigration:

    @pytest.mark.asyncio
    async def test_table_created(self, db):
        """迁移 v3 应创建 conversation_turns 表"""
        row = await db.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_turns'"
        )
        assert row is not None

    @pytest.mark.asyncio
    async def test_column_structure(self, db):
        """表应有正确列结构"""
        columns = await db.fetch_all("PRAGMA table_info(conversation_turns)")
        col_names = {r["name"] for r in columns}
        assert col_names == {"id", "session_id", "user_id", "role", "content", "timestamp", "emotion_json"}

    @pytest.mark.asyncio
    async def test_role_check_constraint(self, db):
        """CHECK 约束应拒绝无效 role 值"""
        await db.execute(
            "INSERT INTO conversation_turns (session_id, user_id, role, content, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            ("s_t", "u_t", "user", "有效", 1000.0),
        )
        with pytest.raises(Exception):
            await db.execute(
                "INSERT INTO conversation_turns (session_id, user_id, role, content, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                ("s_t", "u_t", "invalid", "无效", 1001.0),
            )

    @pytest.mark.asyncio
    async def test_index_exists(self, db):
        """idx_turns_session_ts 索引应存在"""
        row = await db.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_turns_session_ts'"
        )
        assert row is not None

    @pytest.mark.asyncio
    async def test_migration_version_applied(self, db):
        """迁移 v3 应被记录"""
        row = await db.fetch_one("SELECT version FROM schema_version WHERE version = 3")
        assert row is not None
