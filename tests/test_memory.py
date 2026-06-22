"""
记忆引擎测试

覆盖 User Story 5 验收标准：
1. DB-T02: 四层记忆表迁移正确
2. B-T14: 情景/事实/语义记忆 CRUD + FTS5 检索
3. B-T15: 工作记忆快照与恢复
4. B-T16: 遗忘策略
"""

import os
import tempfile

import pytest

from mirror_core.infrastructure.database import Database
from mirror_core.memory.engine import (
    ConversationTurn,
    EpisodicMemory,
    MemoryEngine,
)


# ===== Fixtures =====

@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
async def db(db_path):
    db = Database(db_path)
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
async def engine(db):
    return MemoryEngine(db)


# ===== DB-T02: 迁移测试 =====

class TestMemoryMigrations:

    async def test_episodic_memory_table(self, db):
        r = await db.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='episodic_memory'"
        )
        assert r is not None

    async def test_fact_memory_table(self, db):
        r = await db.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_memory'"
        )
        assert r is not None

    async def test_semantic_memory_table(self, db):
        r = await db.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='semantic_memory'"
        )
        assert r is not None

    async def test_episodic_fts_virtual_table(self, db):
        r = await db.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='episodic_fts'"
        )
        assert r is not None

    async def test_fact_index_exists(self, db):
        r = await db.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_fact_user'"
        )
        assert r is not None

    async def test_migration_v4_applied(self, db):
        r = await db.fetch_one("SELECT version FROM schema_version WHERE version = 4")
        assert r is not None

    async def test_migration_v5_applied(self, db):
        r = await db.fetch_one("SELECT version FROM schema_version WHERE version = 5")
        assert r is not None

    async def test_episodic_user_ts_index_exists(self, db):
        """迁移 v5 应创建复合索引（F-006 修复验证）"""
        r = await db.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_episodic_user_ts'"
        )
        assert r is not None

    async def test_episodic_columns(self, db):
        cols = await db.fetch_all("PRAGMA table_info(episodic_memory)")
        names = {c["name"] for c in cols}
        expected = {"id", "user_id", "session_id", "timestamp", "summary",
                    "emotion_json", "intensity", "tags", "embedding", "fts_content", "created_at"}
        assert names == expected

    async def test_fact_unique_constraint(self, db):
        await db.execute(
            "INSERT INTO fact_memory (user_id, fact_type, key, value) VALUES (?, ?, ?, ?)",
            ("u1", "preference", "color", "blue"),
        )
        with pytest.raises(Exception):
            await db.execute(
                "INSERT INTO fact_memory (user_id, fact_type, key, value) VALUES (?, ?, ?, ?)",
                ("u1", "preference", "color", "red"),
            )


# ===== B-T14: 情景记忆 CRUD =====

class TestEpisodicCRUD:

    async def test_store_and_get(self, engine):
        mem = EpisodicMemory(
            user_id="u1", session_id="s1", summary="用户说今天很开心",
            emotion_json='{"mood": 0.8}', intensity=0.7, tags="开心,日常",
        )
        mid = await engine.store_episodic(mem)
        assert len(mid) == 32  # uuid4 hex

        fetched = await engine.get_episodic(mid)
        assert fetched is not None
        assert fetched.user_id == "u1"
        assert fetched.summary == "用户说今天很开心"
        assert fetched.intensity == 0.7

    async def test_store_multiple(self, engine):
        for i in range(5):
            await engine.store_episodic(EpisodicMemory(
                user_id="u1", summary=f"记忆 #{i}", timestamp=float(i),
            ))

        memories = await engine.retrieve_by_user("u1", limit=10)
        assert len(memories) == 5

    async def test_delete(self, engine):
        mid = await engine.store_episodic(EpisodicMemory(user_id="u1", summary="待删除"))
        assert await engine.delete_episodic(mid) is True
        assert await engine.get_episodic(mid) is None
        assert await engine.delete_episodic(mid) is False

    async def test_get_nonexistent(self, engine):
        assert await engine.get_episodic("nonexistent") is None


# ===== B-T14: 事实记忆 CRUD =====

class TestFactCRUD:

    async def test_update_and_get(self, engine):
        await engine.update_fact("u1", "preference", "color", "blue", 0.9)
        fact = await engine.get_fact("u1", "preference", "color")
        assert fact is not None
        assert fact.value == "blue"
        assert fact.confidence == 0.9

    async def test_update_idempotent(self, engine):
        await engine.update_fact("u1", "preference", "color", "blue")
        await engine.update_fact("u1", "preference", "color", "red")
        fact = await engine.get_fact("u1", "preference", "color")
        assert fact.value == "red"

    async def test_get_nonexistent(self, engine):
        assert await engine.get_fact("u1", "preference", "nonexistent") is None

    async def test_list_facts(self, engine):
        await engine.update_fact("u1", "preference", "color", "blue")
        await engine.update_fact("u1", "preference", "food", "pizza")
        await engine.update_fact("u1", "birthday", "date", "2020-01-01")
        await engine.update_fact("u2", "preference", "color", "green")

        assert len(await engine.list_facts("u1")) == 3
        assert len(await engine.list_facts("u1", fact_type="preference")) == 2
        assert len(await engine.list_facts("u2")) == 1

    async def test_delete_fact(self, engine):
        await engine.update_fact("u1", "preference", "color", "blue")
        assert await engine.delete_fact("u1", "preference", "color") is True
        assert await engine.get_fact("u1", "preference", "color") is None

    async def test_confidence_filter(self, engine):
        await engine.update_fact("u1", "preference", "a", "1", confidence=0.9)
        await engine.update_fact("u1", "preference", "b", "2", confidence=0.3)

        all_facts = await engine.list_facts("u1")
        assert len(all_facts) == 2

        high_facts = await engine.list_facts("u1", min_confidence=0.5)
        assert len(high_facts) == 1


# ===== B-T14: 语义记忆 CRUD（F-001+F-002 修复适配） =====

class TestSemanticCRUD:

    async def test_get_nonexistent_returns_empty_dict(self, engine):
        """设计文档要求返回 Dict，不存在时返回 {}"""
        assert await engine.get_semantic("u1") == {}

    async def test_update_creates_new(self, engine):
        """update_semantic(user_id, relation, delta) 设计文档签名"""
        result = await engine.update_semantic("u1", "trust_score", 0.1)
        assert result["trust_score"] == 0.6  # 0.5 + 0.1
        assert result["intimacy_score"] == 0.3  # 未变
        assert result["relationship_stage"] == "acquaintance"

    async def test_update_trust_and_intimacy_separately(self, engine):
        """信任和亲密度可分别更新"""
        await engine.update_semantic("u1", "trust_score", 0.3)
        result = await engine.update_semantic("u1", "intimacy_score", 0.4)
        assert result["trust_score"] == 0.8  # 0.5 + 0.3
        assert result["intimacy_score"] == 0.7  # 0.3 + 0.4
        assert result["relationship_stage"] == "close"

    async def test_clamp_to_range(self, engine):
        result = await engine.update_semantic("u1", "trust_score", 10.0)
        assert result["trust_score"] == 1.0
        result = await engine.update_semantic("u1", "intimacy_score", -10.0)
        assert result["intimacy_score"] == 0.0

    async def test_get_semantic_returns_dict(self, engine):
        """get_semantic 返回 Dict 而非对象（F-002 修复验证）"""
        await engine.update_semantic("u1", "intimacy_score", 0.1)
        result = await engine.get_semantic("u1")
        assert isinstance(result, dict)
        assert "trust_score" in result
        assert "intimacy_score" in result
        assert "relationship_stage" in result

    async def test_relationship_stage_transitions(self, engine):
        assert engine._infer_stage(0.3) == "acquaintance"
        assert engine._infer_stage(0.4) == "friend"
        assert engine._infer_stage(0.6) == "close"
        assert engine._infer_stage(0.8) == "soulmate"


# ===== B-T14: FTS5 检索 =====

class TestFTS5Retrieval:

    async def test_fts5_search_finds_relevant(self, engine):
        await engine.store_episodic(EpisodicMemory(
            user_id="u1", summary="用户说今天心情很好，想去海边散步",
            tags="开心,海边",
        ))
        await engine.store_episodic(EpisodicMemory(
            user_id="u1", summary="用户抱怨工作压力太大",
            tags="压力,工作",
        ))

        results = await engine.retrieve("u1", "海边 散步", top_k=5)
        assert len(results) >= 1
        assert "海边" in results[0].summary

    async def test_fts5_search_empty_query(self, engine):
        await engine.store_episodic(EpisodicMemory(user_id="u1", summary="测试"))
        results = await engine.retrieve("u1", "", top_k=5)
        assert isinstance(results, list)

    async def test_fts5_search_no_results(self, engine):
        await engine.store_episodic(EpisodicMemory(user_id="u1", summary="abc"))
        results = await engine.retrieve("u1", "nonexistentkeywordxxx", top_k=5)
        assert len(results) == 0

    async def test_fts5_user_isolation(self, engine):
        await engine.store_episodic(EpisodicMemory(user_id="u1", summary="u1的秘密"))
        await engine.store_episodic(EpisodicMemory(user_id="u2", summary="u2的秘密"))

        results = await engine.retrieve("u1", "秘密", top_k=5)
        assert len(results) == 1
        assert results[0].user_id == "u1"

    async def test_batch_fetch_preserves_order(self, engine):
        """_batch_fetch_by_rowids 应按传入行号顺序返回（F-004 修复验证）"""
        ids = []
        for i in range(3):
            mid = await engine.store_episodic(EpisodicMemory(
                user_id="u1", summary=f"test {i}", timestamp=float(i),
            ))
            rowid = await engine._get_rowid(mid)
            ids.append(rowid)

        # 逆序传入，验证返回按传入顺序
        results = await engine._batch_fetch_by_rowids(list(reversed(ids)), "u1", 5)
        assert results[0].summary == "test 2"  # rowid 最大 = timestamp 2
        assert results[1].summary == "test 1"
        assert results[2].summary == "test 0"


# ===== B-T15: 工作记忆（F-003 修复适配） =====

class TestWorkingMemory:

    async def test_add_to_working_memory(self, engine):
        turn = ConversationTurn(session_id="s1", user_id="u1", role="user",
                                content="你好", timestamp=1000.0)
        engine.add_to_working_memory(turn)
        assert len(engine._working_memory.get("s1", [])) == 1

    async def test_snapshot_and_restore(self, engine):
        """snapshot_working_memory 接受 (session_id, turns) 签名（F-003 修复）"""
        turns = [
            ConversationTurn(session_id="s1", user_id="u1", role=r,
                             content=f"消息{i}", timestamp=float(i))
            for i, r in enumerate(["user", "assistant", "user"])
        ]

        # 快照
        saved = await engine.snapshot_working_memory("s1", turns)
        assert saved == 3

        # 恢复
        restored = await engine.restore_working_memory("s1", limit=10)
        assert len(restored) == 3
        assert restored[0].content == "消息0"

    async def test_snapshot_idempotent(self, engine):
        """重复快照不应创建重复记录"""
        turn = ConversationTurn(session_id="s1", user_id="u1", role="user",
                                content="你好", timestamp=100.0)
        await engine.snapshot_working_memory("s1", [turn])
        saved = await engine.snapshot_working_memory("s1", [turn])
        assert saved == 0

    async def test_restore_empty(self, engine):
        restored = await engine.restore_working_memory("nonexistent")
        assert len(restored) == 0

    async def test_snapshot_without_turns(self, engine):
        saved = await engine.snapshot_working_memory("s1", [])
        assert saved == 0


# ===== B-T16: 遗忘策略 =====

class TestForgetting:

    async def test_apply_forgetting_reduces_confidence(self, engine):
        await engine.update_fact("u1", "preference", "a", "1", confidence=0.5)
        await engine.update_fact("u1", "preference", "b", "2", confidence=0.3)
        await engine.update_fact("u1", "preference", "c", "3", confidence=0.05)

        deleted = await engine.apply_forgetting(decay_factor=0.1, threshold=0.1)
        # c: 0.05 - 0.1 = 0.0 → < 0.1 → deleted
        assert deleted == 1

        facts = await engine.list_facts("u1")
        assert len(facts) == 2

    async def test_apply_forgetting_noop_on_high_confidence(self, engine):
        await engine.update_fact("u1", "preference", "a", "1", confidence=1.0)
        deleted = await engine.apply_forgetting(decay_factor=0.1, threshold=0.1)
        assert deleted == 0
        facts = await engine.list_facts("u1")
        assert len(facts) == 1
        assert facts[0].confidence == 0.9

    async def test_apply_forgetting_removes_below_threshold(self, engine):
        await engine.update_fact("u1", "preference", "a", "1", confidence=0.05)
        deleted = await engine.apply_forgetting(decay_factor=0.0, threshold=0.1)
        assert deleted == 1

    async def test_apply_forgetting_empty(self, engine):
        deleted = await engine.apply_forgetting()
        assert deleted == 0


# ===== 统计信息 =====

class TestStats:

    async def test_get_stats_empty_user(self, engine):
        stats = await engine.get_stats("u1")
        assert stats["user_id"] == "u1"
        assert stats["episodic_count"] == 0
        assert stats["fact_count"] == 0
        assert stats["semantic"] is None

    async def test_get_stats_with_data(self, engine):
        await engine.store_episodic(EpisodicMemory(user_id="u1", summary="test"))
        await engine.update_fact("u1", "preference", "color", "blue")
        await engine.update_semantic("u1", "trust_score", 0.1)

        stats = await engine.get_stats("u1")
        assert stats["episodic_count"] == 1
        assert stats["fact_count"] == 1
        assert stats["semantic"]["trust_score"] == 0.6
        assert stats["semantic"]["relationship_stage"] == "acquaintance"
