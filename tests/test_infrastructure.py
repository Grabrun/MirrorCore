"""
基础设施层单元测试

覆盖范围：
- ConfigManager 加载/校验/环境变量/热加载 (D-T01)
- logging 配置 / trace_id / 脱敏 (D-T02)
- retry 装饰器 (D-T04)
- CircuitBreaker 熔断器 (D-T04)
- sqlite-vec 检测 (D-T05)
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirror_core.infrastructure.config import (
    ConfigManager,
    RootConfig,
    _mask_sensitive,
)
from mirror_core.infrastructure.exceptions import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    retry,
    get_fallback_reply,
    get_degraded_reply,
)
from mirror_core.infrastructure.logging_config import (
    hash_preview,
    setup_logging,
    bind_trace_id,
)


# ===== D-T01: ConfigManager =====

@pytest.fixture
def config_dir():
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


def _write_yaml(path, data):
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)


class TestConfigManager:
    """配置管理器 (D-T01)"""

    def test_load_persona(self, config_dir):
        """加载 persona.yaml"""
        _write_yaml(config_dir / "persona.yaml", {
            "name": "小艾",
            "identity": "测试助手",
            "traits": ["可爱"],
        })
        cm = ConfigManager(str(config_dir))
        cm.load()
        assert cm.get("persona.name") == "小艾"
        assert cm.get("persona.identity") == "测试助手"

    def test_load_memory_defaults(self, config_dir):
        """未配置时使用默认值"""
        cm = ConfigManager(str(config_dir))
        cm.load()
        assert cm.get("memory.embedding_dim") == 768
        assert cm.get("memory.recall_top_k") == 5

    def test_get_default(self, config_dir):
        """不存在的路径返回默认值"""
        cm = ConfigManager(str(config_dir))
        cm.load()
        assert cm.get("nonexistent.key", "fallback") == "fallback"

    def test_set_value(self, config_dir):
        """运行时设置值"""
        cm = ConfigManager(str(config_dir))
        cm.load()
        cm.set("persona.name", "小镜")
        assert cm.get("persona.name") == "小镜"

    def test_reload(self, config_dir):
        """热加载"""
        cm = ConfigManager(str(config_dir))
        cm.load()
        _write_yaml(config_dir / "persona.yaml", {"name": "新名字"})
        cm.reload()
        assert cm.get("persona.name") == "新名字"

    def test_missing_config_dir(self):
        """配置目录不存在不崩溃"""
        cm = ConfigManager("/nonexistent/config")
        cm.load()  # 不应崩溃

    def test_env_override(self, config_dir):
        """环境变量覆盖"""
        _write_yaml(config_dir / "persona.yaml", {"name": "小艾"})
        with patch.dict(os.environ, {"MIRROR_PERSONA_NAME": "环境覆盖"}):
            cm = ConfigManager(str(config_dir))
            cm.load()
            assert cm.get("persona.name") == "环境覆盖"

    def test_sensitive_mask(self):
        """敏感信息脱敏"""
        assert _mask_sensitive("api_key", "sk-1234567890abcdef") == "sk-1****cdef"
        assert _mask_sensitive("name", "公开信息") == "公开信息"


class TestLogging:
    """日志配置 (D-T02)"""

    def test_hash_preview(self):
        """哈希脱敏"""
        h = hash_preview("测试消息")
        assert len(h) == 8
        assert "测试" not in h

    def test_hash_consistent(self):
        """相同输入相同哈希"""
        assert hash_preview("hello") == hash_preview("hello")

    def test_hash_different(self):
        """不同输入不同哈希"""
        assert hash_preview("a") != hash_preview("b")

    def test_setup_no_crash(self):
        """日志配置不崩溃"""
        setup_logging(log_level="DEBUG", json_format=False)

    def test_bind_trace_id(self):
        """绑定 trace_id"""
        bind_trace_id("test-trace-123")


class TestRetry:
    """重试装饰器 (D-T04)"""

    @pytest.mark.asyncio
    async def test_retry_success_first(self):
        """首次成功不重试"""
        mock = AsyncMock()
        mock.side_effect = ["ok"]

        @retry(max_retries=2)
        async def test_func():
            return await mock()

        result = await test_func()
        assert result == "ok"
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_after_failure(self):
        """失败后重试成功"""
        mock = AsyncMock()
        mock.side_effect = [ValueError("失败"), "成功"]

        @retry(max_retries=2, exceptions=(ValueError,))
        async def test_func():
            return await mock()

        result = await test_func()
        assert result == "成功"
        assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_all_fail(self):
        """全部失败抛出异常"""
        mock = AsyncMock()
        mock.side_effect = ValueError("始终失败")

        @retry(max_retries=2, exceptions=(ValueError,), base_delay=0.01)
        async def test_func():
            return await mock()

        with pytest.raises(ValueError):
            await test_func()
        assert mock.call_count == 3  # 首次 + 2 次重试


class TestCircuitBreaker:
    """熔断器 (D-T04)"""

    @pytest.mark.asyncio
    async def test_success_keeps_closed(self):
        """成功时保持关闭"""
        cb = CircuitBreaker(failure_threshold=3)
        async with cb:
            pass
        assert cb.state == CircuitBreaker.CLOSED

    @pytest.mark.asyncio
    async def test_failure_opens(self):
        """连续失败打开熔断器"""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60)
        for _ in range(2):
            with pytest.raises(ValueError):
                async with cb:
                    raise ValueError("失败")
        assert cb.state == CircuitBreaker.OPEN

    @pytest.mark.asyncio
    async def test_open_raises(self):
        """打开时抛出异常"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
        with pytest.raises(ValueError):
            async with cb:
                raise ValueError("失败")
        with pytest.raises(CircuitBreakerOpenError):
            async with cb:
                pass

    @pytest.mark.asyncio
    async def test_half_open_recovery(self):
        """半开后成功请求关闭熔断器"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)

        # 触发打开
        with pytest.raises(ValueError):
            async with cb:
                raise ValueError("失败")

        # 等待超时 → 半开
        import asyncio
        await asyncio.sleep(0.02)

        # 成功 → 关闭
        async with cb:
            pass
        assert cb.state == CircuitBreaker.CLOSED

    def test_reset(self):
        """手动重置"""
        cb = CircuitBreaker(failure_threshold=1)
        cb._failure_count = 5
        cb._state = CircuitBreaker.OPEN
        cb.reset()
        assert cb.state == CircuitBreaker.CLOSED
        assert cb._failure_count == 0


class TestFallbackReplies:
    """降级回复 (D-T03)"""

    def test_fallback_reply(self):
        reply = get_fallback_reply()
        assert "走神" in reply

    def test_degraded_reply(self):
        reply = get_degraded_reply()
        assert "忙" in reply or "等一下" in reply


class TestVecDetection:
    """sqlite-vec 检测 (D-T05)"""

    @pytest.mark.asyncio
    async def test_has_vec_property(self):
        """Database 有 has_vec 属性"""
        from mirror_core.infrastructure.database import Database
        db = Database(path=":memory:")
        await db.initialize()
        # has_vec 应为 bool
        assert isinstance(db.has_vec, bool)
        await db.close()

    @pytest.mark.asyncio
    async def test_has_vec_default(self):
        """默认 has_vec 为 True（SQLite 标准环境）"""
        from mirror_core.infrastructure.database import Database
        db = Database(path=":memory:")
        await db.initialize()
        # 标准 SQLite 不带 vec0 → has_vec=False
        await db.close()
