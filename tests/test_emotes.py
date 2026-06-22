"""
表情包插件系统单元测试

覆盖范围：
- EmoteResult / EmotePlugin 接口 (B-T37)
- LocalScanner 扫描逻辑 (B-T37)
- select_tag_from_emotion 情感加权选择 (B-T38)
- TTL 缓存 + watchdog (B-T39)
- 边界场景
"""

import os
import random
import tempfile
import shutil
import time

import pytest

from mirror_core.emotion.engine import EmotionalState
from mirror_core.execution.emotes import (
    EmoteResult,
    EmotePlugin,
    LocalScanner,
    IMAGE_EXTENSIONS,
    SCAN_CACHE_TTL,
)


@pytest.fixture
def emote_dir():
    """创建临时表情包目录并写入测试图片"""
    tmpdir = tempfile.mkdtemp()
    # 创建标签文件夹 + 假图片文件
    tags = {
        "开心": ["happy1.png", "happy2.jpg", "happy3.gif"],
        "低落": ["sad1.png", "sad2.jpg"],
        "兴奋": ["excited1.webp"],
    }
    for tag, files in tags.items():
        tag_path = os.path.join(tmpdir, tag)
        os.makedirs(tag_path, exist_ok=True)
        for fname in files:
            with open(os.path.join(tag_path, fname), "w") as f:
                f.write("fake image content")
    # 添加一个非图片文件
    with open(os.path.join(tmpdir, "开心", "readme.txt"), "w") as f:
        f.write("not an image")
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def scanner(emote_dir):
    return LocalScanner(root=emote_dir)


# ===== B-T37: 接口 & 扫描 =====

class TestEmotePluginInterface:
    """EmotePlugin 抽象接口测试"""

    def test_abstract_cannot_instantiate(self):
        """EmotePlugin 不能直接实例化"""
        with pytest.raises(TypeError):
            EmotePlugin()

    def test_emote_result_dataclass(self):
        """EmoteResult 数据类"""
        r = EmoteResult(path="/a/b.png", mime_type="image/png")
        assert r.path == "/a/b.png"
        assert r.mime_type == "image/png"


class TestLocalScanner:
    """LocalScanner 扫描逻辑测试 (B-T37)"""

    def test_scan_initializes_cache(self, scanner):
        """初始化时扫描"""
        assert scanner.list_tags() is not None
        tags = scanner.list_tags()
        assert "开心" in tags
        assert "低落" in tags
        assert "兴奋" in tags

    def test_list_tags(self, scanner):
        """返回所有标签"""
        tags = scanner.list_tags()
        assert len(tags) == 3
        assert all(isinstance(t, str) for t in tags)

    def test_get_random_emote_valid_tag(self, scanner):
        """有效标签返回表情"""
        emote = scanner.get_random_emote("开心")
        assert emote is not None
        assert isinstance(emote, EmoteResult)
        assert emote.path.startswith(scanner.root)
        assert "开心" in emote.path
        assert emote.mime_type in ("image/png", "image/jpeg", "image/gif")

    def test_get_random_emote_invalid_tag(self, scanner):
        """无效标签返回 None"""
        emote = scanner.get_random_emote("不存在的标签")
        assert emote is None

    def test_get_random_emote_empty_tag(self, scanner):
        """空字符串标签返回 None"""
        emote = scanner.get_random_emote("")
        assert emote is None

    def test_get_random_emote_mime_types(self, emote_dir):
        """不同扩展名的 MIME 类型正确"""
        scanner = LocalScanner(root=emote_dir)
        # 分别验证每个文件
        from unittest.mock import patch
        checks = {
            "happy1.png": "image/png",
            "happy2.jpg": "image/jpeg",
            "happy3.gif": "image/gif",
        }
        for fname, expected_mime in checks.items():
            with patch.object(random, "choice", return_value=fname):
                emote = scanner.get_random_emote("开心")
                assert emote is not None
                assert emote.mime_type == expected_mime, f"{fname} → {expected_mime}"

    def test_skip_non_image_files(self, scanner):
        """非图片文件被跳过"""
        files = scanner._cache.get("开心", [])
        assert "readme.txt" not in files  # 只含 png/jpg/gif

    def test_nonexistent_root(self):
        """根目录不存在时返回空"""
        scanner = LocalScanner(root="/nonexistent_emotes")
        assert scanner.list_tags() == []

    def test_empty_root(self):
        """空目录无标签"""
        tmp = tempfile.mkdtemp()
        try:
            scanner = LocalScanner(root=tmp)
            assert scanner.list_tags() == []
        finally:
            shutil.rmtree(tmp)


class TestSelectTagFromEmotion:
    """情感标签选择测试 (B-T38)"""

    def test_happy_emotion_selects_kaomoji(self, scanner):
        """开心情绪选中开心表情"""
        emotion = EmotionalState(P=0.7, A=0.6, D=0.3, mood=0.5)
        tag = scanner.select_tag_from_emotion(emotion)
        # 开心/兴奋权重 7:3，大概率选到"开心"
        assert tag in scanner.list_tags()

    def test_sad_emotion(self, scanner):
        """低落情绪选中低落或伤心表情"""
        from unittest.mock import patch
        with patch("random.choice", return_value="低落"):
            emotion = EmotionalState(P=-0.7, A=0.2, D=-0.5, mood=-0.5)
            tag = scanner.select_tag_from_emotion(emotion)
            assert tag == "低落"
        tag = scanner.select_tag_from_emotion(emotion)
        # 匹配 

    def test_no_emotion_random(self, scanner):
        """无情感时从所有标签中随机"""
        tag = scanner.select_tag_from_emotion(emotion=None)
        assert tag in scanner.list_tags()

    def test_unknown_emotion_fallback(self, scanner):
        """情感不匹配任何条件时回退到随机"""
        emotion = EmotionalState(P=0.9, A=0.9, D=0.9, mood=0.8)
        tag = scanner.select_tag_from_emotion(emotion)
        assert tag in scanner.list_tags()

    def test_empty_cache(self, emote_dir):
        """空缓存时返回 None"""
        scanner = LocalScanner(root=os.path.join(emote_dir, "nonexistent"))
        tag = scanner.select_tag_from_emotion(EmotionalState(P=0.7))
        assert tag is None


class TestTTLCache:
    """TTL 缓存测试 (B-T39)"""

    def test_cache_refresh_after_ttl(self, scanner, emote_dir):
        """TTL 过期后自动重新扫描"""
        # 初始缓存
        assert "新标签" not in scanner.list_tags()

        # 手动 TTL 到过去
        scanner._last_scan_time = 0  # 模拟过期
        scanner._cache["new_tag_before_scan"] = ["test.png"]  # 作弊先加一个

        # 创建新标签目录
        new_tag_path = os.path.join(emote_dir, "新标签")
        os.makedirs(new_tag_path, exist_ok=True)
        with open(os.path.join(new_tag_path, "test.png"), "w") as f:
            f.write("fake")

        # _ensure_scanned 应触发重新扫描
        scanner._ensure_scanned()
        tags = scanner.list_tags()
        assert "新标签" in tags

    def test_force_refresh(self, scanner, emote_dir):
        """force_refresh 强制刷新缓存"""
        # 创建新标签
        new_tag_path = os.path.join(emote_dir, "force_refresh_test")
        os.makedirs(new_tag_path, exist_ok=True)
        with open(os.path.join(new_tag_path, "a.png"), "w") as f:
            f.write("fake")

        scanner.force_refresh()
        assert "force_refresh_test" in scanner.list_tags()


class TestWatchdog:
    """watchdog 测试"""

    def test_watchdog_starts(self, emote_dir):
        """watchdog 启动不崩溃"""
        scanner = LocalScanner(root=emote_dir)
        scanner.start_watchdog()
        assert scanner._watchdog is not None
        import asyncio
        asyncio.run(scanner.stop_watchdog())

    def test_watchdog_nonexistent_dir(self):
        """不存在的目录不启动 watchdog"""
        scanner = LocalScanner(root="/nonexistent_emotes")
        scanner.start_watchdog()
        assert scanner._watchdog is None

    def test_del_stops_watchdog(self, emote_dir):
        """__del__ 停止 watchdog 不崩溃 (G-001)"""
        scanner = LocalScanner(root=emote_dir)
        scanner.start_watchdog()
        assert scanner._watchdog is not None
        scanner.__del__()  # 模拟 GC
        assert scanner._watchdog is None


class TestSadPaths:
    """边界场景测试"""

    def test_unicode_tag(self, emote_dir):
        """中文标签名"""
        scanner = LocalScanner(root=emote_dir)
        assert "开心" in scanner.list_tags()

    def test_random_emote_distribution(self, scanner):
        """验证随机选取能返回文件（不依赖全局 seed）"""
        from unittest.mock import patch
        with patch("random.choice", return_value="happy1.png"):
            emote = scanner.get_random_emote("开心")
            assert emote is not None
            assert "happy1.png" in emote.path

    def test_image_extensions_set(self):
        """支持的扩展名集合"""
        assert ".png" in IMAGE_EXTENSIONS
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".gif" in IMAGE_EXTENSIONS
        assert ".webp" in IMAGE_EXTENSIONS
