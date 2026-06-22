"""
表情包插件系统

B-T37: EmotePlugin 抽象接口 + LocalScanner
B-T38: 情感状态标签加权随机选择
B-T39: TTL 缓存 + watchdog 驱动更新
"""

from __future__ import annotations

import logging
import os
import random
import time as _time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from mirror_core.emotion.engine import (
    EMOTION_TAG_MAP,
    EmotionalState,
    _meets_condition,
)

logger = logging.getLogger("mirror_core.execution.emotes")


@dataclass
class EmoteResult:
    """表情包结果"""
    path: str
    mime_type: str = "image/png"


class EmotePlugin(ABC):
    """表情包插件抽象接口"""

    @abstractmethod
    def get_random_emote(self, tag: str) -> Optional[EmoteResult]:
        """返回指定标签的随机表情包。"""
        ...

    @abstractmethod
    def list_tags(self) -> List[str]:
        """返回所有可用标签。"""
        ...


# ===== 支持的文件扩展名 =====
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

# TTL 缓存时长（秒）
SCAN_CACHE_TTL = 60

# MIME 类型映射
MIME_TYPE_MAP = {
    ".gif": "image/gif",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class LocalScanner(EmotePlugin):
    """
    本地文件夹扫描器。

    扫描指定根目录下按标签分文件夹的图片文件。
    支持 TTL 缓存和 watchdog 文件变更驱动刷新。
    """

    def __init__(self, root: str):
        """
        Args:
            root: 表情包根目录，每个子文件夹为一个标签
        """
        self._root = root
        self._cache: Dict[str, List[str]] = {}
        self._last_scan_time: float = 0.0
        self._watchdog = None

    @property
    def root(self) -> str:
        return self._root

    # ---- B-T37: 扫描逻辑 ----

    def _scan(self) -> None:
        """扫描文件系统，刷新缓存。"""
        self._cache.clear()
        if not os.path.isdir(self._root):
            logger.warning("表情包根目录不存在: %s", self._root)
            self._last_scan_time = _time.time()
            return

        for entry in os.scandir(self._root):
            if not entry.is_dir():
                continue
            tag = entry.name
            files = []
            for fname in os.listdir(entry.path):
                ext = os.path.splitext(fname)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    files.append(fname)
            if files:
                self._cache[tag] = files

        self._last_scan_time = _time.time()
        logger.debug("表情包扫描完成: %d 个标签", len(self._cache))

    def _ensure_scanned(self) -> None:
        """检查缓存是否过期，过期则重新扫描。"""
        if _time.time() - self._last_scan_time > SCAN_CACHE_TTL:
            self._scan()

    # ---- B-T37: 接口实现 ----

    def get_random_emote(self, tag: str) -> Optional[EmoteResult]:
        """
        从指定标签文件夹中随机返回一张图片。

        Args:
            tag: 标签名（文件夹名）

        Returns:
            随机表情包，标签不存在或文件夹为空时返回 None
        """
        self._ensure_scanned()
        files = self._cache.get(tag)
        if not files:
            return None

        chosen = random.choice(files)
        ext = os.path.splitext(chosen)[1].lower()

        mime_type = MIME_TYPE_MAP.get(ext, "application/octet-stream")

        return EmoteResult(
            path=os.path.join(self._root, tag, chosen),
            mime_type=mime_type,
        )

    def list_tags(self) -> List[str]:
        """返回所有已扫描到的标签列表。"""
        self._ensure_scanned()
        return list(self._cache.keys())

    # ---- B-T38: 情感状态标签选择 ----

    def select_tag_from_emotion(
        self,
        emotion: Optional[EmotionalState] = None,
    ) -> Optional[str]:
        """
        根据情感状态加权随机选择一个标签。

        使用 EMOTION_TAG_MAP（定义于 emotion.engine）进行匹配，
        每个标签按权重重复放入候选池，然后随机抽取。

        Args:
            emotion: 当前情感状态，None 时从所有标签中随机选择

        Returns:
            选中的标签名，无可用标签时返回 None
        """
        self._ensure_scanned()
        if not self._cache:
            return None

        if emotion is None:
            # 无情感状态时从所有标签中随机选
            return random.choice(list(self._cache.keys()))

        candidates: List[str] = []
        for condition, tags in EMOTION_TAG_MAP:
            if _meets_condition(condition, emotion):
                for tag, weight in tags:
                    candidates.extend([tag] * weight)

        if not candidates:
            return random.choice(list(self._cache.keys()))

        # 去重后只保留实际存在的标签
        available = set(self._cache.keys())
        selected_tag = random.choice(candidates)
        if selected_tag not in available:
            # 回退到随机
            return random.choice(list(self._cache.keys()))

        return selected_tag

    # ---- B-T39: TTL 缓存 + watchdog ----

    def force_refresh(self) -> None:
        """强制刷新扫描缓存。"""
        self._scan()

    def start_watchdog(self) -> None:
        """
        启动文件系统监听，文件变更时自动刷新缓存（B-T39）。

        watchdog 监听表情包根目录的文件变更事件，
        变更时调用 force_refresh() 使缓存下次访问时重新扫描。
        """
        if not os.path.isdir(self._root):
            logger.warning("表情包根目录不存在，watchdog 不启动: %s", self._root)
            return
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class _EmoteHandler(FileSystemEventHandler):
                def __init__(self, scanner: LocalScanner):
                    self.scanner = scanner

                def on_modified(self, event):
                    self.scanner.force_refresh()

                def on_created(self, event):
                    self.scanner.force_refresh()

                def on_deleted(self, event):
                    self.scanner.force_refresh()

            self._watchdog = Observer()
            self._watchdog.schedule(
                _EmoteHandler(self),
                self._root,
                recursive=True,
            )
            self._watchdog.start()
            logger.debug("表情包 watchdog 已启动: %s", self._root)
        except Exception as exc:
            logger.warning("表情包 watchdog 启动失败: %s", exc)

    async def stop_watchdog(self) -> None:
        """停止文件监听。"""
        if self._watchdog:
            self._watchdog.stop()
            self._watchdog.join()
            self._watchdog = None
            logger.debug("表情包 watchdog 已停止")

    # ---- G-001: 防止 watchdog 僵尸线程 ----

    def __del__(self) -> None:
        """析构时确保 watchdog 线程被停止。"""
        if self._watchdog:
            try:
                self._watchdog.stop()
                self._watchdog.join(timeout=2)
            except Exception:
                pass
            self._watchdog = None
