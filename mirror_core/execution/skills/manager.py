"""
SkillManager — 技能管理器

B-T34: SkillManager 加载/重载/查询技能
B-T35: watchdog 文件变更监听与热加载
B-T36: 关键词+情绪双维度匹配评分

设计文档 §3.4.3:

匹配逻辑:
    score = 0
    if any(trigger in user_text for trigger in skill.triggers):
        score += 1
    if emotion_matches(skill.emotion_match, current_emotion):
        score += 1
    return skill if score >= 2
"""

from __future__ import annotations

import logging
import os
import time as _time
from typing import Dict, List, Optional

from mirror_core.emotion.engine import EmotionalState, _meets_condition
from mirror_core.execution.skills.loader import SkillMeta, parse_skill_file

logger = logging.getLogger("mirror_core.execution.skills.manager")


class SkillManager:
    """
    技能管理器。

    管理 skills_root 下所有 SKILL.md 的加载、缓存、匹配和热加载。
    """

    def __init__(self, skills_root: str, enable_watchdog: bool = True):
        """
        Args:
            skills_root: 技能文件根目录
            enable_watchdog: 是否启用文件监听（默认 True，B-T35）
        """
        self._skills_root = os.path.abspath(skills_root)
        self._skills: Dict[str, SkillMeta] = {}
        self._enable_watchdog = enable_watchdog
        self._watchdog = None

    # ---- B-T34: 加载 ----

    async def load_all(self) -> int:
        """
        扫描 skills_root 下所有 SKILL.md 并加载。

        目录结构约定:
            skills_root/
                calming/
                    SKILL.md
                greeting/
                    SKILL.md
                ...

        Returns:
            成功加载的技能数量
        """
        self._skills.clear()
        count = 0

        if not os.path.isdir(self._skills_root):
            logger.warning("技能根目录不存在: %s", self._skills_root)
            self._start_watchdog()  # 目录创建后自动加载
            return 0

        for entry in os.scandir(self._skills_root):
            if not entry.is_dir():
                continue
            skill_path = os.path.join(entry.path, "SKILL.md")
            if not os.path.isfile(skill_path):
                continue

            meta = parse_skill_file(skill_path)
            if meta:
                self._skills[entry.name] = meta
                count += 1

        logger.info("技能加载完成: %d 个", count)
        self._start_watchdog()
        return count

    async def reload(self) -> int:
        """
        重新加载所有技能（热加载用）。

        Returns:
            加载后的技能数量
        """
        return await self.load_all()

    # ---- B-T36: 匹配 ----

    def match(
        self,
        user_text: str,
        emotion: Optional[EmotionalState] = None,
    ) -> List[SkillMeta]:
        """
        根据用户输入和情绪状态筛选候选技能。

        评分规则:
            关键词匹配 +1 分
            情绪状态匹配 +1 分
            score >= 2 入选候选列表

        Args:
            user_text: 用户输入文本
            emotion: 当前情感状态（可选）

        Returns:
            按评分降序排列的候选技能列表
        """
        if not self._skills:
            return []

        scored: List[tuple] = []

        for skill_name, meta in self._skills.items():
            score = 0

            # 关键词匹配
            if meta.triggers:
                if self._match_keywords(user_text, meta.triggers):
                    score += 1

            # 情绪状态匹配
            if emotion and meta.emotion_match:
                if self._match_emotion(emotion, meta.emotion_match):
                    score += 1

            if score >= 2:
                scored.append((score, skill_name, meta))

        # 按分数降序排列
        scored.sort(key=lambda x: x[0], reverse=True)
        return [meta for _, _, meta in scored]

    @staticmethod
    def _match_keywords(user_text: str, triggers: List[str]) -> bool:
        """检查用户输入是否包含任意触发关键词。"""
        text_lower = user_text.lower()
        for trigger in triggers:
            if trigger.lower() in text_lower:
                return True
        return False

    @staticmethod
    def _match_emotion(
        emotion: EmotionalState,
        emotion_match: dict,
    ) -> bool:
        """
        检查当前情绪是否匹配技能的情感条件。

        使用 emotion.engine._meets_condition 进行评估。
        emotion_match 格式: {"P": "< -0.4", "A": "> 0.5"}
        """
        condition_parts = []
        for field, expr in emotion_match.items():
            condition_parts.append(f"{field}{expr}")
        condition_str = ",".join(condition_parts)
        return _meets_condition(condition_str, emotion)

    # ---- B-T35: watchdog 文件监听 ----

    def _start_watchdog(self) -> None:
        """启动文件系统监听（可选）。"""
        if not self._enable_watchdog or self._watchdog is not None:
            return
        if not os.path.isdir(self._skills_root):
            return

        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class _SkillHandler(FileSystemEventHandler):
                def __init__(self, manager: SkillManager):
                    self.manager = manager

                def on_modified(self, event):
                    if event.src_path.endswith("SKILL.md"):
                        logger.info("技能文件变更: %s", event.src_path)
                        import asyncio
                        asyncio.ensure_future(self.manager.reload())

                def on_created(self, event):
                    if event.src_path.endswith("SKILL.md"):
                        logger.info("技能文件新增: %s", event.src_path)
                        import asyncio
                        asyncio.ensure_future(self.manager.reload())

                def on_deleted(self, event):
                    if event.src_path.endswith("SKILL.md"):
                        logger.info("技能文件删除: %s", event.src_path)
                        import asyncio
                        asyncio.ensure_future(self.manager.reload())

            self._watchdog = Observer()
            self._watchdog.schedule(
                _SkillHandler(self),
                self._skills_root,
                recursive=True,
            )
            self._watchdog.start()
            logger.debug("技能文件监听已启动: %s", self._skills_root)
        except Exception as exc:
            logger.warning("技能文件监听启动失败 (watchdog 不可用?): %s", exc)

    async def stop_watchdog(self) -> None:
        """停止文件监听。"""
        if self._watchdog:
            self._watchdog.stop()
            self._watchdog.join()
            self._watchdog = None
            logger.debug("技能文件监听已停止")

    # ---- 便捷查询 ----

    def get_prompt(self, skill_name: str) -> str:
        """
        获取指定技能的提示词正文。

        Args:
            skill_name: 技能名称（目录名）

        Returns:
            技能 Markdown 正文，不存在时返回空字符串
        """
        meta = self._skills.get(skill_name)
        return meta.body if meta else ""

    def list_skills(self) -> List[str]:
        """返回所有已加载的技能名称列表。"""
        return list(self._skills.keys())

    @property
    def skills_root(self) -> str:
        return self._skills_root

    @property
    def count(self) -> int:
        return len(self._skills)
