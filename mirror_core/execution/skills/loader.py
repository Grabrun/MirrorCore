"""
Skills - SKILL.md 加载与解析

B-T34: 解析 YAML frontmatter 与 Markdown body

SKILL.md 格式:
    ---
    description: "情绪安抚技能"
    author: "community"
    version: "1.0"
    triggers: ["焦虑", "难过", "害怕"]
    emotion_match: { "P": "< -0.4" }
    ---

    当用户表现出焦虑或悲伤时，用温和的语气回应...
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger("mirror_core.execution.skills.loader")

# YAML frontmatter 分隔符
_FM_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass
class SkillMeta:
    """技能元数据"""
    name: str = ""
    description: str = ""
    author: str = ""
    version: str = ""
    triggers: List[str] = field(default_factory=list)
    emotion_match: Dict[str, str] = field(default_factory=dict)
    body: str = ""


def parse_skill_file(filepath: str) -> Optional[SkillMeta]:
    """
    解析单条 SKILL.md 文件。

    Args:
        filepath: SKILL.md 的完整路径

    Returns:
        解析成功返回 SkillMeta，失败返回 None
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except (OSError, IOError) as exc:
        logger.warning("读取技能文件失败: %s — %s", filepath, exc)
        return None

    match = _FM_PATTERN.match(content)
    if not match:
        logger.warning("技能文件缺少 YAML frontmatter: %s", filepath)
        return None

    try:
        fm = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        logger.warning("技能文件 frontmatter YAML 解析失败: %s — %s", filepath, exc)
        return None

    if not isinstance(fm, dict):
        logger.warning("技能文件 frontmatter 格式错误: %s", filepath)
        return None

    body = content[match.end():].strip()
    name = os.path.splitext(os.path.basename(os.path.dirname(filepath)))[0] or \
           os.path.splitext(os.path.basename(filepath))[0]

    return SkillMeta(
        name=fm.get("name", name),
        description=fm.get("description", ""),
        author=fm.get("author", ""),
        version=fm.get("version", ""),
        triggers=fm.get("triggers", []),
        emotion_match=fm.get("emotion_match", {}),
        body=body,
    )
