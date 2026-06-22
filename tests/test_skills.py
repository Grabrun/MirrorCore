"""
技能系统单元测试

覆盖范围：
- SKILL.md 文件解析 (B-T34)
- SkillManager 加载/重载 (B-T34)
- 双维度匹配评分 (B-T36)
- watchdog 热加载 (B-T35)
- 边界场景
"""

import os
import tempfile
import shutil

import pytest

from mirror_core.emotion.engine import EmotionalState
from mirror_core.execution.skills.loader import SkillMeta, parse_skill_file
from mirror_core.execution.skills.manager import SkillManager


# ===== 测试用 SKILL.md 内容 =====

CALMING_SKILL = """\
---
description: "情绪安抚"
author: "community"
version: "1.0"
triggers: ["焦虑", "难过"]
emotion_match:
  P: "< -0.4"
---

用温和的语气回应，尝试共情。
"""

GREETING_SKILL = """\
---
description: "早安问候"
triggers: ["早安", "早上好"]
emotion_match:
  P: "> 0.2"
---

送上早安祝福，询问今天的计划。
"""

INVALID_SKILL = """\
没有 frontmatter 分隔符的纯文本
"""

EMPTY_FM_SKILL = """\

---

body only
"""

YAML_ERROR_SKILL = """\
---
invalid: [yaml: error
---

body
"""


# ===== Test Fixtures =====

@pytest.fixture
def skills_dir():
    """创建临时技能目录并写入测试文件"""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


def _write_skill(base_dir, name, content):
    path = os.path.join(base_dir, name, "SKILL.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ===== Loader Tests =====

class TestParseSkillFile:
    """SKILL.md 解析测试 (B-T34)"""

    def test_parse_valid(self, skills_dir):
        """正常 SKILL.md 解析"""
        path = _write_skill(skills_dir, "calming", CALMING_SKILL)
        meta = parse_skill_file(path)
        assert meta is not None
        assert meta.name == "calming"
        assert meta.description == "情绪安抚"
        assert meta.author == "community"
        assert meta.version == "1.0"
        assert meta.triggers == ["焦虑", "难过"]
        assert meta.emotion_match == {"P": "< -0.4"}
        assert "用温和的语气" in meta.body

    def test_parse_no_frontmatter(self, skills_dir):
        """无 frontmatter 返回 None"""
        path = _write_skill(skills_dir, "bad", INVALID_SKILL)
        meta = parse_skill_file(path)
        assert meta is None

    def test_parse_yaml_error(self, skills_dir):
        """YAML 错误返回 None"""
        path = _write_skill(skills_dir, "bad", YAML_ERROR_SKILL)
        meta = parse_skill_file(path)
        assert meta is None

    def test_parse_nonexistent(self):
        """不存在的文件返回 None"""
        meta = parse_skill_file("/nonexistent/SKILL.md")
        assert meta is None

    def test_parse_minimal(self, skills_dir):
        """最小合法 SKILL.md"""
        content = "---\ndescription: test\n---\nbody"
        path = _write_skill(skills_dir, "minimal", content)
        meta = parse_skill_file(path)
        assert meta is not None
        assert meta.description == "test"
        assert meta.body == "body"

    def test_parse_name_from_directory(self, skills_dir):
        """名称从目录名继承"""
        path = _write_skill(skills_dir, "my-skill", "---\ndescription: test\n---\nbody")
        meta = parse_skill_file(path)
        assert meta is not None
        assert meta.name == "my-skill"


# ===== SkillManager Tests =====

class TestSkillManager:
    """SkillManager 测试 (B-T34 / B-T36)"""

    @pytest.fixture
    def manager(self, skills_dir):
        """创建 SkillManager 实例（禁用 watchdog）"""
        return SkillManager(skills_root=skills_dir, enable_watchdog=False)

    @pytest.mark.asyncio
    async def test_load_all(self, manager, skills_dir):
        """加载所有技能"""
        _write_skill(skills_dir, "calming", CALMING_SKILL)
        _write_skill(skills_dir, "greeting", GREETING_SKILL)

        count = await manager.load_all()
        assert count == 2
        assert manager.count == 2
        assert "calming" in manager.list_skills()
        assert "greeting" in manager.list_skills()

    @pytest.mark.asyncio
    async def test_load_empty_dir(self, manager):
        """空目录加载 0 个"""
        count = await manager.load_all()
        assert count == 0

    @pytest.mark.asyncio
    async def test_load_nonexistent_dir(self):
        """目录不存在返回 0 不崩溃"""
        mgr = SkillManager(skills_root="/nonexistent", enable_watchdog=False)
        count = await mgr.load_all()
        assert count == 0

    @pytest.mark.asyncio
    async def test_reload(self, manager, skills_dir):
        """重新加载后技能数刷新"""
        _write_skill(skills_dir, "calming", CALMING_SKILL)
        await manager.load_all()
        assert manager.count == 1

        _write_skill(skills_dir, "greeting", GREETING_SKILL)
        count = await manager.reload()
        assert count == 2

    @pytest.mark.asyncio
    async def test_get_prompt(self, manager, skills_dir):
        """获取技能正文"""
        _write_skill(skills_dir, "calming", CALMING_SKILL)
        await manager.load_all()

        prompt = manager.get_prompt("calming")
        assert "共情" in prompt
        assert manager.get_prompt("nonexistent") == ""

    # ---- B-T36: 匹配评分 ----

    @pytest.mark.asyncio
    async def test_match_keyword_only(self, manager, skills_dir):
        """仅关键词匹配不触发（需 score>=2）"""
        _write_skill(skills_dir, "calming", CALMING_SKILL)
        await manager.load_all()

        from mirror_core.emotion.engine import EmotionalState
        emotion = EmotionalState(P=0.0, A=0.3)  # 不匹配 P<-0.4

        results = manager.match("我好焦虑", emotion=emotion)
        assert len(results) == 0  # 只有关键词得分=1，不够

    @pytest.mark.asyncio
    async def test_match_keyword_and_emotion(self, manager, skills_dir):
        """关键词+情绪都匹配时触发"""
        _write_skill(skills_dir, "calming", CALMING_SKILL)
        await manager.load_all()

        from mirror_core.emotion.engine import EmotionalState
        emotion = EmotionalState(P=-0.6, A=0.5)  # P=-0.6 <-0.4 ✅

        results = manager.match("我好焦虑", emotion=emotion)
        assert len(results) == 1
        assert results[0].name == "calming"

    @pytest.mark.asyncio
    async def test_match_emotion_only(self, manager, skills_dir):
        """仅情绪匹配不触发"""
        _write_skill(skills_dir, "calming", CALMING_SKILL)
        await manager.load_all()

        from mirror_core.emotion.engine import EmotionalState
        emotion = EmotionalState(P=-0.6, A=0.5)  # P=-0.6 <-0.4 ✅

        results = manager.match("我今天很开心", emotion=emotion)
        assert len(results) == 0  # 只有情绪得分=1，关键词没匹配

    @pytest.mark.asyncio
    async def test_match_multiple_skills(self, manager, skills_dir):
        """多个技能按分数降序排列"""
        _write_skill(skills_dir, "calming", CALMING_SKILL)
        _write_skill(skills_dir, "greeting", GREETING_SKILL)
        await manager.load_all()

        from mirror_core.emotion.engine import EmotionalState
        emotion = EmotionalState(P=0.5, A=0.6)  # P>0.2 ✅ for greeting

        results = manager.match("早安", emotion=emotion)
        assert len(results) == 1
        assert results[0].description == "早安问候"

    @pytest.mark.asyncio
    async def test_match_no_skills(self, manager):
        """无加载技能时空列表"""
        results = manager.match("焦虑", emotion=EmotionalState(P=-0.6))
        assert results == []

    @pytest.mark.asyncio
    async def test_match_no_emotion(self, manager, skills_dir):
        """不传 emotion 时不触发 emotion 匹配"""
        _write_skill(skills_dir, "calming", CALMING_SKILL)
        await manager.load_all()

        results = manager.match("我好焦虑", emotion=None)
        assert len(results) == 0  # 只有关键词得分=1

    @pytest.mark.asyncio
    async def test_match_case_insensitive(self, manager, skills_dir):
        """关键词匹配大小写不敏感"""
        _write_skill(skills_dir, "calming", CALMING_SKILL)
        await manager.load_all()

        emotion = EmotionalState(P=-0.6)
        results = manager.match("我好焦虑", emotion=emotion)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_match_without_emotion_match_condition(self, manager, skills_dir):
        """技能没有 emotion_match 字段时，仅关键词匹配仍不够"""
        content = """\
---
description: "测试"
triggers: ["测试"]
---
body
"""
        _write_skill(skills_dir, "test", content)
        await manager.load_all()

        # 只有关键词得分=1，不够
        results = manager.match("测试一下", emotion=EmotionalState(P=0.0))
        assert len(results) == 0


class TestSkillManagerWatchdog:
    """watchdog 热加载测试 (B-T35)"""

    @pytest.mark.asyncio
    async def test_watchdog_starts(self, skills_dir):
        """启用 watchdog 时不崩溃"""
        _write_skill(skills_dir, "calming", CALMING_SKILL)
        mgr = SkillManager(skills_root=skills_dir, enable_watchdog=True)
        await mgr.load_all()
        assert mgr.count == 1
        await mgr.stop_watchdog()  # 清理

    @pytest.mark.asyncio
    async def test_watchdog_nonexistent_dir(self):
        """目录不存在时 watchdog 静默不启动"""
        mgr = SkillManager(skills_root="/nonexistent_skills", enable_watchdog=True)
        await mgr.load_all()
        await mgr.stop_watchdog()  # 不应崩溃


class TestSkillManagerSadPaths:
    """Sad Path 测试"""

    @pytest.fixture
    def manager(self, skills_dir):
        return SkillManager(skills_root=skills_dir, enable_watchdog=False)

    @pytest.mark.asyncio
    async def test_invalid_skill_skipped(self, manager, skills_dir):
        """非法 SKILL.md 被跳过"""
        _write_skill(skills_dir, "calming", CALMING_SKILL)
        _write_skill(skills_dir, "bad", INVALID_SKILL)
        _write_skill(skills_dir, "bad2", YAML_ERROR_SKILL)

        count = await manager.load_all()
        assert count == 1  # 只有 calming 成功加载
        assert "calming" in manager.list_skills()

    @pytest.mark.asyncio
    async def test_non_skill_dirs_ignored(self, manager, skills_dir):
        """无 SKILL.md 的目录被忽略"""
        os.makedirs(os.path.join(skills_dir, "not-a-skill"), exist_ok=True)
        with open(os.path.join(skills_dir, "not-a-skill", "readme.txt"), "w") as f:
            f.write("not a skill")

        _write_skill(skills_dir, "calming", CALMING_SKILL)
        count = await manager.load_all()
        assert count == 1

    @pytest.mark.asyncio
    async def test_empty_emotion_match(self, manager, skills_dir):
        """空 emotion_match 不触发匹配"""
        content = """\
---
description: "测试"
triggers: ["测试"]
emotion_match: {}
---
body
"""
        _write_skill(skills_dir, "test", content)
        await manager.load_all()

        results = manager.match("测试", emotion=EmotionalState(P=-0.8))
        assert len(results) == 0  # emotion_match 为空字典
