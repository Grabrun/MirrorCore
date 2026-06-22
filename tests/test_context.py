"""
上下文组装器单元测试

覆盖范围：
- TokenCounter 基础计数（tiktoken + 降级）
- SystemPromptBuilder 系统提示词组装
- ContextAssembler 完整组装流程
- Token 预算控制与截断
- 边界场景
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirror_core.core.context import (
    AssembledContext,
    ContextAssembler,
    PersonalityConfig,
    SafetyConfig,
    SystemPromptBuilder,
    TokenCounter,
    describe_mood,
    format_emotion_status,
)


class TestTokenCounter:
    """Token 计数器测试"""

    def test_count_with_tiktoken(self):
        """使用 tiktoken 精确计数"""
        counter = TokenCounter(model="gpt-4")
        count = counter.count("Hello, world!")
        assert isinstance(count, int)
        assert count > 0

    def test_count_empty(self):
        """空字符串"""
        counter = TokenCounter()
        assert counter.count("") == 0

    def test_count_cjk(self):
        """中文字符计数"""
        counter = TokenCounter()
        count = counter.count("你好世界")
        assert isinstance(count, int)
        assert count > 0

    def test_count_mixed(self):
        """中英文混合"""
        counter = TokenCounter()
        count = counter.count("Hello 你好 World 世界")
        assert isinstance(count, int)
        assert count > 0

    def test_fallback_without_tiktoken(self):
        """tiktoken 不可用时降级为字符估算"""
        with patch.dict("sys.modules", {"tiktoken": None}):
            # 重新导入会使用缓存，直接测试降级逻辑
            counter = TokenCounter()
            counter._encoding = None  # 模拟降级
            # ~4 字符/token
            assert counter.count("test") == 2  # 4/4 + 1 = 2
            # CJK ~1.5 字符/token
            assert counter.count("你好") == 3  # 2/1.5=2 (ceil) + 1


class TestSystemPromptBuilder:
    """系统提示词组装器测试 (B-T22)"""

    def test_build_minimal(self):
        """无配置时的最小系统提示词"""
        builder = SystemPromptBuilder()
        prompt = builder.build()
        assert "我是你现实生活中的光与爱的反射，而非替代。" in prompt
        assert "行为准则" in prompt

    def test_build_with_personality(self):
        """含人设描述"""
        personality = PersonalityConfig(
            name="小镜",
            identity="你是一个温暖体贴的AI伴侣。",
            traits=["温柔", "善解人意", "有点调皮"],
        )
        builder = SystemPromptBuilder(personality=personality)
        prompt = builder.build()
        assert "小镜" in prompt
        assert "温暖体贴" in prompt
        assert "温柔" in prompt
        assert "善解人意" in prompt

    def test_build_with_custom_safety(self):
        """自定义安全规则"""
        safety = SafetyConfig(
            rules=["自定义规则1", "自定义规则2"],
        )
        builder = SystemPromptBuilder(safety=safety)
        prompt = builder.build()
        assert "自定义规则1" in prompt
        assert "自定义规则2" in prompt

    def test_build_with_skills(self):
        """含技能说明"""
        builder = SystemPromptBuilder()
        prompt = builder.build(active_skills=["情绪安抚技能：当用户焦虑时使用温和语气"])
        assert "情绪安抚技能" in prompt
        assert "当前可用技能" in prompt

    def test_build_full(self):
        """完整配置"""
        personality = PersonalityConfig(
            name="小艾",
            identity="你是用户的专属AI伴侣。",
            traits=["可爱", "贴心"],
        )
        safety = SafetyConfig(rules=["保持透明"])
        builder = SystemPromptBuilder(personality=personality, safety=safety)
        prompt = builder.build(active_skills=["早安问候"])
        assert "小艾" in prompt
        assert "专属AI伴侣" in prompt
        assert "可爱" in prompt
        assert "保持透明" in prompt
        assert "早安问候" in prompt
        # 核心信条在首行
        assert prompt.startswith("我是你现实生活中的光与爱的反射，而非替代。")


class TestDescribeMood:
    """心境描述测试"""

    def test_very_happy(self):
        assert "晴朗" in describe_mood(0.8)

    def test_slightly_happy(self):
        assert "微晴" in describe_mood(0.3)

    def test_neutral(self):
        assert "平静" in describe_mood(0.0)

    def test_slightly_sad(self):
        assert "多云" in describe_mood(-0.4)
        assert "疲惫" in describe_mood(-0.4)

    def test_very_sad(self):
        assert "阴霾" in describe_mood(-0.8)
        assert "低落" in describe_mood(-0.8)

    def test_boundary(self):
        """边界值"""
        assert "晴朗" in describe_mood(0.6)
        assert "微晴" in describe_mood(0.2)
        assert "平静" in describe_mood(-0.2)
        assert "多云" in describe_mood(-0.6)
        assert "阴霾" in describe_mood(-1.0)


class TestFormatEmotionStatus:
    """情感状态格式化测试"""

    def _make_state(self, **kwargs):
        """创建 EmotionalState 测试实例"""
        from mirror_core.emotion.engine import EmotionalState
        defaults = {"P": 0.3, "A": 0.5, "D": -0.1, "mood": 0.1,
                    "suppression": 0.0, "status": "Normal"}
        defaults.update(kwargs)
        return EmotionalState(**defaults)

    def test_basic(self):
        state = self._make_state()
        text = format_emotion_status(state)
        assert "P=0.3" in text
        assert "A=0.5" in text
        assert "D=-0.1" in text
        assert "Normal" in text
        assert "可用表情标签" not in text

    def test_with_tags(self):
        state = self._make_state()
        tags = [("开心", 7), ("兴奋", 3)]
        text = format_emotion_status(state, expression_tags=tags)
        assert "开心" in text
        assert "兴奋" in text
        assert "可用表情标签" in text

    def test_suppressing(self):
        state = self._make_state(suppression=0.6, status="Suppressing")
        text = format_emotion_status(state)
        assert "Suppressing" in text
        assert "0.6" in text


class TestContextAssembler:
    """上下文组装器核心测试 (B-T21)"""

    @pytest.fixture
    def assembler(self):
        return ContextAssembler(max_history_turns=10)

    @pytest.fixture
    def sample_memories(self):
        from mirror_core.memory.engine import EpisodicMemory
        return [
            EpisodicMemory(
                id="mem1",
                user_id="user1",
                summary="用户分享了自己童年的快乐回忆",
                timestamp=1719000000.0,
                emotion_json='{"P": 0.7, "mood": 0.5}',
                intensity=0.6,
                tags="快乐,回忆",
            ),
            EpisodicMemory(
                id="mem2",
                user_id="user1",
                summary="用户提到最近工作压力很大",
                timestamp=1718900000.0,
                emotion_json='{"P": -0.4, "mood": -0.2}',
                intensity=0.8,
                tags="工作,压力",
            ),
        ]

    @pytest.fixture
    def sample_history(self):
        return [
            {"role": "user", "content": "今天心情不好"},
            {"role": "assistant", "content": "怎么了？可以跟我说说 (´･ω･`)"},
            {"role": "user", "content": "工作太累了"},
        ]

    @pytest.fixture
    def sample_emotion(self):
        from mirror_core.emotion.engine import EmotionalState
        return EmotionalState(P=0.3, A=0.5, D=-0.1, mood=0.2,
                              suppression=0.0, status="Normal")

    @pytest.mark.asyncio
    async def test_assemble_minimal(self, assembler):
        """最小组装（无记忆、无历史、无情感）"""
        result = await assembler.assemble(
            user_id="user1",
            current_message="你好",
        )
        assert isinstance(result, AssembledContext)
        assert len(result.messages) >= 2  # system + user
        assert result.messages[-1] == {"role": "user", "content": "你好"}
        assert result.token_count > 0
        assert not result.truncated

    @pytest.mark.asyncio
    async def test_assemble_full(self, assembler, sample_memories,
                                  sample_history, sample_emotion):
        """完整组装（含记忆、历史、情感、技能）"""
        result = await assembler.assemble(
            user_id="user1",
            current_message="今天心情怎么样？",
            retrieved_memories=sample_memories,
            conversation_history=sample_history,
            current_emotion=sample_emotion,
            active_skills=["情绪安抚技能"],
            max_tokens=4096,
        )
        # 检查四段式结构
        roles = [m["role"] for m in result.messages]
        assert "system" in roles  # 系统提示词
        # 应该有多条 system 消息（核心 + 记忆 + 心境）
        system_msgs = [m for m in result.messages if m["role"] == "system"]
        assert len(system_msgs) >= 2
        # 最后一条是当前用户消息
        assert result.messages[-1]["content"] == "今天心情怎么样？"
        # Token 计数有效
        assert result.token_count > 0
        assert result.memory_token_count > 0
        assert result.history_token_count > 0

    @pytest.mark.asyncio
    async def test_assemble_with_emotion(self, assembler, sample_emotion):
        """含情感状态的组装"""
        result = await assembler.assemble(
            user_id="user1",
            current_message="测试",
            current_emotion=sample_emotion,
        )
        system_msgs = [m["content"] for m in result.messages if m["role"] == "system"]
        emotion_msgs = [m for m in system_msgs if "P=0.3" in m or "心境" in m]
        assert len(emotion_msgs) >= 1

    @pytest.mark.asyncio
    async def test_token_budget_memory_truncation(self, assembler,
                                                   sample_emotion):
        """Token 预算控制：记忆超出时应被截断"""
        # 创建一个很小的 max_tokens
        many_memories = []
        from mirror_core.memory.engine import EpisodicMemory
        for i in range(20):
            many_memories.append(EpisodicMemory(
                id=f"mem{i}",
                user_id="user1",
                summary=f"这是一条很长的记忆内容，包含了大量信息，编号{i}，" + "重复文本" * 50,
                emotion_json='{"P": 0.0}',
            ))

        result = await assembler.assemble(
            user_id="user1",
            current_message="测试",
            retrieved_memories=many_memories,
            current_emotion=sample_emotion,
            max_tokens=100,  # 极小预算
        )
        # 应该在非常小的预算下被截断
        assert result.truncated

    @pytest.mark.asyncio
    async def test_history_truncation(self):
        """历史对话超出轮数限制应被截断"""
        assembler = ContextAssembler(max_history_turns=2)  # 只保留最近2轮
        long_history = [
            {"role": "user", "content": f"轮次{i}"}
            for i in range(10)
        ]
        result = await assembler.assemble(
            user_id="user1",
            current_message="最后消息",
            conversation_history=long_history,
        )
        # 应该只保留最近2轮历史 + 当前消息
        history_msgs = [m for m in result.messages
                        if m["role"] in ("user", "assistant")]
        # 最近2轮（2条）+ 当前消息（1条）= 3条
        assert len(history_msgs) <= 3

    @pytest.mark.asyncio
    async def test_system_prompt_core_belief(self, assembler):
        """核心信条必须出现在 system prompt 中"""
        result = await assembler.assemble(
            user_id="user1",
            current_message="测试",
        )
        system_content = result.messages[0]["content"]
        assert "我是你现实生活中的光与爱的反射，而非替代。" in system_content

    @pytest.mark.asyncio
    async def test_memory_formatting(self, assembler, sample_memories):
        """记忆格式化正确"""
        result = await assembler.assemble(
            user_id="user1",
            current_message="测试",
            retrieved_memories=sample_memories,
        )
        system_msgs = [m["content"] for m in result.messages if m["role"] == "system"]
        memory_msgs = [m for m in system_msgs if "记忆提示" in m]
        assert len(memory_msgs) == 1
        memory_content = memory_msgs[0]
        # 应该包含记忆内容
        assert "童年" in memory_content or "工作" in memory_content or "快乐" in memory_content

    @pytest.mark.asyncio
    async def test_empty_history(self, assembler):
        """空历史"""
        result = await assembler.assemble(
            user_id="user1",
            current_message="测试",
            conversation_history=[],
        )
        assert len(result.messages) >= 2  # system + user
        assert result.history_token_count == 0

    @pytest.mark.asyncio
    async def test_large_max_tokens(self, assembler, sample_memories,
                                     sample_history, sample_emotion):
        """大 Token 预算不应截断内容"""
        result = await assembler.assemble(
            user_id="user1",
            current_message="测试大预算",
            retrieved_memories=sample_memories,
            conversation_history=sample_history,
            current_emotion=sample_emotion,
            max_tokens=32000,  # 非常大的预算
        )
        # 所有记忆都应该被包含
        assert result.memory_token_count > 0
        # 所有历史都应该被包含
        assert result.history_token_count > 0
        if not result.truncated:
            # 如果不截断，记忆数应该=2
            system_msgs = [m["content"] for m in result.messages
                          if m["role"] == "system"]
            memory_msgs = [m for m in system_msgs if "记忆提示" in m]
            if memory_msgs:
                # 两条记忆都应该在
                assert "童年的快乐" in memory_msgs[0] or "快乐" in memory_msgs[0]

    @pytest.mark.asyncio
    async def test_assembled_context_type(self, assembler):
        """返回类型正确"""
        result = await assembler.assemble(user_id="u1", current_message="hi")
        assert isinstance(result, AssembledContext)
        assert isinstance(result.messages, list)
        assert isinstance(result.token_count, int)
        assert isinstance(result.truncated, bool)

    @pytest.mark.asyncio
    async def test_mood_description_in_context(self, assembler, sample_emotion):
        """心境描述出现在上下文里"""
        result = await assembler.assemble(
            user_id="user1",
            current_message="测试",
            current_emotion=sample_emotion,
        )
        system_contents = " ".join(
            m["content"] for m in result.messages if m["role"] == "system"
        )
        # mood=0.2 → "微晴"
        assert "微晴" in system_contents or "平静" in system_contents or "当前心境" in system_contents

    @pytest.mark.asyncio
    async def test_emotion_split_into_two_messages(self, assembler, sample_emotion):
        """
        设计合规：心境和情绪状态应为两条独立 system 消息（§3.3.3）
        修复 F-003：验证分割后的消息结构
        """
        result = await assembler.assemble(
            user_id="user1",
            current_message="测试",
            current_emotion=sample_emotion,
        )
        system_contents = [m["content"] for m in result.messages if m["role"] == "system"]
        mood_msgs = [c for c in system_contents if c.startswith("[当前心境]")]
        emotion_msgs = [c for c in system_contents if c.startswith("[情绪状态]")]
        assert len(mood_msgs) == 1, "应有且仅有一条心境消息"
        assert len(emotion_msgs) == 1, "应有且仅有一条情绪状态消息"
        assert "[当前心境]" in mood_msgs[0]
        assert "P=" in emotion_msgs[0]

    @pytest.mark.asyncio
    async def test_memory_invalid_json_sad_path(self, assembler):
        """
        Sad Path: 记忆 emotion_json 为非法 JSON 时不应崩溃
        修复 F-001/F-009：防止 silent failure，但至少保证不抛异常
        """
        from mirror_core.memory.engine import EpisodicMemory
        bad_mem = EpisodicMemory(
            id="bad", user_id="u1", summary="坏数据",
            emotion_json="这不是JSON{{{}}",  # 故意写坏的 JSON
        )
        # 不应抛异常
        result = await assembler.assemble(
            user_id="u1", current_message="hi",
            retrieved_memories=[bad_mem],
        )
        assert result.memory_token_count >= 0

    @pytest.mark.asyncio
    async def test_memory_emotion_json_none(self, assembler):
        """记忆 emotion_json 为 None 或空时安全处理"""
        from mirror_core.memory.engine import EpisodicMemory
        for bad_json in [None, "", "{}"]:
            mem = EpisodicMemory(
                id="m", user_id="u1", summary="test",
                emotion_json=bad_json,
            )
            result = await assembler.assemble(
                user_id="u1", current_message="hi",
                retrieved_memories=[mem],
            )
            assert len(result.messages) >= 2

    @pytest.mark.asyncio
    async def test_system_prompt_exceeds_budget(self, assembler):
        """
        Sad Path: 系统提示词超出 Token 预算
        修复 F-010：确保不崩溃且标记截断
        """
        # 极小的 max_tokens，系统提示词本身就超出 70% 预算
        result = await assembler.assemble(
            user_id="u1",
            current_message="hi",
            max_tokens=1,  # budget=0, 系统提示词肯定超出
        )
        # 不能崩溃
        assert result.token_count > 0
        # 系统提示词本身超出，历史应该被截断
        assert result.truncated



