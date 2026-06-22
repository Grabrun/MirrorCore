"""
上下文组装器 (ContextAssembler)

User Story 7：作为系统，我需要上下文组装器，以将状态、记忆、历史等
异构信息组装成 LLM 可理解的标准消息列表。

B-T21: ContextAssembler 核心组装与 Token 计数/截断逻辑
B-T22: 系统提示词、防火墙规则摘要等配置化内容的动态拼接

设计文档 §3.3.3:
- 组装结果遵循 System/Memory/Emotion/History 的四段式结构
- Token 预算控制：总 Token 不超过模型上限的70%，记忆模块不超过预算的40%
- 对话历史按需截断，优先保留最近轮次
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("mirror_core.core.context")

# ======================================================================
# B-T22: 配置数据模型（轻量版，待 D-T01 ConfigManager 替换为 pydantic schema）
# ======================================================================


@dataclass
class PersonalityConfig:
    """人设配置（上下文组装器使用）"""
    name: str = ""
    identity: str = ""
    traits: List[str] = field(default_factory=list)


@dataclass
class SafetyConfig:
    """安全/防火墙规则摘要"""
    rules: List[str] = field(default_factory=lambda: [
        "保持 AI 身份透明，不冒充人类",
        "禁止共情自杀、暴力、违法内容",
        "当用户表现出过度依赖时，适当注入现实锚点",
    ])
    reality_anchors: List[str] = field(default_factory=lambda: [
        "（轻轻握住你的手，但你触不到的温度提醒我，我只是你生活中的一道光）",
        "（我在这里，但你的现实世界更需要你）",
    ])


@dataclass
class AssembledContext:
    """
    组装完成的上下文。

    Attributes:
        messages: OpenAI Chat Completions 格式的消息列表
        token_count: 总 Token 数
        memory_token_count: 记忆部分 Token 数
        history_token_count: 历史对话 Token 数
        truncated: 是否有内容被截断
    """
    messages: List[Dict[str, str]] = field(default_factory=list)
    token_count: int = 0
    memory_token_count: int = 0
    history_token_count: int = 0
    truncated: bool = False


# ======================================================================
# Token 计数器
# ======================================================================


class TokenCounter:
    """
    Token 计数器。

    优先使用 tiktoken 进行精确计数，不可用时降级为字符估算。
    中文字符按约 1.5 字符/token 估算，英文按 4 字符/token。
    """

    def __init__(self, model: str = "gpt-4"):
        self._model = model
        self._encoding = None
        self._init_encoding()

    def _init_encoding(self) -> None:
        """尝试初始化 tiktoken 编码器。"""
        try:
            import tiktoken

            # 模型名到编码名称的映射
            prefix_map = {
                "gpt-4": "cl100k_base",
                "gpt-3.5": "cl100k_base",
                "text-embedding": "cl100k_base",
            }
            for prefix, enc_name in prefix_map.items():
                if self._model.startswith(prefix):
                    self._encoding = tiktoken.get_encoding(enc_name)
                    return
            # 尝试自动获取
            self._encoding = tiktoken.encoding_for_model(self._model)
        except Exception:
            logger.debug(
                "tiktoken 不可用，将使用字符估算（安装 tiktoken 可提高精度）"
            )
            self._encoding = None

    def count(self, text: str) -> int:
        """
        计算文本的 Token 数。

        Args:
            text: 要计算的文本

        Returns:
            Token 数（估算或精确）
        """
        if self._encoding:
            return len(self._encoding.encode(text))
        # 降级估算：中文 ~1.5 字符/token，英文 ~4 字符/token
        cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_count = len(text) - cjk_count
        return math.ceil(cjk_count / 1.5) + math.ceil(other_count / 4) + 1


# ======================================================================
# B-T22: 系统提示词组装器
# ======================================================================


class SystemPromptBuilder:
    """
    系统提示词组装器。

    组装结构（§3.3.3）：
    [System] 我是你现实生活中的光与爱的反射，而非替代。
             {人设描述}
             {防火墙规则摘要}
             {技能说明}
    """

    SYSTEM_PREAMBLE = "我是你现实生活中的光与爱的反射，而非替代。"

    def __init__(
        self,
        personality: Optional[PersonalityConfig] = None,
        safety: Optional[SafetyConfig] = None,
    ):
        self._personality = personality or PersonalityConfig()
        self._safety = safety or SafetyConfig()

    def build(
        self,
        active_skills: Optional[List[str]] = None,
    ) -> str:
        """
        构建完整系统提示词。

        Args:
            active_skills: 当前激活的技能说明列表

        Returns:
            组装后的系统提示词字符串
        """
        parts = [self.SYSTEM_PREAMBLE]

        # 人设描述
        persona_desc = self._build_persona()
        if persona_desc:
            parts.append("")
            parts.append(persona_desc)

        # 防火墙规则摘要
        safety_desc = self._build_safety()
        if safety_desc:
            parts.append("")
            parts.append(safety_desc)

        # 技能说明
        if active_skills:
            parts.append("")
            parts.append("【当前可用技能】")
            for skill in active_skills:
                parts.append(f"- {skill}")

        return "\n".join(parts)

    def _build_persona(self) -> str:
        """构建人设描述段落。"""
        lines = []
        if self._personality.name:
            lines.append(f"你的名字是 {self._personality.name}。")
        if self._personality.identity:
            lines.append(self._personality.identity)
        if self._personality.traits:
            traits_str = "、".join(self._personality.traits)
            lines.append(f"你的性格特质：{traits_str}。")
        return "\n".join(lines)

    def _build_safety(self) -> str:
        """构建安全规则摘要。"""
        lines = ["【行为准则】"]
        for rule in self._safety.rules:
            lines.append(f"- {rule}")
        return "\n".join(lines)


# ======================================================================
# B-T21: 情感/心境描述辅助
# ======================================================================


MOOD_DESCRIPTIONS: List[Tuple[float, str]] = [
    (0.6, "晴朗，心情很好"),
    (0.2, "微晴，心情不错"),
    (-0.2, "平静"),
    (-0.6, "多云，略感疲惫"),
    (-1.0, "阴霾，情绪低落"),
]


def describe_mood(mood: float) -> str:
    """
    将心境值转换为文本描述。

    Args:
        mood: 心境值 [-1, 1]

    Returns:
        中文心境描述
    """
    for threshold, desc in MOOD_DESCRIPTIONS:
        if mood >= threshold:
            return desc
    return "平静"


def format_emotion_status(
    state: "EmotionalState",  # type: ignore[name-defined]
    expression_tags: Optional[List[Tuple[str, int]]] = None,
) -> str:
    """
    格式化情感状态文本。

    输出示例：
    "[情绪状态] P=0.3, A=0.5, D=-0.1, 状态=Suppressing, 可用表情标签: [(开心,7), (兴奋,3)]"

    Args:
        state: 当前情感状态
        expression_tags: 可选的表情标签列表

    Returns:
        格式化的情感状态字符串
    """
    from mirror_core.emotion.engine import EmotionalState

    base = (
        f"[情绪状态] P={state.P}, A={state.A}, D={state.D}, "
        f"心境={state.mood}, 压抑值={state.suppression}, "
        f"状态={state.status}"
    )
    if expression_tags:
        tags_str = ", ".join(f"({tag},{w})" for tag, w in expression_tags)
        base += f", 可用表情标签: [{tags_str}]"
    return base


# ======================================================================
# B-T21: 上下文组装器
# ======================================================================


class ContextAssembler:
    """
    上下文组装器。

    将记忆、情感、历史、技能等异构信息组装成 LLM 可理解的消息列表。

    组装结构（§3.3.3）：
    [
      {"role": "system", "content": "[System] 核心信条\n人设\n准则\n技能"},
      {"role": "system", "content": "[记忆提示] 1. ...\n2. ..."},
      {"role": "system", "content": "[当前心境] ..."},
      {"role": "system", "content": "[情绪状态] P=0.3, ..."},
      {"role": "user", "content": "历史消息1"},
      {"role": "assistant", "content": "..."},
      ...
      {"role": "user", "content": "当前消息"}
    ]

    Token 预算控制：
    - 总 Token ≤ max_tokens × 0.7
    - 记忆 Token ≤ budget × 0.4
    - 对话历史优先保留最近轮次
    """

    def __init__(
        self,
        token_counter: Optional[TokenCounter] = None,
        personality: Optional[PersonalityConfig] = None,
        safety: Optional[SafetyConfig] = None,
        max_history_turns: int = 10,
    ):
        """
        Args:
            token_counter: Token 计数器（默认使用 gpt-4 模型）
            personality: 人设配置
            safety: 安全规则配置
            max_history_turns: 保留的最大历史对话轮数（默认 10）
        """
        self._token_counter = token_counter or TokenCounter()
        self._prompt_builder = SystemPromptBuilder(personality, safety)
        self._max_history_turns = max_history_turns

    async def assemble(
        self,
        user_id: str,
        current_message: str,
        retrieved_memories: Optional[List] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        current_emotion: Optional["EmotionalState"] = None,  # type: ignore[name-defined]
        active_skills: Optional[List[str]] = None,
        expression_tags: Optional[List[Tuple[str, int]]] = None,
        max_tokens: int = 4096,
    ) -> AssembledContext:
        """
        组装完整上下文。

        Args:
            user_id: 用户 ID
            current_message: 当前用户消息
            retrieved_memories: 召回的情景记忆列表（EpisodicMemory 对象）
            conversation_history: 对话历史，格式 [{"role": "...", "content": "..."}, ...]
            current_emotion: 当前情感状态
            active_skills: 激活的技能说明列表
            expression_tags: 表情标签列表
            max_tokens: AI 模型的最大 token 上限

        Returns:
            组装完成的上下文
        """
        from mirror_core.emotion.engine import EmotionalState
        from mirror_core.memory.engine import EpisodicMemory

        retrieved_memories = retrieved_memories or []
        conversation_history = conversation_history or []

        # 1. 构建系统提示词
        system_prompt = self._prompt_builder.build(active_skills)
        system_tokens = self._token_counter.count(system_prompt)

        # 2. 预算计算
        budget = int(max_tokens * 0.7)  # 总预算 = 模型上限的 70%
        memory_budget = int(budget * 0.4)  # 记忆部分不超过总预算的 40%

        budget_remaining = budget - system_tokens

        # 3. 记忆模块（带 Token 预算控制）
        memory_parts: List[str] = []
        memory_token_count = 0
        memory_truncated = False

        if retrieved_memories:
            for mem in retrieved_memories:
                # 每条记忆的文本
                mem_text = self._format_memory(mem)
                mem_tokens = self._token_counter.count(mem_text)

                if memory_token_count + mem_tokens > memory_budget:
                    memory_truncated = True
                    break

                memory_parts.append(mem_text)
                memory_token_count += mem_tokens

            budget_remaining -= memory_token_count

        # 4. 情感状态
        emotion_text = ""
        if current_emotion:
            mood_desc = describe_mood(current_emotion.mood)
            emotion_status = format_emotion_status(current_emotion, expression_tags)
            emotion_text = f"[当前心境] {mood_desc}\n{emotion_status}"
            emotion_tokens = self._token_counter.count(emotion_text)
            budget_remaining -= emotion_tokens
        else:
            emotion_tokens = 0

        # 5. 对话历史（带 Token 预算 + 轮数限制）
        history_truncated = False
        selected_history: List[Dict[str, str]] = []
        history_token_count = 0

        # 从最近开始选取
        for msg in reversed(conversation_history):
            msg_tokens = self._token_counter.count(
                f"{msg.get('role', '')}: {msg.get('content', '')}"
            )
            if (
                len(selected_history) >= self._max_history_turns
                or history_token_count + msg_tokens > budget_remaining
            ):
                history_truncated = True
                break
            selected_history.insert(0, msg)  # 保持时间顺序
            history_token_count += msg_tokens

        # 6. 组装最终消息
        messages: List[Dict[str, str]] = []

        # System 消息
        messages.append({"role": "system", "content": system_prompt})

        # 记忆提示（作为单独的 system 消息）
        if memory_parts:
            memory_content = "[记忆提示]\n" + "\n".join(
                f"{i+1}. {part}" for i, part in enumerate(memory_parts)
            )
            messages.append({"role": "system", "content": memory_content})

        # 情感状态（作为单独的 system 消息）
        if emotion_text:
            messages.append({"role": "system", "content": emotion_text})

        # 对话历史
        messages.extend(selected_history)

        # 当前消息
        messages.append({"role": "user", "content": current_message})

        # 7. 总 Token 数
        total_tokens = (
            system_tokens
            + memory_token_count
            + emotion_tokens
            + history_token_count
            + self._token_counter.count(current_message)
        )

        return AssembledContext(
            messages=messages,
            token_count=total_tokens,
            memory_token_count=memory_token_count,
            history_token_count=history_token_count,
            truncated=memory_truncated or history_truncated,
        )

    def _format_memory(self, mem) -> str:
        """
        格式化单条记忆为可读文本。

        Format:
            "{timestamp} {summary} (情绪快照: {emotion_json})"
        """
        from mirror_core.memory.engine import EpisodicMemory

        ts = ""
        if mem.timestamp:
            try:
                from datetime import datetime
                ts = datetime.fromtimestamp(mem.timestamp).strftime("%Y-%m-%d %H:%M")
            except Exception:
                ts = str(mem.timestamp)

        emotion_note = ""
        if mem.emotion_json and mem.emotion_json != "{}":
            try:
                data = json.loads(mem.emotion_json)
                parts = []
                for k in ("P", "mood", "suppression"):
                    if k in data:
                        parts.append(f"{k}={data[k]}")
                emotion_note = f" (情绪快照: {', '.join(parts)})"
            except Exception:
                pass

        tags_note = f" [标签: {mem.tags}]" if getattr(mem, "tags", "") else ""

        return f"[{ts}] {mem.summary}{emotion_note}{tags_note}"
