"""
AI 提供商适配层 + 文本生成器单元测试

覆盖范围：
- AIProvider 抽象接口
- Provider 工厂
- TextGenerator 重试 / 超时 / 降级
- post_process 颜文字 + 锚点注入
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mirror_core.execution.ai_providers import AIProvider, ChatResponse
from mirror_core.execution.ai_providers.factory import create_provider
from mirror_core.execution.llm import TextGenerator, FALLBACK_REPLIES, MAX_RETRIES


class TestAIProviderInterface:
    """AIProvider 抽象接口测试 (B-T29)"""

    def test_abstract_methods(self):
        """AIProvider 不能直接实例化"""
        with pytest.raises(TypeError):
            AIProvider()

    def test_concrete_implementation(self):
        """具体实现必须实现所有抽象方法"""

        class MockProvider(AIProvider):
            @property
            def embedding_dim(self): return 768

            @property
            def max_tokens(self): return 4096

            async def chat(self, messages, **kwargs):
                return ChatResponse(content="hi", model="mock")

            async def embed(self, text):
                return [0.1] * 768

        p = MockProvider()
        assert p.embedding_dim == 768
        assert p.max_tokens == 4096


class TestProviderFactory:
    """Provider 工厂测试 (B-T30)"""

    def test_create_openai_compat(self):
        """创建 OpenAI 兼容适配器"""
        provider = create_provider({
            "type": "openai-compat",
            "api_key": "sk-test",
            "model": "gpt-4",
        })
        from mirror_core.execution.ai_providers.openai_compat import (
            OpenAICompatProvider,
        )
        assert isinstance(provider, OpenAICompatProvider)
        assert provider._model == "gpt-4"

    def test_create_deepseek(self):
        """创建 DeepSeek 适配器"""
        provider = create_provider({
            "type": "deepseek",
            "api_key": "sk-test",
        })
        from mirror_core.execution.ai_providers.openai_compat import (
            DeepSeekCompatProvider,
        )
        assert isinstance(provider, DeepSeekCompatProvider)

    def test_create_anthropic(self):
        """创建 Anthropic 适配器"""
        provider = create_provider({
            "type": "anthropic",
            "api_key": "sk-ant-test",
        })
        from mirror_core.execution.ai_providers.anthropic_compat import (
            AnthropicCompatProvider,
        )
        assert isinstance(provider, AnthropicCompatProvider)

    def test_create_glm(self):
        """创建 GLM 适配器"""
        provider = create_provider({
            "type": "glm",
            "api_key": "glm-test",
        })
        from mirror_core.execution.ai_providers.glm import GLMProvider
        assert isinstance(provider, GLMProvider)

    def test_create_unknown_type(self):
        """未知类型应抛出 ValueError"""
        with pytest.raises(ValueError, match="未知的 AI Provider 类型"):
            create_provider({"type": "unknown"})

    def test_create_case_insensitive(self):
        """类型名大小写不敏感"""
        from mirror_core.execution.ai_providers.openai_compat import (
            OpenAICompatProvider,
        )
        provider = create_provider({"type": "OpenAI-COMPAT", "api_key": ""})
        assert isinstance(provider, OpenAICompatProvider)


class TestTextGenerator:
    """文本生成器测试 (B-T32)"""

    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock(spec=AIProvider)
        provider.chat = AsyncMock()
        provider.embedding_dim = 768
        provider.max_tokens = 4096
        return provider

    @pytest.fixture
    def generator(self, mock_provider):
        return TextGenerator(provider=mock_provider)

    @pytest.mark.asyncio
    async def test_successful_call(self, generator, mock_provider):
        """正常调用返回内容"""
        mock_provider.chat.return_value = ChatResponse(
            content="你好！今天过得怎么样？",
            model="gpt-4o",
            usage={"prompt_tokens": 50, "completion_tokens": 10},
        )
        result = await generator.generate_response(
            messages=[{"role": "user", "content": "你好"}],
        )
        assert result == "你好！今天过得怎么样？"

    @pytest.mark.asyncio
    async def test_timeout_retry(self, generator, mock_provider):
        """超时后重试，最终成功"""
        mock_provider.chat.side_effect = [
            asyncio.TimeoutError(),  # 第一次超时
            ChatResponse(content="重试成功！"),  # 第二次成功
        ]
        result = await generator.generate_response(
            messages=[{"role": "user", "content": "测试"}],
        )
        assert result == "重试成功！"
        assert mock_provider.chat.await_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_fail_fallback(self, generator, mock_provider):
        """全部重试失败后使用降级回复"""
        mock_provider.chat.side_effect = asyncio.TimeoutError()

        result = await generator.generate_response(
            messages=[{"role": "user", "content": "测试"}],
        )
        # 应降级
        assert result in FALLBACK_REPLIES
        assert mock_provider.chat.await_count == MAX_RETRIES + 1

    @pytest.mark.asyncio
    async def test_exception_retry(self, generator, mock_provider):
        """异常后重试"""
        mock_provider.chat.side_effect = [
            RuntimeError("API 错误"),
            ChatResponse(content="第二次成功"),
        ]
        result = await generator.generate_response(
            messages=[{"role": "user", "content": "test"}],
        )
        assert result == "第二次成功"

    @pytest.mark.asyncio
    async def test_rate_limit_429(self, generator, mock_provider):
        """429 速率限制后自动等待并重试"""
        mock_provider.chat.side_effect = [
            RuntimeError("429 Too Many Requests"),  # 429
            ChatResponse(content="成功"),
        ]
        result = await generator.generate_response(
            messages=[{"role": "user", "content": "test"}],
        )
        assert result == "成功"

    @pytest.mark.asyncio
    async def test_provider_property(self, generator, mock_provider):
        """provider 属性返回注入的 provider"""
        assert generator.provider is mock_provider


class TestPostProcess:
    """后处理测试 (B-T33)"""

    @pytest.fixture
    def mock_provider(self):
        p = MagicMock(spec=AIProvider)
        p.chat = AsyncMock(return_value=ChatResponse(content="ok"))
        p.embedding_dim = 768
        p.max_tokens = 4096
        return p

    @pytest.fixture
    def safety(self):
        s = MagicMock()
        s.inject_reality_anchor = AsyncMock(side_effect=lambda response, **kw: response)
        return s

    @pytest.fixture
    def emotion(self):
        from mirror_core.emotion.engine import EmotionalState
        return EmotionalState(P=0.7, A=0.6, D=0.3, mood=0.5)

    @pytest.mark.asyncio
    async def test_kaomoji_added(self, mock_provider, emotion):
        """颜文字追加到回复后"""
        generator = TextGenerator(provider=mock_provider)
        result = await generator.post_process(
            raw_text="今天天气真好",
            emotion=emotion,
        )
        # P=0.7,A=0.6 → "P>0.5,A>0.5" → kaomoji
        assert "今天天气真好" in result
        assert any(km in result for km in ["(≧▽≦)", "ヽ(>∀<)ノ"])

    @pytest.mark.asyncio
    async def test_no_emotion_no_kaomoji(self, mock_provider):
        """无情感时不追加颜文字"""
        generator = TextGenerator(provider=mock_provider)
        result = await generator.post_process(
            raw_text="hello",
            emotion=None,
        )
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_anchor_injected(self, mock_provider, safety):
        """高依赖度时注入锚点"""
        safety.inject_reality_anchor = AsyncMock(
            return_value="回复内容\n（锚点文本）"
        )
        generator = TextGenerator(provider=mock_provider, safety_engine=safety)
        result = await generator.post_process(
            raw_text="回复内容",
            dependency_score=0.8,
        )
        assert "锚点文本" in result

    @pytest.mark.asyncio
    async def test_no_safety_no_anchor(self, mock_provider):
        """无安全引擎时不注入锚点"""
        generator = TextGenerator(provider=mock_provider)
        result = await generator.post_process(
            raw_text="hello",
            dependency_score=0.9,
        )
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_full_pipeline(self, mock_provider, safety, emotion):
        """完整后处理流水线: 颜文字 + 锚点"""
        safety.inject_reality_anchor = AsyncMock(
            return_value="今天开心吗？ (≧▽≦)\n（现实锚点）"
        )
        generator = TextGenerator(provider=mock_provider, safety_engine=safety)
        result = await generator.post_process(
            raw_text="今天开心吗？",
            emotion=emotion,
            dependency_score=0.8,
        )
        # 颜文字和锚点都应存在
        assert "今天开心吗？" in result
        assert any(km in result for km in ["(≧▽≦)", "ヽ(>∀<)ノ"])
        assert "现实锚点" in result

    def test_kaomoji_selection_sad(self):
        """消极情绪选择正确的颜文字"""
        from mirror_core.emotion.engine import EmotionalState
        from mirror_core.execution.llm import TextGenerator
        from unittest.mock import MagicMock

        generator = TextGenerator(provider=MagicMock())
        emotion = EmotionalState(P=-0.7, A=0.2, mood=-0.4)  # P<-0.5,A<0.3

        kaomoji = generator._select_kaomoji(emotion)
        assert kaomoji in ["(´；ω；`)", "(◞‸◟)"]


class TestCloseChain:
    """G-001: close 传播链测试"""

    def test_close_propagates_to_provider(self):
        """TextGenerator.close() 应调用 provider.close()"""
        provider = MagicMock(spec=AIProvider)
        provider.close = AsyncMock()

        generator = TextGenerator(provider=provider)
        import asyncio
        asyncio.run(generator.close())

        provider.close.assert_awaited_once()

    def test_async_context_manager(self):
        """async with 退出时自动 close"""
        provider = MagicMock(spec=AIProvider)
        provider.close = AsyncMock()
        provider.chat = AsyncMock(return_value=ChatResponse(content="ok"))
        provider.embedding_dim = 768
        provider.max_tokens = 4096

        async def run():
            async with TextGenerator(provider=provider) as gen:
                assert gen.provider is provider
            # 退出后应自动 close
            provider.close.assert_awaited_once()

        import asyncio
        asyncio.run(run())


class TestHttpRequestFormat:
    """G-002: HTTP 请求格式验证（通过 mock transport）"""

    @pytest.mark.asyncio
    async def test_openai_chat_request_payload(self):
        """OpenAI chat() 发送正确的请求体"""
        import json
        from unittest.mock import AsyncMock

        # 模拟 httpx 响应
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "你好！"}}],
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_response.raise_for_status = MagicMock()

        mock_post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient.post", mock_post):
            from mirror_core.execution.ai_providers.openai_compat import (
                OpenAICompatProvider,
            )
            provider = OpenAICompatProvider(api_key="sk-test")
            result = await provider.chat([{"role": "user", "content": "你好"}])

        # 验证请求体
        call_kwargs = mock_post.call_args.kwargs
        payload = call_kwargs["json"]
        assert payload["model"] == "gpt-4o"
        assert payload["messages"] == [{"role": "user", "content": "你好"}]
        assert result.content == "你好！"

    @pytest.mark.asyncio
    async def test_openai_embed_request_payload(self):
        """OpenAI embed() 发送正确的请求体"""
        from unittest.mock import AsyncMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
            "model": "text-embedding-3-small",
        }
        mock_response.raise_for_status = MagicMock()

        mock_post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient.post", mock_post):
            from mirror_core.execution.ai_providers.openai_compat import (
                OpenAICompatProvider,
            )
            provider = OpenAICompatProvider(api_key="sk-test")
            result = await provider.embed("测试文本")

        call_kwargs = mock_post.call_args.kwargs
        payload = call_kwargs["json"]
        assert payload["model"] == "text-embedding-3-small"
        assert payload["input"] == "测试文本"
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_anthropic_chat_request_payload(self):
        """Anthropic chat() 发送正确的请求体（Messages API 格式）"""
        from unittest.mock import AsyncMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "你好！"}],
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 12, "output_tokens": 8},
        }
        mock_response.raise_for_status = MagicMock()

        mock_post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient.post", mock_post):
            from mirror_core.execution.ai_providers.anthropic_compat import (
                AnthropicCompatProvider,
            )
            provider = AnthropicCompatProvider(api_key="sk-ant-test")
            result = await provider.chat([
                {"role": "system", "content": "你是助手"},
                {"role": "user", "content": "你好"},
            ])

        call_kwargs = mock_post.call_args.kwargs
        payload = call_kwargs["json"]
        # Anthropic 格式：system 字段在外层，messages 只含 user/assistant
        assert payload["system"] == "你是助手"
        assert payload["messages"] == [{"role": "user", "content": "你好"}]
        assert payload["model"] == "claude-sonnet-4-20250514"
        assert result.content == "你好！"


class TestAnthropicAdapter:
    """Anthropic 适配器测试 (B-T31)"""

    def test_message_conversion(self):
        """OpenAI 格式转 Anthropic 格式"""
        from mirror_core.execution.ai_providers.anthropic_compat import (
            AnthropicCompatProvider,
        )
        system, messages = AnthropicCompatProvider._convert_messages([
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
        ])
        assert system == "你是助手"
        assert messages == [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
        ]

    def test_embed_not_implemented(self):
        """Anthropic 的 embed 应抛出 NotImplementedError"""
        from mirror_core.execution.ai_providers.anthropic_compat import (
            AnthropicCompatProvider,
        )
        provider = AnthropicCompatProvider(api_key="test")
        import pytest
        with pytest.raises(NotImplementedError):
            import asyncio
            asyncio.run(provider.embed("test"))


class TestFactorySadPaths:
    """工厂 Sad Path 测试"""

    def test_empty_config(self):
        """空配置（type 为空字符串）应抛出 ValueError"""
        with pytest.raises(ValueError, match="未知的 AI Provider 类型"):
            create_provider({})

    def test_none_config(self):
        """None 配置应抛出异常"""
        with pytest.raises((TypeError, AttributeError)):
            create_provider(None)  # type: ignore
