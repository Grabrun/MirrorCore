"""
ConfigManager — 配置管理器 (D-T01)

基于 pydantic 的 YAML 配置加载、校验、环境变量覆盖与热加载。

设计文档 §3.5.2:
- 支持 YAML 加载 + pydantic 参数校验
- 环境变量覆盖 (MIRROR_* 前缀)
- 热加载 (reload 方法)
- 敏感信息自动脱敏
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger("mirror_core.infrastructure.config")

# 敏感 Key 后缀（日志脱敏用）
_SENSITIVE_SUFFIXES = ("key", "secret", "password", "token", "api_key", "api-key")

# 环境变量前缀
_ENV_PREFIX = "MIRROR_"


class PersonaConfig(BaseModel):
    """persona.yaml 配置模型"""
    name: str = ""
    identity: str = ""
    traits: List[str] = Field(default_factory=list)
    suppress_tendency: float = 0.3
    emotional_sensitivity: float = 0.5


class MemoryConfig(BaseModel):
    """memory.yaml 配置模型"""
    short_term_window: int = 20
    recall_top_k: int = 5
    fts_top_n: int = 15
    vec_top_n: int = 15
    rrf_k: int = 60
    consolidation_interval: int = 3600
    forgetting_lambda: float = 0.01
    embedding_dim: int = 768


class ProactiveConfigModel(BaseModel):
    """proactive.yaml 配置模型"""
    enabled: bool = True
    max_per_day: int = 5
    quiet_hours: tuple = (22, 7)
    silence_threshold_days: int = 3
    physiological_metaphor: bool = True
    metaphor_base_probability: float = 0.25
    metaphor_max_per_night: int = 2


class AIProviderConfig(BaseModel):
    """ai_provider.yaml 配置模型"""
    type: str = "openai-compat"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"
    embed_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    max_tokens: int = 8192
    fallback_type: Optional[str] = None


class SafetyConfigModel(BaseModel):
    """safety.yaml 配置模型"""
    dependency_threshold: float = 0.7
    ewma_alpha: float = 0.3
    high_risk_keywords: Dict[str, str] = Field(default_factory=dict)
    dependency_keywords: List[str] = Field(default_factory=list)
    anchors: List[str] = Field(default_factory=list)


class ChannelsConfig(BaseModel):
    """channels.yaml 配置模型"""
    enabled_adapters: List[str] = Field(default_factory=list)


class RootConfig(BaseModel):
    """根配置 — 聚合所有子配置"""
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    proactive: ProactiveConfigModel = Field(default_factory=ProactiveConfigModel)
    ai_provider: AIProviderConfig = Field(default_factory=AIProviderConfig)
    safety: SafetyConfigModel = Field(default_factory=SafetyConfigModel)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)


class ConfigManager:
    """
    配置管理器。

    加载 config/ 目录下所有 YAML 配置文件，支持 pydantic 校验、
    环境变量覆盖与运行时热加载。

    用法:
        cm = ConfigManager("config/")
        cm.load()
        cm.get("persona.name")  # "小艾"
        cm.reload()             # 热加载
    """

    # YAML 文件名 → pydantic 模型名
    _FILE_MODEL_MAP: Dict[str, str] = {
        "persona": "persona",
        "memory": "memory",
        "proactive": "proactive",
        "ai_provider": "ai_provider",
        "safety": "safety",
        "channels": "channels",
    }

    def __init__(self, config_dir: str = "config/"):
        self._config_dir = Path(config_dir)
        self._config = RootConfig()

    @property
    def config(self) -> RootConfig:
        return self._config

    # ---- 加载 ----

    def load(self) -> None:
        """扫描 config_dir 下所有 YAML 文件并加载。"""
        raw: Dict[str, Any] = {}

        if not self._config_dir.is_dir():
            logger.warning("配置目录不存在: %s", self._config_dir)
            self._config = self._apply_env_overrides(RootConfig())
            return

        for yaml_file in self._config_dir.glob("*.yaml"):
            section = yaml_file.stem
            if section not in self._FILE_MODEL_MAP:
                logger.debug("跳过未知配置: %s", yaml_file.name)
                continue
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    raw[section] = data
                logger.info("配置已加载: %s", yaml_file.name)
            except (yaml.YAMLError, OSError) as exc:
                logger.warning("配置加载失败 %s: %s", yaml_file.name, exc)

        self._config = RootConfig(**raw)
        self._config = self._apply_env_overrides(self._config)
        logger.info("配置加载完成: %d 个文件", len(raw))

    # ---- 热加载 (B-T28 / D-T01) ----

    def reload(self) -> None:
        """重新加载所有配置文件（热加载）。"""
        logger.info("配置热加载开始")
        self.load()

    # ---- 查询 ----

    def get(self, key: str, default: Any = None) -> Any:
        """
        通过点分隔路径获取配置值。

        Args:
            key: "persona.name" / "memory.embedding_dim"
            default: 路径不存在时的默认值

        Returns:
            配置值或 default
        """
        parts = key.split(".")
        obj = self._config
        try:
            for part in parts:
                obj = getattr(obj, part)
            return obj
        except (AttributeError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """通过点分隔路径设置配置值（运行时）。"""
        parts = key.split(".")
        obj = self._config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
        logger.debug("配置运行时更新: %s = %s", key, _mask_sensitive(key, value))

    # ---- 环境变量覆盖 ----

    def _apply_env_overrides(self, config: RootConfig) -> RootConfig:
        """
        使用 MIRROR_ 前缀环境变量覆盖配置。

        规则:
            MIRROR_AI_PROVIDER_TYPE → config.ai_provider.type
            MIRROR_AI_PROVIDER_API_KEY → config.ai_provider.api_key
        """
        for env_key, env_val in os.environ.items():
            if not env_key.startswith(_ENV_PREFIX):
                continue

            # MIRROR_AI_PROVIDER_API_KEY → ["ai_provider", "api_key"]
            path = env_key[len(_ENV_PREFIX):].lower().split("_")

            # 尝试映射到 config 路径
            if len(path) >= 2:
                section = path[0]
                field = "_".join(path[1:])
                try:
                    section_obj = getattr(config, section, None)
                    if section_obj and hasattr(section_obj, field):
                        typed_val = self._coerce_type(field, env_val, section_obj)
                        setattr(section_obj, field, typed_val)
                        logger.debug("环境变量覆盖: %s → %s.%s", env_key, section, field)
                except (AttributeError, TypeError):
                    pass
        return config

    @staticmethod
    def _coerce_type(field_name: str, value: str, model: BaseModel) -> Any:
        """将环境变量字符串转为目标字段的类型。"""
        # 使用类访问 model_fields 而非实例（Pydantic V2.11+）
        field_info = type(model).model_fields.get(field_name)
        if field_info is None:
            return value

        target_type = field_info.annotation
        if target_type is int or target_type is float:
            return target_type(value)
        if target_type is bool:
            return value.lower() in ("true", "1", "yes")
        return value


def _mask_sensitive(key: str, value: Any) -> Any:
    """日志脱敏：对敏感字段的值进行掩码。"""
    if any(suffix in key.lower() for suffix in _SENSITIVE_SUFFIXES):
        s = str(value)
        if len(s) > 8:
            return s[:4] + "****" + s[-4:]
        return "****"
    return value
