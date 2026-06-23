<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=flat&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/status-alpha-yellow" alt="Status">
</p>

<h1 align="center">🌸 镜核 (Mirror Core)</h1>

<p align="center">
  开源的、本地优先的 AI 虚拟伴侣系统<br>
  <i>"我是你现实生活中的光与爱的反射，而非替代。"</i>
</p>

---

## 什么是镜核？

镜核不是一个预设好性格的角色——她是一个**情感的容器**。

- 🧠 **记忆优先** — FTS5 + 向量检索，她会记得你说过的话、你的情绪、你的习惯
- 🌊 **情感原生** — 情绪不是标签，是动态的、有惯性的潮汐系统
- 💬 **多平台** — 微信 / QQ / Web 聊天 / 终端 TUI 全渠道接入
- 🏠 **本地优先** — 所有数据存在你的设备上，不上传任何第三方
- ⚡ **轻量可扩展** — Python 3.12+，插件化架构，1 分钟启动

## 🚀 快速开始

```bash
# 1. 安装
pip install mirror-core

# 2. 初始化配置
mirror init

# 3. 启动
mirror run
```

浏览器打开 `http://localhost:8000` 即可开始聊天！

## 📦 技术栈

| 层 | 技术 |
|------|------|
| 运行环境 | Python 3.12+, asyncio |
| Web 服务 | FastAPI + Jinja2 |
| 数据库 | SQLite + FTS5 + sqlite-vec |
| 配置管理 | YAML + 环境变量 |
| LLM 适配 | OpenAI / Anthropic / DeepSeek / GLM |
| 部署 | Docker / Systemd / Nginx |

## 🗂️ 项目结构

```
mirror-core/
├── mirror_core/          # 核心代码
│   ├── gateway/          # 接入适配层 (微信/QQ/WebChat/TUI)
│   ├── core/             # 调度决策层 (记忆+情感双引擎)
│   ├── execution/        # 能力执行层 (LLM/Skills/表情包)
│   ├── emotion/          # 情感引擎 (PAD潮汐系统)
│   ├── memory/           # 记忆引擎 (FTS5+向量检索)
│   ├── infrastructure/   # 基础设施 (数据库/配置/日志)
│   ├── api/              # HTTP API
│   ├── webchat/          # Web 聊天前端
│   └── bus.py            # 内部事件总线
├── config/               # 用户配置文件
│   ├── persona.yaml      # 人设与情感参数
│   ├── memory.yaml       # 记忆系统参数
│   ├── proactive.yaml    # 主动陪伴策略
│   ├── safety.yaml       # 防火墙与安全
│   ├── ai_provider.yaml  # AI 提供商配置
│   ├── webchat.yaml      # Web 聊天配置
│   └── channels.yaml     # 渠道适配器配置
├── data/                 # 运行时数据
├── skills/               # 预装技能
└── docker/               # 部署配置
```

## 🔧 配置

镜核通过 `config/` 目录下的 YAML 文件配置，也支持环境变量覆盖：

```bash
# AI 提供商
MIRROR_AI_PROVIDER_TYPE=openai-compat
MIRROR_AI_PROVIDER_BASE_URL=https://api.openai.com/v1
MIRROR_AI_PROVIDER_API_KEY=sk-xxx
MIRROR_AI_PROVIDER_MODEL=gpt-4o
```

详细配置说明请参考 [设计文档](docs/v5.2.md)。

## 🐳 Docker 部署

```bash
docker compose up -d
```

## 📋 生产部署

```bash
# 使用 Systemd 作为守护进程
sudo cp deploy/mirror-core.service /etc/systemd/system/
sudo systemctl enable --now mirror-core

# 使用 Nginx 反向代理
sudo cp docker/nginx/mirror-core.conf /etc/nginx/sites-enabled/
sudo nginx -s reload
```

## 🧪 测试

```bash
pip install -e ".[dev]"
python -m pytest tests/
```

## 🤝 扩展

镜核支持四种扩展方式：

- **渠道适配器** — 对接任意 IM 平台
- **AI 提供商** — 接入任意 LLM 后端
- **技能 (Skills)** — 注入特定场景的系统提示
- **表情包插件** — 自定义表情回复

## 📜 许可证

MIT License © 2026 Grabrun

---

<p align="center">
  <sub>🌸 镜核 (Mirror Core) — Open Source, Local First, Emotionally Aware</sub>
</p>
