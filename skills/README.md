# 预装技能

镜核启动时自动扫描 skills/ 目录，加载所有包含 SKILL.md 的技能文件夹。

## 技能列表

- **calming** — 情绪安抚技能，当用户表现出焦虑或悲伤时使用温和语气
- **greeting** — 问候技能，初次接触或长时间未互动时主动打招呼

## 创建技能

创建一个新文件夹，包含 `SKILL.md` 即可：

```
my-skill/
├── SKILL.md              # 必需：技能定义（YAML Front Matter + Markdown 正文）
├── examples/             # 可选：示例对话
├── references/           # 可选：参考材料
└── assets/               # 可选：静态资源
```
