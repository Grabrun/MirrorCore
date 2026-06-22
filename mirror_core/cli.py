#!/usr/bin/env python3
"""
镜核 (Mirror Core) 命令行入口

设计文档 §10:
    mirror init         - 生成默认 config/ 和 data/ 目录
    mirror run          - 启动所有启用的适配器
    mirror doctor       - 系统诊断
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("mirror_core.cli")

# 项目根目录
ROOT = Path(__file__).parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mirror",
        description="镜核 (Mirror Core) - AI 虚拟伴侣系统",
    )
    parser.add_argument(
        "--config", "-c",
        default=str(ROOT / "config"),
        help="配置文件目录 (默认: config/)",
    )
    parser.add_argument(
        "--data",
        default=str(ROOT / "data"),
        help="数据文件目录 (默认: data/)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # mirror init
    init_parser = sub.add_parser("init", help="初始化配置和数据目录")
    init_parser.add_argument("--force", action="store_true", help="覆盖已有文件")

    # mirror run
    run_parser = sub.add_parser("run", help="启动镜核系统")
    run_parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    run_parser.add_argument("--port", type=int, default=8000, help="监听端口")

    # mirror doctor
    sub.add_parser("doctor", help="系统诊断")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "run":
        asyncio.run(cmd_run(args))
    elif args.command == "doctor":
        cmd_doctor(args)


def cmd_init(args) -> None:
    """初始化配置和数据目录。"""
    config_dir = Path(args.config)
    data_dir = Path(args.data)

    # 创建目录
    config_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 生成默认配置文件
    defaults = {
        "persona.yaml": PERSONA_DEFAULT,
        "memory.yaml": MEMORY_DEFAULT_YAML,
        "proactive.yaml": PROACTIVE_DEFAULT_YAML,
    }

    for name, content in defaults.items():
        path = config_dir / name
        if path.exists() and not args.force:
            print(f"  ⏭️  跳过 {name} (已存在)")
        else:
            path.write_text(content, encoding="utf-8")
            print(f"  ✅ 创建 {name}")

    # 创建 skills 目录
    skills_dir = ROOT / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✅ 创建 skills/ 目录")

    print(f"\n镜核初始化完成！运行 mirror run 启动系统。")


async def cmd_run(args) -> None:
    """启动镜核系统。"""
    from mirror_core.infrastructure.logging_config import setup_logging
    from mirror_core.infrastructure.database import Database
    from mirror_core.infrastructure.config import ConfigManager

    setup_logging(log_level=args.log_level, json_format=False)
    logger.info("镜核正在启动...")

    db = Database(path=os.path.join(args.data, "mirror.db"))
    await db.initialize()

    cm = ConfigManager(config_dir=args.config)
    cm.load()

    logger.info("镜核启动完成 (API: http://%s:%s)", args.host, args.port)

    # 保持运行
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        logger.info("镜核正在关闭...")
    finally:
        await db.close()


def cmd_doctor(args) -> None:
    """系统诊断。"""
    print(f"\n🌸 镜核 (Mirror Core) — 系统诊断")
    print(f"  {'✅' if True else '❌'} Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"  ✅ 项目根目录: {ROOT}")
    print(f"  {'✅' if Path(args.config).is_dir() else '❌'} 配置目录: {args.config}")
    print(f"  {'✅' if Path(args.data).is_dir() else '❌'} 数据目录: {args.data}")
    print()


# ===== 配置模板 =====

PERSONA_DEFAULT = """\
# 人设配置
name: ""
identity: "你是用户的专属AI伴侣。"
traits: ["温柔", "贴心", "善解人意"]
suppress_tendency: 0.3
emotional_sensitivity: 0.5
"""

MEMORY_DEFAULT_YAML = """\
# 记忆系统配置
short_term_window: 20
recall_top_k: 5
fts_top_n: 15
vec_top_n: 15
rrf_k: 60
consolidation_interval: 3600
forgetting_lambda: 0.01
embedding_dim: 768
"""

PROACTIVE_DEFAULT_YAML = """\
# 主动陪伴配置
enabled: true
max_per_day: 5
quiet_hours: [22, 7]
silence_threshold_days: 3
"""


if __name__ == "__main__":
    main()
