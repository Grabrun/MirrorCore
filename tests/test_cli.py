"""
CLI 入口 + 项目结构验证测试
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


class TestProjectStructure:
    """项目目录结构验证"""

    def test_cli_exists(self):
        """cli.py 存在"""
        cli = Path(__file__).parent.parent / "mirror_core" / "cli.py"
        assert cli.is_file(), "cli.py 不存在"

    def test_config_dir_exists(self):
        """config/ 目录存在"""
        d = Path(__file__).parent.parent / "config"
        assert d.is_dir(), "config/ 目录不存在"

    def test_config_has_persona(self):
        """config/persona.yaml 存在"""
        f = Path(__file__).parent.parent / "config" / "persona.yaml"
        assert f.is_file(), "config/persona.yaml 不存在"
        content = f.read_text(encoding="utf-8")
        assert "小镜" in content

    def test_skills_dir_exists(self):
        """skills/ 目录存在"""
        d = Path(__file__).parent.parent / "skills"
        assert d.is_dir(), "skills/ 目录不存在"

    def test_data_dir_exists(self):
        """data/ 目录存在"""
        d = Path(__file__).parent.parent / "data"
        assert d.is_dir(), "data/ 目录不存在"

    def test_pyproject_has_entry_point(self):
        """pyproject.toml 有 console_scripts 入口"""
        f = Path(__file__).parent.parent / "pyproject.toml"
        content = f.read_text(encoding="utf-8")
        assert "mirror = \"mirror_core.cli:main\"" in content


class TestCLI:
    """CLI 入口测试"""

    def test_cli_help(self):
        """--help 不崩溃"""
        from mirror_core.cli import main
        with patch.object(sys, "argv", ["mirror", "--help"]):
            with pytest.raises(SystemExit):
                main()

    def test_cli_init(self, tmp_path):
        """mirror init 创建配置文件"""
        from mirror_core.cli import cmd_init

        class Args:
            config = str(tmp_path / "config")
            data = str(tmp_path / "data")
            force = False

        cmd_init(Args())

        assert (tmp_path / "config" / "persona.yaml").is_file()
        assert (tmp_path / "data").is_dir()

    def test_cli_doctor(self, capsys):
        """mirror doctor 输出诊断信息"""
        from mirror_core.cli import cmd_doctor

        class Args:
            config = str(Path(__file__).parent.parent / "config")
            data = str(Path(__file__).parent.parent / "data")

        cmd_doctor(Args())
        captured = capsys.readouterr()
        assert "镜核" in captured.out
        assert "配置目录" in captured.out
