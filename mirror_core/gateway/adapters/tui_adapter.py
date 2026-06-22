"""
TUI 终端文本界面适配器

B-T07: 基于 textual 或 prompt_toolkit 的终端文本界面适配器

适合本地运行，非图形渠道自动降级图片为标签文本。
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
from typing import Callable, Optional

from mirror_core.gateway.base import ChannelAdapter, MessageContent, RawMessage

logger = logging.getLogger("mirror_core.gateway.adapters.tui")


class TuiAdapter(ChannelAdapter):
    """
    TUI 终端文本界面适配器

    提供简单的命令行输入/输出循环，消息显示在终端。
    图片表情包降级为标签文本显示。

    Args:
        ingress_callback: Gateway.ingress 回调
        user_id: TUI 用户的固定 ID（调试用），默认 "tui_user"
    """

    def __init__(
        self,
        ingress_callback: Callable,
        user_id: str = "tui_user",
    ):
        self._ingress = ingress_callback
        self._user_id = user_id
        self._running = False
        self._input_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """启动 TUI：开始监听终端输入"""
        self._running = True
        self._input_task = asyncio.create_task(self._input_loop())
        logger.info("TUI 适配器已启动")

    async def _input_loop(self) -> None:
        """异步输入循环（使用 executor 避免阻塞事件循环）"""
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                text = await loop.run_in_executor(None, self._read_line)
                if text and text.strip():
                    raw_msg = RawMessage(
                        platform="tui",
                        platform_user_id=self._user_id,
                        text=text.strip(),
                        timestamp=_time.time(),
                    )
                    await self._ingress(raw_msg)
            except EOFError:
                break
            except Exception:
                logger.exception("TUI 输入读取异常")

    @staticmethod
    def _read_line() -> str:
        """读取一行终端输入"""
        try:
            return input("> ")
        except (EOFError, KeyboardInterrupt):
            return ""

    async def stop(self) -> None:
        """停止 TUI"""
        self._running = False
        if self._input_task:
            self._input_task.cancel()
            try:
                await self._input_task
            except (asyncio.CancelledError, Exception):
                pass
            self._input_task = None
        logger.info("TUI 适配器已停止")

    async def send_message(self, target_id: str, content: MessageContent) -> bool:
        """
        在终端显示消息。

        图片自动降级为标签文本 [表情: 标签名]。
        """
        try:
            display = content.display_text
            print(f"[{self.platform_name}] {display}")
            return True
        except Exception:
            logger.exception("TUI 消息显示失败")
            return False

    @property
    def platform_name(self) -> str:
        return "tui"

    @property
    def status(self) -> str:
        return "connected" if self._running else "disconnected"
