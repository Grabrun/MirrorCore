"""
镜核 (Mirror Core) 内部事件总线

提供进程内异步发布/订阅机制，所有模块通过事件类型解耦通信。

特性：
- 按 EventType 发布事件，订阅者按注册顺序依次执行
- 单个处理器异常不中断后续处理器，异常被完整记录
- 基于 event_id 的 LRU 幂等性校验，防止重复消费
- 优雅关闭：阻止新事件，等待进行中的处理完成后退出

额外公共接口（扩展自设计文档原定契约）：
- unsubscribe(): 取消注册指定处理器
- subscriber_count(): 查询指定或全部类型的事件订阅者数量
- wait_closed(): 等待总线完全关闭（shutdown 完成后返回）
- is_shutting_down: 关闭状态只读属性

架构说明 — 幂等性持久化层：
设计文档 §3.3.1.3 中定义了 processed_events SQLite 持久化表，
用于系统重启后防止日志重放时的重复消费。该持久化层将在
DB-T02（数据库迁移脚本）中实现，本模块仅提供运行时内存 LRU 层。
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger("mirror_core.bus")


class EventType(Enum):
    """事件类型枚举"""
    USER_MESSAGE = "user_message"
    SYSTEM_TICK = "system_tick"
    ANNIVERSARY_CHECK = "anniversary_check"
    PROACTIVE_CHANCE = "proactive_chance"
    EMOTION_CHANGED = "emotion_changed"
    STATE_TRANSITION = "state_transition"
    DECISION_ACTION = "decision_action"
    REPLY_READY = "reply_ready"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True)
class BaseEvent:
    """基础事件数据类，所有事件的基类"""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    type: EventType = EventType.ERROR
    timestamp: float = field(default_factory=_time.time)
    source: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)


class _LRUSet:
    """基于 OrderedDict 的 LRU 集合，用于幂等性校验"""

    def __init__(self, maxsize: int = 1000):
        self._dict: OrderedDict[str, None] = OrderedDict()
        self._maxsize = maxsize

    def add(self, item: str) -> None:
        self._dict[item] = None
        self._dict.move_to_end(item)
        if len(self._dict) > self._maxsize:
            self._dict.popitem(last=False)

    def __contains__(self, item: str) -> bool:
        """检查并更新访问顺序（LRU 语义）"""
        if item in self._dict:
            self._dict.move_to_end(item)
            return True
        return False

    def clear(self) -> None:
        self._dict.clear()

    def __len__(self) -> int:
        return len(self._dict)


# 处理器类型：接收 BaseEvent 的异步函数
Handler = Callable[[BaseEvent], Coroutine[Any, Any, None]]


class EventBus:
    """
    内部事件总线

    使用方式：
        bus = EventBus()
        bus.subscribe(EventType.USER_MESSAGE, my_handler)
        await bus.publish(BaseEvent(type=EventType.USER_MESSAGE, ...))
        await bus.shutdown()

    Args:
        idempotent_cache_size: 幂等性 LRU 缓存的最大条目数（默认 1000）
        shutdown_timeout: shutdown() 等待进行中处理器的超时秒数（默认 10.0）
    """

    def __init__(
        self,
        idempotent_cache_size: int = 1000,
        shutdown_timeout: float = 10.0,
    ):
        self._subscribers: Dict[EventType, List[Handler]] = {}
        self._processed_ids = _LRUSet(maxsize=idempotent_cache_size)
        self._shutdown_timeout = shutdown_timeout
        self._shutting_down = False
        self._shutdown_event = asyncio.Event()
        self._running_count = 0

    # ---- 订阅管理 ----

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        """
        注册一个异步处理器到指定事件类型。

        Args:
            event_type: 要订阅的事件类型
            handler: 异步回调函数，接收 BaseEvent 参数

        Raises:
            TypeError: handler 不是协程函数
        """
        if not asyncio.iscoroutinefunction(handler):
            raise TypeError(
                f"handler 必须是协程函数 (async def)，got {type(handler).__name__}"
            )
        self._subscribers.setdefault(event_type, []).append(handler)
        logger.debug(
            "订阅注册",
            extra={"event_type": event_type.value, "handler": handler.__name__},
        )

    def unsubscribe(self, event_type: EventType, handler: Handler) -> bool:
        """
        取消注册一个处理器。

        Returns:
            是否成功移除
        """
        handlers = self._subscribers.get(event_type)
        if handlers and handler in handlers:
            handlers.remove(handler)
            if not handlers:
                del self._subscribers[event_type]
            logger.debug(
                "取消订阅",
                extra={"event_type": event_type.value, "handler": handler.__name__},
            )
            return True
        return False

    def subscriber_count(self, event_type: Optional[EventType] = None) -> int:
        """
        返回指定事件类型或所有类型的订阅者数量。

        Args:
            event_type: 事件类型，为 None 时返回全局总数
        """
        if event_type:
            return len(self._subscribers.get(event_type, []))
        return sum(len(hs) for hs in self._subscribers.values())

    # ---- 事件发布 ----

    async def publish(self, event: BaseEvent) -> None:
        """
        发布事件，按类型依次通知所有订阅者（顺序执行，保证因果顺序）。

        特性：
        - 如果 shutdown() 已调用，新事件将被静默忽略
        - 重复 event_id 的事件会被幂等性校验跳过
        - 单个处理器异常不会中断后续处理器

        Args:
            event: 要发布的事件
        """
        if self._shutting_down:
            logger.warning(
                "总线正在关闭，新事件被忽略",
                extra={"event_type": event.type.value, "event_id": event.event_id},
            )
            return

        # 幂等性校验
        if event.event_id in self._processed_ids:
            logger.debug(
                "重复事件已跳过",
                extra={"event_type": event.type.value, "event_id": event.event_id},
            )
            return

        handlers = self._subscribers.get(event.type, [])
        if not handlers:
            return

        # 先标记再执行（防止并发重复事件）
        self._processed_ids.add(event.event_id)

        # 顺序执行所有处理器（保证因果顺序）
        for handler in handlers:
            if self._shutting_down:
                logger.warning(
                    "关闭中，中断后续处理器",
                    extra={"handler": handler.__name__, "event_id": event.event_id},
                )
                break

            self._running_count += 1
            try:
                await handler(event)
            except asyncio.CancelledError:
                logger.info(
                    "处理器被取消",
                    extra={"handler": handler.__name__, "event_id": event.event_id},
                )
            except Exception:
                logger.exception(
                    "事件处理异常",
                    extra={
                        "handler": handler.__name__,
                        "event_type": event.type.value,
                        "event_id": event.event_id,
                    },
                )
                # 异常不中断后续处理器
            finally:
                self._running_count -= 1

    # ---- 优雅关闭 ----

    async def shutdown(self) -> None:
        """
        优雅关闭总线。

        流程：
        1. 设置关闭标志，阻止新事件处理
        2. 发布 SHUTDOWN 事件（紧急通知所有订阅者）
        3. 等待所有正在执行的处理器完成（最多 shutdown_timeout 秒）
        4. 清理资源
        """
        if self._shutting_down:
            return

        self._shutting_down = True

        # 发布 SHUTDOWN 事件：直接顺序执行，不经过幂等
        shutdown_event = BaseEvent(
            type=EventType.SHUTDOWN,
            source="event_bus",
            payload={"reason": "graceful_shutdown"},
        )
        sh_handlers = self._subscribers.get(EventType.SHUTDOWN, [])
        for handler in sh_handlers:
            self._running_count += 1
            try:
                await handler(shutdown_event)
            except Exception:
                logger.exception(
                    "SHUTDOWN 处理器异常",
                    extra={"handler": handler.__name__},
                )
            finally:
                self._running_count -= 1

        # 等待进行中的非 SHUTDOWN 处理器完成
        if self._running_count > 0:
            deadline = asyncio.get_event_loop().time() + self._shutdown_timeout
            while self._running_count > 0:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    logger.warning(
                        "超时，%d 个处理器未完成",
                        self._running_count,
                    )
                    break
                await asyncio.sleep(0.01)

        self._processed_ids.clear()
        self._shutdown_event.set()

        logger.info("事件总线已优雅关闭")

    async def wait_closed(self) -> None:
        """等待总线完全关闭（shutdown 完成后返回）"""
        await self._shutdown_event.wait()

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down
