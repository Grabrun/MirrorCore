"""
聊天历史查询 API

B-T10: 实现聊天历史查询 API，从 conversation_turns 表读取数据
User Story 3：作为用户，我希望在 WebChat 端能回看聊天记录

验收标准：
1. 提供 GET /api/v1/chat/history?session_id=xxx&limit=20 接口
2. 按 timestamp 降序返回结构正确的历史消息，包含 role, content, timestamp

鉴权说明：
- 当前使用轻量 Bearer Token 鉴权（通过环境变量 MIRROR_API_TOKEN 配置）
- 未配置 TOKEN 时自动进入本地开发模式，跳过鉴权
- 待 B-T42（匿名 JWT 认证）实现后，应替换为 JWT 签名验证
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from mirror_core.infrastructure.database import Database

logger = logging.getLogger("mirror_core.api.chat")

# ---- 鉴权依赖（轻量版，待 B-T42 替换为 JWT） ----


async def verify_token(authorization: Optional[str] = Header(None)) -> None:
    """
    Bearer Token 鉴权依赖。

    环境变量 MIRROR_API_TOKEN 配置令牌：
    - 未设置 → 本地开发模式，跳过鉴权
    - 已设置 → 校验 Authorization: Bearer <token>
    """
    expected_token = os.environ.get("MIRROR_API_TOKEN")
    if not expected_token:
        # 本地开发模式：无 token 配置时跳过鉴权
        return

    if not authorization:
        raise HTTPException(status_code=401, detail="缺少认证头")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="认证格式错误，需使用 Bearer <token>")
    actual_token = authorization.removeprefix("Bearer ").strip()
    if not actual_token:
        raise HTTPException(status_code=401, detail="Token 为空")
    if actual_token != expected_token:
        raise HTTPException(status_code=403, detail="Token 无效")


# ---- API 路由 ----

def create_chat_router(db: Database) -> APIRouter:
    """
    创建聊天历史 REST API 路由。

    Args:
        db: 数据库实例，用于查询 conversation_turns 表
    """
    router = APIRouter(prefix="/api/v1/chat", dependencies=[Depends(verify_token)])

    @router.get("/history")
    async def get_chat_history(
        session_id: str = Query(..., description="会话 ID"),
        user_id: str = Query(..., description="用户 ID（数据隔离校验）"),
        limit: int = Query(20, ge=1, le=200, description="返回消息数量上限"),
        before: Optional[float] = Query(
            None, description="可选：只返回早于此时间戳的消息（用于滚动分页）"
        ),
    ):
        """
        获取会话聊天历史。

        按 timestamp 降序排列，最新的消息在前。
        支持基于 before 参数的分页查询。
        通过 user_id + session_id 双重过滤实现数据隔离。
        """
        try:
            # 使用 user_id 做数据隔离：只返回属于该用户的消息
            conditions = "WHERE session_id = ? AND user_id = ?"
            params: list = [session_id, user_id]

            if before is not None:
                conditions += " AND timestamp < ?"
                params.append(before)

            rows = await db.fetch_all(
                f"""
                SELECT role, content, timestamp
                FROM conversation_turns
                {conditions}
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (*params, limit),
            )

            return {
                "messages": [
                    {
                        "role": row["role"],
                        "content": row["content"],
                        "timestamp": row["timestamp"],
                    }
                    for row in rows
                ]
            }

        except RuntimeError:
            logger.error("数据库未初始化", extra={"session_id": session_id})
            raise HTTPException(status_code=503, detail="服务暂不可用")
        except Exception:
            logger.exception(
                "查询聊天历史失败",
                extra={"session_id": session_id},
            )
            raise HTTPException(
                status_code=500,
                detail="查询聊天历史失败",
            )

    return router
