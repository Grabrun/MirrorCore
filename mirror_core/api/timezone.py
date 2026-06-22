"""
时区上报 API

B-T44: POST /api/v1/user/timezone
补充文档 v1.1: 用户时区数据来源方案
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from mirror_core.infrastructure.database import Database

logger = logging.getLogger("mirror_core.api.timezone")


def create_timezone_router(db: Database) -> APIRouter:
    """
    创建时区上报 REST API 路由。

    补充文档 v1.1 变更 1:
    用户时区优先级: persona.yaml > POST /api/v1/user/timezone > 系统时区
    """
    router = APIRouter(prefix="/api/v1/user")

    @router.post("/timezone")
    async def report_timezone(
        user_id: str,
        timezone: str,
    ):
        """
        接收并存储用户时区信息。

        以 fact_type='preference', key='timezone' 格式存入 fact_memory 表。
        """
        if not user_id or not timezone:
            raise HTTPException(status_code=400, detail="缺少 user_id 或 timezone")

        try:
            await db.execute(
                """
                INSERT OR REPLACE INTO fact_memory
                    (user_id, fact_type, key, value, confidence, last_updated)
                VALUES (?, ?, ?, ?, 1.0, unixepoch())
                """,
                (user_id, "preference", "timezone", timezone),
            )
            logger.info("时区已记录: user=%s, tz=%s", user_id, timezone)
            return {"status": "ok", "user_id": user_id, "timezone": timezone}
        except Exception:
            logger.exception("时区记录失败", extra={"user_id": user_id})
            raise HTTPException(status_code=500, detail="时区记录失败")

    return router
