"""
轻量级 JWT 认证模块

提供无外部依赖的 JWT 令牌签发与验证（HMAC-SHA256），
用于 WebChat 适配器的 WebSocket 认证。

B-T42: POST /api/v1/auth/anonymous 匿名认证
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time as _time
import uuid
from typing import Optional

logger = logging.getLogger("mirror_core.gateway.auth")


def _base64url_encode(data: bytes) -> str:
    """Base64url 编码（无填充）"""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _base64url_decode(s: str) -> bytes:
    """Base64url 解码（自动补齐填充）"""
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def generate_secret() -> str:
    """生成随机密钥"""
    return uuid.uuid4().hex + uuid.uuid4().hex


def generate_token(user_id: str, secret: str, expire_days: int = 7) -> str:
    """
    签发 JWT 令牌。

    Args:
        user_id: 用户标识
        secret: HMAC 密钥
        expire_days: 过期天数（默认 7）

    Returns:
        JWT 字符串 (header.payload.signature)
    """
    header = _base64url_encode(
        json.dumps({"alg": "HS256", "typ": "JWT"}).encode()
    )
    now = int(_time.time())
    payload = _base64url_encode(
        json.dumps({
            "user_id": user_id,
            "exp": now + expire_days * 86400,
            "iat": now,
        }).encode()
    )
    signature_input = f"{header}.{payload}".encode()
    signature = _base64url_encode(
        hmac.new(secret.encode(), signature_input, hashlib.sha256).digest()
    )
    return f"{header}.{payload}.{signature}"


def verify_token(token: str, secret: str) -> Optional[str]:
    """
    验证 JWT 令牌并返回 user_id。

    Args:
        token: JWT 字符串
        secret: HMAC 密钥

    Returns:
        user_id（验证通过），None（验证失败或已过期）
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, payload_b64, sig_b64 = parts

        # 验证签名
        expected_sig = _base64url_encode(
            hmac.new(
                secret.encode(),
                f"{header_b64}.{payload_b64}".encode(),
                hashlib.sha256,
            ).digest()
        )
        if not hmac.compare_digest(sig_b64, expected_sig):
            logger.warning("JWT 签名不匹配")
            return None

        # 解码 payload
        payload_data = json.loads(_base64url_decode(payload_b64))
        user_id = payload_data.get("user_id")

        # 检查过期
        exp = payload_data.get("exp", 0)
        if exp < _time.time():
            logger.warning("JWT 已过期")
            return None

        if not user_id:
            logger.warning("JWT payload 缺少 user_id")
            return None

        return user_id

    except (json.JSONDecodeError, ValueError, IndexError, Exception):
        logger.exception("JWT 验证异常")
        return None


def create_anonymous_token_handler(secret: str) -> callable:
    """
    创建匿名认证 API 处理器。

    返回一个 FastAPI 路由处理器函数，用于 POST /api/v1/auth/anonymous。
    """
    from fastapi import APIRouter

    router = APIRouter(prefix="/api/v1/auth")

    @router.post("/anonymous")
    async def anonymous_auth():
        """匿名认证：生成 user_id 并签发 JWT 令牌"""
        user_id = f"u_{uuid.uuid4().hex[:16]}"
        token = generate_token(user_id, secret)
        logger.info("匿名认证", extra={"user_id": user_id})
        return {
            "user_id": user_id,
            "token": token,
            "expire_days": 7,
            "token_type": "Bearer",
        }

    return router
