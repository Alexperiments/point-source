"""Redis-backed fixed-window rate limiting helpers."""

import hashlib
from dataclasses import dataclass

from fastapi import HTTPException, Request, status
from loguru import logger
from redis.asyncio import Redis
from redis.exceptions import RedisError


@dataclass(frozen=True, slots=True)
class FixedWindowRateLimit:
    """Single fixed-window rate limit rule."""

    bucket: str
    limit: int
    window_seconds: int
    error_message: str


def normalize_rate_limit_identifier(identifier: str) -> str:
    """Canonicalize identifiers before hashing them into Redis keys."""
    return identifier.strip().lower()


def get_client_ip(request: Request) -> str:
    """Resolve the best available client IP from proxy headers or the socket."""
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if forwarded_for:
        return forwarded_for.split(",", maxsplit=1)[0].strip()

    for header_name in ("cf-connecting-ip", "x-real-ip"):
        header_value = request.headers.get(header_name, "").strip()
        if header_value:
            return header_value

    if request.client is not None and request.client.host:
        return request.client.host

    return "unknown"


def _build_rate_limit_key(bucket: str, identifier: str) -> str:
    normalized = normalize_rate_limit_identifier(identifier)
    identifier_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"rate_limit:{bucket}:{identifier_hash}"


async def enforce_fixed_window_rate_limit(
    redis: Redis,
    rule: FixedWindowRateLimit,
    identifier: str,
) -> None:
    """Increment the Redis counter and raise when the configured limit is exceeded."""
    key = _build_rate_limit_key(rule.bucket, identifier)

    try:
        created = await redis.set(key, "1", ex=rule.window_seconds, nx=True)
        if created:
            return

        count = await redis.incr(key)
        retry_after = await redis.ttl(key)
        if retry_after is None or retry_after < 0:
            await redis.expire(key, rule.window_seconds)
            retry_after = rule.window_seconds
    except RedisError:
        logger.exception("Failed to enforce rate limit for bucket {}", rule.bucket)
        return

    if count <= rule.limit:
        return

    headers: dict[str, str] = {}
    if retry_after > 0:
        headers["Retry-After"] = str(retry_after)

    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=rule.error_message,
        headers=headers or None,
    )
