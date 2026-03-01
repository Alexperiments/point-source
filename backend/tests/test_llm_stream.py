"""Tests for LLM streaming endpoint framing."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.v1.llm import get_redis
from src.core.rag_config import AGENT_SETTINGS
from src.core.security import hash_password
from src.main import app
from src.models.user import User
from src.services.llm_service import LLMService, LLMStreamEvent


class _FakeRedis:
    def __init__(self) -> None:
        self._values: dict[str, int] = {}

    async def incr(self, key: str) -> int:
        value = self._values.get(key, 0) + 1
        self._values[key] = value
        return value

    async def expire(self, key: str, _seconds: int) -> bool:
        return key in self._values

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_chat_stream_emits_valid_sse_framing(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure streaming endpoint uses valid SSE delimiters and status messages."""
    password = "SecurePass123"
    user = User(
        name="Stream User",
        email="stream@example.com",
        hashed_password=hash_password(password),
    )
    db_session.add(user)
    await db_session.commit()

    login_response = await client.post(
        "/v1/auth/token",
        json={"email": user.email, "password": password},
    )
    token = login_response.json()["access_token"]

    async def fake_stream(
        self: LLMService,
        user: User,
        user_prompt: str,
        message_history=None,  # noqa: ANN001
    ):
        assert user.email == "stream@example.com"
        assert user_prompt == "Hello stream"
        assert message_history == []
        yield LLMStreamEvent(kind="status", value="retrieving_documents")
        yield LLMStreamEvent(kind="delta", value="Hello")
        yield LLMStreamEvent(kind="delta", value=" world")

    async def fake_redis() -> _FakeRedis:
        return _FakeRedis()

    monkeypatch.setattr(LLMService, "run_agent_stream", fake_stream)
    app.dependency_overrides[get_redis] = fake_redis

    try:
        response = await client.post(
            "/v1/llm/chat/stream",
            headers={"Authorization": f"Bearer {token}"},
            json={"messages": [{"role": "user", "content": "Hello stream"}]},
        )
    finally:
        app.dependency_overrides.pop(get_redis, None)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    body = response.text
    assert "data: [DONE]\n\n" in body
    assert "data: [DONE]\\n\\n" not in body
    assert '"type": "status"' in body
    assert '"status": "retrieving_documents"' in body
    assert '"content": "Hello"' in body
    assert '"content": " world"' in body


@pytest.mark.asyncio
async def test_chat_stream_enforces_daily_limit_except_for_premium_users(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-premium users should be rate-limited daily while premium users bypass."""
    password = "SecurePass123"
    regular_user = User(
        name="Regular User",
        email="regular@example.com",
        hashed_password=hash_password(password),
    )
    premium_user = User(
        name="Premium User",
        email="premium@example.com",
        hashed_password=hash_password(password),
        is_premium=True,
    )
    db_session.add_all([regular_user, premium_user])
    await db_session.commit()

    regular_login = await client.post(
        "/v1/auth/token",
        json={"email": regular_user.email, "password": password},
    )
    regular_token = regular_login.json()["access_token"]

    premium_login = await client.post(
        "/v1/auth/token",
        json={"email": premium_user.email, "password": password},
    )
    premium_token = premium_login.json()["access_token"]

    async def fake_stream(
        self: LLMService,
        user: User,
        user_prompt: str,
        message_history=None,  # noqa: ANN001
    ):
        del self, user, user_prompt, message_history
        yield LLMStreamEvent(kind="delta", value="ok")

    redis_client = _FakeRedis()

    async def fake_redis() -> _FakeRedis:
        return redis_client

    monkeypatch.setattr(LLMService, "run_agent_stream", fake_stream)
    monkeypatch.setattr(AGENT_SETTINGS, "daily_message_limit", 1, raising=False)
    app.dependency_overrides[get_redis] = fake_redis

    try:
        regular_first = await client.post(
            "/v1/llm/chat/stream",
            headers={"Authorization": f"Bearer {regular_token}"},
            json={"messages": [{"role": "user", "content": "one"}]},
        )
        regular_second = await client.post(
            "/v1/llm/chat/stream",
            headers={"Authorization": f"Bearer {regular_token}"},
            json={"messages": [{"role": "user", "content": "two"}]},
        )
        premium_first = await client.post(
            "/v1/llm/chat/stream",
            headers={"Authorization": f"Bearer {premium_token}"},
            json={"messages": [{"role": "user", "content": "one"}]},
        )
        premium_second = await client.post(
            "/v1/llm/chat/stream",
            headers={"Authorization": f"Bearer {premium_token}"},
            json={"messages": [{"role": "user", "content": "two"}]},
        )
    finally:
        app.dependency_overrides.pop(get_redis, None)

    assert regular_first.status_code == 200
    assert regular_second.status_code == 429
    assert "Daily message limit reached (1)" in regular_second.json()["detail"]

    assert premium_first.status_code == 200
    assert premium_second.status_code == 200
