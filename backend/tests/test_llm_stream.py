"""Tests for LLM streaming endpoint framing."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.v1.llm import get_redis
from src.core.security import hash_password
from src.main import app
from src.models.user import User
from src.services.llm_service import LLMService


class _FakeRedis:
    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_chat_stream_emits_valid_sse_framing(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure streaming endpoint uses real SSE line delimiters."""
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
    ):
        assert user.email == "stream@example.com"
        assert user_prompt == "Hello stream"
        yield "Hello"
        yield " world"

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
    assert '"content": "Hello"' in body
    assert '"content": " world"' in body
