"""Tests for thread history APIs and persistence."""

import pytest
from httpx import AsyncClient
from pydantic_ai.messages import ModelRequest, ModelResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.v1.llm import get_redis
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
async def test_thread_history_roundtrip(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist messages to a user thread and delete it."""
    password = "SecurePass123"
    user = User(
        name="History User",
        email="history@example.com",
        hashed_password=hash_password(password),
    )
    db_session.add(user)
    await db_session.commit()

    login_response = await client.post(
        "/v1/auth/token",
        json={"email": user.email, "password": password},
    )
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    create_response = await client.post(
        "/v1/threads",
        headers=headers,
        json={"title": "Dark matter"},
    )
    assert create_response.status_code == 201
    thread_id = create_response.json()["id"]

    call_count = 0

    async def fake_stream(
        self: LLMService,
        user: User,
        user_prompt: str,
        message_history=None,  # noqa: ANN001
    ):
        nonlocal call_count
        call_count += 1

        assert user.email == "history@example.com"
        if call_count == 1:
            assert user_prompt == "What is dark matter?"
            assert message_history == []
            yield LLMStreamEvent(kind="delta", value="Dark matter ")
            yield LLMStreamEvent(kind="delta", value="is unseen mass.")
            return

        assert call_count == 2
        assert user_prompt == "Why is it inferred?"
        assert message_history is not None
        assert len(message_history) == 2
        assert isinstance(message_history[0], ModelRequest)
        assert isinstance(message_history[1], ModelResponse)
        assert getattr(message_history[0].parts[0], "content", None) == "What is dark matter?"
        assert message_history[1].text == "Dark matter is unseen mass."
        yield LLMStreamEvent(kind="delta", value="From gravitational effects.")

    async def fake_redis() -> _FakeRedis:
        return _FakeRedis()

    monkeypatch.setattr(LLMService, "run_agent_stream", fake_stream)
    app.dependency_overrides[get_redis] = fake_redis

    try:
        stream_response = await client.post(
            "/v1/llm/chat/stream",
            headers=headers,
            json={
                "thread_id": thread_id,
                "messages": [{"role": "user", "content": "What is dark matter?"}],
            },
        )
        stream_response_second = await client.post(
            "/v1/llm/chat/stream",
            headers=headers,
            json={
                "thread_id": thread_id,
                "messages": [{"role": "user", "content": "Why is it inferred?"}],
            },
        )
    finally:
        app.dependency_overrides.pop(get_redis, None)

    assert stream_response.status_code == 200
    assert stream_response_second.status_code == 200

    list_response = await client.get("/v1/threads", headers=headers)
    assert list_response.status_code == 200
    threads = list_response.json()
    assert len(threads) == 1
    assert threads[0]["id"] == thread_id
    assert [message["role"] for message in threads[0]["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert threads[0]["messages"][0]["content"] == "What is dark matter?"
    assert threads[0]["messages"][1]["content"] == "Dark matter is unseen mass."
    assert threads[0]["messages"][2]["content"] == "Why is it inferred?"
    assert threads[0]["messages"][3]["content"] == "From gravitational effects."

    delete_response = await client.delete(f"/v1/threads/{thread_id}", headers=headers)
    assert delete_response.status_code == 204

    empty_list_response = await client.get("/v1/threads", headers=headers)
    assert empty_list_response.status_code == 200
    assert empty_list_response.json() == []
