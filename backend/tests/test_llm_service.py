"""Tests for LLM service agent execution limits and timeout behavior."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from src.core.rag_config import AGENT_SETTINGS
from src.models.user import User
from src.services.llm_service import LLMService, LLMServiceError


class _FakeRedis:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self._values = values or {}

    async def get(self, key: str) -> str | None:
        return self._values.get(key)

    async def incr(self, key: str) -> int:
        next_value = int(self._values.get(key, "0")) + 1
        self._values[key] = str(next_value)
        return next_value

    async def expire(self, key: str, _seconds: int) -> bool:
        return key in self._values


class _FakeStreamRun:
    def __init__(self, tokens: list[str], *, delay_seconds: float = 0.0) -> None:
        self._tokens = tokens
        self._delay_seconds = delay_seconds

    async def __aenter__(self) -> _FakeStreamRun:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def stream_text(self, *, delta: bool = True):  # noqa: ANN001
        del delta
        if self._delay_seconds > 0:
            await asyncio.sleep(self._delay_seconds)
        for token in self._tokens:
            yield token


class _FakeAgent:
    def __init__(self, stream_run: _FakeStreamRun) -> None:
        self._stream_run = stream_run
        self.run_kwargs: dict[str, object] = {}
        self.run_stream_kwargs: dict[str, object] = {}

    async def run(self, **kwargs):  # noqa: ANN003
        self.run_kwargs = kwargs
        return SimpleNamespace(output="ok")

    def run_stream(self, **kwargs):  # noqa: ANN003
        self.run_stream_kwargs = kwargs
        return self._stream_run


def _test_user() -> User:
    return User(
        name="Test User",
        email="test@example.com",
        hashed_password="hashed",
    )


@pytest.mark.asyncio
async def test_run_agent_passes_usage_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLMService.run_agent should apply configured usage limits."""
    fake_agent = _FakeAgent(_FakeStreamRun(tokens=[]))
    monkeypatch.setattr(
        "src.services.llm_service.get_main_agent",
        lambda: fake_agent,
    )

    service = LLMService(session=None, redis=None)  # type: ignore[arg-type]
    output = await service.run_agent(user=_test_user(), user_prompt="hello")

    assert output == "ok"
    usage_limits = fake_agent.run_kwargs["usage_limits"]
    assert getattr(usage_limits, "request_limit") == AGENT_SETTINGS.request_limit
    assert (
        getattr(usage_limits, "tool_calls_limit")
        == AGENT_SETTINGS.tool_calls_limit
    )


@pytest.mark.asyncio
async def test_get_daily_message_usage_for_regular_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Usage summary should expose daily cap, usage, and remaining requests."""
    user = _test_user()
    now = datetime.now(tz=UTC)
    key = f"daily_message_limit:{now.date().isoformat()}:{user.id}"
    redis = _FakeRedis({key: "2"})
    monkeypatch.setattr(AGENT_SETTINGS, "daily_message_limit", 3, raising=False)

    usage = await LLMService(session=None, redis=redis).get_daily_message_usage(user)  # type: ignore[arg-type]

    assert usage.is_premium is False
    assert usage.daily_message_limit == 3
    assert usage.requests_used == 2
    assert usage.requests_remaining == 1
    assert usage.reset_at.tzinfo is UTC
    assert usage.reset_in_seconds > 0


@pytest.mark.asyncio
async def test_get_daily_message_usage_for_premium_user() -> None:
    """Premium usage summary should expose unlimited rate limits."""
    user = _test_user()
    user.is_premium = True
    now = datetime.now(tz=UTC)
    key = f"daily_message_limit:{now.date().isoformat()}:{user.id}"
    redis = _FakeRedis({key: "7"})

    usage = await LLMService(session=None, redis=redis).get_daily_message_usage(user)  # type: ignore[arg-type]

    assert usage.is_premium is True
    assert usage.daily_message_limit is None
    assert usage.requests_used == 7
    assert usage.requests_remaining is None


@pytest.mark.asyncio
async def test_run_agent_stream_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLMService.run_agent_stream should fail fast on stream timeout."""
    fake_agent = _FakeAgent(_FakeStreamRun(tokens=[], delay_seconds=0.05))
    monkeypatch.setattr(
        "src.services.llm_service.get_main_agent",
        lambda: fake_agent,
    )
    monkeypatch.setattr(
        AGENT_SETTINGS,
        "stream_timeout_seconds",
        0.01,
        raising=False,
    )

    service = LLMService(session=None, redis=None)  # type: ignore[arg-type]
    events: list[tuple[str, str]] = []

    with pytest.raises(LLMServiceError, match="timed out"):
        async for event in service.run_agent_stream(
            user=_test_user(),
            user_prompt="hello",
        ):
            events.append((event.kind, event.value))

    assert events
    assert events[0] == ("status", "thinking")


@pytest.mark.asyncio
async def test_run_agent_stream_emits_deltas_and_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLMService.run_agent_stream should emit thinking + text deltas."""
    fake_agent = _FakeAgent(_FakeStreamRun(tokens=["Hello", " world"]))
    monkeypatch.setattr(
        "src.services.llm_service.get_main_agent",
        lambda: fake_agent,
    )

    service = LLMService(session=None, redis=None)  # type: ignore[arg-type]
    events = [
        (event.kind, event.value)
        async for event in service.run_agent_stream(
            user=_test_user(),
            user_prompt="hello",
        )
    ]

    assert events == [
        ("status", "thinking"),
        ("delta", "Hello"),
        ("delta", " world"),
    ]

    usage_limits = fake_agent.run_stream_kwargs["usage_limits"]
    assert getattr(usage_limits, "request_limit") == AGENT_SETTINGS.request_limit
    assert (
        getattr(usage_limits, "tool_calls_limit")
        == AGENT_SETTINGS.tool_calls_limit
    )
