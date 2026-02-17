"""LLM service for running agents."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from pydantic_ai.usage import UsageLimits
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.agentic_system.agents.main_agent import (
    MainAgentDependencies,
    get_main_agent,
)
from src.core.rag_config import AGENT_SETTINGS
from src.models.user import User


class LLMServiceError(Exception):
    """Exception for LLM service errors."""


@dataclass(frozen=True, slots=True)
class LLMStreamEvent:
    """A stream event emitted while generating an assistant response."""

    kind: Literal["status", "delta"]
    value: str


class LLMService:
    """Service for running LLM agents."""

    def __init__(self, session: AsyncSession, redis: Redis) -> None:
        """Initialize the LLM service.

        Args:
            session: The database session to use.
            redis: The Redis client to use.

        """
        self.session = session
        self.redis = redis

    @staticmethod
    def _build_usage_limits() -> UsageLimits:
        kwargs: dict[str, int] = {
            "request_limit": AGENT_SETTINGS.request_limit,
        }
        if AGENT_SETTINGS.tool_calls_limit is not None:
            kwargs["tool_calls_limit"] = AGENT_SETTINGS.tool_calls_limit
        return UsageLimits(**kwargs)

    async def run_agent(
        self,
        user: User,
        user_prompt: str,
    ) -> str:
        """Run the main agent with a user prompt.

        Args:
            user: The user running the agent.
            user_prompt: The user's prompt/query.

        Returns:
            The agent's response as a string.

        Raises:
            LLMServiceError: If the agent run fails.

        """
        try:
            agent_run = await get_main_agent().run(
                user_prompt=user_prompt,
                deps=MainAgentDependencies(
                    user_name=user.name or user.email,
                    session=self.session,
                    redis=self.redis,
                ),
                usage_limits=self._build_usage_limits(),
            )
        except Exception as e:
            raise LLMServiceError(f"Failed to run agent: {e!s}") from e
        else:
            return agent_run.output

    async def run_agent_stream(
        self,
        user: User,
        user_prompt: str,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Run the main agent and stream status + response token deltas.

        Args:
            user: The user running the agent.
            user_prompt: The user's prompt/query.

        Yields:
            `LLMStreamEvent` items for status transitions and text deltas.

        Raises:
            LLMServiceError: If the agent run fails.

        """
        queue: asyncio.Queue[LLMStreamEvent | Exception | None] = asyncio.Queue()

        async def emit_status(status: str) -> None:
            await queue.put(LLMStreamEvent(kind="status", value=status))

        async def produce() -> None:
            await emit_status("thinking")
            try:
                async with asyncio.timeout(AGENT_SETTINGS.stream_timeout_seconds):
                    async with get_main_agent().run_stream(
                        user_prompt=user_prompt,
                        deps=MainAgentDependencies(
                            user_name=user.name or user.email,
                            session=self.session,
                            redis=self.redis,
                            status_callback=emit_status,
                        ),
                        usage_limits=self._build_usage_limits(),
                    ) as agent_run:
                        async for token in agent_run.stream_text(delta=True):
                            await queue.put(LLMStreamEvent(kind="delta", value=token))
            except TimeoutError:
                await queue.put(
                    LLMServiceError(
                        "Agent stream timed out before producing a final response.",
                    ),
                )
            except Exception as e:  # noqa: BLE001
                await queue.put(e)
            finally:
                await queue.put(None)

        producer_task = asyncio.create_task(produce())

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    if isinstance(item, LLMServiceError):
                        raise item
                    raise LLMServiceError(
                        f"Failed to run agent stream: {item!s}",
                    ) from item
                yield item
        finally:
            await producer_task
