"""LLM service for running agents."""

from collections.abc import AsyncIterator

import logfire
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.agentic_system.agents.main_agent import (
    MainAgentDependencies,
    main_agent,
)
from src.models.user import User


class LLMServiceError(Exception):
    """Exception for LLM service errors."""


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
            with logfire.span("agent.run"):
                agent_run = await main_agent.run(
                    user_prompt=user_prompt,
                    deps=MainAgentDependencies(
                        user_name=user.name or user.email,
                        session=self.session,
                        redis=self.redis,
                    ),
                )
        except Exception as e:
            raise LLMServiceError(f"Failed to run agent: {e!s}") from e
        else:
            return agent_run.output

    async def run_agent_stream(
        self,
        user: User,
        user_prompt: str,
    ) -> AsyncIterator[str]:
        """Run the main agent and stream its response tokens.

        Args:
            user: The user running the agent.
            user_prompt: The user's prompt/query.

        Yields:
            Token deltas as they are produced.

        Raises:
            LLMServiceError: If the agent run fails.

        """
        try:
            with logfire.span("agent.stream"):
                async with main_agent.run_stream(
                    user_prompt=user_prompt,
                    deps=MainAgentDependencies(
                        user_name=user.name or user.email,
                        session=self.session,
                        redis=self.redis,
                    ),
                ) as agent_run:
                    async for token in agent_run.stream_text(delta=True):
                        yield token
        except Exception as e:
            raise LLMServiceError(f"Failed to run agent stream: {e!s}") from e
