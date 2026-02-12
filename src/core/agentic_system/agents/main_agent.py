"""Main agent."""

from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from string import Template

from pydantic_ai import Agent, RunContext
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.agentic_system.utils import get_chat_model
from src.core.rag_config import AGENT_SETTINGS
from src.schemas.retrieval import RetrievalFilters, RetrievedChunk
from src.services.prompt_service import PromptService
from src.services.retrieval_service import RetrievalService


@dataclass(frozen=True, slots=True)
class MainAgentDependencies:
    """Main agent dependencies."""

    user_name: str
    session: AsyncSession
    redis: Redis


def register_main_agent(
    agent: Agent[MainAgentDependencies, str],
) -> Agent[MainAgentDependencies, str]:
    """Connect the main agent to instructions prefix and tools."""

    @agent.instructions
    async def main_agent_instructions(
        ctx: RunContext[MainAgentDependencies],
    ) -> str:
        """Main agent instructions."""
        system_prompt = await PromptService.get_cached_content(
            session=ctx.deps.session,
            redis=ctx.deps.redis,
            slug=AGENT_SETTINGS.instruction_slug,
        )
        return Template(system_prompt).safe_substitute(
            user_name=ctx.deps.user_name,
            date_time=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
        )

    @agent.tool
    async def retrieve_chunks(
        ctx: RunContext[MainAgentDependencies],
        query: str,
        filters: RetrievalFilters | None = None,
        use_prev_next: bool | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant document chunks invoking the relative service."""
        service = RetrievalService(session=ctx.deps.session)
        return await service.retrieve(
            query=query,
            filters=filters,
            use_prev_next=use_prev_next,
        )

    return agent


@lru_cache(maxsize=1)
def get_main_agent() -> Agent[MainAgentDependencies, str]:
    """Return the main agent."""
    agent = Agent[MainAgentDependencies, str](
        name=AGENT_SETTINGS.name,
        model=get_chat_model(AGENT_SETTINGS.model_name),
        deps_type=MainAgentDependencies,
    )
    return register_main_agent(agent)
