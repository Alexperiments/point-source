"""Main agent."""

from datetime import UTC, datetime
from typing import Any

import logfire
from attr import dataclass
from pydantic_ai import Agent, ModelSettings, RunContext
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.agentic_system.utils import get_chat_model
from src.core.rag_config import AGENT_SETTINGS
from src.schemas.retrieval import RetrievalFilters, RetrievedChunk
from src.services.prompt_service import PromptService
from src.services.retrieval_service import RetrievalService


@dataclass
class MainAgentDependencies:
    """Main agent dependencies."""

    user_name: str
    session: AsyncSession
    redis: Redis


_model_settings_kwargs = {
    "temperature": AGENT_SETTINGS.temperature,
    "max_tokens": AGENT_SETTINGS.max_tokens,
}
extra_body: dict[str, Any] = {}
if AGENT_SETTINGS.enable_thinking:
    extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = (
        AGENT_SETTINGS.enable_thinking
    )

custom_provider = AGENT_SETTINGS.custom_llm_provider
if not custom_provider and AGENT_SETTINGS.model_name.startswith("custom/"):
    custom_provider = "openai"
if custom_provider:
    extra_body["custom_llm_provider"] = custom_provider

if extra_body:
    _model_settings_kwargs["extra_body"] = extra_body

DEFAULT_MODEL_SETTINGS = ModelSettings(**_model_settings_kwargs)

main_agent = Agent[MainAgentDependencies, str](
    name=AGENT_SETTINGS.name,
    model=get_chat_model(AGENT_SETTINGS.model_name, DEFAULT_MODEL_SETTINGS),
    deps_type=MainAgentDependencies,
)


@main_agent.instructions
async def main_agent_instructions(
    ctx: RunContext[MainAgentDependencies],
) -> str:
    """Main agent instructions."""
    system_prompt = await PromptService.get_cached_content(
        session=ctx.deps.session,
        redis=ctx.deps.redis,
        slug=AGENT_SETTINGS.instruction_slug,
    )
    return system_prompt.format(
        user_name=ctx.deps.user_name,
        date_time=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
    )


@main_agent.tool
async def retrieve_chunks(
    ctx: RunContext[MainAgentDependencies],
    query: str,
    filters: RetrievalFilters | None = None,
    use_prev_next: bool | None = None,
) -> list[RetrievedChunk]:
    """Retrieve relevant document chunks for factual or research queries."""
    service = RetrievalService(session=ctx.deps.session, redis=ctx.deps.redis)
    with logfire.span("agent.tool.retrieve_chunks"):
        return await service.retrieve(
            query=query,
            filters=filters,
            use_prev_next=use_prev_next,
        )
