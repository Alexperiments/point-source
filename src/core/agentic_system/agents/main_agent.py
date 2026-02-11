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
extra_body: dict[str, Any] = {
    "chat_template_kwargs": {
        "enable_thinking": AGENT_SETTINGS.enable_thinking,
    },
}

custom_provider = AGENT_SETTINGS.custom_llm_provider
if not custom_provider and AGENT_SETTINGS.model_name.startswith("custom/"):
    custom_provider = "openai"
if custom_provider:
    extra_body["custom_llm_provider"] = custom_provider

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
    with logfire.span("agent.system_prompt"):
        template = await PromptService.get_cached_content(
            session=ctx.deps.session,
            redis=ctx.deps.redis,
            slug=AGENT_SETTINGS.instruction_slug,
        )
        rendered = _render_prompt(
            template,
            user_name=ctx.deps.user_name,
            date_time=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
        )
        logfire.info(
            "agent.system_prompt.rendered",
            prompt=rendered,
            prompt_len=len(rendered),
        )
        return rendered


def _render_prompt(template: str, **values: str) -> str:
    """Render prompt placeholders without interpreting other braces."""
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{key}}}", value)
    return rendered


@main_agent.tool
async def retrieve_chunks(
    ctx: RunContext[MainAgentDependencies],
    query: str,
    filters: RetrievalFilters | None = None,
    use_prev_next: bool | None = None,
) -> list[RetrievedChunk]:
    """Retrieve relevant document chunks for factual or research queries."""
    service = RetrievalService(session=ctx.deps.session, redis=ctx.deps.redis)
    filters_payload = (
        filters.model_dump(mode="json", exclude_none=True) if filters else None
    )
    with logfire.span(
        "agent.tool.retrieve_chunks",
        query_preview=" ".join(query.split())[:200],
        query_len=len(query),
        filters=filters_payload,
        use_prev_next=use_prev_next,
    ):
        results = await service.retrieve(
            query=query,
            filters=filters,
            use_prev_next=use_prev_next,
        )
        with logfire.span(
            "agent.tool.retrieve_chunks.to_llm",
            chunk_count=len(results),
        ):
            for idx, chunk in enumerate(results, start=1):
                logfire.info(
                    "agent.tool.retrieve_chunks.chunk",
                    index=idx,
                    chunk=repr(chunk),
                )
        return results
