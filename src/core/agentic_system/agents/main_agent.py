"""Main agent."""

from datetime import UTC, datetime

from attr import dataclass
from pydantic_ai import Agent, ModelSettings, RunContext
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.agentic_system.utils import get_chat_model
from src.schemas.retrieval import RetrievalFilters, RetrievedChunk
from src.services.prompt_service import PromptService
from src.services.retrieval_service import RetrievalService


@dataclass
class MainAgentDependencies:
    """Main agent dependencies."""

    user_name: str
    session: AsyncSession
    redis: Redis


DEFAULT_MODEL_SETTINGS = ModelSettings(
    temperature=0.3,
    max_tokens=2048,
    # Qwen3 supports disabling thinking via chat_template_kwargs on OpenAI-compatible servers.
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

main_agent = Agent[MainAgentDependencies, str](
    name="Main Agent",
    model=get_chat_model("qwen3-8b-mlx-6bit", DEFAULT_MODEL_SETTINGS),
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
        slug="main_agent_instructions",
    )
    base_prompt = system_prompt.format(
        user_name=ctx.deps.user_name,
        date_time=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
    )
    return (
        f"{base_prompt}\n\n"
        "/no_think\n"
        "Respond with the final answer only. Do not include chain-of-thought, "
        "analysis, or process notes."
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
    return await service.retrieve(
        query=query,
        filters=filters,
        use_prev_next=use_prev_next,
    )
