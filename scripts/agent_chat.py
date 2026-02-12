"""Minimal CLI to chat with the main agent."""

from __future__ import annotations

import argparse
import asyncio
import sys

import logfire

from src.core.agentic_system.agents.main_agent import (
    MainAgentDependencies,
    get_main_agent,
)
from src.core.config import PROJECT_INFO
from src.core.database.base import get_async_session
from src.core.database.redis import get_redis_pool


logfire.configure(
    service_name=PROJECT_INFO["name"],
    service_version=PROJECT_INFO["version"],
)
logfire.instrument_pydantic_ai()
MAIN_AGENT = get_main_agent()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with the main agent.",
    )
    parser.add_argument(
        "--prompt",
        help="Prompt text (or pipe via stdin).",
    )
    return parser.parse_args()


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return input("You: ").strip()


async def main() -> None:
    """Run the CLI chat loop for the main agent."""
    args = _parse_args()
    prompt = _read_prompt(args)
    if not prompt:
        return

    redis = await get_redis_pool()
    async for session in get_async_session():
        deps = MainAgentDependencies(
            user_name="agent_chat",
            session=session,
            redis=redis,
        )
        agent_run = await MAIN_AGENT.run(user_prompt=prompt, deps=deps)
        if agent_run.output:
            sys.stdout.write(f"{agent_run.output}\n")
            sys.stdout.flush()
        break

    if redis is not None:
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
