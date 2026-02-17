"""LLM API."""

import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from src.api.v1.auth import get_current_user
from src.core.database.base import get_async_session
from src.core.database.redis import get_redis_pool
from src.models.user import User
from src.schemas.llm import LLMRequest, LLMResponse, LLMStreamRequest
from src.services.llm_service import LLMService, LLMServiceError


router = APIRouter()


async def get_redis() -> Redis:
    """Get Redis client dependency."""
    return await get_redis_pool()


def _resolve_stream_prompt(request: LLMStreamRequest) -> str:
    """Resolve latest user prompt from stream payload."""
    if request.prompt:
        prompt = request.prompt.strip()
        if prompt:
            return prompt

    for message in reversed(request.messages):
        if message.role != "user":
            continue
        prompt = message.content.strip()
        if prompt:
            return prompt

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Missing user prompt.",
    )


@router.post("/chat", response_model=LLMResponse)
async def chat(
    request: LLMRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_async_session)],
    redis: Annotated[Redis, Depends(get_redis)],
) -> LLMResponse:
    """Chat with the LLM agent."""
    try:
        llm_service = LLMService(session=db, redis=redis)
        response = await llm_service.run_agent(
            user=current_user,
            user_prompt=request.prompt,
        )
        return LLMResponse(response=response)
    except LLMServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
    finally:
        await redis.aclose()


@router.post("/chat/stream")
async def chat_stream(
    request: LLMStreamRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_async_session)],
    redis: Annotated[Redis, Depends(get_redis)],
) -> StreamingResponse:
    """Stream chat response in OpenAI-like SSE payload format."""
    prompt = _resolve_stream_prompt(request)

    async def event_stream() -> AsyncIterator[str]:
        try:
            llm_service = LLMService(session=db, redis=redis)
            async for token in llm_service.run_agent_stream(
                user=current_user,
                user_prompt=prompt,
            ):
                payload = {
                    "choices": [
                        {
                            "delta": {
                                "content": token,
                            },
                        },
                    ],
                }
                yield f"data: {json.dumps(payload)}\n\n"

            yield "data: [DONE]\n\n"
        except LLMServiceError:
            payload = {
                "choices": [
                    {
                        "delta": {
                            "content": "\\n\\n[Error: failed to stream response.]",
                        },
                    },
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            await redis.aclose()

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=headers,
    )
