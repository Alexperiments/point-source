"""LLM API."""

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
)
from redis.asyncio import Redis
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from src.api.v1.auth import get_current_user
from src.core.database.base import get_async_session
from src.core.database.redis import get_redis_pool
from src.core.rag_config import AGENT_SETTINGS
from src.models.message import Message, MessageRole
from src.models.thread import Thread
from src.models.user import User
from src.schemas.llm import LLMRequest, LLMResponse, LLMStreamRequest
from src.services.llm_service import (
    DailyMessageLimitExceededError,
    LLMService,
    LLMServiceError,
)


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


def _thread_title_from_prompt(prompt: str) -> str:
    title = prompt.strip().replace("\n", " ")
    if not title:
        return "New chat"
    return title[:40]


async def _resolve_thread(
    db: AsyncSession,
    current_user: User,
    request: LLMStreamRequest,
    prompt: str,
) -> Thread:
    if request.thread_id is None:
        thread = Thread(
            user_id=current_user.id,
            title=_thread_title_from_prompt(prompt),
        )
        db.add(thread)
        await db.flush()
        return thread

    result = await db.execute(
        select(Thread).where(
            Thread.id == request.thread_id,
            Thread.user_id == current_user.id,
        ),
    )
    thread = result.scalar_one_or_none()
    if thread is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found.",
        )
    return thread


async def _append_thread_message(
    db: AsyncSession,
    thread: Thread,
    role: MessageRole,
    content: str,
) -> None:
    next_seq_result = await db.execute(
        select(func.coalesce(func.max(Message.sequence_num), 0) + 1).where(
            Message.thread_id == thread.id,
        ),
    )
    next_sequence = int(next_seq_result.scalar_one())

    db.add(
        Message(
            thread_id=thread.id,
            sequence_num=next_sequence,
            role=role,
            content=content,
        ),
    )
    thread.updated_at = datetime.now(tz=UTC)
    await db.flush()


async def _load_thread_message_history(
    db: AsyncSession,
    thread: Thread,
) -> list[ModelMessage]:
    if AGENT_SETTINGS.history_max_messages <= 0:
        return []

    result = await db.execute(
        select(Message)
        .where(Message.thread_id == thread.id)
        .order_by(Message.sequence_num.desc())
        .limit(AGENT_SETTINGS.history_max_messages),
    )
    persisted_messages = list(reversed(result.scalars().all()))

    history: list[ModelMessage] = []
    for message in persisted_messages:
        if message.role == MessageRole.USER:
            history.append(ModelRequest.user_text_prompt(message.content))
            continue

        if message.role == MessageRole.ASSISTANT:
            history.append(ModelResponse(parts=[TextPart(content=message.content)]))
            continue

        if message.role == MessageRole.SYSTEM:
            history.append(
                ModelRequest(parts=[SystemPromptPart(content=message.content)]),
            )

    return history


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
        await llm_service.enforce_daily_message_limit(current_user)
        response = await llm_service.run_agent(
            user=current_user,
            user_prompt=request.prompt,
        )
        return LLMResponse(response=response)
    except DailyMessageLimitExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
        ) from e
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
    """Stream chat response and status updates in SSE payload format."""
    llm_service = LLMService(session=db, redis=redis)
    try:
        prompt = _resolve_stream_prompt(request)
        thread = await _resolve_thread(
            db=db,
            current_user=current_user,
            request=request,
            prompt=prompt,
        )
        message_history = await _load_thread_message_history(db=db, thread=thread)
        await llm_service.enforce_daily_message_limit(current_user)
        await _append_thread_message(
            db=db,
            thread=thread,
            role=MessageRole.USER,
            content=prompt,
        )
    except DailyMessageLimitExceededError as e:
        await redis.aclose()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
        ) from e
    except Exception:
        await redis.aclose()
        raise

    async def event_stream() -> AsyncIterator[str]:
        assistant_so_far = ""
        try:
            async for event in llm_service.run_agent_stream(
                user=current_user,
                user_prompt=prompt,
                message_history=message_history,
            ):
                if event.kind == "status":
                    payload = {
                        "type": "status",
                        "status": event.value,
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    continue

                payload = {
                    "choices": [
                        {
                            "delta": {
                                "content": event.value,
                            },
                        },
                    ],
                }
                assistant_so_far += event.value
                yield f"data: {json.dumps(payload)}\n\n"

            if assistant_so_far.strip():
                await _append_thread_message(
                    db=db,
                    thread=thread,
                    role=MessageRole.ASSISTANT,
                    content=assistant_so_far,
                )
            yield "data: [DONE]\n\n"
        except LLMServiceError as e:
            logger.exception("LLM stream failed: {}", e)
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
