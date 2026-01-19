"""LLM API."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.v1.auth import get_current_user
from src.database.database import get_async_session
from src.database.redis import get_redis_pool
from src.models.user import User
from src.schemas.llm import LLMRequest, LLMResponse
from src.services.llm_service import LLMService, LLMServiceError


router = APIRouter()


async def get_redis() -> Redis:
    """Get Redis client dependency."""
    return await get_redis_pool()


@router.post("/chat", response_model=LLMResponse)
async def chat(
    request: LLMRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_async_session)],
    redis: Annotated[Redis, Depends(get_redis)],
) -> LLMResponse:
    """Chat with the LLM agent.

    Args:
        request: The LLM request containing the user's prompt.
        current_user: The currently authenticated user.
        db: The database session.
        redis: The Redis client.

    Returns:
        The LLM response.

    Raises:
        HTTPException: If the agent run fails.

    """
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
