"""Thread history API."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.v1.auth import get_current_user
from src.core.database.base import get_async_session
from src.models.thread import Thread
from src.models.user import User
from src.schemas.thread import ThreadCreateRequest, ThreadResponse


router = APIRouter()


@router.get("", response_model=list[ThreadResponse])
async def list_threads(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_async_session)],
) -> list[Thread]:
    """List all chat threads owned by the authenticated user."""
    result = await db.execute(
        select(Thread)
        .where(
            Thread.user_id == current_user.id,
            Thread.is_archived.is_(False),
        )
        .options(selectinload(Thread.messages))
        .order_by(Thread.updated_at.desc()),
    )
    return result.scalars().unique().all()


@router.post("", response_model=ThreadResponse, status_code=status.HTTP_201_CREATED)
async def create_thread(
    payload: ThreadCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_async_session)],
) -> ThreadResponse:
    """Create a new empty thread for the authenticated user."""
    title = (payload.title or "").strip() or "New chat"
    thread = Thread(
        user_id=current_user.id,
        title=title,
    )
    db.add(thread)
    await db.flush()
    await db.refresh(thread)
    return ThreadResponse(
        id=thread.id,
        title=thread.title,
        created_at=thread.created_at,
        updated_at=thread.updated_at,
        messages=[],
    )


@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_thread(
    thread_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_async_session)],
) -> Response:
    """Delete a thread owned by the authenticated user."""
    result = await db.execute(
        select(Thread).where(
            Thread.id == thread_id,
            Thread.user_id == current_user.id,
        ),
    )
    thread = result.scalar_one_or_none()
    if thread is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found.",
        )

    await db.delete(thread)
    await db.flush()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
