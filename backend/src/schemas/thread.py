"""Thread schemas."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ThreadCreateRequest(BaseModel):
    """Schema for creating a new thread."""

    title: str | None = Field(
        default=None,
        max_length=255,
        description="Optional thread title.",
    )


class ThreadMessageResponse(BaseModel):
    """Schema for a chat message returned in thread history."""

    id: uuid.UUID = Field(..., description="Message ID")
    role: str = Field(..., description="Message role")
    content: str = Field(..., description="Message text")
    sequence_num: int = Field(..., description="Message order in the thread")
    created_at: datetime = Field(..., description="Message creation timestamp")

    model_config = ConfigDict(from_attributes=True)


class ThreadResponse(BaseModel):
    """Schema for thread history responses."""

    id: uuid.UUID = Field(..., description="Thread ID")
    title: str | None = Field(None, description="Thread title")
    created_at: datetime = Field(..., description="Thread creation timestamp")
    updated_at: datetime = Field(..., description="Thread update timestamp")
    messages: list[ThreadMessageResponse] = Field(
        default_factory=list,
        description="Thread messages ordered by sequence.",
    )

    model_config = ConfigDict(from_attributes=True)
