"""LLM schemas."""

import uuid
from typing import Literal

from pydantic import BaseModel, Field


class LLMRequest(BaseModel):
    """LLM request schema."""

    prompt: str = Field(
        ...,
        description="The user's prompt/query",
        min_length=1,
    )


class LLMResponse(BaseModel):
    """LLM response schema."""

    response: str = Field(..., description="The agent's response")


class LLMStreamMessage(BaseModel):
    """Chat message schema used by streaming endpoint."""

    role: Literal["system", "user", "assistant"] = Field(
        ...,
        description="Message role.",
    )
    content: str = Field(..., description="Message content", min_length=1)


class LLMStreamRequest(BaseModel):
    """Streaming request schema."""

    messages: list[LLMStreamMessage] = Field(default_factory=list)
    prompt: str | None = Field(default=None, min_length=1)
    thread_id: uuid.UUID | None = Field(
        default=None,
        description="Existing thread ID for persisting chat history.",
    )
