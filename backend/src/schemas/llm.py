"""LLM schemas."""

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
