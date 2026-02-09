"""Retrieval schemas."""

import re
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RetrievalFilters(BaseModel):
    """Optional filters for retrieval."""

    model_config = ConfigDict(extra="ignore")

    document_id: uuid.UUID | None = None
    source_id: str | None = None
    category: str | None = None


class RetrievedChunk(BaseModel):
    """Structured retrieval result for a single chunk."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    source_id: str
    path: str | None = None
    text: str
    text_rank: int | None = None
    vector_rank: int | None = None
    rrf_score: float = 0.0
    citations: list[str] = Field(default_factory=list)

    @property
    def text_snippet(self) -> str:
        """Return a compact single-line snippet for display."""
        snippet = re.sub(r"\s+", " ", self.text or "").strip()
        return snippet[:160]

    def __repr__(self) -> str:
        path = self.path or ""
        snippet = self.text_snippet
        return f"[{self.source_id}] {path} — {snippet}"

    def to_cache_dict(self) -> dict[str, Any]:
        """Serialize for caching."""
        return self.model_dump(mode="json")
