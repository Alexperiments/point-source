"""Retrieval schemas."""

import re
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, HttpUrl, computed_field


class RetrievedChunk(BaseModel):
    """Structured retrieval result for a single chunk."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: uuid.UUID
    document_id: str
    doi_url: HttpUrl
    authors: str | None = None
    title: str | None = None
    journal_ref: str | None = None
    path: str | None = None
    text: str
    text_rank: int | None = None
    vector_rank: int | None = None
    rrf_score: float = 0.0
    relevance_score: float | None = None

    @property
    def text_snippet(self) -> str:
        """Return a compact single-line snippet for display."""
        snippet = re.sub(r"\s+", " ", self.text or "").strip()
        return snippet[:160]

    @computed_field
    @property
    def citation(self) -> str:
        """Return a compact citation string."""
        citation_strings = [
            self.authors,
            self.title,
            self.journal_ref,
        ]
        citation = ". ".join(filter(None, citation_strings))
        return f"[{citation}]({self.doi_url})"

    def __repr__(self) -> str:
        path = self.path or ""
        snippet = self.text_snippet
        return f"[{self.doi_url}] {path} — {snippet}"

    def to_cache_dict(self) -> dict[str, Any]:
        """Serialize for caching."""
        return self.model_dump(mode="json")
