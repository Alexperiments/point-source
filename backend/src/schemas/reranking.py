"""Reranking model interfaces and result types."""

import uuid

from pydantic import BaseModel, ConfigDict

from src.core.rag_config import RerankerSettings


class Passage(BaseModel):
    """Single passage to rerank with a stable source identifier."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    id: uuid.UUID
    passage: str


class RerankingModelInput(BaseModel):
    """Strict reranking request payload."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    query: str
    passages: list[Passage]


class RerankingModelResult(BaseModel):
    """Structured result of a reranking model."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    query: str
    passage: Passage
    relevance_score: float
    run_id: str | None = None


class RerankingModel:
    """Reranker model interface."""

    def __init__(self, settings: RerankerSettings) -> None:
        """Initialize the reranker with the given model name and settings."""
        self.settings = settings

    def score(
        self,
        request: RerankingModelInput,
    ) -> list[RerankingModelResult]:
        """Return relevance scores for each passage given the query."""
        raise NotImplementedError("Subclasses must implement the score method.")

    def rerank(self, request: RerankingModelInput) -> list[RerankingModelResult]:
        """Rerank candidates using the configured strategy."""
        if not request.passages:
            return []

        reranking_result = self.score(request)

        passage_first_index: dict[uuid.UUID, int] = {}
        for index, passage in enumerate(request.passages):
            passage_first_index.setdefault(passage.id, index)

        return sorted(
            reranking_result,
            key=lambda item: (
                item.relevance_score,
                -passage_first_index.get(item.passage.id, len(request.passages)),
            ),
            reverse=True,
        )
