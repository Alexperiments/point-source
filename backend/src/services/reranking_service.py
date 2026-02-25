"""Reranking service."""

from __future__ import annotations

from src.core.model_provider.litellm import get_litellm_reranking_model
from src.core.rag_config import RERANKER_SETTINGS
from src.schemas.reranking import Passage, RerankingModelInput, RerankingModelResult


class RerankingServiceError(Exception):
    """Exception for LLM service errors."""


class RerankingService:
    """Service for reranking retrieval candidates."""

    def __init__(self) -> None:
        """Initialize reranking service and lazily load its model."""
        self.model = get_litellm_reranking_model(RERANKER_SETTINGS)

    def rerank(
        self,
        query: str,
        passages: list[Passage],
    ) -> list[RerankingModelResult]:
        """Rerank candidates using the provided model."""
        try:
            request = RerankingModelInput(query=query, passages=passages)
            return self.model.rerank(request)
        except Exception as e:
            raise RerankingServiceError(f"Failed to run reranking: {e!s}") from e
