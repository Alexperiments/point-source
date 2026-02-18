"""Reranking service."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import litellm
from loguru import logger

from src.core.config import settings as app_settings
from src.core.rag_config import RERANKER_SETTINGS


if TYPE_CHECKING:
    from src.schemas.protocols import Reranker
    from src.schemas.retrieval import RetrievedChunk


_TOKEN_RE = re.compile(r"\w+")


@dataclass(frozen=True, slots=True)
class RerankingModelSettings:
    """Settings used to initialize a reranking model."""

    model_name: str
    timeout_seconds: int


class BasicLexicalReranker:
    """Simple lexical reranker used as the default local strategy."""

    def __init__(self, settings: RerankingModelSettings) -> None:
        """Store model metadata and runtime scoring parameters."""
        self.model_name = settings.model_name

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return lexical relevance scores for each passage."""
        query_terms = self._token_set(query)
        if not query_terms:
            return [0.0] * len(passages)

        return self._score_passages(query, query_terms, passages)

    def _score_passages(
        self,
        query: str,
        query_terms: set[str],
        passages: list[str],
    ) -> list[float]:
        query_normalized = " ".join(_TOKEN_RE.findall(query.lower()))
        out: list[float] = []

        for passage in passages:
            passage_terms = self._token_set(passage)
            if not passage_terms:
                out.append(0.0)
                continue

            overlap = len(query_terms & passage_terms) / len(query_terms)
            density = len(query_terms & passage_terms) / len(passage_terms)

            phrase_boost = 0.0
            if query_normalized:
                passage_normalized = " ".join(_TOKEN_RE.findall(passage.lower()))
                if query_normalized in passage_normalized:
                    phrase_boost = 0.2

            score = (0.8 * overlap) + (0.2 * density) + phrase_boost
            out.append(score)

        return out

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return {token.lower() for token in _TOKEN_RE.findall(text)}


class LiteLLMModelReranker:
    """Reranker backed by LiteLLM's native rerank endpoint."""

    def __init__(
        self,
        settings: RerankingModelSettings,
        fallback: Reranker | None = None,
    ) -> None:
        """Initialize reranking client and fallback strategy."""
        self.model_name = settings.model_name
        self.timeout_seconds = settings.timeout_seconds
        self._fallback = fallback or BasicLexicalReranker(settings)

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return relevance scores from the configured LLM reranker."""
        if not passages:
            return []

        try:
            return self._score_with_model(query, passages)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Model reranking failed for model {} ({}). Falling back to lexical reranker.",
                self.model_name,
                exc,
            )
            return self._fallback.score(query, passages)

    def _score_with_model(self, query: str, passages: list[str]) -> list[float]:
        response = litellm.rerank(
            model=self.model_name,
            query=query,
            documents=passages,
            custom_llm_provider="litellm_proxy",
            top_n=len(passages),
            return_documents=False,
            api_base=app_settings.litellm_base_url.unicode_string(),
            api_key=app_settings.litellm_api_key.get_secret_value(),
            timeout=self.timeout_seconds,
        )
        return self._scores_from_response(
            response=response,
            expected_len=len(passages),
        )

    @staticmethod
    def _scores_from_response(
        response: object,
        *,
        expected_len: int,
    ) -> list[float]:
        results = getattr(response, "results", None)
        if not isinstance(results, list):
            msg = "Reranker response did not include a valid 'results' list."
            raise TypeError(msg)

        indexed: dict[int, float] = {}
        for item in results:
            if isinstance(item, dict):
                index = item.get("index")
                score = item.get("relevance_score")
            else:
                index = getattr(item, "index", None)
                score = getattr(item, "relevance_score", None)
            if not isinstance(index, int):
                continue
            if score is None:
                continue
            indexed[index] = float(score)

        if len(indexed) != expected_len or any(
            i not in indexed for i in range(expected_len)
        ):
            msg = (
                "Reranker returned invalid number of scores: "
                f"{len(indexed)} != {expected_len}"
            )
            raise ValueError(msg)

        return [indexed[index] for index in range(expected_len)]


class RerankingService:
    """Service for reranking retrieval candidates."""

    def __init__(
        self,
        model: Reranker | None = None,
    ) -> None:
        """Initialize reranking service and lazily load its model."""
        self.model_settings = RerankingModelSettings(
            model_name=RERANKER_SETTINGS.model_name,
            timeout_seconds=RERANKER_SETTINGS.timeout_seconds,
        )
        self.model = model or get_reranking_model(self.model_settings)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Rerank candidates using the configured strategy."""
        if not RERANKER_SETTINGS.enabled:
            return candidates
        if not candidates:
            return []

        top_k = max(RERANKER_SETTINGS.top_k, 0)
        if top_k == 0:
            return []

        scores = self.model.score(query, [candidate.text for candidate in candidates])

        ranked = sorted(
            enumerate(candidates),
            key=lambda item: (
                scores[item[0]],
                item[1].rrf_score,
                -item[0],
            ),
            reverse=True,
        )
        return [item[1] for item in ranked[:top_k]]


@lru_cache(maxsize=8)
def get_reranking_model(
    settings: RerankingModelSettings | None = None,
) -> Reranker:
    """Get a cached reranking model instance."""
    resolved = settings or RerankingModelSettings(
        model_name=RERANKER_SETTINGS.model_name,
        timeout_seconds=RERANKER_SETTINGS.timeout_seconds,
    )

    if resolved.model_name.lower().strip() in {
        "",
        "mock-reranker",
        "basic-lexical",
        "basic-lexical-reranker",
    }:
        return BasicLexicalReranker(resolved)

    return LiteLLMModelReranker(
        resolved,
        fallback=BasicLexicalReranker(resolved),
    )
