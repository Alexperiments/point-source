"""LiteLLM provider utilities."""

import math

import litellm
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_ai import Embedder
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.litellm import LiteLLMProvider

from src.core.config import settings as app_settings
from src.core.rag_config import AgentSettings, EmbeddingSettings, RerankerSettings
from src.schemas.reranking import (
    RerankingModel,
    RerankingModelInput,
    RerankingModelResult,
)


class _LiteLLMRerankResult(BaseModel):
    """Validated LiteLLM rerank result row."""

    model_config = ConfigDict(extra="ignore")

    index: int
    relevance_score: float


class _LiteLLMRerankResponse(BaseModel):
    """Validated LiteLLM rerank response payload."""

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    results: list[_LiteLLMRerankResult]


def get_litellm_chat_model(
    settings: AgentSettings,
) -> OpenAIChatModel:
    """Get a model from the model registry."""
    return OpenAIChatModel(
        model_name=settings.model_name,
        settings={
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
        },
        provider=LiteLLMProvider(
            api_base=app_settings.litellm_base_url.unicode_string(),
            api_key=app_settings.litellm_api_key.get_secret_value(),
        ),
    )


def get_litellm_embedding_model(settings: EmbeddingSettings) -> Embedder:
    """Get a LiteLLM embedding model instance."""
    embedding_model = OpenAIEmbeddingModel(
        model_name=settings.model_name,
        settings={"dimensions": settings.embedding_size},
        provider=LiteLLMProvider(
            api_base=app_settings.litellm_base_url.unicode_string(),
            api_key=app_settings.litellm_api_key.get_secret_value(),
        ),
    )

    return Embedder(
        model=embedding_model,
    )


class LiteLLMRerankingModel(RerankingModel):
    """Pydantic-ai-like reranking model backed by LiteLLM."""

    def __init__(
        self,
        settings: RerankerSettings,
    ) -> None:
        """Initialize reranking client and fallback strategy."""
        self.model_name = settings.model_name
        self.settings = settings

    def score(
        self,
        request: RerankingModelInput,
    ) -> list[RerankingModelResult]:
        """Return relevance scores from the configured LLM reranker."""
        passages = list(request.passages)
        documents = [passage.passage for passage in passages]

        response = litellm.rerank(
            model=self.model_name,
            query=request.query,
            documents=documents,
            custom_llm_provider=self.settings.custom_llm_provider,
            top_n=len(documents),
            return_documents=False,
            api_base=app_settings.litellm_base_url.unicode_string(),
            api_key=app_settings.litellm_api_key.get_secret_value(),
            timeout=self.settings.timeout_seconds,
        )
        try:
            parsed_response = _LiteLLMRerankResponse.model_validate(response)
        except ValidationError as exc:
            raise ValueError("Invalid LiteLLM rerank response schema.") from exc

        reranked: list[RerankingModelResult] = []
        for result in parsed_response.results:
            if result.index < 0 or result.index >= len(passages):
                raise ValueError(
                    f"Invalid rerank index returned by provider: {result.index}",
                )
            if not math.isfinite(result.relevance_score):
                raise ValueError(
                    "Invalid rerank relevance score returned by provider: must be finite.",
                )
            reranked.append(
                RerankingModelResult(
                    query=request.query,
                    passage=passages[result.index],
                    relevance_score=result.relevance_score,
                    run_id=parsed_response.id,
                ),
            )
        return reranked


def get_litellm_reranking_model(settings: RerankerSettings) -> RerankingModel:
    """Get a LiteLLM reranking model instance."""
    return LiteLLMRerankingModel(settings=settings)
