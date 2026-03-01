"""Embedding service."""

from collections.abc import Sequence
from functools import lru_cache

from pydantic_ai import Embedder, EmbeddingResult

from src.core.model_provider.litellm import get_litellm_embedding_model
from src.core.rag_config import EMBEDDING_SETTINGS, EmbeddingSettings


class EmbeddingService:
    """Service responsible for generating embeddings for input text."""

    def __init__(self, model: Embedder | None = None) -> None:
        """Initialize the embedding model."""
        self.model_settings = EMBEDDING_SETTINGS
        self.model = model or get_embedding_model()

    async def embed_query(self, queries: str | Sequence[str]) -> EmbeddingResult:
        """Embed a list of query strings."""
        return await self.model.embed_query(queries)

    async def embed_documents(self, documents: str | Sequence[str]) -> EmbeddingResult:
        """Embed a list of document strings."""
        return await self.model.embed_documents(documents)


def get_mlx_embedding_model(settings: EmbeddingSettings) -> Embedder:
    """Lazily load the MLX provider so non-MLX environments can import this module."""
    try:
        from src.core.model_provider.mlx import (
            get_mlx_embedding_model as _get_mlx_embedding_model,
        )
    except ModuleNotFoundError as exc:
        msg = (
            "MLX dependencies are not installed. Install the `local-mps` "
            "dependency group (for example `uv sync --group local-mps`) or "
            "use a non-MLX embedding model."
        )
        raise RuntimeError(msg) from exc

    return _get_mlx_embedding_model(settings)


def _build_embedding_model(resolved: EmbeddingSettings) -> Embedder:
    model_name = resolved.model_name.lower().strip()
    if model_name == "mlx" or model_name.startswith(("mlx/", "mlx-community/")):
        return get_mlx_embedding_model(resolved)

    return get_litellm_embedding_model(resolved)


@lru_cache(maxsize=1)
def _get_default_embedding_model() -> Embedder:
    """Get a cached embedding model instance built from default settings."""
    return _build_embedding_model(EMBEDDING_SETTINGS)


def get_embedding_model(
    settings: EmbeddingSettings | None = None,
) -> Embedder:
    """Get an embedding model instance."""
    if settings is None or settings is EMBEDDING_SETTINGS:
        return _get_default_embedding_model()
    return _build_embedding_model(settings)
