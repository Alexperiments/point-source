"""Configuration for embedding benchmark tables and models."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingModelSpec:
    """Embedding model configuration used by the benchmark runner."""

    key: str
    model_name: str
    dimension: int
    provider: str
    enabled: bool = True


EMBEDDING_MODEL_SPECS: tuple[EmbeddingModelSpec, ...] = (
    EmbeddingModelSpec(
        key="qwen3_embedding_0_6b",
        model_name="Qwen/Qwen3-Embedding-0.6B",
        dimension=1024,
        provider="sentence_transformers",
    ),
    EmbeddingModelSpec(
        key="embeddinggemma_300m",
        model_name="google/embeddinggemma-300m",
        dimension=768,
        provider="sentence_transformers",
    ),
    EmbeddingModelSpec(
        key="e5_large_v2",
        model_name="intfloat/e5-large-v2",
        dimension=1024,
        provider="sentence_transformers",
    ),
    EmbeddingModelSpec(
        key="gte_large_en_v1_5",
        model_name="Alibaba-NLP/gte-large-en-v1.5",
        dimension=1024,
        provider="sentence_transformers",
    ),
    EmbeddingModelSpec(
        key="bge_m3",
        model_name="BAAI/bge-m3",
        dimension=1024,
        provider="sentence_transformers",
    ),
    EmbeddingModelSpec(
        key="text_embedding_3_small",
        model_name="text-embedding-3-small",
        dimension=1536,
        provider="openai",
        enabled=True,
    ),
    EmbeddingModelSpec(
        key="voyage_3_5",
        model_name="voyage-3.5",
        dimension=1024,
        provider="voyage",
        enabled=False,
    ),
)


def document_embedding_table_name(spec: EmbeddingModelSpec) -> str:
    """Table name for document embeddings for a specific model."""
    return f"document_embeddings_{spec.key}"


def query_embedding_table_name(spec: EmbeddingModelSpec) -> str:
    """Table name for query embeddings for a specific model."""
    return f"query_embeddings_{spec.key}"
