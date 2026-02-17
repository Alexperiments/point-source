"""Protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    import numpy as np

    from src.services.tokenization_service import TokenOffsets


class EmbeddingService(Protocol):
    """Embedding service protocol."""

    def encode_query(self, text_list: list[str], batch_size: int) -> np.ndarray:
        """Embed the input list of queries with the embedding model."""
        ...

    def encode_document(self, text_list: list[str], batch_size: int) -> np.ndarray:
        """Embed the input list of documents with the embedding model."""
        ...


class TokenizerService(Protocol):
    """Tokenizer service protocol."""

    def tokenize_with_offsets_mapping(self, text: str) -> TokenOffsets:
        """Tokenize the input string returning an offsets mapping."""
        ...
