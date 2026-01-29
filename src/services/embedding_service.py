"""Embedding and tokenizer service."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import cache

import logfire
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding


embedded_chunks_metric = logfire.metric_counter(
    "embedded_chunks",
    unit="1",
    description="Number of embedded chunks.",
)

Offset = tuple[int, int]


@cache
def get_embedding_model(name: str) -> SentenceTransformer:
    """Returns and caches the embedding model specified by the input name."""
    return SentenceTransformer(name)


@cache
def get_tokenizer(name: str):  # noqa: ANN201
    """Returns and caches tokenizer related to the model specified by the input name."""
    return AutoTokenizer.from_pretrained(name, use_fast=True)


@dataclass(frozen=True, slots=True)
class TokenOffsets:
    """Token-to-character offsets for a specific text.

    offsets[i] = (char_start, char_end) for token i.
    starts/ends are cached for fast bisect span->token span lookup.
    """

    offsets: list[Offset]
    starts: list[int]
    ends: list[int]

    @classmethod
    def from_offsets_mapping(cls, offsets: list[Offset]) -> "TokenOffsets":
        """Build TokenOffsets with cached start/end arrays for fast lookup."""
        return cls(
            offsets=offsets,
            starts=[start for start, _ in offsets],
            ends=[end for _, end in offsets],
        )

    def count_tokens_from_span(self, span_start: int, span_end: int) -> int:
        """Return the token span [start, end) overlapping the char span."""
        if span_start >= span_end:
            raise ValueError("Span start must be smaller than span end!")

        start_token = bisect_right(self.ends, span_start)
        end_token = bisect_left(self.starts, span_end)
        end_token = max(end_token, start_token)
        return end_token - start_token


class EmbeddingService:
    """Embedding service. Provides embedding and tokenization utilities."""

    def __init__(self, *, embedding_model_name: str) -> None:
        """Default constructor."""
        self.embedding_model_name = embedding_model_name

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Embedding model."""
        return get_embedding_model(self.embedding_model_name)

    @property
    def tokenizer(self):  # noqa: ANN201
        """Tokenizer."""
        return get_tokenizer(self.embedding_model_name)

    def encode(self, text_list: list[str]) -> np.ndarray:
        """Embed the input list of strings with the embedding model."""
        return self.embedding_model.encode(text_list)

    def tokenize(self, text: str) -> BatchEncoding:
        """Tokenize the input string with tokenizer."""
        return self.tokenizer(text)

    def tokenize_with_offsets_mapping(self, text: str) -> TokenOffsets:
        """Tokenize the input string returning an offsets mapping."""
        encoded = self.tokenizer(text, return_offsets_mapping=True)
        offsets_mapping = encoded["offset_mapping"]
        return TokenOffsets.from_offsets_mapping(offsets=offsets_mapping)
