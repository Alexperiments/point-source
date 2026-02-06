"""Tokenization service and token offset utilities."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Any

from transformers import AutoTokenizer


Offset = tuple[int, int]


@dataclass(frozen=True, slots=True)
class TokenOffsets:
    """Token-to-character offsets for a specific text.

    offsets[i] = (char_start, char_end) for token i.
    starts/ends are cached for fast bisect span lookup.
    """

    offsets: list[Offset]
    starts: list[int]
    ends: list[int]

    @classmethod
    def from_offsets_mapping(cls, offsets: list[Offset]) -> TokenOffsets:
        """Build cached start/end arrays from an offsets mapping."""
        return cls(
            offsets=offsets,
            starts=[start for start, _ in offsets],
            ends=[end for _, end in offsets],
        )

    def count_tokens_from_span(self, span_start: int, span_end: int) -> int:
        """Return the number of tokens overlapping the [start, end) span."""
        if span_start >= span_end:
            raise ValueError("Span start must be smaller than span end!")

        start_token = bisect_right(self.ends, span_start)
        end_token = bisect_left(self.starts, span_end)
        end_token = max(end_token, start_token)
        return end_token - start_token


class TokenizerService:
    """Tokenizer service for offset mapping lookups."""

    def __init__(
        self,
        *,
        model_name: str,
        tokenizer_kwargs: dict[str, Any] | None = None,
        use_fast: bool = True,
    ) -> None:
        """Initialize an AutoTokenizer configured for offset mapping."""
        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=use_fast,
            **(tokenizer_kwargs or {}),
        )

    def tokenize_with_offsets_mapping(self, text: str) -> TokenOffsets:
        """Tokenize text and return token-to-character offsets."""
        encoded = self._tokenizer(text, return_offsets_mapping=True)
        offsets_mapping = encoded["offset_mapping"]
        return TokenOffsets.from_offsets_mapping(offsets=offsets_mapping)
