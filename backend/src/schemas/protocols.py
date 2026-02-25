"""Protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from src.services.tokenization_service import TokenOffsets


class TokenizerService(Protocol):
    """Tokenizer service protocol."""

    def tokenize_with_offsets_mapping(self, text: str) -> TokenOffsets:
        """Tokenize the input string returning an offsets mapping."""
        ...
