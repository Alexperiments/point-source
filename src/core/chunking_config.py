"""Chunking configuration settings."""

import os
import re
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


parent_dir = Path(__file__).resolve().parents[2]
CURRENT_ENV = os.getenv("ENVIRONMENT", "development").lower()
env_file_path = parent_dir / f".env.{CURRENT_ENV}"


class ChunkingSettings(BaseSettings):
    """Configuration settings for chunking."""

    model_config = SettingsConfigDict(
        env_file=env_file_path,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="CHUNKING_",
    )

    embedding_model_name: str = Field(default="Qwen/Qwen3-Embedding-0.6B")
    max_tokens: int = Field(default=256)
    overlap_tokens: int = Field(default=0)

    inline_latex_math_patterns: re.Pattern = re.compile(
        r"(?<!\$)\$(?!\$)(?:\\.|[^$\n\\])+?\$(?!\$)|\\\((?:\\.|[^\\)\n])+?\\\)",
    )
    block_latex_math_patterns: re.Pattern = re.compile(
        r"\$\$.*?\$\$|\\\[(?:.|\n)*?\\\]|\\begin\{(?:equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}.*?\\end\{(?:equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}",
        re.DOTALL,
    )
    table_patterns: re.Pattern = re.compile(
        r"^\s*\|?.*\|.*\n^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*(?:\n^\s*\|?.*\|.*)*",
        re.MULTILINE,
    )
    header_patterns: re.Pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    paragraph_patterns: re.Pattern = re.compile(r"(?:\r?\n)\s*(?:\r?\n)+")
    sentence_patterns: re.Pattern = re.compile(
        r"[.!?]+[\"')\]]*(?:\s+|$)",
        re.MULTILINE,
    )
    citation_command_prefixes: tuple[str, ...] = (
        "\\cite",
        "\\citet",
        "\\citep",
        "\\citealp",
        "\\citeauthor",
        "\\citeyear",
        "\\parencite",
        "\\textcite",
        "\\footcite",
        "\\ref",
        "\\eqref",
        "\\autoref",
        "\\cref",
        "\\Cref",
    )


CHUNKING_SETTINGS = ChunkingSettings()
