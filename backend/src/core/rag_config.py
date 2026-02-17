"""RAG configuration settings."""

from __future__ import annotations

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

    tokenizer_model_name: str = Field(default="Qwen/Qwen3-Embedding-0.6B")
    max_tokens: int = Field(default=256)
    overlap_tokens: int = Field(default=0)
    min_chunk_chars: int = Field(default=20)

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
    drop_section_title_prefixes: tuple[str, ...] = (
        "references",
        "reference list",
        "list of references",
        "bibliography",
        "works cited",
        "citations",
        "acknowledgements",
        "acknowledgments",
        "acknowledgement",
        "acknowledgment",
        "thanks",
        "thank you",
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


class EmbeddingSettings(BaseSettings):
    """Configuration settings for embeddings."""

    model_config = SettingsConfigDict(
        env_file=env_file_path,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="EMBEDDING_",
    )

    model_name: str = Field(
        default="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    )
    max_tokens: int = Field(default=256)
    query_batch_size: int = Field(default=8)
    document_batch_size: int = Field(default=8)
    query_instruction: str = Field(
        default="Given a web search query, retrieve relevant passages that answer the query.",
    )


class RetrievalSettings(BaseSettings):
    """Configuration settings for retrieval."""

    model_config = SettingsConfigDict(
        env_file=env_file_path,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="RETRIEVAL_",
    )

    top_n: int = Field(default=5)
    text_top_k: int = Field(default=50)
    vector_top_k: int = Field(default=50)
    rrf_k: int = Field(default=60)
    semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    text_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    use_prev_next: bool = Field(default=False)
    max_merged_chars: int = Field(default=4000)
    cache_prefix: str = Field(default="retrieval_cache:")
    cache_ttl_seconds: int = Field(default=900)


class RerankerSettings(BaseSettings):
    """Configuration settings for re-ranking."""

    model_config = SettingsConfigDict(
        env_file=env_file_path,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="RERANKER_",
    )

    enabled: bool = Field(default=True)
    model_name: str = Field(default="mock-reranker")
    top_k: int = Field(default=5)
    batch_size: int = Field(default=8)


class AgentSettings(BaseSettings):
    """Configuration settings for the main agent."""

    model_config = SettingsConfigDict(
        env_file=env_file_path,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="AGENT_",
    )

    name: str = Field(default="Main Agent")
    model_name: str = Field(default="custom/qwen3-4b-mlx-4bit")
    custom_llm_provider: str | None = Field(
        default=None,
        description="LiteLLM custom provider override for custom model routing.",
    )
    temperature: float = Field(default=0.3, ge=0.0)
    max_tokens: int = Field(default=2048, ge=1)
    instruction_slug: str = Field(default="main_agent_instructions")


CHUNKING_SETTINGS = ChunkingSettings()
EMBEDDING_SETTINGS = EmbeddingSettings()
RETRIEVAL_SETTINGS = RetrievalSettings()
RERANKER_SETTINGS = RerankerSettings()
AGENT_SETTINGS = AgentSettings()
