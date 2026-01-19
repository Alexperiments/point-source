"""Node structure definition and mapping to database."""

import json
import textwrap
import uuid
from enum import Enum
from hashlib import sha256
from typing import Literal

from pgvector.sqlalchemy import Vector
from pydantic.config import ConfigDict
from sqlalchemy import (
    JSON,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.database import AbstractBase


DEFAULT_TEXT_NODE_TMPL = "{metadata_str}\n\n{content}"
DEFAULT_METADATA_TMPL = "{key}: {value}"

TRUNCATE_LENGTH = 350
WRAP_WIDTH = 70

EMBEDDING_KIND = Literal["sparse", "dense"]


class MetadataMode(str, Enum):
    """Enumeration of metadata modes."""

    ALL = "all"  # Include all metadata
    EMBED = "embed"  # Include only metadata relevant for embedding
    LLM = "llm"  # Include only metadata relevant for LLM
    NONE = "none"  # Exclude all metadata


class BaseNode(AbstractBase):
    """Abstract text node interface."""

    __abstract__ = True

    model_config = ConfigDict(populate_by_name=True, validate_assignment=True)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    embedding: Mapped[list[float]] = mapped_column(
        Vector(768),
        nullable=True,
        default=list,
    )

    text: Mapped[str] = mapped_column(Text(), nullable=False)

    start_char_idx: Mapped[int | None] = mapped_column(
        Integer(),
        nullable=True,
        default=None,
    )

    end_char_idx: Mapped[int | None] = mapped_column(
        Integer(),
        nullable=True,
        default=None,
    )

    node_metadata: Mapped[dict[str, object]] = mapped_column(
        JSON(),
        nullable=True,
        default=dict,
    )

    excluded_embed_metadata_keys: Mapped[list[str]] = mapped_column(
        JSON(),
        nullable=True,
        default=list,
    )

    excluded_llm_metadata_keys: Mapped[list[str]] = mapped_column(
        JSON(),
        nullable=True,
        default=list,
    )

    text_template: Mapped[str] = mapped_column(Text(), default=DEFAULT_TEXT_NODE_TMPL)

    metadata_template: Mapped[str] = mapped_column(
        Text(),
        default=DEFAULT_METADATA_TMPL,
    )

    metadata_separator: Mapped[str] = mapped_column(Text(), default="\n")

    hash_: Mapped[str] = mapped_column(String(), nullable=True, default="")

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        """Default constructor."""
        super().__init__(**kwargs)
        self.node_metadata = kwargs.get("node_metadata", {}) or {}
        self.metadata_separator = kwargs.get("metadata_separator", "\n") or "\n"
        self.metadata_template = (
            kwargs.get("metadata_template", DEFAULT_METADATA_TMPL)
            or DEFAULT_METADATA_TMPL
        )
        self.text_template = (
            kwargs.get("text_template", DEFAULT_TEXT_NODE_TMPL)
            or DEFAULT_TEXT_NODE_TMPL
        )

    @property
    def hash(self) -> str:
        """Hash sha-256 of content + metadata."""
        doc_identity: str = self.text + json.dumps(self.node_metadata, sort_keys=True)
        return str(sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest())

    def get_content(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        """Get object content."""
        metadata_str: str = self.get_metadata_str(mode=metadata_mode).strip()
        if metadata_mode == MetadataMode.NONE or not metadata_str:
            return self.text

        return self.text_template.format(
            content=self.text,
            metadata_str=metadata_str,
        ).strip()

    def get_metadata_str(self, mode: MetadataMode = MetadataMode.ALL) -> str:
        """Metadata info string."""
        if mode == MetadataMode.NONE:
            return ""

        usable_metadata_keys: set[str] = set(self.node_metadata.keys())
        if mode == MetadataMode.LLM:
            for key in self.excluded_llm_metadata_keys:
                if key in usable_metadata_keys:
                    usable_metadata_keys.discard(key)
        elif mode == MetadataMode.EMBED:
            for key in self.excluded_embed_metadata_keys:
                if key in usable_metadata_keys:
                    usable_metadata_keys.discard(key)

        return self.metadata_separator.join(
            [
                self.metadata_template.format(key=key, value=str(value))
                for key, value in self.node_metadata.items()
                if key in usable_metadata_keys
            ],
        )

    def set_content(self, value: str) -> None:
        """Set the content of the node."""
        self.text = value

    def get_embedding(self) -> list[float]:
        """Get embedding. Errors if embedding is None."""
        if self.embedding is None:
            raise ValueError("embedding not set.")
        return self.embedding

    def get_text(self) -> str:
        """Get node content."""
        return self.get_content(metadata_mode=MetadataMode.NONE)

    def __str__(self) -> str:
        source_text_truncated: str = self.get_content().strip()[:TRUNCATE_LENGTH]
        source_text_wrapped: str = textwrap.fill(
            f"Text: {source_text_truncated}\n",
            width=WRAP_WIDTH,
        )
        return f"Node ID: {self.node_id}\n{source_text_wrapped}"


class TextNode(BaseNode):
    """Text Node Object.

    Generic interface for retrievable nodes containing text data.
    """

    __tablename__: str = "document_chunks"

    source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id"),
        nullable=True,
        default=None,
    )

    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_chunks.id"),
        nullable=True,
        default=None,
    )

    prev_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_chunks.id"),
        nullable=True,
        default=None,
    )

    parent: Mapped["TextNode"] = relationship(
        "TextNode",
        remote_side="TextNode.id",
        back_populates="children",
        foreign_keys=[parent_id],
    )

    children: Mapped[list["TextNode"]] = relationship(
        "TextNode",
        back_populates="parent",
        cascade="all, delete-orphan",
        foreign_keys=[parent_id],
        single_parent=True,
    )

    document: Mapped["DocumentNode"] = relationship(back_populates="children")

    prev_node: Mapped["TextNode | None"] = relationship(
        "TextNode",
        remote_side="TextNode.id",
        back_populates="next_node",
        foreign_keys=[prev_id],
    )

    next_node: Mapped["TextNode | None"] = relationship(
        "TextNode",
        back_populates="prev_node",
        foreign_keys=[prev_id],
        uselist=False,
    )

    @property
    def children_ids(self) -> list[uuid.UUID]:
        """Returns a list of IDs of the children nodes."""
        return [child.id for child in self.children]

    @property
    def next_id(self) -> uuid.UUID | None:
        """Convenience property to get next_id without loading the object."""
        return self.next_node.id if self.next_node else None

    def get_relationship_ids(self) -> dict[str, uuid.UUID | list[uuid.UUID] | None]:
        """Get node as RelatedNodeInfo."""
        return {
            "source_id": self.source_id,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "next_id": self.next_id,
            "prev_id": self.prev_id,
        }


class DocumentNode(BaseNode):
    """Document node object."""

    __tablename__: str = "documents"

    children: Mapped[list["TextNode"]] = relationship(back_populates="document")

    @property
    def children_ids(self) -> list[uuid.UUID]:
        """Returns a list of IDs of the children nodes."""
        return [child.id for child in self.children]
