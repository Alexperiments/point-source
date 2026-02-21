"""Node structure definition and mapping to database."""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    MappedAsDataclass,
    foreign,
    mapped_column,
    relationship,
    synonym,
)

from src.core.database.base import Base


class NodeBase(MappedAsDataclass, DeclarativeBase):
    """Declarative base for dataclass-mapped node models."""

    metadata = Base.metadata
    __abstract__ = True


class TextNode(NodeBase):
    """Text Node Object.

    Generic interface for retrievable nodes containing text data.
    """

    __tablename__: str = "document_chunks"
    __table_args__ = {"schema": "processed"}

    text: Mapped[str] = mapped_column(Text(), nullable=False)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default_factory=uuid.uuid4,
        init=False,
    )

    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(1024),
        nullable=True,
        default=None,
        init=False,
    )

    node_metadata: Mapped[dict[str, object] | None] = mapped_column(
        JSON(),
        nullable=True,
        default=None,
        init=False,
    )

    document_id: Mapped[str] = mapped_column(
        Text(),
        ForeignKey("processed.documents.id"),
        nullable=False,
        init=False,
    )
    document: Mapped["DocumentNode"] = relationship(
        back_populates="children",
        init=False,
    )

    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("processed.document_chunks.id"),
        nullable=True,
        default=None,
    )
    parent: Mapped["TextNode"] = relationship(
        "TextNode",
        remote_side="TextNode.id",
        back_populates="children",
        foreign_keys=[parent_id],
        init=False,
    )

    prev_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("processed.document_chunks.id"),
        nullable=True,
        default=None,
    )
    prev_node: Mapped["TextNode | None"] = relationship(
        "TextNode",
        remote_side="TextNode.id",
        back_populates="next_node",
        foreign_keys=[prev_id],
        init=False,
    )

    next_node: Mapped["TextNode | None"] = relationship(
        "TextNode",
        back_populates="prev_node",
        uselist=False,
        primaryjoin=lambda: TextNode.id == foreign(TextNode.prev_id),
        init=False,
    )

    children: Mapped[list["TextNode"]] = relationship(
        "TextNode",
        back_populates="parent",
        cascade="all, delete-orphan",
        foreign_keys=[parent_id],
        single_parent=True,
        init=False,
    )


class DocumentNode(NodeBase):
    """Document object sourced from processed documents table."""

    __tablename__: str = "documents"
    __table_args__ = {"schema": "processed"}

    id: Mapped[str] = mapped_column(Text(), primary_key=True)
    url: Mapped[str] = mapped_column(Text(), nullable=False)
    authors: Mapped[str | None] = mapped_column(Text(), nullable=True, default=None)
    title: Mapped[str | None] = mapped_column(Text(), nullable=True, default=None)
    comments: Mapped[str | None] = mapped_column(Text(), nullable=True, default=None)
    journal_ref: Mapped[str | None] = mapped_column(
        Text(),
        nullable=True,
        default=None,
    )
    doi: Mapped[str | None] = mapped_column(Text(), nullable=True, default=None)
    report_no: Mapped[str | None] = mapped_column(Text(), nullable=True, default=None)
    categories: Mapped[str | None] = mapped_column(Text(), nullable=True, default=None)
    license: Mapped[str | None] = mapped_column(Text(), nullable=True, default=None)
    created: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )
    last_updated: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )
    text: Mapped[str | None] = mapped_column(Text(), nullable=True, default=None)

    # Compatibility alias for callers that still reference source_id.
    source_id = synonym("id")

    children: Mapped[list["TextNode"]] = relationship(
        back_populates="document",
        init=False,
        viewonly=True,
    )
