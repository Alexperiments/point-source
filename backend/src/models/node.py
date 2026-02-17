"""Node structure definition and mapping to database."""

import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
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
)

from src.core.database.base import Base


class NodeBase(MappedAsDataclass, DeclarativeBase):
    """Declarative base for dataclass-mapped node models."""

    metadata = Base.metadata
    __abstract__ = True


class BaseNode(NodeBase):
    """Abstract text node interface."""

    __abstract__ = True

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


class TextNode(BaseNode):
    """Text Node Object.

    Generic interface for retrievable nodes containing text data.
    """

    __tablename__: str = "document_chunks"

    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id"),
        nullable=True,
        default=None,
    )
    document: Mapped["DocumentNode"] = relationship(
        back_populates="children",
        init=False,
    )

    parent_id: Mapped[uuid.UUID | None] = mapped_column(
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
        init=False,
    )

    prev_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_chunks.id"),
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


class DocumentNode(BaseNode):
    """Document node object."""

    __tablename__: str = "documents"

    source_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    children: Mapped[list["TextNode"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        init=False,
    )
