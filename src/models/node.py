"""Node structure definition and mapping to database."""

import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    ForeignKey,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, foreign, mapped_column, relationship

from src.core.database.base import AbstractBase


class BaseNode(AbstractBase):
    """Abstract text node interface."""

    __abstract__ = True

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(768),
        nullable=True,
        default=None,
    )

    text: Mapped[str] = mapped_column(Text(), nullable=False)

    node_metadata: Mapped[dict[str, object] | None] = mapped_column(
        JSON(),
        nullable=True,
        default=None,
    )

    def __repr__(self) -> str:
        return f"<node_id={self.id} text={self.text.strip()[:50]} node_metadata={self.node_metadata}>"


class TextNode(BaseNode):
    """Text Node Object.

    Generic interface for retrievable nodes containing text data.
    """

    __tablename__: str = "document_chunks"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id"),
        nullable=True,
        default=None,
    )
    document: Mapped["DocumentNode"] = relationship(back_populates="children")

    parent_id: Mapped[uuid.UUID] = mapped_column(
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
    )

    next_node: Mapped["TextNode | None"] = relationship(
        "TextNode",
        back_populates="prev_node",
        uselist=False,
        primaryjoin=lambda: TextNode.id == foreign(TextNode.prev_id),
    )

    children: Mapped[list["TextNode"]] = relationship(
        "TextNode",
        back_populates="parent",
        cascade="all, delete-orphan",
        foreign_keys=[parent_id],
        single_parent=True,
    )


class DocumentNode(BaseNode):
    """Document node object."""

    __tablename__: str = "documents"

    source_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    children: Mapped[list["TextNode"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )
