"""Node structure definition and mapping to database."""

import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    ForeignKey,
    Integer,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, foreign, mapped_column, relationship

from src.core.database.base import AbstractBase
from src.schemas.protocols import PydanticBaseModelProtocol


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

    @classmethod
    def from_pydantic_model(
        cls: type["BaseNode"],
        pydantic_model: PydanticBaseModelProtocol,
    ) -> "BaseNode":
        """Constructor: build an ORM model from a Pydantic model."""
        return cls(**pydantic_model.model_dump())

    def __repr__(self) -> str:
        return f"<node_id={self.id} text={self.text.strip()[:50]} node_metadata={self.node_metadata}>"


class TextNode(BaseNode):
    """Text Node Object.

    Generic interface for retrievable nodes containing text data.
    """

    __tablename__: str = "document_chunks"

    max_char_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    start_char_index: Mapped[int | None] = mapped_column(
        Integer(),
        nullable=True,
        default=None,
    )

    end_char_index: Mapped[int | None] = mapped_column(
        Integer(),
        nullable=True,
        default=None,
    )

    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id"),
        nullable=True,
        default=None,
    )

    parent_id: Mapped[uuid.UUID] = mapped_column(
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

    source: Mapped["DocumentNode"] = relationship(back_populates="children")

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

    @property
    def children_ids(self) -> list[uuid.UUID]:
        """Returns a list of IDs of the children nodes."""
        return [child.id for child in self.children]

    @property
    def next_id(self) -> uuid.UUID | None:
        """Convenience property to get next_id without loading the object."""
        return self.next_node.id if self.next_node else None


class DocumentNode(BaseNode):
    """Document node object."""

    __tablename__: str = "documents"

    source_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    children: Mapped[list["TextNode"]] = relationship(
        back_populates="source",
        cascade="all, delete-orphan",
    )

    @property
    def children_ids(self) -> list[uuid.UUID]:
        """Returns a list of IDs of the children nodes."""
        return [child.id for child in self.children]
