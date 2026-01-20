"""Node structure definition and mapping to database."""

import uuid

from pydantic import BaseModel, ConfigDict

from src.schemas.protocols import ORMBaseNodeProtocol


DEFAULT_TEXT_NODE_TMPL = "{metadata_str}\n\n{content}"
DEFAULT_METADATA_TMPL = "{key}: {value}"

TRUNCATE_LENGTH = 350
WRAP_WIDTH = 70


class BaseNode(BaseModel):
    """Pydantic model of a Base Node."""

    text: str
    embedding: list[float] | None = None
    node_metadata: dict | None = None

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        from_attributes=True,
    )


class BaseNodeCreate(BaseNode):
    """Pydantic model for a base node input."""


class BaseNodeRead(BaseNode):
    """Pydantic model for a base node output."""

    id: uuid.UUID

    @classmethod
    def from_orm_model(cls, orm_model: ORMBaseNodeProtocol) -> "BaseNode":
        """Validate a SQLAlchemy ORM model for a BaseNode."""
        return cls.model_validate(orm_model)


class TextNodeCreate(BaseNodeCreate):
    """Text node create validation model."""

    max_char_size: int
    start_char_index: int
    end_char_index: int


class TextNodeRead(BaseNodeRead):
    """Text node read validation model."""

    max_char_size: int
    start_char_index: int
    end_char_index: int
    source_id: uuid.UUID | None = None
    parent_id: uuid.UUID | None = None
    children_ids: list[uuid.UUID] | None = None
    prev_id: uuid.UUID | None = None
    next_id: uuid.UUID | None = None


class DocumentNodeCreate(BaseNodeCreate):
    """Document node create validation model."""


class DocumentNodeRead(BaseNodeRead):
    """Document node read validation model."""

    children_ids: list[uuid.UUID] | None = None


def get_node_metadata_str(
    text_node: BaseNode,
    metadata_template: str = DEFAULT_METADATA_TMPL,
) -> str:
    """Get metadata string formatted according to metadata template."""
    if text_node.node_metadata is not None:
        return "\n".join(
            metadata_template.format(key=k, value=str(v))
            for k, v in text_node.node_metadata.items()
        )
    return ""


def get_content(
    text_node: BaseNode,
    text_template: str = DEFAULT_TEXT_NODE_TMPL,
) -> str:
    """Get object content according to the text template, using the specified metadata mode."""
    metadata_str = get_node_metadata_str(text_node).strip()

    return text_template.format(
        content=text_node.text,
        metadata_str=metadata_str,
    ).strip()
