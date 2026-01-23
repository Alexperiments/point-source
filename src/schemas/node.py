"""Node structure definition and mapping to database."""

import uuid
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    ValidationInfo,
    model_validator,
)


DEFAULT_TEXT_NODE_TMPL = "{metadata_str}\n\n{content}"
DEFAULT_METADATA_TMPL = "{key}: {value}"


def is_positive(value: int) -> int:
    """Utility function for pydantic validators."""
    if value < 0:
        raise ValueError(f"{value} is not a positive number")
    return value


PositiveInteger = Annotated[int, AfterValidator(is_positive)]


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

    @model_validator(mode="before")
    @classmethod
    def apply_default_metadata(cls, data: object, info: ValidationInfo) -> object:
        """Merge base_metadata from validation context into node_metadata.

        - context["base_metadata"] comes from validate_python(..., context=...)
        - user-provided node_metadata wins over base_metadata on key conflicts
        """
        if not isinstance(data, dict):
            return data

        base_metadata = (info.context or {}).get("base_metadata") or {}
        if not base_metadata:
            return data

        node_metadata = data.get("node_metadata")
        if node_metadata is None:
            data["node_metadata"] = base_metadata
        else:
            data["node_metadata"] = {**base_metadata, **node_metadata}
        return data


class BaseNodeRead(BaseNode):
    """Pydantic model for a base node output."""

    id: uuid.UUID


class TextNodeCreate(BaseNodeCreate):
    """Text node create validation model."""

    max_length: PositiveInteger
    start_index: PositiveInteger
    end_index: PositiveInteger

    @model_validator(mode="after")
    def check_text_length(self) -> "TextNodeCreate":
        """After validator to check text length."""
        if len(self.text) > self.max_length:
            raise ValueError(f"text is longer than max_length: {self.max_length}")
        if len(self.text) != (self.end_index - self.start_index):
            raise ValueError(
                f"text length ({len(self.text)}) is different from the length implied by the indexes ({self.end_index - self.start_index})",
            )
        return self

    @model_validator(mode="after")
    def check_start_end_consistency(self) -> "TextNodeCreate":
        """After validator to check that end_index >= start_index."""
        if self.end_index < self.start_index:
            raise ValueError(
                f"end_index is smaller than start_index ({self.end_index} < {self.start_index}).",
            )
        return self


class TextNodeRead(BaseNodeRead):
    """Text node read validation model."""

    max_char_size: PositiveInteger
    start_char_index: PositiveInteger
    end_char_index: PositiveInteger
    source_id: uuid.UUID | None = None
    parent_id: uuid.UUID | None = None
    children_ids: list[uuid.UUID] | None = None
    prev_id: uuid.UUID | None = None
    next_id: uuid.UUID | None = None


class DocumentNodeCreate(BaseNodeCreate):
    """Document node create validation model."""

    source_id: str


class DocumentNodeRead(BaseNodeRead):
    """Document node read validation model."""

    source_id: str
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
