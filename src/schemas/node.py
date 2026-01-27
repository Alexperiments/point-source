"""Node structure definition and mapping to database."""

import uuid

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    model_validator,
)


DEFAULT_TEXT_NODE_TMPL = "{metadata_str}\n\n{content}"
DEFAULT_METADATA_TMPL = "{key}: {value}"


class BaseNode(BaseModel):
    """Pydantic model of a Base Node."""

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        from_attributes=True,
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    text: str
    embedding: list[float] | None = None
    node_metadata: dict | None = None

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

    def get_node_metadata_str(
        self,
        metadata_template: str = DEFAULT_METADATA_TMPL,
    ) -> str:
        """Get metadata string formatted according to metadata template."""
        if self.node_metadata is not None:
            return "\n".join(
                metadata_template.format(key=k, value=str(v))
                for k, v in self.node_metadata.items()
            )
        return ""

    def get_content(
        self,
        text_template: str = DEFAULT_TEXT_NODE_TMPL,
    ) -> str:
        """Get object content according to the text template, using the specified metadata mode."""
        metadata_str = self.get_node_metadata_str().strip()

        return text_template.format(
            content=self.text,
            metadata_str=metadata_str,
        ).strip()


class TextNode(BaseNode):
    """A hierarchical chunk of text extracted from a document."""

    title: str
    path: str
    document_id: uuid.UUID | None = None
    parent_id: uuid.UUID | None = None
    prev_id: uuid.UUID | None = None
    next_id: uuid.UUID | None = None
    children_ids: list[uuid.UUID] = Field(default_factory=list)


class DocumentNodeCreate(BaseNode):
    """Document node create validation model."""

    source_id: str


class DocumentNodeRead(BaseNode):
    """Document node read validation model."""

    source_id: str
    children_ids: list[uuid.UUID] | None = None
