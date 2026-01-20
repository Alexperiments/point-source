"""Protocols."""

from typing import Protocol


class PydanticBaseModelProtocol(Protocol):
    """Pydantic BaseModel protocol, to avoid circular dependencies between ORM layer and app layer."""

    def model_dump(self) -> dict[str, object]:
        """Generate a dictionary representation of the model, optionally specifying which fields to include or exclude."""
        ...


class ORMBaseNodeProtocol(Protocol):
    """ORM BaseNode protocol, to avoid circular dependencies between ORM layer and app layer."""
