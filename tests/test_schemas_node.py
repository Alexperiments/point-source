"""Tests for src/schemas/node.py module."""
import uuid
import pytest
from pydantic import ValidationError
from src.schemas.node import (
    BaseNode,
    BaseNodeCreate,
    BaseNodeRead,
    TextNodeCreate,
    TextNodeRead,
    DocumentNodeCreate,
    DocumentNodeRead,
    get_node_metadata_str,
    get_content,
    DEFAULT_TEXT_NODE_TMPL,
    DEFAULT_METADATA_TMPL,
)


class TestBaseNode:
    """Tests for BaseNode class."""

    def test_base_node_minimal_creation(self):
        """Test creating a BaseNode with only required fields."""
        node = BaseNode(text="Sample text")
        assert node.text == "Sample text"
        assert node.embedding is None
        assert node.node_metadata is None

    def test_base_node_with_embedding(self):
        """Test creating a BaseNode with embedding."""
        embedding = [0.1, 0.2, 0.3]
        node = BaseNode(text="Sample text", embedding=embedding)
        assert node.embedding == embedding

    def test_base_node_with_metadata(self):
        """Test creating a BaseNode with metadata."""
        metadata = {"key1": "value1", "key2": 123}
        node = BaseNode(text="Sample text", node_metadata=metadata)
        assert node.node_metadata == metadata

    def test_base_node_empty_embedding(self):
        """Test creating a BaseNode with empty embedding list."""
        node = BaseNode(text="Sample text", embedding=[])
        assert node.embedding == []

    def test_base_node_empty_metadata(self):
        """Test creating a BaseNode with empty metadata dict."""
        node = BaseNode(text="Sample text", node_metadata={})
        assert node.node_metadata == {}

    def test_base_node_missing_text_raises_error(self):
        """Test that missing text field raises validation error."""
        with pytest.raises(ValidationError):
            BaseNode() # pyright: ignore[reportCallIssue]

    def test_base_node_validate_assignment(self):
        """Test that validate_assignment config works."""
        node = BaseNode(text="Original")
        node.text = "Updated"
        assert node.text == "Updated"

    def test_base_node_nested_metadata(self):
        """Test BaseNode with nested metadata structures."""
        metadata = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "mixed": {"a": [1, 2], "b": "text"}
        }
        node = BaseNode(text="text", node_metadata=metadata)
        assert node.node_metadata == metadata


class TestBaseNodeCreate:
    """Tests for BaseNodeCreate class."""

    def test_base_node_create_inherits_from_base_node(self):
        """Test that BaseNodeCreate has same fields as BaseNode."""
        node = BaseNodeCreate(text="Sample text")
        assert hasattr(node, "text")
        assert hasattr(node, "embedding")
        assert hasattr(node, "node_metadata")

    def test_base_node_create_with_all_fields(self):
        """Test BaseNodeCreate with all fields populated."""
        node = BaseNodeCreate(
            text="Sample text",
            embedding=[0.1, 0.2],
            node_metadata={"key": "value"}
        )
        assert node.text == "Sample text"
        assert node.embedding == [0.1, 0.2]
        assert node.node_metadata == {"key": "value"}


class TestBaseNodeRead:
    """Tests for BaseNodeRead class."""

    def test_base_node_read_requires_id(self):
        """Test that BaseNodeRead requires an id field."""
        node_id = uuid.uuid4()
        node = BaseNodeRead(text="Sample text", id=node_id)
        assert node.id == node_id

    def test_base_node_read_missing_id_raises_error(self):
        """Test that missing id raises validation error."""
        with pytest.raises(ValidationError):
            BaseNodeRead(text="Sample text") # type: ignore

    def test_base_node_read_invalid_id_type_raises_error(self):
        """Test that invalid id type raises validation error."""
        with pytest.raises(ValidationError):
            BaseNodeRead(text="Sample text", id="not-a-uuid") # type: ignore

    def test_base_node_read_with_all_fields(self):
        """Test BaseNodeRead with all fields."""
        node_id = uuid.uuid4()
        node = BaseNodeRead(
            text="Sample text",
            id=node_id,
            embedding=[0.1, 0.2],
            node_metadata={"key": "value"}
        )
        assert node.id == node_id
        assert node.text == "Sample text"

    def test_from_orm_model(self):
        """Test from_orm_model class method."""
        # Mock ORM model
        class MockORMModel:
            def __init__(self):
                self.id = uuid.uuid4()
                self.text = "ORM text"
                self.embedding = [0.1, 0.2]
                self.node_metadata = {"key": "value"}

        orm_model = MockORMModel()
        node = BaseNodeRead.from_orm_model(orm_model)
        assert node.id == orm_model.id # pyright: ignore[reportAttributeAccessIssue]
        assert node.text == orm_model.text
        assert node.embedding == orm_model.embedding
        assert node.node_metadata == orm_model.node_metadata


class TestTextNodeCreate:
    """Tests for TextNodeCreate class."""

    def test_text_node_create_with_required_fields(self):
        """Test creating TextNodeCreate with required fields."""
        node = TextNodeCreate(
            text="Sample text",
            max_char_size=100,
            start_char_index=0,
            end_char_index=11
        )
        assert node.text == "Sample text"
        assert node.max_char_size == 100
        assert node.start_char_index == 0
        assert node.end_char_index == 11

    def test_text_node_create_missing_char_fields_raises_error(self):
        """Test that missing char fields raises validation error."""
        with pytest.raises(ValidationError):
            TextNodeCreate(text="Sample text") # pyright: ignore[reportCallIssue]

    def test_text_node_create_negative_indices(self):
        """Test TextNodeCreate accepts negative indices."""
        node = TextNodeCreate(
            text="Sample text",
            max_char_size=100,
            start_char_index=-5,
            end_char_index=5
        )
        assert node.start_char_index == -5
        assert node.end_char_index == 5

    def test_text_node_create_zero_max_char_size(self):
        """Test TextNodeCreate accepts zero max_char_size."""
        node = TextNodeCreate(
            text="",
            max_char_size=0,
            start_char_index=0,
            end_char_index=0
        )
        assert node.max_char_size == 0


class TestTextNodeRead:
    """Tests for TextNodeRead class."""

    def test_text_node_read_with_required_fields(self):
        """Test creating TextNodeRead with required fields."""
        node_id = uuid.uuid4()
        node = TextNodeRead(
            id=node_id,
            text="Sample text",
            max_char_size=100,
            start_char_index=0,
            end_char_index=11
        )
        assert node.id == node_id
        assert node.max_char_size == 100

    def test_text_node_read_with_optional_relationships(self):
        """Test TextNodeRead with all optional relationship fields."""
        node_id = uuid.uuid4()
        source_id = uuid.uuid4()
        parent_id = uuid.uuid4()
        child_id1 = uuid.uuid4()
        child_id2 = uuid.uuid4()
        prev_id = uuid.uuid4()
        next_id = uuid.uuid4()

        node = TextNodeRead(
            id=node_id,
            text="Sample text",
            max_char_size=100,
            start_char_index=0,
            end_char_index=11,
            source_id=source_id,
            parent_id=parent_id,
            children_ids=[child_id1, child_id2],
            prev_id=prev_id,
            next_id=next_id
        )
        assert node.source_id == source_id
        assert node.parent_id == parent_id
        assert len(node.children_ids) == 2
        assert node.prev_id == prev_id
        assert node.next_id == next_id

    def test_text_node_read_empty_children_ids(self):
        """Test TextNodeRead with empty children_ids list."""
        node_id = uuid.uuid4()
        node = TextNodeRead(
            id=node_id,
            text="Sample text",
            max_char_size=100,
            start_char_index=0,
            end_char_index=11,
            children_ids=[]
        )
        assert node.children_ids == []

    def test_text_node_read_null_optional_fields(self):
        """Test that optional fields default to None."""
        node_id = uuid.uuid4()
        node = TextNodeRead(
            id=node_id,
            text="Sample text",
            max_char_size=100,
            start_char_index=0,
            end_char_index=11
        )
        assert node.source_id is None
        assert node.parent_id is None
        assert node.children_ids is None
        assert node.prev_id is None
        assert node.next_id is None


class TestDocumentNodeCreate:
    """Tests for DocumentNodeCreate class."""

    def test_document_node_create_minimal(self):
        """Test creating DocumentNodeCreate with minimal fields."""
        node = DocumentNodeCreate(text="Document text")
        assert node.text == "Document text"

    def test_document_node_create_with_all_fields(self):
        """Test DocumentNodeCreate with all fields."""
        node = DocumentNodeCreate(
            text="Document text",
            embedding=[0.1, 0.2, 0.3],
            node_metadata={"title": "My Document", "author": "John Doe"}
        )
        assert node.text == "Document text"
        assert node.embedding == [0.1, 0.2, 0.3]
        assert node.node_metadata["title"] == "My Document"


class TestDocumentNodeRead:
    """Tests for DocumentNodeRead class."""

    def test_document_node_read_with_required_fields(self):
        """Test creating DocumentNodeRead with required fields."""
        node_id = uuid.uuid4()
        node = DocumentNodeRead(id=node_id, text="Document text")
        assert node.id == node_id
        assert node.text == "Document text"

    def test_document_node_read_with_children_ids(self):
        """Test DocumentNodeRead with children_ids."""
        node_id = uuid.uuid4()
        child_id1 = uuid.uuid4()
        child_id2 = uuid.uuid4()
        child_id3 = uuid.uuid4()

        node = DocumentNodeRead(
            id=node_id,
            text="Document text",
            children_ids=[child_id1, child_id2, child_id3]
        )
        assert len(node.children_ids) == 3
        assert child_id1 in node.children_ids

    def test_document_node_read_empty_children_ids(self):
        """Test DocumentNodeRead with empty children_ids."""
        node_id = uuid.uuid4()
        node = DocumentNodeRead(
            id=node_id,
            text="Document text",
            children_ids=[]
        )
        assert node.children_ids == []

    def test_document_node_read_null_children_ids(self):
        """Test DocumentNodeRead with null children_ids."""
        node_id = uuid.uuid4()
        node = DocumentNodeRead(id=node_id, text="Document text")
        assert node.children_ids is None


class TestGetNodeMetadataStr:
    """Tests for get_node_metadata_str function."""

    def test_get_node_metadata_str_with_metadata(self):
        """Test getting metadata string with metadata present."""
        node = BaseNode(
            text="Sample text",
            node_metadata={"key1": "value1", "key2": "value2"}
        )
        result = get_node_metadata_str(node)
        assert "key1: value1" in result
        assert "key2: value2" in result

    def test_get_node_metadata_str_without_metadata(self):
        """Test getting metadata string when node_metadata is None."""
        node = BaseNode(text="Sample text")
        result = get_node_metadata_str(node)
        assert result == ""

    def test_get_node_metadata_str_with_empty_metadata(self):
        """Test getting metadata string with empty metadata dict."""
        node = BaseNode(text="Sample text", node_metadata={})
        result = get_node_metadata_str(node)
        assert result == ""

    def test_get_node_metadata_str_custom_template(self):
        """Test getting metadata string with custom template."""
        node = BaseNode(
            text="Sample text",
            node_metadata={"author": "John", "title": "Test"}
        )
        custom_template = "{key}={value}"
        result = get_node_metadata_str(node, metadata_template=custom_template)
        assert "author=John" in result
        assert "title=Test" in result

    def test_get_node_metadata_str_with_numeric_values(self):
        """Test metadata string converts numeric values to strings."""
        node = BaseNode(
            text="Sample text",
            node_metadata={"count": 42, "score": 3.14}
        )
        result = get_node_metadata_str(node)
        assert "count: 42" in result
        assert "score: 3.14" in result

    def test_get_node_metadata_str_with_complex_values(self):
        """Test metadata string converts complex values to strings."""
        node = BaseNode(
            text="Sample text",
            node_metadata={
                "list": [1, 2, 3],
                "dict": {"nested": "value"}
            }
        )
        result = get_node_metadata_str(node)
        assert "list:" in result
        assert "dict:" in result

    def test_get_node_metadata_str_newline_separation(self):
        """Test that metadata entries are separated by newlines."""
        node = BaseNode(
            text="Sample text",
            node_metadata={"first": "1", "second": "2", "third": "3"}
        )
        result = get_node_metadata_str(node)
        lines = result.split("\n")
        assert len(lines) == 3


class TestGetContent:
    """Tests for get_content function."""

    def test_get_content_with_metadata(self):
        """Test getting content with metadata."""
        node = BaseNode(
            text="Main content",
            node_metadata={"key": "value"}
        )
        result = get_content(node)
        assert "Main content" in result
        assert "key: value" in result

    def test_get_content_without_metadata(self):
        """Test getting content without metadata returns only text."""
        node = BaseNode(text="Main content")
        result = get_content(node)
        assert result == "Main content"

    def test_get_content_with_empty_metadata(self):
        """Test getting content with empty metadata returns only text."""
        node = BaseNode(text="Main content", node_metadata={})
        result = get_content(node)
        assert result == "Main content"

    def test_get_content_custom_template(self):
        """Test getting content with custom template."""
        node = BaseNode(
            text="Main content",
            node_metadata={"key": "value"}
        )
        custom_template = "Metadata: {metadata_str}\nContent: {content}"
        result = get_content(node, text_template=custom_template)
        assert "Metadata: key: value" in result
        assert "Content: Main content" in result

    def test_get_content_strips_whitespace(self):
        """Test that get_content strips final result."""
        node = BaseNode(text="Content")
        result = get_content(node)
        assert result == result.strip()

    def test_get_content_with_multiple_metadata_entries(self):
        """Test content formatting with multiple metadata entries."""
        node = BaseNode(
            text="Main content",
            node_metadata={
                "author": "John Doe",
                "date": "2024-01-01",
                "version": "1.0"
            }
        )
        result = get_content(node)
        assert "Main content" in result
        assert "author: John Doe" in result
        assert "date: 2024-01-01" in result
        assert "version: 1.0" in result

    def test_get_content_template_with_only_content(self):
        """Test custom template with only content placeholder."""
        node = BaseNode(
            text="Main content",
            node_metadata={"key": "value"}
        )
        result = get_content(node, text_template="{content}")
        assert result == "Main content"
        assert "key: value" not in result

    def test_get_content_template_with_only_metadata(self):
        """Test custom template with only metadata placeholder."""
        node = BaseNode(
            text="Main content",
            node_metadata={"key": "value"}
        )
        result = get_content(node, text_template="{metadata_str}")
        assert result == "key: value"
        assert "Main content" not in result

    def test_get_content_default_template_format(self):
        """Test that default template uses correct format."""
        node = BaseNode(
            text="Content text",
            node_metadata={"key": "value"}
        )
        result = get_content(node)
        # Default template is "{metadata_str}\n\n{content}"
        expected = "key: value\n\nContent text"
        assert result == expected


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_metadata_with_none_values(self):
        """Test metadata with None values are converted to string."""
        node = BaseNode(
            text="text",
            node_metadata={"key": None}
        )
        result = get_node_metadata_str(node)
        assert "key: None" in result

    def test_metadata_with_boolean_values(self):
        """Test metadata with boolean values are converted to string."""
        node = BaseNode(
            text="text",
            node_metadata={"is_active": True, "is_deleted": False}
        )
        result = get_node_metadata_str(node)
        assert "is_active: True" in result
        assert "is_deleted: False" in result

    def test_text_node_with_end_index_before_start(self):
        """Test TextNodeCreate accepts logically invalid index ordering."""
        node = TextNodeCreate(
            text="text",
            max_char_size=100,
            start_char_index=50,
            end_char_index=10
        )
        assert node.start_char_index > node.end_char_index

    def test_duplicate_uuids_in_children_ids(self):
        """Test nodes accept duplicate UUIDs in children_ids."""
        node_id = uuid.uuid4()
        child_id = uuid.uuid4()
        node = TextNodeRead(
            id=node_id,
            text="text",
            max_char_size=100,
            start_char_index=0,
            end_char_index=10,
            children_ids=[child_id, child_id, child_id]
        )
        assert len(node.children_ids) == 3
        assert all(cid == child_id for cid in node.children_ids)

    def test_circular_reference_ids(self):
        """Test nodes accept self-referencing IDs."""
        node_id = uuid.uuid4()
        node = TextNodeRead(
            id=node_id,
            text="text",
            max_char_size=100,
            start_char_index=0,
            end_char_index=10,
            parent_id=node_id,
            children_ids=[node_id]
        )
        assert node.parent_id == node.id
        assert node.id in node.children_ids

    def test_invalid_uuid_string_in_create_raises_error(self):
        """Test that invalid UUID strings raise validation error."""
        with pytest.raises(ValidationError):
            BaseNodeRead(text="text", id="invalid-uuid-string")

    def test_integer_as_uuid_raises_error(self):
        """Test that integer values for UUID fields raise validation error."""
        with pytest.raises(ValidationError):
            BaseNodeRead(text="text", id=12345)
