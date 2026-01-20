import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db_models.node import BaseNode, DocumentNode, TextNode
from src.database.database import AbstractBase


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def engine():
    engine = create_engine("sqlite:///:memory:", echo=False)
    BaseNode.metadata.create_all(engine)
    yield engine
    BaseNode.metadata.drop_all(engine)

# 2. The transactional session fixture
@pytest.fixture
def session(engine):
    connection = engine.connect()
    transaction = connection.begin()

    # Bind a session to the connection
    SessionLocal = sessionmaker(bind=connection)
    session = SessionLocal()

    # Begin a nested transaction (savepoint)
    nested = connection.begin_nested()

    @pytest.fixture(autouse=True)
    def rollback_nested():
        # This ensures that after the test, we roll back to the savepoint
        yield
        if nested.is_active:
            nested.rollback()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def sample_document_node(session):
    """Create and persist a sample document node."""
    doc = DocumentNode(
        text="Document content",
        node_metadata={"doc_key": "doc_value"},
    )
    session.add(doc)
    session.flush()
    return doc


@pytest.fixture
def sample_text_node(session, sample_document_node):
    """Create and persist a sample text node."""
    node = TextNode(
        text="Sample text",
        node_metadata={"key1": "value1", "key2": "value2"},
        embedding=[0.1] * 768,  # Match the Vector(768) dimension
        max_char_size=1000,
        source_id=sample_document_node.id,
        parent_id=uuid.uuid4(),  # Placeholder parent
    )
    session.add(node)
    session.flush()
    return node


# ============================================================================
# BaseNode Initialization (via TextNode)
# ============================================================================

class TestBaseNodeInitialization:
    def test_default_initialization(self, session, sample_document_node):
        """Test creating a TextNode with minimal required fields."""
        parent_id = uuid.uuid4()
        node = TextNode(
            text="Test",
            max_char_size=1000,
            source_id=sample_document_node.id,
            parent_id=parent_id,
        )
        session.add(node)
        session.flush()

        assert node.id is not None
        assert node.text == "Test"
        assert node.node_metadata is None
        assert node.start_char_index is None
        assert node.end_char_index is None
        assert node.embedding is None
        assert node.max_char_size == 1000

    def test_initialization_with_all_parameters(self, session, sample_document_node):
        """Test creating a TextNode with all fields populated."""
        node_id = uuid.uuid4()
        parent_id = uuid.uuid4()
        prev_id = uuid.uuid4()

        node = TextNode(
            id=node_id,
            text="Sample text",
            node_metadata={"key": "value"},
            embedding=[0.1] * 768,
            max_char_size=1000,
            start_char_index=0,
            end_char_index=11,
            source_id=sample_document_node.id,
            parent_id=parent_id,
            prev_id=prev_id,
        )
        session.add(node)
        session.flush()

        assert node.id == node_id
        assert node.start_char_index == 0
        assert node.end_char_index == 11
        assert node.node_metadata == {"key": "value"}
        assert node.embedding == [0.1] * 768

    def test_none_metadata_stays_none(self, session, sample_document_node):
        """Test that None metadata is preserved as None."""
        parent_id = uuid.uuid4()
        node = TextNode(
            text="Test",
            node_metadata=None,
            max_char_size=1000,
            source_id=sample_document_node.id,
            parent_id=parent_id,
        )
        session.add(node)
        session.flush()

        assert node.node_metadata is None


# ============================================================================
# Embedding
# ============================================================================

class TestEmbedding:
    def test_embedding_storage_and_retrieval(self, session, sample_document_node):
        """Test that embeddings are correctly stored and retrieved."""
        parent_id = uuid.uuid4()
        embedding = [0.1, 0.2, 0.3] + [0.0] * 765  # 768 dimensions total
        node = TextNode(
            text="Test",
            embedding=embedding,
            max_char_size=1000,
            source_id=sample_document_node.id,
            parent_id=parent_id,
        )
        session.add(node)
        session.flush()

        assert node.embedding == embedding

    def test_none_embedding_allowed(self, session, sample_document_node):
        """Test that None embedding is allowed."""
        parent_id = uuid.uuid4()
        node = TextNode(
            text="Test",
            embedding=None,
            max_char_size=1000,
            source_id=sample_document_node.id,
            parent_id=parent_id,
        )
        session.add(node)
        session.flush()

        assert node.embedding is None


# ============================================================================
# String Representation
# ============================================================================

class TestStringRepresentation:
    def test_repr_includes_id_and_text(self, session, sample_document_node):
        """Test that __repr__ includes node ID and text."""
        parent_id = uuid.uuid4()
        node = TextNode(
            text="Sample text",
            max_char_size=1000,
            source_id=sample_document_node.id,
            parent_id=parent_id,
        )
        session.add(node)
        session.flush()

        repr_str = repr(node)
        assert f"node_id={node.id}" in repr_str
        assert "text=Sample text" in repr_str


# ============================================================================
# Relationships
# ============================================================================

class TestTextNodeRelationships:
    def test_children_ids_empty_by_default(self, session, sample_document_node):
        """Test that a node with no children has an empty children_ids list."""
        parent_id = uuid.uuid4()
        parent = TextNode(
            text="Parent",
            max_char_size=1000,
            source_id=sample_document_node.id,
            parent_id=parent_id,
        )
        session.add(parent)
        session.flush()

        assert parent.children_ids == []

    def test_parent_child_relationship(self, session, sample_document_node):
        """Test parent-child relationship between text nodes."""
        grandparent_id = uuid.uuid4()
        parent = TextNode(
            text="Parent",
            max_char_size=1000,
            source_id=sample_document_node.id,
            parent_id=grandparent_id,
        )
        session.add(parent)
        session.flush()

        child = TextNode(
            text="Child",
            max_char_size=500,
            source_id=sample_document_node.id,
            parent_id=parent.id,
        )
        session.add(child)
        session.flush()

        # Refresh to load relationships
        session.refresh(parent)
        session.refresh(child)

        assert child.parent_id == parent.id
        assert child in parent.children
        assert parent.children_ids == [child.id]

    def test_prev_next_chain(self, session, sample_document_node):
        """Test previous/next node chaining."""
        parent_id = uuid.uuid4()
        n1 = TextNode(
            text="one",
            max_char_size=1000,
            source_id=sample_document_node.id,
            parent_id=parent_id,
        )
        session.add(n1)
        session.flush()

        n2 = TextNode(
            text="two",
            max_char_size=1000,
            source_id=sample_document_node.id,
            parent_id=parent_id,
            prev_id=n1.id,
        )
        session.add(n2)
        session.flush()

        # Refresh to load relationships
        session.refresh(n1)
        session.refresh(n2)

        assert n1.next_id == n2.id
        assert n2.prev_id == n1.id
        assert n1.next_node == n2
        assert n2.prev_node == n1

    def test_source_relationship(self, session, sample_document_node):
        """Test that TextNode correctly references its DocumentNode source."""
        parent_id = uuid.uuid4()
        chunk = TextNode(
            text="Chunk",
            max_char_size=1000,
            source_id=sample_document_node.id,
            parent_id=parent_id,
        )
        session.add(chunk)
        session.flush()
        session.refresh(chunk)
        session.refresh(sample_document_node)

        assert chunk.source_id == sample_document_node.id
        assert chunk.source == sample_document_node


# ============================================================================
# DocumentNode
# ============================================================================

class TestDocumentNode:
    def test_creation_minimal(self, session):
        """Test creating a DocumentNode with minimal fields."""
        node = DocumentNode(text="Document")
        session.add(node)
        session.flush()

        assert node.id is not None
        assert node.text == "Document"
        assert node.node_metadata is None

    def test_children_relationship(self, session):
        """Test DocumentNode's relationship with TextNode children."""
        doc = DocumentNode(text="Document")
        session.add(doc)
        session.flush()

        parent_id = uuid.uuid4()
        child = TextNode(
            text="Chunk",
            max_char_size=1000,
            source_id=doc.id,
            parent_id=parent_id,
        )
        session.add(child)
        session.flush()

        session.refresh(doc)
        session.refresh(child)

        assert doc.children_ids == [child.id]
        assert child.source_id == doc.id
        assert child in doc.children

    def test_multiple_children(self, session):
        """Test DocumentNode with multiple TextNode children."""
        doc = DocumentNode(text="Document")
        session.add(doc)
        session.flush()

        parent_id = uuid.uuid4()
        children = []
        for i in range(3):
            child = TextNode(
                text=f"Chunk {i}",
                max_char_size=1000,
                source_id=doc.id,
                parent_id=parent_id,
            )
            children.append(child)
            session.add(child)

        session.flush()
        session.refresh(doc)

        assert len(doc.children_ids) == 3
        for c in children:
            assert c.id in doc.children_ids


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_empty_text(self, session, sample_document_node):
        """Test that empty text is allowed."""
        parent_id = uuid.uuid4()
        node = TextNode(
            text="",
            max_char_size=1000,
            source_id=sample_document_node.id,
            parent_id=parent_id,
        )
        session.add(node)
        session.flush()

        assert node.text == ""

    def test_metadata_with_none_value(self, session, sample_document_node):
        """Test metadata with None as a value."""
        parent_id = uuid.uuid4()
        node = TextNode(
            text="Text",
            node_metadata={"key": None},
            max_char_size=1000,
            source_id=sample_document_node.id,
            parent_id=parent_id,
        )
        session.add(node)
        session.flush()

        assert node.node_metadata == {"key": None}

    def test_unicode_metadata(self, session, sample_document_node):
        """Test that unicode characters work in metadata."""
        parent_id = uuid.uuid4()
        node = TextNode(
            text="Text",
            node_metadata={"键": "值", "emoji": "🎉"},
            max_char_size=1000,
            source_id=sample_document_node.id,
            parent_id=parent_id,
        )
        session.add(node)
        session.flush()

        assert node.node_metadata == {"键": "值", "emoji": "🎉"}

    def test_next_id_none_when_no_next(self, session, sample_document_node):
        """Test that next_id returns None when there is no next node."""
        parent_id = uuid.uuid4()
        node = TextNode(
            text="Isolated",
            max_char_size=1000,
            source_id=sample_document_node.id,
            parent_id=parent_id,
        )
        session.add(node)
        session.flush()

        assert node.next_id is None

    def test_cascade_delete_document(self, session, sample_document_node):
        doc_id = sample_document_node.id
        child = TextNode(text="child", max_char_size=100, source_id=doc_id, parent_id=uuid.uuid4())
        session.add(child)
        session.commit()

        session.delete(sample_document_node)
        session.commit()

        assert session.get(TextNode, child.id) is None


    def test_node_chain_integrity(self, session, sample_document_node):
        """Test a 3-node sequence to verify sibling and parent relationships."""
        parent_id = uuid.uuid4()

        # Create 3 nodes in a sequence
        n1 = TextNode(text="Node 1", max_char_size=100, source_id=sample_document_node.id, parent_id=parent_id)
        session.add(n1)
        session.flush()

        n2 = TextNode(text="Node 2", max_char_size=100, source_id=sample_document_node.id, parent_id=parent_id, prev_id=n1.id)
        session.add(n2)
        session.flush()

        n3 = TextNode(text="Node 3", max_char_size=100, source_id=sample_document_node.id, parent_id=parent_id, prev_id=n2.id)
        session.add(n3)
        session.flush()

        # Verify the chain forward and backward
        assert n1.next_node == n2
        assert n2.next_node == n3
        assert n3.prev_node == n2
        assert n2.prev_node == n1

        # Verify convenience property
        assert n1.next_id == n2.id
        assert n2.next_id == n3.id


# ============================================================================
# from_pydantic_model
# ============================================================================

class TestFromPydanticModel:
    def test_from_pydantic_model(self, session):
        """Test creating a node from a Pydantic model."""
        from unittest.mock import Mock

        # Create a mock Pydantic model
        pydantic_model = Mock()
        pydantic_model.model_dump.return_value = {
            "text": "Test document",
            "node_metadata": {"key": "value"},
        }

        node = DocumentNode.from_pydantic_model(pydantic_model)
        session.add(node)
        session.flush()

        assert node.text == "Test document"
        assert node.node_metadata == {"key": "value"}
        pydantic_model.model_dump.assert_called_once()
