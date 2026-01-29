import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.database.base import Base
from src.models.node import DocumentNode, TextNode


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def engine():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def session(engine):
    connection = engine.connect()
    transaction = connection.begin()

    SessionLocal = sessionmaker(bind=connection)
    session = SessionLocal()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def sample_document_node(session):
    """Create and persist a sample document node."""
    doc = DocumentNode(
        text="Document content",
        source_id="source-1",
    )
    doc.node_metadata = {"doc_key": "doc_value"}
    session.add(doc)
    session.flush()
    return doc


# ============================================================================
# BaseNode Initialization (via TextNode)
# ============================================================================

class TestBaseNodeInitialization:
    def test_default_initialization(self, session):
        """Test creating a TextNode with only required fields."""
        node = TextNode(text="Test")
        session.add(node)
        session.flush()

        assert node.id is not None
        assert node.text == "Test"
        assert node.node_metadata is None
        assert node.embedding is None
        assert node.document_id is None
        assert node.parent_id is None
        assert node.prev_id is None

    def test_setting_metadata_and_embedding(self, session):
        """Test assigning metadata and embeddings after creation."""
        node = TextNode(text="Sample text")
        node.node_metadata = {"key": "value"}
        node.embedding = [0.1] * 1024
        session.add(node)
        session.flush()

        assert node.node_metadata == {"key": "value"}
        assert node.embedding == [0.1] * 768


# ============================================================================
# String Representation
# ============================================================================

class TestStringRepresentation:
    def test_repr_includes_id_text_and_metadata(self, session):
        """Test that __repr__ includes node ID, text, and metadata."""
        node = TextNode(text="  Sample text  ")
        node.node_metadata = {"key": "value"}
        session.add(node)
        session.flush()

        repr_str = repr(node)
        assert repr_str.startswith("TextNode(")
        assert f"id=UUID('{node.id}')" in repr_str
        assert "text='  Sample text  '" in repr_str
        assert "node_metadata={'key': 'value'}" in repr_str


# ============================================================================
# Relationships
# ============================================================================

class TestTextNodeRelationships:
    def test_children_empty_by_default(self, session):
        """Test that a node with no children has an empty children list."""
        parent = TextNode(text="Parent")
        session.add(parent)
        session.flush()

        assert parent.children == []

    def test_parent_child_relationship(self, session, sample_document_node):
        """Test parent-child relationship between text nodes."""
        parent = TextNode(text="Parent")
        parent.document = sample_document_node
        session.add(parent)
        session.flush()

        child = TextNode(text="Child")
        child.document = sample_document_node
        child.parent = parent
        session.add(child)
        session.flush()

        session.refresh(parent)
        session.refresh(child)

        assert child.parent_id == parent.id
        assert child in parent.children

    def test_prev_next_chain(self, session, sample_document_node):
        """Test previous/next node chaining."""
        n1 = TextNode(text="one")
        n1.document = sample_document_node
        session.add(n1)
        session.flush()

        n2 = TextNode(text="two")
        n2.document = sample_document_node
        n2.prev_node = n1
        session.add(n2)
        session.flush()

        session.refresh(n1)
        session.refresh(n2)

        assert n2.prev_id == n1.id
        assert n1.next_node == n2
        assert n2.prev_node == n1

    def test_document_relationship(self, session, sample_document_node):
        """Test that TextNode correctly references its DocumentNode."""
        chunk = TextNode(text="Chunk")
        chunk.document = sample_document_node
        session.add(chunk)
        session.flush()

        session.refresh(chunk)
        session.refresh(sample_document_node)

        assert chunk.document_id == sample_document_node.id
        assert chunk in sample_document_node.children


# ============================================================================
# DocumentNode
# ============================================================================

class TestDocumentNode:
    def test_creation_minimal(self, session):
        """Test creating a DocumentNode with required fields."""
        node = DocumentNode(text="Document", source_id="source-1")
        session.add(node)
        session.flush()

        assert node.id is not None
        assert node.text == "Document"
        assert node.source_id == "source-1"
        assert node.node_metadata is None

    def test_children_relationship(self, session):
        """Test DocumentNode's relationship with TextNode children."""
        doc = DocumentNode(text="Document", source_id="source-1")
        session.add(doc)
        session.flush()

        child = TextNode(text="Chunk")
        child.document = doc
        session.add(child)
        session.flush()

        session.refresh(doc)
        session.refresh(child)

        assert child.document_id == doc.id
        assert child in doc.children

    def test_cascade_delete_document(self, session):
        doc = DocumentNode(text="Document", source_id="source-1")
        child = TextNode(text="Child")
        child.document = doc
        session.add_all([doc, child])
        session.flush()

        session.delete(doc)
        session.flush()
        session.expunge_all()

        assert session.get(TextNode, child.id) is None


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_empty_text(self, session):
        """Test that empty text is allowed."""
        node = TextNode(text="")
        session.add(node)
        session.flush()

        assert node.text == ""

    def test_metadata_with_none_value(self, session):
        """Test metadata with None as a value."""
        node = TextNode(text="Text")
        node.node_metadata = {"key": None}
        session.add(node)
        session.flush()

        assert node.node_metadata == {"key": None}
