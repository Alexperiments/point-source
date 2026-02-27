import re

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.database.base import Base
from src.models.node import DocumentNode, TextNode
from src.services.chunking_service import MarkdownChunker


class _DummyTokenOffsets:
    def count_tokens_from_span(self, start: int, end: int) -> int:
        return max(1, end - start)


class _DummyTokenizerService:
    def tokenize_with_offsets_mapping(self, _text: str) -> _DummyTokenOffsets:
        return _DummyTokenOffsets()


class _NoSplitStrategy:
    def split_oversized_span(  # noqa: PLR0913
        self,
        token_offsets,  # noqa: ANN001
        text: str,
        span_start: int,
        span_end: int,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        return [text[span_start:span_end]]


def _build_chunker() -> MarkdownChunker:
    chunker = MarkdownChunker.__new__(MarkdownChunker)
    chunker._max_tokens = 10_000  # noqa: SLF001
    chunker._overlap_tokens = 0  # noqa: SLF001
    chunker._min_chunk_chars = 0  # noqa: SLF001
    chunker._header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)  # noqa: SLF001
    chunker._drop_prefixes = ()  # noqa: SLF001
    chunker._strategy = _NoSplitStrategy()  # noqa: SLF001
    chunker._strategy_name = "NoSplitStrategy"  # noqa: SLF001
    chunker._tokenizer_name = "dummy-tokenizer"  # noqa: SLF001
    chunker._tokenizer_service = _DummyTokenizerService()  # noqa: SLF001
    return chunker


@pytest.fixture(scope="session")
def engine():
    engine = create_engine(
        "sqlite:///:memory:",
        echo=False,
        execution_options={"schema_translate_map": {"processed": None}},
    )
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


def test_chunk_returns_new_text_nodes(session) -> None:
    chunker = _build_chunker()
    doc = DocumentNode(
        id="astro-ph/1234567",
        doi_url="https://www.doi.org/10.48550/arXiv.astro-ph/1234567",
        text="# Intro\n\nChunk body.",
    )
    session.add(doc)
    session.flush()

    nodes = chunker.chunk([doc], metadata={"dataset": "unit-test"})
    session.add_all(nodes)
    session.flush()

    assert len(nodes) == 1
    node = nodes[0]
    assert isinstance(node, TextNode)
    assert node.document == doc
    assert node.document_id == doc.id
    assert node.parent is None
    assert node.text == "Chunk body."
    assert node.node_metadata is not None
    assert node.node_metadata["path"] == "Intro"


def test_create_part_children_links_document_and_prev_next(session) -> None:
    chunker = _build_chunker()
    doc = DocumentNode(
        id="astro-ph/7654321",
        doi_url="https://www.doi.org/10.48550/arXiv.astro-ph/7654321",
        text="# Intro\n\nChunk body.",
    )
    parent = TextNode(text="Parent section")
    parent.document = doc
    parent.node_metadata = {"path": "Intro"}
    session.add_all([doc, parent])
    session.flush()

    children = chunker._create_part_children(  # noqa: SLF001
        parent=parent,
        chunks=["part one", "part two"],
        base_metadata={"dataset": "unit-test"},
    )
    session.add_all(children)
    session.flush()

    assert len(children) == 2
    first, second = children
    assert first.document_id == doc.id
    assert second.document_id == doc.id
    assert first.parent_id == parent.id
    assert second.parent_id == parent.id
    assert second.prev_id == first.id
    assert first.node_metadata is not None
    assert first.node_metadata["path"] == "Intro/part 1"
    assert second.node_metadata is not None
    assert second.node_metadata["path"] == "Intro/part 2"
