"""Markdown chunking service."""

from __future__ import annotations

import re  # noqa: TC003
from collections.abc import Sequence  # noqa: TC003
from dataclasses import dataclass
from typing import Protocol

import logfire

from src.core.chunking_config import CHUNKING_SETTINGS
from src.models.node import DocumentNode, TextNode
from src.services.embedding_service import EmbeddingService, TokenOffsets


chunked_documents_metric = logfire.metric_counter(
    "documents_chunked",
    unit="1",
    description="Number of documents chunked.",
)


@dataclass(frozen=True, slots=True)
class _Heading:
    """Parsed markdown heading with level, title, and char span."""

    level: int
    title: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _SpanRange:
    """Half-open character span [start, end)."""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _SpanUnit:
    """Span plus token count, used for window packing."""

    start: int
    end: int
    tokens: int


@dataclass(frozen=True, slots=True)
class _SectionStackEntry:
    """Stack entry tracking the active heading path and node."""

    level: int
    node: TextNode
    path: str


class ChunkingStrategy(Protocol):
    """Strategy for splitting oversized spans into chunkable text."""

    def split_oversized_span(
        self,
        token_offsets: TokenOffsets,
        text: str,
        span_start: int,
        span_end: int,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        """Split a span into chunks that honor token limits and overlap."""
        ...


# -----------------------------
# Chunking strategy
# -----------------------------


class ParagraphSentenceMathChunkingStrategy:
    """Chunk by paragraph, fall back to sentences, and keep math intact."""

    def __init__(
        self,
        *,
        paragraph_break_pattern: re.Pattern = CHUNKING_SETTINGS.paragraph_patterns,
        sentence_boundary_pattern: re.Pattern = CHUNKING_SETTINGS.sentence_patterns,
        inline_math_pattern: re.Pattern = CHUNKING_SETTINGS.inline_latex_math_patterns,
        block_math_pattern: re.Pattern = CHUNKING_SETTINGS.block_latex_math_patterns,
        citation_command_prefixes: tuple[
            str,
            ...,
        ] = CHUNKING_SETTINGS.citation_command_prefixes,
    ) -> None:
        """Store regex patterns for paragraph breaks, sentences, and math blocks."""
        self._paragraph_break_pattern = paragraph_break_pattern
        self._sentence_boundary_pattern = sentence_boundary_pattern
        self._inline_math_pattern = inline_math_pattern
        self._block_math_pattern = block_math_pattern
        self._citation_command_prefixes = citation_command_prefixes

    def split_oversized_span(
        self,
        token_offsets: TokenOffsets,
        text: str,
        span_start: int,
        span_end: int,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        """Split a large span by paragraph/sentence boundaries without cutting math."""
        if span_start >= span_end:
            return [text[span_start:span_end]]

        math_spans, block_math_spans = self._collect_math_span_sets(
            text,
            span_start,
            span_end,
        )
        units = self._build_split_units(
            token_offsets=token_offsets,
            text=text,
            span_start=span_start,
            span_end=span_end,
            math_spans=math_spans,
            block_math_spans=block_math_spans,
            max_tokens=max_tokens,
        )

        chunk_spans = self._pack_units_into_windows(
            units=units,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )

        chunks = [
            text[span.start : span.end] for span in chunk_spans if span.start < span.end
        ]
        return chunks or [text[span_start:span_end]]

    def _build_split_units(
        self,
        *,
        token_offsets: TokenOffsets,
        text: str,
        span_start: int,
        span_end: int,
        math_spans: list[_SpanRange],
        block_math_spans: list[_SpanRange],
        max_tokens: int,
    ) -> list[_SpanUnit]:
        """Build paragraph/sentence units with cached token counts."""
        units: list[_SpanUnit] = []
        paragraph_spans = self._split_paragraph_spans(
            text=text,
            span_start=span_start,
            span_end=span_end,
            math_spans=math_spans,
            block_math_spans=block_math_spans,
        )

        for paragraph in paragraph_spans:
            if self._is_blank_span(text, paragraph):
                continue
            para_tokens = token_offsets.count_tokens_from_span(
                paragraph.start,
                paragraph.end,
            )
            if para_tokens <= max_tokens:
                units.append(_SpanUnit(paragraph.start, paragraph.end, para_tokens))
                continue

            sentence_spans = self._split_sentence_spans(
                text=text,
                para_start=paragraph.start,
                para_end=paragraph.end,
                math_spans=math_spans,
            )
            if len(sentence_spans) <= 1:
                units.append(_SpanUnit(paragraph.start, paragraph.end, para_tokens))
                continue

            for sentence in sentence_spans:
                if self._is_blank_span(text, sentence):
                    continue
                sent_tokens = token_offsets.count_tokens_from_span(
                    sentence.start,
                    sentence.end,
                )

                units.append(_SpanUnit(sentence.start, sentence.end, sent_tokens))

        if not units:
            total_tokens = token_offsets.count_tokens_from_span(span_start, span_end)
            units.append(_SpanUnit(span_start, span_end, total_tokens))
        return units

    def _split_paragraph_spans(
        self,
        *,
        text: str,
        span_start: int,
        span_end: int,
        math_spans: list[_SpanRange],
        block_math_spans: list[_SpanRange],
    ) -> list[_SpanRange]:
        """Split into paragraph spans, skipping math and glued block math."""
        spans: list[_SpanRange] = []
        cursor = span_start

        for match in self._paragraph_break_pattern.finditer(text, span_start, span_end):
            sep_start, sep_end = match.start(), match.end()
            if sep_start <= cursor:
                continue
            if self._boundary_hits_span(
                start=sep_start,
                end=sep_end,
                lower_bound=span_start,
                spans=math_spans,
            ):
                continue
            if self._should_skip_paragraph_split(
                text=text,
                sep_start=sep_start,
                sep_end=sep_end,
                span_start=span_start,
                span_end=span_end,
                block_math_spans=block_math_spans,
            ):
                continue

            spans.append(_SpanRange(cursor, sep_start))
            cursor = sep_end

        if cursor < span_end:
            spans.append(_SpanRange(cursor, span_end))

        return spans

    def _split_sentence_spans(
        self,
        *,
        text: str,
        para_start: int,
        para_end: int,
        math_spans: list[_SpanRange],
    ) -> list[_SpanRange]:
        """Split a paragraph into sentence spans, avoiding math interiors."""
        boundaries: list[int] = []
        for match in self._sentence_boundary_pattern.finditer(
            text,
            para_start,
            para_end,
        ):
            boundary = match.end()
            if boundary <= para_start or boundary >= para_end:
                continue
            if self._boundary_hits_span(
                start=match.start(),
                end=boundary,
                lower_bound=para_start,
                spans=math_spans,
            ):
                continue
            next_idx = self._find_next_nonspace(text, boundary, para_end)
            if next_idx is not None and self._is_sentence_continuation(text, next_idx):
                continue
            boundaries.append(boundary)

        spans: list[_SpanRange] = []
        cursor = para_start
        for boundary in boundaries:
            if boundary <= cursor:
                continue
            spans.append(_SpanRange(cursor, boundary))
            cursor = boundary

        if cursor < para_end:
            spans.append(_SpanRange(cursor, para_end))
        return spans

    def _collect_math_span_sets(
        self,
        text: str,
        span_start: int,
        span_end: int,
    ) -> tuple[list[_SpanRange], list[_SpanRange]]:
        """Return merged spans for all math and for block math only."""
        block_spans = self._collect_pattern_spans(
            text=text,
            span_start=span_start,
            span_end=span_end,
            patterns=[self._block_math_pattern],
        )
        inline_spans = self._collect_pattern_spans(
            text=text,
            span_start=span_start,
            span_end=span_end,
            patterns=[self._inline_math_pattern],
        )
        block_math_spans = self._merge_overlapping_spans(block_spans)
        math_spans = self._merge_overlapping_spans([*block_spans, *inline_spans])
        return math_spans, block_math_spans

    @staticmethod
    def _collect_pattern_spans(
        *,
        text: str,
        span_start: int,
        span_end: int,
        patterns: list[re.Pattern],
    ) -> list[_SpanRange]:
        """Collect raw spans for the provided regex patterns."""
        return [
            _SpanRange(match.start(), match.end())
            for pattern in patterns
            for match in pattern.finditer(text, span_start, span_end)
        ]

    @staticmethod
    def _merge_overlapping_spans(spans: list[_SpanRange]) -> list[_SpanRange]:
        """Merge overlapping spans into a sorted, non-overlapping list."""
        if not spans:
            return []

        spans = sorted(spans, key=lambda span: span.start)
        merged: list[_SpanRange] = []
        current_start = spans[0].start
        current_end = spans[0].end
        for span in spans[1:]:
            if span.start <= current_end:
                current_end = max(current_end, span.end)
            else:
                merged.append(_SpanRange(current_start, current_end))
                current_start = span.start
                current_end = span.end
        merged.append(_SpanRange(current_start, current_end))
        return merged

    @classmethod
    def _boundary_hits_span(
        cls,
        *,
        start: int,
        end: int,
        lower_bound: int,
        spans: list[_SpanRange],
    ) -> bool:
        """Return True if a boundary intersects any span."""
        if not spans:
            return False
        end_check = end - 1
        has_end_check = end_check >= lower_bound
        for span in spans:
            if span.start <= start < span.end:
                return True
            if has_end_check and span.start <= end_check < span.end:
                return True
        return False

    @staticmethod
    def _find_prev_nonspace(text: str, start: int, lower_bound: int) -> int | None:
        """Return the previous non-space character index."""
        index = min(start, len(text) - 1)
        while index >= lower_bound:
            if not text[index].isspace():
                return index
            index -= 1
        return None

    @staticmethod
    def _find_next_nonspace(text: str, start: int, upper_bound: int) -> int | None:
        """Return the next non-space character index."""
        index = max(start, 0)
        while index < upper_bound:
            if not text[index].isspace():
                return index
            index += 1
        return None

    @classmethod
    def _is_sentence_end(cls, text: str, index: int) -> bool:
        """Heuristic: trailing punctuation (ignoring quotes/brackets) ends a sentence."""
        cursor = index
        while cursor >= 0 and text[cursor].isspace():
            cursor -= 1
        while cursor >= 0 and text[cursor] in "\"')]}":
            cursor -= 1
        return cursor >= 0 and text[cursor] in ".!?"

    def _is_sentence_continuation(self, text: str, index: int) -> bool:
        """Heuristic: lowercase/digit/leading bracket implies continuation."""
        char = text[index]
        if char.islower() or char.isdigit():
            return True
        if char in "([{":
            return True
        if char == "\\":
            if index + 1 < len(text) and text[index + 1] in "[(":
                return True
            if text.startswith("\\\\", index):
                return True
            return any(
                text.startswith(prefix, index)
                for prefix in self._citation_command_prefixes
            )
        return False

    def _should_skip_paragraph_split(
        self,
        *,
        text: str,
        sep_start: int,
        sep_end: int,
        span_start: int,
        span_end: int,
        block_math_spans: list[_SpanRange],
    ) -> bool:
        """Skip splitting when block math borders a continuing sentence."""
        prev_idx = self._find_prev_nonspace(text, sep_start - 1, span_start)
        next_idx = self._find_next_nonspace(text, sep_end, span_end)
        if prev_idx is None or next_idx is None:
            return False

        prev_inside_block = self._boundary_hits_span(
            start=prev_idx,
            end=prev_idx + 1,
            lower_bound=span_start,
            spans=block_math_spans,
        )
        next_inside_block = self._boundary_hits_span(
            start=next_idx,
            end=next_idx + 1,
            lower_bound=span_start,
            spans=block_math_spans,
        )
        if not (prev_inside_block or next_inside_block):
            return False

        prev_sentence_end = self._is_sentence_end(text, prev_idx)
        next_continuation = self._is_sentence_continuation(text, next_idx)
        return (not prev_sentence_end) or next_continuation

    @staticmethod
    def _pack_units_into_windows(
        *,
        units: list[_SpanUnit],
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[_SpanRange]:
        """Pack token-counted spans into windows with optional overlap."""
        if not units:
            return []

        chunks: list[_SpanRange] = []
        index = 0
        total_units = len(units)

        while index < total_units:
            token_total = 0
            end_index = index

            while end_index < total_units:
                unit_tokens = units[end_index].tokens
                if token_total + unit_tokens > max_tokens:
                    break
                token_total += unit_tokens
                end_index += 1

            if end_index == index:
                chunks.append(_SpanRange(units[index].start, units[index].end))
                index += 1
                continue

            chunks.append(_SpanRange(units[index].start, units[end_index - 1].end))

            if overlap_tokens <= 0:
                index = end_index
                continue

            target_advance = max(1, token_total - overlap_tokens)
            advanced = 0
            while index < end_index and advanced < target_advance:
                advanced += units[index].tokens
                index += 1

        return chunks

    @staticmethod
    def _is_blank_span(text: str, span: _SpanRange) -> bool:
        """Return True when the span contains only whitespace."""
        return not text[span.start : span.end].strip()


# -----------------------------
# Main chunker
# -----------------------------


class MarkdownChunker:
    """Chunk markdown documents into a TextNode tree."""

    def __init__(
        self,
        *,
        embedding_model_name: str = CHUNKING_SETTINGS.embedding_model_name,
        max_tokens: int = CHUNKING_SETTINGS.max_tokens,
        overlap_tokens: int = CHUNKING_SETTINGS.overlap_tokens,
        header_pattern: re.Pattern = CHUNKING_SETTINGS.header_patterns,
        strategy: ChunkingStrategy | None = None,
    ) -> None:
        """Initialize a chunker with tokenizer, limits, and strategy."""
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._header_pattern = header_pattern

        self._strategy = strategy or ParagraphSentenceMathChunkingStrategy()

        self._embedding_service = EmbeddingService(
            embedding_model_name=embedding_model_name,
        )

    @logfire.instrument(
        "chunking_service.chunk",
        extract_args=False,
    )
    def chunk(
        self,
        documents: Sequence[DocumentNode],
        *,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Chunk documents into TextNode trees rooted at each DocumentNode."""
        with logfire.span(f"chunking {len(documents)} documents."):
            self._chunk_internal(documents, metadata=metadata)
            logfire.info("documents_chunking completed")

    def _chunk_internal(
        self,
        documents: Sequence[DocumentNode],
        *,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Build section nodes for each document based on headings."""
        base_metadata = dict(metadata) if metadata else None
        for document in documents:
            markdown_text = document.text
            token_offsets = self._embedding_service.tokenize_with_offsets_mapping(
                markdown_text,
            )
            headings = self._extract_headings(markdown_text)

            if not headings:
                raise ValueError(f"Headers not detected for document ID: {document.id}")

            stack: list[_SectionStackEntry] = []

            for heading, next_heading in zip(
                headings,
                [*headings[1:], None],
                strict=False,
            ):
                next_start = (
                    next_heading.start
                    if next_heading is not None
                    else len(markdown_text)
                )
                section_span = self._trim_whitespace_span(
                    markdown_text,
                    heading.end,
                    next_start,
                )

                while stack and stack[-1].level >= heading.level:
                    stack.pop()

                parent_trail = stack[-1] if stack else None
                parent_node = parent_trail.node if parent_trail else None
                parent_path = parent_trail.path if parent_trail else ""
                path = self._join_path(parent_path, heading.title)

                node = self._create_section_node(
                    document=document,
                    parent=parent_node,
                    title=heading.title,
                    path=path,
                    base_metadata=base_metadata,
                )

                self._assign_or_split_section_text(
                    node=node,
                    full_text=markdown_text,
                    span_start=section_span.start,
                    span_end=section_span.end,
                    token_offsets=token_offsets,
                    base_metadata=base_metadata,
                )
                document.children.append(node)

                stack.append(_SectionStackEntry(heading.level, node, path))

            chunked_documents_metric.add(1)

    # -------------------------
    # Parsing
    # -------------------------

    def _extract_headings(self, text: str) -> list[_Heading]:
        """Return headings in order using the configured header pattern."""
        return [
            _Heading(
                level=len(m.group(1)),
                title=m.group(2).strip(),
                start=m.start(),
                end=m.end(),
            )
            for m in self._header_pattern.finditer(text)
        ]

    @staticmethod
    def _trim_whitespace_span(text: str, start: int, end: int) -> _SpanRange:
        """Trim leading/trailing whitespace in a span and return new bounds."""
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return _SpanRange(start, end)

    # -------------------------
    # Section text assignment & splitting
    # -------------------------

    def _assign_or_split_section_text(
        self,
        *,
        node: TextNode,
        full_text: str,
        span_start: int,
        span_end: int,
        token_offsets: TokenOffsets,
        base_metadata: dict[str, object] | None = None,
    ) -> None:
        """Assign section text or split into part-N children if oversized."""
        if span_start >= span_end:
            node.text = ""
            return

        token_count = token_offsets.count_tokens_from_span(
            span_start,
            span_end,
        )

        if token_count <= self._max_tokens:
            node.text = full_text[span_start:span_end]
            return

        chunks = self._strategy.split_oversized_span(
            token_offsets=token_offsets,
            text=full_text,
            span_start=span_start,
            span_end=span_end,
            max_tokens=self._max_tokens,
            overlap_tokens=self._overlap_tokens,
        )

        node.text = ""
        non_empty_chunks = [chunk for chunk in chunks if chunk]
        if not non_empty_chunks:
            return

        self._create_part_children(
            parent=node,
            chunks=non_empty_chunks,
            base_metadata=base_metadata,
        )

    def _create_part_children(
        self,
        *,
        parent: TextNode,
        chunks: list[str],
        base_metadata: dict[str, object] | None = None,
    ) -> None:
        """Create part-N child nodes for each chunk and link them sequentially."""
        prev_node: TextNode | None = None
        parent_path = self._node_path(parent)

        part_index = 0
        for chunk_text in chunks:
            if not chunk_text:
                continue

            part_index += 1
            title = f"part {part_index}"
            path = self._join_path(parent_path, title)

            child = TextNode(text=chunk_text)
            child.parent = parent
            if base_metadata:
                child.node_metadata = dict(base_metadata)
            child.node_metadata = self._metadata_for(child, title=title, path=path)
            if prev_node is not None:
                child.prev_node = prev_node
                prev_node.next_node = child
            prev_node = child
            parent.document.children.append(child)

    @staticmethod
    def _metadata_for(node: TextNode, *, title: str, path: str) -> dict[str, object]:
        """Build node metadata while preserving any existing values."""
        return {
            **(node.node_metadata or {}),
            "title": title,
            "path": path,
        }

    @staticmethod
    def _node_path(node: TextNode) -> str:
        """Return the node path from metadata, or an empty string."""
        return str((node.node_metadata or {}).get("path") or "")

    @staticmethod
    def _join_path(parent_path: str, title: str) -> str:
        """Join a parent path with a title component."""
        return f"{parent_path}/{title}" if parent_path else title

    @classmethod
    def _create_section_node(
        cls,
        *,
        document: DocumentNode,
        parent: TextNode | None,
        title: str,
        path: str,
        base_metadata: dict[str, object] | None = None,
    ) -> TextNode:
        """Create a section node with parent/doc linkage and metadata."""
        node = TextNode(text="")
        node.parent = parent
        node.document = document
        if base_metadata:
            node.node_metadata = dict(base_metadata)
        node.node_metadata = cls._metadata_for(node, title=title, path=path)
        return node
