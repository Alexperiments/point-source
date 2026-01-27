"""Chunking service."""

from __future__ import annotations

import re  # noqa: TC003
from dataclasses import dataclass

from src.core.chunking_config import CHUNKING_SETTINGS
from src.models.node import DocumentNode, TextNode
from src.services.tokenizer_service import Tokenizer, TokenizerFactory, TokenOffsets


@dataclass(frozen=True)
class _Heading:
    level: int
    title: str
    start: int
    end: int


@dataclass(frozen=True)
class _SpanRange:
    start: int
    end: int


@dataclass(frozen=True)
class _SpanUnit:
    start: int
    end: int
    tokens: int


@dataclass(frozen=True)
class _SectionTrail:
    level: int
    node: TextNode
    path: str


@dataclass(frozen=True)
class _ChunkInternalResult:
    documents: list[DocumentNode]
    nodes: list[TextNode]


# -----------------------------
# Chunking strategy
# -----------------------------


class ChunkingStrategy:
    """Base class for chunking strategies.

    This ecapsulates a naive chunking strategy only based on token-window split.
    Reimplement split_oversized_span to improve the strategy.
    """

    def split_oversized_span(
        self,
        token_offsets: TokenOffsets,
        span_start: int,
        span_end: int,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        """Naive token-window split. Designed as a stable seam for later "smart" chunking.

        - Uses global offsets (single tokenizer call).
        - Produces chunks <= max_tokens tokens each.
        - Optional overlap (currently applied; set overlap_tokens=0 to disable).
        """
        token_range = token_offsets.token_range_for_char_span(
            span_start,
            span_end,
        )
        if token_range.start >= token_range.end:
            return [token_offsets.text[span_start:span_end]]

        chunks: list[str] = []
        cursor = token_range.start
        stride = max(1, max_tokens - max(0, overlap_tokens))

        while cursor < token_range.end:
            end_token = min(cursor + max_tokens, token_range.end)
            piece = token_offsets.slice_text_by_token_range(
                start_token=cursor,
                end_token=end_token,
                clamp_start=span_start,
                clamp_end=span_end,
            )
            if piece:
                chunks.append(piece)
            cursor += stride

        return chunks


class ParagraphEquationSentenceChunkingStrategy(ChunkingStrategy):
    """Chunk by paragraphs; fall back to sentences; never split equations."""

    def __init__(
        self,
        *,
        paragraph_break_pattern: re.Pattern = CHUNKING_SETTINGS.paragraph_patterns,
        sentence_boundary_pattern: re.Pattern = CHUNKING_SETTINGS.sentence_patterns,
        inline_math_pattern: re.Pattern = CHUNKING_SETTINGS.inline_latex_math_patterns,
        block_math_pattern: re.Pattern = CHUNKING_SETTINGS.block_latex_math_patterns,
    ) -> None:
        """Initialize the strategy with paragraph, sentence, and math patterns."""
        self._paragraph_break_pattern = paragraph_break_pattern
        self._sentence_boundary_pattern = sentence_boundary_pattern
        self._inline_math_pattern = inline_math_pattern
        self._block_math_pattern = block_math_pattern

    def split_oversized_span(
        self,
        token_offsets: TokenOffsets,
        span_start: int,
        span_end: int,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        """Split a span without breaking equations, preferring paragraph boundaries."""
        text = token_offsets.text
        if span_start >= span_end:
            return [text[span_start:span_end]]

        equation_spans = self._collect_equation_spans(text, span_start, span_end)
        units = self._build_units(
            token_offsets=token_offsets,
            text=text,
            span_start=span_start,
            span_end=span_end,
            equation_spans=equation_spans,
            max_tokens=max_tokens,
        )

        chunk_spans = self._pack_units(
            units=units,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )

        chunks = [
            text[span.start : span.end] for span in chunk_spans if span.start < span.end
        ]
        return chunks or [text[span_start:span_end]]

    def _build_units(
        self,
        *,
        token_offsets: TokenOffsets,
        text: str,
        span_start: int,
        span_end: int,
        equation_spans: list[_SpanRange],
        max_tokens: int,
    ) -> list[_SpanUnit]:
        units: list[_SpanUnit] = []
        paragraph_spans = self._split_paragraphs(
            text=text,
            span_start=span_start,
            span_end=span_end,
            equation_spans=equation_spans,
        )

        for paragraph in paragraph_spans:
            if not text[paragraph.start : paragraph.end].strip():
                continue
            para_tokens = self._token_count(
                token_offsets,
                paragraph.start,
                paragraph.end,
            )
            if para_tokens <= max_tokens:
                units.append(_SpanUnit(paragraph.start, paragraph.end, para_tokens))
                continue

            sentence_spans = self._split_sentences(
                text=text,
                para_start=paragraph.start,
                para_end=paragraph.end,
                equation_spans=equation_spans,
            )
            if len(sentence_spans) <= 1:
                units.append(_SpanUnit(paragraph.start, paragraph.end, para_tokens))
                continue

            for sentence in sentence_spans:
                if not text[sentence.start : sentence.end].strip():
                    continue
                sent_tokens = self._token_count(
                    token_offsets,
                    sentence.start,
                    sentence.end,
                )
                units.append(_SpanUnit(sentence.start, sentence.end, sent_tokens))

        if not units:
            total_tokens = self._token_count(token_offsets, span_start, span_end)
            units.append(_SpanUnit(span_start, span_end, total_tokens))
        return units

    def _split_paragraphs(
        self,
        *,
        text: str,
        span_start: int,
        span_end: int,
        equation_spans: list[_SpanRange],
    ) -> list[_SpanRange]:
        spans: list[_SpanRange] = []
        cursor = span_start

        for match in self._paragraph_break_pattern.finditer(text, span_start, span_end):
            sep_start, sep_end = match.start(), match.end()
            if sep_start <= cursor:
                continue
            if self._is_inside_equation(sep_start, equation_spans) or (
                sep_end - 1 >= span_start
                and self._is_inside_equation(sep_end - 1, equation_spans)
            ):
                continue

            spans.append(_SpanRange(cursor, sep_start))
            cursor = sep_end

        if cursor < span_end:
            spans.append(_SpanRange(cursor, span_end))

        return spans

    def _split_sentences(
        self,
        *,
        text: str,
        para_start: int,
        para_end: int,
        equation_spans: list[_SpanRange],
    ) -> list[_SpanRange]:
        boundaries: list[int] = []
        for match in self._sentence_boundary_pattern.finditer(
            text,
            para_start,
            para_end,
        ):
            boundary = match.end()
            if boundary <= para_start or boundary >= para_end:
                continue
            if self._is_inside_equation(match.start(), equation_spans) or (
                boundary - 1 >= para_start
                and self._is_inside_equation(boundary - 1, equation_spans)
            ):
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

    def _collect_equation_spans(
        self,
        text: str,
        span_start: int,
        span_end: int,
    ) -> list[_SpanRange]:
        raw_spans: list[_SpanRange] = []
        patterns = [self._block_math_pattern, self._inline_math_pattern]
        for pattern in patterns:
            matches = [
                _SpanRange(match.start(), match.end())
                for match in pattern.finditer(text, span_start, span_end)
            ]
            raw_spans.extend(matches)

        if not raw_spans:
            return []

        raw_spans.sort(key=lambda span: span.start)
        merged: list[_SpanRange] = []
        current_span = raw_spans[0]
        current_start = current_span.start
        current_end = current_span.end
        for span in raw_spans[1:]:
            if span.start <= current_end:
                current_end = max(current_end, span.end)
            else:
                merged.append(_SpanRange(current_start, current_end))
                current_start = span.start
                current_end = span.end
        merged.append(_SpanRange(current_start, current_end))
        return merged

    @staticmethod
    def _is_inside_equation(position: int, spans: list[_SpanRange]) -> bool:
        return any(span.start <= position < span.end for span in spans)

    @staticmethod
    def _token_count(
        token_offsets: TokenOffsets,
        span_start: int,
        span_end: int,
    ) -> int:
        token_range = token_offsets.token_range_for_char_span(
            span_start,
            span_end,
        )
        return token_range.end - token_range.start

    @staticmethod
    def _pack_units(
        *,
        units: list[_SpanUnit],
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[_SpanRange]:
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
                if token_total + unit_tokens > max_tokens and end_index > index:
                    break
                if token_total + unit_tokens > max_tokens and end_index == index:
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
            if index == end_index:
                index = end_index

        return chunks


# -----------------------------
# Main chunker
# -----------------------------


class MarkdownChunker:
    """THE markdown chunker.

    Goals:
    - Single tokenizer call for the full text (for providers that support offsets).
    - Fast span->token lookups (bisect on precomputed starts/ends).
    - Clear separation of concerns: parsing vs. tree-building vs. splitting.
    - Extensible provider support via TokenizerFactory.
    - Extensible splitting via ChunkingStrategy.
    """

    def __init__(
        self,
        *,
        tokenizer_name: str = CHUNKING_SETTINGS.tokenizer_name,
        max_tokens: int = CHUNKING_SETTINGS.max_tokens,
        overlap_tokens: int = CHUNKING_SETTINGS.overlap_tokens,
        header_pattern: re.Pattern = CHUNKING_SETTINGS.header_patterns,
        strategy: ChunkingStrategy | None = None,
    ) -> None:
        """Initialize a markdown chunker with tokenizer, limits, and strategy."""
        self._tokenizer_name = tokenizer_name
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._header_pattern = header_pattern

        self._strategy = strategy or ParagraphEquationSentenceChunkingStrategy()

        self._token_offsets_cache: TokenOffsets | None = None
        self._token_offsets_text: str | None = None
        self._tokenizer_instance: Tokenizer | None = None

    def chunk(self, documents: list[DocumentNode]) -> list[DocumentNode]:
        """Chunk documents into TextNode trees rooted at each DocumentNode.

        Behavior (kept consistent with the existing implementation):
        - Detect headings using configured regex.
        - Section text is content between a heading line and the next heading.
        - If a section exceeds max_tokens, create "part N" child nodes, clear parent text.
        - If no headings exist, create a single root TextNode containing the whole text
          (and apply splitting if it exceeds max_tokens).
        - Returns the input documents; relationships are set on ORM objects.
        """
        result = self._chunk_internal(documents)
        return result.documents

    def chunk_nodes(self, documents: list[DocumentNode]) -> list[TextNode]:
        """Chunk documents and return the created TextNodes."""
        result = self._chunk_internal(documents)
        return result.nodes

    def _chunk_internal(
        self,
        documents: list[DocumentNode],
    ) -> _ChunkInternalResult:
        nodes: list[TextNode] = []
        for document in documents:
            markdown_text = document.text
            token_offsets = self._get_or_build_token_offsets(markdown_text)
            headings = self._extract_headings(markdown_text)

            if not headings:
                root = TextNode(text="")
                root.document = document
                root.node_metadata = {
                    **(root.node_metadata or {}),
                    "title": "Document",
                    "path": "Document",
                }
                nodes.append(root)

                span = self._trim_whitespace_span(
                    markdown_text,
                    0,
                    len(markdown_text),
                )
                self._assign_or_split_section_text(
                    node=root,
                    full_text=markdown_text,
                    span_start=span.start,
                    span_end=span.end,
                    token_offsets=token_offsets,
                    nodes=nodes,
                )
                continue

            stack: list[_SectionTrail] = []

            for i, heading in enumerate(headings):
                next_start = (
                    headings[i + 1].start
                    if i + 1 < len(headings)
                    else len(markdown_text)
                )
                section_span = self._trim_whitespace_span(
                    markdown_text,
                    heading.end,
                    next_start,
                )

                while stack and stack[-1].level >= heading.level:
                    stack.pop()

                parent_node = stack[-1].node if stack else None
                parent_path = stack[-1].path if stack else ""
                path = (
                    f"{parent_path}/{heading.title}" if parent_path else heading.title
                )

                node = TextNode(text="")
                node.document = document
                node.parent = parent_node
                node.node_metadata = {
                    **(node.node_metadata or {}),
                    "title": heading.title,
                    "path": path,
                }
                nodes.append(node)

                self._assign_or_split_section_text(
                    node=node,
                    full_text=markdown_text,
                    span_start=section_span.start,
                    span_end=section_span.end,
                    token_offsets=token_offsets,
                    nodes=nodes,
                )

                stack.append(_SectionTrail(heading.level, node, path))

        return _ChunkInternalResult(documents, nodes)

    # -------------------------
    # Parsing
    # -------------------------

    def _extract_headings(self, text: str) -> list[_Heading]:
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
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return _SpanRange(start, end)

    # -------------------------
    # Token offsets (single pass)
    # -------------------------

    def _get_or_build_token_offsets(self, text: str) -> TokenOffsets:
        if self._token_offsets_text == text and self._token_offsets_cache is not None:
            return self._token_offsets_cache

        tokenizer = self._get_tokenizer()
        offsets = tokenizer.tokenize_with_offsets(text)

        self._token_offsets_text = text
        self._token_offsets_cache = offsets
        return offsets

    def _get_tokenizer(self) -> Tokenizer:
        if self._tokenizer_instance is None:
            self._tokenizer_instance = TokenizerFactory.create(self._tokenizer_name)
        return self._tokenizer_instance

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
        nodes: list[TextNode],
    ) -> None:
        """Assign section text to node, or split into part-N children if too large."""
        if span_start >= span_end:
            node.text = ""
            return

        section_text = full_text[span_start:span_end]
        if not section_text:
            node.text = ""
            return

        token_range = token_offsets.token_range_for_char_span(
            span_start,
            span_end,
        )
        token_count = token_range.end - token_range.start

        if token_count <= self._max_tokens:
            node.text = section_text
            return

        chunks = self._strategy.split_oversized_span(
            token_offsets=token_offsets,
            span_start=span_start,
            span_end=span_end,
            max_tokens=self._max_tokens,
            overlap_tokens=self._overlap_tokens,
        )

        node.text = ""
        nodes.extend(self._create_part_children(parent=node, chunks=chunks))

    @staticmethod
    def _create_part_children(
        *,
        parent: TextNode,
        chunks: list[str],
    ) -> list[TextNode]:
        prev_node: TextNode | None = None
        children: list[TextNode] = []

        part_index = 0
        for chunk_text in chunks:
            if not chunk_text:
                continue

            part_index += 1
            parent_path = ""
            if parent.node_metadata is not None:
                parent_path = str(parent.node_metadata.get("path") or "")
            path = (
                f"{parent_path}/part {part_index}"
                if parent_path
                else f"part {part_index}"
            )
            child = TextNode(text=chunk_text)
            child.document = parent.document
            child.parent = parent
            child.node_metadata = {
                **(child.node_metadata or {}),
                "title": f"part {part_index}",
                "path": path,
            }
            if prev_node is not None:
                child.prev_node = prev_node
                prev_node.next_node = child
            prev_node = child
            children.append(child)
        return children
