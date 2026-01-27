"""Chunking service."""

from __future__ import annotations

import re  # noqa: TC003
import time
import uuid  # noqa: TC003
from dataclasses import dataclass
from typing import NamedTuple

from src.core.chunking_config import CHUNKING_SETTINGS
from src.schemas.node import TextNode
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
    node_id: uuid.UUID
    path: str


class ChunkingProfileResult(NamedTuple):
    """Result of chunking with profiling information."""

    nodes: list[TextNode]
    profile: dict[str, float]


@dataclass(frozen=True)
class _ChunkInternalResult:
    nodes: list[TextNode]
    profile: dict[str, float] | None


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

    def chunk(self, markdown_text: str) -> list[TextNode]:
        """Chunk a markdown document into a hierarchy of TextNodes.

        Behavior (kept consistent with the existing implementation):
        - Detect headings using configured regex.
        - Build hierarchical paths "Parent/Child".
        - Section text is content between a heading line and the next heading.
        - If a section exceeds max_tokens, create "part N" child nodes, clear parent text.
        - If no headings exist, create a single "Document" root node containing the whole text
          (and apply splitting if it exceeds max_tokens).
        - Returns a LIST of nodes (not a dict). Parent/child relationships are via uuid.UUID fields.
        """
        result = self._chunk_internal(markdown_text, profile=None)
        return result.nodes

    def chunk_with_profile(
        self,
        markdown_text: str,
    ) -> ChunkingProfileResult:
        """Chunk text and return nodes with a coarse timing breakdown."""
        profile: dict[str, float] = {}
        result = self._chunk_internal(markdown_text, profile=profile)
        return ChunkingProfileResult(result.nodes, result.profile or {})

    def _chunk_internal(
        self,
        markdown_text: str,
        profile: dict[str, float] | None,
    ) -> _ChunkInternalResult:
        start_total = time.perf_counter()

        def _record(name: str, start_time: float) -> None:
            if profile is None:
                return
            profile[name] = profile.get(name, 0.0) + (time.perf_counter() - start_time)

        timer_start = time.perf_counter()
        token_offsets = self._get_or_build_token_offsets(markdown_text)
        _record("tokenize", timer_start)

        timer_start = time.perf_counter()
        headings = self._extract_headings(markdown_text)
        _record("extract_headings", timer_start)
        nodes_by_id: dict[uuid.UUID, TextNode] = {}

        if not headings:
            root = TextNode(title="Document", path="Document", text="", parent_id=None)
            nodes_by_id[root.id] = root

            span = self._trim_whitespace_span(
                markdown_text,
                0,
                len(markdown_text),
            )
            assign_start = time.perf_counter()
            self._assign_or_split_section_text(
                node=root,
                full_text=markdown_text,
                span_start=span.start,
                span_end=span.end,
                token_offsets=token_offsets,
                nodes_by_id=nodes_by_id,
            )
            _record("assign_or_split", assign_start)
            if profile is not None:
                profile["total"] = time.perf_counter() - start_total
            return _ChunkInternalResult(list(nodes_by_id.values()), profile)

        stack: list[_SectionTrail] = []

        for i, heading in enumerate(headings):
            next_start = (
                headings[i + 1].start if i + 1 < len(headings) else len(markdown_text)
            )
            section_span = self._trim_whitespace_span(
                markdown_text,
                heading.end,
                next_start,
            )

            while stack and stack[-1].level >= heading.level:
                stack.pop()

            parent_id = stack[-1].node_id if stack else None
            parent_path = stack[-1].path if stack else ""
            path = f"{parent_path}/{heading.title}" if parent_path else heading.title

            node = TextNode(
                title=heading.title,
                path=path,
                text="",
                parent_id=parent_id,
            )
            nodes_by_id[node.id] = node

            if parent_id is not None:
                nodes_by_id[parent_id].children_ids.append(node.id)

            assign_start = time.perf_counter()
            self._assign_or_split_section_text(
                node=node,
                full_text=markdown_text,
                span_start=section_span.start,
                span_end=section_span.end,
                token_offsets=token_offsets,
                nodes_by_id=nodes_by_id,
            )
            _record("assign_or_split", assign_start)

            stack.append(_SectionTrail(heading.level, node.id, path))

        if profile is not None:
            profile["total"] = time.perf_counter() - start_total
        return _ChunkInternalResult(list(nodes_by_id.values()), profile)

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
        nodes_by_id: dict[uuid.UUID, TextNode],
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
        self._create_part_children(parent=node, chunks=chunks, nodes_by_id=nodes_by_id)

    @staticmethod
    def _create_part_children(
        *,
        parent: TextNode,
        chunks: list[str],
        nodes_by_id: dict[uuid.UUID, TextNode],
    ) -> None:
        prev_id: uuid.UUID | None = None

        part_index = 0
        for chunk_text in chunks:
            if not chunk_text:
                continue

            part_index += 1
            title = f"part {part_index}"
            child = TextNode(
                title=title,
                path=f"{parent.path}/{title}",
                text=chunk_text,
                parent_id=parent.id,
                prev_id=prev_id,
            )
            nodes_by_id[child.id] = child
            parent.children_ids.append(child.id)

            if prev_id is not None:
                nodes_by_id[prev_id].next_id = child.id
            prev_id = child.id
