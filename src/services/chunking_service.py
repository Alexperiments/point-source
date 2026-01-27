"""Chunking service."""

from __future__ import annotations

import re  # noqa: TC003
from dataclasses import dataclass

import logfire

from src.core.chunking_config import CHUNKING_SETTINGS
from src.models.node import DocumentNode, TextNode
from src.services.tokenizer_service import TokenizerFactory, TokenOffsets


chunked_documents_metric = logfire.metric_counter(
    "documents_chunked",
    unit="1",
    description="Number of documents chunked.",
)


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
        block_equation_spans = self._collect_block_equation_spans(
            text,
            span_start,
            span_end,
        )
        units = self._build_units(
            token_offsets=token_offsets,
            text=text,
            span_start=span_start,
            span_end=span_end,
            equation_spans=equation_spans,
            block_equation_spans=block_equation_spans,
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
        block_equation_spans: list[_SpanRange],
        max_tokens: int,
    ) -> list[_SpanUnit]:
        units: list[_SpanUnit] = []
        paragraph_spans = self._split_paragraphs(
            text=text,
            span_start=span_start,
            span_end=span_end,
            equation_spans=equation_spans,
            block_equation_spans=block_equation_spans,
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
        block_equation_spans: list[_SpanRange],
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
            if self._should_merge_paragraph_break(
                text=text,
                sep_start=sep_start,
                sep_end=sep_end,
                span_start=span_start,
                span_end=span_end,
                block_equation_spans=block_equation_spans,
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
            next_idx = self._find_next_nonspace(text, boundary, para_end)
            if next_idx is not None and self._looks_like_continuation(text, next_idx):
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

    def _collect_block_equation_spans(
        self,
        text: str,
        span_start: int,
        span_end: int,
    ) -> list[_SpanRange]:
        matches = [
            _SpanRange(match.start(), match.end())
            for match in self._block_math_pattern.finditer(text, span_start, span_end)
        ]
        if not matches:
            return []

        matches.sort(key=lambda span: span.start)
        merged: list[_SpanRange] = []
        current_start = matches[0].start
        current_end = matches[0].end
        for span in matches[1:]:
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
    def _find_prev_nonspace(text: str, start: int, lower_bound: int) -> int | None:
        index = min(start, len(text) - 1)
        while index >= lower_bound:
            if not text[index].isspace():
                return index
            index -= 1
        return None

    @staticmethod
    def _find_next_nonspace(text: str, start: int, upper_bound: int) -> int | None:
        index = max(start, 0)
        while index < upper_bound:
            if not text[index].isspace():
                return index
            index += 1
        return None

    @classmethod
    def _looks_like_sentence_end(cls, text: str, index: int) -> bool:
        cursor = index
        while cursor >= 0 and text[cursor].isspace():
            cursor -= 1
        while cursor >= 0 and text[cursor] in "\"')]}":
            cursor -= 1
        return cursor >= 0 and text[cursor] in ".!?"

    @classmethod
    def _looks_like_continuation(cls, text: str, index: int) -> bool:
        char = text[index]
        if char.islower() or char.isdigit():
            return True
        return char in "([{"

    def _should_merge_paragraph_break(
        self,
        *,
        text: str,
        sep_start: int,
        sep_end: int,
        span_start: int,
        span_end: int,
        block_equation_spans: list[_SpanRange],
    ) -> bool:
        prev_idx = self._find_prev_nonspace(text, sep_start - 1, span_start)
        next_idx = self._find_next_nonspace(text, sep_end, span_end)
        if prev_idx is None or next_idx is None:
            return False

        prev_inside_block = self._is_inside_equation(prev_idx, block_equation_spans)
        next_inside_block = self._is_inside_equation(next_idx, block_equation_spans)
        if not (prev_inside_block or next_inside_block):
            return False

        prev_sentence_end = self._looks_like_sentence_end(text, prev_idx)
        next_continuation = self._looks_like_continuation(text, next_idx)
        return (not prev_sentence_end) or next_continuation

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
    """Main markdown chunker service."""

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

        self._tokenizer = TokenizerFactory.create(self._tokenizer_name)

    @logfire.instrument(
        "document_chunking_service",
        extract_args=["documents"],
    )
    def chunk(self, documents: list[DocumentNode]) -> None:
        """Chunk documents into TextNode trees rooted at each DocumentNode."""
        self._chunk_internal(documents)
        chunked_documents_metric.add(len(documents))
        logfire.info("documents_chunking completed")

    def _chunk_internal(
        self,
        documents: list[DocumentNode],
    ) -> None:
        for document in documents:
            markdown_text = document.text
            token_offsets = self._tokenizer.tokenize_with_offsets(
                markdown_text,
            )
            headings = self._extract_headings(markdown_text)

            if not headings:
                raise ValueError(f"Headers not detected for document ID: {document.id}")

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
                node.parent = parent_node
                node.document = document
                node.node_metadata = {
                    **(node.node_metadata or {}),
                    "title": heading.title,
                    "path": path,
                }

                self._assign_or_split_section_text(
                    node=node,
                    full_text=markdown_text,
                    span_start=section_span.start,
                    span_end=section_span.end,
                    token_offsets=token_offsets,
                )
                document.children.append(node)

                stack.append(_SectionTrail(heading.level, node, path))

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
        self._create_part_children(parent=node, chunks=chunks)

    @staticmethod
    def _create_part_children(
        *,
        parent: TextNode,
        chunks: list[str],
    ) -> None:
        prev_node: TextNode | None = None

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
            parent.document.children.append(child)
