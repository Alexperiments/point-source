"""Markdown chunking service."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence  # noqa: TC003
from dataclasses import dataclass
from typing import Protocol

import logfire

from src.core.rag_config import CHUNKING_SETTINGS
from src.models.node import DocumentNode, TextNode
from src.services.tokenization_service import TokenizerService, TokenOffsets


@dataclass(frozen=True, slots=True)
class _Heading:
    level: int
    title: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _Span:
    """Half-open span [start, end)."""

    start: int
    end: int

    def empty(self) -> bool:
        return self.start >= self.end


@dataclass(frozen=True, slots=True)
class _Unit:
    span: _Span
    tokens: int


@dataclass(frozen=True, slots=True)
class _StackEntry:
    level: int
    node: TextNode
    path: str


class ChunkingStrategy(Protocol):
    """Chunking strategy protocol."""

    def split_oversized_span(
        self,
        token_offsets: TokenOffsets,
        text: str,
        span_start: int,
        span_end: int,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        """Schema for method to split oversized spans."""


def _trim_span(text: str, start: int, end: int) -> _Span:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return _Span(start, end)


def _merge_spans(spans: Iterable[_Span]) -> list[_Span]:
    spans = sorted(spans, key=lambda s: s.start)
    if not spans:
        return []
    out: list[_Span] = []
    cur = spans[0]
    for s in spans[1:]:
        if s.start <= cur.end:
            cur = _Span(cur.start, max(cur.end, s.end))
        else:
            out.append(cur)
            cur = s
    out.append(cur)
    return out


def _intersects_any(spans: list[_Span], start: int, end: int) -> bool:
    """True if [start, end) intersects any span. Assumes spans list is small."""
    if not spans:
        return False
    if start > end:
        return False
    if start == end:
        return any(s.start <= start < s.end for s in spans)
    a = start
    b = end - 1
    return any(s.start <= a < s.end or s.start <= b < s.end for s in spans)


def _scan_nonspace(text: str, idx: int, bound: int, step: int) -> int | None:
    """Scan for next non-space starting at idx moving by step until passing bound."""
    if step < 0:
        i = min(idx, len(text) - 1)
        while i >= bound:
            if not text[i].isspace():
                return i
            i += step
        return None

    i = max(idx, 0)
    while i < bound:
        if not text[i].isspace():
            return i
        i += step
    return None


def _pattern_spans(text: str, start: int, end: int, pattern: re.Pattern) -> list[_Span]:
    return [_Span(m.start(), m.end()) for m in pattern.finditer(text, start, end)]


class ParagraphSentenceMathChunkingStrategy:
    """Chunk by paragraph, fall back to sentences, and keep math intact.

    Notes:
      - We treat block math specially to avoid splitting it away from surrounding prose if it looks like continuation.
      - Sentence splitting uses a boundary regex plus a few heuristics (abbreviations, latex/citation continuation).

    """

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
        min_chunk_chars: int = CHUNKING_SETTINGS.min_chunk_chars,
    ) -> None:
        """Configure regex patterns and minimum chunk size for splitting."""
        self._p_break = paragraph_break_pattern
        self._s_boundary = sentence_boundary_pattern
        self._inline_math = inline_math_pattern
        self._block_math = block_math_pattern
        self._citation_prefixes = citation_command_prefixes
        self._min_chunk_chars = min_chunk_chars
        self._abbr = re.compile(r"(?:\b(?:e|i)\.g\.|\b(?:e|i)\.e\.)$", re.IGNORECASE)

    def split_oversized_span(
        self,
        token_offsets: TokenOffsets,
        text: str,
        span_start: int,
        span_end: int,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[str]:
        """Split an oversized span into chunkable text respecting token limits."""
        span = _Span(span_start, span_end)
        if span.empty():
            return [text[span_start:span_end]]

        math_spans, block_spans = self._collect_math_spans(text, span)
        units = self._units_for_span(
            token_offsets,
            text,
            span,
            math_spans,
            block_spans,
            max_tokens,
        )

        windows = self._pack(
            units,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        windows = [w for w in windows if not w.empty()]

        if self._min_chunk_chars > 0 and len(windows) > 1:
            windows = self._merge_small(windows)

        return [text[w.start : w.end] for w in windows]

    def _collect_math_spans(
        self,
        text: str,
        span: _Span,
    ) -> tuple[list[_Span], list[_Span]]:
        block = _merge_spans(self._collect_block_math(text, span))
        inline = _pattern_spans(text, span.start, span.end, self._inline_math)
        merged = _merge_spans([*block, *inline])
        return merged, block

    @staticmethod
    def _collect_block_math(text: str, span: _Span) -> list[_Span]:
        r"""Collect $$...$$, \\[...\\], and \\begin{...}...\\end{...} spans (best-effort)."""
        out: list[_Span] = []
        cursor = span.start
        end_limit = span.end

        while cursor < end_limit:
            starts = [
                ("dollars", text.find("$$", cursor, end_limit)),
                ("bracket", text.find("\\[", cursor, end_limit)),
                ("begin", text.find("\\begin{", cursor, end_limit)),
            ]
            starts = [(k, p) for k, p in starts if p != -1]
            if not starts:
                break

            kind, start = min(starts, key=lambda kp: kp[1])

            if kind == "dollars":
                end = text.find("$$", start + 2, end_limit)
                end_len = 2
            elif kind == "bracket":
                end = text.find("\\]", start + 2, end_limit)
                end_len = 2
            else:
                end_brace = text.find("}", start + 7, end_limit)
                if end_brace == -1:
                    break
                env = text[start + 7 : end_brace]
                end_tag = f"\\end{{{env}}}"
                end = text.find(end_tag, end_brace + 1, end_limit)
                end_len = len(end_tag)

            if end == -1:
                break

            out.append(_Span(start, end + end_len))
            cursor = end + end_len

        return out

    def _units_for_span(
        self,
        token_offsets: TokenOffsets,
        text: str,
        span: _Span,
        math_spans: list[_Span],
        block_math_spans: list[_Span],
        max_tokens: int,
    ) -> list[_Unit]:
        units: list[_Unit] = []
        for p in self._split_paragraphs(text, span, math_spans, block_math_spans):
            if not text[p.start : p.end].strip():
                continue

            p_tokens = token_offsets.count_tokens_from_span(p.start, p.end)
            if p_tokens <= max_tokens:
                units.append(_Unit(p, p_tokens))
                continue

            sents = self._split_sentences(text, p, math_spans)
            if len(sents) <= 1:
                units.append(_Unit(p, p_tokens))
                continue

            for s in sents:
                if not text[s.start : s.end].strip():
                    continue
                units.append(
                    _Unit(s, token_offsets.count_tokens_from_span(s.start, s.end)),
                )

        if not units:
            units.append(
                _Unit(span, token_offsets.count_tokens_from_span(span.start, span.end)),
            )
        return units

    def _split_paragraphs(
        self,
        text: str,
        span: _Span,
        math_spans: list[_Span],
        block_math_spans: list[_Span],
    ) -> list[_Span]:
        out: list[_Span] = []
        cursor = span.start

        for m in self._p_break.finditer(text, span.start, span.end):
            sep_s, sep_e = m.start(), m.end()
            if sep_s <= cursor:
                continue
            if _intersects_any(math_spans, sep_s, sep_e):
                continue
            if self._skip_paragraph_split(text, sep_s, sep_e, span, block_math_spans):
                continue

            out.append(_Span(cursor, sep_s))
            cursor = sep_e

        if cursor < span.end:
            out.append(_Span(cursor, span.end))
        return out

    def _split_sentences(
        self,
        text: str,
        para: _Span,
        math_spans: list[_Span],
    ) -> list[_Span]:
        boundaries: list[int] = []
        for m in self._s_boundary.finditer(text, para.start, para.end):
            b = m.end()
            if not (para.start < b < para.end):
                continue
            if _intersects_any(math_spans, m.start(), b) or self._ends_with_abbr(
                text,
                m.start(),
            ):
                continue
            nxt = _scan_nonspace(text, b, para.end, +1)
            if nxt is None or self._is_continuation(text, nxt):
                continue
            boundaries.append(b)

        if not boundaries:
            return [para]

        out: list[_Span] = []
        cursor = para.start
        for b in boundaries:
            if b > cursor:
                out.append(_Span(cursor, b))
                cursor = b
        if cursor < para.end:
            out.append(_Span(cursor, para.end))
        return out

    def _skip_paragraph_split(
        self,
        text: str,
        sep_start: int,
        sep_end: int,
        span: _Span,
        block_math_spans: list[_Span],
    ) -> bool:
        prev_i = _scan_nonspace(text, sep_start - 1, span.start, -1)
        next_i = _scan_nonspace(text, sep_end, span.end, +1)
        if prev_i is None or next_i is None:
            return False

        prev_in_block = _intersects_any(block_math_spans, prev_i, prev_i + 1)
        next_in_block = _intersects_any(block_math_spans, next_i, next_i + 1)

        if prev_in_block or (
            next_in_block
            and (
                (not self._is_sentence_end(text, prev_i))
                or self._is_continuation(
                    text,
                    next_i,
                )
            )
        ):
            return True

        return self._is_soft_wrap(text, prev_i, next_i)

    def _is_soft_wrap(self, text: str, prev_i: int, next_i: int) -> bool:
        if text[next_i] == "#":
            return False

        end_i = self._normalize_end(text, prev_i)
        if end_i < 0:
            return False

        prev_c, next_c = text[end_i], text[next_i]
        if (
            self._ends_with_abbr(text, end_i)
            or prev_c in ":;,([{"
            or (prev_c == "-" and next_c.islower())
        ):
            return True

        return (not self._is_sentence_end(text, prev_i)) and (
            self._is_continuation(text, next_i) or next_c.islower()
        )

    def _is_sentence_end(self, text: str, idx: int) -> bool:
        i = self._normalize_end(text, idx)
        return i >= 0 and text[i] in ".!?" and not self._ends_with_abbr(text, i)

    def _is_continuation(self, text: str, idx: int) -> bool:
        c = text[idx]
        if c.islower() or c.isdigit() or c in "([{" or c == "$":
            return True
        if c != "\\":
            return False

        nxt_is_bracket = (idx + 1 < len(text)) and text[idx + 1] in "[("
        is_double_slash = text.startswith("\\\\", idx)
        is_citation = any(text.startswith(pfx, idx) for pfx in self._citation_prefixes)
        return nxt_is_bracket or is_double_slash or is_citation

    @staticmethod
    def _normalize_end(text: str, idx: int) -> int:
        i = idx
        while i >= 0 and text[i].isspace():
            i -= 1
        while i >= 0 and text[i] in "\"')]}":
            i -= 1
        return i

    def _ends_with_abbr(self, text: str, punct_index: int) -> bool:
        if punct_index < 0:
            return False
        snippet = text[max(0, punct_index - 8) : punct_index + 1]
        return bool(self._abbr.search(snippet))

    @staticmethod
    def _pack(
        units: list[_Unit],
        *,
        max_tokens: int,
        overlap_tokens: int,
    ) -> list[_Span]:
        if not units:
            return []

        out: list[_Span] = []
        i, n = 0, len(units)

        while i < n:
            total = 0
            j = i
            while j < n and total + units[j].tokens <= max_tokens:
                total += units[j].tokens
                j += 1

            if j == i:
                out.append(units[i].span)
                i += 1
                continue

            out.append(_Span(units[i].span.start, units[j - 1].span.end))

            if overlap_tokens <= 0:
                i = j
                continue

            advance_target = max(1, total - overlap_tokens)
            advanced = 0
            while i < j and advanced < advance_target:
                advanced += units[i].tokens
                i += 1

        return out

    def _merge_small(
        self,
        spans: list[_Span],
    ) -> list[_Span]:
        """Merge spans whose character length is below min_chunk_chars.

        Behavior matches previous implementation: prefer merging forward, else backward for trailing small.
        """
        out: list[_Span] = []
        i = 0
        n = len(spans)

        while i < n:
            s = spans[i]
            t = s.end - s.start
            if n == 1 or t >= self._min_chunk_chars:
                out.append(s)
                i += 1
                continue

            if i + 1 < n:
                spans[i + 1] = _Span(s.start, spans[i + 1].end)
                i += 1
                continue

            if out:
                prev = out.pop()
                out.append(_Span(prev.start, s.end))
            else:
                out.append(s)
            i += 1

        return out


class MarkdownChunker:
    """Chunk markdown documents into a TextNode tree."""

    def __init__(
        self,
        *,
        tokenizer_model_name: str = CHUNKING_SETTINGS.tokenizer_model_name,
        max_tokens: int = CHUNKING_SETTINGS.max_tokens,
        overlap_tokens: int = CHUNKING_SETTINGS.overlap_tokens,
        min_chunk_chars: int = CHUNKING_SETTINGS.min_chunk_chars,
        header_pattern: re.Pattern = CHUNKING_SETTINGS.header_patterns,
        drop_section_title_prefixes: tuple[
            str,
            ...,
        ] = CHUNKING_SETTINGS.drop_section_title_prefixes,
        strategy: ChunkingStrategy | None = None,
    ) -> None:
        """Initialize a chunker with tokenizer, limits, and splitting strategy."""
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._min_chunk_chars = min_chunk_chars
        self._header_pattern = header_pattern
        self._drop_prefixes = tuple(
            self._norm_title(p) for p in drop_section_title_prefixes
        )

        self._strategy = strategy or ParagraphSentenceMathChunkingStrategy()
        self._strategy_name = type(self._strategy).__name__
        self._tokenizer_name = tokenizer_model_name
        self._tokenizer_service = TokenizerService(
            model_name=tokenizer_model_name,
        )

    @logfire.instrument("chunking_service.chunk", extract_args=False)
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
        base_md = dict(metadata) if metadata else None

        for doc in documents:
            text = doc.text
            offsets = self._tokenizer_service.tokenize_with_offsets_mapping(text)
            headings = self._extract_headings(text)
            if not headings:
                raise ValueError(f"Headers not detected for document ID: {doc.id}")

            carry_text = ""
            drop_level: int | None = None
            stack: list[_StackEntry] = []

            for idx, h in enumerate(headings):
                nxt = headings[idx + 1] if idx + 1 < len(headings) else None
                if drop_level is not None and h.level <= drop_level:
                    drop_level = None

                drop_here = self._should_drop(h.title)
                if drop_level is not None or drop_here:
                    drop_level = h.level if drop_here else drop_level
                    continue

                next_start = nxt.start if nxt else len(text)
                section_span = _trim_span(text, h.end, next_start)
                raw = text[section_span.start : section_span.end]
                combined = "\n".join(p for p in (carry_text, raw) if p)

                while stack and stack[-1].level >= h.level:
                    stack.pop()
                parent_entry = stack[-1] if stack else None
                parent = parent_entry.node if parent_entry else None
                parent_path = parent_entry.path if parent_entry else ""
                path = self._join_path(parent_path, h.title)

                node = self._new_section_node(
                    doc,
                    parent,
                    title=h.title,
                    path=path,
                    base_metadata=base_md,
                )

                should_carry = (
                    nxt is not None
                    and self._min_chunk_chars > 0
                    and combined
                    and len(combined) < self._min_chunk_chars
                )
                if should_carry:
                    carry_text = combined
                    node.text = ""
                else:
                    if carry_text:
                        full_text = combined
                        span = _Span(0, len(combined))
                        span_offsets = (
                            self._tokenizer_service.tokenize_with_offsets_mapping(
                                combined,
                            )
                        )
                    else:
                        full_text = text
                        span = section_span
                        span_offsets = offsets

                    self._assign_or_split(
                        node=node,
                        full_text=full_text,
                        span=span,
                        token_offsets=span_offsets,
                        base_metadata=base_md,
                    )
                    carry_text = ""

                doc.children.append(node)
                stack.append(_StackEntry(h.level, node, path))

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
    def _norm_title(title: str) -> str:
        t = title.lower().replace("&", "and")
        t = re.sub(r"[^\w\s]", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    def _should_drop(self, title: str) -> bool:
        norm = self._norm_title(title)
        return any(norm == p or norm.startswith(f"{p} ") for p in self._drop_prefixes)

    def _assign_or_split(
        self,
        *,
        node: TextNode,
        full_text: str,
        span: _Span,
        token_offsets: TokenOffsets,
        base_metadata: dict[str, object] | None,
    ) -> None:
        if span.empty():
            node.text = ""
            return

        token_count = token_offsets.count_tokens_from_span(span.start, span.end)
        if token_count <= self._max_tokens:
            node.text = full_text[span.start : span.end]
            return

        chunks = self._strategy.split_oversized_span(
            token_offsets=token_offsets,
            text=full_text,
            span_start=span.start,
            span_end=span.end,
            max_tokens=self._max_tokens,
            overlap_tokens=self._overlap_tokens,
        )
        chunks = [c for c in chunks if c]
        node.text = ""
        if chunks:
            self._create_part_children(
                parent=node,
                chunks=chunks,
                base_metadata=base_metadata,
            )

    def _create_part_children(
        self,
        *,
        parent: TextNode,
        chunks: list[str],
        base_metadata: dict[str, object] | None,
    ) -> None:
        parent_path = self._node_path(parent)
        prev: TextNode | None = None

        for i, chunk in enumerate(chunks, start=1):
            title = f"part {i}"
            path = self._join_path(parent_path, title)

            child = TextNode(text=chunk)
            child.parent = parent
            if base_metadata:
                child.node_metadata = dict(base_metadata)
            child.node_metadata = self._with_metadata(
                child.node_metadata,
                title=title,
                path=path,
            )

            if prev:
                child.prev_node = prev
                prev.next_node = child
            prev = child

            parent.document.children.append(child)

    # ---- node/metadata helpers

    def _with_metadata(
        self,
        existing: dict[str, object] | None,
        *,
        title: str,
        path: str,
    ) -> dict[str, object]:
        return {
            **(existing or {}),
            "title": title,
            "path": path,
            "max_tokens": self._max_tokens,
            "overlap_tokens": self._overlap_tokens,
            "tokenizer_name": self._tokenizer_name,
            "strategy_name": self._strategy_name,
        }

    @staticmethod
    def _node_path(node: TextNode) -> str:
        return str((node.node_metadata or {}).get("path") or "")

    @staticmethod
    def _join_path(parent_path: str, title: str) -> str:
        return f"{parent_path}/{title}" if parent_path else title

    def _new_section_node(
        self,
        document: DocumentNode,
        parent: TextNode | None,
        *,
        title: str,
        path: str,
        base_metadata: dict[str, object] | None,
    ) -> TextNode:
        node = TextNode(text="")
        node.parent = parent
        node.document = document
        md = dict(base_metadata) if base_metadata else None
        node.node_metadata = self._with_metadata(md, title=title, path=path)
        return node
