"""CHUNKING (Smart, Semantic).

Smart: structure-aware. A chunk never crosses a page or heading boundary, a record (CSV row, JSON object,
XML element) is never split unless it alone is longer than chunk_size, and prose is packed sentence by
sentence with an overlap carried into the next chunk.

Semantic: prose sentences are embedded and the text is split where the meaning shifts (cosine distance
between neighbouring sentences above a percentile threshold). Each semantic segment is then packed with
the smart rules, so no chunk is longer than chunk_size.
"""

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import groupby
from typing import Literal

import numpy as np

from Search_Engine.Schema.chunk import Chunk
from Search_Engine.Schema.document import Document, Section

EmbedFn = Callable[[list[str]], np.ndarray]
Strategy = Literal["auto", "smart", "semantic"]

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")


@dataclass(frozen=True)
class _Unit:
    """Smallest piece the chunker moves around: a sentence, or a whole record."""

    text: str
    starts_block: bool  # first sentence of a paragraph, or a record: joined with a blank line


def chunk_document(
    doc: Document,
    *,
    strategy: Strategy,
    chunk_size: int,
    overlap: int,
    embed_fn: EmbedFn | None = None,
    breakpoint_percentile: float = 90.0,
) -> list[Chunk]:
    if not 0 <= overlap < chunk_size:
        raise ValueError(f"overlap ({overlap}) must be >= 0 and smaller than chunk_size ({chunk_size})")
    if strategy == "auto":
        strategy = "smart" if doc.structured else "semantic"
    if strategy == "semantic" and embed_fn is None:
        raise ValueError("semantic chunking needs an embed_fn")

    chunks = []
    for (page, heading), sections in groupby(doc.sections, key=lambda s: (s.page, s.heading)):
        units = list(_units(sections, doc.structured, chunk_size))
        if strategy == "semantic":
            texts = _semantic_chunks(units, embed_fn, chunk_size, overlap, breakpoint_percentile)
        else:
            texts = _pack(units, chunk_size, overlap)

        metadata = {"section": heading} if heading else {}
        chunks.extend(
            Chunk(text=text, source=doc.source, modality=doc.modality, page=page, metadata=dict(metadata))
            for text in texts
        )
    return chunks


def _units(sections: Iterable[Section], structured: bool, chunk_size: int) -> Iterator[_Unit]:
    for section in sections:
        blocks = [section.text] if structured else _PARAGRAPH_BREAK.split(section.text)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            sentences = [block] if structured else _SENTENCE_END.split(block)
            first = True
            for sentence in sentences:
                for part in _hard_split(sentence.strip(), chunk_size):
                    yield _Unit(part, starts_block=first)
                    first = False


def _hard_split(text: str, limit: int) -> list[str]:
    """Last resort for one sentence/record longer than the limit: cut at a line break or space."""
    pieces = []
    while len(text) > limit:
        cut = text.rfind("\n", limit // 2, limit + 1)
        if cut <= 0:
            cut = text.rfind(" ", 0, limit + 1)
        if cut <= 0:
            cut = limit
        pieces.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        pieces.append(text)
    return pieces


def _pack(units: list[_Unit], chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[_Unit] = []
    for unit in units:
        if current and len(_join([*current, unit])) > chunk_size:
            chunks.append(_join(current))
            current = _overlap_tail(current, overlap)
            if current and len(_join([*current, unit])) > chunk_size:
                current = []  # the overlap plus this unit doesn't fit: start clean
        current.append(unit)
    if current:
        chunks.append(_join(current))
    return chunks


def _overlap_tail(units: list[_Unit], overlap: int) -> list[_Unit]:
    """The last whole units of a chunk that fit in `overlap` characters."""
    tail: list[_Unit] = []
    for unit in reversed(units):
        if len(_join([unit, *tail])) > overlap:
            break
        tail.insert(0, unit)
    return tail


def _join(units: list[_Unit]) -> str:
    parts = []
    for index, unit in enumerate(units):
        if index:
            parts.append("\n\n" if unit.starts_block else " ")
        parts.append(unit.text)
    return "".join(parts)


def _semantic_chunks(
    units: list[_Unit], embed_fn: EmbedFn, chunk_size: int, overlap: int, percentile: float
) -> list[str]:
    if len(units) < 3:
        return _pack(units, chunk_size, overlap)

    texts = [unit.text for unit in units]
    # Embed each sentence together with its neighbours so one short sentence doesn't cause a false breakpoint.
    windows = [" ".join(texts[max(i - 1, 0) : i + 2]) for i in range(len(texts))]
    vectors = np.asarray(embed_fn(windows), dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    distances = 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)  # distances[i]: between unit i and i+1
    threshold = np.percentile(distances, percentile)

    segments, start = [], 0
    for i, distance in enumerate(distances, start=1):
        if distance > threshold:
            segments.append(units[start:i])
            start = i
    segments.append(units[start:])

    # No overlap across a semantic breakpoint (it's a topic change); _pack adds overlap inside long segments.
    return [text for segment in segments for text in _pack(segment, chunk_size, overlap)]
