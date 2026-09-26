"""CHUNKING (Smart, Semantic): Document sections -> Chunks.

Each kind of content gets the strategy that fits it (chunking_strategy="auto"):
- prose (txt, Markdown, PDF page text): Semantic, cut where the meaning shifts
- records (CSV rows, JSON objects, XML elements): Smart, whole records packed together, never cut in half
- tables (PDF tables as Markdown): Smart, split between rows, header row repeated in every chunk
- video transcripts: kept as the timestamped sections the loader made, so each chunk's start time stays exact

Sections are chunked in groups of the same (page, heading, timestamp, kind), so a chunk never crosses a page,
a heading, a transcript section or the edge of a table. Pictures don't pass through here: in the diagram they go
PICTURES -> Image Embeddings (CLIP).
"""

import re
from itertools import groupby

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from search_engine.core.config import Settings, get_settings
from search_engine.schemas.chunk import Chunk, Modality
from search_engine.schemas.document import Document, SectionKind

# Largest boundary first: record/paragraph, line, sentence, clause, word, character.
_SMART_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
_RECORD_SEPARATOR = "\n\n"
_LETTER = re.compile(r"[^\W\d_]")  # any letter, in any script
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")


def chunk_document(
    doc: Document, embeddings: Embeddings | None = None, settings: Settings | None = None
) -> list[Chunk]:
    """Spark Streaming: Chunking. `embeddings` is needed only when prose is chunked semantically."""
    settings = settings or get_settings()
    size, overlap = settings.chunk_size, settings.chunk_overlap
    if not 0 <= overlap < size:
        raise ValueError(f"chunk_overlap ({overlap}) must be >= 0 and smaller than chunk_size ({size})")

    chunks: list[Chunk] = []
    groups = groupby(doc.sections, key=lambda s: (s.page, s.heading, s.timestamp, s.kind))
    for (page, heading, timestamp, kind), group in groups:
        texts = [text for section in group if (text := section.text.strip())]
        if not texts:
            continue

        if kind == SectionKind.TABLE:
            content, pieces = "table", [piece for table in texts for piece in _split_table(table, size)]
        elif doc.structured:
            content, pieces = "record", _split_records(texts, size)
        elif doc.modality == Modality.VIDEO_TRANSCRIPT:
            content, pieces = "text", [piece for text in texts for piece in _fit(text, size)]
        else:
            content, pieces = "text", _split_prose("\n\n".join(texts), embeddings, settings)
            # Page numbers and similar leftovers ("12", "- 4 -") carry nothing to search for.
            pieces = [piece for piece in pieces if _LETTER.search(piece)]

        metadata = {"content": content, **({"section": heading} if heading else {})}
        chunks.extend(
            Chunk(
                text=piece,
                source=doc.source,
                modality=doc.modality,
                page=page,
                timestamp=timestamp,
                metadata=dict(metadata),
            )
            for piece in pieces
        )
    return chunks


# Prose


def _split_prose(text: str, embeddings: Embeddings | None, settings: Settings) -> list[str]:
    strategy = settings.chunking_strategy
    if strategy == "auto":
        strategy = "semantic"
    if strategy == "smart":
        return _smart(settings.chunk_size, settings.chunk_overlap).split_text(text)

    if embeddings is None:
        raise ValueError("semantic chunking of prose needs an embeddings model (pass `embeddings`)")
    pieces = _semantic_split(
        text, embeddings, settings.semantic_breakpoint_percentile, settings.semantic_min_chunk_size
    )
    pieces = _merge_short_tail(pieces, settings.semantic_min_chunk_size, settings.chunk_size)
    # One long topic is still one semantic piece: split it again so no chunk exceeds chunk_size.
    smart = _smart(settings.chunk_size, settings.chunk_overlap)
    return [part for piece in pieces for part in (smart.split_text(piece) if len(piece) > settings.chunk_size else [piece])]


def _semantic_split(text: str, embeddings: Embeddings, percentile: float, min_chunk_size: int) -> list[str]:
    """Cut where the meaning shifts.

    For each gap between two sentences, the two sentences before it are compared with the two after it (cosine
    distance). Two sentences at a time, so one odd sentence doesn't look like a topic change, and the two sides
    never overlap, so the biggest distance falls exactly on the gap where the topic changes. The text is cut at
    gaps in the top (100 - percentile)% of distances, unless the piece so far is shorter than min_chunk_size.
    """
    sentences = [s for s in _SENTENCE_END.split(text) if s.strip()]
    if len(sentences) < 2:
        return [text]

    gaps = range(len(sentences) - 1)  # gap g lies between sentence g and g + 1
    before = [" ".join(sentences[max(0, g - 1) : g + 1]) for g in gaps]
    after = [" ".join(sentences[g + 1 : g + 3]) for g in gaps]
    unique = list(dict.fromkeys(before + after))  # most windows appear on both sides: embed each once
    vectors = np.asarray(embeddings.embed_documents(unique), dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    index = {window: i for i, window in enumerate(unique)}
    left = vectors[[index[w] for w in before]]
    right = vectors[[index[w] for w in after]]
    distances = 1.0 - np.sum(left * right, axis=1)
    threshold = np.percentile(distances, percentile)

    pieces: list[str] = []
    current = [sentences[0]]
    for sentence, distance in zip(sentences[1:], distances, strict=True):
        if distance > threshold and len(" ".join(current)) >= min_chunk_size:
            pieces.append(" ".join(current))
            current = []
        current.append(sentence)
    pieces.append(" ".join(current))
    return pieces


def _merge_short_tail(pieces: list[str], min_chunk_size: int, chunk_size: int) -> list[str]:
    """min_chunk_size can't apply to the final piece while splitting: merge it into the previous one if it fits."""
    if min_chunk_size and len(pieces) > 1 and len(pieces[-1]) < min_chunk_size:
        merged = f"{pieces[-2]} {pieces[-1]}"
        if len(merged) <= chunk_size:
            return [*pieces[:-2], merged]
    return pieces


# Records, tables, transcripts: no overlap, so no record, row or sentence appears in two chunks.


def _split_records(records: list[str], size: int) -> list[str]:
    """Pack whole records into chunks. A record bigger than a chunk is split at its field lines."""
    chunks: list[str] = []
    current: list[str] = []
    for record in records:
        if len(record) > size:
            if current:
                chunks.append(_RECORD_SEPARATOR.join(current))
                current = []
            chunks.extend(_split_big_record(record, size))
        elif current and len(_RECORD_SEPARATOR.join([*current, record])) > size:
            chunks.append(_RECORD_SEPARATOR.join(current))
            current = [record]
        else:
            current.append(record)
    if current:
        chunks.append(_RECORD_SEPARATOR.join(current))
    return chunks


def _split_big_record(record: str, size: int) -> list[str]:
    """Repeat the record's first line (usually its id) at the top of every piece, so each piece says whose it is."""
    first, *rest = record.split("\n")
    if not rest or len(first) > size // 2:  # no usable first line to repeat
        return _fit(record, size)
    return _pack_lines(rest, prefix=first, size=size)


def _split_table(table: str, size: int) -> list[str]:
    """Split a Markdown table between rows, repeating the header row and its |---| line in every chunk."""
    if len(table) <= size:
        return [table]
    lines = table.split("\n")
    has_header = len(lines) > 2 and set(lines[1].replace("|", "").strip()) <= set("-: ")
    header = "\n".join(lines[:2]) if has_header else ""
    if len(header) > size // 2:  # a huge header would leave no room for rows
        return _fit(table, size)
    return _pack_lines(lines[2:] if has_header else lines, prefix=header, size=size)


def _pack_lines(lines: list[str], prefix: str, size: int) -> list[str]:
    """Pack whole lines into chunks that each start with `prefix`. A line too long to fit is split at word boundaries."""
    head = f"{prefix}\n" if prefix else ""
    budget = size - len(head)
    chunks: list[str] = []
    current: list[str] = []
    for line in lines:
        for part in _fit(line, budget):
            if current and len("\n".join([*current, part])) > budget:
                chunks.append(head + "\n".join(current))
                current = []
            current.append(part)
    if current:
        chunks.append(head + "\n".join(current))
    return chunks


def _fit(text: str, size: int) -> list[str]:
    """Text that fits stays whole; longer text is split at the largest boundary that fits, without overlap."""
    return [text] if len(text) <= size else _smart(size, 0).split_text(text)


def _smart(size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=_SMART_SEPARATORS,
        keep_separator="end",  # "end" keeps the full stop on its sentence instead of starting the next chunk
    )
