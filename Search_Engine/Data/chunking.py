"""CHUNKING (Smart, Semantic).

Smart: LangChain's RecursiveCharacterTextSplitter splits on the largest boundary that fits: record or
paragraph, then line, sentence, clause, word. A record (CSV row, JSON object, XML element) or paragraph
stays whole unless it alone is longer than chunk_size. Markdown heading structure is handled earlier, in the
loader (ExperimentalMarkdownSyntaxTextSplitter).

Semantic: LangChain's SemanticChunker embeds each sentence together with its neighbours and splits where
the meaning shifts. A semantic chunk longer than chunk_size is split again with the smart splitter.

Both strategies chunk each (page, heading) group on its own, so a chunk never crosses a page or heading.
"""

from itertools import groupby
from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from Search_Engine.Schema.chunk import Chunk
from Search_Engine.Schema.document import Document

Strategy = Literal["auto", "smart", "semantic"]

# Largest boundary first: record/paragraph, line, sentence, clause, word, character.
_SMART_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def chunk_document(
    doc: Document,
    *,
    strategy: Strategy,
    chunk_size: int,
    overlap: int,
    embeddings: Embeddings | None = None,
    breakpoint_percentile: float = 90.0,
    min_chunk_size: int | None = None,
) -> list[Chunk]:
    if not 0 <= overlap < chunk_size:
        raise ValueError(f"overlap ({overlap}) must be >= 0 and smaller than chunk_size ({chunk_size})")
    if strategy == "auto":
        strategy = "smart" if doc.structured else "semantic"
    if strategy == "semantic" and embeddings is None:
        raise ValueError("semantic chunking needs embeddings")

    smart = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=_SMART_SEPARATORS,
        keep_separator="end",  # "end" keeps the full stop on its sentence instead of starting the next chunk
    )
    semantic = (
        SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=breakpoint_percentile,
            min_chunk_size=min_chunk_size,
        )
        if strategy == "semantic"
        else None
    )

    chunks = []
    for (page, heading), sections in groupby(doc.sections, key=lambda s: (s.page, s.heading)):
        text = "\n\n".join(stripped for section in sections if (stripped := section.text.strip()))
        if not text:
            continue

        if semantic is None:
            texts = smart.split_text(text)
        else:
            pieces = _merge_short_tail(semantic.split_text(text), min_chunk_size, chunk_size)
            texts = [part for piece in pieces for part in (smart.split_text(piece) if len(piece) > chunk_size else [piece])]

        metadata = {"section": heading} if heading else {}
        chunks.extend(
            Chunk(text=text, source=doc.source, modality=doc.modality, page=page, metadata=dict(metadata))
            for text in texts
        )
    return chunks


def _merge_short_tail(pieces: list[str], min_chunk_size: int | None, chunk_size: int) -> list[str]:
    """SemanticChunker's min_chunk_size doesn't apply to the final piece: merge it into the previous one if it fits."""
    if min_chunk_size and len(pieces) > 1 and len(pieces[-1]) < min_chunk_size:
        merged = f"{pieces[-2]} {pieces[-1]}"
        if len(merged) <= chunk_size:
            return [*pieces[:-2], merged]
    return pieces
