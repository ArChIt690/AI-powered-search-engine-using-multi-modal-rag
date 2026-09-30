"""The numbered passages the LLM answers from: the reranked chunks, plus whatever its TOOLS fetch later.

Every passage has one number for the whole question, so a tool's new passages continue the numbering and the
answer's [n] marks always point to one chunk.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from search_engine.core.config import Settings
from search_engine.schemas.chunk import Modality
from search_engine.schemas.query import SearchFilters

if TYPE_CHECKING:
    from search_engine.retrieval.rerank import Reranker

_PICTURES = {Modality.IMAGE.value, Modality.VIDEO_FRAME.value}


@dataclass
class AnswerContext:
    """What one question's tools share: the settings, the search they can run, and the passages so far."""

    settings: Settings
    chunks: list[Document]
    filters: SearchFilters = field(default_factory=SearchFilters)
    reranker: "Reranker | None" = None
    tools_used: list[str] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)

    def add(self, docs: Sequence[Document]) -> str:
        """Adds passages not already in the context and returns them numbered, ready to show the LLM."""
        known = {doc.metadata.get("chunk_id") for doc in self.chunks}
        new = [doc for doc in docs if doc.metadata.get("chunk_id") not in known]
        first = len(self.chunks) + 1
        self.chunks.extend(new)
        if not new:
            return "No new passages: everything found is already in the context."
        return "\n\n".join(format_passage(number, doc) for number, doc in enumerate(new, start=first))


def format_context(chunks: Sequence[Document]) -> str:
    """Numbered passages, each with where it comes from, so the LLM can cite [n] and the reader can check it."""
    return "\n\n".join(format_passage(number, doc) for number, doc in enumerate(chunks, start=1))


def format_passage(number: int, doc: Document) -> str:
    m = doc.metadata
    where = ", ".join(part for part in (m.get("file_name"), _location(m), m.get("content")) if part)
    if m.get("modality") in _PICTURES:
        return f"[{number}] ({where}) A picture; only this description is known: {doc.page_content}"
    return f"[{number}] ({where})\n{doc.page_content}"


def _location(metadata: dict) -> str | None:
    if metadata.get("page") is not None:
        return f"page {metadata['page']}"
    if metadata.get("timestamp") is not None:
        seconds = int(metadata["timestamp"])
        return f"at {seconds // 60:02d}:{seconds % 60:02d}"
    return None
