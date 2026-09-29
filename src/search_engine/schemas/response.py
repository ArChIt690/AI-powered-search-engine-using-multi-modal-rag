"""What goes back to the USER: the answer, its citations and where it came from."""

from enum import StrEnum

from pydantic import BaseModel, Field

from search_engine.schemas.chunk import Modality


class AnswerSource(StrEnum):
    """Which diagram box produced the answer."""

    FAISS_SEMANTIC_CACHE = "faiss_semantic_cache"  # similar question answered before
    REDIS_PROMPT_CACHE = "redis_prompt_cache"  # exact or similar prompt answered before
    LLM = "llm"
    NO_RESULTS = "no_results"  # nothing matched the query and filters, so the LLM wasn't called


class Citation(BaseModel):
    """One retrieved chunk the answer cites as [number]."""

    number: int = Field(description="The [n] the answer uses for this chunk.")
    chunk_id: str
    file_name: str
    source: str = Field(description="Path of the source file.")
    modality: Modality
    content: str = Field(description="text, table, record, image, chart or frame.")
    page: int | None = Field(default=None, description="PDF page.")
    timestamp: float | None = Field(default=None, description="Seconds into the video.")
    snippet: str = Field(description="The start of the chunk's text (for pictures, their label).")


class SearchResponse(BaseModel):
    query: str = Field(description="The question as the user asked it.")
    enhanced_query: str = Field(description="The question after QUERY Enhancement.")
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    source: AnswerSource
    session_id: str | None = None
    guardrail_reason: str | None = Field(default=None, description="Why GUARDRAIL blocked the answer (Part 3).")


class CachedAnswer(BaseModel):
    """What the FAISS Semantic Cache and Redis Prompt Caching store and return on a HIT."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
