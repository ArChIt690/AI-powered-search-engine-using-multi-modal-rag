"""USER -> QUERY: what a search request carries, and what QUERY Enhancement makes of it."""

from datetime import date

from pydantic import BaseModel, Field, field_validator

from search_engine.schemas.chunk import ContentKind, Modality


class SearchFilters(BaseModel):
    """METADATA FILTERING options, on the fields Metadata Enrichment (1.4) sets. Empty = no filter.

    Set by the USER (API/CLI options) or found in the query by QUERY Enhancement ("only the PDFs").
    """

    model_config = {"extra": "forbid"}  # an unknown filter fails loudly instead of being silently ignored

    modality: list[Modality] = Field(
        default_factory=list, description="Kinds of content: text, image, video_transcript, video_frame."
    )
    content: list[ContentKind] = Field(
        default_factory=list, description="What the chunk holds: text, table, record, image, chart, frame."
    )
    file_type: list[str] = Field(
        default_factory=list, description="File types without the dot, e.g. pdf, md, csv, json, xml, png, mp4."
    )
    file_name: list[str] = Field(default_factory=list, description="Exact file names with extension, e.g. report.pdf.")
    created_after: date | None = Field(default=None, description="Only files created on or after this date.")
    created_before: date | None = Field(default=None, description="Only files created on or before this date.")

    @field_validator("file_type")
    @classmethod
    def _normalise_file_type(cls, values: list[str]) -> list[str]:
        return [value.strip().lower().lstrip(".") for value in values if value.strip()]

    @field_validator("file_name")
    @classmethod
    def _strip_file_name(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]

    def is_empty(self) -> bool:
        return self == SearchFilters()


class SearchRequest(BaseModel):
    """USER -> QUERY."""

    query: str = Field(min_length=1, max_length=2000, description="The user's question.")
    session_id: str | None = Field(
        default=None, max_length=128, description="Groups a user's queries into a session (Sessional Queries)."
    )
    filters: SearchFilters = Field(default_factory=SearchFilters, description="Explicit METADATA FILTERING options.")
    top_k: int | None = Field(default=None, ge=1, le=50, description="Chunks given to the LLM; default TOP_K.")

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()


class EnhancedQuery(BaseModel):
    """QUERY Enhancement's output (the LLM's structured output schema; the descriptions are its instructions)."""

    query: str = Field(
        description=(
            "The latest question rewritten so it can be understood without the conversation: resolve pronouns "
            "and references ('it', 'that chart', 'page 3') and fix typos. Keep its meaning and language; do not "
            "answer it. If it already stands alone, return it unchanged."
        )
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Up to 5 extra search terms (synonyms, expanded abbreviations) that are not already in the query.",
    )
    filters: SearchFilters = Field(
        default_factory=SearchFilters,
        description=(
            "Only filters the user explicitly asked for, e.g. 'in the PDFs' -> file_type ['pdf'], 'in the video' -> "
            "modality ['video_transcript', 'video_frame'], 'show me charts' -> content ['chart']. Leave empty otherwise."
        ),
    )

    @property
    def search_text(self) -> str:
        """What the hybrid search runs on: the rewritten query plus the new keywords."""
        extra = [word for word in self.keywords if word.lower() not in self.query.lower()]
        return " ".join([self.query, *extra])
