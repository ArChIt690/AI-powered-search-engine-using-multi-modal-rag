from datetime import date

import pytest
from pydantic import ValidationError

from search_engine.schemas.chunk import Modality
from search_engine.schemas.query import EnhancedQuery, SearchFilters, SearchRequest
from search_engine.schemas.response import AnswerSource, Citation, SearchResponse


def test_request_defaults_and_strips_the_query():
    request = SearchRequest(query="  what is a goroutine?  ")

    assert request.query == "what is a goroutine?"
    assert request.session_id is None and request.top_k is None
    assert request.filters.is_empty()


@pytest.mark.parametrize("bad", [{"query": ""}, {"query": "   "}, {"query": "q", "top_k": 0}, {"query": "q", "top_k": 51}])
def test_request_rejects_bad_input(bad):
    with pytest.raises(ValidationError):
        SearchRequest(**bad)


def test_filters_normalise_file_types_and_names():
    filters = SearchFilters(file_type=[".PDF", " md ", ""], file_name=[" report.pdf ", " "])

    assert filters.file_type == ["pdf", "md"]
    assert filters.file_name == ["report.pdf"]
    assert not filters.is_empty()


def test_filters_reject_unknown_fields_and_values():
    with pytest.raises(ValidationError):
        SearchFilters(author=["me"])  # not a filter
    with pytest.raises(ValidationError):
        SearchFilters(modality=["audio"])  # not a Modality
    with pytest.raises(ValidationError):
        SearchFilters(content=["video"])  # not a ContentKind


def test_filters_parse_modalities_and_dates():
    filters = SearchFilters(modality=["video_frame"], created_after="2024-01-31")

    assert filters.modality == [Modality.VIDEO_FRAME]
    assert filters.created_after == date(2024, 1, 31)


def test_search_text_adds_only_new_keywords():
    enhanced = EnhancedQuery(query="How do Go goroutines work?", keywords=["goroutines", "concurrency", "Go", "threads"])

    assert enhanced.search_text == "How do Go goroutines work? concurrency threads"
    assert EnhancedQuery(query="plain").search_text == "plain"
    assert EnhancedQuery(query="goroutines?", keywords=["go", "Goroutines"]).search_text == "goroutines? go"


def test_response_round_trips():
    citation = Citation(
        number=1, chunk_id="abc-0", file_name="report.pdf", source="/data/report.pdf", modality="text",
        content="table", page=3, snippet="Region | Revenue",
    )
    response = SearchResponse(
        query="q", enhanced_query="q", answer="North leads [1].", citations=[citation], source=AnswerSource.LLM
    )

    assert SearchResponse.model_validate_json(response.model_dump_json()) == response
    assert response.model_dump(mode="json")["source"] == "llm"
