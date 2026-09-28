from datetime import date

from search_engine.ingestion.vector_store import metadata_mappings
from search_engine.retrieval.filters import es_fields, merge_filters, to_es_filter
from search_engine.schemas.query import SearchFilters


def test_no_filters_means_no_clauses():
    assert to_es_filter(SearchFilters()) == []


def test_each_field_becomes_one_clause_and_values_are_ored():
    filters = SearchFilters(
        modality=["video_transcript", "video_frame"], content=["chart"], file_type=["pdf"], file_name=["report.pdf"]
    )

    assert to_es_filter(filters) == [
        {"terms": {"metadata.modality": ["video_transcript", "video_frame"]}},
        {"terms": {"metadata.content": ["chart"]}},
        {"terms": {"metadata.file_type": ["pdf"]}},
        {"terms": {"metadata.file_name.keyword": ["report.pdf"]}},
    ]


def test_dates_become_one_range_clause():
    assert to_es_filter(SearchFilters(created_after=date(2024, 1, 1), created_before=date(2024, 12, 31))) == [
        {"range": {"metadata.created": {"gte": "2024-01-01", "lte": "2024-12-31"}}}
    ]
    assert to_es_filter(SearchFilters(created_before=date(2024, 12, 31))) == [
        {"range": {"metadata.created": {"lte": "2024-12-31"}}}
    ]


def test_user_filters_win_per_field_and_the_rest_come_from_the_query():
    user = SearchFilters(file_type=["md"])
    extracted = SearchFilters(file_type=["pdf"], content=["table"])

    merged = merge_filters(user, extracted)

    assert merged.file_type == ["md"]  # the user's explicit option, not the LLM's guess
    assert merged.content == ["table"]  # nothing from the user, so the query's filter is used


def test_every_filtered_field_exists_in_the_vector_database_mapping():
    mappings = metadata_mappings()
    for field in es_fields():
        name = field.removeprefix("metadata.").removesuffix(".keyword")
        assert name in mappings, field
        if field.endswith(".keyword"):
            assert "keyword" in mappings[name].get("fields", {}), field
        else:
            assert mappings[name]["type"] in {"keyword", "date"}, field
