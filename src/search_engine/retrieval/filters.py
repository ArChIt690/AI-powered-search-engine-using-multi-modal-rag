"""METADATA FILTERING: the user's and QUERY Enhancement's filters -> Elasticsearch filter clauses.

The clauses go into both hybrid searches (text and image store), on the `metadata.*` fields 1.5 maps.
"""

import hashlib
import json
from typing import Any

from search_engine.schemas.query import SearchFilters

# SearchFilters field -> the Elasticsearch field it filters (file_name is a text field; exact match on .keyword)
_TERM_FIELDS = {
    "modality": "metadata.modality",
    "content": "metadata.content",
    "file_type": "metadata.file_type",
    "file_name": "metadata.file_name.keyword",
}
_CREATED_FIELD = "metadata.created"


def merge_filters(user: SearchFilters, extracted: SearchFilters) -> SearchFilters:
    """Per field, the user's explicit option wins; otherwise the filter QUERY Enhancement found in the query."""
    merged = {
        name: getattr(user, name) if getattr(user, name) else getattr(extracted, name)
        for name in SearchFilters.model_fields
    }
    return SearchFilters(**merged)


def to_es_filter(filters: SearchFilters) -> list[dict[str, Any]]:
    """Elasticsearch `filter` clauses: values within a field are OR-ed, different fields are AND-ed."""
    clauses: list[dict[str, Any]] = []
    for name, es_field in _TERM_FIELDS.items():
        values = getattr(filters, name)
        if values:
            clauses.append({"terms": {es_field: [str(value) for value in values]}})
    created = {}
    if filters.created_after:
        created["gte"] = filters.created_after.isoformat()
    if filters.created_before:
        created["lte"] = filters.created_before.isoformat()
    if created:
        clauses.append({"range": {_CREATED_FIELD: created}})
    return clauses


def filters_key(filters: SearchFilters) -> str:
    """A short stable key for a set of filters (value order ignored), so cached answers are reused only for the
    same filters: "revenue in the PDFs" must not get the answer cached for "revenue in the video"."""
    data = filters.model_dump(mode="json")
    normalised = {name: sorted(value) if isinstance(value, list) else value for name, value in data.items()}
    return hashlib.sha1(json.dumps(normalised, sort_keys=True).encode()).hexdigest()[:16]


def es_fields() -> set[str]:
    """Every Elasticsearch field the filters use (checked against the 1.5 mappings in the tests)."""
    return {*_TERM_FIELDS.values(), _CREATED_FIELD}
