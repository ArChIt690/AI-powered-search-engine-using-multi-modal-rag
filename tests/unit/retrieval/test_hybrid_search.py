from types import SimpleNamespace

from langchain_core.documents import Document

from search_engine.core.config import Settings
from search_engine.retrieval.hybrid_search import HYBRID, HybridSearch, to_document
from search_engine.schemas.query import SearchFilters


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, query):
        return self.docs


class FakeStore:
    """Stands in for an ElasticsearchStore: records the search kwargs each retriever was built with."""

    def __init__(self, docs, exists=True):
        self.docs, self.search_kwargs = docs, []
        self.client = SimpleNamespace(indices=SimpleNamespace(exists=lambda index: exists))

    def as_retriever(self, search_kwargs):
        self.search_kwargs.append(search_kwargs)
        return FakeRetriever(self.docs)


def _search(text_docs=(), image_docs=(), text_exists=True, image_exists=True):
    text, images = FakeStore(list(text_docs), text_exists), FakeStore(list(image_docs), image_exists)
    return HybridSearch(Settings(retrieval_candidates=30), text_store=text, image_store=images), text, images


def test_returns_both_ranked_lists():
    search, _, _ = _search([Document("goroutines")], [Document("chart from go.pdf")])

    results = search.search("go")

    assert [d.page_content for d in results.text] == ["goroutines"]
    assert [d.page_content for d in results.images] == ["chart from go.pdf"]


def test_both_stores_get_the_metadata_filter_clauses_and_sizes():
    search, text, images = _search()

    search.search("q", SearchFilters(file_type=["pdf"]))

    for store in (text, images):
        kwargs = store.search_kwargs[0]
        assert kwargs["filter"] == [{"terms": {"metadata.file_type": ["pdf"]}}]
        assert kwargs["k"] == 30 and kwargs["fetch_k"] >= 30 and kwargs["doc_builder"] is to_document


def test_a_store_the_filters_rule_out_is_not_searched():
    search, _, _ = _search()

    assert set(search.retrievers(SearchFilters(modality=["video_frame"]))) == {"images"}
    assert set(search.retrievers(SearchFilters(content=["table"]))) == {"text"}
    assert set(search.retrievers(SearchFilters(modality=["text", "image"]))) == {"text", "images"}
    assert search.retrievers(SearchFilters(modality=["text"], content=["chart"])) == {}  # nothing can match


def test_a_missing_index_is_skipped_instead_of_failing():
    search, _, _ = _search([Document("t")], [Document("i")], image_exists=False)

    results = search.search("q")

    assert results.images == [] and len(results.text) == 1


def test_hits_keep_their_chunk_id():
    hit = {"_id": "abc-3", "_source": {"text": "Goroutines", "metadata": {"page": 2}}}

    doc = to_document(hit)

    assert doc.id == "abc-3" and doc.page_content == "Goroutines"
    assert doc.metadata == {"page": 2, "chunk_id": "abc-3"}


def test_text_store_uses_langchains_hybrid_search_without_es_rrf():
    assert HYBRID.hybrid is True and HYBRID.rrf is False  # ES's RRF returns 403 on the basic license
