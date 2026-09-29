from langchain_community.cross_encoders import BaseCrossEncoder
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from search_engine.core.config import Settings
from search_engine.retrieval.rerank import Reranker


class ListRetriever(BaseRetriever):
    docs: list[Document]

    def _get_relevant_documents(self, query, *, run_manager):
        return self.docs


class FakeSearch:
    def __init__(self, text=(), images=()):
        self.lists = {"text": ListRetriever(docs=list(text)), "images": ListRetriever(docs=list(images))}
        self.filters = None

    def retrievers(self, filters):
        self.filters = filters
        return {name: retriever for name, retriever in self.lists.items() if retriever.docs}


class KeywordCrossEncoder(BaseCrossEncoder):
    """Scores a (query, text) pair by how many query words the text contains."""

    def __init__(self):
        self.calls = 0

    def score(self, text_pairs):
        self.calls += 1
        return [sum(word in text.lower() for word in query.lower().split()) for query, text in text_pairs]


def text(chunk_id, content):
    return Document(page_content=content, metadata={"chunk_id": chunk_id, "modality": "text"})


def image(chunk_id, modality="image"):
    return Document(page_content=f"image {chunk_id}", metadata={"chunk_id": chunk_id, "modality": modality})


def reranker(search, **settings):
    return Reranker(Settings(**{"top_k": 10, **settings}), search=search, cross_encoder=KeywordCrossEncoder())


def test_rrf_merges_a_chunk_found_by_both_lists_and_ranks_it_first():
    shared = text("shared", "goroutines")
    search = FakeSearch(text=[text("t1", "a"), shared], images=[text("t2", "b"), shared])

    fused = reranker(search).fuse("q", filters="F")

    assert [d.metadata["chunk_id"] for d in fused][0] == "shared"  # rank 2 in both beats rank 1 in one
    assert len(fused) == 3 and search.filters == "F"


def test_nothing_to_search_returns_nothing():
    assert reranker(FakeSearch()).retrieve("q", filters=None) == []


def test_cross_encoder_reorders_text_and_images_follow_within_their_slots():
    texts = [text("t1", "about cakes"), text("t2", "go goroutines and channels"), text("t3", "go channels")]
    images = [image("i1"), image("i2", "video_frame"), image("i3")]

    result = reranker(FakeSearch(), rerank_image_slots=2).rerank("go goroutines channels", texts + images)

    assert [d.metadata["chunk_id"] for d in result] == ["t2", "t3", "t1", "i1", "i2"]


def test_top_k_keeps_the_best_text_and_images_never_take_half():
    texts = [text(f"t{i}", "match" if i == 3 else "no") for i in range(6)]

    assert [d.metadata["chunk_id"] for d in reranker(FakeSearch()).rerank("match", texts + [image("i1")], top_k=1)] == ["t3"]
    result = reranker(FakeSearch()).rerank("match", texts + [image("i1"), image("i2")], top_k=4)
    assert [d.metadata["chunk_id"] for d in result][0] == "t3" and len(result) == 4
    assert sum(d.metadata["modality"] == "image" for d in result) == 2


def test_images_fill_the_list_when_there_is_no_text():
    images = [image(f"i{i}") for i in range(5)]
    ranker = reranker(FakeSearch(), rerank_image_slots=2)

    result = ranker.rerank("charts", images, top_k=4)

    assert [d.metadata["chunk_id"] for d in result] == ["i0", "i1", "i2", "i3"]
    assert ranker.cross_encoder.calls == 0  # nothing for the cross encoder to read


def test_retrieve_runs_fusion_then_reranking():
    search = FakeSearch(text=[text("t1", "cakes"), text("t2", "goroutines")], images=[image("i1")])

    result = reranker(search).retrieve("goroutines", filters=None)

    assert [d.metadata["chunk_id"] for d in result] == ["t2", "t1", "i1"]
