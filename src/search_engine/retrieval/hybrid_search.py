"""ELASTICSEARCH HYBRID SEARCH over the Vector Database (the two 1.5 stores), through LangChain.

- text store: LangChain's built-in hybrid search, `DenseVectorStrategy(hybrid=True)`: KEYWORD SEARCH (BM25 on
  `text`) + SEMANTIC SEARCH (kNN on the bge `text_embedding`) in one Elasticsearch query
- image store: SEMANTIC SEARCH (kNN on the CLIP `image_embedding`, query embedded by CLIP's text encoder), so a text
  query finds PDF charts, images and video frames
Both apply the METADATA FILTERING clauses. The two ranked lists are fused in RERANKING.
"""

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_elasticsearch import DenseVectorStrategy, ElasticsearchStore

from search_engine.core.config import Settings, get_settings
from search_engine.ingestion.text_embed import TextEmbedder
from search_engine.ingestion.vector_store import build_image_store, build_text_store
from search_engine.retrieval.filters import to_es_filter
from search_engine.schemas.chunk import Modality
from search_engine.schemas.query import SearchFilters

# rrf=False: Elasticsearch's own RRF needs a paid license (403 on basic); BM25 and kNN scores are summed instead,
# and rank fusion across the text and image lists happens in RERANKING.
HYBRID = DenseVectorStrategy(hybrid=True, rrf=False, text_field="text")

_TEXT_MODALITIES = {Modality.TEXT, Modality.VIDEO_TRANSCRIPT}
_IMAGE_MODALITIES = {Modality.IMAGE, Modality.VIDEO_FRAME}
_TEXT_CONTENT = {"text", "table", "record"}
_IMAGE_CONTENT = {"image", "chart", "frame"}


@dataclass
class HybridResults:
    text: list[Document] = field(default_factory=list)  # ranked, best first
    images: list[Document] = field(default_factory=list)


class HybridSearch:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        text_embedder: TextEmbedder | None = None,
        text_store: ElasticsearchStore | None = None,
        image_store: ElasticsearchStore | None = None,
    ):
        self.settings = settings or get_settings()
        self.text_store = text_store or build_text_store(self.settings, text_embedder, strategy=HYBRID)
        self.image_store = image_store or build_image_store(self.settings)

    def retrievers(self, filters: SearchFilters) -> dict[str, BaseRetriever]:
        """LangChain retrievers for the stores the filters can match ("text", "images"), for RERANKING's fusion.

        A store is left out when the filters rule it out (e.g. only charts) or its index doesn't exist yet.
        """
        clauses = to_es_filter(filters)
        s = self.settings
        wanted = {
            "text": (self.text_store, s.es_text_index, _allows(filters, _TEXT_MODALITIES, _TEXT_CONTENT)),
            "images": (self.image_store, s.es_image_index, _allows(filters, _IMAGE_MODALITIES, _IMAGE_CONTENT)),
        }
        return {
            name: store.as_retriever(search_kwargs=self._search_kwargs(clauses))
            for name, (store, index, allowed) in wanted.items()
            if allowed and store.client.indices.exists(index=index)
        }

    def search(self, query: str, filters: SearchFilters | None = None) -> HybridResults:
        """ELASTICSEARCH HYBRID SEARCH: the text-hybrid and image-kNN hit lists for one query."""
        found = {name: retriever.invoke(query) for name, retriever in self.retrievers(filters or SearchFilters()).items()}
        return HybridResults(text=found.get("text", []), images=found.get("images", []))

    def _search_kwargs(self, clauses: list[dict[str, Any]]) -> dict[str, Any]:
        k = self.settings.retrieval_candidates
        # num_candidates (fetch_k) > k gives the approximate kNN a wider pool, so filtered searches still fill k.
        return {"k": k, "fetch_k": max(100, 2 * k), "filter": clauses, "doc_builder": to_document}


def to_document(hit: dict[str, Any]) -> Document:
    """An Elasticsearch hit -> a LangChain Document that keeps the chunk id (for fusion and citations)."""
    source = hit["_source"]
    return Document(
        id=hit["_id"],
        page_content=source.get("text", ""),
        metadata={**source.get("metadata", {}), "chunk_id": hit["_id"]},
    )


def _allows(filters: SearchFilters, modalities: set[Modality], contents: set[str]) -> bool:
    """False when a modality or content filter can only match the other store."""
    if filters.modality and not modalities & set(filters.modality):
        return False
    if filters.content and not contents & set(filters.content):
        return False
    return True
