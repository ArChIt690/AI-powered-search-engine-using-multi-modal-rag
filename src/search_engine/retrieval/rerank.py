"""RERANKING (RANKFUSION) using Cross Encoder Models.

1. Rank fusion: Reciprocal Rank Fusion of the text-hybrid and image-kNN lists (LangChain `EnsembleRetriever`).
   RRF uses ranks only, so the bge and CLIP scores, which are not comparable, are never mixed.
2. Cross encoder (`BAAI/bge-reranker-base`, LangChain `CrossEncoderReranker`) rereads each fused text hit with the
   query and reorders them. It reads text only, and an image's text is just its label, so the best fused image hits
   are added after the text hits instead. They go last because CLIP's text-to-image scores barely separate related
   from unrelated pictures (measured on the real corpus: 0.64 for a matching chart, 0.645 for an unrelated query),
   so images must never push text evidence out; the LLM is told pictures are known only by their description.
"""

from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import BaseCrossEncoder
from langchain_core.documents import Document

from search_engine.core.config import Settings, get_settings
from search_engine.infra import models
from search_engine.retrieval.hybrid_search import HybridSearch
from search_engine.schemas.chunk import Modality
from search_engine.schemas.query import SearchFilters

RRF_C = 60  # the standard RRF constant: a hit at rank r scores 1 / (60 + r)
_PICTURES = {Modality.IMAGE.value, Modality.VIDEO_FRAME.value}


class Reranker:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        search: HybridSearch | None = None,
        cross_encoder: BaseCrossEncoder | None = None,
    ):
        self.settings = settings or get_settings()
        self.search = search or HybridSearch(self.settings)
        self._cross_encoder = cross_encoder

    @property
    def cross_encoder(self) -> BaseCrossEncoder:
        return self._cross_encoder or models.get_cross_encoder()

    def retrieve(self, query: str, filters: SearchFilters, top_k: int | None = None) -> list[Document]:
        """ELASTICSEARCH HYBRID SEARCH -> RERANKING: the best `top_k` chunks for the query, best first."""
        return self.rerank(query, self.fuse(query, filters), top_k)

    def fuse(self, query: str, filters: SearchFilters) -> list[Document]:
        """Rank fusion (RRF) of the hybrid search lists; a chunk found by both lists is merged, not repeated."""
        retrievers = list(self.search.retrievers(filters).values())
        if not retrievers:
            return []
        ensemble = EnsembleRetriever(
            retrievers=retrievers, weights=[1.0] * len(retrievers), c=RRF_C, id_key="chunk_id"
        )
        return ensemble.invoke(query)

    def rerank(self, query: str, fused: list[Document], top_k: int | None = None) -> list[Document]:
        """Cross encoder over the fused text hits, then the best `rerank_image_slots` image hits."""
        top_k = top_k or self.settings.top_k
        images = [doc for doc in fused if is_picture(doc)]
        texts = [doc for doc in fused if not is_picture(doc)]
        if not texts:  # only pictures matched (e.g. only charts were asked for): they fill the list
            return images[:top_k]
        # Otherwise images get at most `rerank_image_slots`, and never half the list.
        image_quota = min(len(images), self.settings.rerank_image_slots, top_k // 2)
        reranker = CrossEncoderReranker(model=self.cross_encoder, top_n=top_k - image_quota)
        return list(reranker.compress_documents(texts, query)) + images[:image_quota]


def is_picture(doc: Document) -> bool:
    return doc.metadata.get("modality") in _PICTURES
