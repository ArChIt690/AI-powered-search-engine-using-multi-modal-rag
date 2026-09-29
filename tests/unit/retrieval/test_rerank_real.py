"""The real cross encoder (run with `pytest -m slow`; the first run downloads BAAI/bge-reranker-base)."""

import pytest
from langchain_core.documents import Document

from search_engine.core.config import Settings
from search_engine.retrieval.rerank import Reranker

pytestmark = pytest.mark.slow


def test_cross_encoder_puts_the_answering_chunk_first_even_when_fusion_ranked_it_last():
    fused = [
        Document("Bake the cake at 180 degrees for forty minutes.", metadata={"chunk_id": "cake", "modality": "text"}),
        Document("The Go runtime schedules goroutines onto OS threads.", metadata={"chunk_id": "sched", "modality": "text"}),
        Document("Channels let goroutines send values to each other safely.", metadata={"chunk_id": "chan", "modality": "text"}),
    ]

    result = Reranker(Settings(top_k=3)).rerank("how do goroutines talk to each other", fused)

    assert result[0].metadata["chunk_id"] == "chan"
    assert result[-1].metadata["chunk_id"] == "cake"
