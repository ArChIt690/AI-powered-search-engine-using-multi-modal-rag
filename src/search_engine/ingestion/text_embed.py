"""TEXT EMBEDDINGS: text chunks -> bge vectors (Chunk.text_embedding)."""

import numpy as np
from langchain_core.embeddings import Embeddings

from search_engine.core.config import Settings, get_settings
from search_engine.infra import models
from search_engine.schemas.chunk import Chunk


class TextEmbedder(Embeddings):
    """bge embeddings, loaded once (infra/models.py) on first use.

    Also a LangChain `Embeddings`, so semantic Chunking and query embedding share this model
    instead of loading a second copy.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Unit-length vectors, one row per text, so cosine similarity is a dot product."""
        if not texts:
            return np.empty((0, self.settings.text_embedding_dim), dtype=np.float32)
        return models.get_text_model().encode(
            texts,
            batch_size=self.settings.embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Spark Streaming: Text Embeddings. Fills text_embedding on each chunk, in order."""
        vectors = self.embed([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.text_embedding = vector.tolist()
        return chunks

    # LangChain Embeddings interface

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        # bge is trained with this prefix on search queries only; documents are embedded without it.
        return self.embed([self.settings.text_query_instruction + text])[0].tolist()
