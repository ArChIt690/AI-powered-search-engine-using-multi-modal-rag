"""TEXT EMBEDDINGS."""

import numpy as np
from sentence_transformers import SentenceTransformer

from Search_Engine.Schema.chunk import Chunk


class TextEmbedder:
    """Sentence-transformers model, loaded on first use (loading takes a few seconds)."""

    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return self.model.get_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Unit-length vectors, one row per text, so cosine similarity is a dot product."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        vectors = self.embed([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.text_embedding = vector.tolist()
        return chunks
