import hashlib

import numpy as np
import pytest
from PIL import Image

from search_engine.infra import models


class FakeSentenceModel:
    """Stands in for bge / CLIP: a deterministic unit vector per input, so no model is downloaded or loaded."""

    def __init__(self, dim: int):
        self.dim = dim

    def encode(self, items, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar):
        vectors = np.stack([self._vector(item) for item in items]) if items else np.empty((0, self.dim))
        return vectors.astype(np.float32)

    def _vector(self, item) -> np.ndarray:
        key = item.tobytes() if isinstance(item, Image.Image) else str(item).encode()
        seed = int.from_bytes(hashlib.sha1(key).digest()[:4], "big")
        vector = np.random.default_rng(seed).normal(size=self.dim)
        return vector / np.linalg.norm(vector)


@pytest.fixture
def fake_models(monkeypatch):
    """Replace bge (384-dim) and CLIP (512-dim) with fast fakes."""
    text, clip = FakeSentenceModel(384), FakeSentenceModel(512)
    monkeypatch.setattr(models, "get_text_model", lambda: text)
    monkeypatch.setattr(models, "get_clip_model", lambda: clip)
    return text, clip
