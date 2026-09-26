"""Model loading, in one place. Each model loads on first use and is then reused."""

from functools import lru_cache
from typing import TYPE_CHECKING

from search_engine.core.config import get_settings

if TYPE_CHECKING:
    from faster_whisper import WhisperModel
    from sentence_transformers import SentenceTransformer


@lru_cache
def get_text_model() -> "SentenceTransformer":
    """bge text embeddings: Chunking (semantic), Text Embeddings, and query embedding."""
    settings = get_settings()
    return _sentence_transformer(settings.text_embedding_model, settings.text_embedding_dim, "text_embedding_dim")


@lru_cache
def get_clip_model() -> "SentenceTransformer":
    """CLIP: Image Embeddings, and its text encoder for text-to-image search."""
    settings = get_settings()
    return _sentence_transformer(settings.clip_model, settings.image_embedding_dim, "image_embedding_dim")


def _sentence_transformer(name: str, expected_dim: int, setting: str) -> "SentenceTransformer":
    # Imported here: PyTorch takes several seconds to import, and the loaders don't need it.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name, device=_device("auto"))
    dim = model.get_embedding_dimension()
    if dim != expected_dim:
        # The Elasticsearch dense_vector fields are sized from config, so a mismatch would break indexing later.
        raise ValueError(f"{name} makes {dim}-dim vectors but {setting} is {expected_dim}; make them match")
    return model


@lru_cache
def get_whisper() -> "WhisperModel":
    # Imported here: loading CTranslate2 is slow, and only videos need it.
    from faster_whisper import WhisperModel

    settings = get_settings()
    return WhisperModel(
        settings.whisper_model,
        device=_device(settings.whisper_device),
        compute_type=settings.whisper_compute_type,
    )


def _device(device: str) -> str:
    """Resolve 'auto' to cuda only when CUDA really works.

    CTranslate2's own 'auto' picks cuda whenever an NVIDIA GPU is present, then crashes mid-transcription
    if the CUDA libraries (cuBLAS) are missing. PyTorch's check also covers the libraries.
    """
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
