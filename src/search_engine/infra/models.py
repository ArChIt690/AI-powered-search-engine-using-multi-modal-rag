"""Model loading, in one place. Each model loads on first use and is then reused."""

from functools import lru_cache
from typing import TYPE_CHECKING

from search_engine.core.config import get_settings

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


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
