"""IMAGE EMBEDDINGS (CLIP): PICTURES (images, PDF charts, video frames) -> CLIP vectors (Chunk.image_embedding).

Pictures don't go through Chunking: each Picture becomes one Chunk here, as in the diagram
(PICTURES -> Image Embeddings -> Metadata Enrichment).
"""

import io
import logging
from pathlib import PurePosixPath

import numpy as np
from PIL import Image

from search_engine.core.config import Settings, get_settings
from search_engine.infra import models
from search_engine.schemas.chunk import Chunk
from search_engine.schemas.document import Document, Picture, PictureKind

logger = logging.getLogger(__name__)


class ImageEmbedder:
    """CLIP embeddings, loaded once (infra/models.py) on first use."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def embed_pictures(self, doc: Document) -> list[Chunk]:
        """Spark Streaming: Image Embeddings (CLIP). One chunk per picture; unreadable pictures are skipped."""
        chunks: list[Chunk] = []
        size = self.settings.embedding_batch_size
        # Decode one batch at a time: hundreds of video frames decoded at once would take gigabytes.
        for start in range(0, len(doc.pictures), size):
            batch = doc.pictures[start : start + size]
            decoded = [(picture, image) for picture in batch if (image := _decode(picture, doc.source)) is not None]
            if not decoded:
                continue
            vectors = self._encode([image for _, image in decoded])
            chunks.extend(
                _picture_chunk(doc, picture, vector) for (picture, _), vector in zip(decoded, vectors, strict=True)
            )
        return chunks

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """CLIP's text encoder: puts text in the same space as images, for text-to-image search (Part 2)."""
        return self._encode(texts)

    def _encode(self, items: list) -> np.ndarray:
        if not items:
            return np.empty((0, self.settings.image_embedding_dim), dtype=np.float32)
        return models.get_clip_model().encode(
            items,
            batch_size=self.settings.embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


def _decode(picture: Picture, source: str) -> Image.Image | None:
    try:
        with Image.open(io.BytesIO(picture.data)) as image:
            return image.convert("RGB")  # CLIP expects 3-channel RGB; this also drops transparency and palettes
    except Exception:
        logger.warning("Skipping unreadable %s (page=%s, t=%s) in %s", picture.kind, picture.page, picture.timestamp, source)
        return None


def _picture_chunk(doc: Document, picture: Picture, vector: np.ndarray) -> Chunk:
    return Chunk(
        text=_label(doc, picture),
        source=doc.source,
        modality=picture.modality,
        page=picture.page,
        timestamp=picture.timestamp,
        metadata={"content": picture.kind.value},
        image_embedding=vector.tolist(),
    )


def _label(doc: Document, picture: Picture) -> str:
    """A short text label, so keyword search can still find a picture by file name, page or time."""
    name = PurePosixPath(doc.source).name
    if picture.kind == PictureKind.FRAME:
        return f"video frame from {name} at {_clock(picture.timestamp or 0.0)}"
    if picture.page is not None:
        return f"{picture.kind.value} from {name}, page {picture.page}"
    return f"image {name}"


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"
