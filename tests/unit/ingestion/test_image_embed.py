import io

import numpy as np
import pytest
from PIL import Image

from search_engine.core.config import Settings
from search_engine.infra import models
from search_engine.ingestion.image_embed import ImageEmbedder
from search_engine.schemas.chunk import Modality
from search_engine.schemas.document import Document, Picture, PictureKind

DIM = 512


class _FakeClip:
    """Stands in for CLIP: vector[0] = image width (or text length), so outputs can be traced back to inputs."""

    def __init__(self):
        self.batches: list[int] = []

    def encode(self, items, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar):
        assert normalize_embeddings and convert_to_numpy
        self.batches.append(len(items))
        vectors = np.zeros((len(items), DIM), dtype=np.float32)
        vectors[:, 0] = [item.width if isinstance(item, Image.Image) else len(item) for item in items]
        return vectors


@pytest.fixture
def fake_clip(monkeypatch):
    clip = _FakeClip()
    monkeypatch.setattr(models, "get_clip_model", lambda: clip)
    return clip


def _png(width: int, mode: str = "RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, (width, 10)).save(buffer, format="png")
    return buffer.getvalue()


def _picture(width: int, **fields) -> Picture:
    return Picture(data=_png(width), width=width, height=10, **fields)


def test_pdf_pictures_become_chunks_with_page_and_label(fake_clip):
    doc = Document(
        source="/docs/report.pdf",
        file_type="pdf",
        pictures=[_picture(100, kind=PictureKind.IMAGE, page=1), _picture(200, kind=PictureKind.CHART, page=3)],
    )

    chunks = ImageEmbedder().embed_pictures(doc)

    assert [(c.text, c.page, c.modality, c.metadata["content"]) for c in chunks] == [
        ("image from report.pdf, page 1", 1, Modality.IMAGE, "image"),
        ("chart from report.pdf, page 3", 3, Modality.IMAGE, "chart"),
    ]
    assert [c.image_embedding[0] for c in chunks] == [100.0, 200.0]
    assert all(c.source == "/docs/report.pdf" and c.text_embedding is None for c in chunks)


def test_video_frames_get_timestamp_and_frame_modality(fake_clip):
    doc = Document(
        source="/videos/talk.mp4",
        file_type="mp4",
        modality=Modality.VIDEO_TRANSCRIPT,
        pictures=[_picture(64, kind=PictureKind.FRAME, timestamp=83.4), _picture(64, kind=PictureKind.FRAME, timestamp=3725.0)],
    )

    chunks = ImageEmbedder().embed_pictures(doc)

    assert [(c.text, c.timestamp, c.modality) for c in chunks] == [
        ("video frame from talk.mp4 at 01:23", 83.4, Modality.VIDEO_FRAME),
        ("video frame from talk.mp4 at 1:02:05", 3725.0, Modality.VIDEO_FRAME),
    ]


def test_standalone_image_label_and_transparency_handled(fake_clip):
    doc = Document(
        source="/pics/logo.png",
        file_type="png",
        modality=Modality.IMAGE,
        pictures=[Picture(data=_png(50, mode="RGBA"), width=50, height=10)],
    )

    [chunk] = ImageEmbedder().embed_pictures(doc)

    assert chunk.text == "image logo.png"
    assert chunk.image_embedding[0] == 50.0


def test_unreadable_picture_is_skipped_not_fatal(fake_clip, caplog):
    doc = Document(
        source="/docs/report.pdf",
        file_type="pdf",
        pictures=[Picture(data=b"not an image", width=1, height=1, page=2), _picture(120, page=4)],
    )

    chunks = ImageEmbedder().embed_pictures(doc)

    assert [c.page for c in chunks] == [4]
    assert "Skipping unreadable" in caplog.text


def test_pictures_decoded_and_embedded_in_batches(fake_clip):
    doc = Document(source="/v.mp4", file_type="mp4", pictures=[_picture(10 + i, kind=PictureKind.FRAME, timestamp=i) for i in range(5)])

    chunks = ImageEmbedder(Settings(embedding_batch_size=2)).embed_pictures(doc)

    assert fake_clip.batches == [2, 2, 1]
    assert [c.image_embedding[0] for c in chunks] == [10.0, 11.0, 12.0, 13.0, 14.0]


def test_clip_text_encoder_for_text_to_image_search(fake_clip):
    vectors = ImageEmbedder().embed_texts(["a red square", "cat"])

    assert vectors.shape == (2, DIM)
    assert list(vectors[:, 0]) == [12.0, 3.0]


def test_document_without_pictures_does_not_load_clip(monkeypatch):
    monkeypatch.setattr(models, "get_clip_model", lambda: pytest.fail("model should not load"))

    assert ImageEmbedder().embed_pictures(Document(source="/a.txt", file_type="txt")) == []
