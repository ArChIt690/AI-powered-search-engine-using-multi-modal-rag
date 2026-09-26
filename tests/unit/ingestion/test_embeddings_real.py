"""Real bge and CLIP models (run with `pytest -m slow`; the first run downloads them)."""

import io

import numpy as np
import pytest
from PIL import Image

from search_engine.core.config import Settings
from search_engine.ingestion.chunking import chunk_document
from search_engine.ingestion.image_embed import ImageEmbedder
from search_engine.ingestion.text_embed import TextEmbedder
from search_engine.schemas.chunk import Chunk
from search_engine.schemas.document import Document, Picture, Section

pytestmark = pytest.mark.slow


def _png(colour: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (224, 224), colour).save(buffer, format="png")
    return buffer.getvalue()


def test_bge_vectors_are_384_dim_unit_length_and_rank_relevant_text_first():
    embedder = TextEmbedder()
    chunks = embedder.embed_chunks(
        [
            Chunk(text="Goroutines are lightweight threads managed by the Go runtime.", source="/go.pdf"),
            Chunk(text="The recipe needs two cups of flour and a pinch of salt.", source="/cake.txt"),
        ]
    )
    query = np.asarray(embedder.embed_query("How does concurrency work in Go?"))

    vectors = np.asarray([c.text_embedding for c in chunks])
    assert vectors.shape == (2, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    scores = vectors @ query
    assert scores[0] > scores[1]


def test_clip_vectors_are_512_dim_and_match_text_to_the_right_image():
    embedder = ImageEmbedder()
    doc = Document(
        source="/pics/colours.pdf",
        file_type="pdf",
        pictures=[Picture(data=_png("red"), width=224, height=224, page=1), Picture(data=_png("blue"), width=224, height=224, page=2)],
    )

    chunks = embedder.embed_pictures(doc)
    images = np.asarray([c.image_embedding for c in chunks])
    texts = embedder.embed_texts(["a plain red square", "a plain blue square"])

    assert images.shape == (2, 512)
    assert np.allclose(np.linalg.norm(images, axis=1), 1.0, atol=1e-4)
    similarity = texts @ images.T  # rows: texts, columns: images
    assert similarity[0, 0] > similarity[0, 1]  # "red" text is closest to the red image
    assert similarity[1, 1] > similarity[1, 0]  # "blue" text is closest to the blue image


def test_semantic_chunking_with_real_bge_cuts_at_the_topic_change():
    football = (
        "The striker scored twice in the first half. The goalkeeper made a brilliant save from a penalty. "
        "Fans sang loudly as the match went into extra time. The coach praised the defence after the final whistle. "
        "The team now sits at the top of the league table. "
    )
    baking = (
        "Preheat the oven to 180 degrees before you start. Mix the flour, sugar and butter in a large bowl. "
        "Add the eggs one at a time and whisk until smooth. Pour the batter into a greased cake tin. "
        "Bake for forty minutes until a skewer comes out clean."
    )
    doc = Document(source="/docs/mixed.txt", file_type="txt", sections=[Section(text=football + baking)])

    chunks = chunk_document(doc, TextEmbedder(), Settings(chunk_size=800, semantic_min_chunk_size=100))

    assert len(chunks) == 2
    assert "striker" in chunks[0].text and "oven" not in chunks[0].text
    assert "oven" in chunks[1].text and "striker" not in chunks[1].text
