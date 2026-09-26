import io
from pathlib import Path

import pytest
from PIL import Image

from search_engine.core.config import Settings
from search_engine.ingestion.pipeline import IngestionPipeline
from search_engine.schemas.chunk import Chunk, Modality


class _RecordingStore:
    def __init__(self):
        self.writes: list[list[Chunk]] = []

    def write(self, chunks: list[Chunk]) -> int:
        self.writes.append(chunks)
        return len(chunks)


@pytest.fixture
def pipeline(fake_models):
    store = _RecordingStore()
    return IngestionPipeline(Settings(), store=store), store


def _png(path: Path) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (80, 60), "red").save(buffer, format="png")
    path.write_bytes(buffer.getvalue())


def test_one_file_runs_every_box_and_is_written_enriched(pipeline, tmp_path):
    pipe, store = pipeline
    path = tmp_path / "notes.md"
    path.write_text("# Setup\nInstall the tool. Then run it.\n\n# Usage\nCall the command.", encoding="utf-8")

    written = pipe.ingest_file(path)

    [chunks] = store.writes
    assert written == len(chunks) == 2
    assert [c.metadata["section"] for c in chunks] == ["Setup", "Usage"]  # Chunking
    assert all(len(c.text_embedding) == 384 for c in chunks)  # Text Embeddings
    assert all(c.id and c.metadata["file_name"] == "notes.md" for c in chunks)  # Metadata Enrichment


def test_pictures_are_embedded_with_clip_after_the_text_chunks(pipeline, tmp_path):
    pipe, store = pipeline
    path = tmp_path / "photo.png"
    _png(path)

    pipe.ingest_file(path)

    [[chunk]] = store.writes
    assert chunk.modality == Modality.IMAGE
    assert len(chunk.image_embedding) == 512 and chunk.text_embedding is None
    assert chunk.metadata["content"] == "image"


def test_folder_ingest_reports_files_chunks_skipped_and_failed(pipeline, tmp_path):
    pipe, store = pipeline
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("Alpha text.", encoding="utf-8")
    (tmp_path / "sub" / "b.csv").write_text("name,city\nAlice,Paris\n", encoding="utf-8")
    (tmp_path / "notes.docx").write_bytes(b"")  # unsupported
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")  # fails
    (tmp_path / ".hidden.txt").write_text("ignored", encoding="utf-8")

    report = pipe.ingest_path(tmp_path)

    assert report.files == 2
    assert report.chunks == sum(len(w) for w in store.writes) == 2
    assert [Path(p).name for p in report.skipped] == ["notes.docx"]
    assert [Path(p).name for p in report.failed] == ["broken.json"]
    assert "JSONDecodeError" in next(iter(report.failed.values()))


def test_missing_path_raises(pipeline, tmp_path):
    pipe, _ = pipeline

    with pytest.raises(FileNotFoundError):
        pipe.ingest_path(tmp_path / "nope")
