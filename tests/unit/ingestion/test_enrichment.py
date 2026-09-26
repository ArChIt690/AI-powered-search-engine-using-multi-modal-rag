import pytest
from pydantic import ValidationError

from search_engine.ingestion.enrichment import document_id, enrich
from search_engine.schemas.chunk import Chunk, ChunkMetadata, Modality
from search_engine.schemas.document import Document


def _pdf_doc(source: str = "/docs/report.pdf", **metadata) -> Document:
    return Document(source=source, file_type="pdf", metadata=metadata)


def _pdf_chunks(source: str = "/docs/report.pdf") -> list[Chunk]:
    return [
        Chunk(text="Intro text.", source=source, page=1, metadata={"content": "text"}, text_embedding=[0.1, 0.2]),
        Chunk(text="|a|b|\n|---|---|", source=source, page=2, metadata={"content": "table"}),
        Chunk(
            text="chart from report.pdf, page 3",
            source=source,
            modality=Modality.IMAGE,
            page=3,
            metadata={"content": "chart"},
            image_embedding=[0.5],
        ),
    ]


def test_ids_are_unique_ordered_and_stable_across_runs():
    first = enrich(_pdf_doc(), _pdf_chunks())
    second = enrich(_pdf_doc(), _pdf_chunks())

    doc_id = document_id("/docs/report.pdf")
    assert [c.id for c in first] == [f"{doc_id}-0", f"{doc_id}-1", f"{doc_id}-2"]
    assert [c.id for c in second] == [c.id for c in first]  # re-ingesting gives the same ids
    assert [c.metadata["chunk_index"] for c in first] == [0, 1, 2]
    assert len(doc_id) == 16


def test_different_files_get_different_doc_ids():
    assert document_id("/docs/a.pdf") != document_id("/docs/b.pdf")


def test_pdf_file_metadata_is_copied_to_every_chunk():
    doc = _pdf_doc(title="Quarterly Report", author="Archit", created="2026-01-15", page_count=12)

    chunks = enrich(doc, _pdf_chunks())

    for chunk in chunks:
        meta = chunk.metadata
        assert (meta["title"], meta["author"], meta["created"], meta["page_count"]) == (
            "Quarterly Report",
            "Archit",
            "2026-01-15",
            12,
        )
        assert (meta["file_name"], meta["file_type"]) == ("report.pdf", "pdf")
        assert meta["ingested_at"].endswith("+00:00")
    assert [c.page for c in chunks] == [1, 2, 3]


def test_existing_fields_and_embeddings_are_kept():
    chunks = enrich(_pdf_doc(), _pdf_chunks())

    assert [c.metadata["content"] for c in chunks] == ["text", "table", "chart"]
    assert chunks[0].text_embedding == [0.1, 0.2]
    assert chunks[2].image_embedding == [0.5]
    assert chunks[2].modality == Modality.IMAGE


def test_markdown_section_kept_and_title_falls_back_to_file_name():
    doc = Document(source="/notes/setup-guide.md", file_type="md")
    chunk = Chunk(text="Install it.", source=doc.source, metadata={"content": "text", "section": "Setup > Usage"})

    [enriched] = enrich(doc, [chunk])

    assert enriched.metadata["section"] == "Setup > Usage"
    assert enriched.metadata["title"] == "setup-guide"
    assert "author" not in enriched.metadata  # unknown fields are left out, not stored as null


def test_video_chunks_get_timestamp_duration_and_language():
    doc = Document(
        source="/videos/talk.mp4",
        file_type="mp4",
        modality=Modality.VIDEO_TRANSCRIPT,
        metadata={"duration_s": 95.5, "language": "en", "created": "2026-02-01"},
    )
    chunks = [
        Chunk(text="Welcome.", source=doc.source, modality=Modality.VIDEO_TRANSCRIPT, timestamp=0.0, metadata={"content": "text"}),
        Chunk(text="video frame from talk.mp4 at 00:05", source=doc.source, modality=Modality.VIDEO_FRAME, timestamp=5.0, metadata={"content": "frame"}),
    ]

    enriched = enrich(doc, chunks)

    assert [(c.timestamp, c.metadata["content"]) for c in enriched] == [(0.0, "text"), (5.0, "frame")]
    assert all(c.metadata["duration_s"] == 95.5 and c.metadata["language"] == "en" for c in enriched)


def test_file_modified_set_when_the_file_exists(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")
    doc = Document(source=path.resolve().as_posix(), file_type="txt")

    [chunk] = enrich(doc, [Chunk(text="hello", source=doc.source, metadata={"content": "text"})])
    [missing] = enrich(_pdf_doc("/no/such/file.pdf"), [Chunk(text="x", source="/no/such/file.pdf", metadata={"content": "text"})])

    assert chunk.metadata["file_modified"].endswith("+00:00")
    assert "file_modified" not in missing.metadata


def test_metadata_round_trips_through_the_typed_model():
    for chunk in enrich(_pdf_doc(title="T", page_count=3), _pdf_chunks()):
        assert ChunkMetadata(**chunk.metadata).model_dump(exclude_none=True) == chunk.metadata


def test_unknown_metadata_field_fails_loudly():
    chunk = Chunk(text="x", source="/docs/report.pdf", metadata={"content": "text", "pgae": 3})

    with pytest.raises(ValidationError, match="pgae"):
        enrich(_pdf_doc(), [chunk])


def test_no_chunks_gives_no_chunks():
    assert enrich(_pdf_doc(), []) == []
