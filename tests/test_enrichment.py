from Search_Engine.Data.enrichment import enrich_chunks
from Search_Engine.Schema.chunk import Chunk
from Search_Engine.Schema.document import Document


def test_enrichment_adds_ids_and_filterable_metadata(tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"")
    doc = Document(source=path.as_posix(), file_type="pdf", metadata={"title": "Report"})
    chunks = [Chunk(text="one two three", source=doc.source), Chunk(text="four", source=doc.source)]

    enrich_chunks(chunks, doc)

    first = chunks[0]
    assert first.id.endswith("-00000") and chunks[1].id.endswith("-00001")
    assert first.metadata["file_name"] == "report.pdf"
    assert first.metadata["file_type"] == "pdf"
    assert first.metadata["title"] == "Report"
    assert first.metadata["chunk_count"] == 2
    assert first.metadata["word_count"] == 3
    assert first.metadata["file_modified"] is not None


def test_ids_are_stable_across_runs(tmp_path):
    doc = Document(source=(tmp_path / "a.txt").as_posix(), file_type="txt")

    first = enrich_chunks([Chunk(text="x", source=doc.source)], doc)[0].id
    second = enrich_chunks([Chunk(text="x", source=doc.source)], doc)[0].id

    assert first == second
