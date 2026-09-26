import pytest
from langchain_core.embeddings import Embeddings

from search_engine.core.config import Settings
from search_engine.ingestion.chunking import chunk_document
from search_engine.schemas.chunk import Modality
from search_engine.schemas.document import Document, Section, SectionKind

PROSE = " ".join(f"This is sentence number {i} about the topic." for i in range(40))


def _settings(**overrides) -> Settings:
    return Settings(**{"chunking_strategy": "auto", "chunk_size": 800, "chunk_overlap": 100, **overrides})


def _prose_doc(*sections: Section) -> Document:
    return Document(source="/docs/a.txt", file_type="txt", sections=list(sections))


def _records_doc(records: list[str]) -> Document:
    return Document(
        source="/docs/p.csv", file_type="csv", structured=True, sections=[Section(text=r) for r in records]
    )


class TopicEmbeddings(Embeddings):
    """Fake embeddings: one dimension per topic word."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[t.lower().count("cat") + 0.0, t.lower().count("rocket") + 0.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _sentences(topic: str, count: int) -> str:
    return " ".join(f"The {topic} number {i} does its thing all day." for i in range(count))


# Smart prose


def test_smart_chunks_respect_size_and_overlap():
    settings = _settings(chunking_strategy="smart", chunk_size=200, chunk_overlap=60)

    chunks = chunk_document(_prose_doc(Section(text=PROSE)), settings=settings)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    # The last sentence of each chunk is repeated at the start of the next one.
    for previous, current in zip(chunks, chunks[1:]):
        last_sentence = previous.text.rsplit(". ", 1)[-1]
        assert current.text.startswith(last_sentence)


def test_chunks_never_cross_pages_or_headings():
    doc = _prose_doc(
        Section(text="Page one text.", page=1),
        Section(text="Page two text.", page=2),
        Section(text="Under a heading.", page=2, heading="Intro"),
    )

    chunks = chunk_document(doc, settings=_settings(chunking_strategy="smart"))

    assert [(c.page, c.metadata.get("section"), c.text) for c in chunks] == [
        (1, None, "Page one text."),
        (2, None, "Page two text."),
        (2, "Intro", "Under a heading."),
    ]
    assert all(c.metadata["content"] == "text" for c in chunks)


def test_prose_pieces_without_letters_are_dropped():
    doc = _prose_doc(Section(text="Real content here.", page=1), Section(text="12", page=2), Section(text="- 4 -", page=3))

    chunks = chunk_document(doc, settings=_settings(chunking_strategy="smart"))

    assert [c.text for c in chunks] == ["Real content here."]


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="overlap"):
        chunk_document(_prose_doc(Section(text=PROSE)), settings=_settings(chunk_size=100, chunk_overlap=100))


# Semantic prose


def test_semantic_splits_where_topic_changes():
    doc = _prose_doc(Section(text=f"{_sentences('cat', 5)} {_sentences('rocket', 5)}"))

    chunks = chunk_document(doc, TopicEmbeddings(), _settings(chunk_size=1000, semantic_min_chunk_size=0))

    assert len(chunks) == 2
    assert "cat" in chunks[0].text and "rocket" not in chunks[0].text
    assert "rocket" in chunks[1].text and "cat" not in chunks[1].text


def test_semantic_merges_a_short_final_chunk():
    doc = _prose_doc(Section(text=f"{_sentences('cat', 5)} {_sentences('rocket', 1)}"))

    assert len(chunk_document(doc, TopicEmbeddings(), _settings(chunk_size=1000, semantic_min_chunk_size=0))) == 2
    assert len(chunk_document(doc, TopicEmbeddings(), _settings(chunk_size=1000, semantic_min_chunk_size=100))) == 1


def test_oversized_semantic_chunk_is_split_again():
    doc = _prose_doc(Section(text=_sentences("cat", 20)))  # one topic: a single long semantic chunk

    chunks = chunk_document(doc, TopicEmbeddings(), _settings(chunk_size=200, chunk_overlap=20))

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


def test_auto_uses_semantic_for_prose_so_needs_embeddings():
    with pytest.raises(ValueError, match="embeddings"):
        chunk_document(_prose_doc(Section(text=PROSE)), settings=_settings())


# Records (CSV / JSON / XML)


def test_records_are_packed_whole_without_overlap_or_embeddings():
    records = [f"name: person {i}\ncity: Paris" for i in range(10)]

    chunks = chunk_document(_records_doc(records), settings=_settings(chunk_size=100, chunk_overlap=0))  # auto, no embeddings needed

    assert len(chunks) > 1
    packed = [record for chunk in chunks for record in chunk.text.split("\n\n")]
    assert packed == records  # every record exactly once, whole, in order
    assert all(c.metadata["content"] == "record" for c in chunks)


def test_large_file_gives_many_chunks_within_size():
    records = [f"name: person {i}\ncity: Paris\nage: {20 + i % 50}" for i in range(1000)]

    chunks = chunk_document(_records_doc(records), settings=_settings())

    assert len(chunks) > 30
    assert all(len(c.text) <= 800 for c in chunks)
    assert [record for chunk in chunks for record in chunk.text.split("\n\n")] == records


def test_oversized_record_repeats_its_first_line():
    fields = [f"field_{i}: " + " ".join(["value"] * 10) for i in range(40)]  # ~2,700 characters
    record = "\n".join(["id: 42", *fields])

    chunks = chunk_document(_records_doc([record]), settings=_settings())

    assert len(chunks) > 1
    assert all(c.text.startswith("id: 42\n") for c in chunks)
    assert all(len(c.text) <= 800 for c in chunks)
    # every field line appears whole, exactly once
    assert [line for c in chunks for line in c.text.split("\n")[1:]] == fields


def test_oversized_record_with_one_huge_field_is_split_within_limit():
    record = "id: 7\ndescription: " + "word " * 400

    chunks = chunk_document(_records_doc([record]), settings=_settings(chunk_size=300))

    assert len(chunks) > 1
    assert all(c.text.startswith("id: 7\n") and len(c.text) <= 300 for c in chunks)


# Tables


def _table(rows: int) -> str:
    lines = ["|Name|Score|City|", "|---|---|---|"]
    lines += [f"|Person {i}|{50 + i}|City number {i}|" for i in range(rows)]
    return "\n".join(lines)


def test_table_split_between_rows_with_header_repeated():
    table = _table(60)
    doc = _prose_doc(Section(text=table, page=3, kind=SectionKind.TABLE))

    chunks = chunk_document(doc, settings=_settings(chunk_size=300))  # tables never need embeddings

    assert len(chunks) > 1
    assert all(c.text.startswith("|Name|Score|City|\n|---|---|---|\n") for c in chunks)
    assert all(len(c.text) <= 300 for c in chunks)
    rows = [line for c in chunks for line in c.text.split("\n")[2:]]
    assert rows == table.split("\n")[2:]  # no row cut, lost or repeated
    assert all(c.page == 3 and c.metadata["content"] == "table" for c in chunks)


def test_small_table_is_one_chunk_and_never_mixed_with_page_text():
    doc = _prose_doc(
        Section(text="Some page text.", page=1),
        Section(text=_table(2), page=1, kind=SectionKind.TABLE),
    )

    chunks = chunk_document(doc, settings=_settings(chunking_strategy="smart"))

    assert [(c.metadata["content"], c.text) for c in chunks] == [("text", "Some page text."), ("table", _table(2))]


# Video transcripts


def test_transcript_sections_kept_one_to_one_with_timestamps():
    doc = Document(
        source="/videos/talk.mp4",
        file_type="mp4",
        modality=Modality.VIDEO_TRANSCRIPT,
        sections=[
            Section(text="Welcome to the review.", timestamp=0.0),
            Section(text="Revenue grew by twelve percent.", timestamp=31.5),
        ],
    )

    chunks = chunk_document(doc, settings=_settings())  # auto, no embeddings needed for transcripts

    assert [(c.text, c.timestamp) for c in chunks] == [
        ("Welcome to the review.", 0.0),
        ("Revenue grew by twelve percent.", 31.5),
    ]
    assert all(c.modality == Modality.VIDEO_TRANSCRIPT and c.source == "/videos/talk.mp4" for c in chunks)
