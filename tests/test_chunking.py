import pytest
from langchain_core.embeddings import Embeddings

from Search_Engine.Data.chunking import chunk_document
from Search_Engine.Schema.document import Document, Section

PROSE = " ".join(f"This is sentence number {i} about the topic." for i in range(40))


def _prose_doc(*sections: Section) -> Document:
    return Document(source="/docs/a.txt", file_type="txt", sections=list(sections))


def test_smart_chunks_respect_size_and_overlap():
    chunks = chunk_document(_prose_doc(Section(text=PROSE)), strategy="smart", chunk_size=200, overlap=60)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    # The last sentence of each chunk is repeated at the start of the next one.
    for previous, current in zip(chunks, chunks[1:]):
        last_sentence = previous.text.rsplit(". ", 1)[-1]
        assert current.text.startswith(last_sentence)


def test_smart_chunks_never_cross_pages_or_headings():
    doc = _prose_doc(
        Section(text="Page one text.", page=1),
        Section(text="Page two text.", page=2),
        Section(text="Under a heading.", page=2, heading="Intro"),
    )

    chunks = chunk_document(doc, strategy="smart", chunk_size=500, overlap=50)

    assert [(c.page, c.metadata.get("section"), c.text) for c in chunks] == [
        (1, None, "Page one text."),
        (2, None, "Page two text."),
        (2, "Intro", "Under a heading."),
    ]


def test_records_are_packed_but_never_split():
    records = [Section(text=f"name: person {i}\ncity: Paris") for i in range(10)]
    doc = Document(source="/docs/p.csv", file_type="csv", structured=True, sections=records)

    chunks = chunk_document(doc, strategy="auto", chunk_size=100, overlap=0)

    assert len(chunks) > 1
    for chunk in chunks:
        for record in chunk.text.split("\n\n"):
            assert record.startswith("name: person") and record.endswith("city: Paris")


def test_oversized_record_is_hard_split_within_limit():
    doc = Document(
        source="/docs/big.json", file_type="json", structured=True, sections=[Section(text="word " * 100)]
    )

    chunks = chunk_document(doc, strategy="smart", chunk_size=120, overlap=0)

    assert len(chunks) > 1
    assert all(len(c.text) <= 120 for c in chunks)


class TopicEmbeddings(Embeddings):
    """Fake embeddings: one dimension per topic word."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[t.lower().count("cat") + 0.0, t.lower().count("rocket") + 0.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _sentences(topic: str, count: int) -> str:
    return " ".join(f"The {topic} number {i} does its thing all day." for i in range(count))


def test_semantic_splits_where_topic_changes():
    doc = _prose_doc(Section(text=f"{_sentences('cat', 5)} {_sentences('rocket', 5)}"))

    chunks = chunk_document(doc, strategy="semantic", chunk_size=1000, overlap=100, embeddings=TopicEmbeddings())

    assert len(chunks) == 2
    assert "cat" in chunks[0].text and "rocket" not in chunks[0].text
    assert "rocket" in chunks[1].text and "cat" not in chunks[1].text


def test_semantic_merges_a_short_final_chunk():
    doc = _prose_doc(Section(text=f"{_sentences('cat', 5)} {_sentences('rocket', 1)}"))
    kwargs = dict(strategy="semantic", chunk_size=1000, overlap=100, embeddings=TopicEmbeddings())

    assert len(chunk_document(doc, **kwargs)) == 2
    assert len(chunk_document(doc, **kwargs, min_chunk_size=100)) == 1


def test_oversized_semantic_chunk_is_split_again():
    doc = _prose_doc(Section(text=_sentences("cat", 20)))  # one topic: a single long semantic chunk

    chunks = chunk_document(doc, strategy="semantic", chunk_size=200, overlap=20, embeddings=TopicEmbeddings())

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


def test_auto_uses_semantic_for_prose_so_needs_embeddings():
    with pytest.raises(ValueError, match="embeddings"):
        chunk_document(_prose_doc(Section(text=PROSE)), strategy="auto", chunk_size=200, overlap=20)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="overlap"):
        chunk_document(_prose_doc(Section(text=PROSE)), strategy="smart", chunk_size=100, overlap=100)
