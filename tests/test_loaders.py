import json

import pymupdf
import pytest

from Search_Engine.Data.loaders import load_file


def test_markdown_splits_on_headings(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("Intro line.\n\n# Setup\nInstall it.\n\n## Usage\nRun it.\n", encoding="utf-8")

    doc = load_file(path)

    assert doc.file_type == "md"
    assert [s.heading for s in doc.sections] == [None, "Setup", "Usage"]
    assert doc.sections[1].text == "Setup\n\nInstall it."


def test_txt_is_one_section(tmp_path):
    path = tmp_path / "plain.txt"
    path.write_text("Hello there.\n\nSecond paragraph.", encoding="utf-8")

    doc = load_file(path)

    assert len(doc.sections) == 1
    assert not doc.structured


def test_pdf_one_section_per_page_with_metadata(tmp_path):
    path = tmp_path / "report.pdf"
    pdf = pymupdf.open()
    for text in ("First page text.", "Second page text."):
        pdf.new_page().insert_text((72, 72), text)
    pdf.set_metadata({"title": "Quarterly Report", "author": "Archit"})
    pdf.save(path)
    pdf.close()

    doc = load_file(path)

    assert [s.page for s in doc.sections] == [1, 2]
    assert doc.sections[1].text == "Second page text."
    assert doc.metadata == {"title": "Quarterly Report", "author": "Archit", "page_count": 2}


def test_csv_one_record_per_row_and_skips_empty_cells(tmp_path):
    path = tmp_path / "people.csv"
    path.write_text("name,age,city\nAlice,30,Paris\nBob,,Rome\n", encoding="utf-8")

    doc = load_file(path)

    assert doc.structured
    assert [s.text for s in doc.sections] == ["name: Alice\nage: 30\ncity: Paris", "name: Bob\ncity: Rome"]


def test_json_list_and_nested_objects_are_flattened(tmp_path):
    path = tmp_path / "items.json"
    path.write_text(json.dumps([{"id": 1, "tags": ["a", "b"], "owner": {"name": "Alice"}}, {"id": 2}]))

    doc = load_file(path)

    assert [s.text for s in doc.sections] == ["id: 1\ntags: a, b\nowner.name: Alice", "id: 2"]


def test_json_object_expands_lists_of_records(tmp_path):
    path = tmp_path / "wrapped.json"
    path.write_text(json.dumps({"version": 2, "data": [{"name": "x"}, {"name": "y"}]}))

    doc = load_file(path)

    assert [s.text for s in doc.sections] == ["version: 2", "data.name: x", "data.name: y"]


def test_xml_one_record_per_child_with_attributes(tmp_path):
    path = tmp_path / "library.xml"
    path.write_text(
        '<library xmlns="http://example.com"><book id="1"><title>Dune</title></book>'
        '<book id="2"><title>Emma</title></book></library>',
        encoding="utf-8",
    )

    doc = load_file(path)

    assert [s.text for s in doc.sections] == ["book@id: 1\nbook/title: Dune", "book@id: 2\nbook/title: Emma"]


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="Unsupported"):
        load_file(path)
