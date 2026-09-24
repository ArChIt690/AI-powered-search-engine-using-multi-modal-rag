import json

import pytest
from defusedxml import EntitiesForbidden

from search_engine.core.exceptions import UnsupportedFileTypeError
from search_engine.ingestion.sources import load_file


def test_csv_one_record_per_row_and_skips_empty_cells(tmp_path):
    path = tmp_path / "people.csv"
    path.write_text("name,age,city\nAlice,30,Paris\nBob,,Rome\n", encoding="utf-8")

    doc = load_file(path)

    assert doc.structured
    assert [s.text for s in doc.sections] == ["name: Alice\nage: 30\ncity: Paris", "name: Bob\ncity: Rome"]


def test_csv_detects_semicolon_delimiter(tmp_path):
    path = tmp_path / "people.csv"
    path.write_text("name;city\nAlice;Paris\nBob;Rome\n", encoding="utf-8")

    doc = load_file(path)

    assert [s.text for s in doc.sections] == ["name: Alice\ncity: Paris", "name: Bob\ncity: Rome"]


def test_json_list_and_nested_objects_are_flattened(tmp_path):
    path = tmp_path / "items.json"
    path.write_text(json.dumps([{"id": 1, "tags": ["a", "b"], "owner": {"name": "Alice"}}, {"id": 2}]))

    doc = load_file(path)

    assert doc.structured
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

    assert doc.structured
    assert [s.text for s in doc.sections] == ["book@id: 1\nbook/title: Dune", "book@id: 2\nbook/title: Emma"]


def test_xml_rejects_entity_declarations(tmp_path):
    path = tmp_path / "bomb.xml"
    path.write_text('<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><x>&a;</x>', encoding="utf-8")

    with pytest.raises(EntitiesForbidden):
        load_file(path)


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "notes.docx"
    path.write_bytes(b"")

    with pytest.raises(UnsupportedFileTypeError, match="Unsupported"):
        load_file(path)
