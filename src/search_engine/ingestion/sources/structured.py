"""XML, CSV, JSON: "for others, just text extraction".

Each record (CSV row, JSON object, XML child element) becomes one section written as "field: value" lines,
so the chunker can keep records whole.
"""

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element

from defusedxml import ElementTree

from search_engine.schemas.document import Document, Section

_CSV_DELIMITERS = ",;\t|"


def load_csv(path: Path) -> Document:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as file:
        sample = file.read(64 * 1024)
        file.seek(0)
        rows = list(csv.DictReader(file, dialect=_sniff_dialect(sample)))

    sections = []
    for row in rows:
        # A row with more cells than headers puts the extras under the key None.
        lines = [f"{key}: {value}" for key, value in row.items() if key is not None and value not in (None, "")]
        if lines:
            sections.append(Section(text="\n".join(lines)))
    return _structured(path, "csv", sections)


def load_json(path: Path) -> Document:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    sections = []
    for record in _json_records(data):
        text = "\n".join(_flatten(record))
        if text:
            sections.append(Section(text=text))
    return _structured(path, "json", sections)


def load_xml(path: Path) -> Document:
    # defusedxml blocks entity-expansion attacks (e.g. "billion laughs") in untrusted files.
    root = ElementTree.parse(path).getroot()
    records = list(root) or [root]
    sections = []
    for element in records:
        text = "\n".join(_xml_lines(element, _tag(element.tag)))
        if text:
            sections.append(Section(text=text))
    return _structured(path, "xml", sections)


def _structured(path: Path, file_type: str, sections: list[Section]) -> Document:
    return Document(source=path.resolve().as_posix(), file_type=file_type, structured=True, sections=sections)


def _sniff_dialect(sample: str) -> type[csv.Dialect]:
    """Detect ',', ';', tab or '|' separated files; fall back to plain CSV when unsure."""
    try:
        return csv.Sniffer().sniff(sample, delimiters=_CSV_DELIMITERS)
    except csv.Error:
        return csv.excel


def _json_records(data: Any) -> list[Any]:
    """Top-level list: one record per item. Top-level object: one record per key, expanding lists of objects."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        records = []
        for key, value in data.items():
            if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                records.extend({key: item} for item in value)
            else:
                records.append({key: value})
        return records
    return [data]


def _flatten(value: Any, prefix: str = "") -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _flatten(child, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            items = [str(item) for item in value if item not in (None, "")]
            if items:
                yield f"{prefix}: {', '.join(items)}" if prefix else ", ".join(items)
        else:
            for index, item in enumerate(value):
                yield from _flatten(item, f"{prefix}[{index}]")
    elif value not in (None, ""):
        yield f"{prefix}: {value}" if prefix else str(value)


def _xml_lines(element: Element, path: str) -> Iterator[str]:
    for name, value in element.attrib.items():
        yield f"{path}@{_tag(name)}: {value}"
    if element.text and element.text.strip():
        yield f"{path}: {element.text.strip()}"
    for child in element:
        yield from _xml_lines(child, f"{path}/{_tag(child.tag)}")
        if child.tail and child.tail.strip():  # mixed content: text after a child belongs to the parent
            yield f"{path}: {child.tail.strip()}"


def _tag(name: str) -> str:
    """Drop the namespace from "{http://ns}tag"."""
    return str(name).rsplit("}", 1)[-1]
