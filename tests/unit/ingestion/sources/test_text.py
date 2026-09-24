from search_engine.ingestion.sources import load_file


def test_markdown_splits_on_headings_with_heading_path(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("Intro line.\n\n# Setup\nInstall it.\n\nSecond para.\n\n## Usage\nRun it.\n", encoding="utf-8")

    doc = load_file(path)

    assert doc.file_type == "md"
    assert [s.heading for s in doc.sections] == [None, "Setup", "Setup > Usage"]
    assert doc.sections[1].text == "# Setup\nInstall it.\n\nSecond para."  # paragraph break kept


def test_markdown_ignores_hash_lines_inside_code_blocks(tmp_path):
    path = tmp_path / "code.md"
    path.write_text("# Install\n```bash\n# not a heading\npip install x\n```\n", encoding="utf-8")

    doc = load_file(path)

    assert {s.heading for s in doc.sections} == {"Install"}


def test_txt_is_one_section(tmp_path):
    path = tmp_path / "plain.txt"
    path.write_text("Hello there.\n\nSecond paragraph.", encoding="utf-8")

    doc = load_file(path)

    assert [s.text for s in doc.sections] == ["Hello there.\n\nSecond paragraph."]
    assert not doc.structured
    assert doc.source == path.resolve().as_posix()


def test_txt_drops_byte_order_mark(tmp_path):
    path = tmp_path / "bom.txt"
    path.write_text("Hello.", encoding="utf-8-sig")

    assert load_file(path).sections[0].text == "Hello."


def test_empty_txt_has_no_sections(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("  \n", encoding="utf-8")

    assert load_file(path).sections == []
