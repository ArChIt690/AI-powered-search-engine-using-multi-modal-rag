import pymupdf
import pytest

from search_engine.ingestion.sources import load_file
from search_engine.schemas.document import PictureKind, SectionKind


def _png(width: int, height: int) -> bytes:
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
    pixmap.clear_with(180)
    return pixmap.tobytes("png")


def _draw_table(page: pymupdf.Page, top: float) -> None:
    """A ruled 3x2 table, which find_tables() detects from its grid lines."""
    rows = [("Name", "Score"), ("Alice", "91"), ("Bob", "78")]
    left, col, row = 72, 120, 24
    for i in range(len(rows) + 1):
        page.draw_line((left, top + i * row), (left + 2 * col, top + i * row))
    for j in range(3):
        page.draw_line((left + j * col, top), (left + j * col, top + len(rows) * row))
    for i, cells in enumerate(rows):
        for j, cell in enumerate(cells):
            page.insert_text((left + j * col + 6, top + i * row + 16), cell)


def _draw_bar_chart(page: pymupdf.Page, left: float, bottom: float) -> None:
    page.draw_line((left, bottom), (left + 200, bottom))  # x axis
    page.draw_line((left, bottom), (left, bottom - 150))  # y axis
    for i, height in enumerate((60, 120, 90, 40)):
        x = left + 20 + i * 45
        page.draw_rect(pymupdf.Rect(x, bottom - height, x + 30, bottom), color=(0, 0, 1), fill=(0, 0, 1))


@pytest.fixture
def report_pdf(tmp_path):
    path = tmp_path / "report.pdf"
    pdf = pymupdf.open()

    page1 = pdf.new_page()
    page1.insert_text((72, 72), "Quarterly results intro.")
    _draw_table(page1, top=120)
    page1.insert_image(pymupdf.Rect(72, 300, 272, 450), stream=_png(200, 150))

    page2 = pdf.new_page()
    page2.insert_text((72, 72), "Revenue by region.")
    _draw_bar_chart(page2, left=72, bottom=400)
    page2.insert_image(pymupdf.Rect(400, 72, 416, 88), stream=_png(16, 16))  # icon: too small to keep

    pdf.set_metadata({"title": "Quarterly Report", "author": "Archit", "creationDate": "D:20260115093000"})
    pdf.save(path)
    pdf.close()
    return path


def test_pdf_metadata(report_pdf):
    doc = load_file(report_pdf)

    assert doc.file_type == "pdf"
    assert doc.metadata == {"title": "Quarterly Report", "author": "Archit", "created": "2026-01-15", "page_count": 2}


def test_pdf_text_sections_per_page_without_table_text(report_pdf):
    doc = load_file(report_pdf)

    text_sections = [s for s in doc.sections if s.kind == SectionKind.TEXT]
    assert [(s.page, s.text) for s in text_sections] == [(1, "Quarterly results intro."), (2, "Revenue by region.")]


def test_pdf_tables_become_markdown_sections(report_pdf):
    doc = load_file(report_pdf)

    tables = [s for s in doc.sections if s.kind == SectionKind.TABLE]
    assert len(tables) == 1
    assert tables[0].page == 1
    compact = tables[0].text.replace(" ", "")
    assert "|Name|Score|" in compact
    assert "|Alice|91|" in compact


def test_pdf_images_go_to_pictures_and_small_icons_are_skipped(report_pdf):
    doc = load_file(report_pdf)

    images = [p for p in doc.pictures if p.kind == PictureKind.IMAGE]
    assert [(p.page, p.width, p.height) for p in images] == [(1, 200, 150)]
    assert images[0].data.startswith(b"\x89PNG")


def test_pdf_vector_charts_are_rendered_to_pictures(report_pdf):
    doc = load_file(report_pdf)

    charts = [p for p in doc.pictures if p.kind == PictureKind.CHART]
    assert [p.page for p in charts] == [2]  # the table's grid lines on page 1 are not a chart
    assert charts[0].data.startswith(b"\x89PNG")
    assert charts[0].width > 200 and charts[0].height > 150


def test_pdf_code_block_background_is_not_a_chart(tmp_path):
    path = tmp_path / "code.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    grey = (0.97, 0.97, 0.97)
    for line in range(10):  # how many PDF generators shade code blocks: one filled rectangle per line
        top = 100 + line * 14
        page.draw_rect(pymupdf.Rect(72, top, 540, top + 14), color=None, fill=grey)
        page.insert_text((80, top + 11), f"line {line} of code")
    pdf.save(path)
    pdf.close()

    doc = load_file(path)

    assert doc.pictures == []


def test_pdf_shaded_callout_with_bullets_is_not_a_chart(tmp_path):
    path = tmp_path / "callout.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.draw_rect(pymupdf.Rect(72, 100, 540, 260), color=None, fill=(0.97, 0.55, 0.66))
    for line in range(5):
        top = 120 + line * 25
        page.draw_circle((90, top), 2, color=None, fill=(0, 0, 0))  # bullet dot
        page.insert_text((100, top + 4), f"Key point number {line}")
    pdf.save(path)
    pdf.close()

    doc = load_file(path)

    assert doc.pictures == []


def test_pdf_image_reused_on_every_page_is_kept_once(tmp_path):
    path = tmp_path / "logo.pdf"
    pdf = pymupdf.open()
    logo = _png(100, 100)
    xref = 0
    for _ in range(3):
        page = pdf.new_page()
        xref = page.insert_image(pymupdf.Rect(72, 72, 172, 172), stream=logo, xref=xref)
    pdf.save(path)
    pdf.close()

    doc = load_file(path)

    assert [p.page for p in doc.pictures] == [1]
