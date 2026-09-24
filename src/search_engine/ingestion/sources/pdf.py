"""PDFs: separating images, charts and tables.

- text   -> one section per page -> Chunking
- tables -> Markdown sections (kind="table") -> Chunking
- images and charts from the PDF -> Document.pictures -> PICTURES -> Image Embeddings (CLIP)
"""

import logging
import re
from pathlib import Path

import pymupdf

from search_engine.core.config import Settings, get_settings
from search_engine.schemas.document import Document, Picture, PictureKind, Section, SectionKind

logger = logging.getLogger(__name__)

_TEXT_BLOCK = 0  # PyMuPDF block type: 0 = text, 1 = image
_CHART_MARGIN_PT = 12  # padding around a chart so axis labels just outside its drawings are kept
_PAGE_FILL_RATIO = 0.9  # drawing clusters covering more of the page than this are borders/backgrounds
_TINY_MARK_PT = 12  # drawings smaller than this on both sides are bullets or dots


def load_pdf(path: Path) -> Document:
    settings = get_settings()
    sections: list[Section] = []
    pictures: list[Picture] = []
    seen_images: set[int] = set()  # an image reused on several pages (e.g. a logo) is kept once

    with pymupdf.open(path) as pdf:
        metadata = _metadata(pdf)
        for number, page in enumerate(pdf, start=1):
            tables = page.find_tables().tables
            table_rects = [pymupdf.Rect(table.bbox) for table in tables]

            text = _page_text(page, table_rects)
            if text:
                sections.append(Section(text=text, page=number))
            for table in tables:
                markdown = table.to_markdown().strip()
                if markdown:
                    sections.append(Section(text=markdown, page=number, kind=SectionKind.TABLE))

            pictures.extend(_images(pdf, page, number, seen_images, settings))
            pictures.extend(_charts(page, number, table_rects, settings))

    return Document(
        source=path.resolve().as_posix(),
        file_type="pdf",
        sections=sections,
        pictures=pictures,
        metadata=metadata,
    )


def _metadata(pdf: pymupdf.Document) -> dict:
    info = pdf.metadata or {}
    metadata = {key: value for key in ("title", "author") if (value := info.get(key))}
    if created := _pdf_date(info.get("creationDate")):
        metadata["created"] = created
    metadata["page_count"] = pdf.page_count
    return metadata


def _pdf_date(value: str | None) -> str | None:
    """PDF dates look like 'D:20240131120000+05'30''; keep the ISO date part."""
    match = re.match(r"D:(\d{4})(\d{2})(\d{2})", value or "")
    return "-".join(match.groups()) if match else None


def _page_text(page: pymupdf.Page, table_rects: list[pymupdf.Rect]) -> str:
    """Text blocks in reading order, leaving out text inside tables (it is kept as the table's Markdown)."""
    paragraphs = []
    for block in page.get_text("blocks", sort=True):
        if block[6] != _TEXT_BLOCK:
            continue
        rect = pymupdf.Rect(block[:4])
        center = (rect.tl + rect.br) / 2
        if any(center in table for table in table_rects):
            continue
        if paragraph := _clean(block[4]):
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def _clean(block_text: str) -> str:
    """A PDF block is one paragraph broken into visual lines: re-join the lines and undo hyphenation."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", block_text)
    return re.sub(r"\s+", " ", text).strip()


def _images(
    pdf: pymupdf.Document, page: pymupdf.Page, number: int, seen: set[int], settings: Settings
) -> list[Picture]:
    """Raster images on the page, as PNG."""
    pictures = []
    for xref, smask, width, height, *_ in page.get_images(full=True):
        if xref in seen or min(width, height) < settings.pdf_min_image_px:
            continue
        seen.add(xref)
        try:
            data = _image_png(pdf, xref, smask)
        except Exception:  # broken or exotic image streams should not fail the whole PDF
            logger.warning("Skipping unreadable image xref=%d on page %d of %s", xref, number, pdf.name)
            continue
        pictures.append(Picture(data=data, kind=PictureKind.IMAGE, page=number, width=width, height=height))
    return pictures


def _image_png(pdf: pymupdf.Document, xref: int, smask: int) -> bytes:
    pixmap = pymupdf.Pixmap(pdf, xref)
    if pixmap.colorspace and pixmap.colorspace.n not in (1, 3):  # CMYK and friends: PNG needs gray or RGB
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
    if smask:  # soft mask = the image's transparency, stored as a separate image
        pixmap = pymupdf.Pixmap(pixmap, pymupdf.Pixmap(pdf, smask))
    return pixmap.tobytes("png")


def _charts(
    page: pymupdf.Page, number: int, table_rects: list[pymupdf.Rect], settings: Settings
) -> list[Picture]:
    """Charts and diagrams drawn with vector paths: cluster nearby drawings and render each cluster to PNG."""
    drawings = page.get_drawings()
    if not drawings:
        return []

    page_area = page.rect.get_area()
    pictures = []
    for cluster in page.cluster_drawings(drawings=drawings):
        if cluster.width < settings.pdf_min_chart_pt or cluster.height < settings.pdf_min_chart_pt:
            continue
        if cluster.get_area() > _PAGE_FILL_RATIO * page_area:
            continue
        if any(cluster.intersects(table) for table in table_rects):  # table grid lines, not a chart
            continue
        paths = [drawing for drawing in drawings if _inside(drawing["rect"], cluster)]
        if len(paths) < settings.pdf_min_chart_paths:  # a single box, rule or frame
            continue
        if _is_shading(paths):  # e.g. a code block's grey background, one filled rectangle per line
            continue

        clip = (cluster + (-_CHART_MARGIN_PT, -_CHART_MARGIN_PT, _CHART_MARGIN_PT, _CHART_MARGIN_PT)) & page.rect
        pixmap = page.get_pixmap(clip=clip, dpi=settings.pdf_chart_dpi)
        pictures.append(
            Picture(
                data=pixmap.tobytes("png"),
                kind=PictureKind.CHART,
                page=number,
                width=pixmap.width,
                height=pixmap.height,
            )
        )
    return pictures


def _is_shading(paths: list[dict]) -> bool:
    """A shaded text box (code block, callout), not a chart.

    Ignoring tiny marks such as bullet dots, what is left is only unstroked filled rectangles in one colour.
    Charts have axes, gridlines or several colours.
    """
    shapes = [path for path in paths if max(path["rect"].width, path["rect"].height) >= _TINY_MARK_PT]
    only_filled_rects = all(
        path["type"] == "f" and all(item[0] == "re" for item in path["items"]) for path in shapes
    )
    return only_filled_rects and len({path.get("fill") for path in shapes}) <= 1


def _inside(inner: pymupdf.Rect, outer: pymupdf.Rect, tolerance: float = 1.0) -> bool:
    # Compares coordinates directly, because a straight line's rect has zero width or height,
    # and PyMuPDF treats such empty rects as never contained.
    return (
        inner.x0 >= outer.x0 - tolerance
        and inner.y0 >= outer.y0 - tolerance
        and inner.x1 <= outer.x1 + tolerance
        and inner.y1 <= outer.y1 + tolerance
    )
