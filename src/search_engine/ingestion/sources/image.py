"""PICTURES: standalone image files -> Image Embeddings (CLIP)."""

import io
import re
from pathlib import Path

from PIL import Image, ImageOps

from search_engine.schemas.chunk import Modality
from search_engine.schemas.document import Document, Picture, PictureKind

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"})

_EXIF_ORIENTATION = 0x0112
_EXIF_IFD = 0x8769
_EXIF_DATE_TAKEN = 0x9003  # DateTimeOriginal, inside the Exif IFD
_EXIF_DATE = 0x0132  # DateTime, in the main IFD


def load_image(path: Path) -> Document:
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        image.load()  # decode now, so a corrupt file fails here and not later in CLIP
        image_format = (image.format or path.suffix.lstrip(".")).lower()
        exif = image.getexif()
        created = _exif_date(exif)

        # Phone photos are often stored sideways with an EXIF "rotate me" tag; apply it, so CLIP sees them upright.
        if exif.get(_EXIF_ORIENTATION, 1) != 1:
            image = ImageOps.exif_transpose(image)
            data, image_format = _encode(image, image_format)
        width, height = image.size

    picture = Picture(data=data, image_format=image_format, kind=PictureKind.IMAGE, width=width, height=height)
    return Document(
        source=path.resolve().as_posix(),
        file_type=path.suffix.lower().lstrip("."),
        modality=Modality.IMAGE,
        pictures=[picture],
        metadata={"created": created} if created else {},
    )


def _encode(image: Image.Image, image_format: str) -> tuple[bytes, str]:
    """Re-encode in the original format when Pillow can write it, otherwise as PNG."""
    target = "jpeg" if image_format in ("jpeg", "mpo") else image_format
    if target not in ("jpeg", "png", "webp"):
        target = "png"
    if target == "jpeg" and image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    options = {"quality": 95} if target == "jpeg" else {}
    buffer = io.BytesIO()
    image.save(buffer, format=target, **options)
    return buffer.getvalue(), target


def _exif_date(exif: Image.Exif) -> str | None:
    """EXIF dates look like '2026:01:15 09:30:00'; keep the ISO date part."""
    value = exif.get_ifd(_EXIF_IFD).get(_EXIF_DATE_TAKEN) or exif.get(_EXIF_DATE)
    match = re.match(r"(\d{4}):(\d{2}):(\d{2})", str(value or ""))
    return "-".join(match.groups()) if match else None
