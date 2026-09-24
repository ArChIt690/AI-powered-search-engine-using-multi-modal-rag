import io

import pytest
from PIL import Image, UnidentifiedImageError

from search_engine.ingestion.sources import load_file
from search_engine.schemas.chunk import Modality
from search_engine.schemas.document import PictureKind


def test_png_becomes_one_picture_with_original_bytes(tmp_path):
    path = tmp_path / "diagram.png"
    Image.new("RGB", (120, 80), "red").save(path)

    doc = load_file(path)

    assert doc.file_type == "png"
    assert doc.modality == Modality.IMAGE
    assert doc.sections == []
    [picture] = doc.pictures
    assert (picture.kind, picture.image_format, picture.width, picture.height) == (PictureKind.IMAGE, "png", 120, 80)
    assert picture.data == path.read_bytes()
    assert picture.modality == Modality.IMAGE


def test_jpeg_exif_rotation_is_applied_and_date_taken_kept(tmp_path):
    path = tmp_path / "photo.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90 degrees to display upright
    exif.get_ifd(0x8769)[0x9003] = "2026:01:15 09:30:00"  # DateTimeOriginal
    Image.new("RGB", (200, 100), "blue").save(path, exif=exif)

    doc = load_file(path)

    [picture] = doc.pictures
    assert (picture.width, picture.height) == (100, 200)  # stored sideways, delivered upright
    assert picture.image_format == "jpeg"
    assert Image.open(io.BytesIO(picture.data)).size == (100, 200)
    assert doc.metadata == {"created": "2026-01-15"}


def test_corrupt_image_raises(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not really a jpeg")

    with pytest.raises(UnidentifiedImageError):
        load_file(path)
