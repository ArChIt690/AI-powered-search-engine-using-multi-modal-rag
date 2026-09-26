"""Part 1 "Done when": one file of every input type -> `IngestionPipeline` -> enriched, embedded chunks in
Elasticsearch, each with the right modality.

Real bge and CLIP models and real Elasticsearch + Redis (docker compose up -d). Only Whisper is replaced: the test
video is silent, so a stand-in returns one transcript segment (real Whisper is checked in 1.1 on spoken audio).
Run with `uv run pytest -m "slow and integration"`.
"""

import csv
import io
import json
import uuid
from dataclasses import dataclass

import av
import numpy as np
import pymupdf
import pytest
from elasticsearch import Elasticsearch
from PIL import Image

from search_engine.core.config import Settings, get_settings
from search_engine.infra import models
from search_engine.infra.redis import get_redis
from search_engine.ingestion.pipeline import IngestionPipeline

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _es() -> Elasticsearch | None:
    client = Elasticsearch(get_settings().es_url, request_timeout=30)
    try:
        return client if client.ping() else None
    except Exception:
        return None


@dataclass
class _Segment:
    start: float
    text: str


class _SpeechStandIn:
    def transcribe(self, path, **kwargs):
        info = type("Info", (), {"language": "en"})()
        return iter([_Segment(0.0, "Welcome to the quarterly review. Revenue grew by twelve percent.")]), info


def _png(width: int, height: int, colour: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="png")
    return buffer.getvalue()


def _write_pdf(path) -> None:
    """Page 1: text + a ruled table + a photo. Page 2: text + a vector bar chart."""
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Quarterly results were strong across every region.")
    rows = [("Region", "Revenue"), ("North", "120"), ("South", "95")]
    for i in range(len(rows) + 1):
        page.draw_line((72, 120 + i * 24), (312, 120 + i * 24))
    for j in range(3):
        page.draw_line((72 + j * 120, 120), (72 + j * 120, 120 + len(rows) * 24))
    for i, cells in enumerate(rows):
        for j, cell in enumerate(cells):
            page.insert_text((78 + j * 120, 136 + i * 24), cell)
    page.insert_image(pymupdf.Rect(72, 300, 272, 450), stream=_png(200, 150, "orange"))

    page = pdf.new_page()
    page.insert_text((72, 72), "Revenue by region is shown in the chart below.")
    page.draw_line((72, 400), (272, 400))
    page.draw_line((72, 400), (72, 250))
    for i, height in enumerate((60, 120, 90, 40)):
        x = 92 + i * 45
        page.draw_rect(pymupdf.Rect(x, 400 - height, x + 30, 400), color=(0, 0, 1), fill=(0, 0, 1))
    pdf.set_metadata({"title": "Quarterly Report"})
    pdf.save(path)
    pdf.close()


def _write_video(path) -> None:
    """6 s silent video: 3 s red, 3 s blue; with a silent audio track so the transcript branch runs."""
    with av.open(str(path), "w") as out:
        video = out.add_stream("mpeg4", rate=10)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        audio = out.add_stream("aac", rate=16000, layout="mono")
        for colour in ((255, 0, 0), (0, 0, 255)):
            pixels = np.full((48, 64, 3), colour, dtype=np.uint8)
            for _ in range(30):
                out.mux(video.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")))
        out.mux(video.encode())
        for pts in range(0, 6 * 16000, 1024):
            frame = av.AudioFrame.from_ndarray(np.zeros((1, 1024), dtype=np.float32), format="fltp", layout="mono")
            frame.sample_rate, frame.pts = 16000, pts
            out.mux(audio.encode(frame))
        out.mux(audio.encode())


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "notes.txt").write_text("Goroutines are lightweight threads managed by the Go runtime.", encoding="utf-8")
    (tmp_path / "guide.md").write_text("# Setup\nInstall the tool.\n\n# Usage\nRun the search command.", encoding="utf-8")
    _write_pdf(tmp_path / "report.pdf")
    (tmp_path / "books.xml").write_text("<library><book id='1'><title>Dune</title></book></library>", encoding="utf-8")
    with (tmp_path / "people.csv").open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows([("name", "city"), ("Alice", "Paris"), ("Bob", "Rome")])
    (tmp_path / "items.json").write_text(json.dumps([{"id": 1, "name": "lamp"}, {"id": 2, "name": "desk"}]))
    (tmp_path / "photo.png").write_bytes(_png(120, 80, "green"))
    _write_video(tmp_path / "review.mp4")
    return tmp_path


@pytest.fixture
def settings(monkeypatch):
    if _es() is None:
        pytest.skip("Elasticsearch is not running (docker compose up -d)")
    monkeypatch.setattr(models, "get_whisper", lambda: _SpeechStandIn())
    suffix = uuid.uuid4().hex[:8]
    settings = Settings(
        es_text_index=f"e2e_text_{suffix}", es_image_index=f"e2e_images_{suffix}", index_version_key=f"e2e:version:{suffix}"
    )
    yield settings
    _es().indices.delete(index=f"{settings.es_text_index},{settings.es_image_index}", ignore_unavailable=True)
    get_redis().delete(settings.index_version_key)


def _counts(client: Elasticsearch, index: str, field: str) -> dict[str, int]:
    body = client.search(index=index, size=0, aggs={"k": {"terms": {"field": field, "size": 50}}})
    return {bucket["key"]: bucket["doc_count"] for bucket in body["aggregations"]["k"]["buckets"]}


def test_every_input_type_lands_in_elasticsearch_with_the_right_modality(corpus, settings):
    report = IngestionPipeline(settings).ingest_path(corpus)

    assert report.failed == {} and report.skipped == []
    assert report.files == 8

    client = _es()
    text_content = _counts(client, settings.es_text_index, "metadata.content")
    image_content = _counts(client, settings.es_image_index, "metadata.content")
    modalities = _counts(client, settings.es_text_index, "metadata.modality") | _counts(
        client, settings.es_image_index, "metadata.modality"
    )
    assert {"text", "table", "record"} <= set(text_content)  # prose, PDF table, CSV/JSON/XML records
    assert {"image", "chart", "frame"} <= set(image_content)  # photo + PDF image, PDF chart, video frames
    assert {"text", "image", "video_transcript", "video_frame"} <= set(modalities)
    assert sum(text_content.values()) + sum(image_content.values()) == report.chunks

    # every chunk is embedded and enriched
    for index, vector_field in ((settings.es_text_index, "text_embedding"), (settings.es_image_index, "image_embedding")):
        missing = client.count(index=index, query={"bool": {"must_not": [{"exists": {"field": vector_field}}]}})
        assert missing["count"] == 0
        unenriched = client.count(index=index, query={"bool": {"must_not": [{"exists": {"field": "metadata.doc_id"}}]}})
        assert unenriched["count"] == 0

    # citations: the PDF table keeps its page, the video frames and transcript keep their timestamps
    table = client.search(index=settings.es_text_index, query={"term": {"metadata.content": "table"}})["hits"]["hits"][0]
    assert table["_source"]["metadata"]["page"] == 1 and table["_source"]["metadata"]["title"] == "Quarterly Report"
    frames = client.search(index=settings.es_image_index, query={"term": {"metadata.content": "frame"}})["hits"]["hits"]
    assert sorted(hit["_source"]["metadata"]["timestamp"] for hit in frames) == [0.0, 5.0]
    assert int(get_redis().get(settings.index_version_key)) == 8  # one bump per ingested file
