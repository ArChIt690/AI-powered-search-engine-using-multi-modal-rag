from dataclasses import dataclass

import av
import numpy as np
import pytest

from search_engine.infra import models
from search_engine.ingestion.sources import load_file
from search_engine.schemas.chunk import Modality
from search_engine.schemas.document import PictureKind

FPS = 10


def _write_video(path, scenes, with_audio=False):
    """scenes: list of (seconds, RGB colour). Writes a small MPEG-4 video, optionally with a silent AAC track."""
    with av.open(str(path), "w") as out:
        video = out.add_stream("mpeg4", rate=FPS)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        audio = out.add_stream("aac", rate=16000, layout="mono") if with_audio else None

        for seconds, colour in scenes:
            pixels = np.full((48, 64, 3), colour, dtype=np.uint8)
            for _ in range(int(seconds * FPS)):
                out.mux(video.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")))
        out.mux(video.encode())

        if audio is not None:
            total = int(sum(seconds for seconds, _ in scenes) * 16000)
            for pts in range(0, total, 1024):
                frame = av.AudioFrame.from_ndarray(np.zeros((1, 1024), dtype=np.float32), format="fltp", layout="mono")
                frame.sample_rate, frame.pts = 16000, pts
                out.mux(audio.encode(frame))
            out.mux(audio.encode())


@dataclass
class _Segment:
    start: float
    text: str


class _FakeWhisper:
    def __init__(self, segments, language="en"):
        self.segments, self.language, self.calls = segments, language, 0

    def transcribe(self, path, **kwargs):
        self.calls += 1
        return iter(self.segments), type("Info", (), {"language": self.language})()


@pytest.fixture
def fake_whisper(monkeypatch):
    def install(segments, language="en"):
        whisper = _FakeWhisper(segments, language)
        monkeypatch.setattr(models, "get_whisper", lambda: whisper)
        return whisper

    return install


def test_frames_sampled_per_interval_with_timestamps(tmp_path, fake_whisper):
    path = tmp_path / "clip.mp4"
    _write_video(path, [(3, (255, 0, 0)), (3, (0, 0, 255)), (6, (0, 255, 0))])  # 12 s: red, blue, green
    fake_whisper([])

    doc = load_file(path)

    frames = doc.pictures
    assert [p.timestamp for p in frames] == [0.0, 5.0, 10.0]  # at most one frame per 5 s
    assert all(p.kind == PictureKind.FRAME and p.modality == Modality.VIDEO_FRAME for p in frames)
    assert all(p.image_format == "jpeg" and p.data.startswith(b"\xff\xd8") for p in frames)
    assert (frames[0].width, frames[0].height) == (64, 48)
    assert doc.file_type == "mp4"
    assert doc.metadata["duration_s"] == pytest.approx(12, abs=0.2)


def test_unchanged_shot_gives_one_frame(tmp_path, fake_whisper):
    path = tmp_path / "static.mp4"
    _write_video(path, [(20, (90, 90, 90))])  # 20 s of the same grey slide
    fake_whisper([])

    doc = load_file(path)

    assert [p.timestamp for p in doc.pictures] == [0.0]


def test_video_without_audio_skips_whisper(tmp_path, fake_whisper):
    path = tmp_path / "silent.mp4"
    _write_video(path, [(2, (255, 0, 0))])
    whisper = fake_whisper([_Segment(0.0, "should not appear")])

    doc = load_file(path)

    assert whisper.calls == 0
    assert doc.sections == []
    assert "language" not in doc.metadata


def test_transcript_segments_grouped_into_timestamped_sections(tmp_path, fake_whisper):
    path = tmp_path / "talk.mp4"
    _write_video(path, [(2, (255, 0, 0))], with_audio=True)
    sentence = "word " * 59 + "end."  # 300 characters
    whisper = fake_whisper(
        [_Segment(0.0, sentence), _Segment(12.5, sentence), _Segment(30.0, sentence), _Segment(41.0, "  ")],
        language="hi",
    )

    doc = load_file(path)

    assert whisper.calls == 1
    assert doc.modality == Modality.VIDEO_TRANSCRIPT
    # chunk_size is 800: two 300-character segments fit in one section, the third starts a new one.
    assert [s.timestamp for s in doc.sections] == [0.0, 30.0]
    assert doc.sections[0].text == f"{sentence} {sentence}"
    assert doc.sections[1].text == sentence
    assert doc.metadata["language"] == "hi"
