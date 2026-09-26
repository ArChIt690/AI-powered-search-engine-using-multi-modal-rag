"""VIDEO: splits two ways, as in the diagram.

- Extract Audio & Convert to Text (Whisper) -> timestamped transcript sections -> Text Extraction -> Chunking
- Takes pictures frame by frame, temporarily -> timestamped frames -> PICTURES -> Image Embeddings (CLIP)

Frames are kept in memory only (Picture.data) and never written to disk, so nothing is left behind once they
have been embedded.
"""

import io
import re
from pathlib import Path

import av
import numpy as np
from PIL import Image

from search_engine.core.config import Settings, get_settings
from search_engine.infra import models
from search_engine.schemas.chunk import Modality
from search_engine.schemas.document import Document, Picture, PictureKind, Section

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"})

_CHANGE_THUMB_PX = 32  # frames are compared as tiny grayscale thumbnails, which ignores noise and compression


def load_video(path: Path) -> Document:
    settings = get_settings()
    with av.open(str(path)) as container:
        metadata = _metadata(container)
        has_audio = bool(container.streams.audio)

    sections: list[Section] = []
    if has_audio:
        sections, language = extract_audio_to_text(path, settings)
        if language:
            metadata["language"] = language

    return Document(
        source=path.resolve().as_posix(),
        file_type=path.suffix.lower().lstrip("."),
        modality=Modality.VIDEO_TRANSCRIPT,
        sections=sections,
        pictures=sample_frames(path, settings),
        metadata=metadata,
    )


def extract_audio_to_text(path: Path, settings: Settings | None = None) -> tuple[list[Section], str | None]:
    """Extract Audio & Convert to Text.

    Whisper returns short segments (a sentence or so). They are grouped into sections of up to chunk_size
    characters, so each section usually becomes one chunk and keeps an accurate start time.
    Returns the sections and the detected language.
    """
    settings = settings or get_settings()
    # vad_filter skips silence, where Whisper tends to invent text.
    segments, info = models.get_whisper().transcribe(str(path), vad_filter=True)

    sections: list[Section] = []
    texts: list[str] = []
    start = 0.0
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if texts and len(" ".join([*texts, text])) > settings.chunk_size:
            sections.append(Section(text=" ".join(texts), timestamp=round(start, 2)))
            texts = []
        if not texts:
            start = segment.start
        texts.append(text)
    if texts:
        sections.append(Section(text=" ".join(texts), timestamp=round(start, 2)))
    return sections, info.language


def sample_frames(path: Path, settings: Settings | None = None) -> list[Picture]:
    """Takes pictures frame by frame, temporarily.

    Keeps at most one frame per video_frame_interval_s seconds, and skips a frame when it looks almost the same
    as the last one kept (a static slide or a talking head), so long unchanged shots give one frame, not dozens.
    """
    settings = settings or get_settings()
    pictures: list[Picture] = []
    last_time: float | None = None
    last_thumb: np.ndarray | None = None

    with av.open(str(path)) as container:
        if not container.streams.video:
            return []
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"  # decode on several threads

        for frame in container.decode(stream):
            if frame.time is None or (last_time is not None and frame.time - last_time < settings.video_frame_interval_s):
                continue
            image = frame.to_image()
            thumb = _thumbnail(image)
            if last_thumb is not None and float(np.abs(thumb - last_thumb).mean()) < settings.video_frame_min_change:
                continue

            last_time, last_thumb = frame.time, thumb
            image.thumbnail((settings.video_frame_max_side_px, settings.video_frame_max_side_px))
            buffer = io.BytesIO()
            image.save(buffer, format="jpeg", quality=90)
            pictures.append(
                Picture(
                    data=buffer.getvalue(),
                    image_format="jpeg",
                    kind=PictureKind.FRAME,
                    timestamp=round(frame.time, 2),
                    width=image.width,
                    height=image.height,
                )
            )
    return pictures


def _thumbnail(image: Image.Image) -> np.ndarray:
    small = image.convert("L").resize((_CHANGE_THUMB_PX, _CHANGE_THUMB_PX))
    return np.asarray(small, dtype=np.float32) / 255.0


def _metadata(container: av.container.InputContainer) -> dict:
    metadata: dict = {}
    if container.duration is not None:  # in microseconds (av.time_base)
        metadata["duration_s"] = round(container.duration / av.time_base, 2)
    # Cameras and phones write e.g. '2026-01-15T09:30:00.000000Z'; keep the ISO date part.
    match = re.match(r"(\d{4}-\d{2}-\d{2})", container.metadata.get("creation_time", ""))
    if match:
        metadata["created"] = match.group(1)
    return metadata
