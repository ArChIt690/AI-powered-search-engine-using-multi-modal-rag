"""Adding files and listing them, over HTTP. The routes only call the Ingestion pipeline."""

import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from search_engine.api.deps import get_ingestion
from search_engine.ingestion.pipeline import IngestionPipeline
from search_engine.ingestion.vector_store import file_counts
from search_engine.schemas.response import FileInfo, IngestResponse

router = APIRouter()


# Plain `def`s: ingestion is blocking (models, Elasticsearch), so FastAPI runs them in a thread.
@router.post("/ingest")
def ingest(
    files: list[UploadFile], pipeline: Annotated[IngestionPipeline, Depends(get_ingestion)]
) -> IngestResponse:
    """Saves each upload in the landing folder and ingests it. A file with the same name replaces the old one
    (same path -> same document id), so re-uploading never duplicates chunks."""
    landing = Path(pipeline.settings.landing_dir)
    landing.mkdir(parents=True, exist_ok=True)
    response = IngestResponse(files=0, chunks=0)
    for upload in files:
        # Only the bare name: a name like "../../x" must not write outside the landing folder.
        name = Path(upload.filename or "").name
        if not name or name.startswith("."):
            raise HTTPException(status_code=400, detail=f"Not a usable file name: {upload.filename!r}")
        target = landing / name
        with target.open("wb") as out:
            shutil.copyfileobj(upload.file, out)
        report = pipeline.ingest_path(target)
        response.files += report.files
        response.chunks += report.chunks
        response.skipped += [Path(path).name for path in report.skipped]
        response.failed.update({Path(path).name: error for path, error in report.failed.items()})
        if report.skipped or report.failed:
            target.unlink(missing_ok=True)  # nothing was ingested from it, so don't keep it
    return response


@router.get("/files")
def files(pipeline: Annotated[IngestionPipeline, Depends(get_ingestion)]) -> list[FileInfo]:
    try:
        counts = file_counts(pipeline.settings)
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Elasticsearch is not available: {str(error)[:200]}") from error
    return [FileInfo(file_name=name, **kinds) for name, kinds in counts.items()]
