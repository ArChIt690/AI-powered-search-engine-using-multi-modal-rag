""" "all the eval results are stored separately": one JSON line per answer in `data/eval_results/<date>.jsonl`.

For watching answer quality over time and tuning retrieval. These files are never ingested into the Vector
Database (Ingestion skips this folder), so an LLM answer can't come back later as evidence for another answer.
"""

import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

from search_engine.core.config import Settings, get_settings
from search_engine.schemas.eval import EvalRecord

logger = logging.getLogger(__name__)


class EvalStore:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.directory = Path(self.settings.eval_results_dir)
        self._lock = threading.Lock()

    def save(self, record: EvalRecord) -> Path | None:
        """Appends the record to today's file. A failure is logged and never fails the search."""
        path = self.directory / f"{datetime.now(UTC):%Y-%m-%d}.jsonl"
        try:
            with self._lock:
                self.directory.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as file:
                    file.write(record.model_dump_json() + "\n")
        except OSError as error:
            logger.warning("Could not store the eval result in %s: %s", path, error)
            return None
        return path

    def read(self, day: str | None = None) -> list[EvalRecord]:
        """The records of one day (YYYY-MM-DD, default today), oldest first."""
        path = self.directory / f"{day or f'{datetime.now(UTC):%Y-%m-%d}'}.jsonl"
        if not path.exists():
            return []
        return [EvalRecord.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
