"""Dev retrieval benchmark (not the diagram's EVAL box): recall@k and MRR of each Retrieval stage.

    uv run python -m eval.run                          # retrieval only: no LLM tokens
    uv run python -m eval.run --enhance                # also Query Enhancement (2 LLM calls/question budget!)
    uv run python -m eval.run --rerank-model cross-encoder/ms-marco-MiniLM-L6-v2

It ingests `eval/corpus/` (plus a generated PDF with a table and a chart, and two images) into its own indexes,
asks every question in `eval/questions.jsonl`, and scores three stages: the text hybrid search alone, after rank
fusion with the image search, and after the cross encoder. Its indexes are deleted at the end.
"""

import argparse
import io
import json
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
KS = (1, 5, 10)


@dataclass
class Question:
    question: str
    file: str
    kind: str
    contains: str | None = None
    content: str | None = None

    def is_answered_by(self, doc) -> bool:
        m = doc.metadata
        if m.get("file_name") != self.file:
            return False
        if self.content:
            return m.get("content") == self.content
        return self.contains.lower() in doc.page_content.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--enhance", action="store_true", help="run QUERY Enhancement first (uses the LLM)")
    parser.add_argument("--rerank-model", help="cross encoder to benchmark instead of RERANK_MODEL")
    args = parser.parse_args()

    from search_engine.core.config import Settings
    from search_engine.core.logging import setup_logging

    setup_logging("WARNING")
    suffix = uuid.uuid4().hex[:8]
    overrides = {"rerank_model": args.rerank_model} if args.rerank_model else {}
    settings = Settings(
        es_text_index=f"eval_text_{suffix}", es_image_index=f"eval_images_{suffix}",
        index_version_key=f"eval:version:{suffix}", **overrides,
    )
    questions = [Question(**json.loads(line)) for line in (HERE / "questions.jsonl").read_text().splitlines() if line]
    corpus = Path(tempfile.mkdtemp(prefix="eval_corpus_"))
    try:
        build_corpus(corpus)
        run(settings, questions, corpus, enhance=args.enhance)
    finally:
        cleanup(settings)
        shutil.rmtree(corpus, ignore_errors=True)


def run(settings, questions: list[Question], corpus: Path, *, enhance: bool) -> None:
    from search_engine.infra import models
    from search_engine.ingestion.pipeline import IngestionPipeline
    from search_engine.retrieval.hybrid_search import HybridSearch
    from search_engine.retrieval.query_enhance import QueryEnhancer
    from search_engine.retrieval.rerank import Reranker
    from search_engine.schemas.query import SearchFilters

    models.get_settings = lambda: settings  # so the cross encoder loads `settings.rerank_model`
    ingestion = IngestionPipeline(settings)
    report = ingestion.ingest_path(corpus)
    print(f"Ingested {report.files} files -> {report.chunks} chunks; failed: {report.failed or 'none'}\n")

    search = HybridSearch(settings, text_embedder=ingestion.text_embedder)
    reranker = Reranker(settings, search=search)
    enhancer = QueryEnhancer(settings) if enhance else None
    reranker.rerank("warm up", reranker.fuse("warm up", SearchFilters()))  # load the cross encoder before timing

    stages = {"text hybrid search": [], "+ image search, RRF": [], "+ cross encoder": []}
    seconds = {name: 0.0 for name in stages}
    for q in questions:
        text, filters = q.question, SearchFilters()
        if enhancer:
            enhanced = enhancer.enhance(q.question)
            text, filters = enhanced.search_text, enhanced.filters
        start = time.perf_counter()
        hybrid = search.search(text, filters).text
        seconds["text hybrid search"] += time.perf_counter() - start
        start = time.perf_counter()
        fused = reranker.fuse(text, filters)
        seconds["+ image search, RRF"] += time.perf_counter() - start
        start = time.perf_counter()
        reranked = reranker.rerank(text, fused)
        seconds["+ cross encoder"] += time.perf_counter() - start
        for name, docs in zip(stages, (hybrid, fused, reranked), strict=True):
            stages[name].append(first_hit(q, docs))

    print_table(stages, seconds, len(questions), settings)
    print_misses(questions, stages["+ cross encoder"])


def first_hit(question: Question, docs) -> int | None:
    """1-based rank of the first retrieved chunk that answers the question, or None."""
    return next((rank for rank, doc in enumerate(docs, start=1) if question.is_answered_by(doc)), None)


def print_table(stages: dict[str, list[int | None]], seconds: dict[str, float], n: int, settings) -> None:
    header = f"{'stage':<22}" + "".join(f"{f'recall@{k}':>11}" for k in KS) + f"{'MRR@10':>9}{'ms/query':>10}"
    print(f"cross encoder: {settings.rerank_model}\n{header}\n{'-' * len(header)}")
    for name, ranks in stages.items():
        recalls = "".join(f"{sum(r is not None and r <= k for r in ranks) / n:>11.2f}" for k in KS)
        mrr = sum(1 / r for r in ranks if r is not None and r <= 10) / n
        print(f"{name:<22}{recalls}{mrr:>9.3f}{1000 * seconds[name] / n:>10.0f}")
    print("(stage times are for that step only; the three together are one full retrieval)")


def print_misses(questions: list[Question], ranks: list[int | None]) -> None:
    misses = [(q, r) for q, r in zip(questions, ranks, strict=True) if r is None or r > 1]
    if misses:
        print("\nNot ranked first after the cross encoder:")
        for q, rank in misses:
            print(f"  [{q.kind:<7}] rank {rank or '>10':>3}: {q.question}")


def build_corpus(folder: Path) -> None:
    """The committed text files, plus a generated PDF (text, a ruled table, a bar chart) and two photos."""
    for file in (HERE / "corpus").iterdir():
        shutil.copy(file, folder / file.name)
    _write_pdf(folder / "report.pdf")
    _gradient(folder / "sunset.png", top=(255, 120, 20), bottom=(120, 20, 60), sun=(255, 220, 90))
    _gradient(folder / "forest.png", top=(120, 190, 90), bottom=(20, 70, 25), sun=None)


def _write_pdf(path: Path) -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Quarterly results: revenue rose in every region compared with last quarter.")
    rows = [("Region", "Revenue"), ("North", "120"), ("South", "95"), ("East", "143"), ("West", "78")]
    for i in range(len(rows) + 1):
        page.draw_line((72, 120 + i * 24), (312, 120 + i * 24))
    for j in range(3):
        page.draw_line((72 + j * 120, 120), (72 + j * 120, 120 + len(rows) * 24))
    for i, cells in enumerate(rows):
        for j, cell in enumerate(cells):
            page.insert_text((78 + j * 120, 136 + i * 24), cell)
    page = pdf.new_page()
    page.insert_text((72, 72), "Revenue by region is shown in the bar chart below.")
    page.draw_line((72, 400), (312, 400))
    page.draw_line((72, 400), (72, 220))
    for i, height in enumerate((120, 95, 143, 78)):
        x = 92 + i * 55
        page.draw_rect(pymupdf.Rect(x, 400 - height, x + 35, 400), color=(0, 0, 1), fill=(0.2, 0.4, 0.8))
    pdf.set_metadata({"title": "Quarterly Report"})
    pdf.save(path)
    pdf.close()


def _gradient(path: Path, top, bottom, sun) -> None:
    width, height = 320, 220
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        t = y / height
        draw.line([(0, y), (width, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom, strict=True)))
    if sun:
        draw.ellipse((120, 110, 200, 190), fill=sun)
    else:  # a row of simple trees
        for x in range(10, width, 40):
            draw.polygon([(x, 200), (x + 30, 200), (x + 15, 90)], fill=(10, 60, 15))
    buffer = io.BytesIO()
    image.save(buffer, format="png")
    path.write_bytes(buffer.getvalue())


def cleanup(settings) -> None:
    from search_engine.infra.elasticsearch import get_es_client
    from search_engine.infra.redis import get_redis

    try:
        get_es_client().indices.delete(
            index=f"{settings.es_text_index},{settings.es_image_index}", ignore_unavailable=True
        )
        get_redis().delete(settings.index_version_key)
    except Exception as error:
        print(f"Cleanup failed: {error}")


if __name__ == "__main__":
    main()
