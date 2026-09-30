import argparse
import sys
from pathlib import Path

from search_engine.core.logging import setup_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AI-powered multimodal search engine")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="Ingest a file or folder into the Vector Database")
    ingest.add_argument("path", type=Path)

    search = commands.add_parser("search", help="Ask a question; get a cited answer")
    search.add_argument("query")
    search.add_argument("--session", help="session id, so follow-up questions keep their context")
    search.add_argument("--file-type", action="append", default=[], help="only these file types, e.g. pdf (repeatable)")
    search.add_argument("--file-name", action="append", default=[], help="only this file, e.g. report.pdf (repeatable)")
    search.add_argument(
        "--modality", action="append", default=[], help="text, image, video_transcript or video_frame (repeatable)"
    )
    search.add_argument(
        "--content", action="append", default=[], help="text, table, record, image, chart or frame (repeatable)"
    )
    search.add_argument("--top-k", type=int, help="chunks given to the LLM (default TOP_K)")

    args = parser.parse_args(argv)
    setup_logging()

    if args.command == "ingest":
        # Imported here: the models load slowly, and `--help` shouldn't wait for them.
        from search_engine.ingestion.pipeline import IngestionPipeline

        report = IngestionPipeline().ingest_path(args.path)
        print(f"\nIngested {report.files} files -> {report.chunks} chunks")
        if report.skipped:
            print(f"Skipped {len(report.skipped)} unsupported files")
        for file, error in report.failed.items():
            print(f"FAILED {file}: {error}")

    elif args.command == "search":
        from search_engine.core.exceptions import LLMUnavailableError
        from search_engine.retrieval.pipeline import SearchPipeline

        try:
            response = SearchPipeline().search(search_request(args))
        except LLMUnavailableError as error:
            sys.exit(f"\n{error}")
        print(format_response(response))


def search_request(args: argparse.Namespace):
    from search_engine.schemas.query import SearchFilters, SearchRequest

    filters = SearchFilters(
        file_type=args.file_type, file_name=args.file_name, modality=args.modality, content=args.content
    )
    return SearchRequest(query=args.query, session_id=args.session, filters=filters, top_k=args.top_k)


def format_response(response) -> str:
    lines = ["", response.answer, ""]
    if response.enhanced_query != response.query:
        lines.append(f"(searched as: {response.enhanced_query})")
    for c in response.citations:
        where = f", page {c.page}" if c.page is not None else ""
        if c.timestamp is not None:
            where += f", at {int(c.timestamp) // 60:02d}:{int(c.timestamp) % 60:02d}"
        lines.append(f"[{c.number}] {c.file_name}{where} ({c.content}): {c.snippet[:100]!r}")
    lines.append(f"answered by: {response.source.value}")
    if response.tools_used:
        lines.append(f"tools used: {', '.join(response.tools_used)}")
    if response.eval:
        e = response.eval
        verdict = "passed" if e.passed else "not passed, so not cached"
        lines.append(
            f"eval: faithfulness {e.faithfulness}/5, relevance {e.relevance}/5, citations {e.citation_correctness}/5"
            f" ({verdict})" + (f" - {e.notes}" if e.notes else "")
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
