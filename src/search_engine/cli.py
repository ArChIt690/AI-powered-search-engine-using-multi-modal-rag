import argparse
from pathlib import Path

from search_engine.core.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-powered multimodal search engine")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="Ingest a file or folder into the Vector Database")
    ingest.add_argument("path", type=Path)

    args = parser.parse_args()
    setup_logging()

    if args.command == "ingest":
        # Imported here so `--help` works before the Ingestion pipeline (Part 1) is built.
        from search_engine.ingestion.pipeline import IngestionPipeline

        report = IngestionPipeline().ingest_path(args.path)
        print(f"\nIngested {report.files} files -> {report.chunks} chunks")
        if report.skipped:
            print(f"Skipped {len(report.skipped)} unsupported files")
        for file, error in report.failed.items():
            print(f"FAILED {file}: {error}")


if __name__ == "__main__":
    main()
