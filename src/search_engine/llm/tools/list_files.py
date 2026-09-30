from langchain_core.tools import BaseTool, tool

from search_engine.ingestion.vector_store import file_counts
from search_engine.llm.context import AnswerContext


def build(context: AnswerContext) -> BaseTool:
    @tool
    def list_files() -> str:
        """List the files that have been ingested, with how many text passages and pictures each has. Use it for
        questions about what documents exist, or to find the right file name to search for."""
        files = file_counts(context.settings)
        if not files:
            return "No files have been ingested."
        lines = [f"- {name}: " + ", ".join(_parts(counts)) for name, counts in files.items()]
        return f"{len(files)} files:\n" + "\n".join(lines)

    return list_files


def _parts(counts: dict[str, int]) -> list[str]:
    labels = (("passages", "text passages"), ("pictures", "pictures"))
    return [f"{counts[key]} {label}" for key, label in labels if counts.get(key)]
