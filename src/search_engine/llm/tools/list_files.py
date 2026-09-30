from langchain_core.tools import BaseTool, tool

from search_engine.infra.elasticsearch import get_es_client
from search_engine.llm.context import AnswerContext

_MAX_FILES = 200


def build(context: AnswerContext) -> BaseTool:
    @tool
    def list_files() -> str:
        """List the files that have been ingested, with how many text passages and pictures each has. Use it for
        questions about what documents exist, or to find the right file name to search for."""
        s = context.settings
        counts: dict[str, dict[str, int]] = {}
        for index, kind in ((s.es_text_index, "text passages"), (s.es_image_index, "pictures")):
            reply = get_es_client().search(
                index=index,
                size=0,
                ignore_unavailable=True,
                aggs={"files": {"terms": {"field": "metadata.file_name.keyword", "size": _MAX_FILES}}},
            )
            for bucket in reply.get("aggregations", {}).get("files", {}).get("buckets", []):
                counts.setdefault(bucket["key"], {})[kind] = bucket["doc_count"]
        if not counts:
            return "No files have been ingested."
        lines = [
            f"- {name}: " + ", ".join(f"{count} {kind}" for kind, count in kinds.items())
            for name, kinds in sorted(counts.items())
        ]
        return f"{len(counts)} files:\n" + "\n".join(lines)

    return list_files
