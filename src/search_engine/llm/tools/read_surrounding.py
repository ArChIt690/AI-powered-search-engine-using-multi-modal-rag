from langchain_core.tools import BaseTool, tool

from search_engine.infra.elasticsearch import get_es_client
from search_engine.llm.context import AnswerContext
from search_engine.retrieval.hybrid_search import to_document


def build(context: AnswerContext) -> BaseTool:
    @tool
    def read_surrounding(passage_number: int) -> str:
        """Read the text just before and just after a context passage, from the same file. Use it when a passage
        is cut off, or the answer seems to continue beyond it. `passage_number` is the [n] of the passage.
        Returns new numbered passages you can cite."""
        if not 1 <= passage_number <= len(context.chunks):
            return f"There is no passage [{passage_number}]; the context has [1] to [{len(context.chunks)}]."
        metadata = context.chunks[passage_number - 1].metadata
        doc_id, index = metadata.get("doc_id"), metadata.get("chunk_index")
        if doc_id is None or index is None:
            return "This passage has no neighbours."
        # Chunk ids are <doc_id>-<chunk_index> (Metadata Enrichment), so neighbours are fetched by id.
        ids = [f"{doc_id}-{i}" for i in (index - 1, index + 1) if i >= 0]
        found = get_es_client().mget(index=context.settings.es_text_index, ids=ids)["docs"]
        neighbours = [to_document(hit) for hit in found if hit.get("found")]
        return context.add(neighbours) if neighbours else "This passage has no neighbouring text."

    return read_surrounding
