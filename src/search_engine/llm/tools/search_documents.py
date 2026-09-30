from langchain_core.tools import BaseTool, tool

from search_engine.llm.context import AnswerContext


def build(context: AnswerContext) -> BaseTool:
    @tool
    def search_documents(query: str) -> str:
        """Search the user's files again with a different query. Use it when the context passages don't contain
        the answer, or cover only part of a question with several parts. Returns new numbered passages you can
        cite. `query` should be a short, specific search phrase, not the whole question."""
        if context.reranker is None:
            return "Search is not available."
        docs = context.reranker.retrieve(query, context.filters, context.settings.agent_tool_top_k)
        return context.add(docs)

    return search_documents
