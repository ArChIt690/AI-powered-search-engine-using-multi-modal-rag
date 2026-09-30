"""TOOLS: the local functions the LLM can call, one file per tool. Built per question, around its context."""

from langchain_core.tools import BaseTool

from search_engine.llm.context import AnswerContext
from search_engine.llm.tools import calculator, list_files, read_surrounding, search_documents

_TOOLS = (search_documents, read_surrounding, list_files, calculator)


def build_tools(context: AnswerContext) -> list[BaseTool]:
    return [module.build(context) for module in _TOOLS]
