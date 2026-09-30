"""The LLM box: an agent that answers the question from the reranked chunks, citing them as [n], and can call
TOOLS and MCP tools when those chunks are not enough.

LangChain `create_agent` (LangGraph) runs the loop: the LLM either answers or asks for a tool, the tool's result
goes back to it, and so on, up to `agent_max_tool_calls`. If the agent run fails, the question is answered with one
direct LLM call from the chunks already in hand, so a tool problem never costs the user their answer.
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ModelFallbackMiddleware, ToolCallLimitMiddleware
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from search_engine.core.config import Settings, get_settings
from search_engine.core.exceptions import LLMUnavailableError
from search_engine.infra.llm_client import chat_models
from search_engine.llm.context import AnswerContext, format_context
from search_engine.llm.mcp import load_mcp_tools, run_sync
from search_engine.llm.tools import build_tools
from search_engine.schemas.query import SearchFilters

logger = logging.getLogger(__name__)

_RULES = """You answer questions using only the numbered context passages from the user's own files.
Rules:
- Use only facts stated in the context passages or returned by your tools. Do not use outside knowledge.
- Cite every claim with the number of the passage it comes from, in plain square brackets like [1] or [2][3].
- If the context does not contain the answer, say that the documents don't cover it. Do not guess.
- Pictures (images, charts, video frames) are given only by their description: say what they are and where they
  are, but never describe what they show beyond that description.
- The conversation is there only to understand the question; answer the latest question.
- Be concise."""

_TOOL_RULES = """
Tools:
- First try to answer from the context passages. Call a tool only when they are not enough.
- Passages a tool returns are numbered like the context and can be cited the same way.
- Use `calculator` for every calculation; cite the passages the numbers came from.
- When you have what you need, or the tools found nothing more, give the final answer."""

_HUMAN = """Conversation so far:
{history}

Context:
{context}

Question: {question}"""

_PASSAGE_TOOLS = {"search_documents", "read_surrounding"}
_MAX_TOOL_RESULT_CHARS = 2000

_DIRECT_PROMPT = ChatPromptTemplate.from_messages([("system", _RULES), ("human", _HUMAN)])
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  # [1], [2][3], [1, 3]
# gpt-oss models (Groq) cite in their own style, 【1】 or 【1†L3-L5】; rewritten to [1] so every model reads the same
_FULLWIDTH_CITATION = re.compile(r"【\s*(\d+(?:\s*,\s*\d+)*)(?:†[^】]*)?\s*】")
_FULLWIDTH_OTHER = re.compile(r"\s*【[^】]*】")  # e.g. 【calculator】: a tool named as a source; not a passage


@dataclass
class AgentAnswer:
    text: str
    chunks: list[Document]  # the numbered passages the answer's [n] refer to: the reranked chunks + tool results
    tools_used: list[str] = field(default_factory=list)
    tool_results: list[str] = field(default_factory=list)  # "<tool>: <what it returned>", evidence for EVAL


class LLMAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        models: Sequence[BaseChatModel] | None = None,
        reranker=None,
        mcp_tools: Sequence[BaseTool] | None = None,
    ):
        self.settings = settings or get_settings()
        self._models = list(models) if models is not None else None
        self.reranker = reranker  # lets `search_documents` run the hybrid search + reranking again
        self.mcp_tools = list(mcp_tools) if mcp_tools is not None else load_mcp_tools(self.settings)

    @property
    def models(self) -> list[BaseChatModel]:
        """The answer model and its fallback (the other provider)."""
        return self._models if self._models is not None else chat_models("answer")

    def answer(
        self,
        question: str,
        chunks: Sequence[Document],
        history: Sequence[BaseMessage] = (),
        filters: SearchFilters | None = None,
    ) -> AgentAnswer:
        """LLM (+ MCP, TOOLS): the answer, citing passages by number. Raises LLMUnavailableError."""
        context = AnswerContext(self.settings, list(chunks), filters or SearchFilters(), self.reranker)
        try:
            text = self._run_agent(question, context, history)
        except LLMUnavailableError:
            raise
        except Exception as error:
            logger.warning("The agent run failed, answering directly from the context: %s", str(error)[:200])
            text = self._answer_directly(question, context, history)
        return AgentAnswer(text, context.chunks, context.tools_used, context.tool_results)

    def _run_agent(self, question: str, context: AnswerContext, history: Sequence[BaseMessage]) -> str:
        first, *fallbacks = self.models
        limit = self.settings.agent_max_tool_calls
        middleware = [
            # at the limit, further tool calls are refused and the LLM is told to answer with what it has
            ToolCallLimitMiddleware(run_limit=limit, exit_behavior="continue"),
            ModelCallLimitMiddleware(run_limit=limit + 2, exit_behavior="end"),
        ]
        if fallbacks:
            middleware.insert(0, ModelFallbackMiddleware(*fallbacks))
        agent = create_agent(
            first, [*build_tools(context), *self.mcp_tools], system_prompt=_RULES + _TOOL_RULES, middleware=middleware
        )
        prompt = _HUMAN.format(
            question=question, context=format_context(context.chunks), history=format_history(history)
        )
        # ainvoke: MCP tools are async-only
        state = run_sync(agent.ainvoke({"messages": [HumanMessage(prompt)]}))
        replies = [message for message in state["messages"] if isinstance(message, AIMessage)]
        context.tools_used.extend(call["name"] for reply in replies for call in reply.tool_calls)
        # What the other tools returned is evidence too (a file list, a sum, an MCP lookup). Passages that the
        # search tools found are already in the numbered context.
        context.tool_results.extend(
            f"{message.name}: {str(message.content)[:_MAX_TOOL_RESULT_CHARS]}"
            for message in state["messages"]
            if isinstance(message, ToolMessage) and message.name not in _PASSAGE_TOOLS
        )
        text = normalise_citations(str(replies[-1].text)).strip() if replies else ""
        if not text:
            raise RuntimeError("the agent ended without an answer")
        return text

    def _answer_directly(self, question: str, context: AnswerContext, history: Sequence[BaseMessage]) -> str:
        first, *fallbacks = self.models
        llm = first.with_fallbacks(fallbacks) if fallbacks else first
        prompt = _DIRECT_PROMPT.invoke(
            {"question": question, "context": format_context(context.chunks), "history": format_history(history)}
        )
        try:
            reply = llm.invoke(prompt)
        except Exception as error:
            raise LLMUnavailableError(f"The LLM could not answer: {str(error)[:300]}") from error
        text = normalise_citations(str(reply.text)).strip()
        if not text:
            raise LLMUnavailableError("The LLM returned an empty answer")
        return text


def normalise_citations(answer: str) -> str:
    """Model-specific citation marks -> [n]; marks that name a tool instead of a passage are dropped."""
    answer = _FULLWIDTH_CITATION.sub(lambda match: f"[{match.group(1)}]", answer)
    return _FULLWIDTH_OTHER.sub("", answer)


def format_history(history: Sequence[BaseMessage]) -> str:
    if not history:
        return "(none)"
    return "\n".join(f"{'Assistant' if isinstance(m, AIMessage) else 'User'}: {str(m.text)[:500]}" for m in history)


def cited_numbers(answer: str, count: int) -> list[int]:
    """The passage numbers the answer cites, in first-cited order; numbers outside 1..count are ignored."""
    numbers: list[int] = []
    for group in _CITATION.findall(answer):
        for part in group.split(","):
            number = int(part)
            if 1 <= number <= count and number not in numbers:
                numbers.append(number)
    return numbers
