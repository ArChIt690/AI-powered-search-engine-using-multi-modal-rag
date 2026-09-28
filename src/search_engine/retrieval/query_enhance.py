"""QUERY ENHANCEMENT: one fast LLM call that rewrites the query to stand alone, adds keywords and finds filters."""

import logging
from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from search_engine.core.config import Settings, get_settings
from search_engine.infra.llm_client import get_chat_model
from search_engine.schemas.query import EnhancedQuery

logger = logging.getLogger(__name__)

_SYSTEM = """You prepare search queries for a search engine over the user's own files: text documents, \
PDFs (text, tables, charts, images), CSV/JSON/XML records, images and videos (spoken transcript and frames).
Rewrite the latest question so it stands alone, add a few extra search keywords, and extract only the filters \
the user explicitly asked for. Never answer the question."""

_HUMAN = """Conversation so far:
{history}

Latest question: {query}"""

_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])
_MAX_HISTORY_CHARS = 500  # per message; enough to resolve references, and keeps the call small


class QueryEnhancer:
    def __init__(self, settings: Settings | None = None, *, llm: BaseChatModel | None = None):
        self.settings = settings or get_settings()
        self.chain = _PROMPT | (llm or get_chat_model(fast=True)).with_structured_output(EnhancedQuery)

    def enhance(self, query: str, history: Sequence[BaseMessage] = ()) -> EnhancedQuery:
        """QUERY Enhancement. If the LLM fails (rate limit, outage, bad output), searches the query as typed."""
        try:
            result = self.chain.invoke({"query": query, "history": self._format(history)})
        except Exception as error:  # an outage of a free API must not break search
            logger.warning("Query Enhancement failed, searching the query as typed: %s", str(error)[:200])
            return EnhancedQuery(query=query)
        if not isinstance(result, EnhancedQuery) or not result.query.strip():
            return EnhancedQuery(query=query)
        result.query = result.query.strip()
        result.keywords = [word.strip() for word in result.keywords if word.strip()][:5]
        return result

    def _format(self, history: Sequence[BaseMessage]) -> str:
        recent = list(history)[-2 * self.settings.session_history_turns :]  # a turn = question + answer
        if not recent:
            return "(none)"
        lines = []
        for message in recent:
            role = "Assistant" if isinstance(message, AIMessage) else "User"
            lines.append(f"{role}: {str(message.text)[:_MAX_HISTORY_CHARS]}")
        return "\n".join(lines)
