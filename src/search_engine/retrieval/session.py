"""USER -- Sessional Queries --> REDIS: each session's questions and answers, for follow-up questions.

LangChain `RedisChatMessageHistory` (langchain-community: one Redis list per session id, expiring `session_ttl_s`
after the last question; langchain-redis' class of the same name is deprecated and fails with redis-py 8).
QUERY Enhancement reads it to rewrite follow-ups, and the LLM gets it as conversation context.
"""

import logging

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from search_engine.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class SessionStore:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def history(self, session_id: str | None) -> list[BaseMessage]:
        """The session's last `session_history_turns` question/answer pairs, oldest first ([] without a session)."""
        if not session_id:
            return []
        try:
            messages = self._history(session_id).messages
        except Exception as error:  # without Redis, search still works, just without follow-up context
            logger.warning("Could not read session %s: %s", session_id, error)
            return []
        return messages[-2 * self.settings.session_history_turns :]

    def append(self, session_id: str | None, question: str, answer: str) -> None:
        if not session_id:
            return
        try:
            self._history(session_id).add_messages([HumanMessage(question), AIMessage(answer)])
        except Exception as error:
            logger.warning("Could not save to session %s: %s", session_id, error)

    def _history(self, session_id: str) -> RedisChatMessageHistory:
        return RedisChatMessageHistory(
            session_id, url=self.settings.redis_url, key_prefix="search:session:", ttl=self.settings.session_ttl_s
        )
