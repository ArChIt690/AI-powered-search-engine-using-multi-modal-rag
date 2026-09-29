"""The LLM box: answers the question from the reranked chunks, citing them as [n].

In Part 2 this is one direct LLM call. Part 3 replaces the inside of `LLMAgent.answer` with the full LLM
Architecture (agent with MCP and TOOLS, GUARDRAIL, EVAL), behind the same method.
"""

import re
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from search_engine.core.exceptions import LLMUnavailableError
from search_engine.infra.llm_client import get_chat_model
from search_engine.schemas.chunk import Modality

_SYSTEM = """You answer questions using only the numbered context passages from the user's own files.
Rules:
- Use only facts stated in the context. Do not use outside knowledge.
- Cite every claim with the number of the passage it comes from, in plain square brackets like [1] or [2][3].
- If the context does not contain the answer, say that the documents don't cover it. Do not guess.
- Pictures (images, charts, video frames) are given only by their description: say what they are and where they
  are, but never describe what they show beyond that description.
- The conversation is there only to understand the question; answer the latest question.
- Be concise."""

_HUMAN = """Conversation so far:
{history}

Context:
{context}

Question: {question}"""

_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])
_PICTURES = {Modality.IMAGE.value, Modality.VIDEO_FRAME.value}
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  # [1], [2][3], [1, 3]
# gpt-oss models (Groq) cite in their own style, 【1】 or 【1†L3-L5】; rewritten to [1] so every model reads the same
_FULLWIDTH_CITATION = re.compile(r"【\s*(\d+(?:\s*,\s*\d+)*)(?:†[^】]*)?\s*】")


class LLMAgent:
    def __init__(self, *, llm: BaseChatModel | None = None):
        self._llm = llm

    @property
    def llm(self) -> BaseChatModel:
        return self._llm or get_chat_model()

    def answer(self, question: str, chunks: Sequence[Document], history: Sequence[BaseMessage] = ()) -> str:
        """LLM: the answer text, citing `chunks` by their 1-based position. Raises LLMUnavailableError."""
        prompt = _PROMPT.invoke(
            {"question": question, "context": format_context(chunks), "history": format_history(history)}
        )
        try:
            reply = self.llm.invoke(prompt)
        except Exception as error:
            raise LLMUnavailableError(f"The LLM could not answer: {str(error)[:300]}") from error
        text = normalise_citations(str(reply.text)).strip()
        if not text:
            raise LLMUnavailableError("The LLM returned an empty answer")
        return text


def normalise_citations(answer: str) -> str:
    """Model-specific citation marks -> [n]."""
    return _FULLWIDTH_CITATION.sub(lambda match: f"[{match.group(1)}]", answer)


def format_context(chunks: Sequence[Document]) -> str:
    """Numbered passages, each with where it comes from, so the LLM can cite [n] and the reader can check it."""
    blocks = []
    for number, doc in enumerate(chunks, start=1):
        m = doc.metadata
        where = ", ".join(part for part in (m.get("file_name"), _location(m), m.get("content")) if part)
        if m.get("modality") in _PICTURES:
            blocks.append(f"[{number}] ({where}) A picture; only this description is known: {doc.page_content}")
        else:
            blocks.append(f"[{number}] ({where})\n{doc.page_content}")
    return "\n\n".join(blocks)


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


def _location(metadata: dict) -> str | None:
    if metadata.get("page") is not None:
        return f"page {metadata['page']}"
    if metadata.get("timestamp") is not None:
        seconds = int(metadata["timestamp"])
        return f"at {seconds // 60:02d}:{seconds % 60:02d}"
    return None
