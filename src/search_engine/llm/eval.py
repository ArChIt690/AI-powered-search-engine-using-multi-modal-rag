"""EVAL: an LLM judge scores each safe answer against the passages it was written from.

The judge runs on the other provider than the answering LLM (role "judge": Gemini, falling back to Groq), so an
answer is not marked by the model that wrote it. Scores are 1-5 for faithfulness, relevance and citation
correctness; an answer passes when every score reaches `eval_pass_score`. The user gets the answer either way;
only answers that pass are cached.
"""

import logging
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from search_engine.core.config import Settings, get_settings
from search_engine.infra.llm_client import get_structured_model
from search_engine.llm.context import format_context
from search_engine.schemas.eval import EvalResult, EvalScores

logger = logging.getLogger(__name__)

_SYSTEM = """You are a strict judge of answers produced by a search engine from a user's documents.
You get the QUESTION, the numbered CONTEXT passages the answer was written from, the TOOL RESULTS the system
looked up while answering (a file list, a calculation, an external lookup), and the ANSWER, which cites passages
as [n]. Score the answer from 1 (bad) to 5 (good) on:
- faithfulness: is every claim in the answer supported by the context passages or the tool results? A claim found
  in neither lowers the score. Results of simple arithmetic on numbers in the passages count as supported.
- relevance: does the answer address what the question asks?
- citation_correctness: does each [n] point to a passage that really contains that claim, and are claims that come
  from passages cited? Claims that come from a tool result need no [n].
An answer that correctly says the context does not contain the answer gets 5 for faithfulness and citation
correctness. Judge only from the passages and tool results: do not use your own knowledge to decide whether a
claim is true.
In notes, explain any score below 5 in one or two sentences."""

_HUMAN = "QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nTOOL RESULTS:\n{tool_results}\n\nANSWER:\n{answer}"
_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])


class Evaluator:
    def __init__(self, settings: Settings | None = None, *, model: Runnable | None = None):
        self.settings = settings or get_settings()
        self._model = model

    @property
    def chain(self) -> Runnable:
        return _PROMPT | (self._model or get_structured_model(EvalScores, "judge"))

    def evaluate(
        self, question: str, answer: str, chunks: Sequence[Document], tool_results: Sequence[str] = ()
    ) -> EvalResult | None:
        """EVAL. The scores, or None when the judge can't run on either provider (then nothing is cached)."""
        try:
            scores = self.chain.invoke({
                "question": question,
                "answer": answer,
                "context": format_context(chunks),
                "tool_results": "\n\n".join(tool_results) or "(none)",
            })
        except Exception as error:
            logger.warning("Eval could not run, the answer is returned without scores: %s", str(error)[:200])
            return None
        if not isinstance(scores, EvalScores):
            return None
        lowest = min(scores.faithfulness, scores.relevance, scores.citation_correctness)
        return EvalResult(**scores.model_dump(), passed=lowest >= self.settings.eval_pass_score)
