"""The LLM Architecture, in the diagram's order, for one question:

LLM (+ MCP, TOOLS) -> GUARDRAIL --UNSAFE--> USER, with the reason
                          `--SAFE--> EVAL -> RESULT -> "goes for caching" + FINAL RESULT to USER
                                       `-> "all the eval results are stored separately"
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from search_engine.core.config import Settings, get_settings
from search_engine.llm.agent import LLMAgent, cited_numbers
from search_engine.llm.eval import Evaluator
from search_engine.llm.eval_store import EvalStore
from search_engine.llm.guardrail import Guardrail
from search_engine.schemas.eval import EvalRecord, EvalResult, GuardrailVerdict
from search_engine.schemas.query import SearchFilters


@dataclass
class LLMResult:
    """RESULT: the answer and what GUARDRAIL and EVAL made of it."""

    answer: str
    chunks: list[Document]  # the numbered passages the answer's [n] refer to
    guardrail: GuardrailVerdict
    eval: EvalResult | None = None
    tools_used: list[str] = field(default_factory=list)
    guardrail_checked: bool = True

    @property
    def safe(self) -> bool:
        return self.guardrail.safe

    @property
    def cacheable(self) -> bool:
        """ "goes for caching": only answers that passed GUARDRAIL (really checked) and EVAL."""
        return self.safe and self.guardrail_checked and self.eval is not None and self.eval.passed


class LLMPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        agent: LLMAgent | None = None,
        guardrail: Guardrail | None = None,
        evaluator: Evaluator | None = None,
        store: EvalStore | None = None,
        reranker=None,
    ):
        self.settings = settings or get_settings()
        self.agent = agent or LLMAgent(self.settings, reranker=reranker)
        self.guardrail = guardrail or Guardrail()
        self.evaluator = evaluator or Evaluator(self.settings)
        self.store = store or EvalStore(self.settings)

    def run(
        self,
        question: str,
        chunks: Sequence[Document],
        history: Sequence[BaseMessage] = (),
        filters: SearchFilters | None = None,
    ) -> LLMResult:
        """LLM -> GUARDRAIL -> EVAL -> RESULT. Raises LLMUnavailableError when no LLM can answer."""
        reply = self.agent.answer(question, chunks, history, filters)
        guard = self.guardrail.check(question, reply.text)
        result = LLMResult(reply.text, reply.chunks, guard.verdict, None, reply.tools_used, guard.checked)
        if result.safe:  # UNSAFE goes straight back to the USER with the reason; EVAL is skipped
            result.eval = self.evaluator.evaluate(question, reply.text, reply.chunks, reply.tool_results)
        self.store.save(
            EvalRecord(
                timestamp=datetime.now(UTC).isoformat(),
                question=question,
                answer=result.answer,
                chunk_ids=[doc.metadata.get("chunk_id", "") for doc in result.chunks],
                cited=cited_numbers(result.answer, len(result.chunks)),
                tools_used=result.tools_used,
                guardrail=result.guardrail,
                eval=result.eval,
            )
        )
        return result
