"""GUARDRAIL and EVAL results: what the two boxes decide about an LLM answer, and what is stored about it."""

from typing import Literal

from pydantic import BaseModel, Field

GuardrailCategory = Literal["none", "harmful", "secrets", "personal_data", "prompt_injection", "other"]


class GuardrailVerdict(BaseModel):
    """GUARDRAIL's decision on an answer (also the safety model's structured output)."""

    safe: bool = Field(description="True when the answer may be shown to the user.")
    category: GuardrailCategory = Field(default="none", description="Which rule the answer breaks; 'none' when safe.")
    reason: str = Field(default="", description="One sentence for the user saying why the answer is blocked; empty when safe.")


class EvalScores(BaseModel):
    """EVAL's scores for an answer, 1 (bad) to 5 (good) (the judge's structured output)."""

    faithfulness: int = Field(ge=1, le=5, description="Is every claim in the answer supported by the passages it cites? 5 = all, 1 = none.")
    relevance: int = Field(ge=1, le=5, description="Does the answer address the question that was asked? 5 = fully, 1 = not at all.")
    citation_correctness: int = Field(ge=1, le=5, description="Do the [n] marks point to the passages that actually contain the claim? 5 = all, 1 = none.")
    notes: str = Field(default="", description="One or two sentences explaining any score below 5.")


class EvalResult(EvalScores):
    passed: bool = Field(description="True when every score reaches EVAL_PASS_SCORE; only passed answers are cached.")


class EvalRecord(BaseModel):
    """One line of `data/eval_results/<date>.jsonl`: "all the eval results are stored separately"."""

    timestamp: str
    question: str
    answer: str
    chunk_ids: list[str] = Field(description="The passages the LLM was given, in [n] order.")
    cited: list[int] = Field(description="The passage numbers the answer cites.")
    tools_used: list[str] = Field(default_factory=list)
    guardrail: GuardrailVerdict
    eval: EvalResult | None = Field(description="None when the answer was blocked or the judge could not run.")
