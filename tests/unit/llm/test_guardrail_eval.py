import json

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from search_engine.core.config import Settings
from search_engine.llm.agent import AgentAnswer
from search_engine.llm.eval import Evaluator
from search_engine.llm.eval_store import EvalStore
from search_engine.llm.guardrail import Guardrail, GuardrailResult
from search_engine.llm.pipeline import LLMPipeline
from search_engine.schemas.eval import EvalRecord, EvalResult, EvalScores, GuardrailVerdict

CHUNKS = [Document("|East|143|", metadata={"chunk_id": "r-0", "file_name": "report.pdf", "page": 1, "content": "table"})]


def model(result=None, error=None):
    """Stands in for a structured-output model; records the prompts it was given."""
    seen = []

    def respond(prompt):
        seen.append(prompt.to_string())
        if error:
            raise error
        return result

    runnable = RunnableLambda(respond)
    runnable.seen = seen
    return runnable


# GUARDRAIL

@pytest.mark.parametrize(
    ("answer", "named"),
    [
        ("The key is gsk_abcdefghijklmnopqrstuvwxyz012345 [1].", "an API key"),
        ("Use sk-proj-abcdefghijklmnopqrstuvwxyz for the API [2].", "an API key"),
        ("-----BEGIN RSA PRIVATE KEY-----\nMIIE...", "a private key"),
        ("The admin password is hunter2secret [1].", "a password"),
    ],
)
def test_credentials_are_blocked_by_rule_without_calling_the_model(answer, named):
    safety_model = model(GuardrailVerdict(safe=True))

    result = Guardrail(model=safety_model).check("what is the key?", answer)

    assert result.verdict.safe is False and result.verdict.category == "secrets" and named in result.verdict.reason
    assert safety_model.seen == []


def test_a_safe_verdict_passes_and_the_model_sees_question_and_answer():
    safety_model = model(GuardrailVerdict(safe=True, category="other", reason="fine"))

    result = Guardrail(model=safety_model).check("East revenue?", "East made 143 [1].")

    assert result == GuardrailResult(GuardrailVerdict(safe=True, category="none", reason=""), checked=True)
    assert "QUESTION:\nEast revenue?" in safety_model.seen[0] and "ANSWER:\nEast made 143 [1]." in safety_model.seen[0]
    assert "UNSAFE when" in safety_model.seen[0]  # the written policy is sent


def test_an_unsafe_verdict_keeps_its_category_and_always_has_a_reason():
    harmful = GuardrailVerdict(safe=False, category="harmful", reason="It explains how to make a toxic gas.")
    assert Guardrail(model=model(harmful)).check("q", "a").verdict == harmful

    no_reason = Guardrail(model=model(GuardrailVerdict(safe=False, category="other"))).check("q", "a").verdict
    assert no_reason.safe is False and no_reason.reason == "The answer was blocked by the safety check."


def test_when_the_safety_model_is_down_the_answer_passes_on_rules_only_but_is_marked_unchecked():
    result = Guardrail(model=model(error=RuntimeError("429"))).check("q", "East made 143 [1].")

    assert result.verdict.safe is True and result.checked is False


# EVAL

def test_eval_passes_when_every_score_reaches_the_pass_score():
    judge = model(EvalScores(faithfulness=5, relevance=4, citation_correctness=5))

    result = Evaluator(Settings(eval_pass_score=4), model=judge).evaluate("East revenue?", "East made 143 [1].", CHUNKS)

    assert result.passed is True and result.relevance == 4
    assert "[1] (report.pdf, page 1, table)\n|East|143|" in judge.seen[0] and "ANSWER:\nEast made 143 [1]." in judge.seen[0]


def test_eval_judges_against_tool_results_too():
    judge = model(EvalScores(faithfulness=5, relevance=5, citation_correctness=5))
    evaluator = Evaluator(Settings(), model=judge)

    files = "list_files: 2 files:\n- notes.md\n- report.pdf"
    evaluator.evaluate("Which files are there?", "report.pdf and notes.md.", CHUNKS, [files])
    evaluator.evaluate("q", "a", CHUNKS)

    assert f"TOOL RESULTS:\n{files}\n\nANSWER:" in judge.seen[0]
    assert "TOOL RESULTS:\n(none)" in judge.seen[1]


def test_eval_fails_when_any_score_is_below_the_pass_score():
    judge = model(EvalScores(faithfulness=5, relevance=5, citation_correctness=2, notes="[2] is the wrong passage."))

    result = Evaluator(Settings(eval_pass_score=4), model=judge).evaluate("q", "a [2]", CHUNKS)

    assert result.passed is False and result.notes == "[2] is the wrong passage."


def test_eval_returns_none_when_the_judge_cannot_run():
    assert Evaluator(Settings(), model=model(error=RuntimeError("503"))).evaluate("q", "a", CHUNKS) is None


# "all the eval results are stored separately"

def record(**overrides) -> EvalRecord:
    values = {
        "timestamp": "2026-09-30T10:00:00+00:00", "question": "East revenue?", "answer": "East made 143 [1].",
        "chunk_ids": ["r-0"], "cited": [1], "tools_used": ["calculator"], "guardrail": GuardrailVerdict(safe=True),
        "eval": EvalResult(faithfulness=5, relevance=5, citation_correctness=5, passed=True),
    }
    return EvalRecord(**{**values, **overrides})


def test_eval_results_are_appended_one_json_line_each_and_read_back(tmp_path):
    store = EvalStore(Settings(eval_results_dir=str(tmp_path / "eval_results")))

    path = store.save(record())
    store.save(record(question="second", eval=None))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert path.parent == tmp_path / "eval_results" and path.suffix == ".jsonl" and len(lines) == 2
    assert json.loads(lines[0])["eval"]["passed"] is True and json.loads(lines[1])["eval"] is None
    assert [r.question for r in store.read()] == ["East revenue?", "second"]


def test_a_store_that_cannot_write_does_not_fail_the_search(tmp_path, caplog):
    blocker = tmp_path / "file"
    blocker.write_text("not a folder")

    assert EvalStore(Settings(eval_results_dir=str(blocker))).save(record()) is None
    assert "Could not store the eval result" in caplog.text


# LLM -> GUARDRAIL -> EVAL -> RESULT

class FakeAgent:
    def __init__(self, text="East made 143 [1].", tools=("calculator",)):
        self.text, self.tools, self.calls = text, list(tools), []

    def answer(self, question, chunks, history, filters):
        self.calls.append((question, filters))
        return AgentAnswer(self.text, list(chunks), self.tools, ["calculator: 143"])


def pipeline(tmp_path, *, verdict=GuardrailVerdict(safe=True), scores=EvalScores(faithfulness=5, relevance=5, citation_correctness=5),
             guard_error=None, judge_error=None):
    settings = Settings(eval_results_dir=str(tmp_path / "eval_results"), eval_pass_score=4)
    judge = model(scores, judge_error)
    llm = LLMPipeline(
        settings, agent=FakeAgent(), guardrail=Guardrail(model=model(verdict, guard_error)),
        evaluator=Evaluator(settings, model=judge), store=EvalStore(settings),
    )
    return llm, judge


def test_a_safe_answer_is_evaluated_stored_and_cacheable(tmp_path):
    llm, judge = pipeline(tmp_path)

    result = llm.run("East revenue?", CHUNKS, filters="F")

    assert result.answer == "East made 143 [1]." and result.safe and result.eval.passed and result.cacheable
    assert result.tools_used == ["calculator"] and llm.agent.calls == [("East revenue?", "F")]
    assert "TOOL RESULTS:\ncalculator: 143" in judge.seen[0]  # the judge sees what the tools returned
    [stored] = llm.store.read()
    assert stored.question == "East revenue?" and stored.chunk_ids == ["r-0"] and stored.cited == [1]
    assert stored.eval.passed and stored.guardrail.safe and stored.tools_used == ["calculator"]


def test_an_unsafe_answer_skips_eval_is_not_cacheable_and_is_still_stored(tmp_path):
    unsafe = GuardrailVerdict(safe=False, category="harmful", reason="It explains how to make a toxic gas.")
    llm, judge = pipeline(tmp_path, verdict=unsafe)

    result = llm.run("q", CHUNKS)

    assert not result.safe and result.eval is None and not result.cacheable
    assert judge.seen == []  # UNSAFE goes straight to the USER; EVAL is not run
    [stored] = llm.store.read()
    assert stored.guardrail == unsafe and stored.eval is None


def test_an_answer_that_fails_eval_is_returned_but_not_cacheable(tmp_path):
    llm, _ = pipeline(tmp_path, scores=EvalScores(faithfulness=2, relevance=5, citation_correctness=5))

    result = llm.run("q", CHUNKS)

    assert result.safe and result.eval.passed is False and not result.cacheable


def test_an_answer_the_guardrail_or_judge_could_not_check_is_not_cacheable(tmp_path):
    unchecked, _ = pipeline(tmp_path, guard_error=RuntimeError("429"))
    unjudged, _ = pipeline(tmp_path, judge_error=RuntimeError("503"))

    assert unchecked.run("q", CHUNKS).cacheable is False  # safe by rules only
    result = unjudged.run("q", CHUNKS)
    assert result.safe and result.eval is None and result.cacheable is False
