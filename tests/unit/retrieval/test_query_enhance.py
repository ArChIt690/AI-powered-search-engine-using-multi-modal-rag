from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from search_engine.core.config import Settings
from search_engine.retrieval.query_enhance import QueryEnhancer
from search_engine.schemas.query import EnhancedQuery, SearchFilters


class FakeLLM:
    """Stands in for the chat model: `with_structured_output` returns a canned result (or raises), and keeps the
    prompt it was given."""

    def __init__(self, result=None, error: Exception | None = None):
        self.result, self.error, self.prompts, self.schema, self.method = result, error, [], None, None

    def with_structured_output(self, schema, method=None):
        self.schema, self.method = schema, method
        return RunnableLambda(self._respond)

    def _respond(self, prompt):
        self.prompts.append(prompt.to_string())
        if self.error:
            raise self.error
        return self.result


def test_returns_the_llms_rewrite_keywords_and_filters():
    result = EnhancedQuery(
        query="What revenue does the chart in report.pdf show?",
        keywords=[" sales ", "", "a", "b", "c", "d", "e"],
        filters=SearchFilters(content=["chart"]),
    )
    llm = FakeLLM(result)

    enhanced = QueryEnhancer(Settings(), llm=llm).enhance("what does that chart show?")

    assert llm.schema is EnhancedQuery and llm.method == "json_schema"
    assert enhanced.query == "What revenue does the chart in report.pdf show?"
    assert enhanced.keywords == ["sales", "a", "b", "c", "d"]  # trimmed, blanks dropped, at most 5
    assert enhanced.filters.content == ["chart"]


def test_prompt_holds_the_question_and_only_the_recent_history():
    llm = FakeLLM(EnhancedQuery(query="q"))
    history = [HumanMessage("old question"), AIMessage("old answer")] + [
        message for i in range(3) for message in (HumanMessage(f"question {i}"), AIMessage(f"answer {i}"))
    ]

    QueryEnhancer(Settings(session_history_turns=3), llm=llm).enhance("and page 3?", history)

    prompt = llm.prompts[0]
    assert "Latest question: and page 3?" in prompt
    assert "User: question 0\nAssistant: answer 0" in prompt and "Assistant: answer 2" in prompt
    assert "old question" not in prompt  # only the last 3 turns


def test_no_history_is_stated():
    llm = FakeLLM(EnhancedQuery(query="q"))

    QueryEnhancer(Settings(), llm=llm).enhance("q")

    assert "Conversation so far:\n(none)" in llm.prompts[0]


def test_long_history_messages_are_cut():
    llm = FakeLLM(EnhancedQuery(query="q"))

    QueryEnhancer(Settings(), llm=llm).enhance("q", [AIMessage("x" * 5000)])

    assert "x" * 500 in llm.prompts[0] and "x" * 501 not in llm.prompts[0]


def test_llm_failure_falls_back_to_the_query_as_typed():
    enhanced = QueryEnhancer(Settings(), llm=FakeLLM(error=RuntimeError("503 model overloaded"))).enhance("goroutines")

    assert enhanced == EnhancedQuery(query="goroutines")


def test_empty_or_missing_rewrite_falls_back_to_the_query():
    assert QueryEnhancer(Settings(), llm=FakeLLM(EnhancedQuery(query="  "))).enhance("q1").query == "q1"
    assert QueryEnhancer(Settings(), llm=FakeLLM(None)).enhance("q2").query == "q2"
