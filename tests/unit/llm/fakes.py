"""A chat model for tests that plays back scripted replies and records what it was sent."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ScriptedModel(BaseChatModel):
    """Each call returns the next item of `replies`; an Exception in the list is raised instead."""

    replies: list[Any]
    seen: list[list[Any]] = Field(default_factory=list)  # the messages of every call
    tool_names: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.tool_names[:] = [tool.name for tool in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        if not self.replies:
            raise RuntimeError("ScriptedModel has no replies left")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(generations=[ChatGeneration(message=reply)])


def tool_call(name: str, call_id: str = "call-1", **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])
