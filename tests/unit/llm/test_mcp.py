"""MCP against a real (tiny) MCP server started as a subprocess over stdio."""

import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage

from search_engine.core.config import Settings
from search_engine.llm.agent import LLMAgent
from search_engine.llm.mcp import load_mcp_tools, run_sync
from tests.unit.llm.fakes import ScriptedModel, tool_call

SERVER = {"transport": "stdio", "command": sys.executable, "args": [str(Path(__file__).with_name("mcp_test_server.py"))]}


def test_no_servers_configured_means_no_mcp_tools():
    assert load_mcp_tools(Settings(mcp_servers={})) == []


def test_tools_of_a_configured_server_are_loaded_and_callable():
    [tool] = load_mcp_tools(Settings(mcp_servers={"glossary": SERVER}))

    assert tool.name == "glossary_define" and "definition" in tool.description
    reply = run_sync(tool.ainvoke({"term": "RRF"}))
    assert "Reciprocal Rank Fusion" in str(reply)


def test_a_server_that_cannot_start_is_skipped_and_the_others_still_load(caplog):
    broken = {"transport": "stdio", "command": sys.executable, "args": ["no_such_server.py"]}

    tools = load_mcp_tools(Settings(mcp_servers={"broken": broken, "glossary": SERVER}))

    assert [tool.name for tool in tools] == ["glossary_define"]
    assert "MCP server 'broken' is not available" in caplog.text


def test_the_llm_calls_an_mcp_servers_tool_and_uses_its_result():
    tools = load_mcp_tools(Settings(mcp_servers={"glossary": SERVER}))
    model = ScriptedModel(replies=[tool_call("glossary_define", term="rrf"), AIMessage("RRF is Reciprocal Rank Fusion.")])
    chunk = Document("RRF merges ranked lists.", metadata={"chunk_id": "c-0", "file_name": "notes.md"})

    reply = LLMAgent(Settings(), models=[model], mcp_tools=tools).answer("What does RRF stand for?", [chunk])

    assert reply.tools_used == ["glossary_define"] and reply.text == "RRF is Reciprocal Rank Fusion."
    [result] = [m for m in model.seen[1] if isinstance(m, ToolMessage)]
    assert "Reciprocal Rank Fusion" in str(result.content)
