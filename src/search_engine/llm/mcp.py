"""MCP: the tools of external MCP servers, as LangChain tools the LLM can call next to the local TOOLS.

Servers are listed in `MCP_SERVERS` (.env), in `langchain-mcp-adapters` format, e.g.
    MCP_SERVERS='{"files": {"transport": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "docs"]},
                  "wiki":  {"transport": "streamable_http", "url": "http://localhost:8000/mcp"}}'
With no servers configured, the LLM has the local tools only.
"""

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from search_engine.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


def load_mcp_tools(settings: Settings | None = None) -> list[BaseTool]:
    """Connects to every configured MCP server and returns their tools. A server that can't be reached is
    logged and skipped, so one broken server doesn't take search down."""
    settings = settings or get_settings()
    tools: list[BaseTool] = []
    for name, connection in settings.mcp_servers.items():
        try:
            # tool_name_prefix: "<server>_<tool>", so two servers (or a local tool) can't clash on a name
            client = MultiServerMCPClient({name: connection}, tool_name_prefix=True)
            server_tools = run_sync(client.get_tools())
        except Exception as error:
            logger.warning("MCP server %r is not available, skipping it: %s", name, str(error)[:200])
            continue
        logger.info("MCP server %r: %d tools (%s)", name, len(server_tools), ", ".join(t.name for t in server_tools))
        tools.extend(server_tools)
    return tools


def run_sync(coroutine: Coroutine[Any, Any, Any]) -> Any:
    """Runs async code from our synchronous pipeline. MCP tools are async-only, so the agent runs through this."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    # Already inside an event loop (e.g. called from async code): run on a separate thread's loop.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()
