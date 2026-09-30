"""A tiny MCP server (stdio) for the tests: run by `test_mcp.py` as a subprocess."""

from mcp.server.fastmcp import FastMCP

server = FastMCP("glossary")


@server.tool()
def define(term: str) -> str:
    """Return the glossary definition of a term."""
    return {"rrf": "Reciprocal Rank Fusion"}.get(term.lower(), f"no definition for {term}")


if __name__ == "__main__":
    server.run("stdio")
