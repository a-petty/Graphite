"""Guards the PR 7 tool-surface contract.

The MCP server exposes exactly 8 tools. Any PR that adds or drops a tool
without updating this test is almost certainly expanding the surface Claude
must reason over when choosing tools — which is the exact thing we
consolidated away from in PR 7.
"""

import re
from pathlib import Path


MCP_SERVER = Path(__file__).parent.parent / "graphite" / "mcp_server.py"

EXPECTED_TOOLS = {
    "graphite_status",
    "recall",
    "entity",
    "timeline",
    "remember",
    "ingest_source",
    "register_source",
    "reflect",
}


def _discover_tools() -> set[str]:
    """Parse mcp_server.py for @mcp.tool() decorators and return the set of
    decorated function names. Uses regex rather than importing the module so
    test failures give a readable signal even if the module fails to import.
    """
    src = MCP_SERVER.read_text()
    # `@mcp.tool()` on its own line, followed by `async def NAME(` on the next
    pattern = re.compile(r"^@mcp\.tool\(\)\s*\n\s*async\s+def\s+(\w+)\s*\(", re.MULTILINE)
    return set(pattern.findall(src))


def test_exactly_eight_tools_decorated():
    tools = _discover_tools()
    assert len(tools) == 8, f"expected 8 @mcp.tool() functions, found {len(tools)}: {sorted(tools)}"


def test_expected_tool_names():
    tools = _discover_tools()
    assert tools == EXPECTED_TOOLS, (
        f"surface drift:\n"
        f"  unexpected additions: {sorted(tools - EXPECTED_TOOLS)}\n"
        f"  missing: {sorted(EXPECTED_TOOLS - tools)}"
    )


def test_legacy_tools_still_callable_as_internal_functions():
    """Removing the @mcp.tool() decorator must NOT delete the function body —
    internal callers (including the 8 new tools and existing tests) still
    depend on these helpers.
    """
    import graphite.mcp_server as srv

    for internal in (
        "get_entity_profile",
        "get_timeline",
        "find_relevant_entities",
        "assemble_memory",
        "graphite_ingest_text",
        "graphite_reflect",
        "graphite_ingest",
        "graphite_ingest_sessions",
    ):
        assert hasattr(srv, internal), f"internal helper {internal} disappeared"
