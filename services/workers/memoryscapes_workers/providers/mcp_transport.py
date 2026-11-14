"""Minimal synchronous MCP client for calling SATIE's tools from a worker."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Protocol


class McpError(RuntimeError):
    pass


class McpTransport(Protocol):
    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


def parse_tool_result(structured: Any, texts: list[str], is_error: bool) -> dict[str, Any]:
    """Prefer structured content; else parse the text as JSON; else return it raw."""
    text = "\n".join(texts)
    if is_error:
        raise McpError(text or "MCP tool returned an error")
    if isinstance(structured, dict):
        out = dict(structured)
        out.setdefault("text", text)
        return out
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {"value": value, "text": text}
    except (json.JSONDecodeError, ValueError):
        return {"text": text}


class HttpMcpTransport:
    """Streamable-HTTP MCP client (official `mcp` Python SDK, in the `satie` extra).

    SATIE signs users in with OAuth in interactive clients. A headless worker needs a
    bearer token for the account that pays for takes; how to mint one for server use is
    not documented yet, so confirm with Attractor Labs before deploying.
    """

    def __init__(self, url: str, token: str | None = None, timeout_s: float = 300.0) -> None:
        self.url = url
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.timeout_s = timeout_s

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(self._call(name, arguments))

    async def _call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        from datetime import timedelta

        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(self.url, headers=self.headers) as (read, write, _):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=self.timeout_s)) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
        texts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
        return parse_tool_result(getattr(result, "structuredContent", None), texts, bool(result.isError))
