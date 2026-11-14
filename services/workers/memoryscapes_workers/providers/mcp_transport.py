"""Minimal synchronous MCP client for calling SATIE's tools from a worker."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
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

    ``token`` is a fixed bearer token or a function returning a current one (the saved
    worker login, see ``satie_auth``), asked for before every call so a refreshed token
    is picked up.
    """

    def __init__(self, url: str, token: str | Callable[[], str] | None = None, timeout_s: float = 300.0) -> None:
        self.url = url
        self.token = token
        self.timeout_s = timeout_s

    @property
    def headers(self) -> dict[str, str]:
        token = self.token() if callable(self.token) else self.token
        return {"Authorization": f"Bearer {token}"} if token else {}

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        headers = self.headers  # a sign-in problem surfaces here as SatieAuthError
        try:
            return asyncio.run(self._call(name, arguments, headers))
        except McpError:
            raise
        except Exception as exc:  # the SDK wraps failures in exception groups
            leaf = _first_leaf(exc)
            hint = "" if headers else " (not signed in: run memoryscapes-worker satie-login, or set SATIE_MCP_TOKEN)"
            raise McpError(f"SATIE {name} failed: {type(leaf).__name__}: {leaf}{hint}") from exc

    async def _call(self, name: str, arguments: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        import contextlib
        from datetime import timedelta

        import mcp.client.streamable_http as sh
        from mcp import ClientSession

        async with contextlib.AsyncExitStack() as stack:
            if _sdk_major() >= 2:  # headers ride on the HTTP client; timeouts are floats
                client = await stack.enter_async_context(sh.create_mcp_http_client(headers=headers))
                streams = await stack.enter_async_context(sh.streamable_http_client(self.url, http_client=client))
                timeout: Any = self.timeout_s
            else:  # mcp 1.x: timedelta timeouts
                streams = await stack.enter_async_context(sh.streamablehttp_client(self.url, headers=headers))
                timeout = timedelta(seconds=self.timeout_s)
            session = await stack.enter_async_context(
                ClientSession(streams[0], streams[1], read_timeout_seconds=timeout))
            await session.initialize()
            result = await session.call_tool(name, arguments)
        return _result_dict(result)


def _sdk_major() -> int:
    from importlib.metadata import version

    return int(version("mcp").split(".")[0])


def _first_leaf(exc: BaseException) -> BaseException:
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def _result_dict(result: Any) -> dict[str, Any]:
    """CallToolResult from either SDK version (2.x renamed the fields to snake_case)."""
    texts = [c.text for c in (getattr(result, "content", None) or []) if getattr(c, "type", None) == "text"]
    structured = getattr(result, "structured_content", None)
    if structured is None:
        structured = getattr(result, "structuredContent", None)
    is_error = getattr(result, "is_error", None)
    if is_error is None:
        is_error = getattr(result, "isError", False)
    return parse_tool_result(structured, texts, bool(is_error))
