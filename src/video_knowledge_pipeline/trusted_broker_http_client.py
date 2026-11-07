from __future__ import annotations

import asyncio
import json
import urllib.parse
from typing import Any

from .media_async_client import _is_loopback_base_url


def require_loopback_mcp_url(value: str) -> str:
    """Validate the shared Streamable HTTP Broker boundary and return a clean URL."""

    clean = str(value or "").strip()
    if not _is_loopback_base_url(clean):
        raise ValueError("Trusted Broker only permits an explicit loopback HTTP MCP URL")
    parsed = urllib.parse.urlsplit(clean)
    if parsed.username or parsed.password:
        raise ValueError("Trusted Broker URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("Trusted Broker URL must not contain query or fragment data")
    if not clean.rstrip("/").endswith("/mcp"):
        raise ValueError("Trusted Broker URL must end with /mcp")
    return clean


def call_loopback_broker_tool(
    broker_url: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Call one existing Broker MCP tool; provider execution remains Broker-owned."""

    url = require_loopback_mcp_url(broker_url)
    name = str(tool_name or "").strip()
    if not name:
        raise ValueError("Broker tool_name is required")
    return asyncio.run(_call_broker_tool(url, name, dict(arguments)))


async def _call_broker_tool(
    broker_url: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise RuntimeError("Install VKP MCP support with: pip install -e .[mcp]") from exc

    async with streamablehttp_client(broker_url) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            response = await session.call_tool(tool_name, arguments)
    return decode_broker_tool_response(response)


def decode_broker_tool_response(response: Any) -> dict[str, Any]:
    structured = getattr(response, "structuredContent", None)
    if not isinstance(structured, dict):
        structured = getattr(response, "structured_content", None)
    if isinstance(structured, dict):
        return structured

    text_blocks = [
        text.strip()
        for block in getattr(response, "content", None) or []
        if isinstance((text := getattr(block, "text", None)), str) and text.strip()
    ]
    for text in text_blocks:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    if bool(getattr(response, "isError", False)):
        raise RuntimeError("Trusted Broker rejected the MCP tool call")
    raise ValueError("Trusted Broker returned no machine-readable JSON response")
