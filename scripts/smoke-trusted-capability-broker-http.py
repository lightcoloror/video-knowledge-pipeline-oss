from __future__ import annotations

import argparse
import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def run(url: str) -> dict[str, object]:
    async with streamablehttp_client(url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = sorted(tool.name for tool in tools.tools)
            capability = await session.call_tool("model_connector_capabilities", {})
            return {
                "status": "passed",
                "url": url,
                "tool_count": len(names),
                "tools": names,
                "capability_content_blocks": len(capability.content),
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the local VKP capability broker")
    parser.add_argument("--url", default="http://127.0.0.1:8766/mcp")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.url)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
