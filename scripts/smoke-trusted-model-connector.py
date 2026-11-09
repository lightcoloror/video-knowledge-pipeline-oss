from __future__ import annotations

import asyncio
import json
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def _run() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    launcher = root / "scripts" / "start-trusted-model-connector-with-env.ps1"
    parameters = StdioServerParameters(
        command="powershell.exe",
        args=["-NoProfile", "-File", str(launcher)],
        cwd=str(root),
    )
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            listing = await session.list_tools()
            names = sorted(tool.name for tool in listing.tools)
            capabilities = await session.call_tool("model_connector_capabilities", {})
    expected = {
        "model_connector_capabilities",
        "model_connector_consent_status",
        "execute_consented_model_task_tool",
        "execute_consented_semantic_vision",
        "execute_consented_temporal_vision",
    }
    forbidden = {name for name in names if "create" in name or "confirm" in name}
    return {
        "ok": expected.issubset(set(names)) and not forbidden and not capabilities.isError,
        "tools": names,
        "missing_tools": sorted(expected - set(names)),
        "forbidden_tools": sorted(forbidden),
        "capabilities_call_error": bool(capabilities.isError),
    }


def main() -> int:
    result = asyncio.run(_run())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
