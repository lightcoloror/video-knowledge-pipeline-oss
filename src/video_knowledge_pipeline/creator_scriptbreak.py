"""Project VKP candidates into the existing ScriptBreak project/MCP contract."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .file_hash import sha256_file


UPSTREAM_SHA = "c457f02ec2f0f34bb31af5289af90dd9216297b5"
REQUIRED_BRIEF = ("product", "audience", "goal", "resources")


def validate_brief(value: dict[str, Any] | None) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or any(not isinstance(value.get(k), str) or not value[k].strip() for k in REQUIRED_BRIEF):
        raise ValueError("creative brief requires nonempty product, audience, goal and resources strings")
    if any(len(value[k]) > 4000 for k in REQUIRED_BRIEF):
        raise ValueError("creative brief fields must not exceed 4000 characters")
    return {k: value[k].strip() for k in REQUIRED_BRIEF}


def project_candidate(report: dict[str, Any]) -> dict[str, Any]:
    brief = validate_brief(report.get("brief") or None)
    if not brief:
        raise ValueError("provide a creative brief before exporting a tailored ScriptBreak candidate")
    scenes = []
    for shot in report["shots"]:
        facts = shot["facts"]
        scenes.append({"num": shot["index"], "slug": f"参考段落 {shot['shot_id']}",
                       "intExt": "", "location": "", "tod": "", "eighths": 1,
                       "synopsis": f"参考时段 {shot['start_time']}–{shot['end_time']}；原片未知字段保持未知。",
                       "text": "原片文字（仅供结构参考）：" + str(facts.get("dialogue_or_narration") or "未知"),
                       "characters": [], "elements": {},
                       "shots": [{"num": str(shot["index"]) + "A", "size": "", "angle": "", "move": "", "lens": "",
                                  "desc": "原片证据（未知项不得当事实补齐）：" + json.dumps(facts, ensure_ascii=False)
                                  + "。为新产品创作建议；建议不是原片事实。需求：" + json.dumps(brief, ensure_ascii=False)}]})
    if not scenes:
        raise ValueError("verified technical shots are required for a ScriptBreak candidate")
    return {"app": "scriptbreak", "version": 2, "project": report["title"],
            "state": {"title": report["title"], "look": {"aspect": "9:16", "tone": "创作候选，待复核"},
                      "bibles": {"characters": {}, "locations": {}, "props": {}}, "scenes": scenes},
            "vkp_provenance": {"candidate_only": True, "publication_allowed": False,
                               "input_fingerprint": report["input_fingerprint"], "brief": brief,
                               "upstream_commit": UPSTREAM_SHA,
                               "warning": "Template defaults/completions are creative suggestions, never observations of the source."}}


def export_prompt_pack(project_path: Path, script_path: Path, *, expected_sha256: str, node: str = "node") -> dict[str, Any]:
    """Use the reviewed original stdio server, with an operator-pinned script hash.

    The caller supplies a generated service-owned project, never an end-user path.
    This calls templates only; it does not contact a model or render a video.
    """
    actual = sha256_file(script_path)
    if not expected_sha256 or actual.lower() != expected_sha256.lower():
        raise ValueError("ScriptBreak source hash mismatch; review and pin the original script first")
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "VKP", "version": "1"}}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "export_prompt_pack", "arguments": {"projectPath": str(project_path.resolve()), "generator": "generic"}}},
    ]
    process = subprocess.run([node, str(script_path.resolve())],
                             input="\n".join(json.dumps(row) for row in requests) + "\n",
                             encoding="utf-8", capture_output=True, timeout=30, check=False)
    if process.returncode:
        raise RuntimeError(f"ScriptBreak exited {process.returncode}: {process.stderr[:400]}")
    responses = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    response = next((row for row in responses if row.get("id") == 2), {})
    result = response.get("result") or {}
    if not result or response.get("error") or result.get("isError"):
        raise RuntimeError("ScriptBreak did not return a successful prompt pack")
    text = "\n".join(row["text"] for row in result.get("content", []) if row.get("type") == "text")
    if not text.strip():
        raise RuntimeError("ScriptBreak returned an empty prompt pack")
    return {"text": "# 创作提示词候选\n\n原生模板可能补充未知场景、机位与人物；这些是创作建议，不能回写为原片事实。\n\n" + text,
            "script_sha256": actual, "model_calls_made": 0, "candidate_only": True}
