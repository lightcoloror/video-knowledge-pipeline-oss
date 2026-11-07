from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .storage import write_json


EXTRACTOR_PROBE_SCHEMA = "lecture_extractor_output_probe.v1"


def detect_extractor_output(
    paths: list[str | Path],
    *,
    project: str | Path | None = None,
    topic: str = "课程视频",
    title: str | None = None,
    webui_output_dir: str | Path | None = None,
    output_json: str | Path | None = None,
    handoff_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Detect vidclaude/peepshow/vidwise output folders and suggest import commands."""
    candidates = [_detect_one(Path(path).expanduser().resolve(), project=project, topic=topic) for path in paths]
    result = {
        "schema": EXTRACTOR_PROBE_SCHEMA,
        "count": len(candidates),
        "importable_count": sum(1 for item in candidates if item.get("importable")),
        "candidates": candidates,
        "next": _next_candidate(candidates),
    }
    pipeline = _pipeline_handoff(candidates, project=project, topic=topic, title=title, webui_output_dir=webui_output_dir)
    if pipeline:
        result["recommended_pipeline"] = pipeline
    if handoff_dir:
        _write_handoff_files(result, Path(handoff_dir).expanduser().resolve())
    if output_json:
        out = Path(output_json).expanduser().resolve()
        write_json(out, result)
        result["output_json"] = str(out)
    return result


def _detect_one(path: Path, *, project: str | Path | None, topic: str) -> dict[str, Any]:
    if not path.exists():
        return _candidate(path, kind="unknown", confidence=0, evidence=[], missing=["path"], project=project, topic=topic)
    checks = [_peepshow(path), _vidclaude(path), _vidwise(path)]
    best = sorted(checks, key=lambda item: (item["confidence"], item["kind"]), reverse=True)[0]
    return _candidate(
        path,
        kind=best["kind"],
        confidence=best["confidence"],
        evidence=best["evidence"],
        missing=best["missing"],
        project=project,
        topic=topic,
    )


def _candidate(
    path: Path,
    *,
    kind: str,
    confidence: int,
    evidence: list[str],
    missing: list[str],
    project: str | Path | None,
    topic: str,
) -> dict[str, Any]:
    importable = kind != "unknown" and not missing
    command = _import_command(kind, path, project=project, topic=topic) if importable and project else ""
    return {
        "path": str(path),
        "exists": path.exists(),
        "kind": kind,
        "confidence": confidence,
        "importable": importable,
        "evidence": evidence,
        "missing": missing,
        "command": command,
        "reason": _reason(kind, importable, missing),
    }


def _peepshow(path: Path) -> dict[str, Any]:
    evidence: list[str] = []
    missing: list[str] = []
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        evidence.append("manifest.json")
    else:
        missing.append("manifest.json")
    data = _read_json_object(manifest_path)
    if isinstance(data.get("frames"), list):
        evidence.append("manifest.frames")
    else:
        missing.append("manifest.frames")
    if isinstance(data.get("video"), dict):
        evidence.append("manifest.video")
    if (path / "report.html").exists():
        evidence.append("report.html")
    return {"kind": "peepshow", "confidence": len(evidence) * 30, "evidence": evidence, "missing": missing}


def _vidclaude(path: Path) -> dict[str, Any]:
    evidence: list[str] = []
    missing: list[str] = []
    if (path / "meta.json").exists():
        evidence.append("meta.json")
    else:
        missing.append("meta.json")
    if (path / "transcript.json").exists():
        evidence.append("transcript.json")
    if (path / "timeline.json").exists():
        evidence.append("timeline.json")
    frames = list((path / "frames").glob("*.jpg")) if (path / "frames").exists() else []
    if frames:
        evidence.append("frames/*.jpg")
    elif (path / "evidence.md").exists():
        evidence.append("evidence.md")
    else:
        missing.append("frames/*.jpg or evidence.md")
    return {"kind": "vidclaude", "confidence": len(evidence) * 25, "evidence": evidence, "missing": missing}


def _vidwise(path: Path) -> dict[str, Any]:
    evidence: list[str] = []
    missing: list[str] = []
    if (path / "video.mp4").exists():
        evidence.append("video.mp4")
    else:
        missing.append("video.mp4")
    if (path / "transcript.json").exists():
        evidence.append("transcript.json")
    frames = list((path / "frames").glob("*.png")) if (path / "frames").exists() else []
    if frames:
        evidence.append("frames/*.png")
    else:
        missing.append("frames/*.png")
    return {"kind": "vidwise", "confidence": len(evidence) * 30, "evidence": evidence, "missing": missing}


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _import_command(kind: str, path: Path, *, project: str | Path | None, topic: str) -> str:
    if kind == "peepshow":
        return f'.\\scripts\\video-knowledge.ps1 import-peepshow "{project}" "{path}" --topic "{topic}"'
    if kind == "vidclaude":
        return f'.\\scripts\\video-knowledge.ps1 import-vidclaude "{project}" "{path}" --topic "{topic}"'
    if kind == "vidwise":
        return f'.\\scripts\\video-knowledge.ps1 import-vidwise "{project}" "{path}" --topic "{topic}"'
    return ""


def _reason(kind: str, importable: bool, missing: list[str]) -> str:
    if kind == "unknown":
        return "No known extractor output signature was found."
    if importable:
        return f"Looks like a {kind} output folder and has the minimum files needed by the importer."
    return f"Looks like {kind}, but missing: {', '.join(missing)}"


def _next_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    importable = [item for item in candidates if item.get("importable")]
    if importable:
        return {"status": "ready_to_import", "candidate": importable[0]}
    known = [item for item in candidates if item.get("kind") != "unknown"]
    if known:
        return {"status": "missing_files", "candidate": known[0]}
    return {"status": "no_known_output", "candidate": candidates[0] if candidates else {}}


def _pipeline_handoff(
    candidates: list[dict[str, Any]],
    *,
    project: str | Path | None,
    topic: str,
    title: str | None,
    webui_output_dir: str | Path | None,
) -> dict[str, Any]:
    if not project or not title:
        return {}
    paths = [item["path"] for item in candidates if item.get("importable")]
    if not paths:
        return {}
    command_parts = [
        ".\\scripts\\video-knowledge.ps1",
        "run-detected-lecture-pipeline",
        _quote(project),
        *[_quote(path) for path in paths],
        "--title",
        _quote(title),
        "--topic",
        _quote(topic),
    ]
    args: dict[str, Any] = {
        "project": str(Path(project).expanduser()),
        "output_paths": paths,
        "title": title,
        "topic": topic,
    }
    if webui_output_dir:
        command_parts.extend(["--webui-output-dir", _quote(webui_output_dir)])
        args["webui_output_dir"] = str(Path(webui_output_dir).expanduser())
    return {
        "status": "ready_to_run" if paths else "not_ready",
        "mcp_tool": "run_detected_lecture_pipeline",
        "mcp_args": args,
        "command": " ".join(command_parts),
    }


def _write_handoff_files(result: dict[str, Any], handoff_dir: Path) -> None:
    handoff_dir.mkdir(parents=True, exist_ok=True)
    json_path = handoff_dir / "extractor-output-probe.json"
    markdown_path = handoff_dir / "extractor-output-probe.md"
    result["handoff_dir"] = str(handoff_dir)
    result["handoff_json_path"] = str(json_path)
    result["handoff_markdown_path"] = str(markdown_path)
    pipeline = result.get("recommended_pipeline") if isinstance(result.get("recommended_pipeline"), dict) else {}
    args = pipeline.get("mcp_args") if isinstance(pipeline.get("mcp_args"), dict) else {}
    if args:
        args_path = handoff_dir / "mcp-run-detected-lecture-pipeline.args.json"
        write_json(args_path, args)
        result["recommended_pipeline"]["mcp_args_path"] = str(args_path)
    write_json(json_path, result)
    markdown_path.write_text(render_extractor_probe_markdown(result), encoding="utf-8")


def render_extractor_probe_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Extractor Output Probe",
        "",
        f"- Schema: `{result.get('schema', '')}`",
        f"- Candidates: `{result.get('count', 0)}`",
        f"- Importable: `{result.get('importable_count', 0)}`",
        f"- Next: `{(result.get('next') or {}).get('status', '')}`",
        "",
        "## Candidates",
        "",
    ]
    for item in result.get("candidates") or []:
        if not isinstance(item, dict):
            continue
        lines.extend(
            [
                f"### {item.get('kind', 'unknown')}",
                "",
                f"- Path: `{item.get('path', '')}`",
                f"- Importable: `{item.get('importable', False)}`",
                f"- Confidence: `{item.get('confidence', 0)}`",
                f"- Evidence: {', '.join(item.get('evidence') or []) or 'none'}",
                f"- Missing: {', '.join(item.get('missing') or []) or 'none'}",
                f"- Reason: {item.get('reason', '')}",
            ]
        )
        if item.get("command"):
            lines.extend(["", "```powershell", str(item.get("command")), "```"])
        lines.append("")
    pipeline = result.get("recommended_pipeline") if isinstance(result.get("recommended_pipeline"), dict) else {}
    if pipeline:
        lines.extend(
            [
                "## Recommended Pipeline",
                "",
                f"- MCP tool: `{pipeline.get('mcp_tool', '')}`",
                f"- MCP args: `{pipeline.get('mcp_args_path', '')}`",
                "",
                "```powershell",
                str(pipeline.get("command") or ""),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _quote(value: str | Path) -> str:
    text = str(value)
    return '"' + text.replace('"', '\\"') + '"'

