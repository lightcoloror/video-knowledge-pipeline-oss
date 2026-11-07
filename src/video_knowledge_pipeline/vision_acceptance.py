from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import resolve_vision_execution_profile
from .model_defaults import GEMINI_DEFAULT_MODEL
from .storage import read_json, write_json
from .vision_api import resolve_provider_config


def vision_acceptance_plan(
    bundle_dir: str | Path,
    *,
    provider_config: dict[str, Any] | None = None,
    semantic_limit: int | None = None,
    temporal_limit: int | None = None,
    frame_count: int | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Prepare a no-secret runbook for the first real multimodal API acceptance run."""
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    timeline_path = root / "timeline.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"timeline not found: {timeline_path}")
    manifest = read_json(manifest_path)
    timeline_data = read_json(timeline_path)
    timeline = [item for item in timeline_data if isinstance(item, dict)] if isinstance(timeline_data, list) else []
    profile = resolve_vision_execution_profile(
        provider_config=provider_config,
        multimodal_limit=semantic_limit,
        temporal_limit=temporal_limit,
        frame_count=frame_count,
    )
    semantic_limit_value = int(profile["multimodal_limit"])
    temporal_limit_value = int(profile["temporal_limit"])
    frame_count_value = int(profile["frame_count"])
    cfg = resolve_provider_config(profile["provider_config"])

    semantic_candidates = [
        index
        for index, item in enumerate(timeline, start=1)
        if str(item.get("visual_route") or "") in {"semantic_frame", "mixed"}
        and _has_frames(item)
        and not _has_mapping(item.get("visual_understanding"))
    ]
    temporal_candidates = [
        index
        for index, item in enumerate(timeline, start=1)
        if str(item.get("visual_route") or "") in {"temporal_sequence", "mixed"}
        and _has_list(item.get("temporal_frame_paths"))
        and not _has_mapping(item.get("temporal_visual_understanding"))
    ]
    missing_temporal_groups = [
        index
        for index, item in enumerate(timeline, start=1)
        if str(item.get("visual_route") or "") in {"temporal_sequence", "mixed"} and not _has_list(item.get("temporal_frame_paths"))
    ]

    semantic_selected = semantic_candidates[: max(0, semantic_limit_value)]
    temporal_selected = temporal_candidates[: max(0, temporal_limit_value)]
    ready = bool(cfg.get("api_key")) and len(semantic_selected) >= min(max(0, semantic_limit_value), len(semantic_candidates)) and len(temporal_selected) >= min(max(0, temporal_limit_value), len(temporal_candidates))

    runbook = {
        "schema": "lecture_vision_acceptance_plan.v1",
        "bundle_dir": str(root),
        "manifest_path": str(manifest_path),
        "provider": {
            "provider": cfg.get("provider"),
            "base_url": cfg.get("base_url"),
            "model": cfg.get("model"),
            "api_key_configured": bool(cfg.get("api_key")),
            "timeout_seconds": cfg.get("timeout_seconds"),
        },
        "targets": {
            "semantic_limit": semantic_limit_value,
            "temporal_limit": temporal_limit_value,
            "frame_count": frame_count_value,
        },
        "candidate_counts": {
            "semantic_available": len(semantic_candidates),
            "semantic_selected": len(semantic_selected),
            "temporal_available": len(temporal_candidates),
            "temporal_selected": len(temporal_selected),
            "temporal_without_frame_groups": len(missing_temporal_groups),
        },
        "selected_indexes": {
            "semantic": semantic_selected,
            "temporal": temporal_selected,
            "temporal_missing_groups_sample": missing_temporal_groups[:10],
        },
        "ready_to_execute": ready,
        "blockers": _blockers(cfg, semantic_selected, temporal_selected, missing_temporal_groups, semantic_limit_value, temporal_limit_value),
        "commands": _commands(
            root,
            cfg,
            semantic_limit=semantic_limit_value,
            temporal_limit=temporal_limit_value,
            frame_count=frame_count_value,
            semantic_selected=semantic_selected,
            temporal_selected=temporal_selected,
        ),
        "artifacts": {
            "semantic_report": str(root / "multimodal-frame-analysis-report.md"),
            "temporal_groups_report": str(root / "temporal-frame-groups-report.md"),
            "temporal_report": str(root / "temporal-visual-analysis-report.md"),
            "knowledge_note": str(root / "exports" / "knowledge-note.md"),
        },
    }
    if write:
        json_path = root / "vision-acceptance-plan.json"
        markdown_path = root / "vision-acceptance-plan.md"
        write_json(json_path, runbook)
        markdown_path.write_text(render_vision_acceptance_plan_markdown(runbook), encoding="utf-8")
        if isinstance(manifest, dict):
            manifest["vision_acceptance_plan"] = "vision-acceptance-plan.md"
            manifest["vision_acceptance_plan_json"] = "vision-acceptance-plan.json"
            manifest["mcp_vision_acceptance_plan_args"] = "mcp-vision-acceptance-plan.args.json"
            write_json(root / "mcp-vision-acceptance-plan.args.json", {"bundle_dir": str(root), "semantic_limit": semantic_limit_value, "temporal_limit": temporal_limit_value, "frame_count": frame_count_value, "write": True})
            write_json(manifest_path, manifest)
        runbook["plan_path"] = str(markdown_path)
        runbook["plan_json_path"] = str(json_path)
    return runbook


def render_vision_acceptance_plan_markdown(plan: dict[str, Any]) -> str:
    provider = plan.get("provider") if isinstance(plan.get("provider"), dict) else {}
    counts = plan.get("candidate_counts") if isinstance(plan.get("candidate_counts"), dict) else {}
    selected = plan.get("selected_indexes") if isinstance(plan.get("selected_indexes"), dict) else {}
    lines = [
        "# 多模态真实 API 验收计划",
        "",
        f"- Bundle: `{plan.get('bundle_dir', '')}`",
        f"- Provider: `{provider.get('provider', '')}` / `{provider.get('model', '')}`",
        f"- API key configured: `{provider.get('api_key_configured', False)}`",
        f"- Ready to execute: `{plan.get('ready_to_execute', False)}`",
        "",
        "## 候选规模",
        "",
        f"- 单帧可用/选中：`{counts.get('semantic_available', 0)}` / `{counts.get('semantic_selected', 0)}`",
        f"- 连续片段可用/选中：`{counts.get('temporal_available', 0)}` / `{counts.get('temporal_selected', 0)}`",
        f"- 缺连续帧组：`{counts.get('temporal_without_frame_groups', 0)}`",
        f"- 单帧 index：`{selected.get('semantic', [])}`",
        f"- 连续片段 index：`{selected.get('temporal', [])}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = plan.get("blockers") if isinstance(plan.get("blockers"), list) else []
    lines.extend([f"- `{item.get('key', '')}`: {item.get('message', '')}" for item in blockers if isinstance(item, dict)] or ["- 无"])
    lines.extend(["", "## Commands", ""])
    commands = plan.get("commands") if isinstance(plan.get("commands"), dict) else {}
    for key, command in commands.items():
        lines.extend([f"### {key}", "", "```powershell", str(command), "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def _blockers(
    cfg: dict[str, Any],
    semantic_selected: list[int],
    temporal_selected: list[int],
    missing_temporal_groups: list[int],
    semantic_limit: int,
    temporal_limit: int,
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    if not cfg.get("api_key"):
        blockers.append({"key": "missing_api_key", "message": "配置 GEMINI_API_KEY / OPENAI_API_KEY / AGNES_API_KEY 或 LECTURE_VISION_API_KEY 后才能执行真实 API。"})
    if len(semantic_selected) < int(semantic_limit or 0):
        blockers.append({"key": "semantic_candidates_below_target", "message": f"单帧候选只选到 {len(semantic_selected)} 个，目标是 {int(semantic_limit or 0)} 个。"})
    if len(temporal_selected) < int(temporal_limit or 0):
        blockers.append({"key": "temporal_candidates_below_target", "message": f"连续片段候选只选到 {len(temporal_selected)} 个，目标是 {int(temporal_limit or 0)} 个。"})
    if missing_temporal_groups:
        blockers.append({"key": "temporal_frame_groups_missing", "message": f"还有 {len(missing_temporal_groups)} 个连续片段缺少 5-12 帧组；验收先跑已抽好的小批。"})
    return blockers


def _commands(
    root: Path,
    cfg: dict[str, Any],
    *,
    semantic_limit: int,
    temporal_limit: int,
    frame_count: int,
    semantic_selected: list[int],
    temporal_selected: list[int],
) -> dict[str, str]:
    provider = str(cfg.get("provider") or "gemini")
    model = str(cfg.get("model") or GEMINI_DEFAULT_MODEL)
    provider_json = f'{{"provider":"{provider}","model":"{model}"}}'
    semantic_indexes = ",".join(str(index) for index in semantic_selected)
    temporal_indexes = ",".join(str(index) for index in temporal_selected)
    return {
        "set_gemini_env_example": f'$env:LECTURE_VISION_PROVIDER="gemini"; $env:GEMINI_API_KEY="<your key>"; $env:LECTURE_VISION_MODEL="{GEMINI_DEFAULT_MODEL}"',
        "set_agnes_env_example": '$env:LECTURE_VISION_PROVIDER="agnes"; $env:AGNES_API_KEY="<your key>"; $env:LECTURE_VISION_MODEL="agnes-1.5-flash"',
        "set_openai_env_example": '$env:LECTURE_VISION_PROVIDER="openai"; $env:OPENAI_API_KEY="<your key>"; $env:LECTURE_VISION_MODEL="gpt-4o-mini"',
        "test_provider": f".\\scripts\\video-knowledge.ps1 test-vision-provider --provider-config '{provider_json}' --image-paths \"{root / 'assets' / '0001-001_0000000000ms.jpg'}\"",
        "run_semantic_acceptance": (
            f".\\scripts\\video-knowledge.ps1 run-multimodal-frame-analysis \"{root}\" --execute --limit {semantic_limit} "
            f"--indexes {semantic_indexes} --confirm-vision-calls {len(semantic_selected)} --confirm-vision-indexes {semantic_indexes} "
            f"--provider-config '{provider_json}'"
        ),
        "run_temporal_acceptance": (
            f".\\scripts\\video-knowledge.ps1 run-temporal-visual-analysis \"{root}\" --execute --frame-count {frame_count} --limit {temporal_limit} "
            f"--indexes {temporal_indexes} --confirm-vision-calls {len(temporal_selected)} --confirm-vision-indexes {temporal_indexes} "
            f"--provider-config '{provider_json}'"
        ),
        "export_note": f".\\scripts\\video-knowledge.ps1 export-knowledge-note \"{root}\"",
    }


def _has_mapping(value: object) -> bool:
    return isinstance(value, dict) and any(item not in (None, "", [], {}) for item in value.values())


def _has_list(value: object) -> bool:
    return isinstance(value, list) and any(item not in (None, "", [], {}) for item in value)


def _has_frames(item: dict[str, Any]) -> bool:
    return _has_list(item.get("frame_paths")) or _has_list(item.get("assets"))
