from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .models import now_iso
from .storage import read_json, write_json


SCHEMA = "video_knowledge_pipeline.visual_ab_benchmark_plan.v1"
DEICTIC_TERMS = ("这里", "这个", "大家看", "看屏幕", "如图", "上面", "下面", "这张", "界面", "截图", "演示")
HARD_ISSUES = {
    "ocr_text_empty",
    "ocr_wrapper_only",
    "ocr_text_low_information",
    "screen_text_low_confidence",
    "missing_visual_text",
    "ebook_pipeline_failed",
    "semantic_frame_without_analysis",
    "temporal_sequence_without_analysis",
}


def build_visual_ab_benchmark_plan(
    bundle_dir: str | Path,
    *,
    limit: int = 10,
    min_score: int = 4,
    write: bool = True,
) -> dict[str, Any]:
    root = Path(bundle_dir).expanduser().resolve()
    timeline_path = root / "timeline.json"
    manifest_path = root / "manifest.json"
    if not timeline_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"bundle missing manifest.json or timeline.json: {root}")
    timeline = read_json(timeline_path)
    rows = timeline if isinstance(timeline, list) else []
    candidates = [_candidate(root, raw, position) for position, raw in enumerate(rows, start=1) if isinstance(raw, dict)]
    candidates = [row for row in candidates if row["score"] >= int(min_score) and row["frame_paths"]]
    selected = _stratified_select(candidates, max(1, int(limit)))
    semantic_indexes = [row["index"] for row in selected if row["analysis_kind"] == "semantic"]
    temporal_indexes = [row["index"] for row in selected if row["analysis_kind"] == "temporal"]
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "status": "ready" if selected else "no_candidates",
        "experiment": {
            "A": "corrected transcript only",
            "B": "corrected transcript plus local ebook/OCR evidence",
            "C": "B plus targeted online multimodal evidence for only the selected rows",
            "evaluation": [
                "key_point_recall",
                "proper_name_and_number_accuracy",
                "visual_only_information_recall",
                "unsupported_claim_count",
                "readability_and_actionability",
            ],
        },
        "selection": {
            "strategy": "score_then_uniform_full_duration_coverage",
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "limit": max(1, int(limit)),
            "min_score": int(min_score),
            "semantic_indexes": semantic_indexes,
            "temporal_indexes": temporal_indexes,
        },
        "items": selected,
        "operator_boundary": {
            "plan_only": True,
            "does_not_call_online_model": True,
            "selected_frames_only": True,
            "vision_export_consent_required_for_agent": True,
            "full_video_or_audio_export_allowed": False,
        },
        "next_actions": _next_actions(root, semantic_indexes, temporal_indexes),
        "artifacts": {
            "json": str(root / "visual-ab-benchmark-plan.json"),
            "markdown": str(root / "visual-ab-benchmark-plan.md"),
        },
        "updated_at": now_iso(),
    }
    if write:
        write_json(root / "visual-ab-benchmark-plan.json", result)
        (root / "visual-ab-benchmark-plan.md").write_text(_render_markdown(result), encoding="utf-8")
        manifest = read_json(manifest_path)
        manifest = manifest if isinstance(manifest, dict) else {}
        manifest["visual_ab_benchmark_plan_json"] = "visual-ab-benchmark-plan.json"
        manifest["visual_ab_benchmark_plan_markdown"] = "visual-ab-benchmark-plan.md"
        write_json(manifest_path, manifest)
    return result


def _candidate(root: Path, item: dict[str, Any], position: int) -> dict[str, Any]:
    index = int(item.get("index") or position)
    route = str(item.get("visual_route") or "unknown")
    transcript = str(item.get("corrected_text") or item.get("transcript") or item.get("text") or "").strip()
    visual_text = str(item.get("visual_text") or item.get("ocr_text") or "").strip()
    issues = sorted(set(_strings(item.get("quality_issues")) + _strings(item.get("issues"))))
    frames = _frame_paths(root, item)
    temporal_frames = _path_list(root, item.get("temporal_frame_paths"))
    score = 0
    reasons: list[str] = []
    if route in {"temporal_sequence", "mixed"}:
        score += 5
        reasons.append("continuous_change_or_mixed_route")
    if route == "semantic_frame":
        score += 3
        reasons.append("semantic_frame_route")
    hard = sorted(HARD_ISSUES.intersection(issues))
    if hard:
        score += min(6, 2 + len(hard))
        reasons.extend(hard)
    if any(term in transcript for term in DEICTIC_TERMS):
        score += 3
        reasons.append("speech_points_to_screen")
    if not visual_text:
        score += 2
        reasons.append("local_visual_text_missing")
    else:
        score += 1
        reasons.append("local_ebook_or_ocr_available_for_ab")
    if item.get("structured_visual"):
        score += 1
        reasons.append("structured_visual_available_for_ab")
    if re.search(r"\d", transcript) or re.search(r"[A-Za-z][A-Za-z0-9_.+-]{2,}", transcript):
        score += 1
        reasons.append("entity_or_number_risk")
    analysis_kind = "temporal" if temporal_frames and route in {"temporal_sequence", "mixed"} else "semantic"
    return {
        "index": index,
        "start": _number(item.get("start")),
        "end": _number(item.get("end")),
        "visual_route": route,
        "analysis_kind": analysis_kind,
        "score": score,
        "reasons": list(dict.fromkeys(reasons)),
        "quality_issues": issues,
        "transcript_excerpt": transcript[:600],
        "local_visual_text_excerpt": visual_text[:600],
        "has_structured_visual": bool(item.get("structured_visual")),
        "frame_paths": temporal_frames if analysis_kind == "temporal" else frames,
        "evidence_layers": {
            "A_transcript": bool(transcript),
            "B_local_ebook_or_ocr": bool(visual_text or item.get("structured_visual")),
            "C_targeted_multimodal": False,
        },
    }


def _stratified_select(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(candidates) <= limit:
        return sorted(candidates, key=lambda row: (row["start"], row["index"]))
    maximum = max(float(row["end"] or row["start"] or 0.0) for row in candidates)
    width = maximum / limit if maximum > 0 else 1.0
    selected: list[dict[str, Any]] = []
    selected_indexes: set[int] = set()
    for bucket in range(limit):
        start = bucket * width
        end = math.inf if bucket == limit - 1 else (bucket + 1) * width
        rows = [row for row in candidates if start <= float(row["start"] or 0.0) < end]
        if not rows:
            continue
        choice = max(rows, key=lambda row: (row["score"], len(row["reasons"]), -row["index"]))
        selected.append(choice)
        selected_indexes.add(choice["index"])
    if len(selected) < limit:
        remaining = sorted(candidates, key=lambda row: (-row["score"], row["start"], row["index"]))
        for row in remaining:
            if row["index"] in selected_indexes:
                continue
            selected.append(row)
            selected_indexes.add(row["index"])
            if len(selected) >= limit:
                break
    return sorted(selected[:limit], key=lambda row: (row["start"], row["index"]))


def _frame_paths(root: Path, item: dict[str, Any]) -> list[str]:
    paths = _path_list(root, item.get("frame_paths"))
    for asset in item.get("assets") or []:
        if isinstance(asset, dict):
            paths.extend(_path_list(root, [asset.get("path") or asset.get("source")]))
    return list(dict.fromkeys(paths))


def _path_list(root: Path, values: Any) -> list[str]:
    rows = values if isinstance(values, list) else []
    result: list[str] = []
    for value in rows:
        text = str(value or "").strip()
        if not text:
            continue
        path = Path(text)
        result.append(str(path if path.is_absolute() else (root / path).resolve()))
    return result


def _strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, dict):
        return [str(key) for key, enabled in value.items() if enabled]
    return [str(value)] if str(value or "").strip() else []


def _number(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _next_actions(root: Path, semantic: list[int], temporal: list[int]) -> list[str]:
    semantic_csv = ",".join(str(value) for value in semantic)
    temporal_csv = ",".join(str(value) for value in temporal)
    index_args = []
    if semantic_csv:
        index_args.append(f"--semantic-indexes {semantic_csv}")
    if temporal_csv:
        index_args.append(f"--temporal-indexes {temporal_csv}")
    create = (
        f".\\scripts\\video-knowledge.ps1 vision-export-consent-create '{root}' "
        f"{' '.join(index_args)} --max-calls {len(semantic) + len(temporal)} "
        f"--image-max-edge 512 --image-jpeg-quality 55 --confirm-data-export"
    )
    return [
        create,
        "Run vision-execution-preflight for exactly these indexes.",
        "Execute C only with the active consent; if the agent platform still refuses, use the generated visible PowerShell fallback or a local VLM.",
        "Regenerate summary C, then compare A/B/C using the listed evaluation dimensions.",
    ]


def _render_markdown(result: dict[str, Any]) -> str:
    selection = result.get("selection") or {}
    lines = [
        "# Visual A/B/C Benchmark Plan",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Candidates/selected: `{selection.get('candidate_count', 0)}` / `{selection.get('selected_count', 0)}`",
        f"- Semantic indexes: `{selection.get('semantic_indexes', [])}`",
        f"- Temporal indexes: `{selection.get('temporal_indexes', [])}`",
        "",
        "## Comparison",
        "",
        "- A: corrected transcript only",
        "- B: A plus local ebook/OCR evidence",
        "- C: B plus targeted multimodal evidence for only the selected rows",
        "",
        "## Samples",
        "",
        "| Index | Time | Kind | Score | Local B evidence | Reasons |",
        "|---:|---:|---|---:|---|---|",
    ]
    for row in result.get("items") or []:
        lines.append(
            f"| {row.get('index')} | {row.get('start', 0):.1f}s | {row.get('analysis_kind')} | {row.get('score')} | "
            f"{'yes' if row.get('evidence_layers', {}).get('B_local_ebook_or_ocr') else 'no'} | {', '.join(row.get('reasons') or [])} |"
        )
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {value}" for value in result.get("next_actions") or [])
    return "\n".join(lines).rstrip() + "\n"
