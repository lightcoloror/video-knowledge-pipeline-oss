from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import now_iso
from .storage import read_json, write_json
from .transcript_semantic_correction import build_transcript_semantic_correction_pack


SCHEMA = "video_knowledge_pipeline.local_targeted_asr_plan.v1"
_HIGH_RISK_TYPES = {"number", "proper_noun", "term", "action"}
_EXTERNAL_EVIDENCE = {
    "secondary_asr",
    "platform_subtitle",
    "embedded_subtitle",
    "ocr",
    "structured_visual",
    "visual_understanding",
    "temporal_visual",
    "human_note",
}


def build_local_targeted_asr_plan(
    bundle_dir: str | Path,
    *,
    input_pack: str | Path | None = None,
    pack_data: dict[str, Any] | None = None,
    max_windows: int = 24,
    padding_seconds: float = 3.0,
    write: bool = True,
) -> dict[str, Any]:
    """Create bounded local second-ASR windows for unresolved factual candidates.

    The result deliberately uses the existing ``asr-retry-snippets`` contract.
    It creates evidence only: a second ASR can never replace the main transcript
    by itself.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    if not isinstance(manifest, dict):
        manifest = {}
    pack_path = Path(input_pack).expanduser() if input_pack else root / "transcript-semantic-correction-pack.json"
    if not pack_path.is_absolute():
        pack_path = root / pack_path
    pack_path = pack_path.resolve()
    if pack_data is not None:
        if not isinstance(pack_data, dict):
            raise ValueError("semantic correction pack data must be a JSON object")
        pack = pack_data
    else:
        pack = read_json(pack_path) if pack_path.exists() else build_transcript_semantic_correction_pack(root, write=write)
    if not isinstance(pack, dict):
        raise ValueError("semantic correction pack must be a JSON object")

    limit = max(0, int(max_windows or 0))
    padding = max(0.0, float(padding_seconds or 0.0))
    candidates = [row for row in pack.get("candidates") or [] if isinstance(row, dict)]
    selected = [row for row in candidates if _needs_local_second_asr(row)]
    selected.sort(key=_priority)
    if limit:
        selected = selected[:limit]
    windows = _merge_windows(selected, padding_seconds=padding)
    retry_plan = {
        "windows": windows,
        "window_count": len(windows),
        "source": "unresolved_high_risk_semantic_candidates",
        "requires_independent_local_second_asr": bool(windows),
    }
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "status": "planned" if windows else "no_targeted_evidence_needed",
        "ok": True,
        "pack_json": str(pack_path),
        "candidate_count": len(candidates),
        "eligible_candidate_count": len([row for row in candidates if _needs_local_second_asr(row)]),
        "selected_candidate_count": len(selected),
        "max_windows": limit,
        "padding_seconds": padding,
        "selected_candidates": [_candidate_summary(row) for row in selected],
        "retry_plan": retry_plan,
        "operator_boundary": {
            "local_audio_only": True,
            "second_asr_required": True,
            "candidate_only_evidence": True,
            "does_not_modify_raw_asr": True,
            "does_not_auto_apply_corrections": True,
            "output_is_compatible_with": "asr-retry-snippets",
        },
        "next_actions": _next_actions(root, windows),
        "updated_at": now_iso(),
    }
    if write:
        write_json(root / "local-targeted-asr-plan.json", result)
        (root / "local-targeted-asr-plan.md").write_text(_render(result), encoding="utf-8")
        manifest["local_targeted_asr_plan_json"] = "local-targeted-asr-plan.json"
        manifest["local_targeted_asr_plan_markdown"] = "local-targeted-asr-plan.md"
        manifest["local_targeted_asr_plan_summary"] = {
            "status": result["status"],
            "eligible_candidate_count": result["eligible_candidate_count"],
            "selected_candidate_count": result["selected_candidate_count"],
            "window_count": len(windows),
            "updated_at": result["updated_at"],
        }
        write_json(manifest_path, manifest)
    return result


def _needs_local_second_asr(candidate: dict[str, Any]) -> bool:
    source_types = {str(value) for value in candidate.get("evidence_source_types") or [] if str(value)}
    if source_types & _EXTERNAL_EVIDENCE:
        return False
    # A lone number/action heuristic is not a correction proposal. Sending
    # every such row back through ASR creates thousands of clips but cannot
    # establish what text should replace the original. Reserve the bounded
    # evidence queue for concrete alternatives (for example 中威 -> 钟巍).
    if not _has_concrete_alternative(candidate):
        return False
    kind = str(candidate.get("correction_type") or "")
    high_risk = kind in _HIGH_RISK_TYPES or str(candidate.get("risk_level") or "") == "high"
    deferred = str(candidate.get("llm_review_defer_reason") or "") == "needs_conflicting_external_evidence"
    return bool(high_risk and deferred)


def _has_concrete_alternative(candidate: dict[str, Any]) -> bool:
    return bool(
        str(candidate.get("candidate_text") or candidate.get("suggested_text") or "").strip()
    )


def _priority(candidate: dict[str, Any]) -> tuple[int, float, str]:
    kind = str(candidate.get("correction_type") or "")
    score = 0
    score += 100 if str(candidate.get("risk_level") or "") == "high" else 0
    score += {"proper_noun": 50, "term": 45, "number": 40, "action": 20}.get(kind, 0)
    if str(candidate.get("candidate_text") or "").strip():
        score += 25
    return (-score, _number(candidate.get("start")), str(candidate.get("candidate_id") or ""))


def _merge_windows(candidates: list[dict[str, Any]], *, padding_seconds: float) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    for candidate in candidates:
        start = max(0.0, _number(candidate.get("start")) - padding_seconds)
        end = max(start, _number(candidate.get("end"), start) + padding_seconds)
        raw.append({
            "start": start,
            "end": end,
            "candidate_ids": [str(candidate.get("candidate_id") or "")],
            "source_segment_ids": [str(candidate.get("candidate_id") or "")],
            "reasons": [str(candidate.get("reason") or "unresolved_high_risk_semantic_candidate")],
        })
    raw.sort(key=lambda row: (float(row["start"]), float(row["end"])))
    merged: list[dict[str, Any]] = []
    for row in raw:
        if merged and float(row["start"]) <= float(merged[-1]["end"]) + 1.0:
            current = merged[-1]
            current["end"] = max(float(current["end"]), float(row["end"]))
            current["candidate_ids"] = _unique([*current["candidate_ids"], *row["candidate_ids"]])
            current["source_segment_ids"] = _unique([*current["source_segment_ids"], *row["source_segment_ids"]])
            current["reasons"] = _unique([*current["reasons"], *row["reasons"]])
            continue
        merged.append(dict(row))
    for index, row in enumerate(merged, start=1):
        row["retry_id"] = f"semantic-evidence-{index:04d}"
        row["duration_seconds"] = round(float(row["end"]) - float(row["start"]), 6)
        row["alignment_source"] = "semantic_candidate_time_range_with_padding"
    return merged


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "correction_type": str(candidate.get("correction_type") or ""),
        "risk_level": str(candidate.get("risk_level") or ""),
        "start": _number(candidate.get("start")),
        "end": _number(candidate.get("end")),
        "original_text": str(candidate.get("original_text") or ""),
        "candidate_text": str(candidate.get("candidate_text") or ""),
        "reason": str(candidate.get("reason") or ""),
    }


def _next_actions(root: Path, windows: list[dict[str, Any]]) -> list[str]:
    if not windows:
        return ["No unresolved high-risk candidate lacks independent evidence."]
    return [
        "Extract only the planned local clips with asr-retry-snippets --execute.",
        "Run an independent local ASR preset on each extracted clip, then build asr-local-targeted-evidence.",
        "Register the evidence as a secondary ASR source and rerun transcript-evidence-correction-pipeline.",
    ]


def _render(result: dict[str, Any]) -> str:
    lines = [
        "# Local Targeted ASR Evidence Plan",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Eligible candidates: `{result.get('eligible_candidate_count')}`",
        f"- Planned windows: `{len((result.get('retry_plan') or {}).get('windows') or [])}`",
        "",
        "## Windows",
        "",
    ]
    for row in (result.get("retry_plan") or {}).get("windows") or []:
        lines.append(f"- `{row.get('retry_id')}` {float(row.get('start') or 0):.3f}s - {float(row.get('end') or 0):.3f}s; candidates={', '.join(row.get('candidate_ids') or [])}")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {value}" for value in result.get("next_actions") or [])
    return "\n".join(lines) + "\n"


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
