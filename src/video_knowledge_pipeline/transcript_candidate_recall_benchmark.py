from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any

from .models import now_iso
from .storage import bundle_write_lock, read_json, write_json
from .transcript import parse_transcript, transcript_excerpt
from .transcript_semantic_correction import DOMAIN_SEMANTIC_SUSPECT_CORRECTIONS

SCHEMA = "video_knowledge_pipeline.transcript_candidate_recall_benchmark.v1"


def benchmark_transcript_candidate_recall(
    bundle_dir: str | Path,
    *,
    reference_transcript: str | Path | None = None,
    source_transcript: str | Path | None = None,
    target_pairs_json: str | Path | None = None,
    asr_ab_run_json: str | Path | None = None,
    start_seconds: float = 0.0,
    end_seconds: float = 0.0,
    write: bool = True,
) -> dict[str, Any]:
    """Benchmark ASR variants and semantic-correction candidate recall.

    The reference transcript is evaluation-only. It is never written as a
    sidecar evidence source and must not be used by the correction pipeline.
    """

    root = Path(bundle_dir).expanduser().resolve()
    manifest = _read_manifest(root)
    source_path = _resolve_optional_path(root, source_transcript) or _default_source_transcript(root, manifest)
    reference_path = _resolve_optional_path(root, reference_transcript)
    run_path = _resolve_optional_path(root, asr_ab_run_json)
    pairs_path = _resolve_optional_path(root, target_pairs_json)
    targets = _load_targets(pairs_path)
    source_text = _excerpt_from_path(source_path, start_seconds=start_seconds, end_seconds=end_seconds)
    reference_text = _excerpt_from_path(reference_path, start_seconds=start_seconds, end_seconds=end_seconds)
    artifacts = _artifact_paths(root, manifest)
    target_rows = [
        _score_target(
            target,
            source_text=source_text,
            reference_text=reference_text,
            artifacts=artifacts,
        )
        for target in targets
    ]
    active_targets = [row for row in target_rows if row["active_in_source"] or row["supported_by_reference"]]
    discovered = [row for row in active_targets if row["discovered_by_any_candidate_artifact"]]
    conflict_ready = [row for row in active_targets if row["conflict_index_hit"]]
    applied_or_closed = [row for row in active_targets if row["validation_hit"] or row["corrected_transcript_hit"]]
    asr_variants = _score_asr_variants(run_path, reference_text=reference_text, start_seconds=start_seconds, end_seconds=end_seconds)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "source_transcript": str(source_path) if source_path else "",
        "reference_transcript": str(reference_path) if reference_path else "",
        "reference_role": "evaluation_only_not_correction_evidence" if reference_path else "missing",
        "asr_ab_run_json": str(run_path) if run_path else "",
        "window": {
            "start_seconds": float(start_seconds or 0.0),
            "end_seconds": float(end_seconds or 0.0),
            "enabled": bool(end_seconds and end_seconds > start_seconds),
        },
        "status": _status(active_targets, discovered, conflict_ready, asr_variants, reference_text),
        "summary": {
            "target_count": len(target_rows),
            "active_or_reference_supported_target_count": len(active_targets),
            "candidate_recall_count": len(discovered),
            "candidate_recall_rate": _ratio(len(discovered), len(active_targets)),
            "conflict_index_recall_count": len(conflict_ready),
            "conflict_index_recall_rate": _ratio(len(conflict_ready), len(active_targets)),
            "applied_or_closed_count": len(applied_or_closed),
            "applied_or_closed_rate": _ratio(len(applied_or_closed), len(active_targets)),
            "asr_variant_count": len(asr_variants),
            "best_asr_variant": asr_variants[0]["key"] if asr_variants else "",
            "missed_target_count": len([row for row in active_targets if not row["discovered_by_any_candidate_artifact"]]),
        },
        "targets": target_rows,
        "asr_variants": asr_variants,
        "artifact_inputs": {key: str(value) for key, value in artifacts.items() if value and value.exists()},
        "operator_boundary": {
            "reference_transcript_is_evaluation_only": True,
            "does_not_import_reference_as_evidence": True,
            "does_not_apply_corrections": True,
            "does_not_call_cloud_models": True,
            "use_to_improve_candidate_discovery_recall": True,
        },
        "next_actions": _next_actions(active_targets, discovered, conflict_ready, asr_variants),
        "updated_at": now_iso(),
    }
    if write:
        with bundle_write_lock(root, operation="transcript_candidate_recall_benchmark", timeout_seconds=1.0):
            json_path = root / "transcript-candidate-recall-benchmark.json"
            md_path = root / "transcript-candidate-recall-benchmark.md"
            write_json(json_path, result)
            md_path.write_text(_render_markdown(result), encoding="utf-8")
            manifest["transcript_candidate_recall_benchmark_json"] = "transcript-candidate-recall-benchmark.json"
            manifest["transcript_candidate_recall_benchmark_markdown"] = "transcript-candidate-recall-benchmark.md"
            manifest["transcript_candidate_recall_benchmark_summary"] = {
                "status": result["status"],
                "candidate_recall_rate": result["summary"]["candidate_recall_rate"],
                "missed_target_count": result["summary"]["missed_target_count"],
                "updated_at": result["updated_at"],
            }
            write_json(root / "manifest.json", manifest)
            result["json_path"] = str(json_path)
            result["markdown_path"] = str(md_path)
    return result


def _load_targets(path: Path | None) -> list[dict[str, Any]]:
    if path and path.exists():
        data = read_json(path)
        rows = data.get("targets") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ValueError("target pairs JSON must be a list or an object with targets")
        return [_normalise_target(row) for row in rows if isinstance(row, dict)]
    return [
        {
            "original_text": original,
            "corrected_text": corrected,
            "reason": reason,
            "source": "built_in_domain_semantic_suspects",
        }
        for original, (corrected, reason) in DOMAIN_SEMANTIC_SUSPECT_CORRECTIONS.items()
    ]


def _normalise_target(row: dict[str, Any]) -> dict[str, Any]:
    original = str(row.get("original_text") or row.get("bad_text") or row.get("asr_text") or "").strip()
    corrected = str(row.get("corrected_text") or row.get("expected_text") or row.get("reference_text") or "").strip()
    if not original or not corrected:
        raise ValueError("target pair requires original_text and corrected_text")
    return {
        "original_text": original,
        "corrected_text": corrected,
        "reason": str(row.get("reason") or ""),
        "source": str(row.get("source") or "custom"),
    }


def _score_target(
    target: dict[str, Any],
    *,
    source_text: str,
    reference_text: str,
    artifacts: dict[str, Path],
) -> dict[str, Any]:
    original = str(target.get("original_text") or "")
    corrected = str(target.get("corrected_text") or "")
    source_has_original = _contains(source_text, original)
    source_has_corrected = _contains(source_text, corrected)
    reference_has_corrected = _contains(reference_text, corrected)
    reference_has_original = _contains(reference_text, original)
    hits = {
        "semantic_pack_hit": _artifact_pair_hit(artifacts.get("semantic_pack"), original, corrected),
        "candidate_discovery_hit": _artifact_pair_hit(artifacts.get("candidate_discovery_pack"), original, corrected),
        "candidate_suggestions_hit": _artifact_pair_hit(artifacts.get("candidate_suggestions"), original, corrected),
        "conflict_index_hit": _artifact_pair_hit(artifacts.get("conflict_index"), original, corrected),
        "validation_hit": _artifact_pair_hit(artifacts.get("validation"), original, corrected),
        "corrected_transcript_hit": _corrected_transcript_hit(artifacts.get("corrected_transcript"), original, corrected),
    }
    discovered = any(hits[key] for key in ("semantic_pack_hit", "candidate_discovery_hit", "candidate_suggestions_hit", "conflict_index_hit"))
    return {
        **target,
        "active_in_source": bool(source_has_original and not source_has_corrected),
        "supported_by_reference": bool(reference_has_corrected and not reference_has_original),
        "source_has_original": source_has_original,
        "source_has_corrected": source_has_corrected,
        "reference_has_corrected": reference_has_corrected,
        "reference_has_original": reference_has_original,
        "discovered_by_any_candidate_artifact": bool(discovered),
        **hits,
        "missed": bool((source_has_original or reference_has_corrected) and not discovered),
    }


def _score_asr_variants(
    run_path: Path | None,
    *,
    reference_text: str,
    start_seconds: float,
    end_seconds: float,
) -> list[dict[str, Any]]:
    if not run_path or not run_path.exists():
        return []
    run = read_json(run_path)
    variants = run.get("variants") if isinstance(run, dict) else []
    rows = []
    for row in variants if isinstance(variants, list) else []:
        if not isinstance(row, dict):
            continue
        path = _first_existing(row.get("normalized_json"), row.get("normalized_srt"), row.get("raw_output_json"))
        text = _excerpt_from_path(path, start_seconds=start_seconds, end_seconds=end_seconds)
        rows.append(
            {
                "key": str(row.get("key") or ""),
                "status": str(row.get("status") or ""),
                "transcript_path": str(path) if path else "",
                "char_count": len(text),
                "punctuation_count": _punctuation_count(text),
                "punctuation_per_1000_chars": _ratio(_punctuation_count(text) * 1000, len(text)),
                "reference_similarity": _similarity(text, reference_text) if reference_text else 0.0,
            }
        )
    rows.sort(key=lambda item: (float(item.get("reference_similarity") or 0.0), float(item.get("punctuation_per_1000_chars") or 0.0)), reverse=True)
    return rows


def _artifact_paths(root: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    return {
        "semantic_pack": _bundle_path(root, manifest.get("transcript_semantic_correction_pack_json") or "transcript-semantic-correction-pack.json"),
        "candidate_discovery_pack": _bundle_path(root, manifest.get("transcript_semantic_candidate_discovery_pack_json") or "transcript-semantic-candidate-discovery-pack.json"),
        "candidate_suggestions": _first_existing(
            _bundle_path(root, manifest.get("transcript_semantic_candidate_suggestions_imported_json") or "transcript-semantic-candidate-suggestions-imported.json"),
            _bundle_path(root, manifest.get("transcript_semantic_candidate_suggestions_llm_json") or "transcript-semantic-candidate-suggestions.llm.json"),
            _bundle_path(root, manifest.get("transcript_semantic_candidate_suggestions_codex_json") or "transcript-semantic-candidate-suggestions.codex.json"),
        ),
        "conflict_index": _bundle_path(root, manifest.get("evidence_conflict_index_json") or "evidence-conflict-index.json"),
        "validation": _bundle_path(root, manifest.get("transcript_semantic_correction_validation_json") or "transcript-semantic-correction-validation.json"),
        "corrected_transcript": _bundle_path(root, manifest.get("corrected_transcript_json") or manifest.get("source_arbitrated_transcript_json") or "corrected-transcript.json"),
    }


def _default_source_transcript(root: Path, manifest: dict[str, Any]) -> Path | None:
    for key in ("normalized_transcript_json", "postprocessed_transcript_json", "transcript_json", "source_transcript", "transcript_path"):
        path = _bundle_path(root, manifest.get(key))
        if path.exists():
            return path
    for name in ("normalized-transcript.json", "postprocessed-transcript.json", "transcript.json"):
        path = root / name
        if path.exists():
            return path
    return None


def _excerpt_from_path(path: Path | None, *, start_seconds: float, end_seconds: float) -> str:
    if not path or not path.exists():
        return ""
    try:
        cues = parse_transcript(path)
    except Exception:
        return path.read_text(encoding="utf-8", errors="replace")
    if end_seconds and end_seconds > start_seconds:
        return transcript_excerpt(cues, float(start_seconds), float(end_seconds))
    return " ".join(cue.text for cue in cues).strip()


def _artifact_pair_hit(path: Path | None, original: str, corrected: str) -> bool:
    text = _artifact_text(path)
    if not text:
        return False
    return _contains(text, original) and _contains(text, corrected)


def _corrected_transcript_hit(path: Path | None, original: str, corrected: str) -> bool:
    text = _artifact_text(path)
    if not text:
        return False
    return _contains(text, corrected) and not _contains(text, original)


def _artifact_text(path: Path | None) -> str:
    if not path or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _contains(text: str, needle: str) -> bool:
    return bool(needle and _compact(needle) in _compact(text))


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return round(difflib.SequenceMatcher(a=_compact(left), b=_compact(right), autojunk=False).ratio(), 4)


def _punctuation_count(text: str) -> int:
    return sum(str(text or "").count(ch) for ch in "，。！？；：、,.!?;:")


def _status(active_targets: list[dict[str, Any]], discovered: list[dict[str, Any]], conflict_ready: list[dict[str, Any]], asr_variants: list[dict[str, Any]], reference_text: str) -> str:
    if not reference_text:
        return "missing_reference_transcript"
    if not active_targets:
        return "no_active_reference_targets"
    if len(discovered) < len(active_targets):
        return "candidate_recall_gap"
    if len(conflict_ready) < len(discovered):
        return "candidate_recall_ready_conflict_index_gap"
    if not asr_variants:
        return "candidate_recall_ready_asr_ab_missing"
    return "candidate_recall_ready"


def _next_actions(active_targets: list[dict[str, Any]], discovered: list[dict[str, Any]], conflict_ready: list[dict[str, Any]], asr_variants: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    missed = [row for row in active_targets if not row["discovered_by_any_candidate_artifact"]]
    if missed:
        examples = ", ".join(f"{row['original_text']}=>{row['corrected_text']}" for row in missed[:5])
        actions.append(f"Improve semantic candidate discovery rules or evidence inputs for missed targets: {examples}.")
    if len(conflict_ready) < len(discovered):
        actions.append("Promote discovered real conflicts into evidence-conflict-index before LLM arbitration; pure heuristic risks should remain deferred.")
    if not asr_variants:
        actions.append("Run asr-ab-sample-run on the same fixed sample to decide whether a second ASR source improves content accuracy.")
    elif len(asr_variants) >= 2:
        best = asr_variants[0]
        actions.append(f"Review ASR A/B winner `{best.get('key')}` with reference_similarity={best.get('reference_similarity')}; introduce second ASR only if it fixes term errors, not just punctuation.")
    return actions or ["Candidate recall is ready for this fixed sample; rerun on another sample window before changing defaults."]


def _render_markdown(result: dict[str, Any]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    lines = [
        "# Transcript Candidate Recall Benchmark",
        "",
        f"- Status: `{result.get('status', '')}`",
        f"- Bundle: `{result.get('bundle_dir', '')}`",
        f"- Source transcript: `{result.get('source_transcript', '')}`",
        f"- Reference transcript: `{result.get('reference_transcript', '')}`",
        f"- Reference role: `{result.get('reference_role', '')}`",
        f"- Candidate recall: `{summary.get('candidate_recall_count', 0)}/{summary.get('active_or_reference_supported_target_count', 0)}` = `{summary.get('candidate_recall_rate', 0)}`",
        f"- Conflict-index recall: `{summary.get('conflict_index_recall_count', 0)}/{summary.get('active_or_reference_supported_target_count', 0)}` = `{summary.get('conflict_index_recall_rate', 0)}`",
        "",
        "## Targets",
        "",
        "| Original | Expected | Active | Reference | Candidate | Conflict | Applied/Closed | Missed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in result.get("targets") or []:
        lines.append(
            f"| {row.get('original_text', '')} | {row.get('corrected_text', '')} | `{row.get('active_in_source', False)}` | "
            f"`{row.get('supported_by_reference', False)}` | `{row.get('discovered_by_any_candidate_artifact', False)}` | "
            f"`{row.get('conflict_index_hit', False)}` | `{bool(row.get('validation_hit') or row.get('corrected_transcript_hit'))}` | `{row.get('missed', False)}` |"
        )
    variants = result.get("asr_variants") if isinstance(result.get("asr_variants"), list) else []
    if variants:
        lines.extend(["", "## ASR Variants", "", "| Variant | Status | Similarity | Punctuation/1000 | Chars |", "| --- | --- | ---: | ---: | ---: |"])
        for row in variants:
            lines.append(f"| `{row.get('key', '')}` | `{row.get('status', '')}` | {row.get('reference_similarity', 0)} | {row.get('punctuation_per_1000_chars', 0)} | {row.get('char_count', 0)} |")
    lines.extend(["", "## Next Actions", ""])
    for action in result.get("next_actions") or []:
        lines.append(f"- {action}")
    lines.extend(["", "## Boundary", "", "The reference transcript is used only for benchmark scoring. It is not imported as correction evidence."])
    return "\n".join(lines).rstrip() + "\n"


def _ratio(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator or 0)
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / denominator, 4)


def _resolve_optional_path(root: Path, value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _bundle_path(root: Path, value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else root / path


def _first_existing(*values: Any) -> Path | None:
    for value in values:
        if not value:
            continue
        path = value if isinstance(value, Path) else Path(str(value)).expanduser()
        try:
            path = path.resolve()
        except Exception:
            pass
        if path.exists():
            return path
    return None


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.exists():
        return {}
    value = read_json(path)
    return value if isinstance(value, dict) else {}
