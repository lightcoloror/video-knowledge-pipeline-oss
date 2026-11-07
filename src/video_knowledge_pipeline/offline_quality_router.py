from __future__ import annotations

import hashlib
import re
import statistics
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .file_hash import sha256_file
from .storage import read_json, write_json
from .transcript import parse_transcript


SCHEMA = "video_knowledge_pipeline.offline_quality_route.v1"
STAGE_QUALITY = ("unknown", "missing", "stopped_by_design", "low", "weak", "usable", "good")
FAILURE_CLASSES = (
    "none",
    "artifact_missing",
    "empty_output",
    "punctuation_sparse",
    "segmentation_coarse",
    "coverage_incomplete",
    "text_mutation",
    "execution_not_requested",
    "provider_unavailable",
    "parse_failure",
)
FALLBACK_ACTIONS = (
    "no_action",
    "use_normalized_asr",
    "use_postprocessed_asr",
    "run_local_postprocess",
    "retry_local_stage",
    "keep_local_evidence",
    "retain_image_evidence",
    "generate_optional_review_pack",
)
HUMAN_REVIEW = ("not_required", "optional", "recommended")
_PUNCTUATION_RE = re.compile(r"[，。！？；：、,.!?;:]")
_CONTENT_DROP_RE = re.compile(r"[\s，。！？；：、,.!?;:]+")


def offline_quality_route(
    bundle_dir: str | Path,
    *,
    benchmark_manifest: str | Path | None = None,
    output_dir: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Build a deterministic, local-only quality and fallback routing report."""

    root = Path(bundle_dir).expanduser().resolve()
    manifest = _mapping(root / "manifest.json")
    paths = _transcript_paths(root, manifest)
    transcript_metrics = {key: _transcript_metrics(path) for key, path in paths.items()}
    comparisons = {
        "normalized_to_postprocessed": _compare_text_stages(
            transcript_metrics["normalized"], transcript_metrics["postprocessed"]
        ),
        "normalized_to_corrected": _compare_text_stages(
            transcript_metrics["normalized"], transcript_metrics["corrected"]
        ),
    }
    visual_metrics = _visual_metrics(_timeline(root))
    review = _review_machine_summary(benchmark_manifest, bundle_dir=root)
    stages = {
        "asr": _route_asr(transcript_metrics["normalized"]),
        "punctuation": _route_punctuation(
            transcript_metrics["postprocessed"], comparisons["normalized_to_postprocessed"]
        ),
        "segmentation": _route_segmentation(transcript_metrics["postprocessed"]),
        "ocr": _route_ocr(visual_metrics),
        "vision": _route_vision(visual_metrics),
    }
    result = {
        "schema": SCHEMA,
        "bundle_dir": str(root),
        "input_hashes": _input_hashes(
            [root / "manifest.json", *[path for path in paths.values() if path], root / "timeline.json"]
        ),
        "vocabulary": {
            "stage_quality": list(STAGE_QUALITY),
            "failure_class": list(FAILURE_CLASSES),
            "fallback": list(FALLBACK_ACTIONS),
            "human_review_needed": list(HUMAN_REVIEW),
        },
        "transcript_metrics": transcript_metrics,
        "comparisons": comparisons,
        "visual_metrics": visual_metrics,
        "review_page_machine_summary": review,
        "stages": stages,
        "routing_proposal": _routing_proposal(stages),
        "operator_boundary": {
            "local_only": True,
            "auto_execute": False,
            "cloud_allowed": False,
            "downloads_allowed": False,
            "review_page_presence_is_not_review_completion": True,
        },
    }
    if write:
        out = Path(output_dir).expanduser().resolve() if output_dir else root / "offline-quality-route"
        out.mkdir(parents=True, exist_ok=True)
        result["artifacts"] = {
            "route_json": str(out / "offline-quality-route.json"),
            "route_markdown": str(out / "offline-quality-route.md"),
            "proposal_json": str(out / "routing-proposal.json"),
            "proposal_markdown": str(out / "routing-proposal.md"),
            "review_summary_json": str(out / "review-page-machine-summary.json"),
            "review_summary_markdown": str(out / "review-page-machine-summary.md"),
        }
        write_json(out / "offline-quality-route.json", result)
        (out / "offline-quality-route.md").write_text(_render_route(result), encoding="utf-8")
        write_json(out / "routing-proposal.json", result["routing_proposal"])
        (out / "routing-proposal.md").write_text(_render_proposal(result), encoding="utf-8")
        write_json(out / "review-page-machine-summary.json", review)
        (out / "review-page-machine-summary.md").write_text(_render_review(review), encoding="utf-8")

    return result


def _transcript_paths(root: Path, manifest: dict[str, Any]) -> dict[str, Path | None]:
    return {
        "normalized": _resolve(root, manifest.get("normalized_transcript_json"), "normalized-transcript.json"),
        "postprocessed": _resolve(root, manifest.get("postprocessed_transcript_json"), "postprocessed-transcript.json"),
        "corrected": _resolve(root, manifest.get("corrected_transcript_json"), "corrected-transcript.json"),
    }


def _transcript_metrics(path: Path | None) -> dict[str, Any]:
    base = {
        "path": str(path or ""),
        "exists": bool(path and path.exists()),
        "parse_ok": False,
        "segment_count": 0,
        "character_count": 0,
        "content_character_count": 0,
        "punctuation_count": 0,
        "punctuation_per_100_chars": 0.0,
        "average_segment_chars": 0.0,
        "p95_segment_chars": 0,
        "duration_seconds": 0.0,
        "timeline_coverage_ratio": 0.0,
        "content_fingerprint": "",
    }
    if not path or not path.exists():
        return base
    try:
        cues = parse_transcript(path)
    except Exception as exc:
        return {**base, "parse_error": type(exc).__name__}
    texts = [str(cue.text or "") for cue in cues if str(cue.text or "").strip()]
    joined = "".join(texts)
    content = _content_only(joined)
    lengths = sorted(len(text) for text in texts)
    duration = max((float(cue.end) for cue in cues), default=0.0)
    covered = _covered_seconds(cues)
    punctuation = len(_PUNCTUATION_RE.findall(joined))
    return {
        **base,
        "parse_ok": True,
        "segment_count": len(texts),
        "character_count": len(joined),
        "content_character_count": len(content),
        "punctuation_count": punctuation,
        "punctuation_per_100_chars": round(punctuation * 100 / max(1, len(joined)), 4),
        "average_segment_chars": round(statistics.mean(lengths), 3) if lengths else 0.0,
        "p95_segment_chars": _percentile(lengths, 0.95),
        "duration_seconds": round(duration, 3),
        "timeline_coverage_ratio": round(covered / max(duration, 0.001), 6) if duration else 0.0,
        "content_fingerprint": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def _compare_text_stages(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    if not source.get("exists") or not target.get("exists"):
        return {
            "comparable": False,
            "text_character_preservation_ratio": None,
            "content_exact": False,
            "only_punctuation_or_segmentation_changed": False,
        }
    source_text = _content_from_path(source.get("path"))
    target_text = _content_from_path(target.get("path"))
    ratio = SequenceMatcher(None, source_text, target_text, autojunk=False).ratio() if source_text or target_text else 1.0
    exact = source_text == target_text
    return {
        "comparable": True,
        "text_character_preservation_ratio": round(ratio, 6),
        "content_exact": exact,
        "only_punctuation_or_segmentation_changed": exact,
        "source_content_chars": len(source_text),
        "target_content_chars": len(target_text),
    }


def _timeline(root: Path) -> list[dict[str, Any]]:
    value = read_json(root / "timeline.json") if (root / "timeline.json").exists() else []
    if isinstance(value, dict):
        value = value.get("items") or value.get("timeline") or []
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _visual_metrics(timeline: list[dict[str, Any]]) -> dict[str, Any]:
    document_expected = document_covered = vision_expected = vision_covered = 0
    for row in timeline:
        route = str(row.get("visual_route") or "")
        if route in {"document_visual", "mixed"}:
            document_expected += 1
            if row.get("visual_text") or row.get("structured_visual"):
                document_covered += 1
        if route in {"semantic_frame", "temporal_sequence", "mixed"}:
            vision_expected += 1
            if row.get("visual_understanding") or row.get("temporal_visual_understanding"):
                vision_covered += 1
    return {
        "timeline_item_count": len(timeline),
        "ocr_expected_count": document_expected,
        "ocr_evidence_count": document_covered,
        "ocr_coverage_ratio": round(document_covered / document_expected, 6) if document_expected else None,
        "vision_expected_count": vision_expected,
        "vision_evidence_count": vision_covered,
        "vision_coverage_ratio": round(vision_covered / vision_expected, 6) if vision_expected else None,
    }


def _review_machine_summary(manifest_path: str | Path | None, *, bundle_dir: Path | None = None) -> dict[str, Any]:
    if not manifest_path:
        return {
            "manifest_path": "",
            "review_page_exists": False,
            "sample_count": 0,
            "human_reference_count": 0,
            "asr_prefilled_count": 0,
            "content_reviewed": False,
            "status": "not_configured",
        }
    path = Path(manifest_path).expanduser().resolve()
    manifest = _mapping(path)
    all_samples = [row for row in manifest.get("samples") or [] if isinstance(row, dict)]
    if bundle_dir:
        expected_bundle = str(bundle_dir.resolve())
        samples = [
            row
            for row in all_samples
            if not row.get("bundle_dir")
            or str(Path(str(row["bundle_dir"])).expanduser().resolve()) == expected_bundle
        ]
    else:
        samples = all_samples
    human_count = sum(
        1
        for row in samples
        if str(row.get("reference_text") or "").strip()
        and str(row.get("human_review_status") or "") == "completed"
    )
    review_page = path.parent / "quality-benchmark-review.html"
    reviewed = bool(samples) and human_count == len(samples)
    return {
        "manifest_path": str(path),
        "review_page_path": str(review_page),
        "review_page_exists": review_page.exists(),
        "sample_count": len(samples),
        "human_reference_count": human_count,
        "asr_prefilled_count": sum(1 for row in samples if str(row.get("asr_draft_text") or "").strip()),
        "content_reviewed": reviewed,
        "status": "completed" if reviewed else ("pending" if samples else "empty"),
    }


def _route_asr(metrics: dict[str, Any]) -> dict[str, Any]:
    if not metrics.get("exists"):
        return _stage("missing", 0, ["artifact_missing"], "retry_local_stage", "recommended")
    if not metrics.get("parse_ok"):
        return _stage("low", 0, ["parse_failure"], "retry_local_stage", "recommended")
    if metrics.get("content_character_count", 0) <= 0:
        return _stage("low", 0, ["empty_output"], "retry_local_stage", "recommended")
    quality = "good" if metrics.get("timeline_coverage_ratio", 0) >= 0.98 else "usable"
    failures = [] if quality == "good" else ["coverage_incomplete"]
    return _stage(quality, int(metrics.get("segment_count") or 0), failures, "no_action", "not_required")


def _route_punctuation(metrics: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    if not metrics.get("exists"):
        return _stage("missing", 0, ["artifact_missing"], "run_local_postprocess", "optional")
    failures: list[str] = []
    if float(metrics.get("punctuation_per_100_chars") or 0.0) < 1.5:
        failures.append("punctuation_sparse")
    preservation = comparison.get("text_character_preservation_ratio")
    if preservation is not None and float(preservation) < 0.995:
        failures.append("text_mutation")
    quality = "good" if not failures else ("low" if "text_mutation" in failures else "weak")
    fallback = "use_postprocessed_asr" if not failures else "run_local_postprocess"
    return _stage(quality, int(metrics.get("punctuation_count") or 0), failures, fallback, "optional" if failures else "not_required")


def _route_segmentation(metrics: dict[str, Any]) -> dict[str, Any]:
    if not metrics.get("exists"):
        return _stage("missing", 0, ["artifact_missing"], "use_normalized_asr", "optional")
    coarse = float(metrics.get("average_segment_chars") or 0.0) > 140 or int(metrics.get("p95_segment_chars") or 0) > 240
    return _stage(
        "weak" if coarse else "good",
        int(metrics.get("segment_count") or 0),
        ["segmentation_coarse"] if coarse else [],
        "run_local_postprocess" if coarse else "no_action",
        "optional" if coarse else "not_required",
    )


def _route_ocr(metrics: dict[str, Any]) -> dict[str, Any]:
    expected = int(metrics.get("ocr_expected_count") or 0)
    evidence = int(metrics.get("ocr_evidence_count") or 0)
    if expected == 0:
        return _stage("stopped_by_design", evidence, ["execution_not_requested"], "retain_image_evidence", "not_required")
    if evidence == 0:
        return _stage("missing", 0, ["artifact_missing"], "retry_local_stage", "recommended")
    if evidence / expected < 0.8:
        return _stage("weak", evidence, ["coverage_incomplete"], "retain_image_evidence", "optional")
    return _stage("good", evidence, [], "no_action", "not_required")


def _route_vision(metrics: dict[str, Any]) -> dict[str, Any]:
    expected = int(metrics.get("vision_expected_count") or 0)
    evidence = int(metrics.get("vision_evidence_count") or 0)
    if expected == 0 or evidence == 0:
        review = "optional" if expected else "not_required"
        return _stage("stopped_by_design", evidence, ["execution_not_requested"], "keep_local_evidence", review)
    if evidence / expected < 0.8:
        return _stage("weak", evidence, ["coverage_incomplete"], "keep_local_evidence", "optional")
    return _stage("good", evidence, [], "no_action", "not_required")


def _stage(quality: str, evidence_count: int, failures: list[str], fallback: str, review: str) -> dict[str, Any]:
    return {
        "stage_quality": quality,
        "evidence_count": evidence_count,
        "failure_class": failures or ["none"],
        "fallback": fallback,
        "human_review_needed": review,
        "auto_execute": False,
        "cloud_allowed": False,
    }


def _routing_proposal(stages: dict[str, dict[str, Any]]) -> dict[str, Any]:
    actions = [
        {
            "stage": stage,
            "quality": state["stage_quality"],
            "fallback": state["fallback"],
            "failure_class": state["failure_class"],
            "human_review_needed": state["human_review_needed"],
            "evidence_count": state["evidence_count"],
            "auto_execute": False,
            "cloud_allowed": False,
        }
        for stage, state in stages.items()
    ]
    return {
        "schema": "video_knowledge_pipeline.offline_quality_routing_proposal.v1",
        "status": "proposal_only",
        "actions": actions,
        "production_config_modified": False,
    }


def _content_from_path(path_value: Any) -> str:
    if not path_value:
        return ""
    try:
        return _content_only("".join(str(cue.text or "") for cue in parse_transcript(Path(str(path_value)))))
    except Exception:
        return ""


def _content_only(value: str) -> str:
    return _CONTENT_DROP_RE.sub("", str(value or ""))


def _covered_seconds(cues: list[Any]) -> float:
    intervals = sorted((max(0.0, float(cue.start)), max(0.0, float(cue.end))) for cue in cues)
    total = 0.0
    current_start = current_end = None
    for start, end in intervals:
        if current_start is None:
            current_start, current_end = start, end
        elif start <= float(current_end):
            current_end = max(float(current_end), end)
        else:
            total += float(current_end) - float(current_start)
            current_start, current_end = start, end
    if current_start is not None:
        total += float(current_end) - float(current_start)
    return max(0.0, total)


def _percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, max(0, int((len(values) - 1) * ratio + 0.999)))
    return int(values[index])


def _resolve(root: Path, configured: Any, fallback: str) -> Path | None:
    for value in (configured, fallback):
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return path.resolve()
    return None


def _mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = read_json(path)
    return value if isinstance(value, dict) else {}


def _input_hashes(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for path in paths:
        if not path or not path.exists():
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        rows.append({
            "path": resolved,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


def _render_route(result: dict[str, Any]) -> str:
    lines = [
        "# Offline Quality Route",
        "",
        "| Stage | Quality | Evidence | Failure | Fallback | Human review |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for stage, row in result["stages"].items():
        failures = ", ".join(row["failure_class"])
        lines.append(
            f"| {stage} | {row['stage_quality']} | {row['evidence_count']} | {failures} | "
            f"{row['fallback']} | {row['human_review_needed']} |"
        )
    lines.extend([
        "",
        "## Boundaries",
        "",
        "- Local only: true",
        "- Auto execute: false",
        "- Cloud allowed: false",
        "- Review page existence means reviewed: false",
    ])
    return "\n".join(lines).rstrip() + "\n"


def _render_proposal(result: dict[str, Any]) -> str:
    lines = [
        "# Offline Routing Proposal",
        "",
        "This is proposal-only. It does not modify production configuration.",
        "",
    ]
    for row in result["routing_proposal"]["actions"]:
        lines.append(
            f"- {row['stage']}: quality={row['quality']}, fallback={row['fallback']}, "
            f"review={row['human_review_needed']}, auto_execute=false, cloud_allowed=false"
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_review(review: dict[str, Any]) -> str:
    return "\n".join([
        "# Review Page Machine Summary",
        "",
        f"- Review page exists: {review.get('review_page_exists')}",
        f"- Samples: {review.get('sample_count')}",
        f"- Human references: {review.get('human_reference_count')}",
        f"- ASR prefilled: {review.get('asr_prefilled_count', 0)}",
        f"- Content reviewed: {review.get('content_reviewed')}",
        f"- Status: {review.get('status')}",
        "",
        "The review page existing does not mean its content has been reviewed.",
    ]).rstrip() + "\n"