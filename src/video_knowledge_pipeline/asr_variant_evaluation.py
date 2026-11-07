"""Compare three transcript variants on the same explicitly reviewed references."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .asr_comparison_alignment import critical_differences
from .file_hash import sha256_file
from .quality_benchmark import transcript_quality_metrics
from .storage import read_json, write_json
from .transcript import parse_transcript, transcript_excerpt


def compare_asr_variants(reference_json: str | Path, *, primary: str | Path,
                         secondary: str | Path, integrated: str | Path,
                         output_json: str | Path | None = None) -> dict[str, Any]:
    reference_path = Path(reference_json).resolve()
    reference = read_json(reference_path)
    samples = reference.get("samples", [])
    reviewed = [r for r in samples if r.get("reference_status") == "human_verified"
                and str(r.get("reference_text", "")).strip() and float(r["end"]) > float(r["start"]) >= 0]
    paths = {name: Path(value).resolve() for name, value in (("primary", primary), ("secondary", secondary), ("integrated", integrated))}
    variants = {name: parse_transcript(path) for name, path in paths.items()}
    # An overlapping cue includes text outside the reviewed window. Exclude
    # that sample from ALL arms, rather than scoring incompatible excerpts.
    boundary_excluded = [r for r in reviewed if any(
        c.start < float(r["start"]) < c.end or c.start < float(r["end"]) < c.end
        for cues in variants.values() for c in cues)]
    reviewed = [r for r in reviewed if r not in boundary_excluded]
    result: dict[str, Any] = {"status": "evaluated" if reviewed else "blocked_reference_missing",
        "reference_sha256": sha256_file(reference_path), "reviewed_sample_count": len(reviewed),
        "excluded_sample_count": len(samples) - len(reviewed), "variants": {},
        "boundary_excluded_sample_count": len(boundary_excluded),
        "model_switch_allowed": False, "scope": "same_reviewed_windows_only"}
    if not reviewed and boundary_excluded:
        result["status"] = "blocked_incompatible_sample_boundaries"
    for name, path in paths.items():
        cues = variants[name]
        rows = []
        for sample in reviewed:
            text = transcript_excerpt(cues, float(sample["start"]), float(sample["end"]))
            metrics = transcript_quality_metrics(sample["reference_text"], text,
                protected_entities=sample.get("protected_entities", []))
            # The shared helper has no CER for an empty candidate. Treat total
            # deletion as 100%, rather than omitting it from this comparison.
            if not text.strip():
                metrics["cer"] = 1.0
            rows.append({"sample_id": sample.get("sample_id", sample.get("id")),
                "start": sample["start"], "end": sample["end"], "metrics": metrics,
                "critical_differences": critical_differences(sample["reference_text"], text,
                    sample.get("protected_entities", []))})
        result["variants"][name] = {"path": str(path), "sha256": sha256_file(path), "samples": rows,
            "mean_sample_cer": sum(r["metrics"]["cer"] for r in rows if r["metrics"]["cer"] is not None) / len(rows) if rows else None,
            "critical_error_sample_count": sum(bool(set(r["critical_differences"]) - {"lexical_change"}) for r in rows),
            "elapsed_seconds": None, "timing_status": "not_measured_by_transcript_comparison"}
    if output_json:
        write_json(Path(output_json), result)
    return result
