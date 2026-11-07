from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from .storage import read_json, write_json
from .transcript_stability_evaluation import (
    build_transcript_reference_binding,
    evaluate_transcript_files,
)

SCHEMA = "video_knowledge_pipeline.asr_stability_benchmark_result.v1"


def evaluate_asr_stability_manifest(
    manifest_path: str | Path,
    *,
    sample_ids: Sequence[str] = (),
    write: bool = True,
) -> dict[str, Any]:
    """Finalize exact-bound ASR stability evidence for completed local runs.

    Intent: make stratified ASR evaluation repeatable after each resumable run.
    Decision: orchestrate the existing exact reference binding and transcript
    stability evaluators; do not implement another metric or ASR pipeline.
    Reason: the old local finalizer compared by title/path without mandatory
    video SHA -> GetNote ID -> reference SHA identity.
    Evidence: transcript_stability_evaluation already fail-closes identity,
    duration, topic fingerprint, prompt leakage, and long-form loss.
    Effective scope: local evaluation artifacts only. References never enter
    prompts, hotwords, routes, correction, or canonical transcript promotion.
    """

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = read_json(manifest_file)
    if not isinstance(manifest, dict):
        raise ValueError("benchmark manifest must be a JSON object")
    selected = {str(value) for value in sample_ids if str(value).strip()}
    samples = [
        sample
        for sample in manifest.get("samples") or []
        if isinstance(sample, dict) and str(sample.get("sample_id") or "").strip()
    ]
    known_ids = {str(sample["sample_id"]) for sample in samples}
    unknown_ids = sorted(selected - known_ids)
    if unknown_ids:
        raise ValueError(f"benchmark sample ids not found: {unknown_ids}")

    # Intent: make --sample-id a true incremental finalizer for large media sets.
    # Decision: recompute selected rows and merge untouched rows from the prior exact result.
    # Reason: rehashing every completed multi-GB video after each new sample caused timeouts.
    # Evidence: the 8-sample finalizer exceeded 180 seconds after xlong-02 completed.
    # Effective scope: explicit selected-sample finalization only; omitting --sample-id revalidates all.
    previous_by_id: dict[str, dict[str, Any]] = {}
    previous_result_path = manifest_file.with_name("benchmark-result.json")
    if selected and previous_result_path.is_file():
        previous = read_json(previous_result_path)
        if (
            isinstance(previous, dict)
            and str(previous.get("schema") or "") == SCHEMA
            and str(previous.get("manifest_path") or "") == str(manifest_file)
        ):
            previous_by_id = {
                str(row.get("sample_id") or ""): dict(row)
                for row in previous.get("samples") or []
                if isinstance(row, dict) and str(row.get("sample_id") or "")
            }

    rows: list[dict[str, Any]] = []
    reused_sample_ids: list[str] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        if not selected or sample_id in selected:
            rows.append(_evaluate_sample(manifest_file.parent, sample, write=write))
            continue
        previous_row = previous_by_id.get(sample_id)
        if previous_row is not None:
            rows.append(previous_row)
            reused_sample_ids.append(sample_id)
        else:
            rows.append(_pending_sample_row(sample))

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("bucket") or "unknown")].append(row)
    completed = sum(row.get("run_status") == "completed" for row in rows)
    passed = sum(row.get("evaluation_status") == "passed" for row in rows)
    # Intent: keep content similarity separate from production readiness.
    # Decision: a degraded merge remains review-required even when old-reference comparison passes.
    # Reason: unmatched overlap boundaries can hide duplication or omission that an aggregate distance masks.
    # Evidence: medium-02 had two unmatched boundaries while content distance still passed at 1.86%.
    # Effective scope: benchmark reporting and bulk-rerun decisions only; canonical transcripts are unchanged.
    production_ready = sum(
        row.get("evaluation_status") == "passed"
        and row.get("quality_status") == "completed"
        for row in rows
    )
    review_required = sum(
        row.get("evaluation_status") == "passed"
        and row.get("quality_status") == "degraded"
        for row in rows
    )
    result = {
        "schema": SCHEMA,
        "manifest_path": str(manifest_file),
        "evaluation_only": True,
        "reference_must_not_enter_prompt_hotwords_routing_or_correction": True,
        "incremental_reuse": bool(selected),
        "recomputed_sample_ids": sorted(selected) if selected else sorted(known_ids),
        "reused_sample_ids": sorted(reused_sample_ids),
        "sample_count": len(rows),
        "completed_count": completed,
        "passed_count": passed,
        "production_ready_count": production_ready,
        "review_required_count": review_required,
        "possible_long_form_loss_count": sum(
            bool(row.get("possible_long_form_loss")) for row in rows
        ),
        "failed_chunk_count": sum(int(row.get("failed_chunk_count") or 0) for row in rows),
        "decision": _decision(rows),
        "buckets": {
            key: {
                "sample_count": len(values),
                "completed_count": sum(row.get("run_status") == "completed" for row in values),
                "passed_count": sum(row.get("evaluation_status") == "passed" for row in values),
                "production_ready_count": sum(
                    row.get("evaluation_status") == "passed"
                    and row.get("quality_status") == "completed"
                    for row in values
                ),
                "review_required_count": sum(
                    row.get("evaluation_status") == "passed"
                    and row.get("quality_status") == "degraded"
                    for row in values
                ),
            }
            for key, values in buckets.items()
        },
        "samples": rows,
    }
    if write:
        output = manifest_file.with_name("benchmark-result.json")
        write_json(output, result)
        markdown = manifest_file.with_name("benchmark-result.md")
        markdown.write_text(_render_markdown(result), encoding="utf-8")
        result["result_path"] = str(output)
        result["markdown_path"] = str(markdown)
    return result


def _pending_sample_row(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(sample.get("sample_id") or ""),
        "bucket": str(sample.get("bucket") or ""),
        "duration": str(sample.get("duration") or ""),
        "media_path": str(sample.get("media_path") or ""),
        "reference_path": str(sample.get("reference_path") or ""),
        "candidate_path": "",
        "run_status": "missing",
        "quality_status": "",
        "successful_chunk_count": 0,
        "failed_chunk_count": 0,
        "runtime_metrics": {},
        "evaluation_status": "pending",
    }


def _evaluate_sample(root: Path, sample: dict[str, Any], *, write: bool) -> dict[str, Any]:
    sample_id = str(sample["sample_id"])
    run_root = root / "runs" / sample_id
    candidate = _latest(run_root, "normalized-transcript.json")
    raw_path = _latest(run_root, "raw-asr-output.json")
    raw = read_json(raw_path) if raw_path else {}
    raw = raw if isinstance(raw, dict) else {}
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "bucket": str(sample.get("bucket") or ""),
        "duration": str(sample.get("duration") or ""),
        "media_path": str(sample.get("media_path") or ""),
        "reference_path": str(sample.get("reference_path") or ""),
        "candidate_path": str(candidate or ""),
        "run_status": str(raw.get("status") or "missing"),
        "quality_status": str(raw.get("quality_status") or ""),
        "successful_chunk_count": int(raw.get("successful_chunk_count") or 0),
        "failed_chunk_count": int(raw.get("failed_chunk_count") or 0),
        "runtime_metrics": dict(raw.get("runtime_metrics") or {}),
        "evaluation_status": "pending",
    }
    if candidate is None or row["run_status"] not in {"completed", "degraded"}:
        return row
    reference = Path(row["reference_path"]).expanduser().resolve()
    media = Path(row["media_path"]).expanduser().absolute()
    binding_path = Path(str(sample.get("reference_binding_path") or root / "reference-bindings" / f"{sample_id}.json")).expanduser().resolve()
    strict_path = Path(str(sample.get("strict_report_path") or root / "comparisons" / sample_id / "strict.json")).expanduser().resolve()
    content_path = Path(str(sample.get("content_report_path") or root / "comparisons" / sample_id / "content.json")).expanduser().resolve()
    # Intent: one invalid legacy reference must not crash the whole stratified batch.
    # Decision: persist the invalid exact binding and skip quality comparison for that sample.
    # Reason: duration/topic identity failure is an evaluation blocker, not an ASR execution failure.
    # Evidence: xlong-01 differed from its reference by >60 container seconds because of a long tail.
    # Effective scope: local evaluation orchestration only; the hard binding gate remains unchanged.
    chunk_manifest_path = Path(str(raw.get("chunk_manifest_path") or "")).expanduser()
    media_identity: dict[str, Any] = {}
    if chunk_manifest_path.is_file():
        chunk_manifest = read_json(chunk_manifest_path)
        source = chunk_manifest.get("source") if isinstance(chunk_manifest, dict) else None
        if isinstance(source, dict):
            media_identity = {
                "path": str(source.get("path") or ""),
                "sha256": str(source.get("sha256") or ""),
                "bytes": int(source.get("bytes") or 0),
                "mtime_ns": int(source.get("mtime_ns") or 0),
                "duration_seconds": float(source.get("duration_seconds") or 0.0),
                "source": "audio_chunk_manifest.v1",
            }
    binding = build_transcript_reference_binding(
        media,
        reference,
        candidate_path=candidate,
        media_identity=media_identity or None,
        allow_unavailable_media_identity=bool(media_identity),
        allow_invalid=True,
    )
    binding_status = str(binding.get("status") or "invalid")
    if binding_status != "active":
        if write:
            binding_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(binding_path, binding)
        row.update(
            {
                "evaluation_status": "reference_binding_invalid",
                "reference_binding_status": binding_status,
                "reference_binding_reasons": list(
                    (binding.get("creation_validation") or {}).get("reasons") or []
                ),
                "reference_binding_path": str(binding_path),
            }
        )
        return row
    strict = evaluate_transcript_files(
        candidate,
        reference,
        normalization_profile="strict_v1",
        reference_binding=binding,
        media_path=media,
        media_identity=media_identity or None,
        require_reference_binding=True,
    )
    content = evaluate_transcript_files(
        candidate,
        reference,
        normalization_profile="content_vocal_fillers_v1",
        reference_binding=binding,
        media_path=media,
        media_identity=media_identity or None,
        require_reference_binding=True,
    )
    if write:
        binding_path.parent.mkdir(parents=True, exist_ok=True)
        strict_path.parent.mkdir(parents=True, exist_ok=True)
        content_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(binding_path, binding)
        write_json(strict_path, strict)
        write_json(content_path, content)
    row.update(
        {
            "evaluation_status": str(content.get("status") or "failed"),
            "strict_distance": float((strict.get("metric") or {}).get("value") or 0.0),
            "content_distance": float((content.get("metric") or {}).get("value") or 0.0),
            "possible_long_form_loss": bool((content.get("completion") or {}).get("possible_long_form_loss")),
            "window_count": int((content.get("comparison_windows") or {}).get("window_count") or 0),
            "failed_window_count": int((content.get("comparison_windows") or {}).get("failed_window_count") or 0),
            "reference_binding_status": str((content.get("reference_binding") or {}).get("status") or ""),
            "reference_binding_path": str(binding_path),
            "strict_report_path": str(strict_path),
            "content_report_path": str(content_path),
        }
    )
    return row


def _latest(root: Path, name: str) -> Path | None:
    values = sorted(root.rglob(name), key=lambda path: path.stat().st_mtime, reverse=True)
    return values[0] if values else None


def _decision(rows: list[dict[str, Any]]) -> str:
    if not rows or any(row.get("run_status") not in {"completed", "degraded"} for row in rows):
        return "benchmark_in_progress"
    if any(int(row.get("failed_chunk_count") or 0) for row in rows):
        return "do_not_bulk_rerun_failed_chunks"
    if any(bool(row.get("possible_long_form_loss")) for row in rows):
        return "do_not_bulk_rerun_possible_long_form_loss"
    if any(row.get("evaluation_status") != "passed" for row in rows):
        return "review_failed_samples_before_bulk_rerun"
    if any(row.get("quality_status") == "degraded" for row in rows):
        return "review_overlap_boundaries_before_bulk_rerun"
    return "await_anonymous_blind_review_before_bulk_rerun"


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# ASR 分层稳定性基准",
        "",
        f"- 完成：`{result['completed_count']}/{result['sample_count']}`",
        f"- 内容对比通过：`{result['passed_count']}/{result['sample_count']}`",
        f"- 无需边界复核：`{result['production_ready_count']}/{result['sample_count']}`",
        f"- 需要边界复核：`{result['review_required_count']}`",
        f"- 疑似长内容缺失：`{result['possible_long_form_loss_count']}`",
        f"- 当前决策：`{result['decision']}`",
        "",
        "| 样本 | 档位 | 运行 | 内容差异 | 局部失败 | 长内容缺失 | GPU 峰值 MiB |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in result.get("samples") or []:
        distance = f"{float(row['content_distance']):.4%}" if "content_distance" in row else "-"
        metrics = row.get("runtime_metrics") if isinstance(row.get("runtime_metrics"), dict) else {}
        peak = metrics.get("max_cuda_peak_memory_allocated_mib", "-")
        lines.append(
            f"| {row.get('sample_id', '')} | {row.get('bucket', '')} | {row.get('run_status', '')} | "
            f"{distance} | {row.get('failed_window_count', '-')} | {row.get('possible_long_form_loss', '-')} | {peak} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize exact-bound local ASR stability evidence")
    parser.add_argument("manifest_path")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--no-write", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_asr_stability_manifest(
        args.manifest_path,
        sample_ids=args.sample_id,
        write=not args.no_write,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["decision"] != "benchmark_in_progress" else 2


if __name__ == "__main__":
    raise SystemExit(main())