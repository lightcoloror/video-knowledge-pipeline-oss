from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.asr_stability_benchmark import (
    evaluate_asr_stability_manifest,
)


def test_batch_evaluation_requires_exact_binding_and_surfaces_runtime_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "benchmark"
    run = root / "runs" / "short-01" / "transcripts" / "transcript_x"
    raw_run = root / "runs" / "short-01" / "transcripts" / "asr_run_x"
    run.mkdir(parents=True)
    raw_run.mkdir(parents=True)
    candidate = run / "normalized-transcript.json"
    candidate.write_text(json.dumps({"segments": [{"start": 0, "end": 1, "text": "正文"}]}), encoding="utf-8")
    (raw_run / "raw-asr-output.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "quality_status": "completed",
                "successful_chunk_count": 1,
                "failed_chunk_count": 0,
                "runtime_metrics": {
                    "max_cuda_peak_memory_allocated_mib": 2048.0,
                    "total_child_elapsed_seconds": 12.5,
                },
            }
        ),
        encoding="utf-8",
    )
    media = tmp_path / "lesson.mp4"
    reference = tmp_path / "reference.md"
    media.write_bytes(b"media")
    reference.write_text("reference", encoding="utf-8")
    manifest = root / "benchmark-manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "short-01",
                        "bucket": "short",
                        "duration": "00:00:01",
                        "media_path": str(media),
                        "reference_path": str(reference),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict] = []

    def fake_binding(*args, **kwargs):
        calls.append({"kind": "binding", "args": args, "kwargs": kwargs})
        return {"schema": "binding", "status": "active"}

    def fake_evaluate(*args, **kwargs):
        calls.append({"kind": "evaluation", "args": args, "kwargs": kwargs})
        profile = kwargs["normalization_profile"]
        return {
            "status": "passed",
            "metric": {"value": 0.01 if profile == "strict_v1" else 0.02},
            "completion": {"possible_long_form_loss": False},
            "comparison_windows": {"window_count": 2, "failed_window_count": 1},
            "reference_binding": {"status": "valid"},
        }

    monkeypatch.setattr(
        "video_knowledge_pipeline.asr_stability_benchmark.build_transcript_reference_binding",
        fake_binding,
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.asr_stability_benchmark.evaluate_transcript_files",
        fake_evaluate,
    )

    result = evaluate_asr_stability_manifest(manifest, write=True)

    assert result["completed_count"] == 1
    assert result["passed_count"] == 1
    assert result["production_ready_count"] == 1
    assert result["review_required_count"] == 0
    assert result["decision"] == "await_anonymous_blind_review_before_bulk_rerun"
    row = result["samples"][0]
    assert row["runtime_metrics"]["max_cuda_peak_memory_allocated_mib"] == 2048.0
    assert row["failed_window_count"] == 1
    assert sum(call["kind"] == "binding" for call in calls) == 1
    evaluations = [call for call in calls if call["kind"] == "evaluation"]
    assert len(evaluations) == 2
    assert all(call["kwargs"]["require_reference_binding"] is True for call in evaluations)
    assert all(call["kwargs"]["reference_binding"]["status"] == "active" for call in evaluations)
    assert (root / "reference-bindings" / "short-01.json").exists()
    assert (root / "comparisons" / "short-01" / "strict.json").exists()
    assert (root / "comparisons" / "short-01" / "content.json").exists()
    assert (root / "benchmark-result.md").exists()

def test_batch_evaluation_separates_content_pass_from_degraded_run(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "benchmark"
    run = root / "runs" / "medium-02" / "transcripts" / "transcript_x"
    raw_run = root / "runs" / "medium-02" / "transcripts" / "asr_run_x"
    run.mkdir(parents=True)
    raw_run.mkdir(parents=True)
    (run / "normalized-transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 1, "text": "正文"}]}),
        encoding="utf-8",
    )
    (raw_run / "raw-asr-output.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "quality_status": "degraded",
                "successful_chunk_count": 1,
                "failed_chunk_count": 0,
            }
        ),
        encoding="utf-8",
    )
    media = tmp_path / "lesson.mp4"
    reference = tmp_path / "reference.md"
    media.write_bytes(b"media")
    reference.write_text("reference", encoding="utf-8")
    manifest = root / "benchmark-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "medium-02",
                        "bucket": "medium",
                        "duration": "00:01:00",
                        "media_path": str(media),
                        "reference_path": str(reference),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "video_knowledge_pipeline.asr_stability_benchmark.build_transcript_reference_binding",
        lambda *args, **kwargs: {"schema": "binding", "status": "active"},
    )
    monkeypatch.setattr(
        "video_knowledge_pipeline.asr_stability_benchmark.evaluate_transcript_files",
        lambda *args, **kwargs: {
            "status": "passed",
            "metric": {"value": 0.02},
            "completion": {"possible_long_form_loss": False},
            "comparison_windows": {"window_count": 2, "failed_window_count": 1},
            "reference_binding": {"status": "valid"},
        },
    )

    result = evaluate_asr_stability_manifest(manifest, write=False)

    assert result["passed_count"] == 1
    assert result["production_ready_count"] == 0
    assert result["review_required_count"] == 1
    assert result["decision"] == "review_overlap_boundaries_before_bulk_rerun"

def test_invalid_reference_binding_is_recorded_without_crashing_batch(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "benchmark"
    run = root / "runs" / "xlong-01" / "transcripts" / "transcript_x"
    raw_run = root / "runs" / "xlong-01" / "transcripts" / "asr_run_x"
    run.mkdir(parents=True)
    raw_run.mkdir(parents=True)
    (run / "normalized-transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 1, "text": "正文"}]}),
        encoding="utf-8",
    )
    (raw_run / "raw-asr-output.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "quality_status": "completed",
                "successful_chunk_count": 1,
                "failed_chunk_count": 0,
            }
        ),
        encoding="utf-8",
    )
    media = tmp_path / "lesson.mp4"
    reference = tmp_path / "reference.md"
    media.write_bytes(b"media")
    reference.write_text("reference", encoding="utf-8")
    manifest = root / "benchmark-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "xlong-01",
                        "bucket": "extra_long",
                        "duration": "01:30:00",
                        "media_path": str(media),
                        "reference_path": str(reference),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "video_knowledge_pipeline.asr_stability_benchmark.build_transcript_reference_binding",
        lambda *args, **kwargs: {
            "schema": "binding",
            "status": "invalid",
            "creation_validation": {
                "status": "invalid",
                "reasons": ["reference_duration_mismatch"],
            },
        },
    )

    def fail_if_evaluated(*args, **kwargs):
        raise AssertionError("invalid reference must not enter quality evaluation")

    monkeypatch.setattr(
        "video_knowledge_pipeline.asr_stability_benchmark.evaluate_transcript_files",
        fail_if_evaluated,
    )

    result = evaluate_asr_stability_manifest(manifest, write=True)

    assert result["completed_count"] == 1
    assert result["passed_count"] == 0
    assert result["decision"] == "review_failed_samples_before_bulk_rerun"
    row = result["samples"][0]
    assert row["evaluation_status"] == "reference_binding_invalid"
    assert row["reference_binding_reasons"] == ["reference_duration_mismatch"]
    assert Path(row["reference_binding_path"]).exists()

def test_selected_sample_incrementally_merges_existing_exact_rows(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "benchmark"
    root.mkdir()
    manifest = root / "benchmark-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {"sample_id": "short-01", "bucket": "short"},
                    {"sample_id": "xlong-02", "bucket": "extra_long"},
                ]
            }
        ),
        encoding="utf-8",
    )
    previous_row = {
        "sample_id": "short-01",
        "bucket": "short",
        "run_status": "completed",
        "quality_status": "completed",
        "evaluation_status": "passed",
        "failed_chunk_count": 0,
        "possible_long_form_loss": False,
    }
    (root / "benchmark-result.json").write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.asr_stability_benchmark_result.v1",
                "manifest_path": str(manifest.resolve()),
                "samples": [previous_row],
            }
        ),
        encoding="utf-8",
    )

    evaluated_ids: list[str] = []

    def fake_evaluate(_root, sample, *, write):
        evaluated_ids.append(str(sample["sample_id"]))
        return {
            "sample_id": str(sample["sample_id"]),
            "bucket": str(sample["bucket"]),
            "run_status": "completed",
            "quality_status": "completed",
            "evaluation_status": "passed",
            "failed_chunk_count": 0,
            "possible_long_form_loss": False,
        }

    monkeypatch.setattr(
        "video_knowledge_pipeline.asr_stability_benchmark._evaluate_sample",
        fake_evaluate,
    )

    result = evaluate_asr_stability_manifest(
        manifest,
        sample_ids=["xlong-02"],
        write=True,
    )

    assert evaluated_ids == ["xlong-02"]
    assert result["sample_count"] == 2
    assert result["completed_count"] == 2
    assert result["recomputed_sample_ids"] == ["xlong-02"]
    assert result["reused_sample_ids"] == ["short-01"]
    assert [row["sample_id"] for row in result["samples"]] == [
        "short-01",
        "xlong-02",
    ]
