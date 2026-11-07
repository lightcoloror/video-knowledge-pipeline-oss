from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.cli import build_parser
from video_knowledge_pipeline.transcript_candidate_recall_benchmark import benchmark_transcript_candidate_recall


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_candidate_recall_benchmark_reports_missed_semantic_targets(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_json(
        bundle / "manifest.json",
        {
            "title": "lesson",
            "normalized_transcript_json": "normalized-transcript.json",
            "transcript_semantic_correction_pack_json": "transcript-semantic-correction-pack.json",
            "evidence_conflict_index_json": "evidence-conflict-index.json",
        },
    )
    _write_json(
        bundle / "normalized-transcript.json",
        {
            "segments": [
                {
                    "start": 268,
                    "end": 402,
                    "text": "我可以帮你看是否会有一些买虫的。这里采用二则一的方式约时间。",
                }
            ]
        },
    )
    reference = tmp_path / "getbrain-reference.json"
    _write_json(
        reference,
        {
            "segments": [
                {
                    "start": 268,
                    "end": 402,
                    "text": "我可以帮你看是否会有一些买重的。这里采用二择一的方式约时间。",
                }
            ]
        },
    )
    targets = tmp_path / "targets.json"
    _write_json(
        targets,
        [
            {"original_text": "买虫", "corrected_text": "买重"},
            {"original_text": "二则一", "corrected_text": "二择一"},
        ],
    )
    _write_json(
        bundle / "transcript-semantic-correction-pack.json",
        {"candidates": [{"candidate_id": "c1", "original_text": "买虫", "candidate_text": "买重"}]},
    )
    _write_json(
        bundle / "evidence-conflict-index.json",
        {"conflicts": [{"candidate_id": "c1", "original_text": "买虫", "candidate_text": "买重"}]},
    )
    asr_dir = tmp_path / "asr"
    _write_json(asr_dir / "raw.json", {"segments": [{"start": 268, "end": 402, "text": "是否会有一些买虫的这里采用二则一的方式"}]})
    _write_json(asr_dir / "full.json", {"segments": [{"start": 268, "end": 402, "text": "是否会有一些买重的。这里采用二择一的方式。"}]})
    _write_json(
        asr_dir / "asr-ab-sample-run.json",
        {
            "variants": [
                {"key": "sensevoice_raw", "status": "ok", "normalized_json": str(asr_dir / "raw.json")},
                {"key": "sensevoice_full_punc", "status": "ok", "normalized_json": str(asr_dir / "full.json")},
            ]
        },
    )

    result = benchmark_transcript_candidate_recall(
        bundle,
        reference_transcript=reference,
        target_pairs_json=targets,
        asr_ab_run_json=asr_dir / "asr-ab-sample-run.json",
        start_seconds=268,
        end_seconds=402,
        write=True,
    )

    assert result["status"] == "candidate_recall_gap"
    assert result["summary"]["active_or_reference_supported_target_count"] == 2
    assert result["summary"]["candidate_recall_count"] == 1
    assert result["summary"]["missed_target_count"] == 1
    missed = [row for row in result["targets"] if row["missed"]]
    assert missed[0]["original_text"] == "二则一"
    assert result["asr_variants"][0]["key"] == "sensevoice_full_punc"
    assert result["operator_boundary"]["reference_transcript_is_evaluation_only"] is True
    assert (bundle / "transcript-candidate-recall-benchmark.md").exists()


def test_cli_candidate_recall_benchmark_parse() -> None:
    args = build_parser().parse_args(
        [
            "transcript-candidate-recall-benchmark",
            "bundle",
            "--reference-transcript",
            "reference.json",
            "--target-pairs-json",
            "targets.json",
            "--start-seconds",
            "268",
            "--end-seconds",
            "402",
            "--no-write",
        ]
    )

    assert args.command == "transcript-candidate-recall-benchmark"
    assert args.reference_transcript == "reference.json"
    assert args.start_seconds == 268
