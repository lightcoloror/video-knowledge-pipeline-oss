from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.quality_benchmark_arbitration import (
    build_quality_benchmark_arbitration,
    evaluate_quality_benchmark_arbitration,
)


def _write_transcript(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"segments": [{"index": 1, "start": 0.0, "end": 10.0, "text": text}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return str(path)


def _manifest(tmp_path: Path) -> Path:
    primary = _write_transcript(tmp_path / "primary.json", "今天讲小红书获客三百人")
    secondary = _write_transcript(tmp_path / "secondary.json", "今天讲小红书获客三百名")
    manifest = tmp_path / "quality-benchmark-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "sample-01",
                        "category": "numbers_or_amounts",
                        "start_seconds": 10.0,
                        "end_seconds": 20.0,
                        "audio_clip_path": str(tmp_path / "clip.wav"),
                        "reference_text": "今天讲小红书获客三百人",
                        "variants": {
                            "sensevoice_full_punc": primary,
                            "qwen3_asr_1_7b": secondary,
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest


def test_arbitration_public_and_private_packs_exclude_reference(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    result = build_quality_benchmark_arbitration(manifest, write=True)

    public_text = (tmp_path / "quality-benchmark-arbitration-pack.json").read_text(encoding="utf-8")
    private_text = (tmp_path / "quality-benchmark-arbitration.private.json").read_text(encoding="utf-8")
    assert result["difference_count"] == 1
    assert "reference_text" not in public_text
    assert "reference_text" not in private_text
    assert "candidate_a" in public_text
    assert "primary_variant" not in public_text
    assert result["operator_boundary"]["human_reference_excluded"] is True


def test_arbitration_ignores_punctuation_only_difference(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    secondary = Path(payload["samples"][0]["variants"]["qwen3_asr_1_7b"])
    _write_transcript(secondary, "今天讲小红书获客三百人。")

    result = build_quality_benchmark_arbitration(manifest, write=False)

    assert result["difference_count"] == 0


def test_oracle_is_explicitly_evaluation_only(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    build_quality_benchmark_arbitration(manifest, write=True)

    result = evaluate_quality_benchmark_arbitration(manifest, write=False)

    assert result["sample_count"] == 1
    assert result["metrics"]["local_patch_oracle"]["mean_cer"] == 0.0
    assert result["operator_boundary"]["oracle_is_evaluation_only_never_production"] is True
    assert result["operator_boundary"]["human_reference_used_only_by_evaluator"] is True


def test_reviewed_change_without_evidence_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    pack = build_quality_benchmark_arbitration(manifest, write=True)
    diff = pack["samples"][0]["differences"][0]
    private = json.loads((tmp_path / "quality-benchmark-arbitration.private.json").read_text(encoding="utf-8"))
    private_diff = private["samples"][0]["differences"][0]
    secondary_choice = "A" if diff["candidate_a"] == private_diff["secondary_text"] else "B"
    decisions = tmp_path / "decisions.json"
    decisions.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "diff_id": diff["diff_id"],
                        "choice": secondary_choice,
                        "confidence": 0.99,
                        "evidence_refs": [],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = evaluate_quality_benchmark_arbitration(
        manifest,
        decisions_json=decisions,
        write=False,
    )

    assert result["rows"][0]["reviewed_patch_count"] == 0
    assert result["rejected_decisions"] == [
        {"diff_id": diff["diff_id"], "reason": "confidence_or_evidence_gate"}
    ]
