from __future__ import annotations

import json
from pathlib import Path

from video_knowledge_pipeline.quality_benchmark_punctuation import run_quality_benchmark_punctuation


def _manifest(tmp_path: Path) -> Path:
    transcript = tmp_path / "source.json"
    transcript.write_text(
        json.dumps({"segments": [{"start": 0, "end": 10, "text": "大家好今天讲获客首先建立信任"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "sample-01",
                        "start_seconds": 0,
                        "end_seconds": 10,
                        "reference_text": "大家好，今天讲获客。首先，建立信任。",
                        "variants": {"sensevoice_full_punc": str(transcript)},
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest


def test_punctuation_benchmark_preview_never_runs_model(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    result = run_quality_benchmark_punctuation(
        manifest,
        execute=False,
        write=False,
        model_runner=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    assert result["status"] == "planned"
    assert result["operator_boundary"]["reference_not_sent_to_model"] is True


def test_punctuation_benchmark_scores_character_locked_result(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    def runner(root, blocks, **kwargs):
        assert blocks[0][0].text == "大家好今天讲获客首先建立信任"
        return ["大家好，今天讲获客。首先，建立信任。"], {"device": "cuda"}

    result = run_quality_benchmark_punctuation(
        manifest,
        execute=True,
        write=True,
        model_runner=runner,
    )

    assert result["status"] == "completed"
    assert result["char_lock_passed"] is True
    assert result["metrics"]["content_character_mutation_rate"] == 0.0
    assert result["metrics"]["candidate"]["punctuation_f1"] == 1.0
    assert result["metrics"]["candidate"]["sentence_boundary_f1"] == 1.0
    assert Path(result["rows"][0]["candidate_path"]).exists()


def test_punctuation_benchmark_blocks_text_mutation(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    result = run_quality_benchmark_punctuation(
        manifest,
        execute=True,
        write=True,
        model_runner=lambda *args, **kwargs: (["大家好，今天讲销售。"], {"device": "cpu"}),
    )

    assert result["status"] == "completed_with_blockers"
    assert result["char_lock_passed"] is False
    assert result["metrics"]["content_character_mutation_rate"] > 0
