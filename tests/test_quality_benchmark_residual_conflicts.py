from __future__ import annotations

import json
from pathlib import Path

import video_knowledge_pipeline.quality_benchmark_residual_conflicts as residual_module
from video_knowledge_pipeline.quality_benchmark_residual_conflicts import (
    DISCOURSE_PARTICLE_EQUIVALENT,
    ENTITY_LEXICON_RESOLVED,
    NUMERIC_FORMAT_EQUIVALENT,
    PUNCTUATION_ONLY,
    RESIDUAL_TRUE_CONFLICT,
    build_quality_benchmark_residual_conflicts,
)


def _difference(
    diff_id: str,
    candidate_a: str,
    candidate_b: str,
    *,
    context_before: str = "前文",
    context_after: str = "后文",
    operation: str = "replace",
) -> dict:
    return {
        "diff_id": diff_id,
        "cluster_id": f"cluster-{diff_id}",
        "operation": operation,
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "context_before": context_before,
        "context_after": context_after,
        "estimated_time": {"start": 12.0, "end": 13.0},
    }


def _pack(path: Path, differences: list[dict]) -> Path:
    clusters = [
        {
            "cluster_id": row["cluster_id"],
            "diff_ids": [row["diff_id"]],
            "time_range": "00:00:12.000 - 00:00:13.000",
            "candidate_a": row["candidate_a"],
            "candidate_b": row["candidate_b"],
            "context_before": row["context_before"],
            "context_after": row["context_after"],
            "audio_review_window": {
                "start": 0.0,
                "end": 20.0,
                "audio_clip_path": str(path.parent / "clip.wav"),
            },
        }
        for row in differences
    ]
    path.write_text(
        json.dumps(
            {
                "schema": "video_knowledge_pipeline.quality_benchmark_arbitration_pack.v1",
                "samples": [
                    {
                        "sample_id": "sample-01",
                        "category": "proper_noun_or_tool",
                        "start_seconds": 0.0,
                        "end_seconds": 20.0,
                        "audio_clip_path": str(path.parent / "clip.wav"),
                        "differences": differences,
                        "clusters": clusters,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_filters_punctuation_only_difference(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack.json", [_difference("d1", "你好，", "你好")])

    result = build_quality_benchmark_residual_conflicts(pack, write=True)

    row = result["samples"][0]["differences"][0]
    assert row["classification"] == PUNCTUATION_ONLY
    assert result["residual_difference_count"] == 0
    assert result["residual_cluster_count"] == 0
    todo = json.loads((tmp_path / "quality-benchmark-residual-conflicts.todo.json").read_text(encoding="utf-8"))
    assert todo["rows"] == []
    assert todo["operator_boundary"]["contains_no_answers"] is True


def test_filters_unit_aware_numeric_format_equivalent(tmp_path: Path) -> None:
    pack = _pack(
        tmp_path / "pack.json",
        [_difference("d1", "两百", "200", context_before="保费有", context_after="多万元")],
    )

    result = build_quality_benchmark_residual_conflicts(pack, write=False)

    row = result["samples"][0]["differences"][0]
    assert row["classification"] == NUMERIC_FORMAT_EQUIVALENT
    assert row["numeric_evidence"]["candidate_a_keys"] == ["currency:2000000:元"]
    assert row["numeric_evidence"]["candidate_b_keys"] == ["currency:2000000:元"]
    assert result["residual_difference_count"] == 0


def test_filters_bare_numeric_format_equivalents(tmp_path: Path) -> None:
    differences = [
        _difference("d100", "100", "一百"),
        _difference("d300", "300", "三百"),
        _difference("d8000", "8000", "八千"),
        _difference("dpercent", "30%", "百分之三十"),
    ]
    pack = _pack(tmp_path / "pack.json", differences)

    result = build_quality_benchmark_residual_conflicts(pack, write=False)

    assert {row["classification"] for row in result["samples"][0]["differences"]} == {
        NUMERIC_FORMAT_EQUIVALENT
    }
    assert result["residual_difference_count"] == 0


def test_filters_only_conservative_discourse_particles_and_fillers(tmp_path: Path) -> None:
    differences = [
        _difference("empty", "啊", "", operation="delete"),
        _difference("replace", "嗯，", "呃"),
        _difference("repeat", "啊嗯", "哦"),
    ]
    pack = _pack(tmp_path / "pack.json", differences)

    result = build_quality_benchmark_residual_conflicts(pack, write=False)

    assert {row["classification"] for row in result["samples"][0]["differences"]} == {
        DISCOURSE_PARTICLE_EQUIVALENT
    }
    assert result["classification_counts"][DISCOURSE_PARTICLE_EQUIVALENT] == 3
    assert result["residual_difference_count"] == 0


def test_discourse_filter_keeps_content_words_for_review(tmp_path: Path) -> None:
    differences = [
        _difference("content", "好", "啊"),
        _difference("demonstrative", "这个", "那个"),
        _difference("lexical", "看", "开"),
    ]
    pack = _pack(tmp_path / "pack.json", differences)

    result = build_quality_benchmark_residual_conflicts(pack, write=False)

    assert all(
        row["classification"] == RESIDUAL_TRUE_CONFLICT
        for row in result["samples"][0]["differences"]
    )
    assert result["residual_difference_count"] == 3


def test_explicit_entity_alias_resolves_to_canonical(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "entity-lexicon.json").write_text(
        json.dumps(
            {
                "terms": [
                    {
                        "canonical": "明亚",
                        "aliases": ["明亚", "米娅", "名娅"],
                        "source_types": ["base_lexicon"],
                        "review_required": False,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pack = _pack(tmp_path / "pack.json", [_difference("d1", "米娅", "明亚")])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "sample-01",
                        "bundle_dir": str(bundle),
                        "reference_text": "SECRET_REFERENCE_MUST_NOT_LEAK",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = build_quality_benchmark_residual_conflicts(
        pack,
        manifest_json=manifest,
        write=True,
    )

    row = result["samples"][0]["differences"][0]
    assert row["classification"] == ENTITY_LEXICON_RESOLVED
    assert row["recommended_canonical"] == "明亚"
    assert row["entity_evidence"]["review_required"] is False
    artifacts = "\n".join(
        (tmp_path / name).read_text(encoding="utf-8")
        for name in (
            "quality-benchmark-residual-conflicts.json",
            "quality-benchmark-residual-conflicts.md",
            "quality-benchmark-residual-conflicts.todo.json",
        )
    )
    assert "SECRET_REFERENCE_MUST_NOT_LEAK" not in artifacts
    assert "reference_text" not in artifacts


def test_explicit_entity_lexicon_works_without_manifest_or_bundle_copy(tmp_path: Path) -> None:
    lexicon = tmp_path / "shared-entity-lexicon.json"
    lexicon.write_text(
        json.dumps(
            {
                "terms": [
                    {
                        "canonical": "明亚",
                        "aliases": ["明亚", "米娅"],
                        "source_types": ["base_lexicon"],
                        "review_required": False,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pack = _pack(tmp_path / "pack.json", [_difference("d1", "米娅", "明亚")])

    result = build_quality_benchmark_residual_conflicts(
        pack,
        entity_lexicon_json=lexicon,
        write=False,
    )

    row = result["samples"][0]["differences"][0]
    assert row["classification"] == ENTITY_LEXICON_RESOLVED
    assert row["recommended_canonical"] == "明亚"
    assert result["manifest_used_for_bundle_lookup"] is False
    assert result["explicit_entity_lexicon_path"] == str(lexicon.resolve())
    assert result["samples"][0]["entity_lexicon_paths"] == [str(lexicon.resolve())]


def test_review_required_dynamic_entity_remains_true_conflict(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "entity-lexicon.json").write_text(
        json.dumps(
            {
                "terms": [
                    {
                        "canonical": "明亚",
                        "aliases": ["明亚", "米娅"],
                        "source_types": ["metadata"],
                        "review_required": True,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pack = _pack(tmp_path / "pack.json", [_difference("d1", "米娅", "明亚")])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"samples": [{"sample_id": "sample-01", "bundle_dir": str(bundle)}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = build_quality_benchmark_residual_conflicts(pack, manifest_json=manifest, write=False)

    row = result["samples"][0]["differences"][0]
    assert row["classification"] == RESIDUAL_TRUE_CONFLICT
    assert row["entity_evidence"]["resolved"] is False
    assert "dynamic_entity_requires_review" in row["risk_reasons"]
    assert result["residual_difference_count"] == 1


def test_keeps_unresolved_lexical_conflict_with_review_evidence(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack.json", [_difference("d1", "看", "开")])

    result = build_quality_benchmark_residual_conflicts(pack, write=True)

    row = result["residual_conflicts"][0]
    cluster = result["residual_clusters"][0]
    assert row["classification"] == RESIDUAL_TRUE_CONFLICT
    assert row["sample_id"] == "sample-01"
    assert row["cluster_id"] == "cluster-d1"
    assert row["estimated_time"] == {"start": 12.0, "end": 13.0}
    assert row["audio_clip_path"].endswith("clip.wav")
    assert "lexical_content_conflict" in row["risk_reasons"]
    assert cluster["residual_diff_ids"] == ["d1"]
    todo = json.loads((tmp_path / "quality-benchmark-residual-conflicts.todo.json").read_text(encoding="utf-8"))
    assert todo["rows"] == [
        {
            "sample_id": "sample-01",
            "cluster_id": "cluster-d1",
            "diff_ids": ["d1"],
            "choice": "",
            "confidence": None,
            "evidence_refs": [],
            "reason": "",
        }
    ]


def test_manifest_reference_field_is_never_accessed(monkeypatch, tmp_path: Path) -> None:
    class GuardedSample(dict):
        def get(self, key, default=None):
            if key == "reference_text":
                raise AssertionError("human reference must not be read")
            return super().get(key, default)

    pack_path = tmp_path / "pack.json"
    manifest_path = tmp_path / "manifest.json"
    pack_payload = json.loads(
        _pack(pack_path, [_difference("d1", "看", "开")]).read_text(encoding="utf-8")
    )
    manifest_payload = {
        "samples": [
            GuardedSample(
                sample_id="sample-01",
                bundle_dir=str(tmp_path / "missing-bundle"),
                reference_text="SECRET_REFERENCE_MUST_NOT_BE_READ",
            )
        ]
    }
    original_read_json = residual_module.read_json

    def guarded_read_json(path):
        resolved = Path(path).resolve()
        if resolved == pack_path.resolve():
            return pack_payload
        if resolved == manifest_path.resolve():
            return manifest_payload
        return original_read_json(path)

    monkeypatch.setattr(residual_module, "read_json", guarded_read_json)

    result = build_quality_benchmark_residual_conflicts(
        pack_path,
        manifest_json=manifest_path,
        write=False,
    )

    assert result["operator_boundary"]["human_reference_not_read"] is True
    assert "SECRET_REFERENCE_MUST_NOT_BE_READ" not in json.dumps(result, ensure_ascii=False)


def test_cli_parses_residual_conflict_action() -> None:
    from video_knowledge_pipeline.cli import build_parser

    args = build_parser().parse_args(
        [
            "quality-benchmark",
            "build-residual-conflicts",
            "pack.json",
            "--manifest-json",
            "manifest.json",
            "--entity-lexicon-json",
            "lexicon.json",
        ]
    )
    assert args.action == "build-residual-conflicts"
    assert args.manifest_json == "manifest.json"
    assert args.entity_lexicon_json == "lexicon.json"


def test_mcp_quality_benchmark_declares_entity_lexicon_parameter() -> None:
    import inspect

    from video_knowledge_pipeline import mcp_server

    source = inspect.getsource(mcp_server.main)
    assert 'entity_lexicon_json: str = ""' in source
    assert "entity_lexicon_json=entity_lexicon_json or None" in source
