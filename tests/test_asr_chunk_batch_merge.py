from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.asr_vad_activity_audit import SCHEMA as AUDIT_SCHEMA
from video_knowledge_pipeline.asr_chunk_batch_merge import (
    merge_asr_chunk_batch_reports,
)
from video_knowledge_pipeline.asr_chunk_batch_workflow import (
    build_asr_chunk_batch_workflow,
)
from video_knowledge_pipeline.asr_vad_chunking import SCHEMA as CHUNK_SCHEMA
from video_knowledge_pipeline.cli import main as cli_main
from video_knowledge_pipeline.consented_model_batch import SCHEMA as BATCH_SCHEMA
from video_knowledge_pipeline.model_connector_consent import (
    create_model_connector_consent,
)
from video_knowledge_pipeline.storage import read_json, write_json


PROVIDER = {
    "provider": "custom_openai_compatible_asr",
    "base_url": "https://asr.example/v1",
    "model": "whisper-test",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _suite(
    tmp_path: Path,
    *,
    with_activity_audit: bool = False,
    bundle_dir: Path | None = None,
) -> tuple[Path, list[Path], list[Path]]:
    source_media = tmp_path / "source.mp4"
    source_media.write_bytes(b"source-media-for-chunk-merge")
    vad_path = tmp_path / "vad.json"
    write_json(vad_path, {"segments": [{"id": "vad-1", "start": 0, "end": 20}]})
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    chunks: list[dict[str, object]] = []
    artifacts: list[Path] = []
    boundaries = [
        (0.0, 10.0, 0.0, 11.5),
        (10.0, 20.0, 8.5, 20.0),
    ]
    for position, (core_start, core_end, artifact_start, artifact_end) in enumerate(
        boundaries, start=1
    ):
        artifact = chunk_dir / f"asr-chunk-{position:04d}.mp3"
        artifact.write_bytes(f"chunk-{position}".encode())
        artifacts.append(artifact)
        chunks.append(
            {
                "chunk_id": f"asr-chunk-{position:04d}",
                "position": position,
                "status": "completed",
                "core_start": core_start,
                "core_end": core_end,
                "artifact_start": artifact_start,
                "artifact_end": artifact_end,
                "output_path": str(artifact.resolve()),
                "output_bytes": artifact.stat().st_size,
                "output_sha256": _sha256(artifact),
            }
        )
    manifest_path = chunk_dir / "asr-vad-chunk-manifest.json"
    write_json(
        manifest_path,
        {
            "schema": CHUNK_SCHEMA,
            "status": "completed",
            "ok": True,
            "source_path": str(source_media.resolve()),
            "source_bytes": source_media.stat().st_size,
            "source_sha256": _sha256(source_media),
            "vad_json": str(vad_path.resolve()),
            "vad_sha256": _sha256(vad_path),
            "chunk_count": 2,
            "completed_chunk_count": 2,
            "failed_chunk_count": 0,
            "chunks": chunks,
        },
    )
    consents: list[Path] = []
    for position, artifact in enumerate(artifacts, start=1):
        consent_path = tmp_path / "consents" / f"chunk-{position:04d}.json"
        result = create_model_connector_consent(
            tmp_path,
            task="cloud_asr",
            artifact_paths=[artifact],
            provider_config=PROVIDER,
            output_path=consent_path,
            max_calls=1,
            max_estimated_cost_usd=0.01,
            max_cost_per_call_usd=0.01,
            max_retries_per_call=0,
            confirm_data_export=True,
        )
        consents.append(Path(result["consent_path"]))
    audit_path: Path | None = None
    if with_activity_audit:
        audit_path = tmp_path / "asr-vad-activity-audit.json"
        write_json(
            audit_path,
            {
                "schema": AUDIT_SCHEMA,
                "status": "passed",
                "vad_coverage_verified": True,
                "candidate_gap_count": 0,
                "source_media": {"sha256": _sha256(source_media)},
                "vad_sha256": _sha256(vad_path),
            },
        )
    workflow = build_asr_chunk_batch_workflow(
        manifest_path,
        consents,
        activity_audit_path=audit_path,
        bundle_dir=bundle_dir,
    )
    return Path(workflow["output_path"]), artifacts, consents


def _report(
    path: Path,
    consent_path: Path,
    segments: list[dict[str, object]],
) -> Path:
    consent = read_json(consent_path)
    artifact = consent["artifacts"][0]
    write_json(
        path,
        {
            "schema": "video_knowledge_pipeline.trusted_model_connector.v1",
            "task": "cloud_asr",
            "ok": True,
            "status": "completed",
            "transport_ok": True,
            "contract_ok": True,
            "quality_gate_passed": True,
            "production_qualified": True,
            "consent_id": consent["consent_id"],
            "artifact_paths": [artifact["path"]],
            "upload_manifest": consent["upload_manifest"],
            "route": consent["route"],
            "model_result": {
                "model_type": "asr",
                "runtime_result": {"raw_output": {"segments": segments}},
            },
        },
    )
    return path


def _successful_reports(tmp_path: Path, consents: list[Path]) -> list[Path]:
    first = _report(
        tmp_path / "report-1.json",
        consents[0],
        [
            {
                "id": "a",
                "start": 0.0,
                "end": 8.0,
                "text": "这是第一段完整而清楚的测试文本",
                "words": [{"word": "这是", "start": 0.0, "end": 0.5}],
            },
            {
                "id": "boundary-a",
                "start": 8.0,
                "end": 11.5,
                "text": "边界处重复的完整句子",
            },
        ],
    )
    second = _report(
        tmp_path / "report-2.json",
        consents[1],
        [
            {
                "id": "boundary-b",
                "start": 0.0,
                "end": 3.6,
                "text": "边界处重复的完整句子",
            },
            {
                "id": "c",
                "start": 3.5,
                "end": 11.5,
                "text": "这是最后一段完整而清楚的测试文本",
            },
        ],
    )
    return [first, second]


def test_merge_offsets_words_and_exactly_deduplicates_overlap(tmp_path: Path) -> None:
    workflow, _, consents = _suite(tmp_path)
    reports = _successful_reports(tmp_path, consents)

    result = merge_asr_chunk_batch_reports(workflow, list(reversed(reports)))

    assert result["status"] == "completed"
    assert result["segment_count"] == 3
    assert result["exact_deduplication_count"] == 1
    assert result["boundary_conflict_count"] == 0
    assert result["source_media"]["path"].endswith("source.mp4")
    assert len(result["source_media"]["sha256"]) == 64
    transcript = read_json(Path(result["transcript_path"]))
    assert [row["start"] for row in transcript["segments"]] == [0.0, 8.5, 12.0]
    assert transcript["segments"][0]["metadata"]["words"][0]["start"] == 0.0
    assert transcript["segments"][1]["chunk_id"] == "asr-chunk-0002"
    assert transcript["segments"][1]["segment_id"].startswith("asr-chunk-0002:")
    assert Path(result["srt_path"]).is_file()
    assert result["operator_boundary"]["provider_call_performed"] is False
    assert result["alignment_advisory"]["status"] == "available"
    assert result["alignment_advisory"]["plan_prepared"] is False


def test_merge_excludes_padding_only_segments_and_flags_boundary_conflicts(
    tmp_path: Path,
) -> None:
    workflow, _, consents = _suite(tmp_path)
    first = _report(
        tmp_path / "report-1.json",
        consents[0],
        [
            {"id": "a", "start": 0, "end": 8, "text": "这是第一段完整而清楚的测试文本"},
            {
                "id": "left",
                "start": 8,
                "end": 11.5,
                "text": "左侧边界文本明显不同",
                "words": [
                    {"word": "保险", "start": 8.6, "end": 9.0},
                    {"word": "方案", "start": 9.0, "end": 9.4},
                    {"word": "需要", "start": 9.4, "end": 9.8},
                ],
            },
        ],
    )
    second = _report(
        tmp_path / "report-2.json",
        consents[1],
        [
            {"id": "padding", "start": 0, "end": 2, "text": "仅供上下文的垫片文本"},
            {
                "id": "right",
                "start": 0,
                "end": 3.6,
                "text": "右侧边界文本存在冲突",
                "words": [
                    {"word": "保险", "start": 0.1, "end": 0.5},
                    {"word": "方案", "start": 0.5, "end": 0.9},
                    {"word": "可以", "start": 0.9, "end": 1.3},
                ],
            },
            {"id": "c", "start": 3.5, "end": 11.5, "text": "这是最后一段完整而清楚的测试文本"},
        ],
    )

    result = merge_asr_chunk_batch_reports(workflow, [first, second], write=False)

    assert result["status"] == "review_required"
    assert result["exact_deduplication_count"] == 0
    assert result["boundary_conflict_count"] == 1
    assert result["excluded_padding_segment_count"] == 1
    assert result["boundary_conflicts"][0]["requires_human_review"] is True
    agreement = result["boundary_conflicts"][0]["local_agreement"]
    assert agreement["schema"] == "video_knowledge_pipeline.asr_local_agreement.v1"
    assert agreement["token_mode"] == "character"
    assert agreement["candidate_only"] is True
    assert agreement["automatic_merge_allowed"] is False
    assert agreement["upstream"]["project"] == "ufal/SimulStreaming"
    timestamped = result["boundary_conflicts"][0]["timestamped_local_agreement"]
    assert timestamped["status"] == "available"
    assert timestamped["common_prefix_words"] == ["保险", "方案"]
    assert timestamped["agreement_over_shorter"] == pytest.approx(2 / 3)
    assert timestamped["usable_for_review_ranking"] is True
    assert timestamped["automatic_merge_allowed"] is False
    assert result["operator_boundary"]["fuzzy_text_merge_performed"] is False


def test_merge_preserves_successful_chunks_when_one_report_is_missing(tmp_path: Path) -> None:
    workflow, _, consents = _suite(tmp_path)
    reports = _successful_reports(tmp_path, consents)

    result = merge_asr_chunk_batch_reports(workflow, reports[:1], write=False)

    assert result["status"] == "degraded"
    assert result["successful_chunk_count"] == 1
    assert result["failed_chunk_count"] == 1
    assert result["segment_count"] == 2
    assert result["asr_quality"]["coverage_gap_count"] == 1
    assert result["asr_quality"]["retry_plan"]["requires_new_exact_consent"] is True


def test_merge_rejects_changed_chunk_manifest(tmp_path: Path) -> None:
    workflow, _, consents = _suite(tmp_path)
    reports = _successful_reports(tmp_path, consents)
    workflow_payload = read_json(workflow)
    manifest = Path(workflow_payload["chunk_manifest"])
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="chunk manifest changed"):
        merge_asr_chunk_batch_reports(workflow, reports, write=False)


def test_merge_rejects_changed_chunk_artifact(tmp_path: Path) -> None:
    workflow, artifacts, consents = _suite(tmp_path)
    reports = _successful_reports(tmp_path, consents)
    artifacts[0].write_bytes(b"tampered chunk")

    with pytest.raises(
        ValueError,
        match="artifact byte count changed after workflow compilation",
    ):
        merge_asr_chunk_batch_reports(workflow, reports, write=False)


def test_merge_rejects_changed_activity_audit(tmp_path: Path) -> None:
    workflow, _, consents = _suite(tmp_path, with_activity_audit=True)
    reports = _successful_reports(tmp_path, consents)
    workflow_payload = read_json(workflow)
    audit = Path(workflow_payload["activity_audit"]["path"])
    audit.write_text(audit.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="activity audit changed"):
        merge_asr_chunk_batch_reports(workflow, reports, write=False)

def test_merge_reads_terminal_batch_status(tmp_path: Path) -> None:
    workflow, _, consents = _suite(tmp_path)
    reports = _successful_reports(tmp_path, consents)
    workflow_payload = read_json(workflow)
    status_path = tmp_path / "batch-status.json"
    write_json(
        status_path,
        {
            "schema": BATCH_SCHEMA,
            "terminal": True,
            "status": "completed",
            "job_id": "job-1",
            "items": [
                {
                    "node_id": node["node_id"],
                    "execution_report": str(report),
                }
                for node, report in zip(workflow_payload["nodes"], reports, strict=True)
            ],
        },
    )

    result = merge_asr_chunk_batch_reports(
        workflow, batch_status_path=status_path, write=False
    )

    assert result["status"] == "completed"
    assert result["batch_job_id"] == "job-1"
    assert result["batch_status_path"] == str(status_path.resolve())


def test_merge_can_plan_existing_qwen3_forced_aligner_without_execution(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    write_json(bundle / "manifest.json", {"schema": "fixture"})
    write_json(bundle / "timeline.json", {"items": []})
    workflow, _, consents = _suite(tmp_path, bundle_dir=bundle)
    reports = _successful_reports(tmp_path, consents)
    captured: dict[str, object] = {}

    def planner(
        root: Path,
        media: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        transcript_path = Path(str(kwargs["transcript_path"]))
        assert transcript_path.is_file()
        captured.update(root=root, media=media, **kwargs)
        return {
            "plan_path": str(tmp_path / "qwen3-forced-aligner-plan.json"),
            "model": "Qwen/Qwen3-ForcedAligner-0.6B",
        }

    result = merge_asr_chunk_batch_reports(
        workflow,
        reports,
        prepare_alignment_plan=True,
        alignment_planner=planner,
    )

    assert result["status"] == "completed"
    assert result["alignment_advisory"]["status"] == "planned"
    assert result["alignment_advisory"]["preset"] == "qwen3-forced-aligner"
    assert result["alignment_advisory"]["execution_performed"] is False
    assert result["operator_boundary"]["alignment_plan_prepared"] is True
    assert result["operator_boundary"]["canonical_text_replaced_by_alignment"] is False
    assert result["run_registry"]["status"] == "completed"
    assert Path(result["run_registry"]["run_json"]).is_file()
    assert Path(result["run_registry"]["run_markdown"]).is_file()
    registry = read_json(bundle / "run-artifact-registry.json")
    merge_run = next(
        row for row in registry["runs"] if row["run_type"] == "asr_chunk_merge"
    )
    assert merge_run["artifact_count"] == 2
    assert merge_run["operator_boundary"]["provider_call_performed"] is False
    assert captured["root"] == bundle.resolve()
    assert captured["preset"] == "qwen3-forced-aligner"
    assert captured["language"] == "zh"
    transcript = read_json(Path(result["transcript_path"]))
    assert [row["text"] for row in transcript["segments"]] == [
        "这是第一段完整而清楚的测试文本",
        "边界处重复的完整句子",
        "这是最后一段完整而清楚的测试文本",
    ]


def test_optional_alignment_plan_failure_preserves_completed_merge(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    write_json(bundle / "manifest.json", {"schema": "fixture"})
    write_json(bundle / "timeline.json", {"items": []})
    workflow, _, consents = _suite(tmp_path, bundle_dir=bundle)
    reports = _successful_reports(tmp_path, consents)

    def failed_planner(*_: object, **__: object) -> dict[str, object]:
        raise RuntimeError("fixture alignment runtime unavailable")

    result = merge_asr_chunk_batch_reports(
        workflow,
        reports,
        prepare_alignment_plan=True,
        alignment_planner=failed_planner,
    )

    assert result["status"] == "completed"
    assert result["alignment_advisory"]["status"] == "plan_failed"
    assert result["alignment_advisory"]["error_type"] == "RuntimeError"
    assert result["operator_boundary"]["alignment_plan_prepared"] is False
    assert Path(result["transcript_path"]).is_file()
    assert Path(result["report_path"]).is_file()


def test_alignment_plan_requires_written_merge(tmp_path: Path) -> None:
    workflow, _, consents = _suite(tmp_path)
    reports = _successful_reports(tmp_path, consents)

    with pytest.raises(ValueError, match="requires write=True"):
        merge_asr_chunk_batch_reports(
            workflow,
            reports,
            prepare_alignment_plan=True,
            write=False,
        )


def test_merge_cli_is_local_compile_only(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    workflow, _, consents = _suite(tmp_path)
    reports = _successful_reports(tmp_path, consents)

    assert (
        cli_main(
            [
                "asr-chunk-batch-merge",
                str(workflow),
                "--execution-report",
                str(reports[0]),
                "--execution-report",
                str(reports[1]),
                "--no-write",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert payload["write"] is False
    assert not Path(payload["report_path"]).exists()
