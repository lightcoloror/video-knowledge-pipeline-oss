from __future__ import annotations

from pathlib import Path

from video_knowledge_pipeline import local_targeted_asr_execution as targeted_execution
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.storage import read_json, write_json


def _plan(bundle: Path) -> Path:
    path = bundle / "local-targeted-asr-plan.json"
    write_json(
        path,
        {
            "schema": "video_knowledge_pipeline.local_targeted_asr_plan.v1",
            "retry_plan": {
                "window_count": 1,
                "windows": [
                    {
                        "retry_id": "semantic-evidence-0001",
                        "source_segment_ids": ["risk-person"],
                        "start": 7.0,
                        "end": 16.0,
                        "duration_seconds": 9.0,
                    }
                ],
            },
        },
    )
    return path


def test_local_targeted_asr_execution_registers_verified_candidate_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    media = bundle / "lesson.mp4"
    media.write_bytes(b"media fixture")
    write_json(
        bundle / "manifest.json",
        {
            "media_path": str(media),
            "corrected_transcript_json": "corrected-transcript.json",
        },
    )
    write_json(
        bundle / "corrected-transcript.json",
        {"segments": [{"start": 0.0, "end": 20.0, "text": "中威讲短视频营销。"}]},
    )
    plan = _plan(bundle)
    clip = bundle / "local-targeted-asr-snippets" / "semantic-evidence-0001.wav"
    raw = bundle / "local-targeted-asr-runs" / "semantic-evidence-0001" / "raw-asr-output.json"

    def fake_prepare(media_path, quality_report, output_dir, *, execute):
        assert Path(media_path) == media.resolve()
        assert execute is True
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(b"clip fixture")
        artifact = {
            "retry_id": "semantic-evidence-0001",
            "source_segment_ids": ["risk-person"],
            "start": 7.0,
            "end": 16.0,
            "duration_seconds": 9.0,
            "path": str(clip),
            "status": "completed",
            "sha256": sha256_file(clip),
            "bytes": clip.stat().st_size,
        }
        result = {
            "schema": "video_knowledge_pipeline.asr_retry_snippets.v1",
            "status": "completed",
            "artifacts": [artifact],
            "failed_chunks": [],
        }
        write_json(Path(output_dir) / "asr-retry-snippets.json", result)
        return result

    def fake_plan(workspace, clip_path, **kwargs):
        assert Path(clip_path) == clip.resolve()
        plan_path = Path(workspace) / "asr-run-plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(plan_path, {"preset": kwargs["preset"]})
        return {"plan_path": str(plan_path), "preset": kwargs["preset"], "provider": "qwen3-asr"}

    def fake_run(plan_path, **kwargs):
        assert kwargs["execute"] is True
        assert kwargs["normalize"] is False
        raw.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            raw,
            {
                "ok": True,
                "status": "completed",
                "failed_chunk_count": 0,
                "input_path": str(clip),
                "device": "cuda:0",
                "provider": "qwen3-asr",
                "model": "fixture-qwen3",
                "segments": [{"segment_id": "1", "start": 0.0, "end": 4.0, "text": "钟巍讲短视频营销。"}],
            },
        )
        return {"status": "ok", "raw_output_json": str(raw)}

    monkeypatch.setattr(targeted_execution, "prepare_asr_retry_snippets", fake_prepare)
    monkeypatch.setattr(targeted_execution, "plan_asr_run", fake_plan)
    monkeypatch.setattr(targeted_execution, "run_asr_plan", fake_run)

    result = targeted_execution.run_local_targeted_asr_evidence(
        bundle,
        input_plan=plan,
        execute=True,
    )

    assert result["status"] == "completed"
    assert result["ok"] is True
    evidence = read_json(bundle / "local-targeted-asr-evidence.json")
    assert evidence["candidate_only"] is True
    assert evidence["segments"][0]["start"] == 7.0
    manifest = read_json(bundle / "manifest.json")
    assert str((bundle / "local-targeted-asr-evidence.json").resolve()) in manifest["asr_secondary_transcripts"]
    assert manifest["corrected_transcript_json"] == "corrected-transcript.json"
    assert result["operator_boundary"]["independent_second_asr"] is False
    assert result["operator_boundary"]["model_independence"] == "requires_model_lineage_check"


def test_local_targeted_asr_execution_does_not_accept_partial_clip_extraction(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    media = bundle / "lesson.mp4"
    media.write_bytes(b"media fixture")
    write_json(bundle / "manifest.json", {"media_path": str(media)})
    plan = _plan(bundle)

    def fake_prepare(*args, **kwargs):
        return {
            "schema": "video_knowledge_pipeline.asr_retry_snippets.v1",
            "status": "degraded",
            "artifacts": [],
            "failed_chunks": [{"retry_id": "semantic-evidence-0001"}],
        }

    monkeypatch.setattr(targeted_execution, "prepare_asr_retry_snippets", fake_prepare)
    result = targeted_execution.run_local_targeted_asr_evidence(bundle, input_plan=plan, execute=True)

    assert result["status"] == "snippet_extraction_failed"
    assert not (bundle / "local-targeted-asr-evidence.json").exists()


def test_shared_preflight_failure_preserves_root_cause_and_stops_later_windows(tmp_path, monkeypatch):
    from datetime import datetime, timedelta
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    media = bundle / 'lesson.mp4'
    media.write_bytes(b'fixture')
    write_json(bundle / 'manifest.json', {'media_path': str(media)})
    plan = _plan(bundle)
    payload = read_json(plan)
    payload['retry_plan']['windows'].append({'retry_id': 'second', 'start': 20, 'end': 25})
    write_json(plan, payload)
    monkeypatch.setattr(targeted_execution, 'prepare_asr_retry_snippets', lambda *a, **kw: {
        'artifacts': [{'status': 'completed', 'retry_id': name, 'path': str(media)}
                      for name in ['first', 'second']]})
    monkeypatch.setattr(targeted_execution, 'plan_asr_run', lambda *a, **kw: {
        'plan_path': str(plan), 'provider': 'qwen3_asr'})
    calls = []
    def fail(*a, **kw):
        calls.append(a)
        raw = bundle / 'failed.json'
        started = (datetime.now() - timedelta(seconds=1)).isoformat()
        write_json(raw, {'status': 'failed', 'error_code': 'qwen3_asr_model_not_ready',
                         'error': 'missing weight shards', 'model_readiness': {'ready': False}})
        return {'status': 'failed', 'stderr': 'Local ASR failed', 'started_at': started,
                'expected_output_json': str(raw), 'raw_output_json': ''}
    monkeypatch.setattr(targeted_execution, 'run_asr_plan', fail)
    result = targeted_execution.run_local_targeted_asr_evidence(bundle, input_plan=plan, execute=True)
    assert len(calls) == 1
    assert result['failed_runs'][0]['error'] == 'missing weight shards'
    assert result['failed_runs'][0]['model_readiness'] == {'ready': False}
    assert result['not_run'] == [{'retry_id': 'second', 'reason': 'shared_runtime_preflight_failed'}]
    assert result['status'] == 'local_asr_failed'


def test_stale_runner_failure_does_not_hide_new_failure(tmp_path):
    from datetime import datetime, timedelta
    raw = tmp_path / 'old.json'
    write_json(raw, {'status': 'failed', 'error_code': 'qwen3_asr_model_not_ready'})
    result = targeted_execution._runner_failure({
        'status': 'failed', 'expected_output_json': str(raw),
        'started_at': (datetime.now() + timedelta(seconds=2)).isoformat()})
    assert result == {}
