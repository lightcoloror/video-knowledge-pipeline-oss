from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_knowledge_pipeline.run_artifact_recovery import (
    capture_result_run, inspect_result_run, resume_result_run, main,
)
from video_knowledge_pipeline.run_artifact_registry import build_run_artifact_registry
from video_knowledge_pipeline.storage import write_json


def fixture(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'manifest.json', {'schema': 'synthetic', 'video_id': 'invented'})
    write_json(root / 'dependency-versions.json', {'rules': 'synthetic-v1'})
    write_json(root / 'current.json', {'candidates': [], 'candidate_count': 0})
    write_json(root / 'history.json', {'candidates': [
        {'candidate_id': 'h1', 'original_text': '虚构甲'},
        {'candidate_id': 'h2', 'original_text': '虚构乙'}], 'candidate_count': 263})
    write_json(root / 'ocr.json', {'items': [
        {'index': n, 'execution_status': 'completed' if n < 9 else 'not_run',
         'recognition_status': 'empty' if n == 0 else 'unknown'} for n in range(80)]})
    write_json(root / 'human.json', {'items': [{'index': 79, 'status': 'corrected_visual_text'}]})
    write_json(root / 'prompts.json', {'coverage': {'covered_cue_count': 80}, 'complete': True})
    kinds = [('current', 'current_candidates'), ('history', 'historical_candidates'),
             ('ocr', 'ocr_inventory'), ('human', 'human_evidence'), ('prompts', 'discovery_prompts')]
    request = {'run_id': 'synthetic-v1', 'inputs': ['manifest.json', 'dependency-versions.json'],
               'dependency_versions': {'rules': 'synthetic-v1'},
               'stages': [{'stage_id': n, 'source': n + '.json', 'kind': k} for n, k in kinds]}
    return request


def test_real_cli_incremental_resume_is_idempotent(tmp_path, capsys):
    request = fixture(tmp_path)
    request_path = tmp_path / 'request.json'
    write_json(request_path, request)
    source_before = (tmp_path / 'manifest.json').read_bytes()
    assert main(['capture', str(tmp_path), '--request', str(request_path)]) == 0
    capture = json.loads(capsys.readouterr().out)
    assert capture['pending_stages'] == ['current', 'history', 'ocr', 'human', 'prompts']
    args = ['resume', str(tmp_path), '--run-id', 'synthetic-v1', '--versions', str(tmp_path / 'dependency-versions.json')]
    assert main([*args, '--max-stages', '1']) == 0
    first = json.loads(capsys.readouterr().out)
    assert first['restored_stages'] == ['current']
    current = tmp_path / 'exports/versioned-results/synthetic-v1/current.json'
    mtime = current.stat().st_mtime_ns
    assert main(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert second['restored_stages'] == ['history', 'ocr', 'human', 'prompts']
    assert second['skipped_completed'] == ['current']
    assert current.stat().st_mtime_ns == mtime
    assert main(args) == 0
    third = json.loads(capsys.readouterr().out)
    assert third['restored_count'] == 0 and third['producer_calls'] == 0
    assert third['status'] == 'projection_complete'
    assert third['semantic_acceptance'] is False
    assert (tmp_path / 'manifest.json').read_bytes() == source_before


def test_evidence_dimensions_remain_separate(tmp_path):
    result = capture_result_run(tmp_path, fixture(tmp_path))
    rows = {r['stage_id']: r['evidence'] for r in result['stages']}
    assert rows['history']['declared_count'] == 263
    assert rows['history']['concrete_candidate_count'] == 2
    assert rows['history']['missing_candidate_bodies'] == 261
    assert rows['history']['historical_rebuild_status'] == 'unreconstructable_missing_bodies'
    assert rows['current']['current_zero_candidates'] is True
    assert rows['current']['zero_candidates_means_zero_errors'] is False
    assert rows['ocr']['machine_executed'] == 9
    assert rows['ocr']['machine_not_run'] == 71
    assert rows['ocr']['empty_results'] == 1
    assert rows['human']['record_count'] == 1
    assert rows['human']['included_in_machine_ocr_count'] is False
    assert rows['prompts']['semantic_execution_complete'] is False
    assert result['content_gaps'] == [{'stage_id': 'history', 'missing_candidate_bodies': 261}]
    assert result['content_recovery_complete'] is False


def test_ocr_unknown_state_and_unchecked_versions_stay_unverified(tmp_path):
    req = fixture(tmp_path)
    write_json(tmp_path / 'ocr.json', {'items': [{'execution_status': 'future-state'}, {}]})
    capture_result_run(tmp_path, req)
    result = inspect_result_run(tmp_path, req['run_id'])
    rows = {r['stage_id']: r for r in result['stages']}
    assert rows['ocr']['evidence']['unknown_execution'] == 2
    assert rows['current']['evidence']['captured_zero_candidates'] is True
    assert rows['current']['evidence']['current_zero_candidates'] is False
    assert result['dependency_versions'] == 'not_checked' and result['can_resume'] is False


@pytest.mark.parametrize('change', ['input', 'versions', 'snapshot', 'target', 'binding', 'code'])
def test_drift_blocks_without_repairing_or_overwriting(tmp_path, monkeypatch, change):
    capture_result_run(tmp_path, fixture(tmp_path))
    versions = {'rules': 'synthetic-v1'}
    if change == 'input':
        write_json(tmp_path / 'manifest.json', {'changed': True})
    elif change == 'versions':
        versions = {'rules': 'synthetic-v2'}
    elif change == 'snapshot':
        write_json(tmp_path / 'runs/synthetic-v1/snapshots/current.json', {'candidates': ['corrupt']})
    elif change == 'target':
        write_json(tmp_path / 'exports/versioned-results/synthetic-v1/current.json', {'human_edit': True})
    elif change == 'binding':
        path = tmp_path / 'runs/synthetic-v1/run.json'
        run = json.loads(path.read_text(encoding='utf-8'))
        run['run_id'] = 'foreign-run'
        write_json(path, run)
        with pytest.raises(ValueError, match='identity'):
            resume_result_run(tmp_path, 'synthetic-v1', dependency_versions=versions)
        return
    else:
        monkeypatch.setattr('video_knowledge_pipeline.run_artifact_recovery._producer_versions', lambda: {'changed': True})
    result = resume_result_run(tmp_path, 'synthetic-v1', dependency_versions=versions)
    assert result['status'] == 'blocked'
    assert result['restored_count'] == 0


def test_missing_snapshot_does_not_infer_or_execute_producer(tmp_path):
    req = fixture(tmp_path)
    req['stages'].append({'stage_id': 'lost', 'source': 'never-saved.json', 'kind': 'historical_candidates', 'declared_count': 263})
    capture_result_run(tmp_path, req)
    result = resume_result_run(tmp_path, 'synthetic-v1', dependency_versions=req['dependency_versions'])
    assert result['missing_snapshots'] == ['never-saved.json']
    assert result['status'] == 'blocked' and result['restored_count'] == 5
    assert not (tmp_path / 'exports/versioned-results/synthetic-v1/lost.json').exists()
    assert resume_result_run(tmp_path, 'synthetic-v1', dependency_versions=req['dependency_versions'])['restored_count'] == 0


def test_new_version_does_not_reuse_old_run_by_mtime(tmp_path):
    req = fixture(tmp_path)
    capture_result_run(tmp_path, req)
    resume_result_run(tmp_path, req['run_id'], dependency_versions=req['dependency_versions'])
    req['run_id'] = 'synthetic-v2'
    write_json(tmp_path / 'current.json', {'candidates': [{'candidate_id': 'new', 'text': '新版本虚构'}]})
    result = capture_result_run(tmp_path, req)
    assert result['run_id'] == 'synthetic-v2'
    assert result['stages'][0]['evidence']['concrete_candidate_count'] == 1
    previous = inspect_result_run(tmp_path, 'synthetic-v1', dependency_versions=req['dependency_versions'])['stages'][0]
    assert previous['evidence']['captured_zero_candidates'] is True
    assert previous['evidence']['current_zero_candidates'] is False
    assert previous['source_state'] == 'changed'
    assert previous['evidence_freshness'] == 'stale_or_unverified'
    assert build_run_artifact_registry(tmp_path, write=False)['run_count'] == 2


def test_capture_rejects_reuse_traversal_and_foreign_producer(tmp_path):
    req = fixture(tmp_path)
    req['stages'][0]['source'] = '../outside.json'
    with pytest.raises(ValueError, match='outside'):
        capture_result_run(tmp_path, req)
    req['stages'][0]['source'] = 'current.json'
    write_json(tmp_path / 'current.json', {'run_id': 'original-run', 'candidates': []})
    with pytest.raises(ValueError, match='source_run'):
        capture_result_run(tmp_path, req)
    req['stages'][0]['source_run_id'] = 'original-run'
    capture_result_run(tmp_path, req)
    with pytest.raises(ValueError, match='already_exists'):
        capture_result_run(tmp_path, req)


def test_interruption_recovers_only_pending_stage(tmp_path, monkeypatch):
    req = fixture(tmp_path)
    capture_result_run(tmp_path, req)
    from video_knowledge_pipeline import run_artifact_recovery as mod
    copy = mod.atomic_copy_file
    def failing_copy(source, target):
        if target.name == 'history.json':
            raise OSError('synthetic interruption')
        return copy(source, target)
    monkeypatch.setattr(mod, 'atomic_copy_file', failing_copy)
    with pytest.raises(OSError, match='interruption'):
        resume_result_run(tmp_path, req['run_id'], dependency_versions=req['dependency_versions'])
    monkeypatch.setattr(mod, 'atomic_copy_file', copy)
    result = resume_result_run(tmp_path, req['run_id'], dependency_versions=req['dependency_versions'])
    assert result['skipped_completed'] == ['current']
    assert result['restored_stages'] == ['history', 'ocr', 'human', 'prompts']
