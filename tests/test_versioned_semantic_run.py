from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.versioned_semantic_run import create_run, resume, consume, status, main
from video_knowledge_pipeline.transcript_semantic_correction import (
    build_transcript_semantic_correction_pack, build_transcript_semantic_correction_codex_draft)
from video_knowledge_pipeline.review_session import prepare_review_session


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('network prohibited in offline test')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def fixture(root, rid='semantic-v1', zero=False):
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'manifest.json', {'title': '虚构工具演示', 'normalized_transcript_json': 'normalized-transcript.json'})
    write_json(root / 'normalized-transcript.json', {'segments': [] if zero else [
        {'start': 0, 'end': 4, 'text': '今天讲 play right m c p 工具'}]})
    write_json(root / 'timeline.json', [] if zero else [
        {'index': 1, 'start': 0, 'end': 4, 'visual_text': 'Playwright MCP 工具'}])
    write_json(root / 'history.json', {'candidate_count': 263, 'candidates': [
        {'candidate_id': 'old1', 'text': '虚构历史甲'}, {'candidate_id': 'old2', 'text': '虚构历史乙'}]})
    write_json(root / 'ocr.json', {'items': [{'index': i, 'execution_status': 'completed' if i < 9 else 'not_run'} for i in range(80)]})
    write_json(root / 'human.json', {'items': [{'source': 'synthetic human evidence'}]})
    request = {'run_id': rid, 'dependency_versions': {'rules': 'synthetic-v1'},
        'inputs': ['manifest.json', 'normalized-transcript.json', 'timeline.json', 'history.json', 'ocr.json', 'human.json'],
        'context': [{'path': 'history.json', 'kind': 'historical_candidates'},
                    {'path': 'ocr.json', 'kind': 'ocr_inventory'}, {'path': 'human.json', 'kind': 'human_evidence'}]}
    write_json(root / 'request.json', request)
    write_json(root / 'versions.json', request['dependency_versions'])
    return request


def test_real_producers_resume_only_interrupted_b_and_existing_consumers(tmp_path):
    req = fixture(tmp_path)
    original = {p: (tmp_path / p).read_bytes() for p in req['inputs']}
    create_run(tmp_path, req)
    # An actual filesystem interruption at B's commit; no producer is mocked.
    obstacle = tmp_path / 'runs/semantic-v1/stage-cache/local_draft/result.json'
    obstacle.mkdir(parents=True)
    first = resume(tmp_path, req['run_id'], req['dependency_versions'])
    assert first['executed_stages'] == ['candidates']
    assert first['status'] == 'blocked' and first['failed_stage'] == 'local_draft'
    with pytest.raises(ValueError, match='not_complete'):
        prepare_review_session(tmp_path, run_id=req['run_id'], dependency_versions=req['dependency_versions'])
    a = tmp_path / 'runs/semantic-v1/stage-cache/candidates/result.json'
    saved = a.read_bytes(), a.stat().st_mtime_ns
    obstacle.rmdir()
    second = resume(tmp_path, req['run_id'], req['dependency_versions'])
    assert second['executed_stages'] == ['local_draft'] and second['skipped_stages'] == ['candidates']
    third = resume(tmp_path, req['run_id'], req['dependency_versions'])
    assert third['executed_stages'] == [] and third['skipped_stages'] == ['candidates', 'local_draft']
    assert (a.read_bytes(), a.stat().st_mtime_ns) == saved
    pack = build_transcript_semantic_correction_pack(tmp_path, run_id=req['run_id'], dependency_versions=req['dependency_versions'])
    draft = build_transcript_semantic_correction_codex_draft(tmp_path, run_id=req['run_id'], dependency_versions=req['dependency_versions'])
    assert pack['candidate_count'] > 0 and draft['decision_count'] > 0
    assert any(d['original_text'] == 'm c p' and d['corrected_text'] == 'MCP' for d in draft['decisions'])
    assert pack['version_binding']['binding_sha256'] == draft['version_binding']['binding_sha256']
    reviewed = prepare_review_session(tmp_path, run_id=req['run_id'], dependency_versions=req['dependency_versions'])
    assert reviewed['version_binding'] == pack['version_binding']
    assert reviewed['context'][0]['missing_candidate_bodies'] == 261
    assert reviewed['context'][1]['machine_executed'] == 9
    assert reviewed['context'][1]['machine_not_run'] == 71
    assert reviewed['context'][2]['included_in_machine_ocr_count'] is False
    assert reviewed['model_adjudication'] == 'not_executed' and reviewed['semantic_acceptance'] is False
    assert Path(reviewed['review_html_path']).exists()
    assert all((tmp_path / p).read_bytes() == value for p, value in original.items())
    assert not (tmp_path / 'transcript-semantic-correction-pack.json').exists()


@pytest.mark.parametrize('drift', ['source', 'snapshot', 'version', 'output', 'code', 'extra_input'])
def test_consumer_fails_closed_without_fallback(tmp_path, drift):
    req = fixture(tmp_path)
    create_run(tmp_path, req)
    resume(tmp_path, req['run_id'], req['dependency_versions'])
    versions = req['dependency_versions']
    if drift == 'source':
        write_json(tmp_path / 'normalized-transcript.json', {'segments': []})
    elif drift == 'snapshot':
        write_json(tmp_path / 'runs/semantic-v1/inputs/timeline.json', [])
    elif drift == 'version':
        versions = {'rules': 'synthetic-v2'}
    elif drift == 'output':
        write_json(tmp_path / 'runs/semantic-v1/stage-cache/candidates/result.json', {'candidate_count': 0})
    elif drift == 'extra_input':
        write_json(tmp_path / 'runs/semantic-v1/inputs/secondary-asr.json', {})
    else:
        from video_knowledge_pipeline.versioned_semantic_run import _binding
        path = tmp_path / 'runs/semantic-v1/run.json'
        run = json.loads(path.read_text(encoding='utf-8'))
        run['parameters']['code_versions']['transcript_semantic_correction.py'] = 'old-code'
        run['result_binding_sha256'] = _binding(run)
        write_json(path, run)
    with pytest.raises(ValueError):
        prepare_review_session(tmp_path, run_id=req['run_id'], dependency_versions=versions)


def test_existing_cli_reads_exact_zero_version_and_never_legacy_candidate(tmp_path, capsys):
    from video_knowledge_pipeline.cli import main as cli
    req = fixture(tmp_path, zero=True)
    assert main(['create', str(tmp_path), '--request', str(tmp_path / 'request.json')]) == 0
    capsys.readouterr()
    assert main(['resume', str(tmp_path), '--run-id', req['run_id'], '--versions', str(tmp_path / 'versions.json')]) == 0
    capsys.readouterr()
    write_json(tmp_path / 'transcript-semantic-correction-pack.json', {'candidate_count': 99, 'candidates': [{'text': 'stale'}]})
    assert cli(['prepare-review-session', str(tmp_path), '--run-id', req['run_id'], '--dependency-versions', str(tmp_path / 'versions.json')]) == 0
    reviewed = json.loads(capsys.readouterr().out)
    assert reviewed['candidate_count'] == 0 and reviewed['current_zero_candidates'] is True
    assert reviewed['context'][0]['missing_candidate_bodies'] == 261
    with pytest.raises((ValueError, FileNotFoundError)):
        consume(tmp_path, 'unrecorded-run', req['dependency_versions'], 'candidates')


def test_create_rejects_external_reference_and_preserves_original_run(tmp_path):
    req = fixture(tmp_path)
    write_json(tmp_path / 'manifest.json', {'normalized_transcript_json': '../external.json'})
    with pytest.raises(ValueError, match='unsnapshotted'):
        create_run(tmp_path, req)
    req = fixture(tmp_path)
    create_run(tmp_path, req)
    with pytest.raises(ValueError, match='already_exists'):
        create_run(tmp_path, req)


def test_registry_and_cache_integrity_are_shared_with_legacy_consumers(tmp_path):
    from video_knowledge_pipeline.run_artifact_registry import build_run_artifact_registry
    req = fixture(tmp_path)
    create_run(tmp_path, req)
    resume(tmp_path, req['run_id'], req['dependency_versions'])
    registry = build_run_artifact_registry(tmp_path, write=False)
    assert registry['runs'][0]['run_id'] == req['run_id']
    assert registry['runs'][0]['status'] == 'offline_stages_complete'
    assert registry['runs'][0]['artifact_count'] == 2
    with pytest.raises(ValueError, match='versions'):
        consume(tmp_path, req['run_id'], None, 'candidates')


def test_concise_consumer_keeps_binding_and_runtime_drift_blocks(tmp_path, monkeypatch):
    from video_knowledge_pipeline.cli_output import render_cli_result
    req = fixture(tmp_path)
    create_run(tmp_path, req)
    resume(tmp_path, req['run_id'], req['dependency_versions'])
    pack = consume(tmp_path, req['run_id'], req['dependency_versions'], 'candidates')
    rendered = json.loads(render_cli_result(pack, inline_limit=1))
    assert rendered['version_binding'] == pack['version_binding']
    assert rendered['semantic_acceptance'] is False
    monkeypatch.setattr('video_knowledge_pipeline.versioned_semantic_run._runtime_versions', lambda: {'python': 'changed'})
    with pytest.raises(ValueError, match='runtime_versions_changed'):
        consume(tmp_path, req['run_id'], req['dependency_versions'], 'candidates')
