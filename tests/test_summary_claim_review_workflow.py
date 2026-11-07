"""Synthetic front-door / next-action checks; no Provider or media calls."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from video_knowledge_pipeline import bundle_next, cli, mcp_server, summary_claim_review as review
from video_knowledge_pipeline.storage import write_json
from test_summary_claim_review import fixture, submit


def bundle(root):
    fixture(root)
    write_json(root / 'manifest.json', {})


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def isolate_upstream(monkeypatch):
    for name in ('_next_knowledge_coverage_action', '_next_repair_action', '_next_semantic_correction_action', '_next_readiness_action'):
        monkeypatch.setattr(bundle_next, name, lambda *a: None)
    monkeypatch.setattr(bundle_next, 'refresh_bundle_repair_status', lambda *a: None)
    monkeypatch.setattr(bundle_next, 'audit_knowledge_coverage', lambda *a, **kw: None)


def test_status_is_no_write_and_exposes_current_immutable_task(tmp_path):
    bundle(tmp_path)
    before = snapshot(tmp_path)
    first = review.summary_claim_review_status(tmp_path)
    assert first['status'] == 'waiting_annotation'
    assert first['required'] and first['passed'] is False
    assert first['next_task']['phase'] == 'annotate'
    assert first['next_task']['prepared'] is False
    assert 'sources' not in first['next_task']
    assert first == review.summary_claim_review_status(tmp_path)
    assert snapshot(tmp_path) == before
    task = review.prepare_summary_claim_review(tmp_path)
    current = review.summary_claim_review_status(tmp_path)
    assert current['next_task']['prepared']
    assert current['next_task']['task_id'] == task['task_id']
    assert current['counts'] == {'waiting_annotation': 1}


@pytest.mark.parametrize('mutation', ['source', 'missing_plan', 'bad_state', 'task_pack'])
def test_status_drift_or_missing_never_becomes_ready(tmp_path, mutation):
    bundle(tmp_path)
    task = review.prepare_summary_claim_review(tmp_path)
    if mutation == 'source':
        write_json(tmp_path / 'timeline.json', [{'text': 'changed'}])
    elif mutation == 'missing_plan':
        (tmp_path / review.PLAN).unlink()
    elif mutation == 'bad_state':
        write_json(tmp_path / review.STATE, {'state_sha256': 'bad'})
    else:
        write_json(tmp_path / f'runs/summary-claim-review/tasks/{task["task_id"]}.json', {'tampered': True})
    before = snapshot(tmp_path)
    status = review.summary_claim_review_status(tmp_path)
    assert status['status'] == 'blocked' and status['passed'] is False
    assert status['required'] and status['errors']
    assert snapshot(tmp_path) == before


def test_legacy_is_not_assessed_and_no_outputs_created(tmp_path):
    write_json(tmp_path / 'manifest.json', {})
    before = snapshot(tmp_path)
    result = review.summary_claim_review_status(tmp_path)
    assert result['status'] == 'legacy_not_assessed'
    assert result['passed'] is None and not result['required']
    assert snapshot(tmp_path) == before


def test_cli_mcp_and_powershell_use_existing_importer(tmp_path, capsys):
    root = tmp_path / "中文 bundle's"
    bundle(root)
    assert cli.main(['prepare-summary-claim-review', str(root), '--no-write']) == 0
    assert 'task_id' in capsys.readouterr().out
    assert not (root / 'runs').exists()
    funcs = cli._mcp_callables()
    for name in ('prepare_summary_claim_review', 'import_summary_claim_review', 'summary_claim_review_status'):
        assert funcs[name] is getattr(review, name)
        assert f'def {name}(' in inspect.getsource(mcp_server.main)
    task = funcs['prepare_summary_claim_review'](str(root))
    from test_summary_claim_review import answer
    write_json(root / 'answer.json', answer(task))
    assert cli.main(['import-summary-claim-review', str(root), '--input-json', str(root / 'answer.json')]) == 0
    capsys.readouterr()
    assert cli.main(['summary-claim-review-status', str(root)]) == 0
    assert 'waiting_independent_review' in capsys.readouterr().out
    wrapper = (Path(__file__).parents[1] / 'scripts/video-knowledge.ps1').read_text(encoding='utf-8-sig')
    offline = wrapper.split('$offlineProjectCommands = @(', 1)[1].split(')', 1)[0]
    assert all(f'"{x}"' in offline for x in ('prepare-summary-claim-review', 'import-summary-claim-review', 'summary-claim-review-status'))


def test_next_and_advance_prepare_but_never_invent_review(tmp_path, monkeypatch):
    bundle(tmp_path)
    isolate_upstream(monkeypatch)
    nxt = bundle_next.bundle_next_action(tmp_path, refresh=False)['next_action']
    assert nxt['status'] == 'summary_claim_review_pending'
    assert nxt['mcp_tool'] == 'prepare_summary_claim_review'
    assert not nxt['unattended_execution']
    dry = bundle_next.bundle_advance(tmp_path, execute=False)
    assert not dry['advanced'] and not (tmp_path / 'runs').exists()
    actual = bundle_next.bundle_advance(tmp_path, execute=True)
    assert actual['advanced'] and actual['action_result']['task_id']
    assert not (tmp_path / review.STATE).exists()
    repeated = bundle_next.bundle_advance(tmp_path, execute=True)
    assert not repeated['advanced']  # prepared task is waiting, not a new step
    assert repeated['after']['next_action']['next_task']['prepared']
    task = review.prepare_summary_claim_review(tmp_path)
    assert submit(tmp_path, task)['ok']
    assert bundle_next.bundle_next_action(tmp_path, refresh=False)['next_action']['next_task']['phase'] == 'review'
    assert submit(tmp_path, review.prepare_summary_claim_review(tmp_path), run='reviewer')['ok']
    assert bundle_next.bundle_next_action(tmp_path, refresh=False)['status'] == 'ready'


def test_existing_upstream_blocker_keeps_priority(tmp_path, monkeypatch):
    bundle(tmp_path)
    isolate_upstream(monkeypatch)
    blocker = {'status': 'coverage_blocked', 'key': 'source_artifacts', 'mcp_tool': 'audit_knowledge_coverage'}
    monkeypatch.setattr(bundle_next, '_next_knowledge_coverage_action', lambda *a: blocker)
    assert bundle_next.bundle_next_action(tmp_path, refresh=False)['next_action'] == blocker


def test_next_never_exports_bad_review_state(tmp_path, monkeypatch):
    bundle(tmp_path)
    isolate_upstream(monkeypatch)
    write_json(tmp_path / review.STATE, {})
    action = bundle_next.bundle_next_action(tmp_path, refresh=False)['next_action']
    assert action['status'] == 'summary_claim_review_blocked'
    assert action['human_required']
    assert action['mcp_tool'] == 'summary_claim_review_status'


def test_workbench_and_console_share_fresh_review_projection(tmp_path):
    from video_knowledge_pipeline.task_console import export_task_console, _render_task_console_html
    from video_knowledge_pipeline.video_workbench import export_video_workbench, _render_workbench_html
    _render_html = lambda view: _render_workbench_html(view, [])
    bundle(tmp_path)
    task = review.prepare_summary_claim_review(tmp_path)
    for export, render in ((export_task_console, _render_task_console_html), (export_video_workbench, _render_html)):
        view = export(tmp_path, write=False)
        state = view['summary_claim_review']
        assert state['next_task']['task_id'] == task['task_id']
        assert state['counts']['waiting_annotation'] == 1
        page = render(view)
        assert '摘要逐条证据审核' in page
        assert 'prepare-summary-claim-review' in page
        assert 'import-summary-claim-review' in page
        assert state['next_task']['path'] in page
        assert '不代表现实事实已证实' in page
    write_json(tmp_path / 'timeline.json', [{'text': '<script>tampered</script>'}])
    for export in (export_task_console, export_video_workbench):
        state = export(tmp_path, write=False)['summary_claim_review']
        assert state['status'] == 'blocked'


def test_review_panel_escapes_source_controlled_strings():
    from video_knowledge_pipeline.task_console import _summary_claim_review_panel_html
    page = _summary_claim_review_panel_html({'required': True, 'status': '<script>x</script>',
        'errors': ['<img src=x onerror=x>'], 'counts': {}, 'commands': {}, 'artifacts': []})
    assert '<script>' not in page and '<img src=x' not in page
    assert '&lt;script&gt;' in page


def test_completed_review_routes_existing_reduce_resume_without_model(tmp_path, monkeypatch):
    bundle(tmp_path)
    isolate_upstream(monkeypatch)
    for run in ('annotator', 'reviewer'):
        assert submit(tmp_path, review.prepare_summary_claim_review(tmp_path), run=run)['ok']
    write_json(tmp_path / 'exports/smart-summary-global-reduce.json', {'status': 'waiting_claim_review'})
    action = bundle_next.bundle_next_action(tmp_path, refresh=False)['next_action']
    assert action['status'] == 'summary_claim_review_ready'
    assert action['mcp_args']['reuse_candidate'] is True
    assert action['mcp_args']['execute'] is False
    import video_knowledge_pipeline.smart_summary_global_reduce as reducer
    calls = []
    def resume(root, **kwargs):
        calls.append(kwargs)
        write_json(root / 'exports/smart-summary-global-reduce.json', {'status': 'completed'})
        return {'status': 'completed', 'ok': True}
    monkeypatch.setattr(reducer, 'run_smart_summary_global_reduce', resume)
    assert not bundle_next.bundle_advance(tmp_path, execute=False)['advanced']
    assert calls == []
    result = bundle_next.bundle_advance(tmp_path, execute=True)
    assert result['advanced'] and result['after']['status'] == 'ready'
    assert calls == [{'reuse_candidate': True, 'execute': False}]


@pytest.mark.parametrize('runner', ['agent-session', 'background'])
def test_autopilot_leases_and_imports_each_review_phase(tmp_path, monkeypatch, runner):
    from video_knowledge_pipeline import autopilot
    from test_autopilot import _submission
    from test_summary_claim_review import answer
    root = tmp_path / 'bundle'
    bundle(root)
    isolate_upstream(monkeypatch)
    monkeypatch.setattr(autopilot, '_refresh_route', lambda job, root: {'resolved_route': 'local-only'})
    monkeypatch.setattr(autopilot, '_default_background_executor', lambda job: None)
    job_root = tmp_path / 'jobs'
    started = autopilot.autopilot_start(root, runner=runner, execution_route='local-only', job_root=job_root)
    job_id = started['job_id']
    fingerprints = []
    def forbidden_background(work):
        pytest.fail('interactive evidence review must not call a background Provider')
    for phase, run in [('annotate', 'annotator'), ('review', 'reviewer')]:
        nxt = autopilot.autopilot_next(job_id, job_root=job_root, background_executor=forbidden_background)
        work = nxt['work_item']
        assert work['task_type'] == 'summary_claim_review'
        assert work['model_policy']['provider_health_check'] == 'not_applicable'
        assert work['model_policy']['remote_fallback_allowed'] is False
        assert len(autopilot._canonical_bytes(work)) <= autopilot.MAX_WORK_ITEM_BYTES
        pack_ref = next(ref for ref in work['input_refs'] if ref['role'] == 'summary_claim_review_task')
        task = json.loads(Path(pack_ref['path']).read_text(encoding='utf-8'))
        assert task['phase'] == phase
        fingerprints.append(work['action_fingerprint'])
        # Same outstanding lease resumes without minting or invoking anything.
        assert autopilot.autopilot_next(job_id, job_root=job_root)['work_item']['lease_id'] == work['lease_id']
        payload = _submission(work, answer(task, run=run))
        if phase == 'annotate':
            invalid = json.loads(json.dumps(payload))
            invalid['result']['observed_source_ids'] = []
            rejected = autopilot.autopilot_submit(job_id, submission=invalid, job_root=job_root)
            assert rejected['status'] == 'rejected' and not (root / review.STATE).exists()
        accepted = autopilot.autopilot_submit(job_id, submission=payload, job_root=job_root)
        assert accepted['status'] == 'succeeded'
        assert accepted['import_result']['transaction']['restore_plan']
    assert fingerprints[0] != fingerprints[1]  # each phase is new evidence work
    assert review.summary_claim_review_status(root)['passed'] is True


def test_autopilot_pack_drift_rejected_without_review_write(tmp_path, monkeypatch):
    from video_knowledge_pipeline import autopilot
    from test_autopilot import _submission
    from test_summary_claim_review import answer
    root = tmp_path / 'bundle'
    bundle(root)
    isolate_upstream(monkeypatch)
    monkeypatch.setattr(autopilot, '_refresh_route', lambda job, root: {'resolved_route': 'local-only'})
    job_root = tmp_path / 'jobs'
    started = autopilot.autopilot_start(root, execution_route='local-only', job_root=job_root)
    work = autopilot.autopilot_next(started['job_id'], job_root=job_root)['work_item']
    ref = next(ref for ref in work['input_refs'] if ref['role'] == 'summary_claim_review_task')
    task = json.loads(Path(ref['path']).read_text(encoding='utf-8'))
    payload = _submission(work, answer(task))
    write_json(Path(ref['path']), {'drift': True})
    receipt = autopilot.autopilot_submit(started['job_id'], submission=payload, job_root=job_root)
    assert receipt['status'] == 'rejected'
    assert not (root / review.STATE).exists()
