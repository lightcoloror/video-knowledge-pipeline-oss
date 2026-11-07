"""Synthetic frontdoor coverage; no model construction or private media."""
import json

import pytest

from test_courseware_material import fixture_bundle, page_result
from video_knowledge_pipeline import courseware_cli as cli
from video_knowledge_pipeline import courseware_incremental_runtime as runtime
from video_knowledge_pipeline.courseware_material import import_courseware_page, read_scan, store_scan


@pytest.mark.parametrize('flags,count,expected', [
    ([], 8, 'checkpoint'),
    (['--worker-mode', 'checkpoint'], 8, 'checkpoint'),
    (['--worker-mode', 'auto'], 8, 'checkpoint'),
    (['--worker-mode', 'auto'], 9, 'batch'),
    (['--worker-mode', 'auto'], 32, 'batch'),
    (['--worker-mode', 'batch'], 1, 'batch'),
    (['--reuse-worker'], 1, 'batch'),
])
def test_explicit_mode_routes_only_supplied_indexes(monkeypatch, capsys, flags, count, expected):
    indexes = list(range(count, 0, -1))
    calls = []
    monkeypatch.setattr(runtime, 'execute', lambda *a, **k: calls.append((a, k)) or {'ok': True})
    assert cli.main(['courseware-incremental-ocr', 'synthetic', '--indexes',
                     ','.join(str(n) for n in indexes), '--model-policy', 'synthetic-policy',
                     '--execute', *flags]) == 0
    args, kwargs = calls[0]
    assert args == ('synthetic', indexes, 'synthetic-policy')
    assert bool(kwargs.get('reuse_worker')) == (expected == 'batch')
    plan = json.loads(capsys.readouterr().out)['worker_plan']
    assert plan['selected_mode'] == expected
    assert [i for group in plan['checkpoint_groups'] for i in group] == indexes
    assert max(map(len, plan['checkpoint_groups'])) <= 8
    assert not plan['execution_performed'] and plan['concurrency'] == 1


@pytest.mark.parametrize('flags,indexes', [
    ([], '1,2,3,4,5,6,7,8,9'),
    (['--worker-mode', 'auto'], ','.join(str(n) for n in range(1, 34))),
    (['--worker-mode', 'batch'], '1,1'),
    (['--reuse-worker'], '0'),
    (['--worker-mode', 'auto'], ''),
])
def test_invalid_plan_rejects_before_runtime(monkeypatch, capsys, flags, indexes):
    monkeypatch.setattr(runtime, 'preflight', lambda *a: pytest.fail('invalid plan reached resource probe'))
    monkeypatch.setattr(runtime, 'execute', lambda *a, **k: pytest.fail('invalid plan reached executor'))
    assert cli.main(['courseware-incremental-ocr', 'synthetic', '--indexes', indexes,
                     '--model-policy', 'unused', *flags]) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'blocked'


def test_auto_preview_retains_blocking_preflight_and_does_not_execute(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(runtime, 'preflight', lambda p: calls.append(p) or {
        'ready': False, 'blockers': ['existing OCR runner active; do not overlap'],
        'execution_performed': False, 'provider_calls': 0})
    monkeypatch.setattr(runtime, 'execute', lambda *a, **k: pytest.fail('preview executed OCR'))
    assert cli.main(['courseware-incremental-ocr', 'synthetic', '--indexes', '9,8,7,6,5,4,3,2,1',
                     '--model-policy', 'synthetic-policy', '--worker-mode', 'auto']) == 0
    value = json.loads(capsys.readouterr().out)
    assert calls == ['synthetic-policy']
    assert value['ready'] is False and value['blockers']
    assert not value['execution_performed'] and value['provider_calls'] == 0
    assert value['worker_plan']['selected_mode'] == 'batch'
    assert not value['worker_plan']['resume_identity_verified']


def test_main_cli_delegates_explicit_batch_preview(monkeypatch, capsys):
    from video_knowledge_pipeline.cli import main
    monkeypatch.setattr(runtime, 'preflight', lambda p: {'ready': True, 'execution_performed': False})
    monkeypatch.setattr(runtime, 'execute', lambda *a, **k: pytest.fail('main CLI executed preview'))
    assert main(['courseware-incremental-ocr', 'synthetic', '--indexes', '1',
                 '--model-policy', 'synthetic-policy', '--worker-mode', 'batch']) == 0
    value = json.loads(capsys.readouterr().out)
    assert value['worker_plan']['selected_mode'] == 'batch'
    assert not value['execution_performed']


def test_refresh_follows_resolved_mode_and_never_promotes_small_auto(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(runtime, 'execute', lambda *a, **k: calls.append(k) or {'ok': True})
    args = ['courseware-incremental-ocr', 'synthetic', '--model-policy', 'unused',
            '--worker-mode', 'auto', '--refresh-material', '--execute']
    assert cli.main([*args, '--indexes', '1,2']) == 2
    assert not calls
    capsys.readouterr()
    assert cli.main([*args, '--indexes', '1,2,3,4,5,6,7,8,9']) == 0
    assert calls == [{'resume': False, 'reuse_worker': True, 'refresh_material': True}]


def test_alias_and_explicit_mode_cannot_conflict():
    with pytest.raises(SystemExit) as exc:
        cli.main(['courseware-incremental-ocr', 'synthetic', '--model-policy', 'unused',
                  '--indexes', '1', '--reuse-worker', '--worker-mode', 'checkpoint'])
    assert exc.value.code == 2


def test_status_suggestion_is_bounded_and_has_no_model_or_process_probes(tmp_path, monkeypatch, capsys):
    root = fixture_bundle(tmp_path, 40)
    protected = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(runtime, 'preflight', lambda *a: pytest.fail('status must not probe runtime'))
    monkeypatch.setattr(runtime, 'execute', lambda *a, **k: pytest.fail('status must not execute'))
    assert cli.main(['courseware-status', str(root)]) == 0
    value = json.loads(capsys.readouterr().out)
    suggestion = value['next_ocr']
    assert suggestion['indexes'] == list(range(1, 33))
    assert suggestion['pending_count'] == 40 and suggestion['remaining_after_suggestion'] == 8
    assert suggestion['worker_plan']['selected_mode'] == 'batch'
    assert suggestion['index_space'] == 'scan_candidate'
    assert '--execute' not in suggestion['preview_argv']
    assert '--resume' not in suggestion['preview_argv']
    assert suggestion['preview_argv'][-2:] == ['--worker-mode', 'auto']
    assert not suggestion['resource_gates_checked'] and not suggestion['execution_performed']
    assert protected == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_status_uses_scan_indexes_and_excludes_uncaptured_or_imported_gaps(tmp_path, capsys):
    root = fixture_bundle(tmp_path, 4)
    result = page_result(root, 'page-1')
    result.update(status='unavailable', blocks=[], gaps=['requires independent review'])
    import_courseware_page(root, result)
    state = read_scan(root)
    state['candidates'][2].update(frame='', frame_sha256='')
    state['candidates'] = [state['candidates'][3], state['candidates'][1],
                           state['candidates'][2], state['candidates'][0]]
    store_scan(root, state)
    assert cli.main(['courseware-status', str(root), '--full-json']) == 0
    value = json.loads(capsys.readouterr().out)
    assert [p['page_id'] for p in value['pages']] == ['page-0', 'page-1', 'page-2', 'page-3']
    assert value['next_ocr']['indexes'] == [1, 4]
    assert value['next_ocr']['worker_plan']['selected_mode'] == 'checkpoint'


def test_status_without_captured_pending_pages_has_no_ocr_command(tmp_path, capsys):
    root = fixture_bundle(tmp_path)
    import_courseware_page(root, page_result(root))
    assert cli.main(['courseware-status', str(root)]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value['next_ocr']['status'] == 'no_captured_unimported_pages'
    assert 'preview_argv' not in value['next_ocr']
    assert not value['complete']  # A routing hint cannot replace independent review.


def test_auto_failure_retains_exact_selection_and_alias_resume_identity(tmp_path, monkeypatch, capsys):
    from test_courseware_dedupe import image_bundle
    from test_courseware_incremental_ocr import picture
    from test_courseware_ocr_batch import batch
    from video_knowledge_pipeline.storage import read_json
    root = image_bundle(tmp_path, [picture()] * 8 + [picture(b=21)])
    def synthetic_execute(bundle, indexes, policy, **kwargs):
        assert kwargs['reuse_worker'] is True
        return batch(bundle, indexes, fail_call=0 if kwargs['resume'] else 2,
                     resume=kwargs['resume'])[0]
    monkeypatch.setattr(runtime, 'execute', synthetic_execute)
    args = ['courseware-incremental-ocr', str(root), '--indexes', '1,2,3,4,5,6,7,8,9',
            '--model-policy', 'synthetic-policy', '--execute']
    assert cli.main([*args, '--worker-mode', 'auto']) == 2
    blocked = json.loads(capsys.readouterr().out)
    assert blocked['worker_plan']['indexes'] == list(range(1, 10))
    assert not blocked['automatic_retry']
    assert not blocked['worker_plan']['resume_identity_verified']
    receipt_path = next((root / 'courseware/incremental-batches').glob('*.json'))
    before = read_json(receipt_path)
    page_before = (root / 'courseware/pages/page-0.json').read_bytes()
    assert cli.main([*args, '--reuse-worker', '--resume']) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed['run_id'] == before['run_id']
    assert resumed['contract'] == before['contract']
    assert resumed['detector_calls'] == 1
    assert (root / 'courseware/pages/page-0.json').read_bytes() == page_before
