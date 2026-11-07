"""Real Reduce snapshot writer + existing journal/restore, synthetic text only."""
import json
from pathlib import Path

import pytest

from video_knowledge_pipeline import smart_summary_global_reduce as reduce
from video_knowledge_pipeline import transcript_semantic_transaction as tx
from video_knowledge_pipeline.storage import write_json


TARGETS = ('manifest.json', 'exports/smart-summary-global-reduce.json',
    'exports/smart-summary-global-reduce.md', 'exports/smart-summary-global-reduce-prompt.md',
    'exports/smart-summary-global-reduce-candidate.md', 'exports/smart-summary-global-reduce-raw-response.txt',
    'exports/smart-summary-reader-plan.json', 'exports/smart-summary-chapter-fact-pack.json')


def seed(root):
    before = {}
    for number, name in enumerate(TARGETS):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if number % 2 == 0 or name == 'manifest.json':
            path.write_bytes(b'{"preserve_user_field": true}\r\n')
            before[name] = path.read_bytes()
        else:
            before[name] = None
    (root / 'timeline.json').write_bytes(b'[{"text":"source remains unchanged"}]')
    return before


def save(root):
    return reduce._write(root, {'schema': reduce.SCHEMA, 'status': 'reduce_quality_failed',
        'ok': False, 'updated_at': 'synthetic'}, prompt='exact prompt\nsecond line',
        candidate='candidate only', raw_candidate='{"raw": "response"}',
        reader_plan={'schema': 'fixture', 'plan': 'new'},
        fact_pack={'schema': 'fixture', 'sections': [], 'revision': 'new'}, write=True)


def assert_before(root, before):
    for name, value in before.items():
        assert ((root / name).read_bytes() if (root / name).exists() else None) == value, name
    assert (root / 'timeline.json').read_bytes() == b'[{"text":"source remains unchanged"}]'


@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('hard', [False, True])
def test_each_snapshot_write_failure_rolls_back_or_has_complete_restore(tmp_path, monkeypatch, target, hard):
    before = seed(tmp_path)
    original = tx.write_text_atomic
    fired = False
    def crash(path, content):
        nonlocal fired
        original(path, content)
        if Path(path) == tmp_path / target and not fired:
            fired = True
            raise (SystemExit('synthetic process loss') if hard else RuntimeError('synthetic write failure'))
    monkeypatch.setattr(tx, 'write_text_atomic', crash)
    with pytest.raises((SystemExit, RuntimeError), match='synthetic'):
        save(tmp_path)
    monkeypatch.setattr(tx, 'write_text_atomic', original)
    plans = list((tmp_path / 'runs/summary-reduce-snapshot').glob('*/restore-plan.json'))
    assert len(plans) == 1
    plan = json.loads(plans[0].read_text(encoding='utf-8'))
    assert {entry['target'] for entry in plan['entries']} == set(TARGETS)
    if not hard:
        assert_before(tmp_path, before)
    restored = tx.apply_transcript_semantic_restore(tmp_path, plans[0], execute=True,
        confirm_transaction_id=plan['transaction_id'])
    assert restored['ok']
    assert_before(tmp_path, before)
    assert tx.apply_transcript_semantic_restore(tmp_path, plans[0], execute=True,
        confirm_transaction_id=plan['transaction_id'])['status'] == 'already_restored'


def test_snapshot_success_repeat_and_restore_preserve_original_bytes(tmp_path):
    before = seed(tmp_path)
    result = save(tmp_path)
    receipt = result['snapshot_transaction']
    assert receipt['profile'] == 'summary_reduce_snapshot'
    assert receipt['artifact_count'] == len(TARGETS)
    assert json.loads((tmp_path / 'manifest.json').read_text())['preserve_user_field'] is True
    journal_before = Path(receipt['journal']).read_bytes()
    again = save(tmp_path)
    assert again['snapshot_transaction']['status'] == 'no_op'
    assert Path(receipt['journal']).read_bytes() == journal_before
    assert tx.apply_transcript_semantic_restore(tmp_path, receipt['restore_plan'], execute=True,
        confirm_transaction_id=receipt['transaction_id'])['ok']
    assert_before(tmp_path, before)


def test_snapshot_restore_never_overwrites_operator_edit(tmp_path):
    seed(tmp_path)
    receipt = save(tmp_path)['snapshot_transaction']
    target = tmp_path / 'exports/smart-summary-reader-plan.json'
    target.write_text('operator edit', encoding='utf-8')
    manifest = (tmp_path / 'manifest.json').read_bytes()
    restored = tx.apply_transcript_semantic_restore(tmp_path, receipt['restore_plan'], execute=True,
        confirm_transaction_id=receipt['transaction_id'])
    assert restored['status'] == 'blocked'
    assert target.read_text() == 'operator edit'
    assert (tmp_path / 'manifest.json').read_bytes() == manifest


def test_same_transaction_replay_does_not_replace_before_backup(tmp_path):
    target = tmp_path / 'summary-claim-review.json'
    target.write_bytes(b'original\r\n')
    kwargs = dict(artifacts={target.name: 'new'}, expected_inputs={},
        operation_identity={'pack_sha256': 'a' * 64}, profile='summary_claim_review')
    first = tx.commit_transcript_semantic_artifacts(tmp_path, **kwargs)
    journal = Path(first['journal']).read_bytes()
    second = tx.commit_transcript_semantic_artifacts(tmp_path, **kwargs)
    assert second['status'] == 'already_committed'
    assert Path(first['journal']).read_bytes() == journal
    assert tx.apply_transcript_semantic_restore(tmp_path, first['restore_plan'], execute=True,
        confirm_transaction_id=first['transaction_id'])['ok']
    assert target.read_bytes() == b'original\r\n'


def test_interrupted_transaction_replay_preserves_recovery_journal(tmp_path, monkeypatch):
    target = tmp_path / 'summary-claim-review.json'
    target.write_text('old')
    kwargs = dict(artifacts={target.name: 'new'}, expected_inputs={},
        operation_identity={'pack_sha256': 'a' * 64}, profile='summary_claim_review')
    original = tx.write_text_atomic
    def crash(path, content):
        original(path, content)
        if Path(path) == target:
            raise SystemExit('process loss')
    monkeypatch.setattr(tx, 'write_text_atomic', crash)
    with pytest.raises(SystemExit):
        tx.commit_transcript_semantic_artifacts(tmp_path, **kwargs)
    monkeypatch.setattr(tx, 'write_text_atomic', original)
    journal = next((tmp_path / 'runs/summary-claim-review').glob('*/transaction-journal.json'))
    before = journal.read_bytes()
    with pytest.raises(RuntimeError, match='existing_transaction_requires_restore'):
        tx.commit_transcript_semantic_artifacts(tmp_path, **kwargs)
    assert journal.read_bytes() == before


def test_summary_profile_rejects_extra_nested_target(tmp_path):
    with pytest.raises(ValueError, match='target_not_allowed'):
        tx.commit_transcript_semantic_artifacts(tmp_path, artifacts={'exports/unregistered.md': 'bad'},
            expected_inputs={}, operation_identity={'pack_sha256': 'a' * 64}, profile='summary_reduce_snapshot')


def test_no_write_snapshot_creates_no_artifacts(tmp_path):
    result = reduce._write(tmp_path, {'status': 'planned'}, prompt='fixture', write=False)
    assert result['status'] == 'planned' and list(tmp_path.iterdir()) == []


def test_interrupted_snapshot_blocks_new_payload_provider_and_admission(tmp_path, monkeypatch):
    seed(tmp_path)
    original = tx.write_text_atomic
    def crash(path, content):
        original(path, content)
        if Path(path) == tmp_path / 'exports/smart-summary-reader-plan.json':
            raise SystemExit('process loss')
    monkeypatch.setattr(tx, 'write_text_atomic', crash)
    with pytest.raises(SystemExit):
        save(tmp_path)
    monkeypatch.setattr(tx, 'write_text_atomic', original)
    before = {name: (tmp_path / name).read_bytes() if (tmp_path / name).is_file() else None for name in TARGETS}
    with pytest.raises(RuntimeError, match='pending_artifact_transaction'):
        reduce._write(tmp_path, {'status': 'new_intent'}, prompt='different', write=True)
    # Front door must stop before even reaching the production/provider route.
    monkeypatch.setattr(reduce, 'evaluate_production_artifact_gate', lambda *a, **k: pytest.fail('must not run'))
    result = reduce.run_smart_summary_global_reduce(tmp_path, execute=True)
    assert result['status'] == 'blocked_snapshot_transaction'
    assert result['provider_call_performed'] is False
    from video_knowledge_pipeline.summary_claim_review import summary_claim_review_quality_gate, summary_claim_review_status
    from video_knowledge_pipeline.summary_stage_coverage import stored_summary_stage_coverage
    assert summary_claim_review_quality_gate(tmp_path, 'candidate')['passed'] is False
    assert stored_summary_stage_coverage(tmp_path)['status'] == 'blocked'
    assert summary_claim_review_status(tmp_path)['status'] == 'blocked'
    for name, data in before.items():
        assert ((tmp_path / name).read_bytes() if (tmp_path / name).is_file() else None) == data
    pointer = json.loads((tmp_path / 'runs/summary-reduce-snapshot/latest-transaction.json').read_text())
    journal = tmp_path / 'runs/summary-reduce-snapshot' / pointer['transaction_id'] / 'transaction-journal.json'
    recovery = tx.apply_transcript_semantic_restore(tmp_path, journal, execute=True,
        confirm_transaction_id=pointer['transaction_id'])
    assert recovery['ok']
    assert tx.artifact_transaction_status(tmp_path, profile='summary_reduce_snapshot')['passed'] is True


@pytest.mark.parametrize('target', ['pointer', 'journal'])
def test_snapshot_transaction_metadata_drift_is_not_legacy_success(tmp_path, target):
    seed(tmp_path)
    result = save(tmp_path)
    path = (tmp_path / 'runs/summary-reduce-snapshot/latest-transaction.json' if target == 'pointer'
            else Path(result['snapshot_transaction']['journal']))
    path.write_text('{"transaction_id": "../../outside"}')
    from video_knowledge_pipeline.summary_stage_coverage import stored_summary_stage_coverage
    assert stored_summary_stage_coverage(tmp_path)['status'] == 'blocked'


def test_failed_input_precondition_does_not_publish_unrecoverable_pointer(tmp_path):
    seed(tmp_path)
    with pytest.raises(RuntimeError, match='input_drift'):
        tx.commit_transcript_semantic_artifacts(tmp_path, artifacts={'manifest.json': '{}'},
            expected_inputs={'manifest.json': '0' * 64}, operation_identity={'pack_sha256': 'a' * 64},
            profile='summary_reduce_snapshot')
    assert not (tmp_path / 'runs/summary-reduce-snapshot/latest-transaction.json').exists()
    assert tx.artifact_transaction_status(tmp_path, profile='summary_reduce_snapshot')['passed']


def test_real_reduce_process_loss_restores_then_uses_saved_response_once(tmp_path, monkeypatch):
    from test_summary_model_input import live_fixture
    from video_knowledge_pipeline.summary_model_input import RESPONSE_SCHEMA
    root, module, calls = live_fixture(tmp_path, monkeypatch)
    original = tx.write_text_atomic
    def crash(path, content):
        original(path, content)
        if Path(path) == root / 'exports/smart-summary-reader-plan.json':
            raise SystemExit('snapshot process loss')
    monkeypatch.setattr(tx, 'write_text_atomic', crash)
    with pytest.raises(SystemExit, match='snapshot process loss'):
        module.run_smart_summary_global_reduce(root, execute=True, install=False)
    monkeypatch.setattr(tx, 'write_text_atomic', original)
    assert len(calls) == 1
    status = tx.artifact_transaction_status(root, profile='summary_reduce_snapshot')
    assert status['status'] == 'blocked'
    assert module.run_smart_summary_global_reduce(root, execute=True, install=False)['status'] == 'blocked_snapshot_transaction'
    assert len(calls) == 1
    responses = [p for p in (root / 'exports/summary-input-records').glob('*.json')
                 if json.loads(p.read_text(encoding='utf-8')).get('schema') == RESPONSE_SCHEMA]
    assert len(responses) == 1
    raw_bytes = responses[0].read_bytes()
    assert tx.apply_transcript_semantic_restore(root, root / status['restore_plan'], execute=True,
        confirm_transaction_id=status['transaction_id'])['ok']
    recovered = module.run_smart_summary_global_reduce(root, recovery_execution_report=responses[0], install=False)
    assert len(calls) == 1 and responses[0].read_bytes() == raw_bytes
    assert recovered['generation_input']['status'] == 'bound_gateway_input'
    assert recovered['recovery']['network_requests_made'] is False
    assert recovered['snapshot_transaction']['status'] == 'committed'
