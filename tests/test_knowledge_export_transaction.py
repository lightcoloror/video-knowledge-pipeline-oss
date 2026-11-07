"""Offline reader-export consistency through the actual exporter and journal."""
from pathlib import Path
import json

import pytest

from video_knowledge_pipeline import knowledge_note_export as export
from video_knowledge_pipeline import transcript_semantic_transaction as tx


def bundle(tmp_path):
    from test_smart_summary_llm_rewrite import _bundle
    root = _bundle(tmp_path)
    (root / 'exports').mkdir(exist_ok=True)
    for name in ('full-body.md', 'full-transcript.md', 'knowledge-note.md', 'smart-summary.md'):
        (root / 'exports' / name).write_bytes(b'previous reader output\r\n')
    return root


def test_actual_export_has_joint_recoverable_reader_projection(tmp_path):
    root = bundle(tmp_path)
    before = (root / 'exports/full-body.md').read_bytes()
    result = export.export_knowledge_note(root, run_transcript_evidence_check=False)
    transaction = result['export_transaction']
    assert transaction['profile'] == 'knowledge_export'
    assert {'exports/full-body.md', 'exports/reader-export-receipt.json',
            'exports/export-summary.json', 'manifest.json'} <= {r['target'] for r in transaction['artifacts']}
    assert tx.apply_transcript_semantic_restore(root, transaction['restore_plan'], execute=True,
        confirm_transaction_id=transaction['transaction_id'])['ok']
    assert (root / 'exports/full-body.md').read_bytes() == before


@pytest.mark.parametrize('target', sorted(tx._profile_targets('knowledge_export')))
def test_export_process_loss_blocks_reentry_and_restores(tmp_path, monkeypatch, target):
    root = bundle(tmp_path)
    source_bytes = {name: (root / name).read_bytes() for name in ('timeline.json', 'normalized-transcript.json')}
    from video_knowledge_pipeline import codex_vision_session as vision
    original = vision._write_bytes_atomic
    def interrupted(path, raw):
        original(path, raw)
        if Path(path) == root / target:
            raise SystemExit('synthetic export loss')
    monkeypatch.setattr(vision, '_write_bytes_atomic', interrupted)
    with pytest.raises(SystemExit, match='synthetic export loss'):
        export.export_knowledge_note(root, run_transcript_evidence_check=False)
    monkeypatch.setattr(vision, '_write_bytes_atomic', original)
    status = tx.artifact_transaction_status(root, profile='knowledge_export')
    assert not status['passed']
    assert not export.canonical_export_integrity_status(root)['passed']
    monkeypatch.setattr(export, 'reconcile_ebook_pipeline_checkpoints',
        lambda *a, **k: pytest.fail('pending export must block before preparatory writes'))
    with pytest.raises(RuntimeError, match='pending_artifact_transaction'):
        export.export_knowledge_note(root, run_transcript_evidence_check=False)
    assert tx.apply_transcript_semantic_restore(root, root / status['restore_plan'], execute=True,
        confirm_transaction_id=status['transaction_id'])['ok']
    assert (root / 'exports/full-body.md').read_bytes() == b'previous reader output\r\n'
    plan = json.loads((root / status['restore_plan']).read_text(encoding='utf-8'))
    for entry in plan['entries']:
        path = root / entry['target']
        if entry['existed_before']:
            assert path.read_bytes() == (root / entry['backup_path']).read_bytes()
        else:
            assert not path.exists()
    assert source_bytes == {name: (root / name).read_bytes() for name in source_bytes}
    assert tx.apply_transcript_semantic_restore(root, root / status['restore_plan'], execute=True,
        confirm_transaction_id=status['transaction_id'])['status'] == 'already_restored'


def test_normalized_source_drift_refuses_reader_commit(tmp_path, monkeypatch):
    root = bundle(tmp_path)
    before = (root / 'exports/full-body.md').read_bytes()
    original = export._render_full_body
    def drift(*args, **kwargs):
        value = original(*args, **kwargs)
        source = root / 'normalized-transcript.json'
        source.write_bytes(source.read_bytes() + b'\n')
        return value
    monkeypatch.setattr(export, '_render_full_body', drift)
    with pytest.raises(RuntimeError, match='semantic_transaction_input_drift'):
        export.export_knowledge_note(root, run_transcript_evidence_check=False)
    assert (root / 'exports/full-body.md').read_bytes() == before


def test_new_canonical_source_during_render_is_not_silently_admitted(tmp_path, monkeypatch):
    root = bundle(tmp_path)
    before = (root / 'exports/full-body.md').read_bytes()
    original = export._render_full_body
    def drift(*args, **kwargs):
        value = original(*args, **kwargs)
        (root / 'source-arbitrated-transcript.json').write_bytes((root / 'normalized-transcript.json').read_bytes())
        return value
    monkeypatch.setattr(export, '_render_full_body', drift)
    with pytest.raises(RuntimeError, match='export_dependency_set_drift'):
        export.export_knowledge_note(root, run_transcript_evidence_check=False)
    assert (root / 'exports/full-body.md').read_bytes() == before


def test_export_no_write_preserves_all_fixture_bytes(tmp_path):
    root = bundle(tmp_path)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result = export.export_knowledge_note(root, run_transcript_evidence_check=False, write=False)
    assert 'export_transaction' not in result
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_reader_receipt_hashes_final_bytes_and_registry_error_is_not_success(tmp_path, monkeypatch):
    root = bundle(tmp_path)
    # Give the real exporter an admitted canonical source so it writes a receipt.
    (root / 'source-arbitrated-transcript.json').write_bytes((root / 'normalized-transcript.json').read_bytes())
    def unavailable(*args, **kwargs):
        raise OSError('synthetic registry unavailable')
    monkeypatch.setattr(export, '_register_export_run', unavailable)
    result = export.export_knowledge_note(root, run_transcript_evidence_check=False)
    assert result['status'] == 'exported_registry_pending'
    assert not result['ok']
    receipt = json.loads((root / 'exports/reader-export-receipt.json').read_text(encoding='utf-8'))
    from video_knowledge_pipeline.reader_export_receipt import receipt_matches_reader_files
    assert receipt_matches_reader_files(receipt, canonical_transcript=root / 'source-arbitrated-transcript.json',
        full_transcript=root / 'exports/full-transcript.md', full_body=root / 'exports/full-body.md',
        reading_note=root / 'exports/knowledge-note.md')


def test_export_restore_refuses_operator_edit(tmp_path):
    root = bundle(tmp_path)
    result = export.export_knowledge_note(root, run_transcript_evidence_check=False)
    transaction = result['export_transaction']
    body = root / 'exports/full-body.md'
    body.write_bytes(b'operator revision')
    assert not tx.apply_transcript_semantic_restore(root, transaction['restore_plan'], execute=True,
        confirm_transaction_id=transaction['transaction_id'])['ok']
    assert body.read_bytes() == b'operator revision'
