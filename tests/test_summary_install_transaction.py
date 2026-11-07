"""Synthetic final-candidate admission; no model, network or media processing."""
from pathlib import Path

import pytest

from video_knowledge_pipeline import smart_summary_codex as codex
from video_knowledge_pipeline import transcript_semantic_transaction as tx
from video_knowledge_pipeline.storage import write_json


TARGETS = ('exports/smart-summary.codex.md', 'exports/smart-summary-dependency-snapshot.json',
    'exports/smart-summary-quality.json', 'exports/smart-summary-quality.md',
    'exports/smart-summary-codex-status.json', 'exports/smart-summary-codex-status.md')


def fixture(root, monkeypatch, *, passed=True):
    write_json(root / 'manifest.json', {'title': 'synthetic'})
    write_json(root / 'timeline.json', [{'text': 'original evidence'}])
    before = {}
    for name in TARGETS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'old\r\n')
        before[name] = path.read_bytes()
    source = root / 'new.md'
    source.write_bytes(b'\xef\xbb\xbf<!-- codex_final -->\r\nnew evidence-based summary\r\n')
    monkeypatch.setattr(codex, 'evaluate_production_artifact_gate', lambda *a, **k: {'formal_generation_allowed': True})
    monkeypatch.setattr(codex, '_refresh_smart_summary_input_pack', lambda *a, **k: {'status': 'fixture'})
    monkeypatch.setattr(codex, 'build_smart_summary_chapter_pack', lambda *a, **k: {'status': 'fixture'})
    monkeypatch.setattr(codex, 'smart_summary_quality_check', lambda *a, **k: {
        'schema': codex.QUALITY_SCHEMA, 'passed': passed, 'status': 'passed' if passed else 'failed', 'checks': []})
    monkeypatch.setattr(codex, '_register_smart_summary_codex_run', lambda *a, **k: {'status': 'completed'})
    return source, before


def assert_before(root, before):
    for name, data in before.items():
        assert (root / name).read_bytes() == data, name


def test_bad_candidate_never_overwrites_installed_summary_or_reports(tmp_path, monkeypatch):
    source, before = fixture(tmp_path, monkeypatch, passed=False)
    result = codex.generate_smart_summary_with_codex(tmp_path, input_md=source)
    assert result['status'] == 'blocked_quality' and not result['ok']
    assert_before(tmp_path, before)
    assert Path(result['validation_receipt']).is_file()


def test_same_prose_reviewed_with_new_evidence_keeps_separate_immutable_admission(tmp_path, monkeypatch):
    import os
    from video_knowledge_pipeline.artifact_freshness import build_dependency_snapshot
    source, before = fixture(tmp_path,monkeypatch,passed=False)
    evidence = tmp_path/'source-reference.md'
    evidence.write_text('first source',encoding='utf-8')
    first_snapshot = build_dependency_snapshot(tmp_path,subject='test',inputs=[{'role':'reference','path':evidence}],source_run_id='test',producer_schema='test')
    first = codex._install_summary_candidate(tmp_path,source=source,write=True,handoff_snapshot=first_snapshot)
    old = Path(first['candidate_checked_path'])
    old_time = old.stat().st_mtime_ns
    evidence.write_text('second source',encoding='utf-8')
    second_snapshot = build_dependency_snapshot(tmp_path,subject='test',inputs=[{'role':'reference','path':evidence}],source_run_id='test',producer_schema='test')
    os.utime(source,ns=(source.stat().st_atime_ns,old_time+1_000_000_000))
    second = codex._install_summary_candidate(tmp_path,source=source,write=True,handoff_snapshot=second_snapshot)
    new = Path(second['candidate_checked_path'])
    assert new != old and new.read_bytes() == old.read_bytes() == source.read_bytes()
    assert new.stat().st_mtime_ns > old_time and old.stat().st_mtime_ns == old_time
    assert first['status'] == second['status'] == 'blocked_quality'
    assert_before(tmp_path,before)


def test_section_apply_rejected_candidate_does_not_replace_installed_summary(tmp_path, monkeypatch):
    from video_knowledge_pipeline import smart_summary_section_apply as section
    source, before = fixture(tmp_path, monkeypatch, passed=False)
    monkeypatch.setattr(section, 'evaluate_production_artifact_gate', lambda *a, **k: {'formal_generation_allowed': True})
    write_json(tmp_path / 'exports/smart-summary-section-workflow.json', {
        'sections': [{'section_id': 'one', 'title': 'Synthetic', 'start': 0, 'end': 20}]})
    revisions = tmp_path / 'section-revisions.json'
    write_json(revisions, {'rows': [{'section_id': 'one', 'final_markdown': 'Candidate evidence.'}]})
    result = section.apply_smart_summary_sections(tmp_path, input_json=revisions)
    assert result['status'] == 'blocked_quality' and result['installed_section_count'] == 0
    assert result['candidate_section_count'] == 1
    assert_before(tmp_path, before)
    assert Path(result['artifacts']['candidate']).is_file()
    assert Path(result['codex_status']['validation_receipt']).is_file()


def test_success_is_byte_preserving_and_all_targets_restore_together(tmp_path, monkeypatch):
    source, before = fixture(tmp_path, monkeypatch)
    original = source.read_bytes()
    result = codex.generate_smart_summary_with_codex(tmp_path, input_md=source)
    receipt = result['installation_transaction']
    assert receipt['profile'] == 'summary_install'
    assert {row['target'] for row in receipt['artifacts']} == set(TARGETS)
    assert (tmp_path / TARGETS[0]).read_bytes() == original == source.read_bytes()
    assert tx.apply_transcript_semantic_restore(tmp_path, receipt['restore_plan'], execute=True,
        confirm_transaction_id=receipt['transaction_id'])['ok']
    assert_before(tmp_path, before)


@pytest.mark.parametrize('target', TARGETS)
def test_install_crash_after_each_target_has_exact_restore(tmp_path, monkeypatch, target):
    source, before = fixture(tmp_path, monkeypatch)
    original = tx.write_text_atomic
    from video_knowledge_pipeline import codex_vision_session as vision
    original_bytes = vision._write_bytes_atomic
    def crash(path, content):
        original(path, content)
        if Path(path) == tmp_path / target:
            raise SystemExit('install crash')
    def crash_bytes(path, content):
        original_bytes(path, content)
        if Path(path) == tmp_path / target:
            raise SystemExit('install crash')
    monkeypatch.setattr(tx, 'write_text_atomic', crash)
    monkeypatch.setattr(vision, '_write_bytes_atomic', crash_bytes)
    with pytest.raises(SystemExit, match='install crash'):
        codex.generate_smart_summary_with_codex(tmp_path, input_md=source)
    monkeypatch.setattr(tx, 'write_text_atomic', original)
    monkeypatch.setattr(vision, '_write_bytes_atomic', original_bytes)
    status = tx.artifact_transaction_status(tmp_path, profile='summary_install')
    assert status['status'] == 'blocked'
    retry = codex.generate_smart_summary_with_codex(tmp_path, input_md=source)
    assert retry['status'] == 'blocked_install_transaction'
    from video_knowledge_pipeline.summary_claim_review import summary_claim_review_quality_gate
    from video_knowledge_pipeline.summary_stage_coverage import stored_summary_stage_coverage
    assert summary_claim_review_quality_gate(tmp_path, 'old')['passed'] is False
    assert stored_summary_stage_coverage(tmp_path)['status'] == 'blocked'
    assert tx.apply_transcript_semantic_restore(tmp_path, tmp_path / status['restore_plan'], execute=True,
        confirm_transaction_id=status['transaction_id'])['ok']
    assert_before(tmp_path, before)


def test_no_write_validates_requested_candidate_not_previous_installed_text(tmp_path, monkeypatch):
    source, before = fixture(tmp_path, monkeypatch)
    seen = []
    def quality(*a, **k):
        seen.append(Path(k['summary_path']).read_bytes())
        return {'passed': False, 'status': 'failed', 'checks': []}
    monkeypatch.setattr(codex, 'smart_summary_quality_check', quality)
    result = codex.generate_smart_summary_with_codex(tmp_path, input_md=source, write=False)
    assert result['quality']['passed'] is False
    assert seen == [source.read_bytes()]
    assert_before(tmp_path, before)
    assert not (tmp_path / 'runs').exists()


def test_quality_race_cannot_install_with_changed_review_input(tmp_path, monkeypatch):
    source, before = fixture(tmp_path, monkeypatch)
    def quality(*a, **k):
        write_json(tmp_path / 'summary-claim-review.json', {'new_operator_review': True})
        return {'passed': True, 'status': 'passed', 'checks': []}
    monkeypatch.setattr(codex, 'smart_summary_quality_check', quality)
    with pytest.raises(RuntimeError, match='input_drift:summary-claim-review.json'):
        codex.generate_smart_summary_with_codex(tmp_path, input_md=source)
    assert_before(tmp_path, before)


def test_quality_race_preserves_operator_edit_to_installed_target(tmp_path, monkeypatch):
    source, before = fixture(tmp_path, monkeypatch)
    target = tmp_path / TARGETS[0]
    def quality(*a, **k):
        target.write_bytes(b'operator edit\r\n')
        return {'passed': True, 'status': 'passed', 'checks': []}
    monkeypatch.setattr(codex, 'smart_summary_quality_check', quality)
    with pytest.raises(RuntimeError, match='input_drift:exports/smart-summary.codex.md'):
        codex.generate_smart_summary_with_codex(tmp_path, input_md=source)
    assert target.read_bytes() == b'operator edit\r\n'
    assert_before(tmp_path, {name: value for name, value in before.items() if name != TARGETS[0]})


def test_arbitrary_empty_prospective_snapshot_cannot_bypass_freshness(tmp_path):
    from video_knowledge_pipeline.artifact_freshness import build_dependency_snapshot
    write_json(tmp_path / 'manifest.json', {})
    write_json(tmp_path / 'timeline.json', [{'text': 'stable evidence'}])
    forged = build_dependency_snapshot(tmp_path, subject='smart-summary-final', inputs=[],
        source_run_id='smart-summary-codex', producer_schema=codex.SCHEMA)
    result = codex._smart_summary_evidence_freshness_gate(tmp_path, snapshot=forged)
    assert result['passed'] is False


def test_binary_and_text_transactions_have_distinct_serialization_identity():
    assert tx._transaction_id({'exports/smart-summary.codex.md': b'a\nb'}, {}, {'pack_sha256': 'x'}) != tx._transaction_id(
        {'exports/smart-summary.codex.md': 'a\nb'}, {}, {'pack_sha256': 'x'})
    assert tx._transaction_id({'exports/smart-summary.codex.md': 'a\nb'}, {}, {'pack_sha256': 'x'}) == 'semcorr-3a26ef95fc0f64bd089e'


def test_reduce_no_write_never_installs_an_old_or_missing_candidate(tmp_path, monkeypatch):
    from test_summary_model_input import live_fixture
    from video_knowledge_pipeline import summary_claim_review
    root, module, calls = live_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(summary_claim_review, 'summary_claim_review_report', lambda *a, **k: {'passed': True})
    monkeypatch.setattr(module, 'generate_smart_summary_with_codex', lambda *a, **k: pytest.fail('no-write must not install'))
    result = module.run_smart_summary_global_reduce(root, execute=True, install=True, write=False)
    assert result['status'] == 'validated_not_installed' and result['ok'] is False
    assert len(calls) == 1  # synthetic requested generation; installation remains disabled
    assert not (root / 'exports/smart-summary.codex.md').exists()


def test_reduce_does_not_report_complete_when_install_registry_is_pending(tmp_path, monkeypatch):
    from test_summary_model_input import live_fixture
    from video_knowledge_pipeline import summary_claim_review
    root, module, calls = live_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(summary_claim_review, 'summary_claim_review_report', lambda *a, **k: {'passed': True})
    monkeypatch.setattr(module, 'generate_smart_summary_with_codex', lambda *a, **k: {
        'status': 'installed_registry_pending', 'ok': False, 'quality': {'passed': True},
        'next_actions': ['Repair registry without regeneration.']})
    result = module.run_smart_summary_global_reduce(root, execute=True, install=True, write=True)
    assert result['status'] == 'installed_registry_pending' and result['ok'] is False
    assert result['next_actions'] == ['Repair registry without regeneration.']
    assert len(calls) == 1


@pytest.mark.parametrize('write,status', [(False, 'validated_not_installed'),
    (True, 'blocked_quality'), (True, 'installed_registry_pending')])
def test_llm_rewrite_preserves_install_outcome(tmp_path, monkeypatch, write, status):
    source, before = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(codex, 'resolve_text_provider_config', lambda *a, **k: {
        'provider': 'fixture', 'base_url': 'http://example.invalid/v1', 'model': 'synthetic'})
    monkeypatch.setattr(codex, 'prepare_smart_summary_llm_rewrite', lambda *a, **k: {'prompt_path': str(source)})
    monkeypatch.setattr(codex, 'call_openai_compatible_text', lambda *a, **k: {'ok': True, 'content': '# Candidate'})
    calls = []
    def install(*args, **kwargs):
        calls.append(kwargs)
        if not write:
            pytest.fail('no-write must not read a stale rewrite candidate')
        return {'ok': False, 'status': status, 'quality': {'passed': status != 'blocked_quality'},
                'next_actions': ['Repair existing candidate without regeneration.']}
    monkeypatch.setattr(codex, 'generate_smart_summary_with_codex', install)
    result = codex.run_smart_summary_llm_rewrite(tmp_path, execute=True, write=write)
    assert result['ok'] is False and result['status'] == status
    assert len(calls) == int(write)
    assert_before(tmp_path, before)
