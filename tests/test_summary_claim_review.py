from __future__ import annotations

import copy
import json

import pytest

from video_knowledge_pipeline.canonical_json import canonical_json_sha256 as digest
from video_knowledge_pipeline.evidence_locator import JsonEvidenceSnapshot
from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.summary_claim_review import prepare_summary_claim_review, import_summary_claim_review, summary_claim_review_report


def fixture(root):
    write_json(root / 'timeline.json', [{'text': '产品甲2026版首次投保等待期90天。'}])
    loc = JsonEvidenceSnapshot.read(root, 'timeline.json').locator([0, 'text'])
    ref = {'evidence_id': 'r', 'snippet': '产品甲2026版首次投保等待期90天。',
           'source_kind': 'asr', 'fact_status': 'candidate_evidence', 'source_locators': [loc]}
    plan = {'overview': {'text': ref['snippet'], 'evidence_ids': ['r']}}
    pack = {'bundle_dir': str(root), 'sections': [{'section_id': 's', 'time_range': '00:00:00 - 00:00:10', 'evidence_refs': [ref]}]}
    write_json(root / 'exports/smart-summary-reader-plan.json', plan)
    write_json(root / 'exports/smart-summary-chapter-fact-pack.json', pack)
    return plan, pack


def answer(task, *, run='annotator-run'):
    slots = dict(subject='产品甲', property='等待期', value='90', unit='天', negated=False,
                 conditions=['首次投保'], time_scope='', version='2026版')
    result = {'task_id': task['task_id'], 'task_pack_sha256': task['task_pack_sha256'],
              'executor_run_id': run, 'executor': 'synthetic', 'model_actual': 'fixture'}
    if task['phase'] == 'annotate':
        result.update(observed_source_ids=['r'], claims=[{'claim_id': 'c', 'evidence_ids': ['r'], 'slots': slots}],
            assertions=[{'assertion_id': 'a', 'evidence_id': 'r', 'start': 0, 'end': len(task['sources'][0]['snippet']), 'slots': copy.deepcopy(slots)}])
    else:
        result.update(observed_source_ids=['r'], source_interpretation_correct=True,
                      output_interpretation_correct=True, decomposition_complete=True,
                      context_association_correct=True, reason='Synthetic exact source and output checked.')
    return result


def submit(root, task, **kwargs):
    path = root / 'result.json'
    write_json(path, answer(task, **kwargs))
    return import_summary_claim_review(root, path)


def test_prepare_import_review_gate_and_restore(tmp_path):
    from video_knowledge_pipeline.transcript_semantic_transaction import apply_transcript_semantic_restore
    plan, pack = fixture(tmp_path)
    before = (tmp_path / 'timeline.json').read_bytes()
    first = prepare_summary_claim_review(tmp_path)
    assert first['phase'] == 'annotate' and len(json.dumps(first, ensure_ascii=False).encode()) < 8192
    from jsonschema import Draft202012Validator
    Draft202012Validator(first['output_schema']).validate(answer(first))
    assert first['sources'][0]['original_objects'] == [plan['overview']['text']]
    assert first == prepare_summary_claim_review(tmp_path)
    imported = submit(tmp_path, first)
    assert imported['ok']
    assert not summary_claim_review_report(tmp_path, plan, pack)['passed']
    review = prepare_summary_claim_review(tmp_path)
    assert review['phase'] == 'review'
    Draft202012Validator(review['output_schema']).validate(answer(review, run='review'))
    assert len(json.dumps(review, ensure_ascii=False).encode()) < 8192
    same = submit(tmp_path, review)
    assert not same['ok'] and 'separate' in str(same['errors'])
    final = submit(tmp_path, review, run='independent-review-run')
    assert final['ok']
    assert summary_claim_review_report(tmp_path, plan, pack)['passed']
    assert prepare_summary_claim_review(tmp_path)['status'] == 'complete_draft'
    tx = final['transaction']
    restored = apply_transcript_semantic_restore(tmp_path, tx['restore_plan'], execute=True, confirm_transaction_id=tx['transaction_id'])
    assert restored['status'] == 'restored'
    assert not summary_claim_review_report(tmp_path, plan, pack)['passed']
    assert (tmp_path / 'timeline.json').read_bytes() == before


@pytest.mark.parametrize('mutate', ['task', 'model', 'missing_observation', 'extra', 'source_drift', 'output_drift'])
def test_invalid_submission_preserves_state_and_saves_raw(tmp_path, mutate):
    fixture(tmp_path)
    task = prepare_summary_claim_review(tmp_path)
    result = answer(task)
    if mutate == 'task': result['task_id'] = 'wrong'
    elif mutate == 'model': result['model_actual'] = ''
    elif mutate == 'missing_observation': result['observed_source_ids'] = []
    elif mutate == 'extra': result['unknown'] = True
    elif mutate == 'source_drift': write_json(tmp_path / 'timeline.json', ['changed'])
    elif mutate == 'output_drift': write_json(tmp_path / 'exports/smart-summary-reader-plan.json', {})
    path = tmp_path / 'result.json'
    write_json(path, result)
    receipt = import_summary_claim_review(tmp_path, path)
    assert not receipt['ok'] and not (tmp_path / 'summary-claim-review.json').exists()
    assert (tmp_path / receipt['raw_result']).read_bytes() == path.read_bytes()


def test_contradiction_cannot_be_approved_by_reviewer(tmp_path):
    plan, pack = fixture(tmp_path)
    plan['overview']['text'] = plan['overview']['text'].replace('90', '30')
    write_json(tmp_path / 'exports/smart-summary-reader-plan.json', plan)
    task = prepare_summary_claim_review(tmp_path)
    result = answer(task)
    result['claims'][0]['slots']['value'] = '30'
    path = tmp_path / 'result.json'
    write_json(path, result)
    assert import_summary_claim_review(tmp_path, path)['ok']
    review = prepare_summary_claim_review(tmp_path)
    assert submit(tmp_path, review, run='review')['ok']
    assert not summary_claim_review_report(tmp_path, plan, pack)['passed']


def test_transaction_profile_cannot_write_transcript(tmp_path):
    from video_knowledge_pipeline.transcript_semantic_transaction import commit_transcript_semantic_artifacts
    with pytest.raises(ValueError, match='not_allowed'):
        commit_transcript_semantic_artifacts(tmp_path, artifacts={'manifest.json': '{}'},
            expected_inputs={}, operation_identity={'pack_sha256': 'a' * 64}, profile='summary_claim_review')


def test_hard_crash_after_state_write_is_recoverable(tmp_path, monkeypatch):
    from video_knowledge_pipeline import transcript_semantic_transaction as tx
    fixture(tmp_path)
    task = prepare_summary_claim_review(tmp_path)
    original = tx.write_json
    def crash(path, value):
        if path.name == 'transaction-journal.json' and value.get('status') == 'committed':
            raise KeyboardInterrupt('synthetic crash after write')
        return original(path, value)
    monkeypatch.setattr(tx, 'write_json', crash)
    with pytest.raises(KeyboardInterrupt):
        submit(tmp_path, task)
    monkeypatch.setattr(tx, 'write_json', original)
    plans = list((tmp_path / 'runs/summary-claim-review').glob('semcorr-*/restore-plan.json'))
    assert len(plans) == 1
    value = json.loads(plans[0].read_text(encoding='utf-8'))
    assert tx.apply_transcript_semantic_restore(tmp_path, plans[0], execute=True,
        confirm_transaction_id=value['transaction_id'])['status'] == 'restored'


def test_public_reduce_waits_then_reuses_review_without_model_or_install(tmp_path, monkeypatch):
    from video_knowledge_pipeline import smart_summary_global_reduce as reducer
    plan, pack = fixture(tmp_path)
    write_json(tmp_path / 'manifest.json', {})
    write_json(tmp_path / 'exports/smart-summary-section-workflow.json', {'sections': [{'section_id': 's'}]})
    write_json(tmp_path / 'exports/smart-summary-section-llm-revisions.json', {'rows': [
        {'section_id': 's', 'final_markdown': plan['overview']['text'], 'time_range': '00:00:00 - 00:00:10'}]})
    # Isolate admission; full reader-format validation has separate regressions.
    monkeypatch.setattr(reducer, 'evaluate_production_artifact_gate', lambda *a, **k: {'formal_generation_allowed': True})
    monkeypatch.setattr(reducer, '_chapter_fact_pack', lambda *a, **k: pack)
    monkeypatch.setattr(reducer, 'validate_reader_plan', lambda *a, **k: {'passed': True, 'errors': []})
    monkeypatch.setattr(reducer, 'normalize_reader_plan_candidate', lambda p, **k: {'plan': p, 'repairs': []})
    monkeypatch.setattr(reducer, 'render_reader_summary', lambda *a, **k: 'Synthetic reviewed output')
    monkeypatch.setattr(reducer, '_shape_quality', lambda *a, **k: {'passed': True})
    monkeypatch.setattr(reducer, 'resolve_model_api_provider_config', lambda *a, **k: {})
    monkeypatch.setattr(reducer, 'resolve_text_provider_config', lambda *a, **k: {})
    monkeypatch.setattr(reducer, 'model_task_api_call', lambda *a, **k: pytest.fail('no model'))
    first = reducer.run_smart_summary_global_reduce(tmp_path, reuse_candidate=True, install=False)
    assert first['status'] == 'waiting_claim_review' and not first['ok']
    assert first['claim_review_task']['phase'] == 'annotate'
    assert not (tmp_path / 'exports/smart-summary.codex.md').exists()
    assert submit(tmp_path, first['claim_review_task'])['ok']
    review = prepare_summary_claim_review(tmp_path)
    assert submit(tmp_path, review, run='review')['ok']
    final = reducer.run_smart_summary_global_reduce(tmp_path, reuse_candidate=True, install=False)
    assert final['status'] == 'completed' and final['model_call']['reused_candidate']
    # --install=false may persist a reviewed candidate, not overwrite the
    # canonical installed summary before its independent installation gate.
    assert (tmp_path / 'exports/smart-summary-global-reduce-candidate.md').read_text() == 'Synthetic reviewed output'
    assert not (tmp_path / 'exports/smart-summary.codex.md').exists()


@pytest.mark.parametrize('before', [b'{"old":1}\n', b'{"old":1}\r\n', b'\xef\xbb\xbf{"old":1}'])
def test_review_transaction_restores_exact_original_bytes(tmp_path, before):
    from video_knowledge_pipeline import transcript_semantic_transaction as tx
    path = tmp_path / 'summary-claim-review.json'
    path.write_bytes(before)
    result = tx.commit_transcript_semantic_artifacts(tmp_path, artifacts={path.name: '{"new":2}'},
        expected_inputs={}, operation_identity={'pack_sha256': 'a' * 64}, profile='summary_claim_review')
    restored = tx.apply_transcript_semantic_restore(tmp_path, result['restore_plan'], execute=True,
                                                  confirm_transaction_id=result['transaction_id'])
    assert restored['status'] == 'restored' and path.read_bytes() == before


def test_duplicate_old_task_cannot_overwrite_annotation(tmp_path):
    fixture(tmp_path)
    task = prepare_summary_claim_review(tmp_path)
    assert submit(tmp_path, task)['ok']
    before = (tmp_path / 'summary-claim-review.json').read_bytes()
    second = submit(tmp_path, task)
    assert not second['ok'] and (tmp_path / 'summary-claim-review.json').read_bytes() == before


def test_task_pack_drift_and_overbudget_do_not_mutate_review_state(tmp_path):
    fixture(tmp_path)
    task = prepare_summary_claim_review(tmp_path)
    path = tmp_path / f'runs/summary-claim-review/tasks/{task["task_id"]}.json'
    write_json(path, {**task, 'quality': 'changed'})
    assert not submit(tmp_path, task)['ok']
    assert not (tmp_path / 'summary-claim-review.json').exists()
    facts_path = tmp_path / 'exports/smart-summary-chapter-fact-pack.json'
    facts = json.loads(facts_path.read_text(encoding='utf-8'))
    facts['sections'][0]['evidence_refs'][0]['snippet'] = '完整但超预算的原文' * 1200
    write_json(facts_path, facts)
    with pytest.raises(ValueError, match='budget'):
        prepare_summary_claim_review(tmp_path)


def test_cli_uses_same_protocol_without_model(tmp_path, capsys):
    from video_knowledge_pipeline.summary_claim_review import main
    fixture(tmp_path)
    assert main(['prepare', str(tmp_path)]) == 0
    task = json.loads(capsys.readouterr().out)
    write_json(tmp_path / 'result.json', answer(task))
    assert main(['import', str(tmp_path), '--input-json', str(tmp_path / 'result.json')]) == 0
    assert json.loads(capsys.readouterr().out)['ok']
    assert main(['status', str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'review_required'


def reviewed_markdown(root):
    from video_knowledge_pipeline.smart_summary_reader_plan import render_reader_summary
    from video_knowledge_pipeline.smart_summary_global_reduce import _normalise_markdown
    plan, _ = fixture(root)
    assert submit(root, prepare_summary_claim_review(root))['ok']
    assert submit(root, prepare_summary_claim_review(root), run='review')['ok']
    text = render_reader_summary(plan, title=root.name, first_time='00:00:00', last_time='00:00:10')
    return _normalise_markdown(text, title=root.name)


def test_quality_gate_checks_actual_rendered_text_and_source_freshness(tmp_path):
    from video_knowledge_pipeline.summary_claim_review import summary_claim_review_quality_gate
    text = reviewed_markdown(tmp_path)
    good = summary_claim_review_quality_gate(tmp_path, text)
    assert good['passed'] and good['evaluated']
    assert good['output_binding'] == 'exact_deterministic_render'
    assert summary_claim_review_quality_gate(tmp_path, text.replace('\n', '\r\n'))['passed']
    for changed in (text.replace('90', '30'), text + '\n补充：续保也有等待期。', text.replace('首次投保', '')):
        bad = summary_claim_review_quality_gate(tmp_path, changed)
        assert not bad['passed'] and bad['status'] == 'output_drift'
    write_json(tmp_path / 'timeline.json', [{'text': 'changed'}])
    assert not summary_claim_review_quality_gate(tmp_path, text)['passed']


def test_public_quality_and_installer_cannot_bypass_pending_review(tmp_path, monkeypatch):
    from video_knowledge_pipeline import smart_summary_codex as codex
    fixture(tmp_path)
    write_json(tmp_path / 'manifest.json', {})
    target = tmp_path / 'exports/smart-summary.codex.md'
    target.write_text('previous installed artifact', encoding='utf-8')
    candidate = tmp_path / 'candidate.md'
    candidate.write_text('new unreviewed candidate', encoding='utf-8')
    monkeypatch.setattr(codex, 'evaluate_production_artifact_gate', lambda *a, **k: {'formal_generation_allowed': True})
    result = codex.generate_smart_summary_with_codex(tmp_path, input_md=candidate)
    assert result['status'] == 'blocked_claim_review' and not result['ok']
    assert target.read_text(encoding='utf-8') == 'previous installed artifact'
    quality = codex.smart_summary_quality_check(tmp_path, write=False)
    check = next(c for c in quality['checks'] if c['key'] == 'summary_claim_review')
    assert not check['passed'] and check['evaluated']
    assert not quality['production_ready']


def test_legacy_gate_is_explicitly_unassessed_and_missing_registered_plan_blocks(tmp_path):
    from video_knowledge_pipeline.summary_claim_review import summary_claim_review_quality_gate
    legacy = summary_claim_review_quality_gate(tmp_path, 'historical text')
    assert not legacy['evaluated'] and legacy['status'] == 'legacy_not_assessed'
    write_json(tmp_path / 'manifest.json', {'smart_summary_reader_plan_json': 'exports/smart-summary-reader-plan.json'})
    missing = summary_claim_review_quality_gate(tmp_path, 'historical text')
    assert not missing['passed'] and missing['evaluated']


def many_sources(root, count=24, *, contrary_last=False):
    plan, pack = fixture(root)
    texts = [{'text': plan['overview']['text']} for _ in range(count)]
    if contrary_last:
        texts[-1]['text'] = texts[-1]['text'].replace('90', '30')
    write_json(root / 'timeline.json', texts)
    snapshot = JsonEvidenceSnapshot.read(root, 'timeline.json')
    refs = [{**pack['sections'][0]['evidence_refs'][0], 'evidence_id': f'r{i:03d}',
             'snippet': row['text'], 'source_locators': [snapshot.locator([i, 'text'])]}
            for i, row in enumerate(texts)]
    refs.append({'evidence_id': 'all', 'source_kind': 'evidence_group',
                 'fact_status': 'candidate_evidence', 'member_evidence_ids': [r['evidence_id'] for r in refs]})
    plan['overview']['evidence_ids'] = ['all']
    pack['sections'][0]['evidence_refs'] = refs
    write_json(root / 'exports/smart-summary-reader-plan.json', plan)
    write_json(root / 'exports/smart-summary-chapter-fact-pack.json', pack)
    return plan, pack


def shard_answer(task):
    result = answer(task, run='reviewer' if task['phase'] == 'review' else 'annotator')
    result['observed_source_ids'] = list(reversed([s['evidence_id'] for s in task['sources']]))
    if task['phase'] == 'annotate':
        result['claims'][0]['evidence_ids'] = task['allowed_citation_ids']
        result['assertions'] = []
        for i, source in enumerate(task['sources']):
            slots = copy.deepcopy(result['claims'][0]['slots'])
            slots['value'] = '30' if '30' in source['snippet'] else '90'
            result['assertions'].append({'assertion_id': f'a{i}', 'evidence_id': source['evidence_id'],
                'start': 0, 'end': len(source['snippet']), 'slots': slots})
    return result


@pytest.mark.parametrize('contrary_last', [False, True])
def test_many_source_shards_keep_all_evidence_and_late_contradiction(tmp_path, contrary_last):
    from video_knowledge_pipeline.canonical_json import canonical_json_bytes
    plan, pack = many_sources(tmp_path, contrary_last=contrary_last)
    seen = []
    partitions = set()
    for _ in range(100):
        task = prepare_summary_claim_review(tmp_path)
        if task['status'] != 'waiting_result':
            break
        assert len(canonical_json_bytes(task)) <= 8192
        assert task == prepare_summary_claim_review(tmp_path)
        partitions.add(task['shard']['partition_sha256'])
        if task['phase'] == 'annotate':
            seen.extend(s['evidence_id'] for s in task['sources'])
        write_json(tmp_path / 'result.json', shard_answer(task))
        result = import_summary_claim_review(tmp_path, tmp_path / 'result.json')
        assert result['ok'], result
    else:
        pytest.fail('review did not terminate')
    assert seen == [f'r{i:03d}' for i in range(24)]
    assert len(partitions) == 1
    report = summary_claim_review_report(tmp_path, plan, pack)
    assert report['passed'] is not contrary_last
    assert report['items'][0]['shard_coverage']['observed_source_count'] == 24
    if contrary_last:
        assert report['items'][0]['structured_audit']['claims'][0]['status'] != 'supported'


def test_partial_shard_review_does_not_complete_and_resume_has_no_replay(tmp_path):
    plan, pack = many_sources(tmp_path)
    first = prepare_summary_claim_review(tmp_path)
    write_json(tmp_path / 'result.json', shard_answer(first))
    assert import_summary_claim_review(tmp_path, tmp_path / 'result.json')['ok']
    review = prepare_summary_claim_review(tmp_path)
    assert review['phase'] == 'review' and review['shard'] == first['shard']
    assert not summary_claim_review_report(tmp_path, plan, pack)['passed']
    write_json(tmp_path / 'result.json', shard_answer(review))
    imported_review = import_summary_claim_review(tmp_path, tmp_path / 'result.json')
    assert imported_review['ok']
    next_task = prepare_summary_claim_review(tmp_path)
    assert next_task['shard']['index'] == 1 and next_task['phase'] == 'annotate'
    before = (tmp_path / 'summary-claim-review.json').read_bytes()
    write_json(tmp_path / 'result.json', shard_answer(first))
    assert not import_summary_claim_review(tmp_path, tmp_path / 'result.json')['ok']
    assert (tmp_path / 'summary-claim-review.json').read_bytes() == before
    from video_knowledge_pipeline.transcript_semantic_transaction import apply_transcript_semantic_restore
    tx = imported_review['transaction']
    assert apply_transcript_semantic_restore(tmp_path, tx['restore_plan'], execute=True,
                                             confirm_transaction_id=tx['transaction_id'])['status'] == 'restored'
    assert prepare_summary_claim_review(tmp_path) == review


def test_shard_annotation_too_large_for_review_is_atomic(tmp_path):
    fixture(tmp_path)
    task = prepare_summary_claim_review(tmp_path)
    result = answer(task)
    # Many independently identified source assertions fit import's outer byte
    # bound, but cannot fit an intact review packet. Never strand saved state.
    result['assertions'] = [{**copy.deepcopy(result['assertions'][0]), 'assertion_id': f'a{i}'} for i in range(32)]
    write_json(tmp_path / 'result.json', result)
    receipt = import_summary_claim_review(tmp_path, tmp_path / 'result.json')
    assert not receipt['ok'] and 'review_task_budget' in str(receipt['errors'])
    assert not (tmp_path / 'summary-claim-review.json').exists()
    assert (tmp_path / receipt['raw_result']).read_bytes() == (tmp_path / 'result.json').read_bytes()


def test_shard_output_interpretation_disagreement_is_not_majority_voted(tmp_path):
    plan, pack = many_sources(tmp_path)
    for _ in range(100):
        task = prepare_summary_claim_review(tmp_path)
        if task['status'] != 'waiting_result':
            break
        result = shard_answer(task)
        if task['phase'] == 'annotate' and task['shard']['index'] == task['shard']['count'] - 1:
            # Numeric values absent from the output now fail at import. Keep
            # this independent shard-merge test focused on interpretation
            # disagreement that still requires the declared semantic audit.
            result['claims'][0]['slots']['negated'] = True
        write_json(tmp_path / 'result.json', result)
        imported = import_summary_claim_review(tmp_path, tmp_path / 'result.json')
        assert imported['ok'], imported
    else:
        pytest.fail('review did not terminate')
    report = summary_claim_review_report(tmp_path, plan, pack)
    assert not report['passed']
    assert report['items'][0]['reasons'] == ['output_interpretation_disagreement']


def test_source_chapter_membership_drift_invalidates_review_even_if_flat_refs_match(tmp_path):
    from video_knowledge_pipeline.summary_citation_audit import build_citation_index
    plan, pack = fixture(tmp_path)
    second = {**pack['sections'][0]['evidence_refs'][0], 'evidence_id': 'second'}
    pack['sections'].append({'section_id': 'unrelated', 'time_range': '00:01:00 - 00:01:10', 'evidence_refs': [second]})
    write_json(tmp_path / 'exports/smart-summary-chapter-fact-pack.json', pack)
    assert submit(tmp_path, prepare_summary_claim_review(tmp_path))['ok']
    assert submit(tmp_path, prepare_summary_claim_review(tmp_path), run='review')['ok']
    assert summary_claim_review_report(tmp_path, plan, pack)['passed']
    moved = copy.deepcopy(pack)
    moved['sections'][0]['evidence_refs'].append(second)
    moved['sections'][1]['evidence_refs'] = []
    assert build_citation_index(pack).snapshot_sha256 == build_citation_index(moved).snapshot_sha256
    assert not summary_claim_review_report(tmp_path, plan, moved)['passed']
    changed_time = copy.deepcopy(pack)
    changed_time['sections'][0]['time_range'] = '00:10:00 - 00:10:10'
    assert not summary_claim_review_report(tmp_path, plan, changed_time)['passed']
