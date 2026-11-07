"""One admitted synthetic Bundle through real summary/review/acceptance gates.

Only generation is substituted. The authored reader-plan fixture is also the
script of the synthetic course: quote round-tripping, NOT measured model/NLI
accuracy or evidence that real speech/frames have been completely extracted.
"""
from copy import deepcopy
from pathlib import Path
import json
import socket

import pytest

from video_knowledge_pipeline import smart_summary_global_reduce as reducer
from video_knowledge_pipeline import summary_claim_review as review
from video_knowledge_pipeline import smart_summary_codex as codex
from video_knowledge_pipeline.canonical_json import canonical_json_bytes
from video_knowledge_pipeline.evidence_locator import JsonEvidenceSnapshot
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.smart_summary_reader_plan import _evidence_bound_items, _plan_text_values
from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline import transcript_semantic_correction as semantic
from video_knowledge_pipeline.transcript_quality_gate import run_transcript_quality_gate
from video_knowledge_pipeline.acceptance_check import acceptance_check
from video_knowledge_pipeline.knowledge_note_export import export_knowledge_note
from video_knowledge_pipeline.summary_stage_coverage import current_summary_stage_coverage, stored_summary_stage_coverage
from video_knowledge_pipeline import transcript_semantic_transaction as tx
from test_smart_summary_reader_plan import _valid_plan


CONFIG = {'provider': 'local', 'model': 'synthetic-fixture', 'base_url': 'http://127.0.0.1:1/v1'}
INSTALL_TARGETS = ('exports/smart-summary.codex.md', 'exports/smart-summary-dependency-snapshot.json',
    'exports/smart-summary-quality.json', 'exports/smart-summary-quality.md',
    'exports/smart-summary-codex-status.json', 'exports/smart-summary-codex-status.md')


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('No socket connection is authorized in this synthetic chain')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket.socket, 'connect_ex', forbidden)


def target_bytes(root):
    return {p: (root / p).read_bytes() if (root / p).exists() else None for p in INSTALL_TARGETS}


def seed(root, plan=None):
    plan = deepcopy(plan) if plan is not None else _valid_plan()
    plan['review_items'] = [{'text': '视觉证据未执行，画面信息待复核。', 'time_range': '',
                             'evidence_ids': [], 'missing_evidence': '本次仅有合成文本证据'}]
    sections, segments = [], []
    texts = dict(_plan_text_values(plan))
    for i, (location, item) in enumerate(_evidence_bound_items(plan)):
        evidence_id = f'source-{i}'
        item['evidence_ids'] = [evidence_id]
        text = '\n'.join(value for name, value in texts.items() if name.rsplit('.', 1)[0] == location)
        segments.append({'index': i + 1, 'start': 0, 'end': 240, 'text': text})
        sections.append({'section_id': f'section-{i}', 'title': item.get('title', '完整课程概览'),
                         'time_range': '00:00:00.000 - 00:04:00.000',
                         'evidence': {'summary_sentences': [text]}, 'citations': []})
    plan['source_section_ids'] = [s['section_id'] for s in sections]
    for i, segment in enumerate(segments):
        segment.update(start=240 * i / len(segments), end=240 * (i + 1) / len(segments))
    write_json(root / 'manifest.json', {'title': '合成证据闭环课程',
        'normalized_transcript_json': 'normalized-transcript.json'})
    write_json(root / 'timeline.json', segments)
    write_json(root / 'normalized-transcript.json', {'segments': segments})
    snap = JsonEvidenceSnapshot.read(root, 'timeline.json')
    for i, section in enumerate(sections):
        section['citations'] = [{'citation_id': f'source-{i}', 'source_kind': 'asr',
            'snippet': segments[i]['text'], 'chunk_kind': 'moment', 'fact_status': 'candidate_evidence',
            'time_range': section['time_range'], 'source_locators_by_kind': {'asr': [snap.locator([i, 'text'])]}}]
    write_json(root / 'exports/smart-summary-section-workflow.json', {'sections': sections})
    write_json(root / 'exports/smart-summary-section-llm-revisions.json', {'rows': [
        {**{k: section[k] for k in ('section_id', 'title', 'time_range')},
         'final_markdown': segments[i]['text']} for i, section in enumerate(sections)]})
    write_json(root / 'quality-benchmark-key-points.json', {'key_points': [
        {'id': 'overview', 'text': plan['overview']['text'], 'evidence_ids': ['source-0']}],
        'source_bindings': [{'path': 'timeline.json', 'sha256': sha256_file(root / 'timeline.json')}]})
    semantic.build_transcript_semantic_correction_pack(root, write=True)
    return plan


def close_authored_transcript(root):
    """Exercise real discovery/import/closure; simulated judgment is explicit.

    No human confirmation is fabricated. The synthetic reviewer reports no
    correction for the authored script. This tests orchestration, not recall.
    """
    original = {p: sha256_file(root / p) for p in
                ('timeline.json', 'normalized-transcript.json')}
    cursor = 0
    for _ in range(16):
        pack = semantic.build_transcript_semantic_candidate_discovery_pack(root, cursor=cursor)
        answer = root / 'fixture-discovery-answer.json'
        write_json(answer, {'schema': semantic.CANDIDATE_SUGGESTIONS_SCHEMA,
            'source': 'synthetic-authored-script-reviewer', 'suggestions': [],
            'discovery_identity': pack['discovery_identity']})
        imported = semantic.import_transcript_semantic_candidate_suggestions(root, input_json=answer)
        assert imported['ok'] and imported['progress_recorded'], imported
        if pack['complete']:
            break
        cursor = pack['next_cursor']
    else:
        raise AssertionError('Synthetic discovery did not terminate')
    pack = read_json(root / 'transcript-semantic-correction-pack.json')
    script = '\n'.join(s['text'] for s in read_json(root / 'timeline.json'))
    decisions = []
    for row in pack['candidates']:
        assert row['original_text'] in script
        decisions.append({'candidate_id': row['candidate_id'], 'accept': False,
            'correction_type': row['correction_type'], 'original_text': row['original_text'],
            'corrected_text': row['original_text'], 'confidence': 0.95,
            'rationale': 'Synthetic authored-script fixture: preserve the supplied wording; not a model accuracy measurement.',
            'evidence_ids': row['evidence_ids'], 'needs_human_review': False})
    answer = root / 'fixture-semantic-answer.json'
    write_json(answer, {'schema': semantic.RESULT_SCHEMA,
        'source': 'synthetic-authored-script-reviewer', 'decisions': decisions})
    closure = semantic.transcript_semantic_correction_closure(root, input_json=answer)
    assert closure['status'] == 'completed_no_text_changes', closure
    impact = semantic.transcript_semantic_correction_impact_report(root)
    readable = semantic.transcript_semantic_correction_readable_impact_report(root)
    quality = run_transcript_quality_gate(root)
    assert quality['ok'], quality
    status = semantic.transcript_semantic_correction_status(root, write=True)
    assert status['ok'], status
    assert original == {p: sha256_file(root / p) for p in original}
    return {'closure': closure, 'impact': impact, 'readable': readable, 'status': status}


def quoted_answer(task, *, allow_partition_without_quote=False):
    text = task['output_text']
    found = [(s, s['snippet'].find(text)) for s in task['sources'] if text in s['snippet']]
    result = {'task_id': task['task_id'], 'task_pack_sha256': task['task_pack_sha256'],
              'executor': 'synthetic-quotation-fixture', 'model_actual': 'no-model-fixture',
              'executor_run_id': 'fixture-' + task['phase'],
              'observed_source_ids': [s['evidence_id'] for s in task['sources']]}
    if task['phase'] == 'annotate':
        slots = {'subject': '合成课文', 'property': '逐字引文', 'value': text, 'unit': '',
                 'negated': False, 'conditions': [], 'time_scope': '', 'version': 'fixture-v1'}
        result.update(claims=[{'claim_id': 'quoted-output', 'evidence_ids': task['allowed_citation_ids'], 'slots': slots}],
            assertions=[{'assertion_id': f'quote-{i}', 'evidence_id': s['evidence_id'],
                         'start': start, 'end': start + len(text), 'slots': deepcopy(slots)}
                        for i, (s, start) in enumerate(found)])
    else:
        assert found or allow_partition_without_quote, 'Fixture reviewer cannot approve a quotation absent from original script'
        result.update(source_interpretation_correct=True, output_interpretation_correct=True,
                      decomposition_complete=True, context_association_correct=True,
                      reason='Exact authored synthetic quotation; no factual interpretation accuracy claimed.')
    return result


def test_single_bundle_reduce_review_install_and_acceptance(tmp_path, monkeypatch):
    root = tmp_path / 'bundle'
    plan = seed(root)
    close_authored_transcript(root)
    exercise_chain(root, plan, monkeypatch)


def exercise_chain(root, plan, monkeypatch, *, answer=quoted_answer, inspect_stage=None,
                   expected_install_block=None):
    """Shared real chain; optional assertions do not replace production gates."""
    calls = []
    def synthetic_gateway(*args, **kwargs):
        calls.append(kwargs['messages'])
        return {'ok': True, 'content': json.dumps(plan, ensure_ascii=False)}
    monkeypatch.setattr(reducer, 'model_task_api_call', synthetic_gateway)
    before = {p: sha256_file(root / p) for p in ('timeline.json', 'normalized-transcript.json')}
    first = reducer.run_smart_summary_global_reduce(root, execute=True, provider_config=CONFIG, install=False)
    assert first['status'] == 'waiting_claim_review', first
    if inspect_stage:
        inspect_stage(root, 'prepared')
    facts = read_json(root / review.FACTS)
    binding, index, texts, citations = review._view(read_json(root / review.PLAN), facts)
    expected = {review.PLAN: sha256_file(root / review.PLAN), review.FACTS: sha256_file(root / review.FACTS),
                review.STATE: '0' * 64}
    # Two phases for every planned source partition; mixed inputs need more
    # than the old text-only fixture's arbitrary 128-iteration ceiling.
    planned_tasks = 2 * sum(len(review._source_batches(binding, location, text,
        review._scope_sources(root, index, facts, citations[location]), citations[location], expected))
        for location, text in texts.items())
    assert 0 < planned_tasks <= 512
    tasks = []
    for _ in range(planned_tasks + 1):
        task = review.prepare_summary_claim_review(root)
        if task.get('status') == 'complete_draft':
            break
        assert task.get('task_id'), task
        assert len(canonical_json_bytes(task)) <= 8192
        assert task['task_id'] not in tasks, 'Successful submit must advance the controller'
        tasks.append(task['task_id'])
        assert review.prepare_summary_claim_review(root)['task_id'] == task['task_id']
        if len(tasks) == 1:
            invalid = answer(task)
            invalid['observed_source_ids'] = []
            write_json(root / 'fixture-invalid.json', invalid)
            rejected = review.import_summary_claim_review(root, root / 'fixture-invalid.json')
            assert not rejected['ok'], rejected
            assert (root / rejected['raw_result']).read_bytes() == (root / 'fixture-invalid.json').read_bytes()
            assert review.prepare_summary_claim_review(root)['task_id'] == task['task_id']
            assert before == {p: sha256_file(root / p) for p in before}
        write_json(root / 'fixture-answer.json', answer(task))
        result = review.import_summary_claim_review(root, root / 'fixture-answer.json')
        assert result['ok'], result
    else:
        raise AssertionError('Synthetic review did not terminate in bounded tasks')
    assert len(tasks) == len(set(tasks))
    assert len(tasks) == planned_tasks
    resumed = reducer.run_smart_summary_global_reduce(root, reuse_candidate=True, provider_config=CONFIG, install=False)
    assert resumed['status'] == 'completed', resumed
    assert len(calls) == 1
    candidate = root / 'exports/smart-summary-global-reduce-candidate.md'
    prior_install = target_bytes(root)
    installed = codex.generate_smart_summary_with_codex(root, input_md=candidate)
    if expected_install_block:
        assert installed['status'] == 'blocked_quality' and not installed['ok'], installed
        failed = [row['key'] for row in installed['quality']['checks'] if not row['passed']]
        assert failed == [expected_install_block], installed['quality']['checks']
        assert target_bytes(root) == prior_install
        assert before == {p: sha256_file(root / p) for p in before}
        assert read_json(Path(installed['validation_receipt']))['status'] == 'blocked_quality'
        print(json.dumps({'fixture': 'mixed-unreviewed-courseware', 'review_tasks': len(tasks),
            'install_status': installed['status'], 'failed_checks': failed,
            'real_model_calls': 0, 'installed_target_mutations': 0}))
        return
    assert installed['ok'], installed
    receipt = installed['installation_transaction']
    restored = tx.apply_transcript_semantic_restore(root, receipt['restore_plan'], execute=True,
        confirm_transaction_id=receipt['transaction_id'])
    assert restored['ok'], restored
    assert target_bytes(root) == prior_install
    installed = codex.generate_smart_summary_with_codex(root, input_md=candidate)
    assert installed['ok'], installed
    exported = export_knowledge_note(root)
    assert exported['ok'], exported
    assert before == {p: sha256_file(root / p) for p in before}
    stages = current_summary_stage_coverage(root)
    assert stages['passed'], stages
    acceptance = acceptance_check(root, refresh=True, write=True)
    assert acceptance['status'] == 'human_review_required', acceptance
    assert acceptance['blockers'] == [], acceptance
    assert stages['whole_media_completeness'] == 'not_assessed'
    assert stages['factual_truth'] == 'not_assessed'
    if inspect_stage:
        inspect_stage(root, 'exported')
    omitted = stored_summary_stage_coverage(root, output_text='')
    assert omitted['passed'] is False, omitted
    assert omitted['points'][0]['first_loss'] == 'output', omitted
    # A byte change in cited original evidence invalidates the actual admission
    # path; cached review success must not protect a stale summary.
    timeline = read_json(root / 'timeline.json')
    timeline[0]['text'] += ' 新的合成补充。'
    write_json(root / 'timeline.json', timeline)
    stable_install = target_bytes(root)
    drifted = codex.generate_smart_summary_with_codex(root, input_md=candidate)
    assert not drifted['ok'], drifted
    assert target_bytes(root) == stable_install
    assert len(calls) == 1
    print(json.dumps({'fixture': 'authored-text-quote-roundtrip', 'review_tasks': len(tasks),
        'synthetic_generation_calls': len(calls), 'real_model_calls': 0,
        'acceptance_status': acceptance['status'], 'acceptance_blockers': acceptance['blockers'],
        'stage_recall': stages['stage_recall'], 'whole_media_completeness': stages['whole_media_completeness'],
        'missing_observation_rejected': True, 'omission_first_loss': omitted['points'][0]['first_loss'],
        'restore_status': restored['status'], 'drift_status': drifted['status']}, ensure_ascii=False))
