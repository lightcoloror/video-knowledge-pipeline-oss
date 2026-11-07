"""Actual review-import checks for invented numeric annotations, synthetic only."""
from __future__ import annotations

from copy import deepcopy

import pytest

from video_knowledge_pipeline import summary_claim_review as review
from video_knowledge_pipeline.evidence_locator import JsonEvidenceSnapshot
from video_knowledge_pipeline.storage import write_json
from test_summary_claim_review import answer, fixture, submit


@pytest.mark.parametrize('mutation', ['source_value', 'output_value', 'both_agree', 'source_span'])
def test_numeric_annotation_not_in_bound_text_is_atomically_rejected(tmp_path, mutation):
    fixture(tmp_path)
    task = review.prepare_summary_claim_review(tmp_path)
    result = answer(task)
    if mutation in ('source_value', 'both_agree'):
        result['assertions'][0]['slots']['value'] = '30'
    if mutation in ('output_value', 'both_agree'):
        result['claims'][0]['slots']['value'] = '30'
    if mutation == 'source_span':
        result['assertions'][0]['end'] = task['sources'][0]['snippet'].index('90')
    raw = tmp_path / 'answer.json'
    write_json(raw, result)
    before = (tmp_path / 'timeline.json').read_bytes()
    receipt = review.import_summary_claim_review(tmp_path, raw)
    assert not receipt['ok'] and not receipt['changed'], receipt
    assert 'numeric_surface' in str(receipt['errors'])
    assert not (tmp_path / review.STATE).exists()
    assert (tmp_path / receipt['raw_result']).read_bytes() == raw.read_bytes()
    assert (tmp_path / 'timeline.json').read_bytes() == before
    assert review.prepare_summary_claim_review(tmp_path) == task
    # Rejection is repairable through the same importer, not a second state machine.
    assert submit(tmp_path, task)['ok']
    assert submit(tmp_path, review.prepare_summary_claim_review(tmp_path), run='reviewer')['ok']


def quantity_fixture(root, source_text, output_text):
    plan, facts = fixture(root)
    write_json(root / 'timeline.json', [{'text': source_text}])
    ref = facts['sections'][0]['evidence_refs'][0]
    ref.update(snippet=source_text, source_locators=[JsonEvidenceSnapshot.read(root, 'timeline.json').locator([0, 'text'])])
    plan['overview']['text'] = output_text
    write_json(root / review.PLAN, plan)
    write_json(root / review.FACTS, facts)
    return plan, facts


@pytest.mark.parametrize(('source', 'output', 'value', 'unit'), [
    ('活动费用两万元。', '活动费用20000元。', '20000', '元'),
    ('讲解持续六个小时。', '讲解持续360分钟。', '360', '分钟'),
    ('比例百分之八十。', '比例80%。', '80', '%'),
])
def test_existing_numeric_normalizer_supports_equivalent_renderings_and_restore(tmp_path, source, output, value, unit):
    from video_knowledge_pipeline.transcript_semantic_transaction import apply_transcript_semantic_restore
    plan, facts = quantity_fixture(tmp_path, source, output)
    task = review.prepare_summary_claim_review(tmp_path)
    result = answer(task)
    slots = dict(subject='活动', property='合成数量', value=value, unit=unit,
                 negated=False, conditions=[], time_scope='', version='')
    result['claims'][0]['slots'] = slots
    result['assertions'][0]['slots'] = deepcopy(slots)
    write_json(tmp_path / 'answer.json', result)
    imported = review.import_summary_claim_review(tmp_path, tmp_path / 'answer.json')
    assert imported['ok'], imported
    task = review.prepare_summary_claim_review(tmp_path)
    final = submit(tmp_path, task, run='reviewer')
    assert final['ok']
    report = review.summary_claim_review_report(tmp_path, plan, facts)
    assert report['passed']
    audit = report['items'][0]['structured_audit']
    assert audit['numeric_surface_audit']['status'] == 'not_refuted'
    assert audit['source_interpretation_accuracy'] == 'not_assessed'
    tx = final['transaction']
    assert apply_transcript_semantic_restore(tmp_path, tx['restore_plan'], execute=True,
        confirm_transaction_id=tx['transaction_id'])['status'] == 'restored'
    assert review.prepare_summary_claim_review(tmp_path) == task


def test_known_unit_mismatch_is_not_approved_by_equal_annotations(tmp_path):
    quantity_fixture(tmp_path, '费用90元。', '费用90元。')
    task = review.prepare_summary_claim_review(tmp_path)
    result = answer(task)
    for row in result['claims'] + result['assertions']:
        row['slots'].update(value='90', unit='%')
    write_json(tmp_path / 'answer.json', result)
    receipt = review.import_summary_claim_review(tmp_path, tmp_path / 'answer.json')
    assert not receipt['ok'] and 'numeric_surface' in str(receipt['errors'])


def test_reprojection_does_not_grandfather_old_equal_but_wrong_annotations(tmp_path, monkeypatch):
    plan, facts = fixture(tmp_path)
    task = review.prepare_summary_claim_review(tmp_path)
    assert submit(tmp_path, task)['ok']
    assert submit(tmp_path, review.prepare_summary_claim_review(tmp_path), run='reviewer')['ok']
    state = review._state(tmp_path)
    record = next(iter(state['sessions'].values()))['overview.text']
    for row in record['annotation']['claims'] + record['annotation']['assertions']:
        row['slots']['value'] = '30'
    # Simulate a previously admitted record; do not bless it by re-importing.
    monkeypatch.setattr(review, '_state', lambda root: state)
    before = (tmp_path / review.STATE).read_bytes()
    report = review.summary_claim_review_report(tmp_path, plan, facts)
    assert not report['passed']
    assert report['items'][0]['status'] == 'review_gap'
    assert 'numeric_surface' in str(report['items'][0]['structured_audit']['errors'])
    assert (tmp_path / review.STATE).read_bytes() == before
