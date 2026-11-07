"""Numeric interpretation must not be grounded only in an altered excerpt."""
from copy import deepcopy
import pytest

from test_summary_claim_review import fixture, answer, submit
from video_knowledge_pipeline import summary_claim_review as review
from video_knowledge_pipeline.evidence_locator import JsonEvidenceSnapshot
from video_knowledge_pipeline.canonical_json import canonical_json_sha256 as digest
from video_knowledge_pipeline.storage import write_json


@pytest.mark.parametrize('source_shape', ['scalar', 'text_object', 'text_with_numeric_metadata'])
def test_changed_excerpt_cannot_validate_agreeing_invented_numeric_slots(tmp_path, source_shape):
    plan, facts = fixture(tmp_path)
    if source_shape != 'scalar':
        source = {'text': plan['overview']['text']}
        if source_shape == 'text_with_numeric_metadata':
            source.update(frame_index=30, confidence=30, metadata={'text': '30天'})
        write_json(tmp_path / 'timeline.json', [source])
        facts['sections'][0]['evidence_refs'][0]['source_locators'] = [JsonEvidenceSnapshot.read(tmp_path, 'timeline.json').locator([0])]
    ref = facts['sections'][0]['evidence_refs'][0]
    ref['snippet'] = ref['snippet'].replace('90', '30')
    plan['overview']['text'] = ref['snippet']
    write_json(tmp_path / review.PLAN, plan)
    write_json(tmp_path / review.FACTS, facts)
    task = review.prepare_summary_claim_review(tmp_path)
    result = answer(task)
    for row in result['claims'] + result['assertions']:
        row['slots']['value'] = '30'
    write_json(tmp_path / 'answer.json', result)
    before = (tmp_path / 'timeline.json').read_bytes()
    receipt = review.import_summary_claim_review(tmp_path, tmp_path / 'answer.json')
    assert not receipt['ok'] and not receipt['changed'], receipt
    assert 'original_numeric' in str(receipt['errors'])
    assert not (tmp_path / review.STATE).exists()
    assert (tmp_path / receipt['raw_result']).read_bytes() == (tmp_path / 'answer.json').read_bytes()
    assert (tmp_path / 'timeline.json').read_bytes() == before


def test_old_equal_but_altered_excerpt_is_rechecked_against_original(tmp_path, monkeypatch):
    plan, facts = fixture(tmp_path)
    assert submit(tmp_path, review.prepare_summary_claim_review(tmp_path))['ok']
    assert submit(tmp_path, review.prepare_summary_claim_review(tmp_path), run='reviewer')['ok']
    record = deepcopy(next(iter(review._state(tmp_path)['sessions'].values()))['overview.text'])
    ref = facts['sections'][0]['evidence_refs'][0]
    ref['snippet'] = ref['snippet'].replace('90', '30')
    plan['overview']['text'] = ref['snippet']
    annotation = record['annotation']
    annotation['claims'][0]['slots']['value'] = '30'
    annotation['claims'][0]['output_canonical_sha256'] = digest(plan['overview']['text'])
    row = annotation['assertions'][0]
    row.update(reference_canonical_sha256=digest(ref),
               span={'start': 0, 'end': len(ref['snippet']), 'canonical_sha256': digest(ref['snippet'])})
    row['slots']['value'] = '30'
    binding = review._view(plan, facts)[0]
    monkeypatch.setattr(review, '_state', lambda _: {'sessions': {binding: {'overview.text': record}}})
    report = review.summary_claim_review_report(tmp_path, plan, facts)
    assert not report['passed']
    assert 'original_numeric' in str(report['items'][0]['structured_audit']['errors'])


@pytest.mark.parametrize('shape', ['text', 'visual_fields', 'temporal_fields', 'legacy_summary'])
def test_known_text_shapes_and_normalized_numbers_remain_reviewable(tmp_path, shape):
    plan, facts = fixture(tmp_path)
    original = '活动费用两万元。'
    source = {'text': original}
    if shape == 'visual_fields': source = {'objects': [original], 'confidence': 30}
    if shape == 'temporal_fields': source = {'state_changes': [original], 'frames': 30}
    if shape == 'legacy_summary': source = {'summary': original}
    write_json(tmp_path / 'timeline.json', [source])
    ref = facts['sections'][0]['evidence_refs'][0]
    ref.update(snippet='活动费用20000元。', source_locators=[JsonEvidenceSnapshot.read(tmp_path, 'timeline.json').locator([0])])
    plan['overview']['text'] = ref['snippet']
    write_json(tmp_path / review.PLAN, plan)
    write_json(tmp_path / review.FACTS, facts)
    task = review.prepare_summary_claim_review(tmp_path)
    result = answer(task)
    for row in result['claims'] + result['assertions']:
        row['slots'].update(subject='活动', property='费用', value='20000', unit='元', conditions=[], version='')
    write_json(tmp_path / 'answer.json', result)
    receipt = review.import_summary_claim_review(tmp_path, tmp_path / 'answer.json')
    assert receipt['ok'], receipt
    assert submit(tmp_path, review.prepare_summary_claim_review(tmp_path), run='reviewer')['ok']
    report = review.summary_claim_review_report(tmp_path, plan, facts)
    assert report['passed']
    assert report['items'][0]['structured_audit']['original_numeric_audit']['status'] == 'not_refuted'
    assert report['factual_truth'] == 'not_assessed'


def test_metadata_only_cannot_supply_a_number(tmp_path):
    plan, facts = fixture(tmp_path)
    write_json(tmp_path / 'timeline.json', [{'frame_index': 90, 'confidence': 90}])
    facts['sections'][0]['evidence_refs'][0]['source_locators'] = [JsonEvidenceSnapshot.read(tmp_path, 'timeline.json').locator([0])]
    write_json(tmp_path / review.FACTS, facts)
    task = review.prepare_summary_claim_review(tmp_path)
    receipt = submit(tmp_path, task)
    assert not receipt['ok'] and 'original_numeric' in str(receipt['errors'])
    assert not (tmp_path / review.STATE).exists()


def test_repair_excerpt_preserves_source_and_does_not_approve_wrong_summary(tmp_path):
    plan, facts = fixture(tmp_path)
    ref = facts['sections'][0]['evidence_refs'][0]
    original = ref['snippet']
    ref['snippet'] = original.replace('90', '30')
    plan['overview']['text'] = ref['snippet']
    write_json(tmp_path / review.PLAN, plan)
    write_json(tmp_path / review.FACTS, facts)
    task = review.prepare_summary_claim_review(tmp_path)
    result = answer(task)
    for row in result['claims'] + result['assertions']: row['slots']['value'] = '30'
    write_json(tmp_path / 'answer.json', result)
    assert not review.import_summary_claim_review(tmp_path, tmp_path / 'answer.json')['ok']
    source_before = (tmp_path / 'timeline.json').read_bytes()
    ref['snippet'] = original
    write_json(tmp_path / review.FACTS, facts)
    repaired = review.prepare_summary_claim_review(tmp_path)
    assert repaired['task_id'] != task['task_id']
    result = answer(repaired)
    result['claims'][0]['slots']['value'] = '30'
    write_json(tmp_path / 'answer.json', result)
    assert review.import_summary_claim_review(tmp_path, tmp_path / 'answer.json')['ok']
    assert submit(tmp_path, review.prepare_summary_claim_review(tmp_path), run='reviewer')['ok']
    report = review.summary_claim_review_report(tmp_path, plan, facts)
    assert not report['passed']
    assert report['items'][0]['structured_audit']['claims'][0]['status'] == 'contradiction'
    assert (tmp_path / 'timeline.json').read_bytes() == source_before
