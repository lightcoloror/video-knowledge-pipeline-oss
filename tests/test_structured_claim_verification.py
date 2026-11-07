"""Declared-claim consistency and composite citation fixtures, not NLI accuracy."""
from copy import deepcopy
import json

import pytest

from video_knowledge_pipeline.canonical_json import canonical_json_sha256 as digest
from video_knowledge_pipeline.summary_citation_audit import build_citation_index
from video_knowledge_pipeline.structured_claim_verification import audit_structured_claims


def slots(**overrides):
    return dict(subject='产品甲', property='等待期', value='90', unit='天', negated=False,
                conditions=['首次投保'], time_scope='合同生效后', version='2026版', **overrides)


def fixture():
    refs = [dict(evidence_id='row', source_kind='ocr', fact_status='candidate_evidence',
                 snippet='产品甲 2026版：合同生效后等待期90天。'),
            dict(evidence_id='note', source_kind='ocr', fact_status='candidate_evidence',
                 snippet='该行脚注：仅限首次投保。'),
            dict(evidence_id='noise', source_kind='asr', fact_status='candidate_evidence',
                 snippet='现在休息十分钟。')]
    pack = {'sections': [{'evidence_refs': refs}]}
    index = build_citation_index(pack)
    complete = slots()
    row_slots = {key: val for key, val in complete.items() if key != 'conditions'}
    footnote = {key: complete[key] for key in ('subject', 'property', 'time_scope', 'version', 'conditions')}
    assertions = []
    for ref, fields in zip(refs, [row_slots, footnote]):
        assertions.append(dict(assertion_id=ref['evidence_id'] + '-assertion',
                               evidence_id=ref['evidence_id'], context_id='page-1:table-1:row-1',
                               reference_canonical_sha256=digest(ref),
                               span={'start': 0, 'end': len(ref['snippet']),
                                     'canonical_sha256': digest(ref['snippet'])}, slots=fields))
    scope = dict(schema='video_knowledge_pipeline.claim_verification_scope.v1',
                 reference_snapshot_sha256=index.snapshot_sha256,
                 required_locations=['overview.text'], assertions=assertions)
    text = '产品甲2026版合同生效后，首次投保的等待期为90天。'
    claim = dict(claim_id='claim-1', output_location='overview.text',
                 output_canonical_sha256=digest(text), evidence_ids=['row', 'note'], slots=complete)
    return index, scope, [claim], {'overview.text': text}


def audit(data):
    index, scope, claims, texts = data
    return audit_structured_claims(claims, scope=scope, citation_index=index,
                                   output_texts=texts, output_citations={'overview.text': ['row', 'note', 'noise']})


def test_composite_row_plus_footnote_supports_without_independent_votes():
    data = fixture()
    result = audit(data)
    assert result['valid'], result['errors']
    assert result['claims'][0]['status'] == 'supported'
    assert result['claims'][0]['minimal_supporting_sets'] == [['note', 'row']]
    assert all(row['status'] == 'insufficient_evidence' for row in result['claims'][0]['assertion_matrix'])
    assert result['factual_truth'] == 'not_assessed'
    assert result['source_interpretation_accuracy'] == 'not_assessed'
    assert result['decomposition_completeness'] == 'not_assessed'


@pytest.mark.parametrize('same_context', [True, False])
def test_bounded_witness_keeps_composite_support_in_large_citation_scope(same_context):
    from video_knowledge_pipeline.structured_claim_verification import select_bounded_claim_citations
    old_index, scope, claims, texts = fixture()
    refs = list(old_index.refs.values())
    refs += [{**old_index.refs['noise'], 'evidence_id': f'noise-{i}'} for i in range(20)]
    refs.append({'evidence_id': 'all', 'source_kind': 'evidence_group', 'fact_status': 'candidate_evidence',
                 'member_evidence_ids': [r['evidence_id'] for r in refs]})
    index = build_citation_index({'sections': [{'evidence_refs': refs}]})
    scope['reference_snapshot_sha256'] = index.snapshot_sha256
    if not same_context:
        scope['assertions'][1]['context_id'] = 'unrelated-page'
    claims[0]['evidence_ids'] = ['all']
    selected = select_bounded_claim_citations(claims[0], scope['assertions'], index)
    assert len(selected) <= 8
    if same_context:
        assert set(selected) == {'note', 'row'}
    claims[0]['evidence_ids'] = selected
    result = audit_structured_claims(claims, scope=scope, citation_index=index, output_texts=texts,
                                     output_citations={'overview.text': ['all']})
    assert result['valid'], result['errors']
    assert (result['claims'][0]['status'] == 'supported') is same_context


@pytest.mark.parametrize(('field', 'wrong'), [('subject', '产品乙'), ('version', '2025版'),
                                             ('time_scope', '合同生效前'), ('property', '保费')])
def test_other_product_version_property_time_cannot_supply_missing_condition(field, wrong):
    data = fixture()
    data[1]['assertions'][1]['slots'][field] = wrong
    result = audit(data)
    assert result['valid']
    assert result['claims'][0]['status'] == 'insufficient_evidence'


def test_unrelated_context_cannot_join_even_with_matching_slot_words():
    data = fixture()
    data[1]['assertions'][1]['context_id'] = 'other-page:unbound-footnote'
    assert audit(data)['claims'][0]['status'] == 'insufficient_evidence'


@pytest.mark.parametrize(('field', 'wrong', 'status'), [('value', '30', 'contradiction'),
    ('negated', True, 'contradiction'), ('conditions', [], 'insufficient_evidence'),
    ('unit', '元', 'contradiction')])
def test_claim_value_negation_scope_or_unit_is_not_substring_approved(field, wrong, status):
    data = fixture()
    data[2][0]['slots'][field] = wrong
    assert audit(data)['claims'][0]['status'] == status


def test_extra_irrelevant_reference_reduces_citation_precision():
    data = fixture()
    data[2][0]['evidence_ids'].append('noise')
    row = audit(data)['claims'][0]
    assert row['status'] == 'supported'
    assert row['irrelevant_citation_ids'] == ['noise']
    assert row['citation_precision'] == pytest.approx(2 / 3)


def test_omitted_counterevidence_is_not_hidden_by_model_citation_choices():
    data = fixture()
    opposing = deepcopy(data[1]['assertions'][0])
    opposing.update(assertion_id='opposing', evidence_id='noise', context_id='another-observation')
    ref = data[0].refs['noise']
    opposing.update(reference_canonical_sha256=digest(ref),
                    span={'start': 0, 'end': len(ref['snippet']), 'canonical_sha256': digest(ref['snippet'])})
    opposing['slots'] = {**slots(), 'value': '30'}
    data[1]['assertions'].append(opposing)
    result = audit(data)
    assert result['claims'][0]['status'] == 'contradiction'
    assert 'opposing' in result['claims'][0]['contradicting_assertion_ids']
    # The synthetic annotation is intentionally contradictory to its excerpt.
    # Identity validation cannot certify the interpretation; never claim it can.
    assert result['source_interpretation_accuracy'] == 'not_assessed'


@pytest.mark.parametrize('drift', ['source', 'span', 'snapshot', 'output'])
def test_hash_drift_is_all_or_nothing(drift):
    data = fixture()
    if drift == 'source':
        data[1]['assertions'][0]['reference_canonical_sha256'] = 'a' * 64
    elif drift == 'span':
        data[1]['assertions'][0]['span']['canonical_sha256'] = 'a' * 64
    elif drift == 'snapshot':
        data[1]['reference_snapshot_sha256'] = 'a' * 64
    else:
        data[2][0]['output_canonical_sha256'] = 'a' * 64
    result = audit(data)
    assert not result['valid']
    assert result['claims'] == []


@pytest.mark.parametrize('mutation', ['duplicate', 'missing_scope', 'extra_scope', 'unknown_ref',
                                    'negative_span', 'long_field', 'partial_quantity'])
def test_invalid_or_incomplete_protocol_fails_closed(mutation):
    data = fixture()
    if mutation == 'duplicate':
        data[2].append(deepcopy(data[2][0]))
    elif mutation == 'missing_scope':
        data[1]['required_locations'].append('themes[1].summary')
    elif mutation == 'extra_scope':
        data[2][0]['output_location'] = 'invented.text'
    elif mutation == 'unknown_ref':
        data[2][0]['evidence_ids'] = ['outside-file']
    elif mutation == 'negative_span':
        data[1]['assertions'][0]['span']['start'] = -1
    elif mutation == 'long_field':
        data[2][0]['slots']['subject'] = 'x' * 513
    else:
        del data[1]['assertions'][0]['slots']['unit']
    result = audit(data)
    assert not result['valid']
    assert result['claims'] == []


def test_existing_numeric_units_are_reused_but_unknown_units_not_discarded():
    data = fixture()
    data[1]['assertions'][0]['slots'].update(value='2', unit='万元')
    data[2][0]['slots'].update(value='20000', unit='元')
    assert audit(data)['claims'][0]['status'] == 'supported'
    data[1]['assertions'][0]['slots'].update(value='2', unit='未知单位甲')
    data[2][0]['slots'].update(value='2', unit='未知单位乙')
    assert audit(data)['claims'][0]['status'] != 'supported'


def test_deterministic_without_mutating_input():
    data = fixture()
    before = deepcopy(data[1:])
    assert audit(data) == audit(data)
    assert data[1:] == before


def test_negating_another_value_is_not_a_direct_contradiction():
    data = fixture()
    data[1]['assertions'][0]['slots'].update(value='30', negated=True)
    assert audit(data)['claims'][0]['status'] == 'insufficient_evidence'


@pytest.mark.parametrize(('source', 'target'), [
    ('20000000000000001', '20000000000000000'), ('0.000000001', '0.000000002')])
def test_quantity_comparison_cannot_approve_float_rounding_collisions(source, target):
    data = fixture()
    data[1]['assertions'][0]['slots'].update(value=source, unit='元')
    data[2][0]['slots'].update(value=target, unit='元')
    assert audit(data)['claims'][0]['status'] == 'contradiction'


def reader_fixture():
    from test_smart_summary_reader_plan import _fact_pack, _valid_plan
    index, scope, claims, texts = fixture()
    pack, plan = _fact_pack(), _valid_plan()
    pack['sections'][0]['evidence_refs'] += list(index.refs.values())
    scope['reference_snapshot_sha256'] = build_citation_index(pack).snapshot_sha256
    pack['claim_verification'] = scope
    plan['overview'] = {'text': texts['overview.text'], 'evidence_ids': ['row', 'note', 'noise']}
    plan['structured_claims'] = claims
    return pack, plan


def validate_reader(pack, plan):
    from video_knowledge_pipeline.smart_summary_reader_plan import validate_reader_plan
    return validate_reader_plan(plan, fact_pack=pack, expected_section_ids=set(plan['source_section_ids']))


def test_structured_scope_is_enforced_by_actual_reader_gate():
    pack, plan = reader_fixture()
    result = validate_reader(pack, plan)
    assert result['passed'], result['errors']
    assert result['structured_claim_audit']['counts']['supported'] == 1
    assert 'core_insights[1].text' in result['structured_claim_audit']['unassessed_output_locations']
    assert result['semantic_support'] == 'not_assessed'  # scoped declarations != entire-summary truth
    plan['structured_claims'][0]['slots']['conditions'] = []
    result = validate_reader(pack, plan)
    assert not result['passed']
    assert any('structured_claim_not_supported:' in error for error in result['errors'])


def test_model_cannot_omit_required_review_annotations_or_forge_scope():
    pack, plan = reader_fixture()
    del plan['structured_claims']
    assert not validate_reader(pack, plan)['passed']
    pack, plan = reader_fixture()
    del pack['claim_verification']
    assert not validate_reader(pack, plan)['passed']


def test_reduce_refuses_invalid_structured_claim_after_normalization():
    from video_knowledge_pipeline.smart_summary_global_reduce import _validated_reader_plan_content
    pack, plan = reader_fixture()
    plan['structured_claims'][0]['slots']['value'] = '30'
    before = deepcopy((pack, plan))
    _, validation, content = _validated_reader_plan_content(json.dumps(plan, ensure_ascii=False),
        fact_pack=pack, expected_ids=set(plan['source_section_ids']), title='合成课件',
        first_time='00:00:00.000', last_time='00:04:00.000')
    assert not validation['passed']
    assert 'structured_claim_audit' in validation
    assert content == ''
    assert (pack, plan) == before


def test_no_source_interpretations_is_an_explicit_evidence_gap():
    data = fixture()
    data[1]['assertions'] = []
    result = audit(data)
    assert result['valid'], result['errors']
    assert result['claims'][0]['status'] == 'insufficient_evidence'


def test_review_only_source_is_not_admitted_as_an_assertion():
    data = fixture()
    ref = data[0].refs['note']
    ref['fact_status'] = 'review_gap_not_fact'
    index = build_citation_index({'sections': [{'evidence_refs': list(data[0].refs.values())}]})
    data = (index, *data[1:])
    data[1]['reference_snapshot_sha256'] = index.snapshot_sha256
    data[1]['assertions'][1]['reference_canonical_sha256'] = digest(ref)
    result = audit(data)
    assert not result['valid']
    assert result['claims'] == []


def test_unused_source_drift_rejects_entire_batch_before_verdicts():
    data = fixture()
    unused = deepcopy(data[1]['assertions'][0])
    unused.update(assertion_id='unused', evidence_id='noise')
    data[1]['assertions'].append(unused)
    result = audit(data)
    assert not result['valid']
    assert result['claims'] == []


def test_internal_source_disagreement_cannot_be_voted_away():
    data = fixture()
    alternate = deepcopy(data[1]['assertions'][0])
    alternate['assertion_id'] = 'different-observation-same-source'
    alternate['slots']['value'] = '30'
    data[1]['assertions'].append(alternate)
    result = audit(data)
    assert result['claims'][0]['status'] != 'supported'
    assert result['claims'][0]['conflicted_context_ids']


def test_different_source_product_does_not_veto_correct_scoped_claim():
    data = fixture()
    alternate = deepcopy(data[1]['assertions'][0])
    alternate['assertion_id'] = 'other-product'
    alternate['slots'] = {**slots(), 'subject': '产品乙', 'value': '30'}
    data[1]['assertions'].append(alternate)
    assert audit(data)['claims'][0]['status'] == 'supported'


def test_duplicate_claims_cannot_inflate_the_count_using_new_ids():
    data = fixture()
    duplicate = deepcopy(data[2][0])
    duplicate['claim_id'] = 'another-id-same-claim'
    data[2].append(duplicate)
    result = audit(data)
    assert not result['valid']
    assert result['claims'] == []


@pytest.mark.parametrize(('field', 'value'), [('negated', 0), ('conditions', '首次投保')])
def test_field_type_coercion_is_forbidden(field, value):
    data = fixture()
    data[2][0]['slots'][field] = value
    assert not audit(data)['valid']
