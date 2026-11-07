"""Synthetic citation admission tests; no model, media or file writes."""
from copy import deepcopy
import json

import pytest

from test_smart_summary_reader_plan import _fact_pack, _valid_plan
from video_knowledge_pipeline.smart_summary_reader_plan import (
    normalize_reader_plan_candidate, validate_reader_plan,
)
from video_knowledge_pipeline.smart_summary_global_reduce import _validated_reader_plan_content


def _validate(pack, plan=None):
    plan = plan or _valid_plan()
    return validate_reader_plan(plan, fact_pack=pack,
                                expected_section_ids=set(plan['source_section_ids']))


def _group(name, members):
    return {'evidence_id': name, 'source_kind': 'evidence_group',
            'fact_status': 'candidate_evidence', 'member_evidence_ids': members,
            'snippet': ''}


@pytest.mark.parametrize('members', [[], ['unknown'], ['grp'], 'ev-0001', ['ev-0001', 'ev-0001']])
def test_invalid_group_cannot_make_citations_eligible(members):
    pack, plan = _fact_pack(), _valid_plan()
    pack['sections'][0]['evidence_refs'].append(_group('grp', members))
    plan['core_insights'][0]['evidence_ids'] = ['grp']
    result = _validate(pack, plan)
    assert not result['passed']
    assert any('evidence_index:' in error for error in result['errors'])


def test_disconnected_cycle_is_refused_even_when_not_cited():
    pack = _fact_pack()
    pack['sections'][0]['evidence_refs'] += [_group('a', ['b']), _group('b', ['a'])]
    assert not _validate(pack)['passed']


def test_group_cannot_promote_review_only_leaf():
    pack, plan = _fact_pack(), _valid_plan()
    pack['sections'][0]['evidence_refs'].append(_group('grp', ['gap-0001']))
    plan['core_insights'][0]['evidence_ids'] = ['grp']
    result = _validate(pack, plan)
    assert not result['passed']
    assert any('review_only_evidence_promoted' in error for error in result['errors'])


def test_nested_group_expands_for_source_quote_but_not_factual_entailment():
    pack, plan = _fact_pack(), _valid_plan()
    pack['sections'][0]['evidence_refs'] += [
        _group('inner', ['ev-0001']), _group('outer', ['inner', 'ev-0002'])]
    plan['reusable_expressions'][0]['evidence_ids'] = ['outer']
    result = _validate(pack, plan)
    assert result['passed'], result['errors']
    audit = result['citation_audit']
    assert audit['semantic_support'] == 'not_assessed'
    quote = next(row for row in audit['items'] if row['location'] == 'reusable_expressions[1]')
    assert quote['leaf_evidence_ids'] == ['ev-0001', 'ev-0002']
    assert quote['quote_matching_evidence_ids'] == ['ev-0001']
    assert quote['quote_citation_precision'] == 0.5
    assert quote['source_freshness'] == 'not_assessed'


@pytest.mark.parametrize('new_text', ['不同的来源被同一个编号覆盖', None])
def test_conflicting_duplicate_evidence_id_is_not_last_writer_wins(new_text):
    pack = _fact_pack()
    other = deepcopy(pack['sections'][0]['evidence_refs'][0])
    other['snippet'] = new_text
    pack['sections'][1]['evidence_refs'].append(other)
    result = _validate(pack)
    assert not result['passed']


def test_identical_reference_reused_across_chapters_is_not_a_conflict():
    pack = _fact_pack()
    pack['sections'][1]['evidence_refs'].append(deepcopy(pack['sections'][0]['evidence_refs'][0]))
    result = _validate(pack)
    assert result['passed'], result['errors']
    assert result['citation_audit']['unique_leaf_count'] == 5


def test_quote_cannot_be_stitched_from_unrelated_sources():
    pack, plan = _fact_pack(), _valid_plan()
    pack['sections'][0]['evidence_refs'][0]['snippet'] = '开头文字。应当保留'
    pack['sections'][1]['evidence_refs'][0]['snippet'] = '全部个人资料。后面文字'
    plan['reusable_expressions'][0].update(text='应当保留全部个人资料', evidence_ids=['ev-0001', 'ev-0002'])
    result = _validate(pack, plan)
    assert not result['passed']
    assert any('verbatim_quote_not_found' in error for error in result['errors'])
    normalized = normalize_reader_plan_candidate(plan, fact_pack=pack)
    assert normalized['plan']['reusable_expressions'][0]['kind'] == 'reusable_expression'
    assert plan['reusable_expressions'][0]['kind'] == 'verbatim_quote'


@pytest.mark.parametrize(('source', 'quote'), [
    ('比率为1.5%，请注意范围。', '比率为15%，请注意范围。'),
    ('温度为-10度，请注意范围。', '温度为10度，请注意范围。'),
    ('选择A/B方案，请注意范围。', '选择AB方案，请注意范围。'),
])
def test_quote_matching_does_not_erase_meaningful_punctuation(source, quote):
    pack, plan = _fact_pack(), _valid_plan()
    pack['sections'][0]['evidence_refs'][0]['snippet'] = source
    plan['reusable_expressions'][0]['text'] = quote
    assert not _validate(pack, plan)['passed']


def test_quote_matching_accepts_layout_whitespace_not_cross_source_join():
    pack, plan = _fact_pack(), _valid_plan()
    pack['sections'][0]['evidence_refs'][0]['snippet'] = '先确认客户真正想解决的问题，\n再选择合适的方法。'
    assert _validate(pack, plan)['passed']


def test_invalid_source_index_blocks_real_reduce_render_after_normalization():
    pack, plan = _fact_pack(), _valid_plan()
    pack['sections'][0]['evidence_refs'].append(_group('bad', ['absent']))
    before = deepcopy((pack, plan))
    _, validation, content = _validated_reader_plan_content(
        json.dumps(plan, ensure_ascii=False), fact_pack=pack,
        expected_ids=set(plan['source_section_ids']), title='合成验收',
        first_time='00:00:00.000', last_time='00:04:00.000')
    assert not validation['passed']
    assert content == ''
    assert (pack, plan) == before


def test_legacy_success_is_not_presented_as_claim_truth():
    result = _validate(_fact_pack())
    assert result['passed']
    assert result['citation_audit']['assessment_scope'] == 'reference_integrity_not_entailment'
    assert result['citation_audit']['semantic_support'] == 'not_assessed'
    assert result['citation_audit']['factual_truth'] == 'not_assessed'


def test_forged_group_snippet_cannot_supply_a_quote():
    pack, plan = _fact_pack(), _valid_plan()
    group = _group('grp', ['ev-0002'])
    group['snippet'] = plan['reusable_expressions'][0]['text']
    pack['sections'][0]['evidence_refs'].append(group)
    plan['reusable_expressions'][0]['evidence_ids'] = ['grp']
    assert not _validate(pack, plan)['passed']


def test_nested_review_group_cannot_launder_an_eligible_leaf_quote():
    pack, plan = _fact_pack(), _valid_plan()
    hidden = _group('review-container', ['ev-0001'])
    hidden['fact_status'] = 'review_gap_not_fact'
    pack['sections'][0]['evidence_refs'] += [hidden, _group('mixed', ['review-container', 'ev-0002'])]
    plan['reusable_expressions'][0]['evidence_ids'] = ['mixed']
    assert not _validate(pack, plan)['passed']


@pytest.mark.parametrize(('source', 'quote'), [
    ('The label says now here.', 'The label says nowhere.'),
    ('编号是1 5，请分别记录。', '编号是15，请分别记录。'),
])
def test_layout_normalization_does_not_join_distinct_words_or_numbers(source, quote):
    pack, plan = _fact_pack(), _valid_plan()
    pack['sections'][0]['evidence_refs'][0]['snippet'] = source
    plan['reusable_expressions'][0]['text'] = quote
    assert not _validate(pack, plan)['passed']


def test_snapshot_hash_changes_without_claiming_underlying_file_freshness():
    pack = _fact_pack()
    first = _validate(pack)['citation_audit']
    pack['sections'][1]['evidence_refs'][0]['snippet'] += '新增加的限定条件。'
    second = _validate(pack)['citation_audit']
    assert first['reference_snapshot_sha256'] != second['reference_snapshot_sha256']
    assert len(first['reference_snapshot_sha256']) == 64
    assert second['semantic_support'] == 'not_assessed'


def test_references_and_group_expansion_are_bounded(monkeypatch):
    import video_knowledge_pipeline.summary_citation_audit as audit
    pack = _fact_pack()
    monkeypatch.setattr(audit, 'MAX_REFS', 4)
    assert not _validate(pack)['passed']
    monkeypatch.setattr(audit, 'MAX_REFS', 4096)
    monkeypatch.setattr(audit, 'MAX_GROUP_DEPTH', 2)
    pack['sections'][0]['evidence_refs'] += [_group('a', ['ev-0001']), _group('b', ['a'])]
    assert not _validate(pack)['passed']


def test_group_alias_is_not_another_source_observation():
    pack, plan = _fact_pack(), _valid_plan()
    pack['sections'][0]['evidence_refs'] += [_group('grp', ['ev-0001'])]
    plan['core_insights'][0]['evidence_ids'] = ['ev-0001', 'grp']
    result = _validate(pack, plan)
    assert result['passed']
    row = next(row for row in result['citation_audit']['items'] if row['location'] == 'core_insights[1]')
    assert row['leaf_evidence_ids'] == ['ev-0001']
    assert row['redundant_reference_path_count'] == 1


def test_model_reference_list_and_id_lengths_are_bounded_before_audit():
    plan = _valid_plan()
    plan['overview']['evidence_ids'] = ['ref-' + str(i) for i in range(257)]
    result = _validate(_fact_pack(), plan)
    assert not result['passed']
    assert any(error.startswith('schema:overview.evidence_ids:') for error in result['errors'])
    assert result['semantic_support'] == 'not_assessed'
    plan['overview']['evidence_ids'] = ['x' * 513]
    result = _validate(_fact_pack(), plan)
    assert any(error.startswith('schema:overview.evidence_ids.0:') for error in result['errors'])
