from copy import deepcopy

import pytest

from video_knowledge_pipeline.transcript_semantic_correction import _source_support_summary


def row(name, kind, text='90 days', **extra):
    return dict(evidence_id=name, source_type=kind, text=text, **extra)


def test_same_frame_different_recognizers_do_not_accumulate_weight():
    rows = [row('ocr', 'ocr', frame_sha256='a'*64),
            row('vlm', 'visual_understanding', frame_sha256='a'*64)]
    before = deepcopy(rows)
    out = _source_support_summary('30 days', '90 days', rows)
    assert out['candidate_weight'] == 70
    assert out['lineage']['correlated_group_count'] == 1
    assert out['lineage']['independent_source_count'] == 0
    assert rows == before


def test_original_source_binds_different_frame_hashes():
    rows = [row('one', 'ocr', source_media_sha256='a'*64, frame_sha256='b'*64),
            row('two', 'visual_understanding', source_media_sha256='a'*64, frame_sha256='c'*64)]
    assert _source_support_summary('30 days', '90 days', rows)['candidate_weight'] == 70


def test_distinct_declared_roots_are_groups_not_verified_independence():
    rows = [row('one', 'ocr', original_source_sha256='a'*64),
            row('two', 'ocr', original_source_sha256='b'*64)]
    out = _source_support_summary('30 days', '90 days', rows)
    assert out['candidate_weight'] == 140
    assert out['lineage']['declared_source_group_count'] == 2
    assert out['lineage']['independent_source_count'] == 0
    assert out['lineage']['independence_is_verified'] is False


def test_model_that_consumes_other_evidence_is_not_independent():
    rows = [row('ocr', 'ocr', original_source_sha256='a'*64),
            row('vlm', 'visual_understanding', original_source_sha256='b'*64,
                derived_from_evidence_ids=['ocr'])]
    out = _source_support_summary('30 days', '90 days', rows)
    assert out['candidate_weight'] == 70
    assert out['lineage']['declared_source_group_count'] == 1
    assert out['lineage']['independent_source_count'] == 0


def test_unknown_lineage_is_not_two_independent_sources():
    out = _source_support_summary('30 days', '90 days', [row('one','ocr'),row('two','visual_understanding')])
    assert out['candidate_weight'] == 70
    assert out['lineage']['independent_source_count'] == 0
    assert out['lineage']['unverified_evidence_count'] == 2


def test_correlated_disagreement_is_preserved_not_majority_washed():
    rows = [row('a','ocr',frame_sha256='a'*64),
            row('b','visual_understanding','30 days',frame_sha256='a'*64),
            row('c','ocr',frame_sha256='a'*64)]
    out = _source_support_summary('30 days', '90 days', rows)
    assert out['candidate_weight'] == 70
    assert out['original_weight'] == 65
    assert out['needs_review_by_source_vote']
    assert out['lineage']['intra_group_conflict_count'] == 1


@pytest.mark.parametrize('reference', [['missing'], ['b']])
def test_unresolved_or_cyclic_parent_is_unverified(reference):
    rows=[row('a','ocr',original_source_sha256='a'*64,derived_from_evidence_ids=reference),
          row('b','visual_understanding',original_source_sha256='b'*64,derived_from_evidence_ids=['a'])]
    out=_source_support_summary('30 days','90 days',rows)
    assert out['lineage']['independent_source_count'] == 0
    assert out['needs_review_by_source_vote']


def test_malformed_hash_never_proves_independence():
    out=_source_support_summary('30 days','90 days',[row('a','ocr',original_source_sha256='bad')])
    assert out['lineage']['independent_source_count'] == 0


def test_multiple_disjoint_cycles_are_all_unverified():
    rows = [row(name, 'ocr', original_source_sha256=digest*64,
                derived_from_evidence_ids=[parent])
            for name, parent, digest in [('a','b','a'),('b','a','b'),
                                         ('c','d','c'),('d','c','d')]]
    lineage = _source_support_summary('30 days', '90 days', rows)['lineage']
    assert lineage['independent_source_count'] == 0
    assert lineage['unresolved_group_count'] == 2


def test_order_invariant_group_identity():
    rows=[row('a','ocr',frame_sha256='a'*64),row('b','visual_understanding',frame_sha256='a'*64)]
    a=_source_support_summary('30 days','90 days',rows)['lineage']
    b=_source_support_summary('30 days','90 days',list(reversed(rows)))['lineage']
    assert a == b


def test_primary_recognition_disagreement_is_not_direct_evidence_conflict():
    out = _source_support_summary('fee', 'premium', [
        row('primary', 'asr_or_subtitle', 'fee'),
        row('one', 'secondary_asr', 'premium', artifact_sha256='a'*64),
        row('two', 'secondary_asr', 'premium', artifact_sha256='b'*64),
    ])
    assert out['lineage']['intra_group_conflict_count'] == 1
    assert out['lineage']['intra_group_direct_conflict_count'] == 0
    assert out['lineage']['independent_source_count'] == 0


def test_validator_recomputes_lineage_and_preserves_omitted_counter_evidence():
    from video_knowledge_pipeline.transcript_semantic_correction import _validate_decision

    evidence = [row('ocr', 'ocr', 'premium', frame_sha256='a'*64),
                row('vlm', 'visual_understanding', 'fee', frame_sha256='a'*64)]
    candidate = dict(candidate_id='one', original_text='fee', candidate_text='premium',
                     evidence=evidence, evidence_ids=['ocr', 'vlm'],
                     source_support_summary={'needs_review_by_source_vote': False})
    decision = dict(candidate_id='one', original_text='fee', corrected_text='premium',
                    evidence_ids=['ocr'], correction_type='ordinary_word',
                    confidence=0.99, semantic_rationale='synthetic hypothesis')
    out = _validate_decision(decision, {'one': candidate}, idx=0, min_confidence=0.88)
    assert not out['accepted']
    assert 'correlated_or_unresolved_evidence_requires_review' in out['reject_reasons']
    reviewed = _validate_decision(dict(decision, human_confirmed=True), {'one': candidate}, idx=0, min_confidence=0.88)
    assert reviewed['accepted']


@pytest.mark.parametrize('overrides', [
    {'provider': 'backend-a'},
    {'artifact_sha256': 'a'*64},
    {'artifact_sha256': ''},
    {'derived_from_evidence_ids': ['one']},
])
def test_duplicate_or_unverified_secondary_recognition_is_not_independent(overrides):
    from video_knowledge_pipeline.transcript_semantic_correction import _has_two_independent_secondary_asr_evidence

    first = row('one', 'secondary_asr', provider='backend-a', artifact_sha256='a'*64)
    second = row('two', 'secondary_asr', provider='backend-b', artifact_sha256='b'*64)
    candidate = {'evidence': [first, second]}
    assert _has_two_independent_secondary_asr_evidence(candidate, ['one', 'two'], corrected_text='90 days')
    second.update(overrides)
    assert not _has_two_independent_secondary_asr_evidence(candidate, ['one', 'two'], corrected_text='90 days')
