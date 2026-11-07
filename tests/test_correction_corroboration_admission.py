"""Synthetic checks at the real correction validator, not vote-report only."""
from copy import deepcopy

import pytest

from video_knowledge_pipeline.transcript_semantic_correction import _validate_decision
from test_transcript_semantic_safety_and_discovery import _candidate, _decision


def case(kind, identity):
    original, corrected = (('中医人寿', '中意人寿') if kind == 'organization'
                           else ('行使权力', '行使权利'))
    evidence = [dict(evidence_id='a', source_type='ocr', text=corrected,
                     evidence_level='confirmed'),
                dict(evidence_id='b', source_type='structured_visual', text=corrected,
                     evidence_level='confirmed')]
    if identity == 'same_frame':
        for row in evidence:
            row['frame_sha256'] = 'a' * 64
    elif identity == 'shared_original':
        for i, row in enumerate(evidence):
            row.update(source_media_sha256='a' * 64, artifact_sha256=str(i + 1) * 64)
    elif identity in {'distinct_roots', 'derived', 'weak'}:
        for i, row in enumerate(evidence):
            row['original_source_sha256'] = str(i + 1) * 64
        if identity == 'derived':
            evidence[1]['derived_from_evidence_ids'] = ['a']
        if identity == 'weak':
            evidence[1]['evidence_level'] = 'candidate_evidence'
    return _candidate('test', kind, original, corrected, evidence=evidence)


def validate(candidate, *, human=False):
    return _validate_decision(_decision(candidate, human_confirmed=human),
                              {'test': candidate}, idx=0, min_confidence=0.88,
                              strict_model_contract=True)


@pytest.mark.parametrize('kind', ['organization', 'ordinary_homophone'])
@pytest.mark.parametrize('identity', ['same_frame', 'shared_original', 'unknown', 'derived'])
def test_two_types_cannot_bypass_corroboration_gate(kind, identity):
    candidate = case(kind, identity)
    before = deepcopy(candidate)
    result = validate(candidate)
    assert result['accepted'] is False
    assert any('two_independent_direct_evidence' in e for e in result['reject_reasons'])
    assert candidate == before
    assert validate(candidate, human=True)['accepted'] is True


@pytest.mark.parametrize('kind', ['organization', 'ordinary_homophone'])
def test_distinct_declared_roots_preserve_scoped_correction_not_factual_truth(kind):
    result = validate(case(kind, 'distinct_roots'))
    assert result['accepted'] is True
    key = 'semantic_anchor_corroboration' if kind == 'organization' else 'linguistic_corroboration'
    support = result[key]
    assert support['declared_support_group_count'] == 2
    assert support['lineage']['independence_is_verified'] is False
    assert support['lineage']['independent_source_count'] == 0


@pytest.mark.parametrize('kind', ['organization', 'ordinary_homophone'])
def test_unconfirmed_observation_cannot_supply_second_strong_support(kind):
    assert validate(case(kind, 'weak'))['accepted'] is False


@pytest.mark.parametrize('kind', ['organization', 'ordinary_homophone'])
def test_unselected_parent_aliases_still_join_supporting_roots(kind):
    candidate = case(kind, 'distinct_roots')
    candidate['evidence'].extend([
        dict(evidence_id='bridge', source_type='tagger', text='',
             original_source_sha256='1' * 64, artifact_sha256='2' * 64),
    ])
    assert candidate['evidence_ids'] == ['a', 'b']
    assert validate(candidate)['accepted'] is False


@pytest.mark.parametrize('kind', ['organization', 'ordinary_homophone'])
def test_added_lineage_veto_does_not_relax_old_source_diversity_gate(kind):
    candidate = case(kind, 'distinct_roots')
    candidate['evidence'][1]['source_type'] = 'ocr'
    assert validate(candidate)['accepted'] is False
