"""Reuse evidence groups and source partitions, not a second object encoding."""
from copy import deepcopy

import pytest

from video_knowledge_pipeline import summary_claim_review as review
from video_knowledge_pipeline.canonical_json import canonical_json_bytes
from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.evidence_locator import JsonEvidenceSnapshot
from video_knowledge_pipeline.video_moment_index import _chunk_from_items
from video_knowledge_pipeline.smart_summary_global_reduce import _citation_evidence_refs


@pytest.mark.parametrize('changed', [False, True])
def test_tag_frame_factoring_preserves_every_original_value_and_is_idempotent(tmp_path, changed):
    observations = [dict(frame_id=f'f{i}', timestamp=i * 0.5, tags=['课件'], confidence=0.9,
        frame_path='synthetic/' + 'nested/' * 15 + f'frame{i}.png', frame_sha256='a' * 64,
        executor='synthetic', model_actual='synthetic-tags', model_revision='fixture',
        run_id='synthetic-run', ocr_text='', boundary_before=False, preserve_transient=False,
        tag_confidences={}) for i in range(8)]
    value = dict(schema='video_knowledge_pipeline.temporal_tag_delta.v2',
                 raw_observations=observations, smoothed_observations=deepcopy(observations),
                 evidence_level='candidate_weak', coarse_summary='画面标签显示可能有课件。')
    if changed:
        value['smoothed_observations'][3]['tags'] = ['课件', '临时标签']
    value.update(decision='coarse_weak_evidence', stable_tags=['课件'])
    timeline = [{'index': i+1, 'start':i, 'end':i+1, 'text':f'第{i}段合成原文。'*12} for i in range(8)]
    timeline[0]['temporal_tag_delta'] = value
    write_json(tmp_path/'timeline.json', timeline)
    before = (tmp_path/'timeline.json').read_bytes()
    snap = JsonEvidenceSnapshot.read(tmp_path, 'timeline.json')
    timeline = snap.data
    chunk = _chunk_from_items(timeline, 1, snapshot=snap, source_positions={id(row): i for i,row in enumerate(timeline)})
    refs = _citation_evidence_refs(dict(citation_id='moment-1', source_kind='asr',
        snippet=chunk['transcript_text'], visual_snippet=chunk['visual_text'],
        source_locators_by_kind=chunk['source_locators_by_kind'],
        source_observations_by_kind=chunk['source_observations_by_kind']))
    facts = {'sections':[{'section_id':'s','evidence_refs':refs}]}
    plan = {'overview':{'text':timeline[0]['text'],'evidence_ids':['moment-1']}}
    binding,index,texts,cites = review._view(plan, facts)
    assert index.refs['moment-1']['source_kind'] == 'evidence_group'
    sources = review._scope_sources(tmp_path, index, facts, ['moment-1'])
    frame_sources = [s for s in sources if s['source_context'].get('tag_observation_array')]
    assert len(frame_sources) == 16
    for source in frame_sources:
        loc = source['source_locators'][0]
        assert source['original_objects'] == [snap.selected(loc['selector'])]
        assert source['evidence_level'] == 'candidate_weak'
        assert source['recognition_provenance']['selected_field'] == loc['selector'][1:]
        assert source['recognition_provenance']['observation_text_sha256'] == review.digest(source['snippet'])
    batches = review._source_batches(binding, 'overview.text', texts['overview.text'], sources, ['moment-1'], {})
    assert len(batches) > 1
    for batch in batches:
        pack = review._task_pack(binding, 'overview.text', texts['overview.text'], batch, ['moment-1'], {}, {})
        assert len(canonical_json_bytes(pack)) <= 8192
        assert all('original_objects_encoding' not in s for s in pack['sources'])
    assert {s['evidence_id'] for batch in batches for s in batch} == {s['evidence_id'] for s in sources}
    assert (tmp_path/'timeline.json').read_bytes() == before
