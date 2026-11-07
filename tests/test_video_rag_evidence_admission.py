"""Offline evidence admission through the public VideoRAG projection."""
import json

import pytest

from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.video_rag_pack import build_video_rag_pack


def bundle(root, *, level='confirmed', human=False, invalid=False):
    row = {'index': 7, 'start': 0, 'end': 2, 'transcript': '合成课程说明',
           'frame_path': 'assets/synthetic.jpg', 'visual_text': '旧机器文字',
           'structured_visual': {'markdown': '独有表格条件'},
           'visual_understanding': {'objects': ['模型看到的对象'], 'evidence_level': level,
               'unavailable_reason': '合成遮挡', 'inference_basis': '合成歧义',
               'recognition_provenance': {'frame_sha256s': ['bad' if invalid else 'a'*64],
                                          'executor': 'synthetic', 'model_actual': 'reported-model'}}}
    if human:
        row['human_corrected_visual_text'] = '人工确认文字'
        row['human_corrected_visual_understanding'] = {'objects': ['人工确认对象']}
    write_json(root / 'manifest.json', {'title': 'synthetic rag'})
    write_json(root / 'timeline.json', [None, row])
    return row


def visual_units(result):
    return [r for r in result['retrieval_units'] if r['metadata']['chunk_kind'] == 'visual_evidence']


def test_public_rag_respects_human_replacement(tmp_path):
    bundle(tmp_path, human=True)
    original = (tmp_path / 'timeline.json').read_bytes()
    result = build_video_rag_pack(tmp_path, write=False)
    text = ' '.join(r['text'] for r in visual_units(result))
    assert '人工确认文字' in text and '人工确认对象' in text
    assert '旧机器文字' not in text and '独有表格条件' not in text
    assert '模型看到的对象' not in text
    assert (tmp_path / 'timeline.json').read_bytes() == original


def test_public_rag_retains_qualified_constituents_and_exact_positions(tmp_path):
    bundle(tmp_path, level='inferred')
    result = build_video_rag_pack(tmp_path, write=False)
    row = visual_units(result)[0]
    assert '推断' in row['text'] and '可能' in row['text']
    assert '独有表格条件' in row['text']
    refs = row['metadata']['source_observations']
    assert {tuple(r['source_locators'][0]['selector']) for r in refs} == {
        (1, 'visual_text'), (1, 'structured_visual'), (1, 'visual_understanding')}
    assert any(r['evidence_level'] == 'inferred' for r in refs)
    moment = next(r for r in result['retrieval_units'] if r['metadata']['chunk_kind'] == 'moment')
    observations = moment['metadata']['source_observations']
    assert {r['evidence_id'] for r in observations if r['source_kind'] == 'visual'} == {r['evidence_id'] for r in refs}
    asr = [r for r in observations if r['source_kind'] == 'asr']
    assert len(asr) == 1 and asr[0]['source_locators'][0]['selector'] == [1, 'transcript']
    assert asr[0]['snippet'] == '合成课程说明' and asr[0]['source_excerpt_truncated'] is False


def test_unavailable_payload_is_review_gap_not_retrieved_visual_fact(tmp_path):
    bundle(tmp_path, level='unavailable')
    result = build_video_rag_pack(tmp_path, write=False)
    assert all('模型看到的对象' not in r['text'] for r in visual_units(result))
    gaps = [r for r in result['retrieval_units'] if r['metadata']['chunk_kind'] == 'review_gap']
    assert any(r['metadata'].get('source_observations') for r in gaps)
    assert all(r['fact_status'] == 'review_gap_not_fact' for gap in gaps
               for r in gap['metadata'].get('source_observations', []))


def test_invalid_source_record_rejected_before_final_rag_install(tmp_path):
    bundle(tmp_path, invalid=True)
    with pytest.raises(ValueError, match='recognition'):
        build_video_rag_pack(tmp_path, write=False)
    assert not (tmp_path / 'exports/video-rag-pack.json').exists()


def test_timeline_drift_between_moment_and_supplemental_rejected(tmp_path, monkeypatch):
    from video_knowledge_pipeline import video_rag_pack as module
    bundle(tmp_path)
    original = module._supplemental_chunks
    def drift(*args, **kwargs):
        rows = original(*args, **kwargs)
        path = tmp_path / 'timeline.json'
        path.write_bytes(path.read_bytes() + b'\n')
        return rows
    monkeypatch.setattr(module, '_supplemental_chunks', drift)
    with pytest.raises(ValueError, match='drift'):
        build_video_rag_pack(tmp_path, write=False)


def test_duplicate_timeline_indexes_keep_distinct_retrieval_ids(tmp_path):
    row = bundle(tmp_path)
    write_json(tmp_path / 'timeline.json', [None, row, dict(row, start=3, end=4)])
    result = build_video_rag_pack(tmp_path, write=False)
    rows = visual_units(result)
    assert len(rows) == 2
    assert len({r['id'] for r in rows}) == 2


def test_actual_human_tile_projection_is_retained(tmp_path):
    from video_knowledge_pipeline.review_session import _apply_non_destructive_corrections
    row = bundle(tmp_path)
    _apply_non_destructive_corrections(row, {'tile_corrections': [
        {'tile_id': 't1', 'status': 'corrected', 'corrected_text': '人工局部限定条件'}]})
    write_json(tmp_path / 'timeline.json', [row])
    result = build_video_rag_pack(tmp_path, write=False)
    assert any('人工局部限定条件' in r['text'] for r in visual_units(result))


@pytest.mark.parametrize('backend', ['keyword', 'sqlite'])
def test_actual_search_retains_observations_and_rejects_stale_source(tmp_path, backend):
    from video_knowledge_pipeline.video_rag_search import search_video_rag
    bundle(tmp_path, level='inferred')
    build_video_rag_pack(tmp_path, write=True)
    result = search_video_rag(tmp_path, query='模型', ensure_pack=False,
                              retrieval_backend=backend, write=False)
    hit = next(r for r in result['hits'] if r['chunk_kind'] == 'visual_evidence')
    assert any(r['evidence_level'] == 'inferred' for r in hit['source_observations'])
    assert hit['evidence_scope'] == 'derived_context_not_independent_source'
    path = tmp_path / 'timeline.json'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='drift'):
        search_video_rag(tmp_path, query='模型', ensure_pack=False,
                         retrieval_backend=backend, write=False)


def test_search_rejects_changed_rendered_text(tmp_path):
    from video_knowledge_pipeline.video_rag_search import search_video_rag
    bundle(tmp_path)
    build_video_rag_pack(tmp_path, write=True)
    path = tmp_path / 'exports/video-rag-chunks.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    rows[0]['text'] = '被替换的结论'
    path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows), encoding='utf-8')
    with pytest.raises(ValueError, match='text.*drift'):
        search_video_rag(tmp_path, query='结论', ensure_pack=False, write=False)


def test_truncated_observation_is_not_rendered_as_visual_fact_in_moment(tmp_path):
    row = bundle(tmp_path)
    row['visual_understanding']['objects'] = ['超长未审核观察' * 2000]
    write_json(tmp_path / 'timeline.json', [row])
    result = build_video_rag_pack(tmp_path, write=False)
    assert all('超长未审核观察' not in r['text'] for r in result['retrieval_units']
               if r['metadata']['chunk_kind'] in {'moment', 'visual_evidence'})
    assert any(ref['source_excerpt_truncated'] for r in result['retrieval_units']
               for ref in r['metadata'].get('source_observations', []))


def test_legacy_search_stays_readable_but_not_verified(tmp_path):
    from video_knowledge_pipeline.video_rag_search import search_video_rag
    bundle(tmp_path)
    path = tmp_path / 'exports/video-rag-chunks.jsonl'
    path.parent.mkdir()
    path.write_text(json.dumps({'id': 'legacy', 'text': 'legacy observation', 'metadata': {}}), encoding='utf-8')
    result = search_video_rag(tmp_path, query='legacy', ensure_pack=False, write=False)
    assert result['hits'][0]['evidence_scope'] == 'legacy_or_other_derived_not_assessed'
    assert result['observation_admission']['status'] == 'not_assessed'
    assert result['observation_admission']['independent_source_count'] == 0


def test_new_projection_cannot_drop_its_required_observation_list(tmp_path):
    from video_knowledge_pipeline.video_rag_pack import _validate_observation_sources
    bundle(tmp_path)
    result = build_video_rag_pack(tmp_path, write=False)
    row = visual_units(result)[0]
    del row['metadata']['source_observations']
    with pytest.raises(ValueError, match='schema'):
        _validate_observation_sources(tmp_path, [row])
