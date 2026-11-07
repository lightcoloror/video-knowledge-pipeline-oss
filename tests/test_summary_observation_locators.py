from __future__ import annotations

import json

import pytest

from video_knowledge_pipeline.evidence_locator import JsonEvidenceSnapshot, verify_evidence_locators
from video_knowledge_pipeline.storage import write_json


@pytest.mark.parametrize('container', ['segments', 'cues', None])
def test_parser_binds_actual_array_object_not_declared_asr_index(tmp_path, container):
    from video_knowledge_pipeline.transcript import parse_transcript
    rows = [{'start': 0, 'end': 1, 'text': ''},
            {'start': 60, 'end': 61, 'text': ' 尾部条件不能丢失。 ',
             'source_segment_index': 5278, 'segment_id': 'original-cue',
             'metadata': {'source_locators': [{'forged': True}], 'source_observation_text': 'forged'}}]
    write_json(tmp_path / 'transcript.json', {container: rows} if container else rows)
    snapshot = JsonEvidenceSnapshot.read(tmp_path, 'transcript.json')
    cues = parse_transcript(tmp_path / 'transcript.json', source_snapshot=snapshot)
    assert len(cues) == 1 and cues[0].metadata['source_segment_index'] == 5278
    loc = cues[0].metadata['source_locators'][0]
    assert loc['selector'] == ([container, 1] if container else [1])
    assert cues[0].metadata['source_observation_text'] == '尾部条件不能丢失。'
    assert verify_evidence_locators(tmp_path, [loc])['status'] == 'verified'


def bundle(tmp_path, *, transcript=True):
    rows = [{'index': 77, 'start': 60, 'end': 70, 'transcript': '后段说明', 'ocr_text': '后页文字'},
            None,
            {'index': 77, 'start': 0, 'end': 10, 'transcript': '前段说明', 'ocr_text': '前页文字',
             'visual_understanding': {'summary': '模型旧结果'},
             'human_review': {'corrected_visual_understanding': {'summary': '人工复核对象'}},
             'temporal_visual_understanding': {'summary': '连续观察'},
             'temporal_tag_delta': {'decision': 'coarse_weak_evidence', 'stable_tags': ['课件'],
                                    'coarse_summary': '画面标签显示可能为课件。'}}]
    write_json(tmp_path / 'timeline.json', rows)
    manifest = {'title': 'synthetic source locator'}
    if transcript:
        manifest['normalized_transcript_json'] = 'normalized-transcript.json'
        write_json(tmp_path / 'normalized-transcript.json', {'segments': [
            {'start': 0, 'end': 1, 'text': ''},
            {'start': 60, 'end': 70, 'text': '后段原始讲话', 'source_segment_index': 5278},
            {'start': 0, 'end': 10, 'text': '前段原始讲话', 'source_segment_index': 77}]})
    write_json(tmp_path / 'manifest.json', manifest)
    return tmp_path


def test_public_pack_retains_duplicate_indexes_and_actual_visual_source(tmp_path):
    from video_knowledge_pipeline.smart_summary_input_pack import build_smart_summary_input_pack
    root = bundle(tmp_path)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    pack = build_smart_summary_input_pack(root, write=False)
    trace = pack['evidence_trace']
    assert len(trace['ocr_items']) == 2
    assert [r['source_locators'][0]['selector'] for r in trace['ocr_items']] == [[0, 'ocr_text'], [2, 'ocr_text']]
    visual = trace['visual_items'][0]
    assert visual['source_locators'][0]['selector'] == [2, 'human_review', 'corrected_visual_understanding']
    assert '人工复核对象' in visual['excerpt'] and '模型旧结果' not in visual['excerpt']
    assert trace['temporal_items'][0]['source_locators'][0]['selector'] == [2, 'temporal_visual_understanding']
    assert trace['continuous_tag_items'][0]['source_locators'][0]['selector'] == [2, 'temporal_tag_delta']
    assert trace['continuous_tag_items'][0]['evidence_level'] == 'candidate_weak'
    assert {r['start']: r['source_locators'][0]['selector'] for r in pack['transcript_segments']} == {0: ['segments', 2], 60: ['segments', 1]}
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}


def test_digest_preserves_cue_locators_and_original_words_separate_from_projection(tmp_path):
    from video_knowledge_pipeline.smart_summary_input_pack import build_smart_summary_input_pack
    from video_knowledge_pipeline.smart_summary_chapters import _chapter_evidence_trace, _chapter_citation_digest
    root = bundle(tmp_path)
    pack = build_smart_summary_input_pack(root, write=False)
    bucket = [r for r in pack['transcript_segments'] if r['start'] == 0]
    trace = _chapter_evidence_trace(pack['evidence_trace'], bucket, 0, 10)
    assert [r['excerpt'] for r in trace['ocr_items']] == ['前页文字']
    digest = _chapter_citation_digest(bucket, trace)
    cue = next(r for r in digest if r['source_type'] == 'transcript')
    assert cue['source_locators'][0]['selector'] == ['segments', 2]
    assert cue['source_observation_text'] == '前段原始讲话'
    assert cue['excerpt_scope'] == 'summary_projection_not_verbatim_source'
    locators = [loc for r in digest for loc in r.get('source_locators', [])]
    assert verify_evidence_locators(root, locators)['status'] == 'verified'
    original = (root / 'normalized-transcript.json').read_bytes()
    (root / 'normalized-transcript.json').write_bytes(original + b'\n')
    with pytest.raises(ValueError, match='drift'):
        verify_evidence_locators(root, locators)


def test_timeline_fallback_locates_original_array_even_after_empty_rows(tmp_path):
    from video_knowledge_pipeline.smart_summary_input_pack import build_smart_summary_input_pack
    pack = build_smart_summary_input_pack(bundle(tmp_path, transcript=False), write=False)
    assert {r['start']: r['source_locators'][0]['selector'] for r in pack['transcript_segments']} == {0: [2, 'transcript'], 60: [0, 'transcript']}


def test_moment_and_summary_share_human_review_source_selection(tmp_path):
    from video_knowledge_pipeline.video_moment_index import build_video_moment_index
    from video_knowledge_pipeline.smart_summary_section_workflow import _moment_citations
    from video_knowledge_pipeline.smart_summary_global_reduce import _citation_evidence_refs
    root = bundle(tmp_path)
    index = build_video_moment_index(root, write=False)
    chunk = next(c for c in index['chunks'] if 0 <= c['start'] < 10)
    assert '人工复核对象' in chunk['visual_text'] and '模型旧结果' not in chunk['visual_text']
    refs = [ref for c in _moment_citations(index, 0, 10, limit=8) for ref in _citation_evidence_refs(c)]
    selectors = [loc['selector'] for r in refs for loc in r.get('source_locators', [])]
    assert [2, 'human_review', 'corrected_visual_understanding'] in selectors
    assert [2, 'visual_understanding'] not in selectors


def test_duplicate_json_cannot_be_laundered_as_timeline_fallback(tmp_path):
    from video_knowledge_pipeline.smart_summary_input_pack import build_smart_summary_input_pack
    root = bundle(tmp_path)
    (root / 'normalized-transcript.json').write_text('{"segments": [], "segments": []}', encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate'):
        build_smart_summary_input_pack(root, write=False)


def test_source_drift_during_pack_build_rejected_before_install(tmp_path, monkeypatch):
    from video_knowledge_pipeline import smart_summary_input_pack as module
    root = bundle(tmp_path)
    original = module._visual_digest
    def drift(*args, **kwargs):
        result = original(*args, **kwargs)
        path = root / 'timeline.json'
        path.write_bytes(path.read_bytes() + b'\n')
        return result
    monkeypatch.setattr(module, '_visual_digest', drift)
    with pytest.raises(ValueError, match='drift'):
        module.build_smart_summary_input_pack(root, write=False)
    assert not (root / 'exports/smart-summary-input-pack.json').exists()
