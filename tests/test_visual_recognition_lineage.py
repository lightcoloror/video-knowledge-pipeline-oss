"""Synthetic producer -> correction -> summary provenance, no model execution."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from video_knowledge_pipeline.transcript_semantic_correction import _evidence_for_cue, _source_support_summary
from video_knowledge_pipeline.smart_summary_input_pack import _compact_evidence_item


def cue():
    return SimpleNamespace(start=0, end=1, text='原始口头说明')


def test_human_ocr_words_not_old_machine_words_are_confirmed():
    item = {'index': 1, 'visual_text': '旧机器错词', 'human_corrected_visual_text': '人工更正文字'}
    before = deepcopy(item)
    row = next(r for r in _evidence_for_cue(cue(), item) if r['source_type'] == 'ocr')
    assert row['text'] == '人工更正文字'
    assert row['evidence_level'] == 'confirmed'
    assert row['recognition_provenance']['selected_field'] == ['human_corrected_visual_text']
    assert row['recognition_provenance']['executor'] == 'human_review'
    assert item == before


def test_real_session_import_preserves_verified_input_identities_without_model_claims(tmp_path):
    from test_codex_vision_session import _bundle, _fill_semantic_results
    from video_knowledge_pipeline.codex_vision_session import prepare_codex_vision_session, import_codex_vision_results
    from video_knowledge_pipeline.smart_summary_chapters import _chapter_citation_digest
    root = _bundle(tmp_path, semantic_count=1)
    pack = prepare_codex_vision_session(root, indexes=[1])
    result_path = _fill_semantic_results(Path(pack['result_template_path']), ['confirmed'])
    result = json.loads(result_path.read_text(encoding='utf-8'))
    result['items'][0]['visual_understanding']['recognition_provenance'] = {'executor': 'forged', 'frame_sha256s': ['f'*64]}
    result_path.write_text(json.dumps(result), encoding='utf-8')
    import_codex_vision_results(root, input_json=result_path)
    item = json.loads((root / 'timeline.json').read_text(encoding='utf-8'))[0]
    provenance = item['visual_understanding']['recognition_provenance']
    assert provenance['frame_sha256s'] == [f['sha256'] for f in pack['items'][0]['frames']]
    assert provenance['executor'] == 'codex_session'
    assert provenance['model_actual'] == 'gpt-5.6-sol'
    assert provenance['task_pack_sha256'] == pack['task_pack_sha256']
    assert len(provenance['context_text_sha256s']) == 2
    rows = _evidence_for_cue(cue(), item)
    visual = next(r for r in rows if r['source_type'] == 'visual_understanding')
    compact = _compact_evidence_item(1, item, kind='visual_understanding')
    assert visual['recognition_provenance'] == compact['recognition_provenance']
    assert not compact['recognition_provenance']['independence_is_verified']
    digest = _chapter_citation_digest([], {'visual_items': [compact]})
    assert digest[0]['recognition_provenance'] == compact['recognition_provenance']
    assert 'forged' not in str(compact['recognition_provenance'])


def test_selected_human_visual_does_not_borrow_machine_identity():
    item = {'index': 1, 'frame_path': 'assets/synthetic.jpg', 'visual_understanding': {'evidence_level': 'confirmed', 'objects': ['旧模型'],
        'model_actual': 'machine-old'}, 'human_corrected_visual_understanding': {'objects': ['人工观察']}}
    row = _compact_evidence_item(1, item, kind='visual_understanding')
    assert row['recognition_provenance']['executor'] == 'human_review'
    assert row['recognition_provenance']['model_actual'] == ''
    assert row['recognition_provenance']['selected_field'] == ['human_corrected_visual_understanding']


def test_recorded_context_consumption_connects_otherwise_separate_roots():
    from video_knowledge_pipeline.canonical_json import canonical_json_sha256
    text_hash = canonical_json_sha256('候选原文')
    rows = [dict(evidence_id='ocr', source_type='ocr', text='候选原文', original_source_sha256='a'*64,
                 recognition_provenance={'observation_text_sha256': text_hash}),
            dict(evidence_id='vlm', source_type='visual_understanding', text='候选原文', original_source_sha256='b'*64,
                 recognition_provenance={'context_text_sha256s': [text_hash]})]
    result = _source_support_summary('原文', '候选原文', rows)
    assert result['candidate_weight'] == 70
    assert result['lineage']['independent_source_count'] == 0


def test_malformed_recorded_inputs_cannot_silently_lose_review_gate():
    rows = [dict(evidence_id='one', source_type='visual_understanding', text='候选',
                 recognition_provenance={'frame_sha256s': ['not-a-hash']})]
    result = _source_support_summary('原文', '候选', rows)
    assert result['needs_review_by_source_vote']


@pytest.mark.parametrize('recorded', [None, {}, {'model_requested': 'requested-not-actual'},
    {'frame_sha256s': ['bad']}, {'context_text_sha256s': ['a'*64]*33}])
def test_projection_preserves_unknown_actual_model_and_malformed_status(recorded):
    from video_knowledge_pipeline.evidence_lineage import project_recognition_provenance
    result = project_recognition_provenance({'recognition_provenance': recorded},
        selected_field=['visual_understanding'], text='synthetic')
    assert result['model_actual'] == ''
    assert not result['independence_is_verified']
    if recorded and ('frame_sha256s' in recorded or 'context_text_sha256s' in recorded):
        assert result['status'] == 'invalid'
        row = dict(evidence_id='one', source_type='visual_understanding', text='synthetic',
                   recognition_provenance=result)
        assert _source_support_summary('old', 'synthetic', [row])['needs_review_by_source_vote']
