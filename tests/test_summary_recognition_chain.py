"""Synthetic moment -> section -> final reader evidence; no model calls."""
from copy import deepcopy
import json

import pytest

from video_knowledge_pipeline.canonical_json import canonical_json_sha256 as digest
from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.video_moment_index import build_video_moment_index
from video_knowledge_pipeline.smart_summary_section_workflow import _moment_citations
from video_knowledge_pipeline.smart_summary_global_reduce import _chapter_fact_pack, _reduce_prompt_plan
from video_knowledge_pipeline.summary_citation_audit import build_citation_index


def _source(tmp_path, *, level='confirmed', human=False, bad=False, long=False):
    provenance = {'frame_sha256s': ['bad' if bad else 'a' * 64],
                  'context_text_sha256s': [], 'model_actual': 'reported-synthetic-model',
                  'executor': 'synthetic-worker'}
    item = {'index': 1, 'start': 0, 'end': 2, 'frame_path': 'assets/synthetic.jpg',
            'visual_text': '课件限定条件',
            'visual_understanding': {'objects': [('合成内容' * 20000 if long else '模型观察')],
                  'evidence_level': level, 'inference_basis': '合成图示有歧义',
                  'recognition_provenance': provenance}}
    if human:
        item['human_corrected_visual_text'] = '人工更正文字'
        item['structured_visual'] = [{'markdown': '旧机器错误不能回来'}]
        item['human_corrected_visual_understanding'] = {'objects': ['人工观察'], 'evidence_level': 'confirmed'}
    write_json(tmp_path / 'manifest.json', {'title': 'synthetic recognition chain'})
    write_json(tmp_path / 'timeline.json', [item])
    return item


def _pack(tmp_path, **kwargs):
    _source(tmp_path, **kwargs)
    index = build_video_moment_index(tmp_path, write=False)
    citations = _moment_citations(index, 0, 2, limit=8)
    section = {'section_id': 'chapter-0001', 'title': '合成章节', 'time_range': '00:00:00.000 - 00:00:02.000',
               'evidence': {'summary_sentences': ['合成课程说明。']}, 'citations': citations}
    pack = _chapter_fact_pack(tmp_path, [section], {'sections': [section]})
    return index, pack


def test_real_moment_section_reducer_keeps_constituents_not_one_model(tmp_path):
    before = _source(tmp_path)
    original = (tmp_path / 'timeline.json').read_bytes()
    moment, pack = _pack(tmp_path)
    index = build_citation_index(pack)
    assert not index.errors, index.errors
    leaves = [r for name, r in index.refs.items() if index.leaves[name] == (name,)]
    visual = [r for r in leaves if r.get('source_kind') == 'visual']
    assert len(visual) == 2
    assert {tuple(r['recognition_provenance']['selected_field']) for r in visual} == {
        ('visual_text',), ('visual_understanding',)}
    vlm = next(r for r in visual if r['recognition_provenance']['model_actual'])
    assert vlm['recognition_provenance']['frame_sha256s'] == ['a' * 64]
    assert vlm['source_locators'][0]['selector'] == [0, 'visual_understanding']
    assert any(r['source_kind'] == 'evidence_group' and r['evidence_id'] == 'moment-0001' for r in index.refs.values())
    assert (tmp_path / 'timeline.json').read_bytes() == original


def test_human_override_cannot_reintroduce_old_structured_machine_words(tmp_path):
    moment, pack = _pack(tmp_path, human=True)
    assert '旧机器错误' not in moment['chunks'][0]['visual_text']
    index = build_citation_index(pack)
    records = [r['recognition_provenance'] for r in index.refs.values() if 'recognition_provenance' in r]
    assert records and all(r['executor'] == 'human_review' for r in records)
    assert all(not r['model_actual'] for r in records)


@pytest.mark.parametrize('ocr_field', ['visual_text', 'ocr_text'])
def test_structured_table_is_not_lost_beside_plain_ocr(tmp_path, ocr_field):
    from video_knowledge_pipeline.summary_claim_review import _scope_sources
    item = _source(tmp_path)
    item.pop('visual_text')
    item[ocr_field] = 'OCR标题'
    item['structured_visual'] = [{'markdown': '|项目|金额|\n|合成项目|20|\n仅限条件甲；不含条件乙。'}]
    write_json(tmp_path / 'timeline.json', [item])
    before = (tmp_path / 'timeline.json').read_bytes()
    moment = build_video_moment_index(tmp_path, write=False)
    assert '仅限条件甲' in moment['chunks'][0]['visual_text']
    section = {'section_id': 'chapter-0001', 'title': '合成章节',
               'evidence': {}, 'citations': _moment_citations(moment, 0, 2, limit=8)}
    pack = _chapter_fact_pack(tmp_path, [section], {'sections': [section]})
    index = build_citation_index(pack)
    assert not index.errors
    rows = [r for r in index.refs.values() if r.get('observation_kind') == 'structured_visual']
    assert len(rows) == 1
    assert rows[0]['source_locators'][0]['selector'] == [0, 'structured_visual']
    assert rows[0]['recognition_provenance']['model_actual'] == ''
    sources = _scope_sources(tmp_path, index, pack, ['moment-0001'])
    assert any('仅限条件甲' in str(row['original_objects']) for row in sources)
    assert (tmp_path / 'timeline.json').read_bytes() == before


def test_summary_input_trace_keeps_complementary_structured_observation(tmp_path):
    from video_knowledge_pipeline.smart_summary_input_pack import _evidence_trace
    item = _source(tmp_path)
    item['structured_visual'] = {'markdown': '独有表格条件：不适用于条件乙'}
    trace = _evidence_trace([item], [], {}, transcript_path=None, manifest={})
    assert any('独有表格条件' in row['excerpt'] for row in trace['ocr_items'])
    item['human_corrected_visual_text'] = '人工审核文字'
    trace = _evidence_trace([item], [], {}, transcript_path=None, manifest={})
    assert all('独有表格条件' not in row['excerpt'] for row in trace['ocr_items'])


def test_recognition_audit_is_not_rendered_as_semantic_content(tmp_path):
    _, pack = _pack(tmp_path)
    rows = [r for r in build_citation_index(pack).refs.values()
            if r.get('observation_kind') == 'visual_understanding']
    assert len(rows) == 1
    assert '模型观察' in rows[0]['snippet']
    assert 'reported-synthetic-model' not in rows[0]['snippet']
    assert 'a' * 64 not in rows[0]['snippet']
    assert rows[0]['recognition_provenance']['model_actual'] == 'reported-synthetic-model'


def test_public_summary_pack_never_resurrects_human_superseded_table(tmp_path):
    from video_knowledge_pipeline.smart_summary_input_pack import build_smart_summary_input_pack
    item = _source(tmp_path, human=True)
    item['transcript'] = '合成讲话，仅用于验证证据投影。'
    write_json(tmp_path / 'timeline.json', [item])
    before = (tmp_path / 'timeline.json').read_bytes()
    pack = build_smart_summary_input_pack(tmp_path, write=False)
    assert '旧机器错误不能回来' not in str(pack['visual_digest'])
    assert '人工更正文字' in str(pack['visual_digest'])
    assert (tmp_path / 'timeline.json').read_bytes() == before


def test_invalid_recognition_metadata_blocks_existing_final_reader_gate(tmp_path):
    from test_smart_summary_reader_plan import _fact_pack, _valid_plan
    from video_knowledge_pipeline.smart_summary_reader_plan import validate_reader_plan
    _, produced = _pack(tmp_path, bad=True)
    pack, plan = _fact_pack(), _valid_plan()
    pack['sections'].extend(produced['sections'])
    result = validate_reader_plan(plan, fact_pack=pack, expected_section_ids=set(plan['source_section_ids']))
    assert not result['passed']
    assert any('recognition' in e for e in result['errors'])


def test_truncated_observation_is_explicit_review_only_not_complete(tmp_path):
    _, pack = _pack(tmp_path, long=True)
    index = build_citation_index(pack)
    truncated = [r for r in index.refs.values() if r.get('source_excerpt_truncated')]
    assert len(truncated) == 1
    assert truncated[0]['evidence_id'] in index.review_only
    assert truncated[0]['source_observation_chars'] > len(truncated[0]['snippet'])


@pytest.mark.parametrize('level', ['inferred', 'candidate_weak'])
def test_final_reader_requires_qualification_for_weak_only_support(level):
    from test_smart_summary_reader_plan import _fact_pack, _valid_plan
    from video_knowledge_pipeline.smart_summary_reader_plan import validate_reader_plan
    pack, plan = _fact_pack(), _valid_plan()
    pack['sections'][0]['evidence_refs'][0]['evidence_level'] = level
    result = validate_reader_plan(plan, fact_pack=pack, expected_section_ids=set(plan['source_section_ids']))
    assert any('weak_evidence_unqualified' in e for e in result['errors'])
    for key in ('overview',):
        if plan[key]['evidence_ids'] == ['ev-0001']:
            plan[key]['text'] = '可能：' + plan[key]['text']
    for key in ('core_insights', 'principles', 'themes', 'actions', 'reusable_expressions'):
        for item in plan.get(key, []):
            if item.get('evidence_ids') == ['ev-0001']:
                for field in ('text', 'summary', 'problem', 'reason', 'method', 'case', 'action'):
                    if item.get(field):
                        item[field] = '可能：' + item[field]
    result = validate_reader_plan(plan, fact_pack=pack, expected_section_ids=set(plan['source_section_ids']))
    assert not any('weak_evidence_unqualified' in e for e in result['errors'])


def test_claim_audit_caps_correlated_support_and_preserves_counterevidence():
    from test_structured_claim_verification import fixture, audit
    data = fixture()
    refs = list(data[0].refs.values())
    for ref in refs[:2]:
        ref['recognition_provenance'] = {'frame_sha256s': ['a' * 64]}
    data = (build_citation_index({'sections': [{'evidence_refs': refs}]}), *data[1:])
    data[1]['reference_snapshot_sha256'] = data[0].snapshot_sha256
    for assertion in data[1]['assertions']:
        assertion['reference_canonical_sha256'] = digest(data[0].refs[assertion['evidence_id']])
    result = audit(data)
    assert result['valid'] and result['claims'][0]['status'] == 'supported'
    lineage = result['claims'][0]['source_lineage']
    assert lineage['independent_source_count'] == 0
    assert lineage['candidate_weight'] == 1  # row + footnote is one correlated source
    assert lineage['correlated_group_count'] == 1
    opposing = deepcopy(data[1]['assertions'][0])
    opposing.update(assertion_id='counter', evidence_id='noise', context_id='other-observation',
                    reference_canonical_sha256=digest(data[0].refs['noise']),
                    span={'start': 0, 'end': len(data[0].refs['noise']['snippet']),
                          'canonical_sha256': digest(data[0].refs['noise']['snippet'])})
    opposing['slots'] = {**data[2][0]['slots'], 'value': '30'}
    data[1]['assertions'].append(opposing)
    result = audit(data)
    assert result['claims'][0]['status'] == 'contradiction'
    assert result['claims'][0]['source_lineage']['original_weight'] == 1


def test_final_prompt_exposes_level_lineage_and_truncation(tmp_path):
    _, pack = _pack(tmp_path, level='inferred')
    rows = [{'section_id': 'chapter-0001', 'title': '合成章节', 'final_markdown': '合成内容', 'time_range': '00:00:00.000 - 00:00:02.000'}]
    prompt = _reduce_prompt_plan(tmp_path, rows, {}, fact_pack=pack, max_input_chars=16000)['prompt']
    assert 'inferred' in prompt
    assert 'recognition' in prompt
    assert 'reported-synthetic-model' in prompt


def test_existing_review_packet_carries_actual_source_level_and_recognition(tmp_path):
    from video_knowledge_pipeline.summary_claim_review import _scope_sources
    _, pack = _pack(tmp_path, level='inferred')
    index = build_citation_index(pack)
    sources = _scope_sources(tmp_path, index, pack, ['moment-0001'])
    visual = next(row for row in sources if row.get('evidence_level') == 'inferred')
    assert visual['recognition_provenance']['model_actual'] == 'reported-synthetic-model'
    assert visual['original_objects'][0]['objects'] == ['模型观察']


def test_tag_only_action_is_rejected_even_with_maybe_qualifier():
    from test_smart_summary_reader_plan import _fact_pack, _valid_plan
    from video_knowledge_pipeline.smart_summary_reader_plan import validate_reader_plan
    pack, plan = _fact_pack(), _valid_plan()
    pack['sections'][0]['evidence_refs'][0]['evidence_level'] = 'candidate_weak'
    plan['core_insights'][0]['text'] = '画面标签显示，他可能随后点击按钮导致页面关闭。'
    result = validate_reader_plan(plan, fact_pack=pack, expected_section_ids=set(plan['source_section_ids']))
    assert any('tag_only_action_or_causality_claim' in error for error in result['errors'])


def test_conflicting_constituent_id_cannot_be_silently_deduplicated():
    from video_knowledge_pipeline.smart_summary_global_reduce import _dedupe_evidence_refs
    with pytest.raises(ValueError, match='conflicting'):
        _dedupe_evidence_refs([{'evidence_id': 'one', 'snippet': 'first'},
                              {'evidence_id': 'one', 'snippet': 'changed'}])


def test_observation_snippet_drift_blocks_reference_index(tmp_path):
    _, pack = _pack(tmp_path)
    ref = next(r for r in pack['sections'][0]['evidence_refs'] if 'recognition_provenance' in r)
    ref['snippet'] = '被替换的片段'
    assert any('snippet_drift' in error for error in build_citation_index(pack).errors)


def test_public_reduce_refuses_bad_source_provenance_before_model_planning(tmp_path):
    from video_knowledge_pipeline.smart_summary_global_reduce import run_smart_summary_global_reduce
    moment, pack = _pack(tmp_path, bad=True)
    row = {'section_id': 'chapter-0001', 'title': '合成章节', 'time_range': '00:00:00.000 - 00:00:02.000',
           'final_markdown': '合成课程说明。', 'citations': _moment_citations(moment, 0, 2, limit=8), 'evidence': {}}
    write_json(tmp_path / 'exports/smart-summary-section-workflow.json', {'sections': [row]})
    write_json(tmp_path / 'exports/smart-summary-section-llm-revisions.json', {'rows': [row]})
    write_json(tmp_path / 'exports/course-map.json', {})
    before = {str(p): p.read_bytes() for p in tmp_path.glob('**/*.json')}
    result = run_smart_summary_global_reduce(tmp_path, execute=False, write=False)
    assert result['ok'] is False
    assert result['provider_call_performed'] is False
    assert 'recognition' in str(result['next_actions'])
    assert before == {str(p): p.read_bytes() for p in tmp_path.glob('**/*.json')}
