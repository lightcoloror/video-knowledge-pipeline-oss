from __future__ import annotations

import copy
import json

import pytest

from video_knowledge_pipeline.evidence_locator import JsonEvidenceSnapshot, verify_evidence_locators
from video_knowledge_pipeline.storage import write_json


def test_snapshot_binds_exact_json_field_and_detects_byte_drift(tmp_path):
    path = tmp_path / 'timeline.json'
    write_json(path, [{'index': 900, 'text': '首次适用，续期除外。'}])
    snapshot = JsonEvidenceSnapshot.read(tmp_path, 'timeline.json')
    locator = snapshot.locator([0, 'text'])
    assert verify_evidence_locators(tmp_path, [locator])['verified_count'] == 1
    before = path.read_bytes()
    path.write_bytes(before + b'\n')
    with pytest.raises(ValueError, match='drift'):
        verify_evidence_locators(tmp_path, [locator])
    assert path.read_bytes() == before + b'\n'


@pytest.mark.parametrize('selector', [[True], [-1], [0, 'missing'], ['0'], [0] * 17])
def test_selector_identity_is_not_coerced(tmp_path, selector):
    write_json(tmp_path / 'timeline.json', [{'text': 'a'}])
    snap = JsonEvidenceSnapshot.read(tmp_path, 'timeline.json')
    with pytest.raises(ValueError):
        snap.locator(selector)


@pytest.mark.parametrize('field,value', [('artifact_path', '../outside.json'),
    ('artifact_path', 'C:/outside.json'), ('artifact_sha256', '0' * 64),
    ('selected_sha256', '0' * 64), ('schema', 'unknown')])
def test_locator_tampering_fails_closed(tmp_path, field, value):
    write_json(tmp_path / 'timeline.json', [{'text': 'a'}])
    loc = JsonEvidenceSnapshot.read(tmp_path, 'timeline.json').locator([0, 'text'])
    loc[field] = value
    with pytest.raises(ValueError):
        verify_evidence_locators(tmp_path, [loc])


def test_bounds_duplicates_and_non_json(tmp_path):
    (tmp_path / 'bad.json').write_text('{"a":1,"a":2}', encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate'):
        JsonEvidenceSnapshot.read(tmp_path, 'bad.json')
    (tmp_path / 'video.mp4').write_bytes(b'never media')
    with pytest.raises(ValueError):
        JsonEvidenceSnapshot.read(tmp_path, 'video.mp4')
    write_json(tmp_path / 'timeline.json', ['12345'])
    with pytest.raises(ValueError, match='budget'):
        JsonEvidenceSnapshot.read(tmp_path, 'timeline.json', max_bytes=2)
    with pytest.raises(ValueError, match='budget'):
        verify_evidence_locators(tmp_path, [{}] * 4097)


def test_moment_to_fact_refs_preserves_original_array_position(tmp_path):
    from video_knowledge_pipeline.video_moment_index import build_video_moment_index
    from video_knowledge_pipeline.smart_summary_section_workflow import _moment_citations
    from video_knowledge_pipeline.smart_summary_global_reduce import _citation_evidence_refs
    write_json(tmp_path / 'manifest.json', {'title': 'fixture'})
    rows = [{'index': 77, 'start': 60, 'end': 70, 'transcript': '后段'},
            None, {'index': 77, 'start': 0, 'end': 10, 'transcript': '先段', 'ocr_text': '页面'}]
    write_json(tmp_path / 'timeline.json', rows)
    before = (tmp_path / 'timeline.json').read_bytes()
    index = build_video_moment_index(tmp_path, target_window_seconds=30, write=False)
    citations = _moment_citations(index, 0, 100, limit=8)
    refs = [ref for c in citations for ref in _citation_evidence_refs(c)]
    locs = [loc for ref in refs for loc in ref.get('source_locators', [])]
    assert {tuple(loc['selector']) for loc in locs} >= {(0, 'transcript'), (2, 'transcript'), (2, 'ocr_text')}
    assert verify_evidence_locators(tmp_path, locs)['status'] == 'verified'
    assert (tmp_path / 'timeline.json').read_bytes() == before


def test_courseware_block_refs_point_to_raw_receipt_without_prompt_pollution(tmp_path):
    from test_courseware_material import fixture_bundle, page_result
    from video_knowledge_pipeline.courseware_material import build_courseware_material, import_courseware_page, render_courseware_model_input
    from video_knowledge_pipeline.smart_summary_global_reduce import _chapter_fact_section
    root = fixture_bundle(tmp_path)
    import_courseware_page(root, page_result(root, text='原始课件：首次适用，续期除外。' * 10))
    original = (root / 'courseware/pages/page-0.json').read_bytes()
    material = build_courseware_material(root, write=False)
    section = _chapter_fact_section({'section_id': 's'}, {'evidence': {'courseware_material': material}})
    ref = next(r for r in section['evidence_refs'] if r['evidence_id'] == 'courseware:page-0:0')
    assert ref['source_locators'][0]['selector'] == ['blocks', 0]
    assert verify_evidence_locators(root, ref['source_locators'])['status'] == 'verified'
    prompt = render_courseware_model_input(material)
    assert '原始课件' in prompt and 'source_locators' not in prompt
    assert (root / 'courseware/pages/page-0.json').read_bytes() == original


def test_actual_reducer_refuses_stale_evidence_before_render(tmp_path):
    from test_smart_summary_reader_plan import _valid_plan, _fact_pack
    from video_knowledge_pipeline.smart_summary_global_reduce import _validated_reader_plan_content
    write_json(tmp_path / 'timeline.json', [{'text': 'source'}])
    pack = _fact_pack()
    refs = pack['sections'][0]['evidence_refs']
    leaf = next(r for r in refs if r.get('snippet'))
    leaf['source_locators'] = [JsonEvidenceSnapshot.read(tmp_path, 'timeline.json').locator([0, 'text'])]
    pack['bundle_dir'] = str(tmp_path)
    plan = _valid_plan()
    before = copy.deepcopy(pack)
    write_json(tmp_path / 'timeline.json', [{'text': 'changed'}])
    _, validation, content = _validated_reader_plan_content(json.dumps(plan), fact_pack=pack,
        expected_ids={'chapter-0001', 'chapter-0002'}, title='fixture', first_time='00:00:00', last_time='00:01:00')
    assert not validation['passed'] and content == ''
    assert any('source_locator' in e for e in validation['errors'])
    assert pack == before


def test_stale_moment_stops_public_reducer_before_provider(tmp_path, monkeypatch):
    from video_knowledge_pipeline import smart_summary_global_reduce as reduce
    from video_knowledge_pipeline.video_moment_index import build_video_moment_index
    from video_knowledge_pipeline.smart_summary_section_workflow import _moment_citations
    write_json(tmp_path / 'manifest.json', {'title': 'fixture'})
    write_json(tmp_path / 'timeline.json', [{'start': 0, 'end': 10, 'transcript': '原始观察'}])
    index = build_video_moment_index(tmp_path, write=False)
    citations = _moment_citations(index, 0, 10, limit=8)
    write_json(tmp_path / 'exports/smart-summary-section-workflow.json', {'sections': [
        {'section_id': 'chapter-0001', 'citations': citations}]})
    write_json(tmp_path / 'exports/smart-summary-section-llm-revisions.json', {'rows': [
        {'section_id': 'chapter-0001', 'final_markdown': '合成摘要', 'time_range': '00:00:00 - 00:00:10'}]})
    write_json(tmp_path / 'timeline.json', [{'start': 0, 'end': 10, 'transcript': '已变化'}])
    monkeypatch.setattr(reduce, 'evaluate_production_artifact_gate', lambda *a, **k: {'formal_generation_allowed': True})
    monkeypatch.setattr(reduce, 'resolve_model_api_provider_config', lambda *a, **k: {})
    monkeypatch.setattr(reduce, 'resolve_text_provider_config', lambda *a, **k: {'model': 'synthetic'})
    monkeypatch.setattr(reduce, 'model_task_api_call', lambda *a, **k: pytest.fail('no Provider'))
    result = reduce.run_smart_summary_global_reduce(tmp_path, execute=True, write=False)
    assert result['status'] == 'blocked_source_locator' and not result['ok']
    assert result['provider_call_performed'] is False


def test_many_courseware_blocks_form_bounded_groups_without_dropping_members(tmp_path):
    from video_knowledge_pipeline.smart_summary_global_reduce import _chapter_fact_section
    from video_knowledge_pipeline.summary_citation_audit import build_citation_index
    blocks = [{'evidence_id': f'cw:{i}', 'kind': 'body', 'text': f'观察{i}'} for i in range(300)]
    material = {'active': True, 'pages': [{'page_id': 'p', 'time': 0, 'end': 1,
        'status': 'extracted', 'gaps': [], 'blocks': blocks}]}
    section = _chapter_fact_section({'section_id': 's'}, {'evidence': {'courseware_material': material}})
    index = build_citation_index({'sections': [section]})
    assert not index.errors
    assert len(index.resolved_ids(['s:eligible-evidence-set'])) == 300
    assert all(len(r.get('member_evidence_ids', [])) <= 256 for r in section['evidence_refs'])


def test_legacy_and_unlocated_sources_never_receive_complete_freshness(tmp_path):
    from video_knowledge_pipeline.evidence_locator import audit_fact_pack_locators
    pack = {'sections': [{'evidence_refs': [{'evidence_id': 'legacy', 'snippet': 'text'}]}]}
    report = audit_fact_pack_locators(tmp_path, pack)
    assert report['status'] == 'not_assessed' and report['unlocated_evidence_ids'] == ['legacy']


def test_duplicate_locators_read_each_artifact_once(tmp_path, monkeypatch):
    write_json(tmp_path / 'timeline.json', [{'text': 'source'}])
    loc = JsonEvidenceSnapshot.read(tmp_path, 'timeline.json').locator([0, 'text'])
    calls = []
    original = JsonEvidenceSnapshot.read
    def counting(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(JsonEvidenceSnapshot, 'read', counting)
    report = verify_evidence_locators(tmp_path, [loc] * 10)
    assert len(calls) == 1 and report['unique_locator_count'] == 1
    assert report['interpretation_accuracy'] == 'not_assessed'
