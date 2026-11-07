from __future__ import annotations

import json

import pytest

from test_courseware_material import fixture_bundle, page_result
from video_knowledge_pipeline.courseware_material import build_courseware_material, import_courseware_page, material_for_range
from video_knowledge_pipeline.storage import write_json


def test_input_pack_actual_markdown_keeps_material_beyond_digest(tmp_path):
    from video_knowledge_pipeline.smart_summary_input_pack import build_smart_summary_input_pack, _render_pack_markdown
    root = fixture_bundle(tmp_path, 81)
    import_courseware_page(root, page_result(root, 'page-80', '后半段独有脚注：不适用于排除情况。' * 80))
    pack = build_smart_summary_input_pack(root, write=False, max_visual_items=1)
    assert len(pack['courseware_material']['pages']) == 81
    assert '后半段独有脚注' in _render_pack_markdown(pack)
    assert len(pack['courseware_material']['pages'][-1]['blocks'][0]['text']) > 1000
    assert pack['companion_courseware'] != pack['courseware_material']


def test_chapter_raw_material_bypasses_term_replacement(tmp_path, monkeypatch):
    from video_knowledge_pipeline import smart_summary_chapters as chapters
    root = fixture_bundle(tmp_path)
    original = '课件写无额外条件；讲者后面可能纠正。'
    import_courseware_page(root, page_result(root, text=original))
    monkeypatch.setattr(chapters, 'load_bundle_term_replacements', lambda *_: [('无额外条件', '有额外条件')])
    result = chapters.build_smart_summary_chapter_pack(root, write=False, target_chapters=1, chapter_mode='fixed')
    assert result['courseware_material']['pages'][0]['blocks'][0]['text'] == original
    assert result['chapters'][0]['courseware_material']['pages'][0]['blocks'][0]['text'] == original


def test_section_prompt_contains_entire_material_or_blocks(tmp_path):
    from video_knowledge_pipeline.smart_summary_section_workflow import _section_evidence
    from video_knowledge_pipeline.smart_summary_section_llm import _section_prompt
    root = fixture_bundle(tmp_path)
    text = '| row | unit |\n|---|---|\n' + ''.join(f'| {i} | percent |\n' for i in range(200)) + '不可删除的最后一条限定'
    import_courseware_page(root, page_result(root, text=text))
    material = build_courseware_material(root, write=False)
    evidence = _section_evidence({'courseware_material': material, 'visual_notes': []})
    section = {'evidence': evidence, 'title': 'synthetic', 'section_id': 'chapter-0001'}
    prompt = _section_prompt(root, section, max_prompt_chars=20000)
    assert text in prompt and 'courseware:page-0:0' in prompt
    with pytest.raises(ValueError, match='budget'):
        _section_prompt(root, section, max_prompt_chars=1200)
    material['revision'] = 'old'
    with pytest.raises(ValueError, match='stale'):
        _section_prompt(root, section, max_prompt_chars=20000)


def test_reduce_prompt_keeps_complete_channel_and_blocks_small_budget(tmp_path):
    from video_knowledge_pipeline.smart_summary_global_reduce import _reduce_prompt_plan
    root = fixture_bundle(tmp_path)
    text = '逐页原文包含否定限定和数字123；' * 160
    import_courseware_page(root, page_result(root, text=text))
    rows = [{'section_id': 'chapter-0001', 'final_markdown': 'spoken later correction', 'time_range': '00:00:00 - 00:00:02'}]
    result = _reduce_prompt_plan(root, rows, {}, max_input_chars=100000)
    assert text in result['prompt']
    small = _reduce_prompt_plan(root, rows, {}, max_input_chars=1200)
    assert len(small['prompt']) > 1200  # Existing reducer's pre-call budget gate rejects, not truncates pages.
    assert text in small['prompt']


def test_revision_freshness_rejects_old_reducer_before_provider(tmp_path, monkeypatch):
    from video_knowledge_pipeline import smart_summary_global_reduce as reducer
    root = fixture_bundle(tmp_path)
    import_courseware_page(root, page_result(root))
    material = build_courseware_material(root, write=False)
    write_json(root / 'exports/smart-summary-section-workflow.json', {'courseware_material': material, 'sections': [{'section_id': 'chapter-0001'}]})
    write_json(root / 'exports/smart-summary-section-llm-revisions.json', {'courseware_material': material, 'rows': [{'section_id': 'chapter-0001', 'final_markdown': 'old', 'courseware_revision': 'old'}]})
    monkeypatch.setattr(reducer, 'evaluate_production_artifact_gate', lambda *a, **k: {'formal_generation_allowed': True})
    monkeypatch.setattr(reducer, 'model_task_api_call', lambda *a, **k: pytest.fail('no Provider request allowed'))
    result = reducer.run_smart_summary_global_reduce(root, execute=True, write=False)
    assert result['status'] == 'blocked_courseware_stale' and not result['ok']


def test_cli_preview_does_not_read_source_or_write(tmp_path, capsys):
    from video_knowledge_pipeline.courseware_cli import main
    assert main(['courseware-scan', str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)['execution_performed'] is False
    assert list(tmp_path.iterdir()) == []


def test_page_interval_assignment_retains_late_candidate(tmp_path):
    root = fixture_bundle(tmp_path, 100)
    material = build_courseware_material(root, write=False)
    assert material_for_range(material, 80, 100)['pages'][-1]['page_id'] == 'page-99'


def test_actual_section_gateway_receives_full_material_and_records_revision(tmp_path, monkeypatch):
    from video_knowledge_pipeline import smart_summary_section_llm as llm
    from video_knowledge_pipeline.storage import read_json
    root = fixture_bundle(tmp_path)
    text = '表格原文保留全部行、单位和最后的限定条件。' * 100
    import_courseware_page(root, page_result(root, text=text))
    material = build_courseware_material(root, write=False)
    section = {'section_id': 'chapter-0001', 'title': 'synthetic', 'start_time': '00:00:00',
               'end_time': '00:00:02', 'evidence': {'courseware_material': material}}
    workflow = {'courseware_material': material, 'sections': [section]}
    monkeypatch.setattr(llm, 'build_smart_summary_section_workflow', lambda *a, **k: workflow)
    monkeypatch.setattr(llm, 'evaluate_production_artifact_gate', lambda *a, **k: {'formal_generation_allowed': True})
    calls = []
    def fake_call(**kwargs):
        calls.append(kwargs)
        assert text in kwargs['messages'][1]['content']
        return {'ok': True, 'content': '### 00:00:00 - 00:00:02 synthetic\n\n' + '课件原文与讲者纠正应分别保留为证据，限定条件必须复核。' * 10}
    monkeypatch.setattr(llm, 'call_openai_compatible_text', fake_call)
    result = llm.run_smart_summary_section_llm_rewrite(root, execute=True, install=False,
              max_prompt_chars=20000, provider_config={'provider': 'openai_compatible', 'base_url': 'http://127.0.0.1:1/v1', 'model': 'synthetic'})
    assert result['ok'] and len(calls) == 1
    revisions = read_json(root / 'exports/smart-summary-section-llm-revisions.json')
    assert revisions['rows'][0]['courseware_revision'] == material['revision']


def test_legacy_manual_pack_contains_complete_courseware(tmp_path):
    from video_knowledge_pipeline.smart_summary_codex import _render_llm_rewrite_pack
    root = fixture_bundle(tmp_path)
    text = '课件末尾特有限定条件。' * 100
    import_courseware_page(root, page_result(root, text=text))
    material = build_courseware_material(root, write=False)
    rendered = _render_llm_rewrite_pack(root, title='synthetic', provider='local',
               pack={'courseware_material': material}, chapter_pack={}, memory_pack={},
               baseline_path=root / 'absent.md', output_path=root / 'output.md')
    assert text in rendered


def test_missing_section_consumption_cannot_pass_quality(tmp_path):
    from video_knowledge_pipeline.courseware_material import courseware_summary_gate, import_courseware_review
    root = fixture_bundle(tmp_path)
    import_courseware_page(root, page_result(root))
    material = build_courseware_material(root, write=False)
    import_courseware_review(root, {'material_revision': material['revision'], 'reviewer': 'human', 'method': 'full-scope',
                                    'scope_reviewed': True, 'page_ids': ['page-0'], 'gaps': []})
    material = build_courseware_material(root, write=False)
    write_json(root / 'exports/smart-summary-section-workflow.json', {'courseware_material': material,
               'sections': [{'section_id': 'one'}, {'section_id': 'two'}]})
    write_json(root / 'exports/smart-summary-section-llm-revisions.json', {'courseware_material': material,
               'rows': [{'section_id': 'one', 'courseware_revision': material['revision']}]})
    (root / 'exports/smart-summary.md').write_text('synthetic', encoding='utf-8')
    assert not courseware_summary_gate(root)['passed']
