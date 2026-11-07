from __future__ import annotations

import json
import os

import pytest

from test_courseware_material import fixture_bundle, page_result
from video_knowledge_pipeline.courseware_material import build_courseware_material, import_courseware_page, read_scan, store_scan


def test_ocr_child_environment_does_not_forward_python_binary_packages(monkeypatch):
    from video_knowledge_pipeline.visual_structure_batch import _child_environment
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join(['D:/project/src', 'D:/runtime/Lib/site-packages', 'D:/other/dist-packages']))
    monkeypatch.setenv('PYTHONHOME', 'D:/wrong/python')
    result = _child_environment()
    assert 'site-packages' not in result['PYTHONPATH']
    assert 'dist-packages' not in result['PYTHONPATH']
    assert 'D:/project/src' in result['PYTHONPATH']
    assert 'PYTHONHOME' not in result
    assert 'site-packages' in os.environ['PYTHONPATH']  # parent unchanged


def test_exact_index_capture_and_stable_child_indexes(tmp_path, monkeypatch):
    from PIL import Image
    from video_knowledge_pipeline import courseware_scan as scan
    from video_knowledge_pipeline import temporal_frame_groups as frames
    from video_knowledge_pipeline import visual_structure_batch as ocr
    from video_knowledge_pipeline.storage import read_json
    root = fixture_bundle(tmp_path, 3)
    state = read_scan(root)
    state['source'].update(width=32, height=20, fps=25)
    for p in state['candidates']:
        p.update(frame='', frame_sha256='')
    store_scan(root, state)
    monkeypatch.setattr(scan, '_check_source', lambda _: tmp_path / 'synthetic')
    calls = []
    def capture(item, **kwargs):
        calls.extend(item['times'])
        for path in item['output_paths']:
            Image.new('RGB', (32, 20), 'white').save(path)
        return {'ok': True, 'returncodes': [0]}
    monkeypatch.setattr(frames, '_run_candidate', capture)
    ocr_indexes = []
    def fake_ocr(*args, **kwargs):
        ocr_indexes.append(kwargs['indexes'])
        return {'ok': True, 'status': 'completed'}
    monkeypatch.setattr(ocr, 'run_visual_structure_ebook_batches', fake_ocr)
    scan.extract_courseware_candidates(root, indexes='3', limit=1, execute_ocr=True)
    child = root / 'courseware/ocr-bundle/timeline.json'
    assert [(r['index'],r['courseware_page_id']) for r in read_json(child)] == [(1,'page-2')]
    scan.extract_courseware_candidates(root, indexes='1', limit=1, execute_ocr=True)
    assert [(r['index'],r['courseware_page_id']) for r in read_json(child)] == [(1,'page-2'),(2,'page-0')]
    assert ocr_indexes == [[1], [2]]
    assert all(t >= 2 for t in calls[:3]) and not any(1 <= t < 2 for t in calls)


@pytest.mark.parametrize('indexes', ['0','4','1,1','1,2','oops'])
def test_invalid_exact_selection_rejected_before_source_read(tmp_path, monkeypatch, indexes):
    from video_knowledge_pipeline import courseware_scan as scan
    root = fixture_bundle(tmp_path, 3)
    monkeypatch.setattr(scan, '_check_source', lambda _: pytest.fail('no source read on invalid selection'))
    with pytest.raises(ValueError):
        scan.extract_courseware_candidates(root, indexes=indexes, limit=1)


def test_cli_status_is_compact_and_read_only(tmp_path, capsys):
    from video_knowledge_pipeline.courseware_cli import main
    root = fixture_bundle(tmp_path, 500)
    assert main(['courseware-status',str(root)]) == 0
    output = capsys.readouterr().out
    value = json.loads(output)
    assert len(output.encode('utf-8')) < 8192
    assert value['candidate_count'] == 500 and value['pending_extraction_count'] == 500
    assert not (root / 'courseware/material.json').exists()


def test_model_input_keeps_raw_content_but_not_thousands_of_gap_stubs(tmp_path):
    from video_knowledge_pipeline.courseware_material import courseware_prompt
    root = fixture_bundle(tmp_path, 500)
    text = '| name | amount | unit |\n|---|---|---|\n' + ''.join(f'| row{n} | {n} | percent |\n' for n in range(150)) + 'NOT APPLICABLE; exact final footnote'
    import_courseware_page(root, page_result(root,'page-499',text))
    material = build_courseware_material(root, write=False)
    prompt = courseware_prompt(material,max_chars=12000)
    assert text in prompt and 'courseware:page-499:0' in prompt
    assert '499' in prompt and 'review_gap' in prompt
    assert not material['complete']
    assert len(material['review_gaps']) == 499


def test_group_plan_is_lossless_and_never_infers_unseen_duplicates(tmp_path):
    from video_knowledge_pipeline.courseware_material import build_courseware_groups
    root = fixture_bundle(tmp_path, 4)
    for page_id in ['page-0','page-1']:
        import_courseware_page(root,page_result(root,page_id,'完全相同的原文，数值为10且不适用例外。'))
    import_courseware_page(root,page_result(root,'page-2','完全相同的原文，数值为100且适用例外。'))
    result = build_courseware_groups(root,write=False)
    assert result['observed_text_group_count'] == 2
    assert result['unobserved_candidate_count'] == 1
    assert sum(len(g['members']) for g in result['observed_text_groups']) == 3
    assert result['candidate_count'] == 4 and result['proves_visual_equivalence'] is False
    assert not result['extraction_complete']
    assert result == build_courseware_groups(root,write=False)


def test_model_input_budget_still_rejects_long_observed_content(tmp_path):
    from video_knowledge_pipeline.courseware_material import courseware_prompt
    root = fixture_bundle(tmp_path)
    import_courseware_page(root,page_result(root,text='原始表格脚注不得静默截断。' * 1000))
    with pytest.raises(ValueError,match='courseware_input_budget'):
        courseware_prompt(build_courseware_material(root,write=False),max_chars=2000)


def test_full_json_opt_in_preserves_status_payload(tmp_path,capsys):
    from video_knowledge_pipeline.courseware_cli import main
    root = fixture_bundle(tmp_path,2)
    assert main(['courseware-status',str(root),'--full-json']) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result['pages']) == 2 and len(result['review_gaps']) == 2
    assert not (root / 'courseware/material.json').exists()
