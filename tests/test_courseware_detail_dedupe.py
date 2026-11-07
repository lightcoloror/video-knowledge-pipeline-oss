from PIL import Image

from test_courseware_dedupe import image_bundle
from test_courseware_material import fixture_bundle, page_result
from video_knowledge_pipeline.courseware_dedupe import plan_courseware_dedupe
from video_knowledge_pipeline.courseware_material import build_courseware_groups, import_courseware_page
from video_knowledge_pipeline.courseware_material import (
    build_courseware_material, render_courseware_model_input, courseware_reading_signature,
)


def test_one_level_local_change_is_retained_and_localized(tmp_path):
    a = Image.new('RGB', (1024, 768), 'white')
    b = a.copy()
    b.putpixel((500, 400), (254, 255, 255))
    root = image_bundle(tmp_path, [a, b])
    result = plan_courseware_dedupe(root, indexes='1,2')
    row = result['rows'][1]
    assert row['ocr_required'] and not row['safe_to_reuse']
    metrics = row['metrics']
    assert metrics['exact_difference_bbox'] == [500, 400, 501, 401]
    assert metrics['native_difference_bbox'] is None
    assert 0 < metrics['detail_tile_pixel_budget'] < 1024*768
    assert any(x <= 500 < r and y <= 400 < b for x,y,r,b in metrics['changed_detail_tiles'])
    assert metrics['detail_tiles_are_review_only']


def test_reading_groups_preserve_raw_and_numbers(tmp_path):
    root = fixture_bundle(tmp_path, 3)
    texts = ['Coverage 10%\nNot covered.', 'Coverage 10%  Not covered.', 'Coverage 11% Not covered.']
    for i, text in enumerate(texts):
        import_courseware_page(root, page_result(root, f'page-{i}', text))
    before = {p.name: p.read_bytes() for p in (root/'courseware/pages').glob('*.json')}
    result = build_courseware_groups(root, write=False)
    assert result['observed_text_group_count'] == 3
    assert result['reading_group_count'] == 2
    assert result['reading_duplicate_occurrence_count'] == 1
    assert not result['reading_groups'][0]['allows_ocr_skip']
    assert len(result['reading_groups'][0]['members']) == 2
    assert before == {p.name: p.read_bytes() for p in (root/'courseware/pages').glob('*.json')}
    assert not result['complete']


def test_lossless_prompt_reuse_keeps_every_occurrence_and_changed_number(tmp_path):
    root = fixture_bundle(tmp_path, 3)
    text = 'Coverage 10%; not covered except under the stated conditions. '*30
    for i in range(3):
        import_courseware_page(root, page_result(root, f'page-{i}', text if i < 2 else text.replace('10%', '11%')))
    material = build_courseware_material(root, write=False)
    baseline = render_courseware_model_input(material, deduplicate=False)
    compact = render_courseware_model_input(material)
    assert len(compact) < len(baseline)
    assert compact.count(text) == 1
    assert '11%' in compact and 'EXACT TEXT REUSE' in compact
    for page in material['pages']:
        assert page['page_id'] in compact
        assert page['blocks'][0]['evidence_id'] in compact
    assert not material['complete']


def test_reading_signature_preserves_tables_and_structured_metadata():
    a = [{'kind': 'table', 'text': 'A  B\n1  2'}]
    b = [{'kind': 'table', 'text': 'A B 1 2'}]
    assert courseware_reading_signature(a) != courseware_reading_signature(b)
    # This signature is text-only. Structure remains per occurrence in the prompt.
    root_block = {'kind':'body','text':'Same '*100, 'evidence_id':'a', 'cells':[1]}
    other = {**root_block, 'evidence_id':'b', 'cells':[2]}
    material = {'active':True,'revision':'fixture','complete':False,'scan_complete':True,
                'extraction_complete':False,'independent_review_complete':False,'review_gaps':[],
                'pages':[{'page_id':str(i),'time':i,'end':i+1,'blocks':[b],'gaps':[]}
                         for i,b in enumerate([root_block,other])]}
    output = render_courseware_model_input(material)
    assert 'EXACT TEXT REUSE' in output
    assert '"cells": [1]' in output and '"cells": [2]' in output
