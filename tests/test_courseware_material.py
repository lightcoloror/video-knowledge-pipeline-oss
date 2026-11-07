from __future__ import annotations

from pathlib import Path

import pytest

from video_knowledge_pipeline.courseware_material import (
    build_courseware_material, courseware_prompt, import_courseware_page,
    import_courseware_review, store_scan, read_scan,
)
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.storage import write_json


def fixture_bundle(tmp_path: Path, count: int = 1) -> Path:
    root = tmp_path / 'bundle'
    root.mkdir()
    write_json(root / 'manifest.json', {'title': 'Synthetic lecture'})
    write_json(root / 'timeline.json', [{'index': 1, 'start': 0, 'end': count + 1, 'transcript': 'spoken evidence'}])
    candidates = []
    for n in range(count):
        path = root / 'courseware' / 'frames' / f'{n}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        # Material integrity uses frame bytes; decoder tests use real PNGs separately.
        path.write_bytes(b'synthetic frame ' + str(n).encode())
        candidates.append({'page_id': f'page-{n}', 'time': float(n), 'end': float(n + 1),
                           'frame': path.relative_to(root).as_posix(), 'frame_sha256': sha256_file(path)})
    store_scan(root, {'schema': 'video_knowledge_pipeline.courseware_scan.v1',
                     'run_id': 'synthetic', 'scope': {'start': 0, 'end': count + 1, 'excluded_intervals': []},
                     'source': {'sha256': 'a' * 64}, 'windows': [{'start': 0, 'end': count + 1}],
                     'completed_windows': [0], 'candidates': candidates, 'failure': None})
    return root


def page_result(root: Path, page_id: str = 'page-0', text: str = '完整课件原文，含否定条件和脚注。') -> dict:
    state = read_scan(root)
    row = next(x for x in state['candidates'] if x['page_id'] == page_id)
    return {'schema': 'video_knowledge_pipeline.courseware_page_result.v1',
            'scan_identity': state['identity'], 'page_id': page_id, 'time': row['time'],
            'frame_sha256': row['frame_sha256'], 'executor': 'synthetic', 'model_actual': 'fixture',
            'blocks': [{'kind': 'body', 'text': text}], 'gaps': [], 'status': 'extracted'}


def test_full_material_preserves_late_pages_and_long_table(tmp_path):
    root = fixture_bundle(tmp_path, 83)
    table = '| item | value | unit |\n|---|---|---|\n' + ''.join(f'| row {i} | {i} | percent |\n' for i in range(150)) + '不得覆盖原文；*脚注仅限指定条件*'
    for i in range(83):
        import_courseware_page(root, page_result(root, f'page-{i}', table if i == 82 else f'第{i}页，内容完整且含限制条件。'))
    material = build_courseware_material(root, write=False)
    assert len(material['pages']) == 83
    assert material['pages'][-1]['blocks'][0]['text'] == table
    assert 'row 149' in courseware_prompt(material, max_chars=100_000)
    assert material['scan_complete'] and material['extraction_complete']
    assert not material['independent_review_complete'] and not material['complete']


def test_bad_import_is_atomic_and_drift_blocks(tmp_path):
    root = fixture_bundle(tmp_path)
    raw = (root / 'timeline.json').read_bytes()
    result = page_result(root)
    result['frame_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='frame'):
        import_courseware_page(root, result)
    assert (root / 'timeline.json').read_bytes() == raw
    assert not (root / 'courseware/pages/page-0.json').exists()
    import_courseware_page(root, page_result(root))
    (root / 'courseware/frames/0.png').write_bytes(b'drift')
    with pytest.raises(ValueError, match='frame'):
        build_courseware_material(root, write=False)


def test_empty_wrapper_and_budget_fail_closed(tmp_path):
    root = fixture_bundle(tmp_path)
    import_courseware_page(root, page_result(root, text='![source image](frame.png)'))
    material = build_courseware_material(root, write=False)
    assert not material['extraction_complete'] and material['review_gaps']
    with pytest.raises(ValueError, match='budget'):
        courseware_prompt(material, max_chars=10)


def test_review_bound_to_material_and_not_ocr_count(tmp_path):
    root = fixture_bundle(tmp_path)
    import_courseware_page(root, page_result(root))
    before = build_courseware_material(root, write=False)
    review = {'material_revision': before['revision'], 'reviewer': 'independent human',
              'method': 'Full in-scope playback compared with candidate inventory and every screenshot',
              'scope_reviewed': True, 'page_ids': ['page-0'], 'gaps': []}
    import_courseware_review(root, review)
    assert build_courseware_material(root, write=False)['complete']
    review['material_revision'] = 'bad'
    with pytest.raises(ValueError, match='revision'):
        import_courseware_review(root, review)


def test_negative_number_change_and_revisit_not_deduplicated(tmp_path):
    root = fixture_bundle(tmp_path, 3)
    for i, text in enumerate(['无外部条件；额度为10。', '有外部条件；额度为100。', '无外部条件；额度为10。']):
        import_courseware_page(root, page_result(root, f'page-{i}', text))
    m = build_courseware_material(root, write=False)
    assert len(m['pages']) == 3
    assert m['pages'][0]['blocks'][0]['text'] == m['pages'][2]['blocks'][0]['text']
    assert m['pages'][1]['blocks'][0]['text'] != m['pages'][0]['blocks'][0]['text']


def test_unconfigured_bundle_compatibility(tmp_path):
    assert build_courseware_material(tmp_path, write=False) == {'active': False}


@pytest.mark.parametrize('field,value', [('time', -1), ('time', float('nan')), ('frame_time', float('nan')), ('frame_time', None), ('model_actual', ''), ('page_id', '../escaped')])
def test_invalid_result_identity_fields(tmp_path, field, value):
    root = fixture_bundle(tmp_path)
    result = page_result(root)
    result[field] = value
    before = (root / 'timeline.json').read_bytes()
    with pytest.raises(ValueError):
        import_courseware_page(root, result)
    assert (root / 'timeline.json').read_bytes() == before


def test_explicit_page_supersession_keeps_history_and_invalidates_review(tmp_path):
    root = fixture_bundle(tmp_path)
    import_courseware_page(root, page_result(root))
    old = build_courseware_material(root, write=False)
    path = root / 'courseware/pages/page-0.json'
    digest = sha256_file(path)
    update = page_result(root, text='经独立复核，补充遗漏的小字脚注和限定条件。')
    with pytest.raises(ValueError, match='supersedes'):
        import_courseware_page(root, update)
    update['supersedes_sha256'] = digest
    import_courseware_page(root, update)
    assert (root / f'courseware/history/{digest}.json').is_file()
    assert build_courseware_material(root, write=False)['revision'] != old['revision']


def test_frame_path_escape_is_rejected(tmp_path):
    root = fixture_bundle(tmp_path)
    state = read_scan(root)
    state['candidates'][0]['frame'] = '../outside.png'
    store_scan(root, state)
    with pytest.raises(ValueError, match='outside'):
        build_courseware_material(root, write=False)


def test_structured_table_rows_and_units_are_not_replaced_by_caption(tmp_path):
    root = fixture_bundle(tmp_path)
    value = page_result(root)
    value['blocks'] = [{'kind': 'table', 'text': '完整表格标题与限制条件。',
                        'rows': [[str(n), n, 'percent'] for n in range(180)],
                        'footnote': 'not applicable when excluded'}]
    import_courseware_page(root, value)
    material = build_courseware_material(root, write=False)
    text = courseware_prompt(material, max_chars=20000)
    assert '["179", 179, "percent"]' in text
    assert 'not applicable when excluded' in text


def test_missing_scan_cannot_disable_previously_active_material(tmp_path):
    write_json(tmp_path / 'courseware/material.json', {'active': True})
    with pytest.raises(ValueError, match='cannot silently disable'):
        build_courseware_material(tmp_path, write=False)


def test_review_change_invalidates_summary_and_export_dependencies(tmp_path):
    from video_knowledge_pipeline.courseware_material import assert_material_fresh
    from video_knowledge_pipeline.knowledge_note_export import _knowledge_export_dependency_inputs
    root = fixture_bundle(tmp_path)
    import_courseware_page(root, page_result(root))
    material = build_courseware_material(root, write=False)
    review = {'material_revision': material['revision'], 'reviewer': 'human', 'method': 'full scope',
              'scope_reviewed': True, 'page_ids': ['page-0'], 'gaps': []}
    import_courseware_review(root, review)
    before = build_courseware_material(root, write=False)
    review['supersedes_sha256'] = sha256_file(root / 'courseware/independent-review.json')
    review['method'] = 'full scope plus footnote inspection'
    import_courseware_review(root, review)
    with pytest.raises(ValueError, match='stale'):
        assert_material_fresh(root, before)
    rows = _knowledge_export_dependency_inputs(root, {})
    assert any(row['role'] == 'courseware_independent_review' for row in rows)
