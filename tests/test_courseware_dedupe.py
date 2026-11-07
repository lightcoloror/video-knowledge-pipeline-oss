from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from test_courseware_material import fixture_bundle
from video_knowledge_pipeline.courseware_material import read_scan, store_scan
from video_knowledge_pipeline.file_hash import sha256_file


def image_bundle(tmp_path, images):
    root = fixture_bundle(tmp_path, len(images))
    scan = read_scan(root)
    for row, image in zip(scan['candidates'], images):
        image.save(root / row['frame'])
        row['frame_sha256'] = sha256_file(root / row['frame'])
    store_scan(root, scan)
    return root


def resave_candidate(root, index, image, **save_kwargs):
    scan = read_scan(root)
    row = scan['candidates'][index - 1]
    image.save(root / row['frame'], format='PNG', **save_kwargs)
    row['frame_sha256'] = sha256_file(root / row['frame'])
    store_scan(root, scan)


def slide():
    im = Image.new('RGB', (640, 360), 'white')
    draw = ImageDraw.Draw(im)
    draw.text((40, 40), 'Policy: 10%; NOT excluded', fill='black')
    draw.rectangle((40, 100, 400, 250), outline='black', width=2)
    return im


def plan(root, **kwargs):
    from video_knowledge_pipeline.courseware_dedupe import plan_courseware_dedupe
    return plan_courseware_dedupe(root, **kwargs)


def test_full_frame_exact_duplicates_and_revisit_keep_all_occurrences(tmp_path):
    a, b = slide(), slide()
    ImageDraw.Draw(b).rectangle((500, 200, 530, 240), fill='red')
    root = image_bundle(tmp_path, [a, a, b, a])
    before = (root / 'courseware/scan.json').read_bytes()
    report = plan(root, indexes='1,2,3,4')
    assert report['counts']['exact_duplicate'] == 2
    assert report['safe_ocr_indexes'] == [1, 3]
    assert report['rows'][3]['representative_index'] == 1
    assert len(report['rows']) == 4
    assert report['independent_review_complete'] is False
    assert report['queue_modified'] is False
    assert (root / 'courseware/scan.json').read_bytes() == before
    assert not (root / 'courseware/dedupe.json').exists()
    assert report == plan(root, indexes='1,2,3,4')


@pytest.mark.parametrize('kind', ['small_text', 'annotation', 'low_contrast', 'color'])
def test_small_important_changes_are_never_auto_skipped(tmp_path, kind):
    a, b = slide(), slide()
    d = ImageDraw.Draw(b)
    if kind == 'small_text':
        d.text((500, 320), '0 NOT', fill='black')
    elif kind == 'annotation':
        d.line((60, 55, 160, 55), fill='red', width=1)
    elif kind == 'low_contrast':
        d.text((500, 320), 'except', fill=(245, 245, 245))
    else:
        d.rectangle((20, 20, 25, 25), fill=(250, 255, 255))
    root = image_bundle(tmp_path, [a, b])
    report = plan(root, indexes='1,2')
    assert report['safe_ocr_indexes'] == [1, 2]
    assert not report['rows'][1]['safe_to_reuse']


def test_roi_is_preview_only_and_cannot_hide_content(tmp_path):
    a, b = slide(), slide()
    ImageDraw.Draw(b).rectangle((550, 250, 620, 350), fill='blue')
    root = image_bundle(tmp_path, [a, b])
    result = plan(root, indexes='1,2', roi=[0, 0, 500, 360])
    assert result['rows'][1]['classification'] == 'roi_exact_review_required'
    assert result['safe_ocr_indexes'] == [1, 2]


def test_representative_not_previous_frame_prevents_incremental_drift(tmp_path):
    images = []
    for n in range(4):
        im = slide()
        ImageDraw.Draw(im).rectangle((500, 300, 550, 340), fill=(255-n*2,)*3)
        images.append(im)
    root = image_bundle(tmp_path, images)
    result = plan(root, indexes='1,2,3,4')
    assert result['rows'][3]['compared_to_index'] == 1
    assert result['safe_ocr_indexes'] == [1, 2, 3, 4]


@pytest.mark.parametrize('indexes', ['0', '3', '1,1', '', 'no', '1,2,3'])
def test_invalid_indexes_fail_before_frame_decode(tmp_path, indexes):
    root = image_bundle(tmp_path, [slide(), slide()])
    with pytest.raises(ValueError):
        plan(root, indexes=indexes)


def test_bound_limit_and_missing_frame_fail_closed(tmp_path):
    root = image_bundle(tmp_path, [slide(), slide()])
    with pytest.raises(ValueError):
        plan(root, indexes='1,2', limit=1)
    state = read_scan(root)
    state['candidates'][0]['frame'] = ''
    store_scan(root, state)
    with pytest.raises(ValueError):
        plan(root, indexes='1')


def test_frame_drift_and_escaping_path_fail_closed(tmp_path):
    root = image_bundle(tmp_path, [slide()])
    row = read_scan(root)['candidates'][0]
    (root / row['frame']).write_bytes(b'bad')
    with pytest.raises(ValueError, match='hash'):
        plan(root, indexes='1')
    state = read_scan(root)
    state['candidates'][0]['frame'] = '../escape.png'
    store_scan(root, state)
    with pytest.raises(ValueError, match='outside'):
        plan(root, indexes='1')


@pytest.mark.parametrize('roi', [[-1, 0, 100, 100], [0, 0, 999, 10], [0, 0, 0, 10], [0, 0, 10], [False, 0, 10, 10]])
def test_invalid_roi_fail_closed(tmp_path, roi):
    root = image_bundle(tmp_path, [slide()])
    with pytest.raises(ValueError):
        plan(root, indexes='1', roi=roi)


def test_cli_plan_is_read_only_and_compact(tmp_path, capsys):
    import json
    from video_knowledge_pipeline.courseware_cli import main
    root = image_bundle(tmp_path, [slide(), slide()])
    assert main(['courseware-dedupe-plan', str(root), '--indexes', '1,2']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['safe_ocr_indexes'] == [1]
    assert result['queue_modified'] is False
    assert 'rows' not in result


def child_rows(root):
    from video_knowledge_pipeline.storage import write_json
    rows = [{'index': n, 'courseware_page_id': p['page_id'],
             'courseware_frame_sha256': p['frame_sha256'], 'frame_paths': [str(root/p['frame'])]}
            for n, p in enumerate(read_scan(root)['candidates'], 1)]
    write_json(root/'courseware/ocr-bundle/timeline.json', rows)


def original_ocr(root):
    from test_courseware_material import page_result
    from video_knowledge_pipeline.courseware_material import import_courseware_page
    value = page_result(root)
    value['executor'] = 'ebook/RapidOCR'
    value['model_actual'] = 'synthetic actual OCR'
    import_courseware_page(root, value)


def original_ocr_with_visual_identity(root):
    from test_courseware_material import page_result
    from video_knowledge_pipeline.courseware_dedupe import frame_visual_identity
    from video_knowledge_pipeline.courseware_material import import_courseware_page, read_scan
    value = page_result(root)
    value['executor'] = 'ebook/RapidOCR'
    value['model_actual'] = 'synthetic actual OCR'
    value['frame_visual_identity'] = frame_visual_identity(root, read_scan(root)['candidates'][0])
    import_courseware_page(root, value)


def test_exact_ocr_reuse_preserves_times_provenance_and_independent_review(tmp_path):
    from video_knowledge_pipeline.courseware_exact_reuse import reuse_exact_courseware_pages
    from video_knowledge_pipeline.courseware_material import build_courseware_material
    from video_knowledge_pipeline.storage import read_json
    root = image_bundle(tmp_path, [slide(), slide()])
    child_rows(root)
    original_ocr(root)
    original = (root/'courseware/pages/page-0.json').read_bytes()
    timeline = (root/'timeline.json').read_bytes()
    result = reuse_exact_courseware_pages(root, child_indexes=[2])
    assert result['ocr_indexes'] == [] and result['reused_indexes'] == [2]
    value = read_json(root/'courseware/pages/page-1.json')
    assert value['time'] == 1.0
    assert value['exact_frame_reuse']['source_page_id'] == 'page-0'
    assert value['exact_frame_reuse']['inference_performed_for_target'] is False
    material = build_courseware_material(root, write=False)
    assert material['extraction_complete'] and not material['independent_review_complete']
    assert material['pages'][1]['exact_frame_reuse'] == value['exact_frame_reuse']
    assert (root/'courseware/pages/page-0.json').read_bytes() == original
    assert (root/'timeline.json').read_bytes() == timeline
    assert reuse_exact_courseware_pages(root, child_indexes=[2])['reused_indexes'] == []


def test_near_match_does_not_skip_ocr(tmp_path):
    from video_knowledge_pipeline.courseware_exact_reuse import reuse_exact_courseware_pages
    b = slide()
    ImageDraw.Draw(b).point((10,10), fill=(254,255,255))
    root = image_bundle(tmp_path, [slide(), b])
    child_rows(root)
    original_ocr(root)
    result = reuse_exact_courseware_pages(root, child_indexes=[2])
    assert result['ocr_indexes'] == [2]
    assert not (root/'courseware/pages/page-1.json').exists()


@pytest.mark.parametrize('drift', ['result', 'frame', 'child'])
def test_exact_reuse_rejects_drift_without_overwriting(tmp_path, drift):
    from video_knowledge_pipeline.courseware_exact_reuse import reuse_exact_courseware_pages
    from video_knowledge_pipeline.courseware_material import build_courseware_material
    from video_knowledge_pipeline.storage import read_json, write_json
    root = image_bundle(tmp_path, [slide(), slide()])
    child_rows(root)
    original_ocr(root)
    if drift == 'child':
        path = root/'courseware/ocr-bundle/timeline.json'
        data = read_json(path)
        data[1]['courseware_frame_sha256'] = '0'*64
        write_json(path, data)
        with pytest.raises(ValueError):
            reuse_exact_courseware_pages(root, child_indexes=[2])
        assert not (root/'courseware/pages/page-1.json').exists()
        return
    reuse_exact_courseware_pages(root, child_indexes=[2])
    target_before = (root/'courseware/pages/page-1.json').read_bytes()
    if drift == 'frame':
        (root/read_scan(root)['candidates'][0]['frame']).write_bytes(b'drift')
    else:
        path = root/'courseware/pages/page-0.json'
        data = read_json(path)
        data['blocks'][0]['text'] = 'different source result'
        write_json(path, data)
    with pytest.raises(ValueError):
        build_courseware_material(root, write=False)
    assert (root/'courseware/pages/page-1.json').read_bytes() == target_before


def test_missing_or_manual_result_not_reused(tmp_path):
    from video_knowledge_pipeline.courseware_exact_reuse import reuse_exact_courseware_pages
    from test_courseware_material import page_result
    from video_knowledge_pipeline.courseware_material import import_courseware_page
    root = image_bundle(tmp_path, [slide(), slide()])
    child_rows(root)
    assert reuse_exact_courseware_pages(root, child_indexes=[2])['ocr_indexes'] == [2]
    value = page_result(root)
    value['executor'] = 'manual'
    import_courseware_page(root, value)
    assert reuse_exact_courseware_pages(root, child_indexes=[2])['ocr_indexes'] == [2]


def test_forged_reuse_content_rejected_atomically(tmp_path):
    from video_knowledge_pipeline.courseware_exact_reuse import reuse_exact_courseware_pages
    from video_knowledge_pipeline.courseware_material import import_courseware_page
    from video_knowledge_pipeline.storage import read_json
    root = image_bundle(tmp_path, [slide(), slide()])
    child_rows(root)
    original_ocr(root)
    reuse_exact_courseware_pages(root, child_indexes=[2])
    path = root/'courseware/pages/page-1.json'
    before = path.read_bytes()
    value = read_json(path)
    value['blocks'][0]['text'] = 'forged changed number 1000'
    with pytest.raises(ValueError, match='content'):
        import_courseware_page(root, value)
    assert path.read_bytes() == before


@pytest.mark.parametrize('indexes', [[], [0], [2,2], [True], list(range(1,10)), [3]])
def test_exact_reuse_index_bounds(tmp_path, indexes):
    from video_knowledge_pipeline.courseware_exact_reuse import reuse_exact_courseware_pages
    root = image_bundle(tmp_path, [slide(), slide()])
    child_rows(root)
    with pytest.raises(ValueError):
        reuse_exact_courseware_pages(root, child_indexes=indexes)


def test_within_subbatch_duplicates_wait_for_real_representative_result(tmp_path):
    from video_knowledge_pipeline.courseware_exact_reuse import reuse_exact_courseware_pages
    root = image_bundle(tmp_path, [slide(), slide()])
    child_rows(root)
    planned = reuse_exact_courseware_pages(root, child_indexes=[1,2], defer_within_batch=True)
    assert planned['ocr_indexes'] == [1] and planned['deferred_indexes'] == [2]
    assert not (root/'courseware/pages/page-1.json').exists()
    # No successful representative result -> nothing is falsely imported.
    assert reuse_exact_courseware_pages(root, child_indexes=[2])['ocr_indexes'] == [2]
    original_ocr(root)
    assert reuse_exact_courseware_pages(root, child_indexes=[2])['reused_indexes'] == [2]


def test_bounded_collect_imports_only_representative_before_reuse(tmp_path):
    from video_knowledge_pipeline.courseware_exact_reuse import reuse_exact_courseware_pages
    from video_knowledge_pipeline.courseware_scan import collect_courseware_extractions
    from video_knowledge_pipeline.storage import read_json, write_json
    root = image_bundle(tmp_path, [slide(), slide()])
    child_rows(root)
    path = root/'courseware/ocr-bundle/timeline.json'
    rows = read_json(path)
    for row in rows:
        row['visual_text'] = '完整可验证的课件文字，不能覆盖人工结果。'
    write_json(path, rows)
    result = collect_courseware_extractions(root, executor='ebook/RapidOCR', model_actual='fixture', indexes=[1], refresh_material=False)
    assert result['material'] is None
    assert (root/'courseware/pages/page-0.json').exists()
    assert read_json(root/'courseware/pages/page-0.json')['frame_visual_identity']['schema'].endswith('.v2')
    assert not (root/'courseware/pages/page-1.json').exists()
    assert reuse_exact_courseware_pages(root, child_indexes=[2])['reused_indexes'] == [2]
    with pytest.raises(ValueError):
        collect_courseware_extractions(root, executor='ebook/RapidOCR', model_actual='fixture', indexes=[1,99])


def test_same_decoded_frame_with_different_png_encoding_reuses_safely(tmp_path):
    from PIL.PngImagePlugin import PngInfo
    from video_knowledge_pipeline.courseware_exact_reuse import reuse_exact_courseware_pages
    from video_knowledge_pipeline.storage import read_json
    root = image_bundle(tmp_path, [slide(), slide()])
    source_sha = read_scan(root)['candidates'][0]['frame_sha256']
    metadata = PngInfo()
    metadata.add_text('Comment', 'different container bytes, same displayed pixels')
    resave_candidate(root, 2, slide(), compress_level=9, pnginfo=metadata)
    target_sha = read_scan(root)['candidates'][1]['frame_sha256']
    assert source_sha != target_sha
    child_rows(root)
    original_ocr_with_visual_identity(root)
    result = reuse_exact_courseware_pages(root, child_indexes=[2], visual_source_indexes=[1],
                                          allow_decoded_identity=True)
    assert result['reused_indexes'] == [2]
    value = read_json(root/'courseware/pages/page-1.json')
    assert value['exact_frame_reuse']['method'] == 'identical_decoded_frame_identity_v2'
    assert value['exact_frame_reuse']['source_frame_sha256'] == source_sha
    assert value['exact_frame_reuse']['target_frame_sha256'] == target_sha


def test_decoded_identity_within_batch_waits_for_direct_ocr_source(tmp_path):
    from PIL.PngImagePlugin import PngInfo
    from video_knowledge_pipeline.courseware_exact_reuse import reuse_exact_courseware_pages
    root = image_bundle(tmp_path, [slide(), slide()])
    metadata = PngInfo()
    metadata.add_text('Comment', 'different bytes')
    resave_candidate(root, 2, slide(), compress_level=9, pnginfo=metadata)
    child_rows(root)
    planned = reuse_exact_courseware_pages(root, child_indexes=[1, 2], defer_within_batch=True,
                                           allow_decoded_identity=True)
    assert planned['ocr_indexes'] == [1]
    assert planned['deferred_indexes'] == [2]
    original_ocr_with_visual_identity(root)
    completed = reuse_exact_courseware_pages(root, child_indexes=[2], visual_source_indexes=[1],
                                             allow_decoded_identity=True)
    assert completed['reused_indexes'] == [2]


def test_decoded_identity_sources_require_explicit_policy(tmp_path):
    from video_knowledge_pipeline.courseware_exact_reuse import reuse_exact_courseware_pages
    root = image_bundle(tmp_path, [slide(), slide()])
    child_rows(root)
    with pytest.raises(ValueError, match='explicit decoded-identity policy'):
        reuse_exact_courseware_pages(root, child_indexes=[2], visual_source_indexes=[1])


def test_collect_resume_preserves_legacy_result_bytes(tmp_path):
    from video_knowledge_pipeline.courseware_scan import collect_courseware_extractions
    from video_knowledge_pipeline.storage import read_json, write_json
    root = image_bundle(tmp_path, [slide()])
    child_rows(root)
    original_ocr(root)
    path = root/'courseware/pages/page-0.json'
    before = path.read_bytes()
    child = root/'courseware/ocr-bundle/timeline.json'
    rows = read_json(child)
    rows[0]['visual_text'] = 'different later OCR must not overwrite the canonical result'
    write_json(child, rows)
    result = collect_courseware_extractions(root, executor='ebook/RapidOCR',
        model_actual='fixture', refresh_material=False)
    assert result['items'][0]['status'] == 'no_op'
    assert path.read_bytes() == before


def test_rendering_metadata_change_is_not_reused(tmp_path):
    from video_knowledge_pipeline.courseware_exact_reuse import reuse_exact_courseware_pages
    root = image_bundle(tmp_path, [slide(), slide()])
    resave_candidate(root, 2, slide(), icc_profile=b'synthetic-different-display-profile')
    child_rows(root)
    original_ocr_with_visual_identity(root)
    result = reuse_exact_courseware_pages(root, child_indexes=[2], visual_source_indexes=[1],
                                          allow_decoded_identity=True)
    assert result['ocr_indexes'] == [2]
    assert result['reused_indexes'] == []


def test_alpha_only_change_is_not_reused(tmp_path):
    from video_knowledge_pipeline.courseware_exact_reuse import reuse_exact_courseware_pages
    first = slide().convert('RGBA')
    second = first.copy()
    second.putpixel((10, 10), (*second.getpixel((10, 10))[:3], 254))
    root = image_bundle(tmp_path, [first, second])
    child_rows(root)
    original_ocr_with_visual_identity(root)
    result = reuse_exact_courseware_pages(root, child_indexes=[2], visual_source_indexes=[1],
                                          allow_decoded_identity=True)
    assert result['ocr_indexes'] == [2]
    assert result['reused_indexes'] == []


def test_v2_state_consensus_separates_transient_and_persistent_changes(tmp_path):
    from video_knowledge_pipeline.courseware_dedupe import plan_courseware_dedupe_v2
    a = slide()
    b = slide()
    ImageDraw.Draw(b).rectangle((500, 200, 530, 240), fill='red')
    transient_dir = tmp_path/'transient'
    transient_dir.mkdir()
    transient_root = image_bundle(transient_dir, [a, b, a])
    transient = plan_courseware_dedupe_v2(transient_root, indexes='1,2,3')
    assert transient['rows'][1]['state_classification'] == 'transient_change_review'
    assert transient['safe_ocr_indexes'] == [1, 2]
    stable_dir = tmp_path/'stable'
    stable_dir.mkdir()
    stable_root = image_bundle(stable_dir, [a, b, b])
    stable = plan_courseware_dedupe_v2(stable_root, indexes='1,2,3')
    assert stable['rows'][1]['state_classification'] == 'stable_new_state_candidate'
    assert stable['rows'][1]['state_persistence'] == 2
    assert stable['safe_ocr_indexes'] == [1, 2]


def test_v2_state_analysis_never_promotes_near_match_to_safe_reuse(tmp_path):
    from video_knowledge_pipeline.courseware_dedupe import plan_courseware_dedupe_v2
    changed = slide()
    ImageDraw.Draw(changed).text((500, 320), 'except', fill=(245, 245, 245))
    root = image_bundle(tmp_path, [slide(), changed, changed])
    result = plan_courseware_dedupe_v2(root, indexes='1,2,3')
    assert result['rows'][1]['state_classification'] == 'stable_new_state_candidate'
    assert result['rows'][1]['safe_to_reuse'] is False
    assert 2 in result['safe_ocr_indexes']
    assert result['automatic_near_reuse'] is False


def test_v2_marks_persistent_approximate_change_for_review_only(tmp_path):
    from video_knowledge_pipeline.courseware_dedupe import plan_courseware_dedupe_v2
    changed = slide()
    ImageDraw.Draw(changed).rectangle((500, 200, 530, 240), fill='red')
    supported = changed.copy()
    ImageDraw.Draw(supported).point((10, 10), fill=(254, 255, 255))
    root = image_bundle(tmp_path, [slide(), changed, supported])
    result = plan_courseware_dedupe_v2(root, indexes='1,2,3')
    assert result['rows'][1]['state_classification'] == 'persistent_visual_change_review'
    assert result['rows'][1]['approximate_state_support_indexes'] == [3]
    assert result['persistent_visual_change_review_indexes'] == [2]
    assert result['safe_ocr_indexes'] == [1, 2, 3]


def test_v2_cli_is_compact_and_read_only(tmp_path, capsys):
    import json
    from video_knowledge_pipeline.courseware_cli import main
    root = image_bundle(tmp_path, [slide(), slide()])
    before = (root/'courseware/scan.json').read_bytes()
    assert main(['courseware-dedupe-plan-v2', str(root), '--indexes', '1,2']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['schema'].endswith('.v2')
    assert result['safe_ocr_indexes'] == [1]
    assert 'rows' not in result
    assert (root/'courseware/scan.json').read_bytes() == before
