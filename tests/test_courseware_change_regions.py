"""Exact changes are review evidence, never an automatic cursor eraser."""
from PIL import Image, ImageDraw
import pytest

from test_courseware_dedupe import image_bundle
from video_knowledge_pipeline.courseware_dedupe import plan_courseware_dedupe_v2


def pair(tmp_path, pixels, *, roi=None):
    before = Image.new('RGB', (640, 360), 'white')
    after = before.copy()
    for x, y, color in pixels:
        after.putpixel((x, y), color)
    root = image_bundle(tmp_path, [before, after])
    report = plan_courseware_dedupe_v2(root, indexes='1,2', roi=roi)
    return root, report, report['rows'][1]['metrics']['change_regions']


def test_one_level_dot_and_distant_new_text_both_localized(tmp_path):
    _, report, detail = pair(tmp_path, [(5, 6, (254, 255, 255)), (600, 330, (0, 0, 0))])
    assert detail['coordinate_space'] == 'decoded_frame_pixels'
    assert detail['changed_pixel_count'] == 2
    assert detail['regions_xyxy'] == [[5, 6, 6, 7], [600, 330, 601, 331]]
    assert detail['allows_ocr_skip'] is False
    assert report['safe_ocr_indexes'] == [1, 2]


def test_roi_reports_native_coordinates_and_unassessed_outside(tmp_path):
    _, report, detail = pair(tmp_path, [(105, 106, (254, 255, 255)), (600, 330, (0, 0, 0))],
                             roi=[100, 100, 400, 300])
    assert detail['regions_xyxy'] == [[105, 106, 106, 107]]
    assert detail['comparison_bounds_xyxy'] == [100, 100, 400, 300]
    assert detail['outside_comparison'] == 'not_assessed'
    assert report['safe_ocr_indexes'] == [1, 2]


def test_roi_equality_does_not_hide_external_change(tmp_path):
    _, report, detail = pair(tmp_path, [(600, 330, (0, 0, 0))], roi=[0, 0, 400, 300])
    assert detail['regions_xyxy'] == []
    assert detail['outside_comparison'] == 'not_assessed'
    assert report['rows'][1]['classification'] == 'roi_exact_review_required'
    assert report['safe_ocr_indexes'] == [1, 2]


def test_many_components_use_complete_bbox_not_truncated_regions(tmp_path):
    pixels = [(5 + 3 * (i % 10), 5 + 3 * (i // 10), (254, 255, 255)) for i in range(70)]
    _, report, detail = pair(tmp_path, pixels)
    assert detail['status'] == 'bbox_fallback'
    assert detail['reason'] == 'component_budget'
    assert detail['regions_xyxy'] == [[5, 5, 33, 24]]
    assert detail['changed_pixel_count'] == 70
    assert report['safe_ocr_indexes'] == [1, 2]


def test_large_diff_avoids_connected_component_work(tmp_path, monkeypatch):
    import cv2
    monkeypatch.setattr(cv2, 'connectedComponentsWithStats', lambda *a, **kw: pytest.fail('unbounded component work'))
    root = image_bundle(tmp_path, [Image.new('RGB', (640, 360), 'white'), Image.new('RGB', (640, 360), 'black')])
    detail = plan_courseware_dedupe_v2(root, indexes='1,2')['rows'][1]['metrics']['change_regions']
    assert detail['status'] == 'bbox_fallback'
    assert detail['reason'] == 'changed_pixel_budget'
    assert detail['regions_xyxy'] == [[0, 0, 640, 360]]


@pytest.mark.parametrize('shape', ['pointer', 'negation'])
def test_transient_overlay_or_text_is_not_a_noise_fact(tmp_path, shape):
    before = Image.new('RGB', (640, 360), 'white')
    changed = before.copy()
    draw = ImageDraw.Draw(changed)
    if shape == 'pointer':
        draw.polygon([(100, 100), (100, 115), (110, 110)], fill='black')
    else:
        draw.text((100, 100), 'NOT', fill='black')
    root = image_bundle(tmp_path, [before, changed, before])
    result = plan_courseware_dedupe_v2(root, indexes='1,2,3')
    review = result['rows'][1]['transient_change_review']
    assert review['classification'] == 'possible_overlay_or_transient_content'
    assert review['cursor_confirmed'] is False and review['allows_ocr_skip'] is False
    assert review['supporting_indexes'] == [1, 2, 3]
    assert result['safe_ocr_indexes'] == [1, 2]


def test_change_plan_is_repeatable_and_read_only(tmp_path):
    root, first, _ = pair(tmp_path, [(50, 20, (0, 0, 0))])
    files = {p: p.read_bytes() for p in [root/'courseware/scan.json', root/'manifest.json', root/'timeline.json']}
    assert first == plan_courseware_dedupe_v2(root, indexes='1,2')
    assert all(p.read_bytes() == data for p, data in files.items())


def test_transient_localization_does_not_mislabel_another_anchor(tmp_path):
    empty = Image.new('RGB', (640, 360), 'white')
    a, b = empty.copy(), empty.copy()
    a.putpixel((100,100),(254,255,255))
    b.putpixel((200,100),(254,255,255))  # Deletes A's dot as well as adding another.
    root = image_bundle(tmp_path,[empty,a,b,a])
    result = plan_courseware_dedupe_v2(root,indexes='1,2,3,4')
    row = result['rows'][2]
    assert row['compared_to_index'] == 1
    assert row['transient_change_review']['supporting_indexes'] == [2,3,4]
    assert row['transient_change_review']['change_regions'] is None
    assert row['transient_change_review']['localization_status'] == 'unavailable_comparison_anchor_differs'
    assert result['safe_ocr_indexes'] == [1,2,3]
