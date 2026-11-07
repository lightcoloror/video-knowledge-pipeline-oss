from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest

from video_knowledge_pipeline.canonical_json import canonical_json_sha256
from video_knowledge_pipeline import courseware_local_detection as module


def quad(left, top, right, bottom):
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def output(boxes, scores=None, dtype='float32'):
    value = {'boxes': boxes, 'scores': scores if scores is not None else [.9] * len(boxes),
             'boxes_dtype': dtype if boxes else None}
    value['output_sha256'] = canonical_json_sha256(value)
    return value


def pair():
    anchor = np.zeros((400, 600, 3), dtype=np.uint8)
    current = anchor.copy()
    current[200, 300, 1] = 1
    return anchor, current, output([quad(10, 10, 70, 30)])


def test_single_color_step_outside_old_text_is_not_ignored():
    anchor, current, previous = pair()
    plan = module.plan_changed_region(anchor, current, previous)
    assert plan == {'mode': 'roi', 'reason': 'changed_region', 'roi': [268, 168, 333, 233],
                    'changed_pixels': 1, 'retained_indexes': [0], 'removed_indexes': [],
                    'outside_sha256': plan['outside_sha256']}
    assert len(plan['outside_sha256']) == 64
    assert module.validate_roi_plan(plan, current.shape, previous) is None
    assert np.count_nonzero(anchor) == 0 and previous == output([quad(10, 10, 70, 30)])


def test_rgb_channels_count_as_one_changed_pixel_and_far_apart_changes_survive():
    anchor, current, previous = pair()
    current[200, 300] = [1, 2, 3]
    current[201, 340, 0] = 1
    plan = module.plan_changed_region(anchor, current, previous)
    assert plan['changed_pixels'] == 2
    assert plan['roi'] == [268, 168, 373, 234]


def test_old_box_expansion_repeats_until_every_intersecting_box_is_whole():
    anchor, current, _ = pair()
    previous = output([quad(400, 200, 460, 220), quad(320, 200, 420, 220),
                       quad(10, 10, 70, 30)])
    plan = module.plan_changed_region(anchor, current, previous)
    assert plan['roi'] == [268, 168, 460, 233]
    assert plan['removed_indexes'] == [0, 1] and plan['retained_indexes'] == [2]
    module.validate_roi_plan(plan, current.shape, previous)


def test_fractional_old_box_expansion_uses_outward_integer_bounds():
    anchor, current, _ = pair()
    previous = output([quad(265.5, 167.25, 335.5, 240.5)])
    plan = module.plan_changed_region(anchor, current, previous)
    assert plan['roi'] == [265, 167, 336, 241]


def test_box_touching_region_edge_is_conservatively_removed():
    anchor, current, _ = pair()
    previous = output([quad(333, 180, 350, 200)])
    plan = module.plan_changed_region(anchor, current, previous)
    assert plan['roi'] == [268, 168, 350, 233] and plan['removed_indexes'] == [0]


def test_outside_hash_covers_all_four_unchanged_strips_and_excludes_roi():
    anchor, current, previous = pair()
    plan = module.plan_changed_region(anchor, current, previous)
    current[200, 300] = 2
    assert module.plan_changed_region(anchor, current, previous)['outside_sha256'] == plan['outside_sha256']
    for y, x in [(0, 0), (399, 599), (200, 0), (200, 599)]:
        different_anchor = anchor.copy()
        different_current = current.copy()
        different_anchor[y, x] = different_current[y, x] = 7
        assert module.plan_changed_region(different_anchor, different_current, previous)['outside_sha256'] != plan['outside_sha256']


@pytest.mark.parametrize('kind,reason', [
    ('missing', 'no_anchor'), ('shape', 'shape_changed'), ('dtype', 'input_shape_or_dtype'),
    ('empty', 'no_anchor_boxes'), ('same', 'no_pixel_change'), ('large', 'roi_area_limit'),
    ('invalid_geometry', 'invalid_anchor_output'), ('invalid_hash', 'invalid_anchor_output'),
])
def test_planner_returns_explicit_full_fallback(kind, reason):
    anchor, current, previous = pair()
    if kind == 'missing':
        anchor = None
    elif kind == 'shape':
        anchor = anchor[:300]
    elif kind == 'dtype':
        current = current.astype(np.float32)
    elif kind == 'empty':
        previous = output([])
    elif kind == 'same':
        current = anchor.copy()
    elif kind == 'large':
        current[0, 0, 0] = 1
        current[-1, -1, 0] = 1
    elif kind == 'invalid_geometry':
        previous = output([quad(-1, 0, 20, 20)])
    else:
        previous['output_sha256'] = 'a' * 64
    plan = module.plan_changed_region(anchor, current, previous)
    assert plan['mode'] == 'full' and plan['reason'] == reason
    assert plan['roi'] is None and plan['outside_sha256'] is None
    assert plan['retained_indexes'] == plan['removed_indexes'] == []


def test_input_budget_checked_before_comparing_pixels(monkeypatch):
    anchor, current, previous = pair()
    monkeypatch.setattr(module, 'MAX_PIXELS', 100)
    assert module.plan_changed_region(anchor, current, previous)['reason'] == 'pixel_budget'


def test_expanding_old_box_over_area_budget_requires_full_detection():
    anchor, current, _ = pair()
    plan = module.plan_changed_region(anchor, current, output([quad(0, 0, 599, 399)]))
    assert plan['mode'] == 'full' and plan['reason'] == 'roi_area_limit'


def test_exactly_35_percent_is_allowed_but_one_more_column_is_not():
    anchor = np.zeros((100, 200, 3), dtype=np.uint8)
    current = anchor.copy()
    current[0, 0, 0] = current[37, 67, 0] = 1
    previous = output([quad(150, 75, 180, 90)])
    plan = module.plan_changed_region(anchor, current, previous)
    assert plan['mode'] == 'roi' and plan['roi'] == [0, 0, 100, 70]
    current[37, 68, 0] = 1
    assert module.plan_changed_region(anchor, current, previous)['reason'] == 'roi_area_limit'


@pytest.mark.parametrize('field,value', [
    ('outside_sha256', 'z' * 64), ('outside_sha256', None), ('mode', 'reuse'),
    ('reason', 'no_pixel_change'), ('changed_pixels', True), ('changed_pixels', 0),
    ('changed_pixels', 240001), ('changed_pixels', 4226), ('roi', [268.0, 168, 333, 233]),
    ('roi', [268, 168, 268, 233]), ('roi', [-1, 168, 333, 233]),
    ('roi', [0, 0, 600, 400]), ('retained_indexes', []), ('retained_indexes', [0, 0]),
    ('retained_indexes', [True]), ('removed_indexes', [0]), ('extra', 1),
])
def test_plan_validation_rejects_forged_fields(field, value):
    anchor, current, previous = pair()
    plan = module.plan_changed_region(anchor, current, previous)
    plan[field] = value
    with pytest.raises(module.LocalDetectionFallback):
        module.validate_roi_plan(plan, current.shape, previous)


def test_plan_validation_rejects_intersecting_old_box_not_fully_in_roi():
    anchor, current, previous = pair()
    plan = module.plan_changed_region(anchor, current, previous)
    previous = output([quad(300, 200, 360, 220)])
    plan.update(retained_indexes=[], removed_indexes=[0])
    with pytest.raises(module.LocalDetectionFallback):
        module.validate_roi_plan(plan, current.shape, previous)


def test_full_plan_validates_structure_without_claiming_pixel_review():
    anchor, _, previous = pair()
    plan = module.plan_changed_region(anchor, anchor.copy(), previous)
    module.validate_roi_plan(plan, anchor.shape, previous)
    for changes in [{'outside_sha256': 'a' * 64}, {'roi': [0, 0, 10, 10]},
                    {'retained_indexes': [0]}, {'reason': 'changed_region'}, {'changed_pixels': -1}]:
        with pytest.raises(module.LocalDetectionFallback):
            module.validate_roi_plan({**plan, **changes}, anchor.shape, previous)


def test_plan_validation_cannot_prove_an_authentic_looking_pixel_digest():
    anchor, current, previous = pair()
    plan = module.plan_changed_region(anchor, current, previous)
    plan['outside_sha256'] = sha256(b'another structurally valid receipt').hexdigest()
    module.validate_roi_plan(plan, current.shape, previous)


def test_merge_shifts_roi_removes_old_intersections_sorts_and_preserves_scores():
    previous = output([quad(20, 20, 70, 40), quad(140, 120, 180, 140),
                       quad(180, 120, 225, 140)], [.7, .8, .6])
    local = output([quad(30, 30, 70, 50), quad(5, 5, 20, 20)], [.91, .92])
    before_previous, before_local = deepcopy(previous), deepcopy(local)
    calls = []

    def sorted_boxes(boxes):
        calls.append(boxes.copy())
        return boxes[[2, 0, 1]]

    merged = module.merge_region_output(previous, local, [100, 100, 200, 200], (400, 600, 3), sorted_boxes)
    assert len(calls) == 1
    assert merged['boxes'] == [quad(105, 105, 120, 120), quad(20, 20, 70, 40),
                               quad(130, 130, 170, 150)]
    assert merged['scores'] == [.92, .7, .91]
    assert merged['boxes_dtype'] == 'float32'
    assert merged['output_sha256'] == canonical_json_sha256({k: merged[k] for k in ('boxes', 'scores', 'boxes_dtype')})
    assert previous == before_previous and local == before_local


def test_merge_duplicate_boxes_retain_each_score_and_count():
    repeated = quad(20, 20, 70, 40)
    previous = output([repeated, deepcopy(repeated)], [.7, .8])
    local = output([quad(30, 30, 70, 50)], [.9])
    merged = module.merge_region_output(previous, local, [100, 100, 200, 200], (400, 600, 3),
                                        lambda boxes: boxes[[2, 0, 1]])
    assert merged['scores'] == [.9, .7, .8]
    assert merged['boxes'].count(repeated) == 2


@pytest.mark.parametrize('edge_box', [quad(2, 10, 20, 30), quad(10, 2, 30, 20),
                                    quad(80, 10, 98, 30), quad(10, 80, 30, 98)])
def test_internal_crop_edge_detection_falls_back(edge_box):
    with pytest.raises(module.LocalDetectionFallback, match='roi_box_near_crop_edge'):
        module.merge_region_output(output([quad(10, 10, 20, 20)]), output([edge_box]),
                                   [100, 100, 200, 200], (400, 600, 3), lambda boxes: boxes)


@pytest.mark.parametrize('roi,edge_box', [
    ([0, 0, 100, 100], quad(0, 0, 20, 20)),
    ([500, 300, 600, 400], quad(80, 80, 100, 100)),
])
def test_true_image_edges_are_not_crop_edges(roi, edge_box):
    merged = module.merge_region_output(output([quad(200, 200, 220, 220)]), output([edge_box]),
                                        roi, (400, 600, 3), lambda boxes: boxes)
    assert len(merged['boxes']) == 2


@pytest.mark.parametrize('kind,reason', [
    ('empty_roi', 'empty_roi_output'), ('invalid_local', 'invalid_roi_output'),
    ('invalid_anchor', 'invalid_anchor_output'), ('scores', 'invalid_roi_output'),
    ('lost_duplicate', 'sorted_boxes_identity'), ('sorter_mutation', 'sorted_boxes_identity'),
    ('sorter_failure', 'sorted_boxes_failed'), ('bad_roi', 'invalid_roi'),
    ('bad_shape', 'invalid_shape'), ('row_budget', 'merged_box_budget'),
])
def test_merge_ambiguity_fails_closed(kind, reason):
    previous = output([quad(10, 10, 30, 30)])
    local = output([quad(10, 10, 30, 30)])
    roi, shape = [100, 100, 200, 200], (400, 600, 3)
    sorter = lambda boxes: boxes
    if kind == 'empty_roi':
        local = output([])
    elif kind == 'invalid_local':
        local = output([[[10, 10], [30, 30], [30, 10], [10, 30]]])
    elif kind == 'invalid_anchor':
        previous = output([quad(-1, 10, 30, 30)])
    elif kind == 'scores':
        local = output([quad(10, 10, 30, 30)], [])
    elif kind == 'lost_duplicate':
        previous = output([quad(10, 10, 30, 30)] * 2)
        sorter = lambda boxes: boxes[[0, 2]]
    elif kind == 'sorter_mutation':
        def sorter(boxes):
            boxes[:, :, 0] += 1
            return boxes
    elif kind == 'sorter_failure':
        def sorter(boxes):
            raise RuntimeError('synthetic')
    elif kind == 'bad_roi':
        roi = [100, 100, 700, 200]
    elif kind == 'bad_shape':
        shape = (400, 600, 4)
    else:
        previous = output([quad(10, 10, 30, 30)] * 512)
    with pytest.raises(module.LocalDetectionFallback, match=reason):
        module.merge_region_output(previous, local, roi, shape, sorter)


def test_mixed_dtypes_preserve_fractional_coordinates_and_native_sort_contract():
    previous = output([quad(10, 10, 30, 30)], dtype='int32')
    local = output([quad(10.25, 10.5, 30.25, 30.5)], dtype='float32')
    merged = module.merge_region_output(previous, local, [100, 100, 200, 200], (400, 600, 3),
                                        lambda boxes: boxes[::-1])
    assert merged['boxes'][0] == quad(110.25, 110.5, 130.25, 130.5)
    assert merged['boxes_dtype'] == 'float64'


def test_crop_origin_does_not_round_away_float32_subpixel_coordinates():
    tiny = float(np.float32(10.00001))
    local = output([quad(tiny, 10, 30, 30)])
    merged = module.merge_region_output(output([quad(10, 10, 30, 30)]), local,
                                        [500, 100, 600, 200], (400, 600, 3), lambda boxes: boxes)
    assert merged['boxes'][1][0][0] == tiny + 500
    assert merged['boxes_dtype'] == 'float64'


def test_sorter_cannot_duplicate_a_box_and_drop_another_with_same_count():
    with pytest.raises(module.LocalDetectionFallback, match='sorted_boxes_identity'):
        module.merge_region_output(output([quad(10, 10, 30, 30)]), output([quad(10, 10, 30, 30)]),
                                   [100, 100, 200, 200], (400, 600, 3), lambda boxes: boxes[[0, 0]])
