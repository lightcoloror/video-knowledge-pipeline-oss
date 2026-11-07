"""Bounded changed-region planning and native-detector result assembly.

This module does not run a detector or OCR, and does not claim that cropping
preserves full-image detector behavior. The opt-in caller owns fallbacks and
records that distinction. ROI coordinates are integer, right/bottom exclusive.
"""
from __future__ import annotations

from collections import defaultdict, deque
from hashlib import sha256
import math
import re

from .canonical_json import canonical_json_bytes, canonical_json_sha256
from .courseware_ocr_geometry import _box

MAX_PIXELS = 16_777_216
MAX_BOXES = 512
CONTEXT_PIXELS = 32
CROP_EDGE_MARGIN = 2
MAX_ROI_PERCENT = 35
_OUTPUT_FIELDS = ('boxes', 'scores', 'boxes_dtype')
_BOX_DTYPES = {'int32', 'int64', 'float32', 'float64'}
_PLAN_FIELDS = {'mode', 'reason', 'roi', 'changed_pixels', 'retained_indexes',
                'removed_indexes', 'outside_sha256'}
_FULL_REASONS = {'no_anchor', 'input_shape_or_dtype', 'shape_changed', 'pixel_budget',
                 'invalid_anchor_output', 'no_anchor_boxes', 'no_pixel_change',
                 'roi_area_limit', 'outside_pixels_changed'}


class LocalDetectionFallback(ValueError):
    """The caller must use its existing full-image detector path."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__('courseware local detection: ' + reason)


def _full(reason, changed_pixels=0):
    return {'mode': 'full', 'reason': reason, 'roi': None, 'changed_pixels': changed_pixels,
            'retained_indexes': [], 'removed_indexes': [], 'outside_sha256': None}


def _checked_shape(shape):
    if (not isinstance(shape, (list, tuple)) or len(shape) != 3
            or any(type(n) is not int or n < 1 for n in shape) or shape[2] != 3
            or shape[0] * shape[1] > MAX_PIXELS):
        raise LocalDetectionFallback('invalid_shape')
    return shape[0], shape[1]


def _image_problem(image):
    import numpy as np
    if (not isinstance(image, np.ndarray) or image.dtype.name != 'uint8'
            or image.ndim != 3 or image.shape[2] != 3
            or image.shape[0] < 1 or image.shape[1] < 1):
        return 'input_shape_or_dtype'
    if image.shape[0] * image.shape[1] > MAX_PIXELS:
        return 'pixel_budget'
    return None


def _checked_roi(roi, shape):
    height, width = _checked_shape(shape)
    if (not isinstance(roi, list) or len(roi) != 4 or any(type(n) is not int for n in roi)
            or not 0 <= roi[0] < roi[2] <= width or not 0 <= roi[1] < roi[3] <= height):
        raise LocalDetectionFallback('invalid_roi')
    return (roi[2] - roi[0]) * (roi[3] - roi[1])


def _checked_output(value, shape, reason):
    """Use the existing native-quad validator; accept cache-record extra fields."""
    height, width = _checked_shape(shape)
    try:
        if not isinstance(value, dict):
            raise ValueError('output mapping required')
        boxes, scores, dtype = (value.get(key) for key in _OUTPUT_FIELDS)
        if (not isinstance(boxes, list) or not isinstance(scores, list)
                or not len(boxes) == len(scores) <= MAX_BOXES
                or (boxes and (not isinstance(dtype, str) or dtype not in _BOX_DTYPES))
                or (not boxes and dtype is not None)):
            raise ValueError('output fields or budget')
        bounds = []
        for box, score in zip(boxes, scores):
            bounds.append(_box(box, width, height))
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('score')
            if dtype in {'int32', 'int64'} and any(type(n) is not int for point in box for n in point):
                raise ValueError('integer dtype')
        if value.get('output_sha256') != canonical_json_sha256({key: value[key] for key in _OUTPUT_FIELDS}):
            raise ValueError('output hash')
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise LocalDetectionFallback(reason) from exc
    return bounds


def _intersects(bounds, roi):
    # Boundary contact is intentionally included: a touching old box is unsafe
    # to retain as an independent observation outside the current detector crop.
    return (bounds['l'] <= roi[2] and bounds['r'] >= roi[0]
            and bounds['t'] <= roi[3] and bounds['b'] >= roi[1])


def _contained(bounds, roi):
    return (roi[0] <= bounds['l'] <= bounds['r'] <= roi[2]
            and roi[1] <= bounds['t'] <= bounds['b'] <= roi[3])


def _partition(bounds, roi):
    removed = [index for index, box in enumerate(bounds) if _intersects(box, roi)]
    removed_set = set(removed)
    retained = [index for index in range(len(bounds)) if index not in removed_set]
    return retained, removed


def _outside_sha256(image, roi):
    """Hash framed shape/ROI plus top, bottom, middle-left, middle-right bytes.

    These four nonoverlapping strips exhaust the outside region. At most one
    strip's byte copy is live at a time, and no per-pixel Python loop is needed.
    """
    left, top, right, bottom = roi
    digest = sha256(canonical_json_bytes({'shape': list(image.shape), 'roi': roi}))
    for strip in (image[:top], image[bottom:], image[top:bottom, :left], image[top:bottom, right:]):
        digest.update(strip.tobytes())
    return digest.hexdigest()


def plan_changed_region(anchor_image, current_image, anchor_output):
    """Return one conservative candidate ROI or an explicit full-image fallback.

    Every nonzero RGB difference is counted once per pixel, including changes
    outside all old text. No threshold, erosion, movement classification, or
    connected-component truncation can erase a small changed region.
    """
    import numpy as np
    problem = _image_problem(current_image)
    if problem is not None:
        return _full(problem)
    if anchor_image is None or anchor_output is None:
        return _full('no_anchor')
    problem = _image_problem(anchor_image)
    if problem is not None:
        return _full(problem)
    if anchor_image.shape != current_image.shape:
        return _full('shape_changed')
    try:
        bounds = _checked_output(anchor_output, current_image.shape, 'invalid_anchor_output')
    except LocalDetectionFallback:
        return _full('invalid_anchor_output')
    if not bounds:
        return _full('no_anchor_boxes')

    changed = np.any(anchor_image != current_image, axis=2)
    changed_pixels = int(np.count_nonzero(changed))
    if not changed_pixels:
        return _full('no_pixel_change')  # Whole-input cache owns the exact-hit path.
    # Axis reductions avoid allocating two index arrays proportional to every
    # changed pixel. Memory and work are bounded by the validated input budget.
    rows = np.flatnonzero(np.any(changed, axis=1))
    columns = np.flatnonzero(np.any(changed, axis=0))
    height, width = current_image.shape[:2]
    roi = [max(0, int(columns[0]) - CONTEXT_PIXELS),
           max(0, int(rows[0]) - CONTEXT_PIXELS),
           min(width, int(columns[-1]) + 1 + CONTEXT_PIXELS),
           min(height, int(rows[-1]) + 1 + CONTEXT_PIXELS)]
    # Each expanding pass includes at least one previously non-contained old
    # box forever, so at most len(bounds) expanding passes are possible.
    for _ in range(len(bounds) + 1):
        previous_roi = roi.copy()
        for box in bounds:
            if _intersects(box, roi):
                roi = [min(roi[0], math.floor(box['l'])), min(roi[1], math.floor(box['t'])),
                       max(roi[2], math.ceil(box['r'])), max(roi[3], math.ceil(box['b']))]
        if (roi[2] - roi[0]) * (roi[3] - roi[1]) * 100 > height * width * MAX_ROI_PERCENT:
            return _full('roi_area_limit', changed_pixels)
        if roi == previous_roi:
            break
    outside_sha256 = _outside_sha256(current_image, roi)
    if outside_sha256 != _outside_sha256(anchor_image, roi):
        return _full('outside_pixels_changed', changed_pixels)
    retained, removed = _partition(bounds, roi)
    plan = {'mode': 'roi', 'reason': 'changed_region', 'roi': roi, 'changed_pixels': changed_pixels,
            'retained_indexes': retained, 'removed_indexes': removed, 'outside_sha256': outside_sha256}
    validate_roi_plan(plan, current_image.shape, anchor_output)
    return plan


def validate_roi_plan(plan, shape, anchor_output):
    """Check receipt structure and geometry, without rechecking source pixels.

    A plausible outside digest and changed-pixel count are not independent
    evidence of pixel equality; only plan_changed_region reads those pixels.
    """
    height, width = _checked_shape(shape)
    if (not isinstance(plan, dict) or set(plan) != _PLAN_FIELDS
            or not isinstance(plan.get('mode'), str) or plan['mode'] not in {'full', 'roi'}
            or not isinstance(plan.get('reason'), str)
            or type(plan.get('changed_pixels')) is not int
            or not 0 <= plan['changed_pixels'] <= height * width):
        raise LocalDetectionFallback('invalid_plan')
    for name in ('retained_indexes', 'removed_indexes'):
        indexes = plan[name]
        if (not isinstance(indexes, list) or len(indexes) > MAX_BOXES
                or any(type(index) is not int or index < 0 for index in indexes)
                or indexes != sorted(set(indexes))):
            raise LocalDetectionFallback('invalid_plan_partition')
    if plan['mode'] == 'full':
        if (plan['reason'] not in _FULL_REASONS or plan['roi'] is not None
                or plan['outside_sha256'] is not None or plan['retained_indexes'] or plan['removed_indexes']
                or (plan['reason'] not in {'roi_area_limit', 'outside_pixels_changed'} and plan['changed_pixels'] != 0)
                or (plan['reason'] in {'roi_area_limit', 'outside_pixels_changed'} and plan['changed_pixels'] == 0)):
            raise LocalDetectionFallback('invalid_full_plan')
        return None
    roi = plan['roi']
    area = _checked_roi(roi, shape)
    if (plan['reason'] != 'changed_region' or not 0 < plan['changed_pixels'] <= area
            or area * 100 > height * width * MAX_ROI_PERCENT
            or not isinstance(plan['outside_sha256'], str)
            or re.fullmatch(r'[a-f0-9]{64}', plan['outside_sha256']) is None):
        raise LocalDetectionFallback('invalid_roi_plan')
    bounds = _checked_output(anchor_output, shape, 'invalid_anchor_output')
    if not bounds:
        raise LocalDetectionFallback('no_anchor_boxes')
    retained, removed = _partition(bounds, roi)
    if (plan['retained_indexes'] != retained or plan['removed_indexes'] != removed
            or any(not _contained(bounds[index], roi) for index in removed)):
        raise LocalDetectionFallback('invalid_plan_partition')
    return None


def merge_region_output(anchor_output, roi_output, roi, shape, sorted_boxes):
    """Remove intersecting old boxes and sort shifted local boxes natively.

    Supply the pinned detector's sorted_boxes callable. Exact quad identities
    map the returned order back to score queues, preserving duplicate counts.
    Scores retain their input pairing; this does not repair upstream detector
    score ordering before the supplied output was captured.
    """
    import numpy as np
    height, width = _checked_shape(shape)
    _checked_roi(roi, shape)
    anchor_bounds = _checked_output(anchor_output, shape, 'invalid_anchor_output')
    left, top, right, bottom = roi
    crop_shape = (bottom - top, right - left, 3)
    roi_bounds = _checked_output(roi_output, crop_shape, 'invalid_roi_output')
    if not roi_bounds:
        raise LocalDetectionFallback('empty_roi_output')
    for box in roi_bounds:
        if ((left > 0 and box['l'] <= CROP_EDGE_MARGIN)
                or (top > 0 and box['t'] <= CROP_EDGE_MARGIN)
                or (right < width and box['r'] >= crop_shape[1] - CROP_EDGE_MARGIN)
                or (bottom < height and box['b'] >= crop_shape[0] - CROP_EDGE_MARGIN)):
            raise LocalDetectionFallback('roi_box_near_crop_edge')
    retained, _ = _partition(anchor_bounds, roi)
    if len(retained) + len(roi_bounds) > MAX_BOXES:
        raise LocalDetectionFallback('merged_box_budget')
    boxes = [[[x, y] for x, y in anchor_output['boxes'][index]] for index in retained]
    scores = [anchor_output['scores'][index] for index in retained]
    boxes.extend([[[x + left, y + top] for x, y in box] for box in roi_output['boxes']])
    scores.extend(roi_output['scores'])
    dtype = (np.result_type(anchor_output['boxes_dtype'], roi_output['boxes_dtype']).name
             if retained else roi_output['boxes_dtype'])
    array = np.asarray(boxes, dtype=dtype)
    if array.tolist() != boxes:
        # Adding an integer crop origin can require more precision than the
        # local float32 coordinates. Preserve coordinates instead of rounding.
        dtype = 'float64'
        array = np.asarray(boxes, dtype=dtype)
    if array.tolist() != boxes:
        raise LocalDetectionFallback('merged_coordinate_precision')
    identities = defaultdict(deque)
    for box, score in zip(array, scores):
        identities[box.tobytes()].append(score)
    try:
        ordered = sorted_boxes(array.copy())
    except Exception as exc:
        raise LocalDetectionFallback('sorted_boxes_failed') from exc
    if (not isinstance(ordered, np.ndarray) or ordered.shape != array.shape
            or ordered.dtype != array.dtype):
        raise LocalDetectionFallback('sorted_boxes_identity')
    ordered_scores = []
    for box in ordered:
        key = box.tobytes()
        if not identities.get(key):
            raise LocalDetectionFallback('sorted_boxes_identity')
        ordered_scores.append(identities[key].popleft())
    if any(remaining for remaining in identities.values()):
        raise LocalDetectionFallback('sorted_boxes_identity')
    value = {'boxes': ordered.tolist(), 'scores': ordered_scores, 'boxes_dtype': dtype}
    value['output_sha256'] = canonical_json_sha256(value)
    _checked_output(value, shape, 'invalid_merged_output')
    return value
