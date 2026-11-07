"""Synthetic network-shape budgeting; no model or detector inference runs."""
import ast
from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from video_knowledge_pipeline.courseware_detector_budget import (
    detector_input_budget, validate_detector_budget,
)


class ConfigDetector:
    def __init__(self, side=736, kind='min'):
        self.side, self.kind, self.calls = side, kind, []

    def get_preprocess(self, max_wh):
        self.calls.append(max_wh)
        return SimpleNamespace(limit_side_len=self.side, limit_type=self.kind)


def test_narrow_roi_can_cost_more_network_pixels_than_the_full_frame():
    detector = ConfigDetector()
    budget = detector_input_budget(detector, [1120, 2000, 3], [100, 2000, 3])
    assert detector.calls == [2000, 2000]
    assert budget['full_shape'] == [1120, 1984]
    assert budget['roi_shape'] == [736, 14720]
    assert budget['roi_pixels'] > budget['full_pixels']
    assert budget['allowed'] is False
    assert budget['reason'] == 'roi_not_smaller'
    assert validate_detector_budget(budget, [1120, 2000, 3], [100, 2000, 3]) == budget


def test_smaller_roi_allowed_and_equal_work_rejected():
    detector = ConfigDetector(736, 'max')
    allowed = detector_input_budget(detector, (1024, 1024), (128, 256))
    assert allowed['allowed'] is True
    assert allowed['full_shape'] == [736, 736]
    assert allowed['roi_shape'] == [128, 256]
    assert allowed['full_pixels'] == 736 * 736
    assert allowed['roi_pixels'] == 128 * 256
    assert validate_detector_budget(allowed, (1024, 1024), (128, 256)) == allowed
    equal = detector_input_budget(detector, (128, 128), (128, 128))
    assert equal['allowed'] is False and equal['reason'] == 'roi_not_smaller'


def test_roi_network_pixel_cap_applies_even_when_full_is_larger():
    budget = detector_input_budget(ConfigDetector(), (736, 40000), (64, 2048))
    assert budget['roi_pixels'] < budget['full_pixels']
    assert budget['roi_pixels'] > 16_777_216
    assert budget['allowed'] is False and budget['reason'] == 'roi_pixel_budget'


def test_get_preprocess_is_selected_separately_for_full_and_roi():
    class AdaptiveDetector:
        def get_preprocess(self, max_wh):
            return SimpleNamespace(limit_side_len=96 if max_wh < 160 else 192,
                                   limit_type='max')
    budget = detector_input_budget(AdaptiveDetector(), [256, 256], [128, 128])
    assert budget['full_preprocess'] == {'limit_side_len': 192, 'limit_type': 'max'}
    assert budget['roi_preprocess'] == {'limit_side_len': 96, 'limit_type': 'max'}
    assert budget['full_shape'] == [192, 192] and budget['roi_shape'] == [96, 96]
    assert validate_detector_budget(budget, [256, 256], [128, 128]) == budget


@pytest.mark.parametrize('detector', [
    object(), SimpleNamespace(get_preprocess=None),
    SimpleNamespace(get_preprocess=lambda _: SimpleNamespace()),
    ConfigDetector(736, 'unknown'), ConfigDetector(0), ConfigDetector(True),
    ConfigDetector(float('nan')), ConfigDetector('736'),
])
def test_unavailable_or_unknown_preprocess_fails_closed(detector):
    budget = detector_input_budget(detector, [1120, 2000], [128, 128])
    assert budget['allowed'] is False
    assert validate_detector_budget(budget, [1120, 2000], [128, 128]) == budget


def test_get_preprocess_failure_fails_closed():
    class Broken:
        def get_preprocess(self, max_wh):
            raise RuntimeError('synthetic unavailable configuration')
    budget = detector_input_budget(Broken(), [1024, 1024], [128, 128])
    assert budget['allowed'] is False
    assert budget['reason'] == 'preprocess_unavailable'
    assert validate_detector_budget(budget, [1024, 1024], [128, 128]) == budget


@pytest.mark.parametrize('shape', [[0, 16], [True, 16], [1.0, 16], [1], [16, 16, 0]])
def test_invalid_input_shape_fails_closed(shape):
    budget = detector_input_budget(ConfigDetector(), [1024, 1024], shape)
    assert budget['allowed'] is False and budget['reason'] == 'invalid_input_shape'


def test_rounding_to_zero_is_rejected_without_allocating_an_array():
    budget = detector_input_budget(ConfigDetector(736, 'max'), [1024, 1024], [1, 64])
    assert budget['roi_shape'] == [0, 64]
    assert budget['allowed'] is False and budget['reason'] == 'invalid_network_shape'
    assert validate_detector_budget(budget, [1024, 1024], [1, 64]) == budget


@pytest.mark.parametrize('mutation', ['allowed', 'reason', 'shape', 'pixels', 'config', 'extra'])
def test_pure_validator_rejects_tampered_capture(mutation):
    budget = detector_input_budget(ConfigDetector(736, 'max'), [1024, 1024], [128, 256])
    changed = deepcopy(budget)
    if mutation == 'allowed':
        changed['allowed'] = False
    elif mutation == 'reason':
        changed['reason'] = 'preprocess_unavailable'
    elif mutation == 'shape':
        changed['roi_shape'] = [64, 64]
    elif mutation == 'pixels':
        changed['roi_pixels'] += 1
    elif mutation == 'config':
        changed['roi_preprocess']['limit_side_len'] = 64
    else:
        changed['uncaptured'] = 1
    with pytest.raises(ValueError, match='detector network budget'):
        validate_detector_budget(changed, [1024, 1024], [128, 256])
    with pytest.raises(ValueError, match='detector network budget'):
        validate_detector_budget(budget, [1024, 1024], [64, 64])


@pytest.fixture(scope='module')
def upstream_preprocess():
    """Execute selected fixed upstream constructors/resize with tiny images."""
    configured_source = os.environ.get('VKP_RAPIDOCR_SOURCE')
    if not configured_source:
        pytest.skip('explicit VKP_RAPIDOCR_SOURCE required; no downloads')
    source_root = Path(configured_source)
    source = source_root / 'python/rapidocr/ch_ppocr_det/utils.py'
    if not source.is_file():
        pytest.skip('fixed local RapidOCR source unavailable; no downloads')
    content = source.read_bytes()
    # Source Inventory commit 7b2d368ac62f635094d2562a9e973f7b5eec083f.
    assert sha256(content).hexdigest() == 'f598961642ea9f9bd18fae4271a094a7eb53a2b3901b12c9a8e4c010c77b4699'
    tree = ast.parse(content.decode('utf-8-sig'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DetPreProcess')
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {'__init__', 'resize'}]
    assert len(cls.body) == 2
    selected = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
    namespace = {'np': np, 'cv2': cv2, 'ResizeImgError': RuntimeError}
    exec(compile(ast.fix_missing_locations(selected), str(source), 'exec'), namespace)
    return namespace['DetPreProcess']


@pytest.mark.parametrize('side,kind,full,roi', [
    (32, 'min', [57, 91], [17, 23]),
    (64, 'min', [128, 192], [35, 51]),
    (736, 'min', [80, 120], [40, 40]),
    (64, 'max', [128, 256], [32, 48]),
    (96, 'max', [113, 177], [81, 65]),
    (736, 'max', [80, 120], [40, 40]),
])
def test_prediction_matches_fixed_upstream_actual_resize(upstream_preprocess, side, kind, full, roi):
    budget = detector_input_budget(ConfigDetector(side, kind), full, roi)
    preprocess = upstream_preprocess(limit_side_len=side, limit_type=kind)
    full_image = preprocess.resize(np.zeros([*full, 3], dtype=np.uint8))
    roi_image = preprocess.resize(np.zeros([*roi, 3], dtype=np.uint8))
    assert budget['full_shape'] == list(full_image.shape[:2])
    assert budget['roi_shape'] == list(roi_image.shape[:2])
    assert validate_detector_budget(budget, full, roi) == budget
