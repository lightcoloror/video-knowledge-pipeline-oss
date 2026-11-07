"""Predict existing RapidOCR detector resize cost without allocating images.

The small resize arithmetic below adapts ``DetPreProcess.resize`` from
RapidAI/RapidOCR, python/rapidocr/ch_ppocr_det/utils.py, fixed source commit
7b2d368ac62f635094d2562a9e973f7b5eec083f (Apache-2.0). The live detector still
selects each preprocessing configuration through its own ``get_preprocess``.
This is a work-budget check, not evidence of equivalent detector predictions.
"""
# SPDX-License-Identifier: Apache-2.0
# Resize arithmetic: Copyright (c) 2020 PaddlePaddle Authors.
from __future__ import annotations

import math

SCHEMA = 'video_knowledge_pipeline.detector_network_budget.v1'
MAX_ROI_NETWORK_PIXELS = 16_777_216
_CONFIG_FIELDS = {'limit_side_len', 'limit_type'}
_NO_CONFIG_REASONS = {'preprocess_unavailable', 'unsupported_preprocess_config'}


def _input_shape(shape):
    if (not isinstance(shape, (list, tuple)) or len(shape) not in (2, 3)
            or any(type(n) is not int or n <= 0 for n in shape)
            or (len(shape) == 3 and shape[2] != 3)):
        return None
    return list(shape[:2])


def _valid_config(config):
    if not isinstance(config, dict) or set(config) != _CONFIG_FIELDS:
        return False
    side = config['limit_side_len']
    return (type(side) in (int, float) and 0 < side <= MAX_ROI_NETWORK_PIXELS
            and math.isfinite(side) and type(config['limit_type']) is str
            and config['limit_type'] in {'min', 'max'})


def _network_shape(shape, config):
    """Match upstream int-before-round-to-32 exactly, including ties to even."""
    h, w = shape
    side = config['limit_side_len']
    ratio = 1.0
    if config['limit_type'] == 'max':
        if max(h, w) > side:
            ratio = float(side) / (h if h > w else w)
    elif min(h, w) < side:
        ratio = float(side) / (h if h < w else w)
    resize_h, resize_w = int(h * ratio), int(w * ratio)
    return [int(round(resize_h / 32) * 32), int(round(resize_w / 32) * 32)]


def _empty(full, roi, reason):
    return {'schema': SCHEMA, 'allowed': False, 'reason': reason,
            'full_input_shape': full, 'roi_input_shape': roi,
            'full_preprocess': None, 'roi_preprocess': None,
            'full_shape': None, 'roi_shape': None,
            'full_pixels': 0, 'roi_pixels': 0}


def _with_config(full, roi, full_config, roi_config):
    budget = _empty(full, roi, 'invalid_network_shape')
    budget.update(full_preprocess=dict(full_config), roi_preprocess=dict(roi_config))
    try:
        full_network = _network_shape(full, full_config)
        roi_network = _network_shape(roi, roi_config)
    except (ArithmeticError, ValueError):
        return budget
    full_pixels = full_network[0] * full_network[1]
    roi_pixels = roi_network[0] * roi_network[1]
    budget.update(full_shape=full_network, roi_shape=roi_network,
                  full_pixels=full_pixels, roi_pixels=roi_pixels)
    if min(*full_network, *roi_network) <= 0:
        return budget
    if roi_pixels > MAX_ROI_NETWORK_PIXELS:
        budget['reason'] = 'roi_pixel_budget'
    elif roi_pixels >= full_pixels:
        budget['reason'] = 'roi_not_smaller'
    else:
        budget.update(allowed=True, reason='smaller_network_input')
    return budget


def detector_input_budget(detector, full_shape, roi_shape):
    """Capture upstream configs and permit only a strictly smaller bounded ROI.

    Arguments accept ``[height, width]`` or uint8 image shape ``[h, w, 3]``.
    Result ``full_shape``/``roi_shape`` are predicted network dimensions.
    No preprocessing, resizing, inference, or image allocation is performed.
    """
    full, roi = _input_shape(full_shape), _input_shape(roi_shape)
    if full is None or roi is None:
        return _empty(full, roi, 'invalid_input_shape')
    try:
        select = getattr(detector, 'get_preprocess', None)
        if not callable(select):
            return _empty(full, roi, 'preprocess_unavailable')
        configs = []
        for shape in (full, roi):
            preprocess = select(max(shape))
            configs.append({name: getattr(preprocess, name, None) for name in _CONFIG_FIELDS})
    except Exception:
        return _empty(full, roi, 'preprocess_unavailable')
    if not all(_valid_config(config) for config in configs):
        return _empty(full, roi, 'unsupported_preprocess_config')
    return _with_config(full, roi, *configs)


def validate_detector_budget(budget, full_input_shape, roi_input_shape):
    """Recompute a captured receipt without a live detector; return it or fail.

    Captured configs describe the producer's configuration observation. This
    validates their supported values and arithmetic, not a new runtime probe.
    Unavailable configuration can only authorize a full-detection fallback.
    """
    def fail():
        raise ValueError('detector network budget: captured configuration/shape/result drift')

    if not isinstance(budget, dict) or type(budget.get('allowed')) is not bool:
        fail()
    full, roi = _input_shape(full_input_shape), _input_shape(roi_input_shape)
    if full is None or roi is None:
        expected = _empty(full, roi, 'invalid_input_shape')
    else:
        configs = [budget.get('full_preprocess'), budget.get('roi_preprocess')]
        if all(_valid_config(config) for config in configs):
            expected = _with_config(full, roi, *configs)
        elif (all(config is None for config in configs)
              and type(budget.get('reason')) is str
              and budget.get('reason') in _NO_CONFIG_REASONS):
            expected = _empty(full, roi, budget['reason'])
        else:
            fail()
    # Dict equality alone treats True == 1 and 32.0 == 32 as equivalent.
    for key in ('full_pixels', 'roi_pixels'):
        if type(budget.get(key)) is not int:
            fail()
    for key in ('full_input_shape', 'roi_input_shape', 'full_shape', 'roi_shape'):
        shape = budget.get(key)
        if shape is not None and (not isinstance(shape, list) or len(shape) != 2
                                  or any(type(n) is not int for n in shape)):
            fail()
    if budget != expected:
        fail()
    return budget
