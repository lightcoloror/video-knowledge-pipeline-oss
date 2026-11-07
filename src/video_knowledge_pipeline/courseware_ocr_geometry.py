"""Thin RapidOCR final-output binding; no detector, OCR or coordinate remapper.

RapidOCR maps boxes back to the original input before returning final output.
Preserve those quads; ebook's rounded axis-aligned bbox is navigation only.
"""
from __future__ import annotations

import math

SCHEMA = 'video_knowledge_pipeline.courseware_ocr_geometry.v1'
CONTRACT = 'rapidocr_final_original_pixels.v1'


def _fail(message):
    raise ValueError('courseware_ocr_geometry:' + message)


def _rows(value):
    if value is None:
        return []
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) > 512:
        _fail('row_budget_or_type')
    return list(value)


def _box(quad, width, height):
    if (not isinstance(quad, list) or len(quad) != 4
            or any(not isinstance(p, list) or len(p) != 2 for p in quad)):
        _fail('native_quad_required')
    if any(type(n) not in (float, int) or not math.isfinite(n) for p in quad for n in p):
        _fail('nonfinite_quad')
    if any(not 0 <= x <= width or not 0 <= y <= height for x, y in quad):
        _fail('quad_outside_frame')
    # Existing OpenCV does geometry validation; do not implement a polygon engine.
    import cv2
    import numpy as np
    contour = np.asarray(quad, dtype=np.float32)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour) <= 0:
        _fail('degenerate_or_crossed_quad')
    return {'l': min(p[0] for p in quad), 't': min(p[1] for p in quad),
            'r': max(p[0] for p in quad), 'b': max(p[1] for p in quad), 'coord_origin': 'TOPLEFT'}


def capture_geometry(root, candidate, raw, blocks):
    """Capture before formal import; only the pinned final-output route calls us."""
    from .courseware_dedupe import frame_visual_identity
    from .courseware_incremental_ocr import _checked_text
    boxes = _rows(getattr(raw, 'boxes', None))
    texts = _rows(getattr(raw, 'txts', None))
    scores = _rows(getattr(raw, 'scores', None))
    if not len(boxes) == len(texts) == len(scores):
        _fail('parallel_output_count')
    identity = frame_visual_identity(root, candidate)
    image = getattr(raw, 'img', None)
    if texts and (image is None or list(image.shape[:2]) != [identity['height'], identity['width']]):
        _fail('original_image_dimensions_required')
    observations = []
    for index, (quad, text, score) in enumerate(zip(boxes, texts, scores)):
        _checked_text(text, score)
        _box(quad, identity['width'], identity['height'])
        observations.append({'index': index, 'text': text, 'score': score, 'quad': quad})
    binding = {key: candidate[key] for key in ('frame', 'frame_sha256')}
    binding.update(coordinate_space='decoded_frame_pixels', frame_visual_identity=identity)
    geometry = {'schema': SCHEMA, 'contract': CONTRACT, 'frame_binding': binding,
                'observations': observations}
    validate_geometry(root, {'ocr_geometry': geometry, 'frame_sha256': candidate['frame_sha256'],
                            'blocks': [{'text': b['text'], 'structure': b} for b in blocks]}, candidate)
    return geometry


def validate_geometry(root, page, candidate):
    from .courseware_structure import validate_frame_binding_identity
    from .courseware_incremental_ocr import _checked_text
    geometry = page.get('ocr_geometry')
    if not isinstance(geometry, dict) or set(geometry) != {'schema', 'contract', 'frame_binding', 'observations'}:
        _fail('geometry_schema')
    if geometry['schema'] != SCHEMA or geometry['contract'] != CONTRACT:
        _fail('geometry_contract')
    binding = geometry['frame_binding']
    if (not isinstance(binding, dict) or set(binding) != {'frame', 'frame_sha256', 'coordinate_space', 'frame_visual_identity'}
            or binding['coordinate_space'] != 'decoded_frame_pixels'
            or binding['frame'] != candidate.get('frame')
            or binding['frame_sha256'] != candidate.get('frame_sha256')
            or binding['frame_sha256'] != page.get('frame_sha256')):
        _fail('frame_binding')
    validate_frame_binding_identity(root, binding, candidate)
    identity = binding['frame_visual_identity']
    rows = _rows(geometry['observations'])
    for i, row in enumerate(rows):
        if (not isinstance(row, dict) or set(row) != {'index', 'text', 'score', 'quad'}
                or type(row['index']) is not int or row['index'] != i):
            _fail('observation_identity')
        _checked_text(row['text'], row['score'])
        _box(row['quad'], identity['width'], identity['height'])
    indices = []
    for block in page['blocks']:
        structure = block.get('structure') or {}
        i = structure.get('index')
        if type(i) is not int or not 1 <= i <= len(rows) or i in indices:
            _fail('normalized_index_identity')
        indices.append(i)
        if rows[i - 1]['text'].strip() != block['text'] or structure.get('text') != block['text']:
            _fail('normalized_text_identity')
        _checked_text(block['text'], structure.get('score'))
        if structure['score'] != round(rows[i - 1]['score'], 4):
            _fail('normalized_score_identity')
        box = _box(rows[i - 1]['quad'], identity['width'], identity['height'])
        expected = [round(box[k], 2) for k in ('l', 't', 'r', 'b')]
        bbox = structure.get('bbox')
        if (not isinstance(bbox, list) or len(bbox) != 4
                or any(type(n) not in (int, float) for n in bbox) or bbox != expected):
            _fail('normalized_bbox_identity')
        if any(k in block for k in ('ocr_evidence_context', 'frame_region')):
            _fail('derived_context_not_importable')
    if set(indices) != {r['index'] + 1 for r in rows if r['text'].strip()}:
        _fail('unmapped_observation')


def observation_region(geometry, observation):
    from .courseware_structure import project_frame_region
    binding = geometry['frame_binding']
    identity = binding['frame_visual_identity']
    box = _box(observation['quad'], identity['width'], identity['height'])
    return project_frame_region(binding, box)


def project_ocr_block(page, block, number, snapshot):
    geometry = page.get('ocr_geometry')
    if geometry is None:
        return {}
    index = block['structure']['index'] - 1
    return {'source_locators': [snapshot.locator(['ocr_geometry', 'observations', index]),
                               snapshot.locator(['ocr_geometry', 'frame_binding']),
                               snapshot.locator(['blocks', number])],
            'ocr_evidence_context': {'ocr_observation_index': index,
                'geometry_kind': 'original_quad_axis_aligned_navigation',
                'frame_region': observation_region(geometry, geometry['observations'][index])}}
