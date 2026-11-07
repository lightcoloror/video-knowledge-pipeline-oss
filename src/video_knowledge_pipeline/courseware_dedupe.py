"""Read-only OCR selection plans over existing, hash-bound native screenshots.

Pillow supplies image differences; existing temporal metrics supply the cheap
comparison. Only complete RGB-pixel equality is an automatic reuse proof.
Near matches and cropped/masked equality are review suggestions, never facts.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

from .canonical_json import canonical_json_sha256
from .courseware_material import _check_frame, read_scan, safe_path
from .file_hash import sha256_file
from .temporal_frame_preprocess import _temporal_image_metrics

SCHEMA = 'video_knowledge_pipeline.courseware_dedupe_plan.v1'
STATE_SCHEMA = 'video_knowledge_pipeline.courseware_dedupe_plan.v2'
VISUAL_IDENTITY_SCHEMA = 'video_knowledge_pipeline.courseware_frame_visual_identity.v2'
MAX_FRAMES = 256
MAX_PIXELS = 16_777_216


def _render_metadata(image):
    """Return only metadata that can change how decoded pixels are displayed.

    Free-form PNG comments and compression settings are intentionally excluded.
    ICC, gamma/sRGB/chromaticity and EXIF orientation remain identity inputs.
    """
    info = image.info
    profile = info.get('icc_profile')
    if isinstance(profile, str):
        profile = profile.encode('utf-8')
    chromaticity = info.get('chromaticity')
    if isinstance(chromaticity, tuple):
        chromaticity = list(chromaticity)
    try:
        orientation = image.getexif().get(274)
    except (AttributeError, OSError, ValueError):
        orientation = None
    return {
        'format': image.format,
        'icc_profile_sha256': hashlib.sha256(profile).hexdigest() if isinstance(profile, bytes) else None,
        'gamma': info.get('gamma'),
        'srgb': info.get('srgb'),
        'chromaticity': chromaticity,
        'exif_orientation': orientation,
    }


def frame_visual_identity(bundle_dir, candidate):
    """Hash all decoded RGBA pixels plus rendering-relevant metadata.

    This is stronger than container-byte equality for safe OCR reuse while
    remaining exact: no perceptual threshold, resize, crop or colour tolerance.
    """
    from PIL import Image
    root = Path(bundle_dir).absolute()
    path = _check_frame(root, candidate)
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('frame byte budget exceeded')
    with Image.open(path) as image:
        if image.width * image.height > MAX_PIXELS or getattr(image, 'n_frames', 1) != 1:
            raise ValueError('static frame pixel budget exceeded')
        image.load()
        rgba = image.convert('RGBA')
        metadata = _render_metadata(image)
        width, height = rgba.size
        pixels = hashlib.sha256(
            f'{width}x{height}:RGBA\n'.encode('ascii') + rgba.tobytes()
        ).hexdigest()
    rgba.close()
    _check_frame(root, candidate)  # detect modification during decode
    value = {
        'schema': VISUAL_IDENTITY_SCHEMA,
        'width': width,
        'height': height,
        'pixel_mode': 'RGBA',
        'pixel_sha256': pixels,
        'render_metadata': metadata,
    }
    return {**value, 'identity_sha256': canonical_json_sha256(value)}


def validate_frame_visual_identity(bundle_dir, candidate, claimed):
    if not isinstance(claimed, dict) or claimed.get('schema') != VISUAL_IDENTITY_SCHEMA:
        raise ValueError('invalid courseware frame visual identity')
    actual = frame_visual_identity(bundle_dir, candidate)
    if actual != claimed:
        raise ValueError('courseware frame visual identity drift')
    return actual


def _selection(state, indexes, limit):
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_FRAMES:
        raise ValueError('courseware dedupe limit must be 1..256')
    raw = indexes.split(',') if isinstance(indexes, str) else indexes
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError('explicit nonempty courseware indexes required')
    if any(isinstance(n, bool) or not str(n).strip().isdigit() for n in raw):
        raise ValueError('positive integer indexes required')
    selected = [int(n) for n in raw]
    if len(selected) > limit or len(set(selected)) != len(selected) or any(n < 1 or n > len(state['candidates']) for n in selected):
        raise ValueError('duplicate/out-of-range indexes or bounded limit exceeded')
    return sorted(selected, key=lambda n: state['candidates'][n-1]['time'])


def _identity(state, selected):
    fields = ('page_id', 'time', 'end', 'frame_time', 'frame', 'frame_sha256')
    return [{'index': n, **{k: state['candidates'][n-1].get(k) for k in fields}} for n in selected]


def _load(root, candidate, roi):
    from PIL import Image
    path = _check_frame(root, candidate)
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('frame byte budget exceeded')
    with Image.open(path) as im:
        if im.width * im.height > MAX_PIXELS or getattr(im, 'n_frames', 1) != 1:
            raise ValueError('static frame pixel budget exceeded')
        rgb = im.convert('RGB')
    if roi is not None:
        if (not isinstance(roi, (list, tuple)) or len(roi) != 4
                or any(isinstance(v, bool) or not isinstance(v, int) for v in roi)):
            raise ValueError('roi requires four integer pixel coordinates')
        x1, y1, x2, y2 = roi
        if not 0 <= x1 < x2 <= rgb.width or not 0 <= y1 < y2 <= rgb.height:
            raise ValueError('roi outside native screenshot')
    visual_identity = frame_visual_identity(root, candidate)
    pixels = visual_identity['pixel_sha256']
    crop = rgb.crop(roi) if roi is not None else rgb
    region = hashlib.sha256(str(crop.size).encode() + crop.tobytes()).hexdigest()
    _check_frame(root, candidate)  # detect modification during decode
    return path, rgb, crop, pixels, region, visual_identity


def _change_regions(peak, *, origin=(0, 0), outside='none'):
    """Bounded OpenCV component projection of *every* nonzero RGB difference.

    This is a review locator, not a segmentation or cursor-exclusion model.
    Too much work returns the complete difference bbox, never a truncated set.
    """
    box = peak.getbbox()
    changed = peak.width * peak.height - peak.histogram()[0]
    def native(bounds):
        return [bounds[0] + origin[0], bounds[1] + origin[1],
                bounds[2] + origin[0], bounds[3] + origin[1]]
    value = {'schema':'video_knowledge_pipeline.courseware_change_regions.v1',
             'coordinate_space':'decoded_frame_pixels',
             'comparison_bounds_xyxy':native([0, 0, peak.width, peak.height]),
             'outside_comparison':outside, 'changed_pixel_count':changed,
             'status':'localized' if box else 'unchanged_in_comparison',
             'regions_xyxy':[], 'allows_ocr_skip':False,
             'pixel_threshold':0, 'max_regions':64, 'comparison_channels':'RGB'}
    if box is None:
        return value
    reason = None
    if changed > 65536:
        reason = 'changed_pixel_budget'
    elif (box[2] - box[0]) * (box[3] - box[1]) > 1_048_576:
        reason = 'component_pixel_budget'
    else:
        try:
            import cv2
            import numpy as np
        except ImportError:
            reason = 'component_backend_unavailable'
        else:
            with peak.crop(box) as crop:
                mask = (np.asarray(crop) != 0).astype(np.uint8)
            count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if count - 1 > value['max_regions']:
                reason = 'component_budget'
            else:
                # At most 64 rows: deterministic display ordering, no tree/image sort.
                regions = [[int(x)+box[0], int(y)+box[1], int(x+w)+box[0], int(y+h)+box[1]]
                           for x, y, w, h, _ in stats[1:]]
                value['regions_xyxy'] = [native(r) for r in sorted(regions, key=lambda r:(r[1],r[0]))]
    if reason:
        value.update(status='bbox_fallback', reason=reason, regions_xyxy=[native(box)])
    return value


def _difference(left, right, paths, *, roi=None):
    from PIL import ImageChops
    if left.size != right.size:
        return {'dimensions_changed': True, 'review_reason': 'native dimensions changed'}
    channels = ImageChops.difference(left, right).split()
    peak = ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2])
    hist = peak.histogram()
    total = left.width * left.height
    coarse, _ = _temporal_image_metrics([str(p) for p in paths])
    last = coarse[-1]
    if 'metric_error' in last or 'difference_rms_to_previous' not in last:
        raise ValueError('existing temporal metric unavailable; no approximate match assumed')
    changed = peak.point([0]*17 + [255]*239)
    box = changed.getbbox()
    # Reuse the existing overlapping tile geometry. No thresholding here:
    # even a one-level pixel change remains in the detail-review workload.
    from .high_res_tile_plan import _dynamic_tile_boxes
    tiles = []
    for tile_box in _dynamic_tile_boxes(left.width, left.height, tile_size=256,
                                         overlap=.12, max_tiles=12):
        with peak.crop(tile_box) as tile:
            if tile.getbbox() is not None:
                tiles.append(list(tile_box))
    return {'dimensions_changed': False, 'changed_pixels': total-hist[0],
            'changed_ratio_above_16': round(sum(hist[17:])/total, 9),
            'changed_pixels_above_64': sum(hist[65:]),
            'native_difference_bbox': list(box) if box else None,
            'exact_difference_bbox': list(peak.getbbox()) if peak.getbbox() else None,
            'changed_detail_tiles': tiles,
            'detail_tiles_are_review_only': True,
            'detail_tile_pixel_budget': sum((b[2]-b[0])*(b[3]-b[1]) for b in tiles),
            'coarse_rms': last['difference_rms_to_previous'],
            # Legacy bbox/tile fields use comparison-crop coordinates. The new
            # projection explicitly maps ROI offsets back to the native frame.
            'legacy_metrics_coordinate_space':'roi_pixels' if roi else 'decoded_frame_pixels',
            'change_regions':_change_regions(peak, origin=roi[:2] if roi else (0,0),
                                             outside='not_assessed' if roi else 'none')}


def plan_courseware_dedupe(bundle_dir, *, indexes, limit=MAX_FRAMES, roi=None):
    """Return an immutable bounded plan, without writing a queue or media.

    New scan captures outside the selected scope do not invalidate the plan;
    changes to any selected frame, manifest or timeline do. Revisit matching is
    global within this explicit selection, while near matching uses an anchor
    (not adjacent-frame chaining). Non-exact matches stay in safe_ocr_indexes.
    """
    root = Path(bundle_dir).absolute()
    state = read_scan(root)
    selected = _selection(state, indexes, limit)
    selected_identity = _identity(state, selected)
    artifacts = {name: sha256_file(safe_path(root, name)) for name in ('manifest.json', 'timeline.json')}
    rows, seen, anchor = [], {}, None
    for n in selected:
        candidate = state['candidates'][n-1]
        path, rgb, crop, pixels, region, visual_identity = _load(root, candidate, roi)
        item = {**selected_identity[selected.index(n)], 'pixel_sha256': pixels,
                'visual_identity_sha256': visual_identity['identity_sha256'],
                'roi_pixel_sha256': region, 'classification': 'new_content',
                'safe_to_reuse': False, 'representative_index': n,
                'ocr_required': True, 'evidence_level': 'comparison_candidate'}
        exact_key = visual_identity['identity_sha256']
        if exact_key in seen:
            item.update(classification='exact_duplicate', safe_to_reuse=True,
                        representative_index=seen[exact_key], ocr_required=False,
                        evidence_level='decoded_frame_identity_v2')
        elif anchor is not None:
            item['compared_to_index'] = anchor['index']
            item['metrics'] = _difference(anchor['crop'], crop, [anchor['path'], path], roi=roi)
            metrics = item['metrics']
            if region == anchor['region'] and roi is not None:
                item.update(classification='roi_exact_review_required', representative_index=anchor['index'])
            elif (not metrics['dimensions_changed'] and metrics['coarse_rms'] <= .01
                  and metrics['changed_ratio_above_16'] <= .01):
                item.update(classification='near_duplicate_review_required', representative_index=anchor['index'])
                if metrics['changed_pixels_above_64']:
                    item['local_change_warning'] = 'Small strong changes may be text, numbers, drawing or pointer; keep for OCR/review.'
            else:
                item['classification'] = 'content_change'
        rows.append(item)
        seen.setdefault(exact_key, n)
        if anchor is None or item['classification'] in {'new_content', 'content_change'}:
            if anchor is not None:
                anchor['crop'].close()
            anchor = {'index': n, 'path': path, 'crop': crop.copy(), 'region': region}
        rgb.close()
        if crop is not rgb:
            crop.close()
    if anchor is not None:
        anchor['crop'].close()
    current = read_scan(root)
    if current['identity'] != state['identity'] or _identity(current, selected) != selected_identity:
        raise ValueError('courseware selected candidate identity drift')
    for n in selected:
        _check_frame(root, current['candidates'][n-1])
    if any(sha256_file(safe_path(root, name)) != digest for name, digest in artifacts.items()):
        raise ValueError('courseware manifest/timeline drift')
    value = {'schema': SCHEMA, 'scan_identity': state['identity'], 'artifact_hashes': artifacts,
             'selected_indexes': selected, 'selected_identity_sha256': canonical_json_sha256(selected_identity),
             'roi': list(roi) if roi is not None else None, 'rows': rows,
             'counts': dict(Counter(r['classification'] for r in rows)),
             'safe_ocr_indexes': [r['index'] for r in rows if r['ocr_required']],
             'approximate_representative_indexes': sorted({r['representative_index'] for r in rows}),
             'queue_modified': False, 'source_video_read': False, 'model_calls': 0,
             'independent_review_complete': False, 'max_frames': limit,
             'implementation': ['Pillow RGB pixel hashing/ImageChops', 'existing _temporal_image_metrics'],
             'limitations': ['No semantic slide-equivalence claim.',
                 'Approximate and ROI matches NEVER authorize OCR skipping.',
                 'Exact reuse requires a separately validated successful representative OCR result.',
                 'All occurrence times and original screenshots remain; this is not an independent review.']}
    return {**value, 'plan_sha256': canonical_json_sha256(value)}


def plan_courseware_dedupe_v2(bundle_dir, *, indexes, limit=MAX_FRAMES, roi=None):
    """Add exact temporal-state persistence without widening automatic reuse.

    A-B-A marks B as a transient review candidate. A-B-B marks B as a stable
    new-state candidate. Both still require OCR unless exact decoded identity
    has already selected a representative.
    """
    prior = plan_courseware_dedupe(bundle_dir, indexes=indexes, limit=limit, roi=roi)
    rows = [dict(row) for row in prior['rows']]
    keys = [row['visual_identity_sha256'] for row in rows]
    keys_by_index = {row['index']:row['visual_identity_sha256'] for row in rows}
    for position, row in enumerate(rows):
        start = position
        while start and keys[start - 1] == keys[position]:
            start -= 1
        end = position
        while end + 1 < len(rows) and keys[end + 1] == keys[position]:
            end += 1
        persistence = end - start + 1
        row['state_persistence'] = persistence
        row['state_run_indexes'] = [rows[n]['index'] for n in range(start, end + 1)]
        approximate_support = []
        if row['classification'] == 'content_change':
            for later in rows[position + 1:]:
                if later.get('representative_index') != row['index']:
                    break
                approximate_support.append(later['index'])
        row['approximate_state_support_indexes'] = approximate_support
        if position > start:
            state = 'repeated_state_occurrence'
        elif position == 0:
            state = 'initial_state_candidate'
        elif persistence >= 2:
            state = 'stable_new_state_candidate'
        elif position + 1 < len(rows) and keys[position - 1] == keys[position + 1]:
            state = 'transient_change_review'
        elif approximate_support:
            state = 'persistent_visual_change_review'
        elif row['classification'] in {'near_duplicate_review_required', 'roi_exact_review_required'}:
            state = 'near_state_occurrence_review'
        else:
            state = 'unresolved_change_review'
        row['state_classification'] = state
        row['state_evidence_level'] = 'exact_decoded_sequence_observation'
        if state == 'transient_change_review':
            # Metrics compare to a stable anchor, not necessarily the previous
            # observation. Never label C-vs-B regions as A-vs-B in C,A,B,A.
            comparison_matches = keys_by_index.get(row.get('compared_to_index')) == keys[position-1]
            row['transient_change_review'] = {
                'classification':'possible_overlay_or_transient_content',
                'supporting_indexes':[rows[position-1]['index'], row['index'], rows[position+1]['index']],
                'cursor_confirmed':False, 'allows_ocr_skip':False,
                'reason':'Exact A-B-A capture sequence; a pointer, annotation or short-lived text may all explain B.',
                'localization_status':('anchor_matches_preceding_state' if comparison_matches
                                       else 'unavailable_comparison_anchor_differs'),
                'change_regions':row.get('metrics',{}).get('change_regions') if comparison_matches else None,
            }
    value = {
        **{key: val for key, val in prior.items() if key not in {'schema', 'rows', 'counts', 'plan_sha256'}},
        'schema': STATE_SCHEMA,
        'source_plan_sha256': prior['plan_sha256'],
        'rows': rows,
        'counts': dict(Counter(row['classification'] for row in rows)),
        'state_counts': dict(Counter(row['state_classification'] for row in rows)),
        'stable_state_indexes': [row['index'] for row in rows
                                 if row['state_classification'] in {'initial_state_candidate', 'stable_new_state_candidate'}],
        'persistent_visual_change_review_indexes': [row['index'] for row in rows
                                                    if row['state_classification'] == 'persistent_visual_change_review'],
        'near_state_review_indexes': [row['index'] for row in rows
                                      if row['state_classification'] == 'near_state_occurrence_review'],
        'transient_review_indexes': [row['index'] for row in rows
                                     if row['state_classification'] == 'transient_change_review'],
        'automatic_near_reuse': False,
        'state_analysis_only': True,
        'limitations': [*prior['limitations'],
            'Temporal persistence classifies captured observations only; it does not prove semantic page completeness.',
            'Transient and unresolved changes remain OCR/review candidates.'],
    }
    return {**value, 'plan_sha256': canonical_json_sha256(value)}
