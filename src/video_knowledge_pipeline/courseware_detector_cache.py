"""Bounded exact caching and opt-in local calls to the RapidOCR detector.

The input is already resized/padded by RapidOCR. Local detection is a review-
required optimization, not equivalence to full detection. No new OCR engine.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from .canonical_json import canonical_json_sha256
from .courseware_material import _check_frame, _page_id, read_scan, safe_path
from .courseware_ocr_geometry import _box
from .file_hash import sha256_file
from .stage_cache import StageCache

DISABLED = 'disabled'
EXACT_DETECTOR = 'exact_detector_input_v1'
CHANGED_REGIONS = 'changed_regions_v1'
DETECTOR_CACHE_POLICIES = (DISABLED, EXACT_DETECTOR, CHANGED_REGIONS)
SCHEMA = 'video_knowledge_pipeline.courseware_detector_cache.v1'
INPUT_SCHEMA = 'video_knowledge_pipeline.detector_input.v1'
EVENT_SCHEMA = 'video_knowledge_pipeline.detector_cache_event.v1'
LOCAL_EVENT_SCHEMA = 'video_knowledge_pipeline.detector_cache_event.v2'
EXECUTION_SCHEMA = 'video_knowledge_pipeline.local_detector_execution.v1'
MAX_CACHE_ROWS = 32
MAX_CACHE_BYTES = 4 * 1024 * 1024
MAX_BOXES = 512
MAX_PIXELS = 16_777_216
MAX_EVENTS = 512
_BOX_DTYPES = {'int32', 'int64', 'float32', 'float64'}
_FRAME_FIELDS = ('page_id', 'frame', 'frame_sha256')
_OUTPUT_FIELDS = ('boxes', 'scores', 'boxes_dtype')
_RECORD_FIELDS = ('key', 'source', 'input_identity', *_OUTPUT_FIELDS, 'output_sha256')


def _fail(reason):
    raise ValueError('courseware detector cache: ' + reason)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r'[a-f0-9]{64}', value) is not None


def _runtime_identity(runtime, adapter_sha256=None):
    if (not isinstance(runtime, dict) or not runtime.get('model_actual')
            or runtime.get('detector_cache_policy') not in {EXACT_DETECTOR, CHANGED_REGIONS}):
        _fail('explicit pinned runtime and detector policy required')
    adapter = adapter_sha256 or sha256_file(Path(__file__))
    if (not _digest(adapter) or runtime.get('detector_adapter_sha256', adapter) != adapter):
        _fail('detector adapter identity drift')
    return canonical_json_sha256({'runtime': runtime, 'adapter_sha256': adapter})


def _binding(root, candidate):
    if not isinstance(candidate, dict):
        _fail('source/target frame identity required')
    value = {name: candidate.get(name) for name in _FRAME_FIELDS}
    _page_id(value['page_id'])
    if not isinstance(value['frame'], str) or not _digest(value['frame_sha256']):
        _fail('source/target frame identity malformed')
    rows = [row for row in read_scan(root)['candidates'] if row.get('page_id') == value['page_id']]
    if len(rows) != 1 or any(rows[0].get(name) != value[name] for name in _FRAME_FIELDS):
        _fail('source/target scan identity drift')
    _check_frame(root, value)
    return value


def _input_identity(image, runtime_sha256, adapter_sha256):
    import numpy as np
    if (not isinstance(image, np.ndarray) or image.dtype.name != 'uint8'
            or image.ndim != 3 or image.shape[2] != 3
            or not 0 < image.shape[0] * image.shape[1] <= MAX_PIXELS):
        _fail('detector input shape/dtype/pixel budget')
    value = {'schema': INPUT_SCHEMA, 'runtime_sha256': runtime_sha256, 'adapter_sha256': adapter_sha256,
             'shape': list(image.shape), 'dtype': 'uint8',
             'pixels_sha256': sha256(image.tobytes()).hexdigest()}
    return {**value, 'input_sha256': canonical_json_sha256(value)}


def _validate_input(value, runtime_sha256):
    if not isinstance(value, dict):
        _fail('input identity missing')
    shape = value.get('shape')
    if (set(value) != {'schema', 'runtime_sha256', 'adapter_sha256', 'shape', 'dtype', 'pixels_sha256', 'input_sha256'}
            or value.get('schema') != INPUT_SCHEMA or value.get('runtime_sha256') != runtime_sha256
            or value.get('dtype') != 'uint8' or not isinstance(shape, list) or len(shape) != 3
            or any(type(n) is not int or n < 1 for n in shape) or shape[2] != 3
            or shape[0] * shape[1] > MAX_PIXELS or not _digest(value.get('pixels_sha256'))
            or not _digest(value.get('adapter_sha256'))):
        _fail('input identity shape/dtype/runtime/budget')
    if value.get('input_sha256') != canonical_json_sha256({k: v for k, v in value.items() if k != 'input_sha256'}):
        _fail('input identity hash drift')
    return shape


def _validate_output(value, shape):
    boxes, scores, dtype = (value.get(key) for key in _OUTPUT_FIELDS)
    if not isinstance(boxes, list) or not isinstance(scores, list) or not len(boxes) == len(scores) <= MAX_BOXES:
        _fail('detector boxes/scores count or budget')
    if (boxes and (not isinstance(dtype, str) or dtype not in _BOX_DTYPES)) or (not boxes and dtype is not None):
        _fail('detector boxes dtype')
    for box, score in zip(boxes, scores):
        _box(box, shape[1], shape[0])
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            _fail('detector score invalid')
        if dtype in {'int32', 'int64'} and any(type(n) is not int for point in box for n in point):
            _fail('detector integer box dtype mismatch')
    expected = canonical_json_sha256({key: value[key] for key in _OUTPUT_FIELDS})
    if value.get('output_sha256') != expected:
        _fail('detector output hash drift')


def _capture_output(output, shape):
    import numpy as np
    raw_boxes, raw_scores = getattr(output, 'boxes', None), getattr(output, 'scores', None)
    if raw_boxes is None:
        boxes, dtype = [], None
    else:
        array = np.asarray(raw_boxes)
        if array.size == 0:
            boxes, dtype = [], None
        elif array.ndim != 3 or array.shape[1:] != (4, 2) or array.shape[0] > MAX_BOXES:
            _fail('detector box shape or budget')
        else:
            boxes, dtype = array.tolist(), array.dtype.name
    scores = raw_scores.tolist() if hasattr(raw_scores, 'tolist') else raw_scores
    scores = list(scores) if isinstance(scores, (list, tuple)) else ([] if scores is None else scores)
    value = {'boxes': boxes, 'scores': scores, 'boxes_dtype': dtype}
    value['output_sha256'] = canonical_json_sha256(value)
    _validate_output(value, shape)
    return value


def _validate_record(root, row, runtime_sha256, *, check_source):
    if not isinstance(row, dict) or set(row) != {*_RECORD_FIELDS, 'content_sha256'}:
        _fail('detector record fields')
    shape = _validate_input(row['input_identity'], runtime_sha256)
    if row['key'] != row['input_identity']['input_sha256']:
        _fail('detector record input hash drift')
    _validate_output(row, shape)
    if row['content_sha256'] != canonical_json_sha256({key: row[key] for key in _RECORD_FIELDS}):
        _fail('detector record content hash drift')
    source = row['source']
    if (not isinstance(source, dict) or set(source) != set(_FRAME_FIELDS)
            or not isinstance(source.get('frame'), str) or not _digest(source.get('frame_sha256'))):
        _fail('detector record source identity')
    _page_id(source.get('page_id'))
    if check_source:
        _binding(root, row['source'])


def _record(source, identity, output):
    row = {'key': identity['input_sha256'], 'source': deepcopy(source),
           'input_identity': deepcopy(identity), **deepcopy(output)}
    return {**row, 'content_sha256': canonical_json_sha256(row)}


def _output(factory, image, row, elapse=0.0):
    import numpy as np
    boxes = np.array(row['boxes'], dtype=row['boxes_dtype'], copy=True) if row['boxes'] else None
    return factory(img=image, boxes=boxes, scores=list(row['scores']) if row['boxes'] else None, elapse=elapse)


class DetectorResultCache:
    def __init__(self, root, runtime_identity, detect, output_factory):
        self.root = Path(root).absolute()
        self.adapter_sha256 = sha256_file(Path(__file__))
        self.identity = _runtime_identity(runtime_identity, self.adapter_sha256)
        self.policy = runtime_identity['detector_cache_policy']
        self.runtime = deepcopy(runtime_identity)
        if self.policy == CHANGED_REGIONS:
            for field, filename in (('local_detection_adapter_sha256','courseware_local_detection.py'),
                                    ('detector_budget_adapter_sha256','courseware_detector_budget.py')):
                if runtime_identity.get(field) != sha256_file(Path(__file__).with_name(filename)):
                    _fail('local detector helper adapter identity drift')
        self.anchor = None  # One full-result anchor only; never a chain of partial pages.
        if not callable(detect) or not callable(output_factory):
            _fail('detector and output factory required')
        self.detect, self.output_factory = detect, output_factory
        self.cache = StageCache(safe_path(self.root, 'courseware/incremental-cache', exists=False),
                                safe_path(self.root, 'manifest.json'))
        self.rows = OrderedDict()
        self._check_paths()
        payload = self.cache.load_json('detector', 'inputs.json', identity=self.identity)
        if payload is not None:
            if (not isinstance(payload, dict) or set(payload) != {'schema', 'rows', 'sha256'}
                    or payload['schema'] != SCHEMA or not isinstance(payload['rows'], list)
                    or len(payload['rows']) > MAX_CACHE_ROWS):
                _fail('detector cache row budget/schema')
            if payload['sha256'] != canonical_json_sha256(payload['rows']):
                _fail('detector cache payload hash drift')
            for row in payload['rows']:
                _validate_record(self.root, row, self.identity, check_source=False)
                if not row['boxes'] or row['key'] in self.rows:
                    _fail('empty/duplicate detector cache entry')
                self.rows[row['key']] = row
        self.begin_frame(None)

    def __getattr__(self, name):
        return getattr(self.detect, name)

    def _check_paths(self):
        for name in ('inputs.json', 'manifest.json'):
            path = safe_path(self.root, f'courseware/incremental-cache/detector/{name}', exists=False)
            limit = MAX_CACHE_BYTES if name == 'inputs.json' else min(MAX_CACHE_BYTES, 32768)
            if path.exists() and (not path.is_file() or path.stat().st_size > limit):
                _fail('detector cache JSON byte budget')

    def _checkpoint(self):
        while self.rows:
            rows = list(self.rows.values())
            payload = {'schema': SCHEMA, 'rows': rows, 'sha256': canonical_json_sha256(rows)}
            if (len(rows) <= MAX_CACHE_ROWS
                    and len(json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8')) <= MAX_CACHE_BYTES):
                self._check_paths()
                self.cache.store_json('detector', 'inputs.json', payload, identity=self.identity)
                return
            self.rows.popitem(last=False)
        _fail('detector cache single record byte budget')

    def begin_frame(self, candidate):
        self.candidate = _binding(self.root, candidate) if candidate is not None else None
        self.events = []
        self.detector_calls = 0
        self.reused_inputs = 0
        self.execution = None
        self.local_frames = 0
        self.local_fallbacks = 0
        self.network_budget = None
        self.frame_inputs = 0

    def __call__(self, image):
        self.frame_inputs += 1
        if self.policy == CHANGED_REGIONS:
            return self._local_call(image)
        return self._detect_input(image)

    def _detect_input(self, image, *, scope='full'):
        if self.candidate is None or len(self.events) >= MAX_EVENTS:
            _fail('bound current frame and detector event budget required')
        target = _binding(self.root, self.candidate)
        identity = _input_identity(image, self.identity, self.adapter_sha256)
        key = identity['input_sha256']
        row = self.rows.get(key) if scope == 'full' else None
        elapse = 0.0
        if row is not None:
            _validate_record(self.root, row, self.identity, check_source=True)
            if row['input_identity'] != identity:
                _fail('detector input identity drift')
            origin = 'cache_reuse'
            result = _output(self.output_factory, image, row)
            self.rows.move_to_end(key)
            self._checkpoint()  # Hits alone must persist LRU touches.
            self.reused_inputs += 1
        else:
            self.detector_calls += 1
            raw = self.detect(image)
            if _input_identity(image, self.identity, self.adapter_sha256) != identity:
                _fail('detector mutated input pixels')
            _binding(self.root, target)
            output = _capture_output(raw, identity['shape'])
            row = _record(target, identity, output)
            origin = 'detector_call'
            elapse = getattr(raw, 'elapse', 0.0)
            if type(elapse) not in (int, float) or not math.isfinite(elapse) or elapse < 0:
                _fail('detector elapsed time invalid')
            result = _output(self.output_factory, image, row, elapse)
            if row['boxes'] and scope == 'full':
                self.rows[key] = row
                self._checkpoint()
        value = {'schema': EVENT_SCHEMA, 'event_index': len(self.events), 'policy': self.policy,
                 'origin': origin, 'target': deepcopy(target), 'source': deepcopy(row['source']),
                 'input_identity': deepcopy(identity),
                 **{name: deepcopy(row[name]) for name in (*_OUTPUT_FIELDS, 'output_sha256')},
                 'record_sha256': row['content_sha256']}
        if self.policy == CHANGED_REGIONS:
            value.update(schema=LOCAL_EVENT_SCHEMA, scope=scope)
        self.events.append({**value, 'event_sha256': canonical_json_sha256(value)})
        return result

    def _local_call(self, image):
        from .courseware_local_detection import (
            LocalDetectionFallback, plan_changed_region, merge_region_output,
        )
        if self.candidate is None or self.events:
            _fail('local detector requires one hook call per bound frame')
        identity = _input_identity(image, self.identity, self.adapter_sha256)
        target = _binding(self.root, self.candidate)
        plan, anchor_events = None, []
        mode, reason = 'full', 'no_anchor'
        if identity['input_sha256'] in self.rows:
            reason = 'exact_full_input_available'
        elif self.anchor is not None:
            anchor_image, anchor_events = self.anchor
            validate_detector_events(self.root, anchor_events, self.runtime)
            anchor_output = _event_output(anchor_events[-1])
            plan = plan_changed_region(anchor_image, image, anchor_output)
            reason = plan['reason']
            if plan['mode'] == 'roi':
                x0, y0, x1, y1 = plan['roi']
                from .courseware_detector_budget import detector_input_budget
                self.network_budget = detector_input_budget(self.detect,identity['shape'],[y1-y0,x1-x0,3])
                if not self.network_budget['allowed']:
                    reason = self.network_budget['reason']
                    result = self._detect_input(image)
                    output = _event_output(self.events[-1])
                    self._execution('full',reason,target,identity,plan,anchor_events,output)
                    self.anchor = (image.copy(),deepcopy(self.events)) if output['boxes'] else None
                    return result
                cropped = image[y0:y1, x0:x1].copy()
                roi_result = self._detect_input(cropped, scope='roi')
                try:
                    merged = merge_region_output(anchor_output, _event_output(self.events[-1]),
                        plan['roi'], identity['shape'], getattr(self.detect, 'sorted_boxes', None))
                except LocalDetectionFallback as exc:
                    if exc.reason not in {'empty_roi_output','roi_box_near_crop_edge','merged_box_budget',
                                          'merged_coordinate_precision','sorted_boxes_failed'}:
                        raise  # Corrupt geometry/sorter output is not ordinary ROI insufficiency.
                    mode, reason = 'fallback_full', exc.reason
                    self.local_fallbacks += 1
                else:
                    if _input_identity(image, self.identity, self.adapter_sha256) != identity:
                        _fail('current full detector input drift')
                    _binding(self.root, target)
                    # Current full image still goes through upstream cropping and
                    # recognition; only box discovery is localized.
                    result = _output(self.output_factory, image, merged, roi_result.elapse)
                    self.local_frames += 1
                    self._execution('roi', reason, target, identity, plan, anchor_events, merged)
                    return result
        result = self._detect_input(image)
        full_output = _event_output(self.events[-1])
        self._execution(mode, reason, target, identity, plan, anchor_events, full_output)
        # Save before upstream's final mapping mutates boxes. An empty full result
        # cannot authorize local reuse. No additional frame copies go to disk.
        self.anchor = (image.copy(), deepcopy(self.events)) if full_output['boxes'] else None
        return result

    def _execution(self, mode, reason, target, identity, plan, anchor_events, output):
        value = {'schema':EXECUTION_SCHEMA, 'policy':CHANGED_REGIONS,
                 'mode':mode, 'reason':reason, 'target':deepcopy(target),
                 'input_identity':deepcopy(identity), 'plan':deepcopy(plan),
                 'anchor_events':deepcopy(anchor_events), 'output':deepcopy(output),
                 'network_budget':deepcopy(self.network_budget),
                 'events_sha256':canonical_json_sha256(self.events),
                 'full_detector_equivalence':False, 'independent_review_complete':False}
        self.execution = {**value, 'execution_sha256':canonical_json_sha256(value)}


def _event_output(event):
    return {name:deepcopy(event[name]) for name in (*_OUTPUT_FIELDS, 'output_sha256')}


@contextmanager
def attach_detector_cache(engine, root, runtime_identity, output_factory=None):
    policy = runtime_identity.get('detector_cache_policy', DISABLED) if isinstance(runtime_identity, dict) else None
    if policy not in DETECTOR_CACHE_POLICIES:
        _fail('unsupported detector cache policy')
    if policy == DISABLED:
        yield None
        return
    if output_factory is None:
        from rapidocr.ch_ppocr_det import TextDetOutput
        output_factory = TextDetOutput
    original = engine.text_det
    cache = DetectorResultCache(root, runtime_identity, original, output_factory)
    engine.text_det = cache
    try:
        yield cache
    finally:
        engine.text_det = original


def validate_detector_events(root, events, runtime_identity):
    """Validate recorded identities; this does not re-run image preprocessing.

Both frames are bound to the current scan and their file bytes are rehashed.
Input/output hashes prove receipt consistency, not independent pixel review.
"""
    root = Path(root).absolute()
    if not isinstance(events, list) or len(events) > MAX_EVENTS:
        _fail('detector events type/budget')
    policy = runtime_identity.get('detector_cache_policy', DISABLED) if isinstance(runtime_identity, dict) else None
    if policy not in DETECTOR_CACHE_POLICIES or (events and policy == DISABLED):
        _fail('detector event policy mismatch')
    metrics = {'detector_calls': 0, 'reused_inputs': 0, 'detector_inputs': len(events)}
    if not events:
        return metrics
    first_input = events[0].get('input_identity') if isinstance(events[0], dict) else None
    captured_adapter = first_input.get('adapter_sha256') if isinstance(first_input, dict) else None
    if not _digest(captured_adapter):
        _fail('captured detector adapter identity missing')
    runtime_sha256 = _runtime_identity(runtime_identity, captured_adapter)
    fields = {'schema', 'event_index', 'policy', 'origin', 'source', 'target', 'input_identity',
              *_OUTPUT_FIELDS, 'output_sha256', 'record_sha256', 'event_sha256'}
    if policy == CHANGED_REGIONS:
        fields.add('scope')
    for index, event in enumerate(events):
        schema = LOCAL_EVENT_SCHEMA if policy == CHANGED_REGIONS else EVENT_SCHEMA
        if (not isinstance(event, dict) or set(event) != fields or event['schema'] != schema
                or type(event['event_index']) is not int or event['event_index'] != index
                or event['policy'] != policy or event['origin'] not in {'detector_call', 'cache_reuse'}):
            _fail('detector event fields/origin/index/policy')
        if policy == CHANGED_REGIONS and (event['scope'] not in {'full','roi'}
                or (event['scope'] == 'roi' and event['origin'] != 'detector_call')):
            _fail('local detector event scope/origin')
        if event['event_sha256'] != canonical_json_sha256({k: v for k, v in event.items() if k != 'event_sha256'}):
            _fail('detector event hash drift')
        _binding(root, event['target'])
        if (not isinstance(event['input_identity'], dict)
                or event['input_identity'].get('adapter_sha256') != captured_adapter):
            _fail('detector event adapter identity mismatch')
        row = {'key': event['input_identity'].get('input_sha256') if isinstance(event['input_identity'], dict) else None,
               'source': event['source'], 'input_identity': event['input_identity'],
               **{name: event[name] for name in (*_OUTPUT_FIELDS, 'output_sha256')},
               'content_sha256': event['record_sha256']}
        _validate_record(root, row, runtime_sha256, check_source=True)
        if event['origin'] == 'detector_call':
            if event['source'] != event['target']:
                _fail('fresh detector source/target mismatch')
            metrics['detector_calls'] += 1
        else:
            if not event['boxes']:
                _fail('empty detector output cannot authorize reuse')
            metrics['reused_inputs'] += 1
    return metrics


def validate_detector_execution(root, events, execution, runtime_identity, target):
    """Verify local merge conservation and receipt identities, not OCR recall.

    Pixel equality is captured at execution, not claimed as an independent
    re-run of RapidOCR preprocessing. Only full anchors are accepted.
    """
    from .courseware_local_detection import validate_roi_plan, merge_region_output
    fields = {'schema','policy','mode','reason','target','input_identity','plan','anchor_events',
              'output','events_sha256','network_budget','full_detector_equivalence','independent_review_complete','execution_sha256'}
    if (not isinstance(execution,dict) or set(execution) != fields
            or execution['schema'] != EXECUTION_SCHEMA or execution['policy'] != CHANGED_REGIONS
            or execution['mode'] not in {'full','roi','fallback_full'}
            or execution['target'] != target or not isinstance(execution['reason'],str)
            or not 1 <= len(execution['reason']) <= 128
            or execution['full_detector_equivalence'] is not False
            or execution['independent_review_complete'] is not False):
        _fail('local detector execution fields/identity')
    if (execution['execution_sha256'] != canonical_json_sha256({k:v for k,v in execution.items() if k!='execution_sha256'})
            or execution['events_sha256'] != canonical_json_sha256(events)):
        _fail('local detector execution hash drift')
    metrics = validate_detector_events(root,events,runtime_identity)
    if not 1 <= len(events) <= 2 or any(e['target'] != target for e in events):
        _fail('local detector call bound/target mismatch')
    current = execution['input_identity']
    runtime_sha = _runtime_identity(runtime_identity,events[0]['input_identity']['adapter_sha256'])
    shape = _validate_input(current,runtime_sha)
    if current['adapter_sha256'] != events[0]['input_identity']['adapter_sha256']:
        _fail('local detector current adapter identity drift')
    _validate_output(execution['output'],shape)
    anchors, plan = execution['anchor_events'], execution['plan']
    if not isinstance(anchors,list) or len(anchors) > 2:
        _fail('local detector anchor bound')
    if anchors:
        validate_detector_events(root,anchors,runtime_identity)
        if (anchors[-1]['scope'] != 'full' or not anchors[-1]['boxes']
                or [a['scope'] for a in anchors] not in [['full'],['roi','full']]
                or anchors[-1]['input_identity']['runtime_sha256'] != runtime_sha
                or anchors[-1]['input_identity']['adapter_sha256'] != current['adapter_sha256']
                or any(a['target'] != anchors[-1]['target'] for a in anchors)):
            _fail('local detector requires full anchor')
    if plan is not None:
        if not anchors:
            _fail('local detector plan without anchor')
        validate_roi_plan(plan,shape,_event_output(anchors[-1]))
    budget = execution['network_budget']
    if budget is not None:
        if not isinstance(plan,dict) or plan.get('mode') != 'roi':
            _fail('local detector network budget without ROI plan')
        from .courseware_detector_budget import validate_detector_budget
        x0,y0,x1,y1 = plan['roi']
        validate_detector_budget(budget,shape,[y1-y0,x1-x0,3])
    mode = execution['mode']
    if mode in {'roi','fallback_full'}:
        if not anchors or anchors[-1]['input_identity']['shape'] != shape:
            _fail('local detector compatible anchor missing')
        validate_roi_plan(plan,shape,_event_output(anchors[-1]))
        if plan['mode'] != 'roi' or events[0]['scope'] != 'roi':
            _fail('local detector ROI plan/call mismatch')
        x0,y0,x1,y1 = plan['roi']
        if events[0]['input_identity']['shape'] != [y1-y0,x1-x0,3]:
            _fail('local detector ROI shape drift')
        if (not isinstance(budget,dict) or budget.get('allowed') is not True
                or type(budget.get('full_pixels')) is not int or type(budget.get('roi_pixels')) is not int
                or not 0 < budget['roi_pixels'] < budget['full_pixels']
                or budget['roi_pixels'] > MAX_PIXELS):
            _fail('local detector network input budget')
    if mode == 'roi':
        if len(events) != 1 or execution['reason'] != plan['reason']:
            _fail('local detector merged call count')
        # Reapply the merge while preserving the recorded order, verifying the
        # exact box multiset and its score mapping rather than implementing a sorter.
        import numpy as np
        def recorded_order(boxes):
            return np.asarray(execution['output']['boxes'],dtype=boxes.dtype)
        expected = merge_region_output(_event_output(anchors[-1]),_event_output(events[0]),
                                       plan['roi'],shape,recorded_order)
        if expected != execution['output']:
            _fail('local detector merge conservation drift')
    else:
        if (events[-1]['scope'] != 'full' or events[-1]['input_identity'] != current
                or execution['output'] != _event_output(events[-1])
                or len(events) != (2 if mode == 'fallback_full' else 1)):
            _fail('local detector full fallback output mismatch')
        if mode == 'fallback_full':
            if execution['reason'] not in {'empty_roi_output','roi_box_near_crop_edge','merged_box_budget',
                                            'merged_coordinate_precision','sorted_boxes_failed'}:
                _fail('local detector fallback reason mismatch')
            from .courseware_local_detection import LocalDetectionFallback
            try:
                merge_region_output(_event_output(anchors[-1]),_event_output(events[0]),plan['roi'],shape,
                                    lambda boxes:boxes.copy())
            except LocalDetectionFallback as exc:
                if exc.reason != execution['reason']:
                    _fail('local detector fallback reason drift')
            else:
                if execution['reason'] != 'sorted_boxes_failed':
                    _fail('local detector fallback without recorded insufficiency')
        elif plan is not None:
            expected_reason = (plan['reason'] if plan['mode']=='full' else
                               budget['reason'] if isinstance(budget,dict) and not budget['allowed'] else None)
            if execution['reason'] != expected_reason:
                _fail('local detector full plan/budget reason mismatch')
        elif (anchors or budget is not None or execution['reason'] not in {'no_anchor','exact_full_input_available'}
              or (execution['reason']=='exact_full_input_available') != (events[-1]['origin']=='cache_reuse')):
            _fail('local detector initial/exact reason mismatch')
    return {**metrics,'detector_frame_inputs':1,'local_detector_frames':int(mode=='roi'),
            'local_detector_fallbacks':int(mode=='fallback_full')}
