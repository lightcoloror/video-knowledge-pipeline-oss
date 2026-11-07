"""RapidOCR recognition memoization with optional exact detector-input reuse.

No OCR algorithm, fuzzy image matching, deletion, or second page writer. Reuse
the upstream recognize_txt/output contract and VKP StageCache/import machinery.
"""
from __future__ import annotations

from collections import OrderedDict, Counter
from contextlib import contextmanager
from hashlib import sha256
import json
import math
from pathlib import Path
from uuid import uuid4

from .canonical_json import canonical_json_sha256
from .courseware_material import _check_frame, _page_id, import_courseware_page, read_scan, safe_path
from .file_hash import sha256_file
from .stage_cache import StageCache
from .storage import bundle_write_lock, read_json, write_json

SCHEMA = 'video_knowledge_pipeline.courseware_incremental_ocr.v1'
MAX_REGIONS = 512
MAX_CACHE_ROWS = 1024
REGION_EVIDENCE_SCHEMA = 'video_knowledge_pipeline.incremental_region_evidence.v1'
EXACT_CROP = 'exact_current_crop'
UNIFORM_PADDING = 'uniform_padding_exact_v1'
CACHE_POLICIES = {EXACT_CROP, UNIFORM_PADDING}


def crop_cache_identity(image, *, runtime_sha256, method=EXACT_CROP):
    """Ignore only provably uniform outside padding, never alter recognizer input.

    The opt-in identity is translation/padding invariant only when every outside
    pixel has exactly one RGB value and the enclosed pixels/shape are identical.
    No threshold, resizing, colour normalization, blur or glyph erosion is used.
    JPEG/warp noise and clipped/border-touching text fall back to exact identity.
    """
    if not isinstance(method, str) or method not in CACHE_POLICIES:
        raise ValueError('unsupported recognition cache policy')
    if (len(image.shape) != 3 or image.shape[2] != 3 or image.dtype.name != 'uint8'
            or image.size > 12_000_000 or not image.size):
        raise ValueError('incremental OCR crop shape/dtype/budget')
    shape = list(image.shape)
    pixels = sha256(image.tobytes()).hexdigest()
    exact = canonical_json_sha256({'runtime':runtime_sha256,'shape':shape,
                                  'dtype':str(image.dtype),'pixels':pixels})
    proof = {'method':EXACT_CROP,'runtime_sha256':runtime_sha256,'shape':shape,
             'dtype':str(image.dtype),'pixels_sha256':pixels,
             'crop_sha256':exact,'cache_key_sha256':exact}
    if method == EXACT_CROP:
        return proof
    # NumPy already owns upstream RapidOCR crop arrays; no new image matcher.
    import numpy as np
    background = image[0, 0]
    if any(not np.all(edge == background) for edge in
           (image[0], image[-1], image[:, 0], image[:, -1])):
        return proof
    content = np.any(image != background, axis=2)
    rows, columns = np.flatnonzero(content.any(axis=1)), np.flatnonzero(content.any(axis=0))
    if not rows.size or not columns.size:
        return proof  # Uniform/blank extents are not reusable text identities.
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(columns[0]), int(columns[-1]) + 1
    interior = image[top:bottom, left:right]
    content_sha = sha256(interior.tobytes()).hexdigest()
    bg = [int(channel) for channel in background]
    key = canonical_json_sha256({'runtime':runtime_sha256,'method':UNIFORM_PADDING,
        'background':bg,'shape':list(interior.shape),'dtype':str(image.dtype),'pixels':content_sha})
    return {**proof,'method':UNIFORM_PADDING,'cache_key_sha256':key,
            'content_box':[left,top,right,bottom],'background':bg,'content_sha256':content_sha}


def _validate_crop_identity(proof):
    """Validate the serialized identity recipe without inventing pixel evidence."""
    from .evidence_lineage import _hash
    if not isinstance(proof, dict):
        raise ValueError('incremental crop identity missing')
    shape = proof.get('shape')
    if (not isinstance(shape, list) or len(shape) != 3 or shape[2] != 3
            or any(type(n) is not int or n <= 0 for n in shape)
            or math.prod(shape) > 12_000_000 or proof.get('dtype') != 'uint8'
            or any(not _hash(proof.get(k)) for k in ('runtime_sha256','pixels_sha256',
                                                    'crop_sha256','cache_key_sha256'))):
        raise ValueError('incremental crop identity shape/hash')
    exact = canonical_json_sha256({'runtime':proof['runtime_sha256'],'shape':shape,
                                  'dtype':proof['dtype'],'pixels':proof['pixels_sha256']})
    key = exact
    if proof.get('method') == UNIFORM_PADDING:
        box, bg = proof.get('content_box'), proof.get('background')
        if (not isinstance(box,list) or len(box) != 4 or any(type(n) is not int for n in box)
                or not 0 < box[0] < box[2] < shape[1] or not 0 < box[1] < box[3] < shape[0]
                or not isinstance(bg,list) or len(bg) != 3
                or any(type(n) is not int or not 0 <= n <= 255 for n in bg)
                or not _hash(proof.get('content_sha256'))):
            raise ValueError('incremental crop identity padding proof')
        key = canonical_json_sha256({'runtime':proof['runtime_sha256'],'method':UNIFORM_PADDING,
            'background':bg,'shape':[box[3]-box[1],box[2]-box[0],3],
            'dtype':proof['dtype'],'pixels':proof['content_sha256']})
    elif proof.get('method') != EXACT_CROP:
        raise ValueError('incremental crop identity method')
    if exact != proof['crop_sha256'] or key != proof['cache_key_sha256']:
        raise ValueError('incremental crop identity hash drift')


def incremental_run_id(scan_identity, indexes, runtime_identity):
    """One identity formula shared by single and batched checkpoint callers."""
    runtime_identity = _detector_runtime(runtime_identity)
    return canonical_json_sha256({'scan':scan_identity,'indexes':indexes,'runtime':runtime_identity,
                                 'adapter_sha256':sha256_file(Path(__file__))})[:24]


def _detector_runtime(runtime_identity):
    from .courseware_detector_cache import CHANGED_REGIONS, DISABLED, DETECTOR_CACHE_POLICIES
    if not isinstance(runtime_identity, dict):
        raise ValueError('pinned runtime identity required')
    detector_policy = runtime_identity.get('detector_cache_policy', DISABLED)
    if not isinstance(detector_policy, str) or detector_policy not in DETECTOR_CACHE_POLICIES:
        raise ValueError('unsupported detector cache policy')
    runtime = dict(runtime_identity)
    if detector_policy != DISABLED:
        adapter_sha = sha256_file(Path(__file__).with_name('courseware_detector_cache.py'))
        if runtime.get('detector_adapter_sha256', adapter_sha) != adapter_sha:
            raise ValueError('detector adapter identity drift')
        runtime['detector_adapter_sha256'] = adapter_sha
    if detector_policy == CHANGED_REGIONS:
        for field, filename in (('local_detection_adapter_sha256','courseware_local_detection.py'),
                                ('detector_budget_adapter_sha256','courseware_detector_budget.py')):
            local_sha = sha256_file(Path(__file__).with_name(filename))
            if runtime.get(field,local_sha) != local_sha:
                raise ValueError('local detector adapter identity drift')
            runtime[field] = local_sha
    return runtime


def _checked_text(text, score):
    if not isinstance(text, str) or len(text) > 8000:
        raise ValueError('invalid incremental OCR text budget')
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('invalid incremental OCR confidence')


def _recognition_observations(events):
    """Factor cache proof mechanics out of model observations, without truncation.

    Full crop proofs remain in the same hash-bound raw artifact. Every original
    text, confidence, origin, frame and recognition identity remains observable.
    This keeps an 8 KiB review task from duplicating two bulky pixel-key recipes.
    """
    if not isinstance(events, list) or len(events) > MAX_REGIONS or any(not isinstance(row, dict) for row in events):
        raise ValueError('incremental recognition observation schema/budget')
    return [{key: value for key, value in event.items()
             if key not in {'cache_identity', 'source_crop_identity'}} for event in events]


class RegionRecognitionCache:
    """Wrap the existing recognizer, retaining current detector order/boxes.

    Hashes cover exact post-detection/classification crop bytes, shape, dtype,
    and caller-supplied pinned runtime/model/config identity. No threshold or
    text correction. Cache misses are sent together to upstream once per frame.
    """
    def __init__(self, root, runtime_identity, recognize, output_factory):
        self.root = Path(root)
        if not isinstance(runtime_identity, dict) or not runtime_identity.get('model_actual'):
            raise ValueError('pinned runtime identity required')
        self.cache_policy = runtime_identity.get('recognition_cache_policy', EXACT_CROP)
        if not isinstance(self.cache_policy, str) or self.cache_policy not in CACHE_POLICIES:
            raise ValueError('unsupported recognition cache policy')
        self.identity = canonical_json_sha256({'runtime':runtime_identity,'adapter_sha256':sha256_file(Path(__file__))})
        self.cache = StageCache(safe_path(self.root, 'courseware/incremental-cache', exists=False),
                                safe_path(self.root, 'manifest.json'))
        self.recognize = recognize
        self.output_factory = output_factory
        payload = self.cache.load_json('regions', 'regions.json', identity=self.identity)
        self.rows = OrderedDict()
        if payload is not None:
            if not isinstance(payload, dict) or payload.get('sha256') != canonical_json_sha256(payload.get('rows')):
                raise ValueError('incremental cache hash drift')
            rows = payload['rows']
            if not isinstance(rows, list) or len(rows) > MAX_CACHE_ROWS:
                raise ValueError('incremental cache row bound')
            for row in rows:
                if not isinstance(row, dict) or row.get('key') in self.rows:
                    raise ValueError('incremental cache duplicate/invalid row')
                _checked_text(row.get('text'), row.get('score'))
                if row.get('recognition_id') != canonical_json_sha256(
                        {'runtime': self.identity, **{k:v for k,v in row.items() if k != 'recognition_id'}}):
                    raise ValueError('incremental cache recognition identity drift')
                _validate_crop_identity(row.get('crop_identity'))
                if (row['crop_identity']['cache_key_sha256'] != row['key']
                        or row['crop_identity']['runtime_sha256'] != self.identity
                        or (self.cache_policy == EXACT_CROP and row['crop_identity']['method'] != EXACT_CROP)):
                    raise ValueError('incremental crop cache identity drift')
                self.rows[row['key']] = row
        self.begin_frame(None)

    def begin_frame(self, candidate):
        self.candidate = candidate
        self.events = []
        self.recognition_calls = 0
        self.recognized_regions = 0
        self.low_confidence_regions = 0
        self.checked_sources = set()

    def _key(self, image):
        return crop_cache_identity(image, runtime_sha256=self.identity,
                                   method=self.cache_policy)['cache_key_sha256']

    def __call__(self, images):
        if not self.candidate or not 1 <= len(images) <= MAX_REGIONS:
            raise ValueError('incremental OCR requires a bound current frame and 1..512 regions')
        if len(self.events) + len(images) > MAX_REGIONS:
            raise ValueError('incremental OCR frame region budget exceeded')
        identities = [crop_cache_identity(image, runtime_sha256=self.identity,
                                         method=self.cache_policy) for image in images]
        keys = [proof['cache_key_sha256'] for proof in identities]
        misses = OrderedDict()
        selected = {}
        for i, key in enumerate(keys):
            row = self.rows.get(key)
            if row is None:
                misses.setdefault(key, i)
                continue
            source = row.get('source')
            if not isinstance(source, dict):
                raise ValueError('incremental cache source missing')
            source_key = (source.get('frame'), source.get('frame_sha256'))
            if source_key not in self.checked_sources:
                _check_frame(self.root, source)
                self.checked_sources.add(source_key)
            selected[key] = row
        if misses:
            self.recognition_calls += 1
            output = self.recognize([images[i] for i in misses.values()])
            txts, scores = output.txts, output.scores
            if txts is None or scores is None or len(txts) != len(misses) or len(scores) != len(misses):
                raise ValueError('incremental OCR result count mismatch')
            # Validate entire upstream response before changing the checkpoint.
            fresh = []
            for key, text, score in zip(misses, txts, scores):
                if isinstance(score, bool):
                    raise ValueError('invalid incremental OCR confidence')
                score = float(score)
                _checked_text(text, score)
                row = {'key':key,'text':text,'score':score,'crop_identity':identities[misses[key]],'source':{
                    k:self.candidate[k] for k in ('page_id','frame','frame_sha256')}}
                row['recognition_id'] = canonical_json_sha256({'runtime':self.identity, **row})
                fresh.append(row)
            _check_frame(self.root, self.candidate)
            for row in fresh:
                selected[row['key']] = row
                # Low-confidence or empty recognition is not a reusable fact.
                if row['text'].strip() and row['score'] >= .80:
                    self.rows[row['key']] = row
            self.recognized_regions += len(misses)
        _check_frame(self.root, self.candidate)
        # A cache hit is a use too. Persist this ordering on hit-only frames so
        # the hottest regions survive checkpoint restarts and cross-page churn.
        previous_order = tuple(self.rows)
        for key in keys:
            if key in self.rows:
                self.rows.move_to_end(key)
        while len(self.rows) > MAX_CACHE_ROWS:
            self.rows.popitem(last=False)
        if misses or tuple(self.rows) != previous_order:
            values = list(self.rows.values())
            if len(json.dumps(values, ensure_ascii=False).encode()) > 12*1024*1024:
                raise ValueError('incremental cache byte budget')
            self.cache.store_json('regions','regions.json',{'rows':values,'sha256':canonical_json_sha256(values)},identity=self.identity)
        rows = [selected[key] for key in keys]
        self.low_confidence_regions += sum(not r['text'].strip() or r['score'] < .80 for r in rows)
        offset = len(self.events)
        self.events.extend({'region_index':offset+i,'crop_sha256':identities[i]['crop_sha256'],
                            'cache_identity':identities[i],
                            'source_crop_identity':selected[key]['crop_identity'],
                            'recognized_now':key in misses and misses[key] == i,
                            'recognition_id':selected[key]['recognition_id'],
                            'text':selected[key]['text'],'score':selected[key]['score'],
                            'origin':('recognized_now' if key in misses and misses[key] == i
                                      else 'same_batch_reuse' if key in misses else 'cache_reuse'),
                            'source':selected[key]['source']}
                           for i,key in enumerate(keys))
        return self.output_factory(imgs=images, txts=tuple(r['text'] for r in rows),
                                   scores=[r['score'] for r in rows],
                                   word_results=tuple(None for _ in rows),elapse=0.0)


@contextmanager
def attach_region_cache(engine, root, runtime_identity, output_factory):
    if getattr(engine, 'return_word_box', False):
        raise ValueError('incremental OCR v1 does not support word-box mode')
    original = engine.recognize_txt
    cache = RegionRecognitionCache(root, runtime_identity, original, output_factory)
    engine.recognize_txt = cache
    try:
        yield cache
    finally:
        engine.recognize_txt = original


def _detector_projection(root, events, runtime, candidate, execution=None):
    from .courseware_detector_cache import validate_detector_events, validate_detector_execution, CHANGED_REGIONS
    target = {k:candidate[k] for k in ('page_id','frame','frame_sha256')}
    if runtime.get('detector_cache_policy') == CHANGED_REGIONS:
        metrics = validate_detector_execution(root,events,execution,runtime,target)
        return {'schema':'video_knowledge_pipeline.detector_evidence.v2',
                'events_sha256':canonical_json_sha256(events), **metrics,
                'execution_sha256':execution['execution_sha256'],
                'mode':execution['mode'], 'fallback_reason':execution['reason'],
                'coordinate_space':'rapidocr_detector_input_pixels',
                'full_detector_equivalence':False,'independent_review_complete':False}
    if execution is not None:
        raise ValueError('local detector proof without opt-in policy')
    metrics = validate_detector_events(root, events, runtime)
    if (len(events) != 1 or metrics['detector_inputs'] != 1
            or events[0].get('target') != target):
        raise ValueError('incremental detector hook/target identity mismatch')
    return {'schema':'video_knowledge_pipeline.detector_evidence.v1',
            'events_sha256':canonical_json_sha256(events), **metrics,
            'coordinate_space':'rapidocr_detector_input_pixels',
            'independent_review_complete':False}


def run_incremental_courseware_ocr(bundle_dir, *, indexes, engine, runtime_identity,
                                  output_factory, normalize_blocks, resume=False,
                                  detector_output_factory=None):
    """Explicit local bounded executor. Caller owns resource/service exclusion.

    Page import is independently atomic; cache checkpoint precedes page import.
    Restart never overwrites a formal/manual page. No automatic retry.
    """
    from .courseware_detector_cache import attach_detector_cache, DISABLED
    runtime_identity = _detector_runtime(runtime_identity)
    detector_policy = runtime_identity.get('detector_cache_policy', DISABLED)
    root = Path(bundle_dir).absolute()
    if (not isinstance(indexes, list) or not 1 <= len(indexes) <= 8 or len(set(indexes)) != len(indexes)
            or any(type(i) is not int or i < 1 for i in indexes)):
        raise ValueError('incremental OCR requires 1..8 explicit indexes')
    state = read_scan(root)
    if max(indexes) > len(state['candidates']):
        raise ValueError('incremental OCR index outside scan')
    for index in indexes:
        _page_id(state['candidates'][index-1]['page_id'])
    job_id = incremental_run_id(state['identity'], indexes, runtime_identity)
    receipt_path = safe_path(root, f'courseware/incremental-runs/{job_id}.json', exists=False)
    if receipt_path.exists() and not resume:
        raise ValueError('existing incremental run requires explicit resume')
    if resume and not receipt_path.exists():
        raise ValueError('resume run identity missing/drifted; use an explicit new run')
    receipt = {'schema':SCHEMA,'run_id':job_id,'runtime':runtime_identity,'indexes':indexes,
               'status':'running','pages':[],'automatic_retries':0,'recognition_calls':0,
               'recognized_regions':0,'detector_calls':0,'detected_regions':0,'reused_regions':0,
               'detector_inputs':0,'reused_detector_inputs':0,'detector_cache_policy':detector_policy,
               'local_detector_frames':0,'local_detector_fallbacks':0,'detector_frame_inputs':0,
               'review_required':True}
    with bundle_write_lock(root, operation='incremental_ocr', lock_name='.courseware-incremental.lock'):
        if receipt_path.exists():
            history = safe_path(root, f'courseware/incremental-runs/history/{sha256_file(receipt_path)}.json', exists=False)
            if not history.exists():
                write_json(history,read_json(receipt_path))
            receipt['previous_attempt'] = history.relative_to(root).as_posix()
        with (attach_region_cache(engine, root, runtime_identity, output_factory) as cache,
              attach_detector_cache(engine, root, runtime_identity, detector_output_factory) as detector):
            for index in indexes:
                candidate = state['candidates'][index-1]
                try:
                    frame = _check_frame(root, candidate)
                    formal = safe_path(root, f"courseware/pages/{candidate['page_id']}.json", exists=False)
                    if formal.exists():
                        from .courseware_material import _validate_page
                        _validate_page(root, state, read_json(formal))
                        receipt['pages'].append({'index':index,'status':'no_op_existing_result'})
                        continue
                    cache.begin_frame(candidate)
                    inputs = {name:sha256_file(safe_path(root,name)) for name in ('manifest.json','timeline.json')}
                    if detector is not None:
                        detector.begin_frame(candidate)
                    else:
                        receipt['detector_calls'] += 1
                        receipt['detector_inputs'] += 1
                        receipt['detector_frame_inputs'] += 1
                    try:
                        raw = engine(str(frame))  # Existing crop, classification, coordinates and merge.
                    except Exception as exc:
                        failure_raw = safe_path(root,f'courseware/incremental-raw/{uuid4().hex}.failure.json',exists=False)
                        write_json(failure_raw,{'schema':SCHEMA,'runtime':runtime_identity,
                            'candidate':candidate,'run_id':job_id,'region_events':cache.events,
                            'detector_events':detector.events if detector is not None else [],
                            'detector_execution':detector.execution if detector is not None else None})
                        failure_validation = failure_raw.with_suffix('.validation.json')
                        write_json(failure_validation,{'ok':False,'changed':False,'error':str(exc),
                            'raw_path':failure_raw.relative_to(root).as_posix(),'raw_sha256':sha256_file(failure_raw)})
                        receipt['failure_evidence'] = failure_validation.relative_to(root).as_posix()
                        raise
                    finally:
                        if detector is not None:
                            receipt['detector_calls'] += detector.detector_calls
                            receipt['reused_detector_inputs'] += detector.reused_inputs
                            receipt['detector_inputs'] += detector.detector_calls + detector.reused_inputs
                            receipt['local_detector_frames'] += detector.local_frames
                            receipt['local_detector_fallbacks'] += detector.local_fallbacks
                            receipt['detector_frame_inputs'] += detector.frame_inputs
                    detector_raw = {'detector_events':detector.events} if detector is not None else {}
                    if detector is not None and detector.execution is not None:
                        detector_raw['detector_execution'] = detector.execution
                    from .courseware_ocr_geometry import _rows
                    final_recognition = {'txts':_rows(getattr(raw,'txts',None)),
                                         'scores':_rows(getattr(raw,'scores',None))}
                    observation_path = safe_path(root, f'courseware/incremental-raw/{uuid4().hex}.observation.json', exists=False)
                    write_json(observation_path, {'schema':SCHEMA,'runtime':runtime_identity,
                        'region_events':cache.events,'final_recognition':final_recognition,
                        'candidate':candidate,'run_id':job_id,**detector_raw})
                    try:
                        blocks = normalize_blocks(raw)
                        if not isinstance(blocks, list) or len(blocks) > MAX_REGIONS:
                            raise ValueError('incremental detector output bound')
                        if blocks and not cache.events:
                            raise ValueError('RapidOCR recognition hook was not used; incompatible runtime')
                        region_evidence = _region_projection(cache.events, final_recognition, blocks,
                                                            cache_policy=cache.cache_policy)
                        detector_evidence = (_detector_projection(root, detector.events, runtime_identity, candidate, detector.execution)
                                             if detector is not None else None)
                    except Exception as exc:
                        write_json(observation_path.with_suffix('.validation.json'), {
                            'ok':False,'changed':False,'error':str(exc),
                            'raw_path':observation_path.relative_to(root).as_posix(),
                            'raw_sha256':sha256_file(observation_path)})
                        raise
                    _check_frame(root, candidate)
                    current = read_scan(root)
                    if current['identity'] != state['identity'] or current['candidates'][index-1] != candidate:
                        raise ValueError('incremental selected frame/scan drift')
                    if any(sha256_file(safe_path(root,name)) != h for name,h in inputs.items()):
                        raise ValueError('incremental manifest/timeline drift')
                    low = (not blocks or cache.low_confidence_regions > 0
                           or any(float(b.get('score',0)) < .80 for b in blocks))
                    result = {'schema':'video_knowledge_pipeline.courseware_page_result.v1',
                        'scan_identity':state['identity'],'page_id':candidate['page_id'],
                        'time':candidate['time'],'frame_time':candidate.get('frame_time',candidate['time']),
                        'frame_sha256':candidate['frame_sha256'],'executor':'RapidOCR/incremental',
                        'model_actual':runtime_identity['model_actual'],
                        'status':'extracted' if blocks else 'unavailable',
                        'blocks':[{'kind':'body','text':b['text'],'structure':b} for b in blocks],
                        'region_evidence':region_evidence,
                        'gaps':['incremental OCR empty/low-confidence; independent review required'] if low else []}
                    if region_evidence['unrepresented_region_count']:
                        result['gaps'].append('recognition output excluded by upstream filtering; inspect region receipt before completeness approval')
                    if detector_evidence is not None:
                        result['detector_evidence'] = detector_evidence
                        if detector_evidence.get('mode') == 'roi':
                            result['gaps'].append('local detector changes scale/context; full-detector equivalence unproven; independent review required')
                    if runtime_identity.get('geometry_contract'):
                        from .courseware_ocr_geometry import CONTRACT, capture_geometry
                        if runtime_identity['geometry_contract'] != CONTRACT:
                            raise ValueError('unsupported OCR geometry contract')
                        result['ocr_geometry'] = capture_geometry(root, candidate, raw, blocks)
                    raw_path = safe_path(root, f'courseware/incremental-raw/{uuid4().hex}.json', exists=False)
                    write_json(raw_path, {'schema':SCHEMA,'result':result,'region_events':cache.events,
                                          'observations':_recognition_observations(cache.events),
                                          'final_recognition':final_recognition,
                                          'runtime':runtime_identity,**detector_raw})
                    result['incremental_ocr'] = {'raw_path':raw_path.relative_to(root).as_posix(),
                        'raw_sha256':sha256_file(raw_path),'run_id':job_id,
                        'region_evidence_schema':REGION_EVIDENCE_SCHEMA,
                        'recognition_calls':cache.recognition_calls,'recognized_regions':cache.recognized_regions}
                    imported = import_courseware_page(root,result)
                    receipt['pages'].append({'index':index,**imported})
                    receipt['recognition_calls'] += cache.recognition_calls
                    receipt['recognized_regions'] += cache.recognized_regions
                    receipt['detected_regions'] += len(cache.events)
                    receipt['reused_regions'] += len(cache.events)-cache.recognized_regions
                    write_json(receipt_path,receipt)
                except Exception as exc:
                    receipt.update(status='blocked',failure_index=index,error=str(exc),
                                   recovery='Explicitly rerun same indexes/runtime with resume=True; never overwrite formal pages.')
                    write_json(receipt_path,receipt)
                    raise
    receipt['status'] = 'completed_review_required'
    write_json(receipt_path,receipt)
    return receipt


def validate_incremental_page(root, value):
    proof = value['incremental_ocr']
    path = safe_path(root, proof['raw_path'])
    if path.stat().st_size > 16*1024*1024 or sha256_file(path) != proof['raw_sha256']:
        raise ValueError('incremental raw receipt hash drift')
    raw = read_json(path)
    if raw.get('schema') != SCHEMA or raw.get('result') != {k:v for k,v in value.items() if k != 'incremental_ocr'}:
        raise ValueError('incremental raw/page mismatch')
    if 'observations' in raw and raw['observations'] != _recognition_observations(raw.get('region_events', [])):
        raise ValueError('incremental recognition observations drift')
    from .courseware_detector_cache import EXACT_DETECTOR, CHANGED_REGIONS, DISABLED
    detector_policy = raw.get('runtime',{}).get('detector_cache_policy', DISABLED)
    if detector_policy != DISABLED or 'detector_evidence' in value or 'detector_events' in raw or 'detector_execution' in raw:
        if detector_policy not in {EXACT_DETECTOR,CHANGED_REGIONS}:
            raise ValueError('incremental detector evidence policy mismatch')
        matches = [row for row in read_scan(root)['candidates'] if row['page_id'] == value['page_id']]
        if len(matches) != 1:
            raise ValueError('incremental detector page identity mismatch')
        expected = _detector_projection(root, raw.get('detector_events'), raw['runtime'], matches[0],raw.get('detector_execution'))
        if value.get('detector_evidence') != expected:
            raise ValueError('incremental detector evidence projection drift')
    if 'region_evidence' in value or 'region_evidence_schema' in proof:
        if proof.get('region_evidence_schema') != REGION_EVIDENCE_SCHEMA:
            raise ValueError('incremental region projection schema missing')
        if any(not isinstance(b.get('structure'),dict) or b['text'] != b['structure'].get('text')
               for b in value['blocks']):
            raise ValueError('incremental region projection outer_text_mismatch')
        expected = _region_projection(raw.get('region_events'), raw.get('final_recognition'),
            [b['structure'] for b in value['blocks']],
            cache_policy=raw.get('runtime', {}).get('recognition_cache_policy', EXACT_CROP))
        if expected != value.get('region_evidence'):
            raise ValueError('incremental region projection drift')
    for event in raw.get('region_events',[]):
        _check_frame(root,event['source'])


def project_incremental_blocks(root, page):
    """Project validated producer observations; reused crops aren't new sources.

    An equal text/score match may identify several crops. Preserve the complete
    candidate set, not a fabricated unique crop-to-box mapping.
    """
    if not page.get('region_evidence'):
        return None  # Legacy pages stay readable, not retrospectively certified.
    from .evidence_locator import JsonEvidenceSnapshot
    from .evidence_lineage import project_recognition_provenance
    raw = JsonEvidenceSnapshot.read(root, page['incremental_ocr']['raw_path'])
    if raw.artifact_sha256 != page['incremental_ocr']['raw_sha256']:
        raise ValueError('incremental raw receipt hash drift')
    events = raw.data['region_events']
    observation_field = 'observations' if 'observations' in raw.data else 'region_events'
    projections = []
    for block, mapping in zip(page['blocks'], page['region_evidence']['blocks']):
        indexes = mapping['candidate_region_indexes']
        matched = [events[i] for i in indexes]
        hashes = sorted({page['frame_sha256'], *(e['source']['frame_sha256'] for e in matched)})
        record = {'executor':page['executor'],'model_actual':page['model_actual'],
                  'frame_sha256s':hashes}
        provenance = project_recognition_provenance({'recognition_provenance':record},
            selected_field=['blocks', mapping['block_index']], text=block['text'])
        # Shared provenance's bound is deliberate. If exceeded, its invalid
        # status is retained and the full source set remains in exact locators.
        projections.append({'recognition_provenance':provenance,
            'incremental_region_context':{
                'recognition_ids':sorted({e['recognition_id'] for e in matched}),
                'origins':sorted({e['origin'] for e in matched}),
                'candidate_region_indexes':indexes,
                'assignment_scope':page['region_evidence']['assignment_scope'],
                'independence_is_verified':False},
            'source_locators':[raw.locator([observation_field,i]) for i in indexes]})
    return {'blocks':projections,'dependency':{'path':raw.artifact_path,'sha256':raw.artifact_sha256}}


def _region_projection(events, final, blocks, *, cache_policy=EXACT_CROP):
    """Conserve upstream text/score multisets; do not redo detection or geometry.

    Equal text at different crops is an ambiguous source set, not proof of one
    exact crop-to-final-box assignment. Existing geometry validates final boxes.
    """
    from .courseware_ocr_geometry import _rows
    def fail(reason):
        raise ValueError('incremental region projection:' + reason)
    if not isinstance(cache_policy, str) or cache_policy not in CACHE_POLICIES:
        fail('cache_policy')
    if not isinstance(events, list) or len(events) > MAX_REGIONS or not isinstance(final, dict):
        fail('schema_or_budget')
    def key(text, score):
        _checked_text(text, score)
        return (text.strip(), round(score, 4))
    event_keys = []
    for i, event in enumerate(events):
        if not isinstance(event, dict) or type(event.get('region_index')) is not int or event['region_index'] != i:
            fail('event_order')
        from .evidence_lineage import _hash
        if (not _hash(event.get('recognition_id')) or not _hash(event.get('crop_sha256'))
                or not isinstance(event.get('source'),dict)
                or not _hash(event['source'].get('frame_sha256'))
                or event.get('origin') not in {'recognized_now','same_batch_reuse','cache_reuse'}
                or type(event.get('recognized_now')) is not bool
                or event['recognized_now'] != (event['origin'] == 'recognized_now')):
            fail('recognition_identity_or_origin')
        if cache_policy == UNIFORM_PADDING or 'cache_identity' in event or 'source_crop_identity' in event:
            current, source = event.get('cache_identity'), event.get('source_crop_identity')
            _validate_crop_identity(current)
            _validate_crop_identity(source)
            if (current['crop_sha256'] != event['crop_sha256']
                    or current['cache_key_sha256'] != source['cache_key_sha256']
                    or current['runtime_sha256'] != source['runtime_sha256']
                    or (cache_policy == EXACT_CROP and any(proof['method'] != EXACT_CROP
                                                          for proof in (current, source)))
                    or (event['recognized_now'] and current != source)):
                fail('crop_identity_binding')
            expected_recognition = canonical_json_sha256({'runtime':source['runtime_sha256'],
                'key':source['cache_key_sha256'],'text':event.get('text'),'score':event.get('score'),
                'crop_identity':source,'source':event['source']})
            if event['recognition_id'] != expected_recognition:
                fail('recognition_identity_binding')
        event_keys.append(key(event.get('text'), event.get('score')))
    texts, scores = _rows(final.get('txts')), _rows(final.get('scores'))
    if len(texts) != len(scores):
        fail('final_parallel_count')
    final_keys = [key(text, score) for text, score in zip(texts, scores)]
    if Counter(final_keys) - Counter(event_keys):
        fail('final_output_not_from_recognition')
    block_keys = [key(block.get('text'), block.get('score')) for block in blocks]
    if Counter(block_keys) != Counter(k for k in final_keys if k[0]):
        fail('normalized_text_score_conservation')
    return {'schema':REGION_EVIDENCE_SCHEMA,
            'region_events_sha256':canonical_json_sha256(events),
            'final_recognition_sha256':canonical_json_sha256(final),
            'recognized_region_occurrences':len(events),
            'final_output_count':len(texts),'normalized_block_count':len(blocks),
            'unrepresented_region_count':len(events)-len(texts),
            'independent_source_count':0,
            'assignment_scope':'matching_text_score_candidates_not_unique_crop_geometry',
            'blocks':[{'block_index':i, 'candidate_region_indexes':[
                n for n,k in enumerate(event_keys) if k == block_key]}
                for i,block_key in enumerate(block_keys)]}
