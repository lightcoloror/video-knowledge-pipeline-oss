"""Offline actual-hook tests: changed-region detector + current crop recognition."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from test_courseware_dedupe import image_bundle
from test_courseware_incremental_ocr import Engine
from test_courseware_detector_integration import normalize
from video_knowledge_pipeline import courseware_incremental_ocr as runner
from video_knowledge_pipeline.courseware_material import read_scan, _validate_page
from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.canonical_json import canonical_json_sha256

POLICY = 'changed_regions_v1'


def slide(value=20):
    image = np.full((240, 400, 3), 255, dtype=np.uint8)
    image[40:60, 40:60] = 10
    image[140:160, 280:300] = value
    return Image.fromarray(image)


class Detector:
    def __init__(self):
        self.shapes = []

    def __call__(self, data):
        self.shapes.append(data.shape)
        boxes = []
        for value in np.unique(data[:, :, 0]):
            if value == 255:
                continue
            ys, xs = np.where(data[:, :, 0] == value)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1
            boxes.append([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
        return SimpleNamespace(img=data,boxes=np.asarray(boxes,dtype=np.float32) if boxes else None,
                               scores=[.95]*len(boxes) if boxes else None,elapse=.01)

    @staticmethod
    def get_preprocess(max_wh):
        return SimpleNamespace(limit_type='max',limit_side_len=640)

    @staticmethod
    def sorted_boxes(boxes):
        return np.asarray(sorted(boxes.tolist(), key=lambda b:(b[0][1],b[0][0])),dtype=boxes.dtype)


class LocalEngine(Engine):
    def __init__(self, fail_call=0):
        super().__init__(fail_call)
        self.detector = Detector()
        self.text_det = self.detector

    def __call__(self, path):
        with Image.open(path) as image:
            data = np.asarray(image.convert('RGB'))
        detected = self.text_det(data)
        crops = [data[int(b[0,1]):int(b[2,1]),int(b[0,0]):int(b[2,0])].copy()
                 for b in detected.boxes] if detected.boxes is not None else []
        raw = self.recognize_txt(crops) if crops else SimpleNamespace(txts=[],scores=[])
        raw.boxes, raw.img = detected.boxes, data
        return raw


def run(root, engine, indexes, **kwargs):
    return runner.run_incremental_courseware_ocr(root,indexes=indexes,engine=engine,
        runtime_identity={'model_actual':'synthetic-local','detector_cache_policy':POLICY,
                          'geometry_contract':'rapidocr_final_original_pixels.v1'},
        output_factory=SimpleNamespace,detector_output_factory=SimpleNamespace,
        normalize_blocks=normalize,**kwargs)


def test_changed_roi_only_detects_local_pixels_and_merges_current_page(tmp_path):
    root = image_bundle(tmp_path,[slide(),slide(21)])
    before = (root/'timeline.json').read_bytes()
    engine = LocalEngine()
    receipt = run(root,engine,[1,2])
    assert engine.detector.shapes[0] == (240,400,3)
    assert np.prod(engine.detector.shapes[1][:2]) < 240*400*.35
    assert receipt['detector_calls'] == 2 and receipt['local_detector_frames'] == 1
    assert receipt['recognized_regions'] == 3 and receipt['reused_regions'] == 1
    page = read_json(root/'courseware/pages/page-1.json')
    assert len(page['blocks']) == 2 and '21;' in str(page['blocks']) and '20;' not in str(page['blocks'])
    raw = read_json(root/page['incremental_ocr']['raw_path'])
    assert raw['detector_execution']['mode'] == 'roi'
    assert page['detector_evidence']['full_detector_equivalence'] is False
    assert page['gaps'] and page['detector_evidence']['independent_review_complete'] is False
    _validate_page(root,read_scan(root),page)
    assert (root/'timeline.json').read_bytes() == before


def test_deleted_region_empty_falls_back_full_and_never_keeps_old_text(tmp_path):
    root = image_bundle(tmp_path,[slide(),slide(255)])
    engine = LocalEngine()
    receipt = run(root,engine,[1,2])
    assert receipt['detector_calls'] == 3 and receipt['local_detector_fallbacks'] == 1
    assert receipt['detector_frame_inputs'] == 2 and receipt['detector_inputs'] == 3
    assert engine.detector.shapes[-1] == (240,400,3)
    page = read_json(root/'courseware/pages/page-1.json')
    assert len(page['blocks']) == 1 and '20;' not in str(page['blocks'])
    _validate_page(root,read_scan(root),page)


def test_resume_reestablishes_full_anchor_and_preserves_completed_pages(tmp_path):
    root = image_bundle(tmp_path,[slide(),slide(21),slide(22)])
    with pytest.raises(RuntimeError,match='injected'):
        run(root,LocalEngine(fail_call=2),[1,2,3])
    first = (root/'courseware/pages/page-0.json').read_bytes()
    engine = LocalEngine()
    result = run(root,engine,[1,2,3],resume=True)
    assert engine.detector.shapes[0] == (240,400,3)
    assert result['status'] == 'completed_review_required'
    assert (root/'courseware/pages/page-0.json').read_bytes() == first
    again = run(root,LocalEngine(),[1,2,3],resume=True)
    assert again['detector_calls'] == 0


def test_partial_results_never_become_cache_or_anchor_and_old_pages_revalidate(tmp_path):
    root = image_bundle(tmp_path,[slide(),slide(21),slide(22)])
    run(root,LocalEngine(),[1,2,3])
    page = read_json(root/'courseware/pages/page-2.json')
    raw = read_json(root/page['incremental_ocr']['raw_path'])
    assert raw['detector_execution']['anchor_events'][-1]['target']['page_id'] == 'page-0'
    entries = read_json(root/'courseware/incremental-cache/detector/inputs.json')['rows']
    assert len(entries) == 1 and entries[0]['source']['page_id'] == 'page-0'
    # History verification is self-contained, independent of current LRU contents.
    (root/'courseware/incremental-cache/detector/inputs.json').write_bytes(b'future cache is unavailable')
    for index in range(3):
        _validate_page(root,read_scan(root),read_json(root/f'courseware/pages/page-{index}.json'))


def test_unknown_upstream_resize_contract_falls_back_before_roi_call(tmp_path):
    root = image_bundle(tmp_path,[slide(),slide(21)])
    engine = LocalEngine()
    engine.detector.get_preprocess = None
    result = run(root,engine,[1,2])
    assert engine.detector.shapes == [(240,400,3),(240,400,3)]
    assert result['local_detector_frames'] == 0
    page = read_json(root/'courseware/pages/page-1.json')
    raw = read_json(root/page['incremental_ocr']['raw_path'])
    assert raw['detector_execution']['network_budget']['allowed'] is False
    _validate_page(root,read_scan(root),page)


def test_failed_recognition_preserves_zero_change_failure_evidence(tmp_path):
    root = image_bundle(tmp_path,[slide(),slide(21)])
    before = (root/'timeline.json').read_bytes()
    with pytest.raises(RuntimeError,match='injected'):
        run(root,LocalEngine(fail_call=2),[1,2])
    assert (root/'timeline.json').read_bytes() == before
    assert not (root/'courseware/pages/page-1.json').exists()
    runs = list((root/'courseware/incremental-runs').glob('*.json'))
    receipt = read_json(runs[0])
    validation = read_json(root/receipt['failure_evidence'])
    assert validation['changed'] is False and validation['ok'] is False
    raw = read_json(root/validation['raw_path'])
    assert raw['detector_execution']['mode'] == 'roi'


def test_local_policy_is_checkpoint_bounded_and_serial_across_nine_pages(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from video_knowledge_pipeline.courseware_ocr_batch import run_ocr_batch
    from video_knowledge_pipeline import courseware_detector_cache as cache_module
    attach = cache_module.attach_detector_cache
    monkeypatch.setattr(cache_module,'attach_detector_cache',
        lambda engine,root,runtime,output_factory=None:attach(engine,root,runtime,SimpleNamespace))
    root = image_bundle(tmp_path,[slide(20+i) for i in range(9)])
    engine = LocalEngine()
    @contextmanager
    def runtime():
        yield engine, {'model_actual':'synthetic-local','detector_cache_policy':POLICY}, SimpleNamespace, normalize
    result = run_ocr_batch(root,indexes=list(range(1,10)),runtime_factory=runtime,
                          policy_identity={'model':'synthetic-local'},guard=lambda:None)
    assert result['model_loads'] == 1 and len(result['subbatches']) == 2
    assert result['local_detector_frames'] == 7
    assert sum(s == (240,400,3) for s in engine.detector.shapes) == 2
    again = run_ocr_batch(root,indexes=list(range(1,10)),runtime_factory=runtime,
        policy_identity={'model':'synthetic-local'},guard=lambda:None,resume=True)
    assert again['model_loads'] == 0 and again['detector_calls'] == 0


@pytest.mark.parametrize('change',['missing','target','plan','merged','anchor','reason','budget','adapter'])
def test_local_execution_proof_fails_closed_even_if_raw_outer_hash_updated(tmp_path,change):
    root = image_bundle(tmp_path,[slide(),slide(21)])
    run(root,LocalEngine(),[1,2])
    page = read_json(root/'courseware/pages/page-1.json')
    path = root/page['incremental_ocr']['raw_path']
    raw = read_json(path)
    if change == 'missing':
        raw.pop('detector_execution')
    else:
        execution = raw['detector_execution']
        if change == 'target': execution['target']['page_id']='page-0'
        if change == 'plan': execution['plan']['retained_indexes']=[]
        if change == 'merged': execution['output']['boxes']=[]
        if change == 'anchor': execution['anchor_events'][-1]['target']['page_id']='page-1'
        if change == 'reason': execution['reason']='forged_reason'
        if change == 'budget': execution['network_budget']['roi_pixels']=1
        if change == 'adapter':
            execution['input_identity']['adapter_sha256']='a'*64
            execution['input_identity']['input_sha256']=canonical_json_sha256({k:v for k,v in execution['input_identity'].items() if k!='input_sha256'})
        execution['execution_sha256']=canonical_json_sha256({k:v for k,v in execution.items() if k!='execution_sha256'})
        page['detector_evidence']['execution_sha256']=execution['execution_sha256']
        raw['result']=deepcopy({k:v for k,v in page.items() if k!='incremental_ocr'})
    write_json(path,raw)
    page['incremental_ocr']['raw_sha256']=sha256_file(path)
    with pytest.raises(ValueError):
        _validate_page(root,read_scan(root),page)


def test_fixed_upstream_crop_sort_and_coordinate_mapping_on_local_merged_boxes(tmp_path):
    """Run pinned upstream pure functions, without loading its model runtime."""
    import ast
    import copy
    import os
    from pathlib import Path
    from dataclasses import dataclass
    import cv2
    from video_knowledge_pipeline.courseware_detector_cache import attach_detector_cache, validate_detector_execution
    source_root = os.environ.get('VKP_RAPIDOCR_SOURCE')
    if not source_root:
        pytest.skip('explicit fixed local RapidOCR source required')
    source = Path(source_root)/'python/rapidocr'
    def selected_class(path,name,methods=None):
        tree = ast.parse((source/path).read_text(encoding='utf-8-sig'))
        node = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
        if methods is not None:
            node.body = [n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in methods]
            assert len(node.body)==len(methods)
        return node
    process = ast.parse((source/'utils/process_img.py').read_text(encoding='utf-8-sig'))
    functions = [n for n in process.body if isinstance(n,ast.FunctionDef)
        and n.name in {'apply_vertical_padding','get_rotate_crop_image','map_boxes_to_original'}]
    body = [ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),
        *functions, selected_class('ch_ppocr_det/utils.py','TextDetOutput'),
        selected_class('ch_ppocr_det/utils.py','DetPreProcess'),
        selected_class('ch_ppocr_det/main.py','TextDetector',{'get_preprocess','sorted_boxes'}),
        selected_class('main.py','RapidOCR',{'detect_and_crop','crop_text_regions'})]
    ns = {'np':np,'cv2':cv2,'copy':copy,'dataclass':dataclass,
          '_BOX_SORT_Y_THRESHOLD':10,'RapidOCRError':RuntimeError}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(source/'main.py'),'exec'),ns)
    upstream = ns['RapidOCR']()
    upstream.width_height_ratio = -1
    upstream.min_height = 0
    detector = Detector()
    detector.sorted_boxes = ns['TextDetector'].sorted_boxes
    preprocess_owner = ns['TextDetector']()
    preprocess_owner.limit_type = 'min'
    preprocess_owner.limit_side_len = 736
    preprocess_owner.mean = preprocess_owner.std = [.5]*3
    detector.get_preprocess = preprocess_owner.get_preprocess
    upstream.text_det = detector
    root = image_bundle(tmp_path,[slide(),slide(21)])
    runtime = runner._detector_runtime({'model_actual':'synthetic-local','detector_cache_policy':POLICY})
    candidates = read_scan(root)['candidates']
    with attach_detector_cache(upstream,root,runtime,ns['TextDetOutput']) as active:
        for candidate, image in zip(candidates,[slide(),slide(21)]):
            active.begin_frame(candidate)
            data = np.asarray(image)
            op = {'preprocess':{'ratio_h':1.,'ratio_w':1.}}
            crops, detected = upstream.detect_and_crop(data,op)
            assert len(crops) == 2 and np.all(crops[0] == 10)
            native = detected.boxes.copy()
            op['preprocess'] = {'ratio_h':.5,'ratio_w':.5}
            ns['map_boxes_to_original'](detected.boxes,op,120,200)
            assert not np.array_equal(native,detected.boxes)
            validate_detector_execution(root,active.events,active.execution,runtime,
                {k:candidate[k] for k in ('page_id','frame','frame_sha256')})
        assert active.execution['mode'] == 'roi'
        assert np.all(crops[1] == 21) and active.local_frames == 1
        assert detected.elapse > 0


@pytest.mark.parametrize('change',['reason','plan','budget'])
def test_full_budget_fallback_explanation_is_validated(tmp_path,change):
    root = image_bundle(tmp_path,[slide(),slide(21)])
    engine = LocalEngine()
    engine.detector.get_preprocess = None
    run(root,engine,[1,2])
    page = read_json(root/'courseware/pages/page-1.json')
    path = root/page['incremental_ocr']['raw_path']
    raw = read_json(path)
    execution = raw['detector_execution']
    if change == 'reason': execution['reason']='forged_reason'
    if change == 'plan': execution['plan']['changed_pixels']=0
    if change == 'budget': execution['network_budget']['allowed']=True
    execution['execution_sha256']=canonical_json_sha256({k:v for k,v in execution.items() if k!='execution_sha256'})
    page['detector_evidence']['execution_sha256']=execution['execution_sha256']
    page['detector_evidence']['fallback_reason']=execution['reason']
    raw['result']=deepcopy({k:v for k,v in page.items() if k!='incremental_ocr'})
    write_json(path,raw)
    page['incremental_ocr']['raw_sha256']=sha256_file(path)
    with pytest.raises(ValueError):
        _validate_page(root,read_scan(root),page)
