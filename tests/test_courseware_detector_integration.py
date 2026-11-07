"""Detection cache remains inside the current page writer and restore path."""
from types import SimpleNamespace
from copy import deepcopy
from contextlib import contextmanager

import numpy as np
from PIL import Image
import pytest

from test_courseware_dedupe import image_bundle
from test_courseware_incremental_ocr import Engine, picture
from video_knowledge_pipeline import courseware_incremental_ocr as module
from video_knowledge_pipeline.courseware_material import read_scan, _validate_page
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.storage import read_json, write_json


POLICY = 'exact_detector_input_v1'


class DetectorEngine(Engine):
    def __init__(self):
        super().__init__()
        self.detections = 0
        self.text_det = self.detect

    def detect(self, data):
        self.detections += 1
        boxes = [[[x, 0], [x+20, 0], [x+20, 20], [x, 20]] for x in (0, 20) if data[0, x, 0] != 255]
        return SimpleNamespace(img=data, boxes=np.asarray(boxes, dtype=np.float32) if boxes else None,
                               scores=[.95]*len(boxes) if boxes else None, elapse=.01)

    def __call__(self, path):
        with Image.open(path) as image:
            data = np.asarray(image.convert('RGB'))
        detected = self.text_det(data)
        images = [data[int(b[0,1]):int(b[2,1]), int(b[0,0]):int(b[2,0])].copy()
                  for b in detected.boxes] if detected.boxes is not None else []
        raw = self.recognize_txt(images) if images else SimpleNamespace(txts=[], scores=[])
        raw.boxes, raw.img = detected.boxes, data
        return raw


def normalize(raw):
    return [{'index':i+1, 'text':t, 'score':round(float(s),4),
             'bbox':[float(b[0,0]),float(b[0,1]),float(b[2,0]),float(b[2,1])]}
            for i,(t,s,b) in enumerate(zip(raw.txts,raw.scores,raw.boxes if raw.boxes is not None else []))]


def run(root, engine, indexes, **kwargs):
    return module.run_incremental_courseware_ocr(root,indexes=indexes,engine=engine,
        runtime_identity={'model_actual':'synthetic','geometry_contract':'rapidocr_final_original_pixels.v1',
                          'detector_cache_policy':POLICY},
        output_factory=SimpleNamespace,detector_output_factory=SimpleNamespace,
        normalize_blocks=normalize,**kwargs)


def test_same_detection_input_reuses_but_current_page_geometry_is_retained(tmp_path):
    root = image_bundle(tmp_path, [picture(),picture(),picture(b=21)])
    before = (root/'timeline.json').read_bytes()
    engine = DetectorEngine()
    result = run(root,engine,[1,2,3])
    assert engine.detections == result['detector_calls'] == 2
    assert result['detector_inputs'] == 3 and result['reused_detector_inputs'] == 1
    assert engine.regions == 3  # existing crop cache still operates
    page = read_json(root/'courseware/pages/page-1.json')
    assert page['ocr_geometry']['frame_binding']['frame'] == 'courseware/frames/1.png'
    assert page['detector_evidence']['reused_inputs'] == 1
    raw = read_json(root/page['incremental_ocr']['raw_path'])
    assert raw['detector_events'][0]['origin'] == 'cache_reuse'
    assert raw['detector_events'][0]['source']['page_id'] == 'page-0'
    assert raw['detector_events'][0]['target']['page_id'] == 'page-1'
    _validate_page(root,read_scan(root),page)
    assert (root/'timeline.json').read_bytes() == before
    assert run(root,DetectorEngine(),[1,2,3],resume=True)['detector_calls'] == 0


def test_import_interruption_resumes_from_detection_and_recognition_cache(tmp_path, monkeypatch):
    root=image_bundle(tmp_path,[picture()])
    original=module.import_courseware_page
    monkeypatch.setattr(module,'import_courseware_page',lambda *a,**kw: (_ for _ in ()).throw(RuntimeError('injected import failure')))
    with pytest.raises(RuntimeError,match='injected'):
        run(root,DetectorEngine(),[1])
    monkeypatch.setattr(module,'import_courseware_page',original)
    engine=DetectorEngine()
    result=run(root,engine,[1],resume=True)
    assert result['detector_calls']==0 and result['reused_detector_inputs']==1
    assert engine.calls==0 and engine.detections==0


@pytest.mark.parametrize('change', ['remove', 'target', 'metrics', 'events_hash'])
def test_formal_detector_proof_tampering_fails_closed(tmp_path,change):
    root=image_bundle(tmp_path,[picture(),picture()])
    run(root,DetectorEngine(),[1,2])
    page=read_json(root/'courseware/pages/page-1.json')
    raw_path=root/page['incremental_ocr']['raw_path']
    raw=read_json(raw_path)
    if change=='remove':
        raw.pop('detector_events')
    elif change=='target':
        raw['detector_events'][0]['target']=deepcopy(raw['detector_events'][0]['source'])
    elif change=='metrics':
        page['detector_evidence']['reused_inputs']=0
        raw['result']['detector_evidence']=deepcopy(page['detector_evidence'])
    else:
        page['detector_evidence']['events_sha256']='f'*64
        raw['result']['detector_evidence']=deepcopy(page['detector_evidence'])
    write_json(raw_path,raw)
    page['incremental_ocr']['raw_sha256']=sha256_file(raw_path)
    with pytest.raises(ValueError,match='detector'):
        _validate_page(root,read_scan(root),page)


def test_policy_rejected_before_loading_models(tmp_path):
    from video_knowledge_pipeline.courseware_incremental_runtime import model_policy
    policy=tmp_path/'models.json'
    write_json(policy,{'schema':'video_knowledge_pipeline.incremental_ocr_models.v1',
                       'detector_cache_policy':'loose_roi'})
    with pytest.raises(ValueError,match='detector cache policy'):
        model_policy(policy)


def test_batch_reuses_detector_across_checkpoints_with_one_model_load(tmp_path,monkeypatch):
    from video_knowledge_pipeline.courseware_ocr_batch import run_ocr_batch
    from video_knowledge_pipeline import courseware_detector_cache as detector_module
    original_attach=detector_module.attach_detector_cache
    monkeypatch.setattr(detector_module,'attach_detector_cache',
                        lambda engine,root,runtime,output_factory=None:
                        original_attach(engine,root,runtime,output_factory or SimpleNamespace))
    root=image_bundle(tmp_path,[picture() for _ in range(9)])
    engine=DetectorEngine()
    @contextmanager
    def runtime():
        yield engine, {'model_actual':'synthetic','detector_cache_policy':POLICY}, SimpleNamespace, normalize
    result=run_ocr_batch(root,indexes=list(range(1,10)),runtime_factory=runtime,
                         policy_identity={'model':'synthetic'},guard=lambda:None)
    assert result['model_loads']==1
    assert result['detector_calls']==1 and result['reused_detector_inputs']==8
    assert result['detector_inputs']==9 and len(result['subbatches'])==2
    for child in result['subbatches']:
        assert (root/child['receipt']).is_file()
    again=run_ocr_batch(root,indexes=list(range(1,10)),runtime_factory=runtime,
                        policy_identity={'model':'synthetic'},guard=lambda:None,resume=True)
    assert again['model_loads']==0 and again['detector_calls']==0


def test_actual_rapidocr_detector_crop_mapping_contract_without_models(tmp_path):
    """Execute fixed local upstream pure functions, not a copied OCR algorithm."""
    import ast
    import copy
    import os
    from pathlib import Path
    from dataclasses import dataclass
    import cv2
    from video_knowledge_pipeline.courseware_detector_cache import attach_detector_cache
    source_root=os.environ.get('VKP_RAPIDOCR_SOURCE')
    if not source_root:
        pytest.skip('explicit local RapidOCR source required; no downloads')
    source=Path(source_root)/'python/rapidocr'
    main=ast.parse((source/'main.py').read_text(encoding='utf-8-sig'))
    upstream=next(n for n in main.body if isinstance(n,ast.ClassDef) and n.name=='RapidOCR')
    upstream.body=[n for n in upstream.body if isinstance(n,ast.FunctionDef)
                   and n.name in {'detect_and_crop','crop_text_regions'}]
    process=ast.parse((source/'utils/process_img.py').read_text(encoding='utf-8-sig'))
    functions=[n for n in process.body if isinstance(n,ast.FunctionDef)
               and n.name in {'apply_vertical_padding','get_rotate_crop_image','map_boxes_to_original'}]
    types=ast.parse((source/'ch_ppocr_det/utils.py').read_text(encoding='utf-8-sig'))
    output_class=next(n for n in types.body if isinstance(n,ast.ClassDef) and n.name=='TextDetOutput')
    assert len(upstream.body)==2 and len(functions)==3
    selected=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),
                             *functions,output_class,upstream],type_ignores=[])
    namespace={'np':np,'cv2':cv2,'copy':copy,'dataclass':dataclass,'RapidOCRError':RuntimeError}
    exec(compile(ast.fix_missing_locations(selected),str(source/'main.py'),'exec'),namespace)
    engine=namespace['RapidOCR']()
    engine.width_height_ratio=-1
    engine.min_height=0
    detector=DetectorEngine()
    engine.text_det=detector.detect
    root=image_bundle(tmp_path,[picture(),picture()])
    candidates=read_scan(root)['candidates']
    data=np.asarray(picture())
    with attach_detector_cache(engine,root,{'model_actual':'synthetic','detector_cache_policy':POLICY},
                               namespace['TextDetOutput']) as cache:
        crops=[]
        for candidate in candidates:
            cache.begin_frame(candidate)
            op={'preprocess':{'ratio_h':1.0,'ratio_w':1.0}}
            current,detected=engine.detect_and_crop(data,op)
            assert isinstance(detected,namespace['TextDetOutput'])
            crops.append([im.copy() for im in current])
            native=detected.boxes.copy()
            op['preprocess']={'ratio_h':.5,'ratio_w':.5}
            namespace['map_boxes_to_original'](detected.boxes,op,20,40)  # mutates upstream output
            assert not np.array_equal(detected.boxes,native)
        assert detector.detections==1 and cache.reused_inputs==1
        assert all(np.array_equal(a,b) for a,b in zip(*crops))
