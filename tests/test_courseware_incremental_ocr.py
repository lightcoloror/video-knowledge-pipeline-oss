from types import SimpleNamespace
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from test_courseware_dedupe import image_bundle
from video_knowledge_pipeline.courseware_incremental_ocr import run_incremental_courseware_ocr
from video_knowledge_pipeline.courseware_material import read_scan, _validate_page
from video_knowledge_pipeline.storage import read_json


class Engine:
    return_word_box = False
    def __init__(self, fail_call=0):
        self.calls = 0
        self.regions = 0
        self.fail_call = fail_call

    def recognize_txt(self, images):
        self.calls += 1
        if self.calls == self.fail_call:
            raise RuntimeError('injected recognizer failure')
        self.regions += len(images)
        return SimpleNamespace(txts=tuple(f'Observed value {int(x[0,0,0])}; condition retained.' for x in images),
                               scores=[.99]*len(images))

    def __call__(self, path):
        with Image.open(path) as image:
            data=np.array(image.convert('RGB'))
        images=[data[:,x:x+20].copy() for x in (0,20) if data[0,x,0] != 255]
        if not images:
            return SimpleNamespace(txts=[],scores=[])
        return self.recognize_txt(images)


def normalize(raw):
    return [{'text':t,'score':s,'bbox':[i*20,0,20,20]} for i,(t,s) in enumerate(zip(raw.txts,raw.scores))]


def picture(a=10,b=20):
    data=np.full((20,40,3),255,dtype=np.uint8)
    data[:,:20]=a
    data[:,20:]=b
    return Image.fromarray(data)


def run(root,engine,indexes,**kwargs):
    return run_incremental_courseware_ocr(root,indexes=indexes,engine=engine,
        runtime_identity=kwargs.pop('runtime',{'model_actual':'synthetic-v1','weights_sha256':'a'*64}),
        output_factory=SimpleNamespace,normalize_blocks=normalize,**kwargs)


def test_current_detector_merge_reuses_regions_and_drops_deleted_text(tmp_path):
    root=image_bundle(tmp_path,[picture(),picture(b=21),picture(b=255)])
    engine=Engine()
    result=run(root,engine,[1,2,3])
    assert result['detector_calls']==3
    assert result['recognition_calls']==2
    assert result['recognized_regions']==3  # baseline 2+2+1 = 5
    final=read_json(root/'courseware/pages/page-2.json')
    assert len(final['blocks'])==1
    assert '10' in final['blocks'][0]['text']
    assert '20;' not in str(final['blocks']) and '21;' not in str(final['blocks'])
    assert final['incremental_ocr']['recognition_calls']==0
    assert result['status']=='completed_review_required'


def test_interruption_resume_only_missing_regions_and_preserves_formal_bytes(tmp_path):
    root=image_bundle(tmp_path,[picture(),picture(b=21),picture(b=21)])
    with pytest.raises(RuntimeError,match='injected'):
        run(root,Engine(fail_call=2),[1,2,3])
    first=(root/'courseware/pages/page-0.json').read_bytes()
    engine=Engine()
    result=run(root,engine,[1,2,3],resume=True)
    assert engine.regions==1
    assert result['recognition_calls']==1
    assert (root/'courseware/pages/page-0.json').read_bytes()==first
    assert (root/result['previous_attempt']).is_file()
    engine=Engine()
    again=run(root,engine,[1,2,3],resume=True)
    assert engine.calls==0 and again['detector_calls']==0


def test_new_runtime_invalidates_region_cache(tmp_path):
    root=image_bundle(tmp_path,[picture(),picture()])
    run(root,Engine(),[1])
    engine=Engine()
    run(root,engine,[2],runtime={'model_actual':'synthetic-v2'})
    assert engine.regions==2


def test_raw_drift_blocks_formal_validation(tmp_path):
    root=image_bundle(tmp_path,[picture()])
    run(root,Engine(),[1])
    value=read_json(root/'courseware/pages/page-0.json')
    Path(root/value['incremental_ocr']['raw_path']).write_text('{}')
    with pytest.raises(ValueError,match='hash drift'):
        _validate_page(root,read_scan(root),value)


def test_blank_frame_does_not_keep_prior_text(tmp_path):
    root=image_bundle(tmp_path,[picture(),picture(255,255)])
    run(root,Engine(),[1,2])
    value=read_json(root/'courseware/pages/page-1.json')
    assert value['blocks']==[] and value['status']=='unavailable' and value['gaps']


def test_cache_corruption_stops_before_recognition(tmp_path):
    root=image_bundle(tmp_path,[picture(),picture()])
    run(root,Engine(),[1])
    (root/'courseware/incremental-cache/regions/regions.json').write_text('{"rows":[],"sha256":"bad"}')
    engine=Engine()
    with pytest.raises(ValueError,match='hash drift'):
        run(root,engine,[2])
    assert engine.calls==0


def test_changed_source_refuses_reuse_and_leaves_target_absent(tmp_path):
    root=image_bundle(tmp_path,[picture(),picture()])
    run(root,Engine(),[1])
    candidate=read_scan(root)['candidates'][0]
    (root/candidate['frame']).write_bytes(b'changed')
    with pytest.raises(ValueError):
        run(root,Engine(),[2])
    assert not (root/'courseware/pages/page-1.json').exists()


def test_low_confidence_is_not_cached(tmp_path):
    root=image_bundle(tmp_path,[picture(),picture()])
    class LowEngine(Engine):
        def recognize_txt(self,images):
            result=super().recognize_txt(images)
            result.scores=[.3]*len(images)
            return result
    engine=LowEngine()
    run(root,engine,[1,2])
    assert engine.regions==4
    assert read_json(root/'courseware/pages/page-1.json')['gaps']


def test_failed_import_resumes_from_recognition_checkpoint(tmp_path,monkeypatch):
    root=image_bundle(tmp_path,[picture()])
    import video_knowledge_pipeline.courseware_incremental_ocr as module
    original=module.import_courseware_page
    def fail(*args):
        raise RuntimeError('injected import failure')
    monkeypatch.setattr(module,'import_courseware_page',fail)
    with pytest.raises(RuntimeError,match='injected'):
        run(root,Engine(),[1])
    monkeypatch.setattr(module,'import_courseware_page',original)
    engine=Engine()
    run(root,engine,[1],resume=True)
    assert engine.calls==0


def test_changed_runtime_cannot_resume_old_run(tmp_path):
    root=image_bundle(tmp_path,[picture()])
    run(root,Engine(),[1])
    with pytest.raises(ValueError,match='identity'):
        run(root,Engine(),[1],runtime={'model_actual':'different'},resume=True)


def test_cli_preview_does_not_execute(monkeypatch,capsys):
    from video_knowledge_pipeline.courseware_cli import main
    import video_knowledge_pipeline.courseware_incremental_runtime as runtime
    monkeypatch.setattr(runtime,'preflight',lambda path:{'ready':False,'execution_performed':False})
    monkeypatch.setattr(runtime,'execute',lambda *a,**k:pytest.fail('preview executed'))
    assert main(['courseware-incremental-ocr','synthetic','--indexes','1,2','--model-policy','models.json'])==0
    assert 'false' in capsys.readouterr().out


def test_model_policy_checks_hash_before_engine(tmp_path):
    from video_knowledge_pipeline.courseware_incremental_runtime import model_policy
    from video_knowledge_pipeline.file_hash import sha256_file
    from video_knowledge_pipeline.storage import write_json
    models={}
    for kind in ('Det','Rec','Cls'):
        file=tmp_path/(kind+'.onnx')
        file.write_bytes(b'synthetic-not-executable')
        models[kind]={'path':file.name,'sha256':sha256_file(file)}
    path=tmp_path/'models.json'
    write_json(path,{'schema':'video_knowledge_pipeline.incremental_ocr_models.v1','rapidocr_version':'3.8.1','models':models})
    assert set(model_policy(path)[1])=={'Det','Rec','Cls'}
    (tmp_path/'Det.onnx').write_bytes(b'drift')
    with pytest.raises(ValueError,match='hash drift'):
        model_policy(path)


def test_active_ocr_blocks_before_lms_or_model_start(monkeypatch):
    import sys
    import video_knowledge_pipeline.courseware_incremental_runtime as runtime
    monkeypatch.setattr(runtime,'model_policy',lambda p:({'rapidocr_version':'3.8.1'},{}))
    monkeypatch.setattr(runtime,'_ocr_python',lambda:'configured-python')
    monkeypatch.setattr(runtime.importlib.util,'find_spec',lambda name:True)
    probes=[]
    def probe(argv,**kwargs):
        probes.append(argv)
        assert argv[0]=='configured-python'
        return SimpleNamespace(stdout='3.8.1\n')
    monkeypatch.setattr(runtime.subprocess,'run',probe)
    fake=SimpleNamespace(virtual_memory=lambda:SimpleNamespace(available=16*1024**3),
        Process=lambda pid:SimpleNamespace(parents=lambda:[]),
        cpu_percent=lambda **k:1,process_iter=lambda args:[SimpleNamespace(pid=-1,info={'cmdline':['full_acceptance_run.py']})])
    monkeypatch.setitem(sys.modules,'psutil',fake)
    result=runtime.preflight('models.json')
    assert not result['ready'] and 'active' in result['blockers'][0]
    assert len(probes)==1 and not result['execution_performed']
    # The waiting serial owner is permitted only when it is an actual ancestor.
    fake.Process=lambda pid:SimpleNamespace(parents=lambda:[SimpleNamespace(pid=-1)])
    monkeypatch.setattr(runtime.shutil,'which',lambda name:'lms-test')
    def serial_probe(argv,**kwargs):
        return SimpleNamespace(stdout='[]' if argv[0]=='lms-test' else '3.8.1\n')
    monkeypatch.setattr(runtime.subprocess,'run',serial_probe)
    assert runtime.preflight('models.json')['ready']


def test_invalid_candidate_id_fails_before_recognizer(tmp_path):
    from video_knowledge_pipeline.courseware_material import store_scan
    root=image_bundle(tmp_path,[picture()])
    state=read_scan(root)
    state['candidates'][0]['page_id']='invalid'
    store_scan(root,state)
    engine=Engine()
    with pytest.raises(ValueError,match='page_id'):
        run(root,engine,[1])
    assert engine.calls==0


def test_manifest_drift_before_import_is_atomic(tmp_path):
    root=image_bundle(tmp_path,[picture()])
    class DriftEngine(Engine):
        def __call__(self,path):
            result=super().__call__(path)
            (root/'manifest.json').write_text('{"changed":true}')
            return result
    with pytest.raises(ValueError,match='manifest/timeline drift'):
        run(root,DriftEngine(),[1])
    assert not (root/'courseware/pages/page-0.json').exists()


@pytest.mark.parametrize('indexes', [[],[0],[1,1],[True],list(range(1,10)),[99]])
def test_bounded_indexes_fail_without_model(tmp_path,indexes):
    root=image_bundle(tmp_path,[picture()])
    engine=Engine()
    with pytest.raises(ValueError):
        run(root,engine,indexes)
    assert engine.calls==0
