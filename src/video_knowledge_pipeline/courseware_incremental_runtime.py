"""Offline, explicit runtime gate for the incremental RapidOCR adapter."""
from __future__ import annotations

import importlib.metadata
import importlib.util
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess
import json
import sys
from unittest.mock import patch

from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file
from .storage import read_json


def _ocr_python():
    """Reuse ebook's selected interpreter; read only its non-secret runtime key."""
    from .path_defaults import workspace_root
    ebook_root = workspace_root()/'ebook_markdown_pipeline'
    sys.path.insert(0,str(ebook_root.parent))
    from ebook_markdown_pipeline.local_env import parse_env_line
    selected = os.environ.get('EBOOK_CONVERTER_RAPIDOCR_PYTHON','')
    env_file = ebook_root/'.env'
    if not selected and env_file.is_file():
        for line in env_file.read_text(encoding='utf-8-sig').splitlines():
            if 'EBOOK_CONVERTER_RAPIDOCR_PYTHON' not in line:
                continue
            entry = parse_env_line(line)
            if entry and entry[0] == 'EBOOK_CONVERTER_RAPIDOCR_PYTHON':
                selected = entry[1]
    path = Path(selected or sys.executable)
    if not path.is_file():
        raise ValueError('selected ebook OCR interpreter missing')
    return str(path.absolute())


def model_policy(path):
    path = Path(path).absolute()
    if path.stat().st_size > 32768:
        raise ValueError('model policy budget')
    value = read_json(path)
    if value.get('schema') != 'video_knowledge_pipeline.incremental_ocr_models.v1':
        raise ValueError('explicit incremental OCR model policy required')
    from .courseware_incremental_ocr import CACHE_POLICIES, EXACT_CROP
    cache_policy = value.get('recognition_cache_policy', EXACT_CROP)
    if not isinstance(cache_policy, str) or cache_policy not in CACHE_POLICIES:
        raise ValueError('unsupported recognition cache policy')
    from .courseware_detector_cache import DETECTOR_CACHE_POLICIES, DISABLED
    detector_policy = value.get('detector_cache_policy', DISABLED)
    if not isinstance(detector_policy, str) or detector_policy not in DETECTOR_CACHE_POLICIES:
        raise ValueError('unsupported detector cache policy')
    models = value.get('models',{})
    if set(models) != {'Det','Cls','Rec'} or not value.get('rapidocr_version'):
        raise ValueError('pin detector/classifier/recognizer and installed version')
    paths = {}
    for kind, record in models.items():
        target = Path(record['path'])
        target = target if target.is_absolute() else path.parent/target
        # Reparse points are forbidden, including parents. Models remain read-only.
        for part in (target,*target.parents):
            if part.is_symlink() or (part.exists() and getattr(part.stat(),'st_file_attributes',0) & 0x400):
                raise ValueError('model reparse path')
        if not target.is_file() or target.suffix.lower() != '.onnx' or target.stat().st_size > 256*1024*1024:
            raise ValueError('missing/bounded local ONNX model required')
        if sha256_file(target) != record.get('sha256'):
            raise ValueError('model hash drift')
        paths[kind] = str(target.absolute())
    return value, paths


def preflight(policy_path):
    value, paths = model_policy(policy_path)
    blockers = []
    interpreter = _ocr_python()
    for module in ('psutil',):
        if importlib.util.find_spec(module) is None:
            blockers.append(f'{module} unavailable in selected interpreter')
    try:
        probe = subprocess.run([interpreter,'-B','-c',
            'import importlib.metadata; print(importlib.metadata.version("rapidocr"))'],
            capture_output=True,text=True,timeout=10,check=True)
        if probe.stdout.strip() != value['rapidocr_version']:
            blockers.append('RapidOCR version drift')
    except (OSError,subprocess.SubprocessError):
        blockers.append('RapidOCR unavailable in ebook selected interpreter')
    if importlib.util.find_spec('psutil') is not None:
        import psutil
        # A serial owner's waiting parent is not a second OCR worker. Any
        # sibling/unrelated worker still blocks; inability to verify fails closed.
        ancestor_ids = {p.pid for p in psutil.Process(os.getpid()).parents()}
        if psutil.virtual_memory().available < 8*1024**3 or psutil.cpu_percent(interval=.2) >= 75:
            blockers.append('RAM/CPU resource gate')
        for process in psutil.process_iter(['pid','name','cmdline']):
            if process.pid == os.getpid() or process.pid in ancestor_ids:
                continue
            command = ' '.join(process.info.get('cmdline') or [])
            if any(name in command for name in ('full_acceptance_run.py','ocr_subbatch.py','rapidocr_worker.py',
                                                'courseware-incremental-ocr','courseware_incremental_runtime')):
                blockers.append('existing OCR runner active; do not overlap')
                break
    result = {'ready':False,'blockers':blockers,'model_paths':paths,'policy':value,
              'ocr_python':interpreter,
              'provider_calls':0,'automatic_download':False,
              'recognition_cache':value.get('recognition_cache_policy','exact_current_crop'),
              'approximate_text_reuse':False,
              'detector_cache_policy':value.get('detector_cache_policy','disabled'),
              'detector_always_runs':value.get('detector_cache_policy','disabled') == 'disabled',
              'changed_detector_inputs_always_run':True,
              'changed_region_local_detection':value.get('detector_cache_policy') == 'changed_regions_v1',
              'full_detector_equivalence':value.get('detector_cache_policy') != 'changed_regions_v1',
              'automatic_roi_exclusion':False,'concurrency':1,'execution_performed':False}
    if blockers:
        return result
    lms = shutil.which('lms')
    if not lms:
        blockers.append('LM Studio mutual exclusion probe unavailable')
    else:
        try:
            output = subprocess.run([lms,'ps','--json'],capture_output=True,text=True,check=True,timeout=10)
            models = json.loads(output.stdout)
            if not isinstance(models,list) or models:
                blockers.append('VLM resident or unknown state; no OCR overlap')
        except (subprocess.SubprocessError,ValueError,OSError):
            blockers.append('LM Studio mutual exclusion probe failed')
    return {**result,'ready':not blockers}


def execute(bundle_dir, indexes, policy_path, *, resume=False, reuse_worker=False, refresh_material=False):
    from .courseware_material import read_scan, _check_frame, _page_id
    maximum = 32 if reuse_worker else 8
    if (not isinstance(indexes,list) or not 1 <= len(indexes) <= maximum
            or any(type(i) is not int or i < 1 for i in indexes) or len(set(indexes)) != len(indexes)):
        raise ValueError(f'incremental OCR requires 1..{maximum} explicit indexes')
    if refresh_material and not reuse_worker:
        raise ValueError('refresh-material requires explicit worker reuse')
    state = read_scan(bundle_dir)
    for i in indexes:
        if i > len(state['candidates']):
            raise ValueError('incremental OCR index out of range')
        _page_id(state['candidates'][i-1]['page_id'])
        _check_frame(Path(bundle_dir),state['candidates'][i-1])
    status = preflight(policy_path)
    if not status['ready']:
        raise ValueError('; '.join(status['blockers']))
    from .storage import bundle_write_lock
    with bundle_write_lock(bundle_dir,operation='incremental_ocr_runtime',lock_name='.incremental-runtime.lock'):
        if Path(status['ocr_python']).resolve() != Path(sys.executable).resolve():
            env = os.environ.copy()
            # Do not inject controller-version binary site-packages into OCR Python.
            env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
            env['OMP_NUM_THREADS'] = '2'
            if reuse_worker:
                script = ('import json,sys; from video_knowledge_pipeline.courseware_incremental_runtime import _execute_local_batch; '
                          'print(json.dumps(_execute_local_batch(sys.argv[1],json.loads(sys.argv[2]),sys.argv[3],'
                          'resume=sys.argv[4]=="1",refresh_material=sys.argv[5]=="1",'
                          'controller_python=sys.argv[6],controller_path=sys.argv[7])))')
                extra = ['1' if refresh_material else '0',sys.executable,os.environ.get('PYTHONPATH','')]
            else:
                script = ('import json,sys; from video_knowledge_pipeline.courseware_incremental_runtime import _execute_local; '
                          'print(json.dumps(_execute_local(sys.argv[1],json.loads(sys.argv[2]),sys.argv[3],resume=sys.argv[4]=="1")))')
                extra = []
            child = subprocess.run([status['ocr_python'],'-B','-c',script,str(Path(bundle_dir).absolute()),
                json.dumps(indexes),str(Path(policy_path).absolute()),'1' if resume else '0',*extra],
                env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=900 if reuse_worker else 180)
            if child.returncode:
                raise RuntimeError('incremental OCR child failed: '+child.stderr[-1200:])
            return json.loads(child.stdout.strip().splitlines()[-1])
        if reuse_worker:
            return _execute_local_batch(bundle_dir,indexes,policy_path,resume=resume,
                                        refresh_material=refresh_material)
        return _execute_local(bundle_dir,indexes,policy_path,resume=resume)


@contextmanager
def local_runtime(policy_path):
    """Existing pinned engine construction, reusable for a bounded worker life."""
    policy, paths = model_policy(policy_path)
    if importlib.metadata.version('rapidocr') != policy['rapidocr_version']:
        raise ValueError('worker RapidOCR version drift')
    from .path_defaults import workspace_root
    ebook_root = workspace_root()/'ebook_markdown_pipeline'
    source = ebook_root/'ocr_providers.py'
    if not source.is_file():
        raise ValueError('existing ebook OCR normalizer unavailable')
    sys.path.insert(0,str(ebook_root.parent))
    from ebook_markdown_pipeline.ocr_providers import normalize_rapidocr_blocks
    from rapidocr import RapidOCR
    from rapidocr.ch_ppocr_rec import TextRecOutput
    params = {f'{kind}.model_path':path for kind,path in paths.items()}
    params.update({'EngineConfig.onnxruntime.use_cuda':False,
                   'EngineConfig.onnxruntime.intra_op_num_threads':2,
                   'EngineConfig.onnxruntime.inter_op_num_threads':1,
                   'Global.return_word_box':False})
    runtime = {'model_actual':'RapidOCR '+policy['rapidocr_version'],
               'policy_sha256':canonical_json_sha256(policy),
               'normalizer_sha256':sha256_file(source),'adapter':'recognition-crop-cache-v1',
               'geometry_contract':'rapidocr_final_original_pixels.v1',
               'geometry_adapter_sha256':sha256_file(Path(__file__).with_name('courseware_ocr_geometry.py')),
               'params_sha256':canonical_json_sha256(params)}
    if 'recognition_cache_policy' in policy:
        runtime['recognition_cache_policy'] = policy['recognition_cache_policy']
    if 'detector_cache_policy' in policy:
        runtime['detector_cache_policy'] = policy['detector_cache_policy']
        runtime['detector_adapter_sha256'] = sha256_file(Path(__file__).with_name('courseware_detector_cache.py'))
    # Same-process runtime only. Missing assets cannot initiate network retrieval.
    with patch('socket.socket.connect',side_effect=RuntimeError('incremental OCR is offline-only')):
        engine = RapidOCR(params=params)
        yield engine, runtime, TextRecOutput, normalize_rapidocr_blocks


def _execute_local(bundle_dir, indexes, policy_path, *, resume=False):
    from .courseware_incremental_ocr import run_incremental_courseware_ocr
    with local_runtime(policy_path) as (engine, runtime, output_factory, normalize):
        return run_incremental_courseware_ocr(bundle_dir,indexes=indexes,engine=engine,
            runtime_identity=runtime,output_factory=output_factory,
            normalize_blocks=normalize,resume=resume)


def _execute_local_batch(bundle_dir, indexes, policy_path, *, resume=False, refresh_material=False,
                         controller_python=None, controller_path=None):
    from .courseware_ocr_batch import run_ocr_batch
    from .path_defaults import workspace_root
    normalizer = workspace_root()/'ebook_markdown_pipeline'/'ocr_providers.py'
    policy, _ = model_policy(policy_path)
    policy_hash = canonical_json_sha256(policy)
    policy_identity = {'models': policy_hash, 'runtime_adapter': sha256_file(Path(__file__)),
                       'geometry_adapter': sha256_file(Path(__file__).with_name('courseware_ocr_geometry.py')),
                       'normalizer': sha256_file(normalizer)}
    from .courseware_detector_cache import DISABLED, CHANGED_REGIONS
    if policy.get('detector_cache_policy', DISABLED) != DISABLED:
        policy_identity['detector_adapter'] = sha256_file(Path(__file__).with_name('courseware_detector_cache.py'))
    if policy.get('detector_cache_policy') == CHANGED_REGIONS:
        policy_identity['local_detector_adapter'] = sha256_file(Path(__file__).with_name('courseware_local_detection.py'))
        policy_identity['detector_budget_adapter'] = sha256_file(Path(__file__).with_name('courseware_detector_budget.py'))

    def guard():
        # The OCR venv need not contain psutil. Reuse the controller's existing
        # preflight; the waiting owner/worker are actual ancestors, not siblings.
        if controller_python:
            env = os.environ.copy()
            env['PYTHONPATH'] = controller_path or ''
            script = ('import json,sys; from video_knowledge_pipeline.courseware_incremental_runtime import preflight; '
                      'print(json.dumps(preflight(sys.argv[1])))')
            output = subprocess.run([controller_python,'-B','-c',script,str(policy_path)],
                env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',check=True,timeout=40)
            result = json.loads(output.stdout.strip().splitlines()[-1])
        else:
            result = preflight(policy_path)
        if not result['ready']:
            raise ValueError('; '.join(result['blockers']))
        if (canonical_json_sha256(result['policy']) != policy_hash
                or ('detector_adapter' in policy_identity and
                    sha256_file(Path(__file__).with_name('courseware_detector_cache.py')) != policy_identity['detector_adapter'])
                or ('local_detector_adapter' in policy_identity and
                    sha256_file(Path(__file__).with_name('courseware_local_detection.py')) != policy_identity['local_detector_adapter'])
                or ('detector_budget_adapter' in policy_identity and
                    sha256_file(Path(__file__).with_name('courseware_detector_budget.py')) != policy_identity['detector_budget_adapter'])
                or sha256_file(Path(__file__).with_name('courseware_ocr_geometry.py')) != policy_identity['geometry_adapter']
                or sha256_file(normalizer) != policy_identity['normalizer']
                or sha256_file(Path(__file__)) != policy_identity['runtime_adapter']):
            raise ValueError('worker model policy drift')

    return run_ocr_batch(bundle_dir,indexes=indexes,runtime_factory=lambda:local_runtime(policy_path),
        policy_identity=policy_identity,guard=guard,resume=resume,refresh_material=refresh_material)
