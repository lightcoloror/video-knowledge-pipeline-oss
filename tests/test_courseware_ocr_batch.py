from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from test_courseware_dedupe import image_bundle
from test_courseware_incremental_ocr import Engine, normalize, picture
from video_knowledge_pipeline.storage import read_json


def batch(root, indexes, *, fail_call=0, guard=lambda: None, **kwargs):
    from video_knowledge_pipeline.courseware_ocr_batch import run_ocr_batch
    loads = []

    @contextmanager
    def factory():
        engine = Engine(fail_call=fail_call)
        loads.append(engine)
        yield engine, {'model_actual': 'synthetic-v1'}, SimpleNamespace, normalize

    return run_ocr_batch(root, indexes=indexes, runtime_factory=factory,
                         policy_identity='synthetic-pinned', guard=guard, **kwargs), loads


def test_one_model_load_for_three_checkpoint_groups(tmp_path):
    root = image_bundle(tmp_path, [picture()] * 17)
    result, loads = batch(root, list(range(1, 18)))
    assert len(loads) == result['model_loads'] == 1
    assert len(result['subbatches']) == 3
    assert result['detector_calls'] == 17 and result['recognized_regions'] == 2
    assert result['material_refresh'] == 'deferred'
    assert result['review_required'] and not result['complete']
    assert 'recognition' in result['timings_seconds']
    assert result['timings_seconds']['file_hashing'] > 0
    assert result['timings_seconds']['page_import'] > 0


def test_resume_retains_successes_and_only_does_missing_work(tmp_path):
    root = image_bundle(tmp_path, [picture()] * 8 + [picture(b=21)])
    with pytest.raises(RuntimeError, match='injected'):
        batch(root, list(range(1, 10)), fail_call=2)
    first = (root / 'courseware/pages/page-0.json').read_bytes()
    failed = list((root / 'courseware/incremental-batches').glob('*.json'))
    assert len(failed) == 1 and read_json(failed[0])['status'] == 'blocked'
    result, loads = batch(root, list(range(1, 10)), resume=True)
    assert loads[0].regions == 1
    assert result['detector_calls'] == 1
    assert (root / 'courseware/pages/page-0.json').read_bytes() == first
    assert (root / result['previous_attempt']).is_file()
    again, loads = batch(root, list(range(1, 10)), resume=True)
    assert again['detector_calls'] == 0 and not loads  # no engine for all-done resume


@pytest.mark.parametrize('indexes', [[], [True], [1, 1], [0], [99], list(range(1, 34))])
def test_invalid_selection_never_loads_model(tmp_path, indexes):
    root = image_bundle(tmp_path, [picture()])
    with pytest.raises(ValueError):
        batch(root, indexes, guard=lambda: pytest.fail('invalid selection reached guard'))


def test_resource_pause_between_groups_keeps_checkpoint(tmp_path):
    root = image_bundle(tmp_path, [picture()] * 9)
    calls = []
    def guard():
        calls.append(True)
        if len(calls) == 2:
            raise ValueError('RAM gate')
    with pytest.raises(ValueError, match='RAM gate'):
        batch(root, list(range(1, 10)), guard=guard)
    assert (root / 'courseware/pages/page-7.json').exists()
    assert not (root / 'courseware/pages/page-8.json').exists()
    result, _ = batch(root, list(range(1, 10)), resume=True)
    assert result['detector_calls'] == 1


def test_input_drift_refuses_old_resume(tmp_path):
    root = image_bundle(tmp_path, [picture()])
    batch(root, [1])
    (root / 'timeline.json').write_text('[]\n ')
    with pytest.raises(ValueError, match='resume'):
        batch(root, [1], resume=True)


def test_guard_and_model_failure_have_receipt(tmp_path):
    root = image_bundle(tmp_path, [picture()])
    def fail():
        raise ValueError('resource blocked')
    with pytest.raises(ValueError, match='resource blocked'):
        batch(root, [1], guard=fail)
    path = next((root / 'courseware/incremental-batches').glob('*.json'))
    receipt = read_json(path)
    assert receipt['status'] == 'blocked' and receipt['model_loads'] == 0
    assert receipt['automatic_retries'] == 0


def test_final_refresh_once_and_never_claims_acceptance(tmp_path, monkeypatch):
    root = image_bundle(tmp_path, [picture()] * 9)
    import video_knowledge_pipeline.courseware_ocr_batch as module
    calls = []
    monkeypatch.setattr(module, 'build_courseware_material', lambda *a, **k: calls.append(True) or {'revision': 'test'})
    result, _ = batch(root, list(range(1, 10)), refresh_material=True)
    assert calls == [True]
    assert result['material_refresh'] == 'full_verified'
    assert not result['complete'] and result['review_required']


def test_collect_cli_forwards_bounded_deferred_refresh(monkeypatch, capsys):
    import video_knowledge_pipeline.courseware_cli as cli
    calls = []
    monkeypatch.setattr(cli, 'collect_courseware_extractions',
        lambda *a, **k: calls.append(k) or {'ok': True})
    assert cli.main(['courseware-collect', 'fixture', '--executor', 'synthetic',
                     '--model-actual', 'synthetic', '--indexes', '1,3', '--defer-material']) == 0
    assert calls[0]['indexes'] == [1, 3] and calls[0]['refresh_material'] is False
    assert 'deferred' in capsys.readouterr().out


def test_worker_flag_keeps_legacy_default(monkeypatch):
    import video_knowledge_pipeline.courseware_cli as cli
    import video_knowledge_pipeline.courseware_incremental_runtime as runtime
    calls = []
    monkeypatch.setattr(runtime, 'execute', lambda *a, **k: calls.append(k) or {'ok': True})
    args = ['courseware-incremental-ocr', 'fixture', '--indexes', '1,2', '--model-policy', 'p', '--execute']
    assert cli.main(args) == 0
    assert not calls[0].get('reuse_worker')
    assert cli.main([*args, '--reuse-worker']) == 0
    assert calls[1]['reuse_worker'] is True


def test_old_and_reused_worker_produce_identical_page_content(tmp_path):
    from test_courseware_incremental_ocr import run
    from video_knowledge_pipeline.file_hash import sha256_file
    pictures = [picture()] * 8 + [picture(b=21)] * 8 + [picture(b=255)]
    (tmp_path / 'old').mkdir()
    (tmp_path / 'new').mkdir()
    old = image_bundle(tmp_path / 'old', pictures)
    new = image_bundle(tmp_path / 'new', pictures)
    protected = [new / 'manifest.json', new / 'timeline.json', *sorted((new / 'courseware/frames').glob('*'))]
    assert len(protected) == 19
    hashes = {p: sha256_file(p) for p in protected}
    old_loads = 0
    for offset in range(0, 17, 8):
        old_loads += 1
        run(old, Engine(), list(range(offset + 1, min(offset + 9, 18))), runtime={'model_actual': 'synthetic-v1'})
    result, loads = batch(new, list(range(1, 18)))
    for i in range(17):
        a = read_json(old / f'courseware/pages/page-{i}.json')
        b = read_json(new / f'courseware/pages/page-{i}.json')
        for field in ('status', 'blocks', 'gaps', 'time', 'frame_sha256', 'executor', 'model_actual'):
            assert a[field] == b[field]
    assert old_loads == 3 and len(loads) == result['model_loads'] == 1
    assert hashes == {p: sha256_file(p) for p in protected}


def test_instrumentation_restores_hooks_on_failure():
    from video_knowledge_pipeline.courseware_ocr_batch import instrument_engine
    engine = Engine(fail_call=1)
    original = engine.recognize_txt
    timings = {}
    with pytest.raises(RuntimeError):
        with instrument_engine(engine, timings):
            engine.recognize_txt([])
    assert engine.recognize_txt == original
    assert timings['recognition'] >= 0


def test_formal_result_drift_blocks_before_model(tmp_path):
    from video_knowledge_pipeline.storage import write_json
    root = image_bundle(tmp_path, [picture()])
    batch(root, [1])
    path = root / 'courseware/pages/page-0.json'
    page = read_json(path)
    page['blocks'][0]['text'] = 'changed without proof'
    write_json(path, page)
    with pytest.raises(ValueError, match='mismatch'):
        batch(root, [1], resume=True, guard=lambda: pytest.fail('must validate formal result first'))


def test_model_init_failure_receipt_retained(tmp_path):
    from video_knowledge_pipeline.courseware_ocr_batch import run_ocr_batch
    root = image_bundle(tmp_path, [picture()])
    @contextmanager
    def fail():
        raise RuntimeError('load failed')
        yield  # pragma: no cover
    with pytest.raises(RuntimeError, match='load failed'):
        run_ocr_batch(root, indexes=[1], runtime_factory=fail,
                      policy_identity='synthetic', guard=lambda: None)
    receipt = read_json(next((root / 'courseware/incremental-batches').glob('*.json')))
    assert receipt['status'] == 'blocked' and receipt['model_loads'] == 0


def test_unknown_policy_resume_does_not_infer_new_run(tmp_path):
    from video_knowledge_pipeline.courseware_ocr_batch import run_ocr_batch
    root = image_bundle(tmp_path, [picture()])
    batch(root, [1])
    with pytest.raises(ValueError, match='resume identity'):
        run_ocr_batch(root, indexes=[1], runtime_factory=lambda: pytest.fail('no engine'),
                      policy_identity='different-model', guard=lambda: None, resume=True)


def test_runtime_forwards_batch_to_one_child_and_preserves_python_env(tmp_path, monkeypatch):
    import os
    import video_knowledge_pipeline.courseware_incremental_runtime as runtime
    root = image_bundle(tmp_path, [picture()] * 9)
    monkeypatch.setattr(runtime, 'preflight', lambda p: {'ready': True, 'ocr_python': 'separate-ocr-python'})
    monkeypatch.setenv('PYTHONPATH', 'controller-only-deps')
    calls = []
    def child(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"status":"completed_review_required"}', stderr='')
    monkeypatch.setattr(runtime.subprocess, 'run', child)
    result = runtime.execute(root, list(range(1, 10)), tmp_path / 'models.json', reuse_worker=True)
    assert result['status'] == 'completed_review_required'
    argv, kwargs = calls[0]
    assert len(calls) == 1 and '_execute_local_batch' in argv[3]
    assert kwargs['timeout'] == 900
    assert 'controller-only-deps' not in kwargs['env']['PYTHONPATH']
    assert argv[-1] == os.environ['PYTHONPATH']
    assert kwargs['env']['OMP_NUM_THREADS'] == '2'


def test_legacy_eight_frame_bound_unchanged(tmp_path, monkeypatch):
    import video_knowledge_pipeline.courseware_incremental_runtime as runtime
    root = image_bundle(tmp_path, [picture()] * 9)
    monkeypatch.setattr(runtime, 'preflight', lambda p: pytest.fail('must reject before preflight'))
    with pytest.raises(ValueError, match='1..8'):
        runtime.execute(root, list(range(1, 10)), 'unused-policy')


def test_deferred_collection_requires_exact_indexes():
    from video_knowledge_pipeline.courseware_cli import main
    assert main(['courseware-collect', 'fixture', '--executor', 'synthetic',
                 '--model-actual', 'synthetic', '--defer-material']) == 2


def test_local_factory_pins_cpu_models_and_reuses_one_instance(tmp_path, monkeypatch):
    import socket
    import sys
    from types import ModuleType
    import video_knowledge_pipeline.courseware_incremental_runtime as runtime
    import video_knowledge_pipeline.path_defaults as paths
    root = image_bundle(tmp_path, [picture()] * 9)
    ebook = tmp_path / 'ebook_markdown_pipeline'
    ebook.mkdir()
    (ebook / 'ocr_providers.py').write_text('# synthetic normalizer identity')
    monkeypatch.setattr(paths, 'workspace_root', lambda: tmp_path)
    policy = {'rapidocr_version': 'test-version'}
    monkeypatch.setattr(runtime, 'model_policy', lambda p: (policy, {k: k + '.onnx' for k in ('Det', 'Cls', 'Rec')}))
    monkeypatch.setattr(runtime, 'preflight', lambda p: {'ready': True, 'policy': policy})
    monkeypatch.setattr(runtime.importlib.metadata, 'version', lambda p: 'test-version')
    providers = ModuleType('ebook_markdown_pipeline.ocr_providers')
    def normalize_final(raw):
        rows = normalize(raw)
        for i, row in enumerate(rows):
            row['index'] = i + 1
            box = raw.boxes[i]
            row['bbox'] = [round(float(box[:, 0].min()), 2), round(float(box[:, 1].min()), 2),
                           round(float(box[:, 0].max()), 2), round(float(box[:, 1].max()), 2)]
        return rows
    providers.normalize_rapidocr_blocks = normalize_final
    parent = ModuleType('ebook_markdown_pipeline')
    parent.ocr_providers = providers
    monkeypatch.setitem(sys.modules, 'ebook_markdown_pipeline', parent)
    monkeypatch.setitem(sys.modules, 'ebook_markdown_pipeline.ocr_providers', providers)
    builds = []
    from test_courseware_ocr_geometry import GeometryEngine
    class LocalEngine(GeometryEngine):
        def __init__(self, *, params):
            super().__init__()
            builds.append(params)
            with socket.socket() as connection:
                with pytest.raises(RuntimeError, match='offline-only'):
                    connection.connect(('127.0.0.1', 1))  # mocked before OS connect
    rapidocr = ModuleType('rapidocr')
    rapidocr.RapidOCR = LocalEngine
    rec = ModuleType('rapidocr.ch_ppocr_rec')
    rec.TextRecOutput = SimpleNamespace
    monkeypatch.setitem(sys.modules, 'rapidocr', rapidocr)
    monkeypatch.setitem(sys.modules, 'rapidocr.ch_ppocr_rec', rec)
    report = runtime._execute_local_batch(root, list(range(1, 10)), 'synthetic-model-policy')
    assert len(builds) == report['model_loads'] == 1
    assert builds[0]['EngineConfig.onnxruntime.use_cuda'] is False
    assert builds[0]['EngineConfig.onnxruntime.intra_op_num_threads'] == 2
    assert set(k for k in builds[0] if k.endswith('.model_path')) == {'Det.model_path', 'Cls.model_path', 'Rec.model_path'}
    assert report['runtime']['model_actual'] == 'RapidOCR test-version'


def test_child_timeout_is_structured_and_never_retried(monkeypatch, capsys):
    import subprocess
    from video_knowledge_pipeline.courseware_cli import main
    import video_knowledge_pipeline.courseware_incremental_runtime as runtime
    calls = []
    def timeout(*a, **k):
        calls.append(True)
        raise subprocess.TimeoutExpired('synthetic-child', 1)
    monkeypatch.setattr(runtime, 'execute', timeout)
    assert main(['courseware-incremental-ocr', 'fixture', '--indexes', '1', '--model-policy', 'p',
                 '--execute', '--reuse-worker']) == 2
    assert calls == [True]
    assert '"status": "blocked"' in capsys.readouterr().out


def test_failed_group_metrics_are_included_once_and_resume_counts_only_new_work(tmp_path):
    from video_knowledge_pipeline.courseware_ocr_batch import COUNTERS
    from video_knowledge_pipeline.file_hash import sha256_file
    root = image_bundle(tmp_path, [picture()] * 8 + [picture(b=21)])
    with pytest.raises(RuntimeError, match='injected'):
        batch(root, list(range(1, 10)), fail_call=2)
    path = next((root / 'courseware/incremental-batches').glob('*.json'))
    failed = read_json(path)
    assert failed['detector_calls'] == 9
    assert failed['detector_frame_inputs'] == 9
    assert failed['metrics_incomplete'] is False
    assert len(failed['subbatches']) == 2
    child = failed['subbatches'][-1]
    assert child['indexes'] == [9] and child['status'] == 'blocked'
    assert child['counts_included'] is True
    child_path = root / child['receipt']
    captured = read_json(child_path)
    assert child['receipt_sha256'] == sha256_file(child_path)
    assert child['metrics'] == {key: captured[key] for key in COUNTERS}
    assert child['metrics']['detector_calls'] == 1
    resumed, _ = batch(root, list(range(1, 10)), resume=True)
    assert resumed['detector_calls'] == resumed['detector_frame_inputs'] == 1
    assert read_json(root / resumed['previous_attempt'])['detector_calls'] == 9
    again, _ = batch(root, list(range(1, 10)), resume=True)
    assert again['detector_calls'] == 0


@pytest.mark.parametrize('damage', ['missing', 'run_id', 'runtime', 'indexes', 'counter',
                                    'input_drift', 'unreadable', 'stale'])
def test_failed_child_metrics_require_current_matching_evidence_and_keep_original_error(tmp_path, monkeypatch, damage):
    import video_knowledge_pipeline.courseware_ocr_batch as module
    from video_knowledge_pipeline.storage import write_json
    from video_knowledge_pipeline.courseware_material import read_scan
    root = image_bundle(tmp_path, [picture()])
    runtime = {'model_actual': 'synthetic-v1'}
    child_id = module.incremental_run_id(read_scan(root)['identity'], [1], runtime)
    child_path = root / f'courseware/incremental-runs/{child_id}.json'
    captured = {'schema': module.courseware_incremental_ocr.SCHEMA, 'run_id': child_id,
                'runtime': runtime, 'indexes': [1], 'status': 'blocked', 'failure_index': 1,
                **dict.fromkeys(module.COUNTERS, 0)}
    captured.update(detector_calls=2, detector_inputs=2, detector_frame_inputs=1,
                    local_detector_fallbacks=1)
    if damage == 'stale':
        write_json(child_path, captured)
    # An unrelated receipt must never be used to fill missing current evidence.
    write_json(root / 'courseware/incremental-runs/unrelated.json', captured)
    original = RuntimeError('injected child failure retained')
    def failed_child(*args, **kwargs):
        if damage not in {'missing', 'stale'}:
            if damage == 'run_id': captured['run_id'] = 'wrong-child'
            if damage == 'runtime': captured['runtime'] = {'model_actual': 'different'}
            if damage == 'indexes': captured['indexes'] = [2]
            if damage == 'counter': captured['detector_calls'] = True
            write_json(child_path, captured)
            if damage == 'input_drift':
                write_json(root / 'timeline.json', {'synthetic_drift': True})
            if damage == 'unreadable':
                child_path.write_text('{broken', encoding='utf-8')
        raise original
    monkeypatch.setattr(module, 'run_incremental_courseware_ocr', failed_child)
    with pytest.raises(RuntimeError, match='retained') as raised:
        batch(root, [1])
    assert raised.value is original
    report = read_json(next((root / 'courseware/incremental-batches').glob('*.json')))
    assert report['metrics_incomplete'] is True
    assert report['metrics_incomplete_details']
    assert report['detector_calls'] == 0  # Known subtotal, explicitly incomplete.
    assert not any(child.get('counts_included') for child in report['subbatches'])
    assert 'injected child failure retained' == report['error']


def test_failed_child_receipt_preserves_all_recorded_roi_call_counters(tmp_path, monkeypatch):
    import video_knowledge_pipeline.courseware_ocr_batch as module
    from video_knowledge_pipeline.storage import write_json
    from video_knowledge_pipeline.courseware_material import read_scan
    root = image_bundle(tmp_path, [picture()])
    expected = {name: i + 1 for i, name in enumerate(module.COUNTERS)}
    def failed_child(*args, **kwargs):
        child_id = module.incremental_run_id(read_scan(root)['identity'], kwargs['indexes'],
                                             kwargs['runtime_identity'])
        write_json(root / f'courseware/incremental-runs/{child_id}.json',
            {'schema': module.courseware_incremental_ocr.SCHEMA, 'run_id': child_id,
             'runtime': kwargs['runtime_identity'], 'indexes': kwargs['indexes'],
             'status': 'blocked', 'failure_index': 1, **expected})
        raise RuntimeError('synthetic ROI then full failure')
    monkeypatch.setattr(module, 'run_incremental_courseware_ocr', failed_child)
    with pytest.raises(RuntimeError, match='ROI then full'):
        batch(root, [1])
    report = read_json(next((root / 'courseware/incremental-batches').glob('*.json')))
    assert report['metrics_incomplete'] is False
    assert {name: report[name] for name in module.COUNTERS} == expected
    assert report['subbatches'][0]['metrics'] == expected
