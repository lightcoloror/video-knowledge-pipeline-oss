"""Bounded model reuse over existing eight-frame OCR checkpoints.

No new page writer, OCR algorithm or long-running service. The runtime caller
owns process/resource exclusion; this adapter retains existing import/restore
evidence and only adds an aggregate performance receipt.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from .canonical_json import canonical_json_sha256
from . import courseware_incremental_ocr
from .courseware_incremental_ocr import run_incremental_courseware_ocr, incremental_run_id
from .courseware_material import (
    _check_frame, _page_id, _validate_page, build_courseware_material, read_scan, safe_path,
)
from .file_hash import sha256_file
from .storage import bundle_write_lock, read_json, write_json

SCHEMA = 'video_knowledge_pipeline.courseware_ocr_batch.v1'
COUNTERS = ('detector_calls', 'recognition_calls', 'detected_regions',
            'recognized_regions', 'reused_regions', 'detector_inputs', 'reused_detector_inputs',
            'local_detector_frames', 'local_detector_fallbacks', 'detector_frame_inputs')
MAX_CHILD_RECEIPT_BYTES = 4 * 1024 * 1024


def selection_identity(root, indexes):
    if (not isinstance(indexes, list) or not 1 <= len(indexes) <= 32
            or any(type(i) is not int or i < 1 for i in indexes)
            or len(set(indexes)) != len(indexes)):
        raise ValueError('worker batch requires 1..32 unique explicit indexes')
    root = Path(root).absolute()
    state = read_scan(root)
    if max(indexes) > len(state['candidates']):
        raise ValueError('worker batch index outside scan')
    selected = [state['candidates'][i - 1] for i in indexes]
    for candidate in selected:
        _page_id(candidate['page_id'])
        _check_frame(root, candidate)
    return {'scan': state['identity'], 'selected': selected,
            'files': {name: sha256_file(safe_path(root, name))
                      for name in ('manifest.json', 'timeline.json')}}


def _measured(timings, name, function):
    @wraps(function)
    def call(*args, **kwargs):
        before = perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            timings[name] = timings.get(name, 0.0) + perf_counter() - before
    return call


@contextmanager
def instrument_engine(engine, timings):
    originals = {}
    try:
        for attr, label in (('detect_and_crop', 'detection_and_crop'),
                            ('cls_and_rotate', 'classification'),
                            ('recognize_txt', 'recognition')):
            function = getattr(engine, attr, None)
            if callable(function):
                originals[attr] = function
                setattr(engine, attr, _measured(timings, label, function))
        yield
    finally:
        for attr, function in originals.items():
            setattr(engine, attr, function)


@contextmanager
def instrument_integrity(timings):
    """Time existing functions, never memoize/skip hashes or change validation.

    Used only within the single serialized worker. These are inclusive timings:
    page_import includes file_hashing, as does checkpoint_groups_total.
    """
    from . import courseware_incremental_ocr, courseware_material
    with ExitStack() as stack:
        for module in (courseware_incremental_ocr, courseware_material):
            stack.enter_context(patch.object(module, 'sha256_file',
                _measured(timings, 'file_hashing', module.sha256_file)))
        stack.enter_context(patch.object(courseware_incremental_ocr, 'import_courseware_page',
            _measured(timings, 'page_import', courseware_incremental_ocr.import_courseware_page)))
        yield


def _child_receipt_marker(path):
    """Capture only this bounded child file; False means freshness is unknown."""
    try:
        if not path.exists():
            return None
        if not path.is_file() or path.stat().st_size > MAX_CHILD_RECEIPT_BYTES:
            return False
        return sha256_file(path)
    except Exception:
        return False


def _record_failed_child(report, root, child_id, part, runtime, indexes, inputs, previous_sha):
    """Keep verified current-attempt metrics without replacing the child error."""
    try:
        path = safe_path(root, f'courseware/incremental-runs/{child_id}.json', exists=False)
        receipt_sha = _child_receipt_marker(path)
        if receipt_sha is None:
            raise ValueError('current child receipt missing; actual counters unavailable')
        if receipt_sha is False or previous_sha is False:
            raise ValueError('current child receipt size/read/freshness unavailable')
        if receipt_sha == previous_sha:
            raise ValueError('current child receipt was not updated by this attempt')
        child = read_json(path)
        if _child_receipt_marker(path) != receipt_sha:
            raise ValueError('current child receipt changed while reading')
        if (not isinstance(child, dict) or child.get('schema') != courseware_incremental_ocr.SCHEMA
                or child.get('run_id') != child_id or child.get('status') != 'blocked'
                or not isinstance(child.get('indexes'), list)
                or any(type(i) is not int for i in child['indexes']) or child['indexes'] != part
                or type(child.get('failure_index')) is not int or child['failure_index'] not in part
                or not isinstance(child.get('runtime'), dict)
                or canonical_json_sha256(child['runtime']) != canonical_json_sha256(runtime)
                or incremental_run_id(inputs['scan'], part, runtime) != child_id):
            raise ValueError('current child receipt run/runtime/index identity mismatch')
        if selection_identity(root, indexes) != inputs:
            raise ValueError('current child input identity drift')
        metrics = {key: child.get(key) for key in COUNTERS}
        if any(type(value) is not int or value < 0 for value in metrics.values()):
            raise ValueError('current child counters missing or invalid')
    except Exception as exc:
        report['metrics_incomplete'] = True
        report['metrics_incomplete_details'].append({'indexes': list(part), 'run_id': child_id,
            'reason': str(exc)[:1000] or type(exc).__name__})
        return
    report['subbatches'].append({'indexes': list(part), 'run_id': child_id,
        'receipt': path.relative_to(root).as_posix(), 'receipt_sha256': receipt_sha,
        'status': child['status'], 'counts_included': True, 'metrics': metrics})
    for key, value in metrics.items():
        report[key] += value


def run_ocr_batch(bundle_dir, *, indexes, runtime_factory, policy_identity, guard,
                  resume=False, refresh_material=False):
    """Process <=32 frames serially, using exactly the old <=8-frame importer.

    Every pending group must pass guard before execution. No automatic retry.
    Finalized pages are validated before deciding whether a model is needed.
    Deferred material is explicitly stale, never evidence of full acceptance.
    """
    root = Path(bundle_dir).absolute()
    started = perf_counter()
    before = perf_counter()
    identity = selection_identity(root, indexes)
    timings = {'input_validation': perf_counter() - before, 'model_load': 0.0,
               'resource_preflight': 0.0, 'detection_and_crop': 0.0,
               'classification': 0.0, 'recognition': 0.0, 'normalization': 0.0,
               'file_hashing': 0.0, 'page_import': 0.0,
               'checkpoint_groups_total': 0.0, 'material_refresh': 0.0}
    contract = {'inputs': identity, 'indexes': indexes, 'policy': policy_identity,
                'checkpoint_adapter_sha256': sha256_file(Path(courseware_incremental_ocr.__file__)),
                'adapter_sha256': sha256_file(Path(__file__))}
    job_id = canonical_json_sha256(contract)[:24]
    receipt_path = safe_path(root, f'courseware/incremental-batches/{job_id}.json', exists=False)
    report = {'schema': SCHEMA, 'run_id': job_id, 'contract': contract,
              'status': 'running', 'indexes': indexes, 'model_loads': 0,
              'subbatches': [], 'automatic_retries': 0, 'review_required': True,
              'metrics_scope': 'current_attempt_only; excludes previous_attempt',
              'metrics_incomplete': False, 'metrics_incomplete_details': [],
              'complete': False, 'material_refresh': 'deferred',
              'material_freshness': 'not_verified', 'timings_seconds': timings,
              'timing_semantics': 'monotonic wall seconds; inclusive, not additive: checkpoint_groups_total/page_import/material_refresh include instrumented file_hashing',
              'recovery': 'Repeat identical indexes/model policy with --reuse-worker --resume.',
              **dict.fromkeys(COUNTERS, 0)}
    with bundle_write_lock(root, operation='courseware_bounded_worker', lock_name='.courseware-batch.lock'):
        if receipt_path.exists() and not resume:
            raise ValueError('existing worker batch requires explicit resume')
        if resume and not receipt_path.exists():
            raise ValueError('worker batch resume identity missing/drifted')
        if receipt_path.exists():
            prior = read_json(receipt_path)
            if prior.get('contract') != contract:
                raise ValueError('worker batch resume contract drift')
            history = safe_path(root, f'courseware/incremental-batches/history/{sha256_file(receipt_path)}.json', exists=False)
            if not history.exists():
                write_json(history, prior)
            report['previous_attempt'] = history.relative_to(root).as_posix()
        write_json(receipt_path, report)
        try:
            with ExitStack() as stack:
                stack.enter_context(instrument_integrity(timings))
                runtime = None
                for offset in range(0, len(indexes), 8):
                    part = indexes[offset:offset + 8]
                    report['current_indexes'] = part
                    if selection_identity(root, indexes) != identity:
                        raise ValueError('worker batch input drift')
                    state = read_scan(root)
                    pending = []
                    for i in part:
                        page = state['candidates'][i - 1]
                        path = safe_path(root, f"courseware/pages/{page['page_id']}.json", exists=False)
                        if path.exists():
                            value = read_json(path)
                            if value.get('page_id') != page['page_id']:
                                raise ValueError('worker batch formal page identity drift')
                            _validate_page(root, state, value)
                        else:
                            pending.append(i)
                    if not pending:
                        report['subbatches'].append({'indexes': part, 'status': 'no_op_existing_results'})
                        write_json(receipt_path, report)
                        continue
                    _measured(timings, 'resource_preflight', guard)()
                    if runtime is None:
                        before = perf_counter()
                        try:
                            runtime = stack.enter_context(runtime_factory())
                        finally:
                            timings['model_load'] += perf_counter() - before
                        report['model_loads'] += 1
                        engine, runtime_identity, output_factory, normalize = runtime
                        report['runtime'] = runtime_identity
                        stack.enter_context(instrument_engine(engine, timings))
                    child_runtime = courseware_incremental_ocr._detector_runtime(runtime_identity)
                    child_id = incremental_run_id(state['identity'], part, child_runtime)
                    child_path = safe_path(root, f'courseware/incremental-runs/{child_id}.json', exists=False)
                    previous_child_sha = _child_receipt_marker(child_path)
                    try:
                        result = _measured(timings, 'checkpoint_groups_total', run_incremental_courseware_ocr)(
                            root, indexes=part, engine=engine, runtime_identity=child_runtime,
                            output_factory=output_factory,
                            normalize_blocks=_measured(timings, 'normalization', normalize),
                            resume=bool(resume and child_path.exists()))
                    except Exception:
                        _record_failed_child(report, root, child_id, part, child_runtime,
                                             indexes, identity, previous_child_sha)
                        raise
                    report['subbatches'].append({'indexes': part, 'run_id': child_id,
                        'receipt': child_path.relative_to(root).as_posix(),
                        'receipt_sha256': sha256_file(child_path), 'status': result['status']})
                    for key in COUNTERS:
                        report[key] += result[key]
                    write_json(receipt_path, report)
                if selection_identity(root, indexes) != identity:
                    raise ValueError('worker batch final input drift')
            if refresh_material:
                with instrument_integrity(timings):
                    material = _measured(timings, 'material_refresh', build_courseware_material)(root)
                report.update(material_refresh='full_verified', material_revision=material['revision'],
                              material_freshness='verified_at_batch_finish')
            report['status'] = 'completed_review_required'
        except Exception as exc:
            report.update(status='blocked', error=str(exc), failure_indexes=report.get('current_indexes', indexes))
            raise
        finally:
            timings['total'] = perf_counter() - started
            write_json(receipt_path, report)
    return report
