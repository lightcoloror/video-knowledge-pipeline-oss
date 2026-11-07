"""Offline semantic producers with version-bound checkpoints in the run registry."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from .artifact_freshness import build_dependency_snapshot, validate_dependency_snapshot
from .artifact_validation import artifact_evidence
from .canonical_json import canonical_json_sha256
from .file_hash import sha256_file
from .run_artifact_recovery import _identifier, _inside, _projection, MAX_BYTES
from .run_artifact_registry import register_bundle_run, _write_run
from .stage_cache import StageCache, atomic_copy_file
from .storage import bundle_write_lock, read_json, write_json, write_text_atomic

PROFILE = 'versioned_semantic_producers.v1'
STAGES = ('candidates', 'local_draft')


def _versions():
    # Import the actual existing producers before collecting their loaded local
    # implementation dependencies. No model execution occurs during import.
    from . import transcript_semantic_correction  # noqa: F401
    import sys
    base = Path(__file__).parent.resolve()
    result = {}
    for name in ('versioned_semantic_run.py', 'review_session.py', 'cli.py', 'cli_output.py', 'evidence_lineage.py'):
        result[name] = sha256_file(base / name)
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('video_knowledge_pipeline.') and path:
            path = Path(path).resolve()
            if path.suffix == '.py' and path.is_relative_to(base):
                result[path.relative_to(base).as_posix()] = sha256_file(path)
    return result


def _binding(run):
    return canonical_json_sha256({k: run[k] for k in ('run_id', 'run_type', 'bundle_dir', 'inputs', 'parameters', 'dependency_snapshot')})


def _runtime_versions():
    import sys
    import unicodedata
    return {'python': sys.version, 'unicode': unicodedata.unidata_version}


def _read(path):
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('metadata_too_large')
    return read_json(path)


def create_run(bundle_dir, request):
    root = Path(bundle_dir).resolve()
    rid = _identifier(request.get('run_id'))
    paths = request.get('inputs')
    versions = request.get('dependency_versions')
    if not isinstance(versions, dict) or not versions or any(not isinstance(v, str) or not v for v in versions.values()):
        raise ValueError('explicit_dependency_versions_required')
    if not isinstance(paths, list) or not 2 <= len(paths) <= 64 or len(set(paths)) != len(paths):
        raise ValueError('explicit_bounded_input_files_required')
    if 'manifest.json' not in paths:
        raise ValueError('manifest_required')
    contexts = request.get('context', [])
    if not isinstance(contexts, list) or len(contexts) > 32:
        raise ValueError('invalid_context')
    for row in contexts:
        if row.get('path') not in paths or row.get('kind') not in {'historical_candidates', 'human_evidence', 'ocr_inventory'}:
            raise ValueError('context_must_be_bound_input')
    frozen = _inside(root, f'runs/{rid}/inputs')
    with bundle_write_lock(root, operation='versioned_semantic_create'):
        if frozen.parent.exists():
            raise ValueError('run_identity_already_exists')
        for value in paths:
            if value.startswith(('runs/', 'exports/versioned-')) or '\\' in value:
                raise ValueError('input_namespace_invalid')
            source = _inside(root, value)
            if not source.is_file() or source.suffix != '.json':
                raise ValueError('explicit_json_metadata_only')
            _read(source)
        manifest = _read(root / 'manifest.json')
        if not isinstance(manifest, dict):
            raise ValueError('manifest_object_required')
        # The offline profile supports explicit transcript, sidecar and metadata
        # references. Never let a manifest direct the producers outside the snapshot.
        from .transcript_semantic_correction import SIDE_SOURCE_MANIFEST_KEYS
        refs = {'normalized_transcript_json', 'raw_transcript_json', 'asr_transcript_json', 'transcript_json', 'page_metadata_json'}
        refs.update(k for k, _ in SIDE_SOURCE_MANIFEST_KEYS)
        for key in refs:
            values = manifest.get(key)
            for value in values if isinstance(values, list) else [values]:
                if value and (not isinstance(value, str) or value not in paths):
                    raise ValueError(f'unsnapshotted_manifest_reference:{key}')
        if not any(manifest.get(k) for k in refs - {'page_metadata_json'}) and 'timeline.json' not in paths:
            raise ValueError('transcript_or_timeline_required')
        dependency = build_dependency_snapshot(root, subject=PROFILE, inputs=paths, source_run_id=rid, producer_schema=PROFILE)
        hashes = {}
        for value in paths:
            source, target = _inside(root, value), _inside(frozen, value)
            before = sha256_file(source)
            atomic_copy_file(source, target)
            if before != sha256_file(target) or before != sha256_file(source):
                raise ValueError('source_changed_during_capture')
            hashes[value] = before
        if not validate_dependency_snapshot(root, dependency)['passed']:
            raise ValueError('input_changed_during_capture')
        run = register_bundle_run(root, run_id=rid, run_type='versioned_semantic', status='planned',
            inputs={'files': paths}, parameters={'profile': PROFILE, 'dependency_versions': versions,
            'runtime_versions': _runtime_versions(),
            'code_versions': _versions(), 'input_hashes': hashes, 'context': contexts},
            dependency_snapshot=dependency, operator_boundary={'model_execution': False, 'semantic_acceptance': False}, write=False)
        run['result_binding_sha256'] = _binding(run)
        _write_run(root, run)
    return status(root, rid, versions)


def _load(root, rid, versions):
    rid = _identifier(rid)
    run = _read(_inside(root, f'runs/{rid}/run.json'))
    if (run.get('run_id') != rid or run.get('run_type') != 'versioned_semantic' or run.get('bundle_dir') != str(root)
            or run.get('parameters', {}).get('profile') != PROFILE or run.get('result_binding_sha256') != _binding(run)):
        raise ValueError('run_binding_mismatch')
    p = run['parameters']
    if versions != p['dependency_versions']:
        raise ValueError('dependency_versions_changed_or_unchecked')
    if p['runtime_versions'] != _runtime_versions():
        raise ValueError('runtime_versions_changed')
    dep = run['dependency_snapshot']
    if dep.get('source_run_id') != rid or dep.get('producer_schema') != PROFILE or not validate_dependency_snapshot(root, dep)['passed']:
        raise ValueError('input_dependency_drift')
    # Compare the captured dependency closure, not unrelated modules imported by
    # a wider consumer such as the legacy CLI.
    base = Path(__file__).parent
    for name, digest in p['code_versions'].items():
        path = _inside(base.resolve(), name)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError('producer_code_drift:' + name)
    frozen = _inside(root, f'runs/{rid}/inputs')
    actual = {f.relative_to(frozen).as_posix() for f in frozen.rglob('*') if f.is_file()}
    if actual != set(p['input_hashes']):
        raise ValueError('snapshot_inventory_drift')
    for name, digest in p['input_hashes'].items():
        if sha256_file(_inside(frozen, name)) != digest:
            raise ValueError('snapshot_input_drift')
    return run, frozen


def _cache(root, run):
    return StageCache(_inside(root, f"runs/{run['run_id']}/stage-cache"), _inside(root, f"runs/{run['run_id']}/inputs/manifest.json"))


def _stage_identity(run, stage):
    previous = next((a['sha256'] for a in run['artifacts'] if a['key'] == 'candidates'), '') if stage == 'local_draft' else ''
    return canonical_json_sha256({'run_binding': run['result_binding_sha256'], 'stage': stage, 'upstream_sha256': previous})


def _completed(root, run, stage):
    artifact = next((a for a in run['artifacts'] if a['key'] == stage), None)
    if artifact is None:
        return False
    cache = _cache(root, run)
    path = cache.stage_path(stage, 'result.json')
    if artifact['path'] != path.relative_to(root).as_posix() or not path.is_file():
        raise ValueError('completed_stage_artifact_missing:' + stage)
    if sha256_file(path) != artifact['sha256'] or not cache.is_valid(stage, identity=_stage_identity(run, stage)):
        raise ValueError('completed_stage_artifact_drift:' + stage)
    value = _read(path).get('version_binding', {})
    if value.get('run_id') != run['run_id'] or value.get('stage') != stage or value.get('binding_sha256') != run['result_binding_sha256']:
        raise ValueError('completed_stage_version_mismatch:' + stage)
    return True


def _context(run, frozen):
    return [{**_projection(row, _read(_inside(frozen, row['path']))), 'input_path': row['path'],
             'input_sha256': run['parameters']['input_hashes'][row['path']]} for row in run['parameters']['context']]


def status(bundle_dir, run_id, versions):
    root = Path(bundle_dir).resolve()
    run, frozen = _load(root, run_id, versions)
    completed = [stage for stage in STAGES if _completed(root, run, stage)]
    return {'schema': PROFILE, 'run_id': run_id, 'binding_sha256': run['result_binding_sha256'],
            'input_sha256': canonical_json_sha256(run['parameters']['input_hashes']),
            'dependency_versions': versions, 'completed_stages': completed,
            'pending_stages': [s for s in STAGES if s not in completed],
            'context': _context(run, frozen), 'status': 'offline_stages_complete' if len(completed) == len(STAGES) else 'pending',
            'model_adjudication': 'not_executed', 'semantic_acceptance': False}


def resume(bundle_dir, run_id, versions):
    root = Path(bundle_dir).resolve()
    executed, skipped = [], []
    with bundle_write_lock(root, operation='versioned_semantic_resume'):
        run, frozen = _load(root, run_id, versions)
        cache = _cache(root, run)
        from .transcript_semantic_correction import build_transcript_semantic_correction_pack, build_transcript_semantic_correction_codex_draft
        for stage in STAGES:
            if _completed(root, run, stage):
                skipped.append(stage)
                continue
            _load(root, run_id, versions)
            try:
                if stage == 'candidates':
                    value = build_transcript_semantic_correction_pack(frozen, write=False)
                else:
                    value = build_transcript_semantic_correction_codex_draft(frozen,
                        input_json=cache.stage_path('candidates', 'result.json'), write=False)
                _load(root, run_id, versions)
                value['version_binding'] = {'run_id': run_id, 'binding_sha256': run['result_binding_sha256'],
                    'input_sha256': canonical_json_sha256(run['parameters']['input_hashes']),
                    'dependency_versions': versions, 'stage': stage}
                value['semantic_acceptance'] = False
                # write=False producers return default destination hints. They
                # are not generated files and must not become consumer links.
                value['producer_artifact_hints'] = {k: value.pop(k) for k in ('artifacts', 'result_json', 'result_markdown') if k in value}
                value['result_json'] = str(cache.stage_path(stage, 'result.json'))
                target = cache.store_json(stage, 'result.json', value, identity=_stage_identity(run, stage))
                run['artifacts'].append({**artifact_evidence(target), 'key': stage, 'path': target.relative_to(root).as_posix()})
                run['status'] = 'offline_stages_complete' if stage == STAGES[-1] else 'running'
                run['failed_items'] = []
                _write_run(root, run)
                executed.append(stage)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                run['status'] = 'needs_retry'
                run['failed_items'] = [{'id': stage, 'reason': type(exc).__name__, 'detail': str(exc)}]
                _write_run(root, run)
                return {**status(root, run_id, versions), 'status': 'blocked', 'failed_stage': stage, 'error': str(exc),
                        'executed_stages': executed, 'skipped_stages': skipped, 'attempted_stage': stage}
    return {**status(root, run_id, versions), 'executed_stages': executed, 'skipped_stages': skipped}


def consume(bundle_dir, run_id, versions, stage):
    if stage not in STAGES:
        raise ValueError('unsupported_consumer_stage')
    root = Path(bundle_dir).resolve()
    with bundle_write_lock(root, operation='versioned_semantic_consume'):
        run, _ = _load(root, run_id, versions)
        if not all(_completed(root, run, s) for s in STAGES[:STAGES.index(stage)+1]):
            raise ValueError('requested_stage_not_complete')
        return _read(_cache(root, run).stage_path(stage, 'result.json'))


def review(bundle_dir, run_id, versions):
    root = Path(bundle_dir).resolve()
    with bundle_write_lock(root, operation='versioned_semantic_review'):
        pack = consume(root, run_id, versions, 'candidates')
        draft = consume(root, run_id, versions, 'local_draft')
        current = status(root, run_id, versions)
        if pack['version_binding']['binding_sha256'] != draft['version_binding']['binding_sha256']:
            raise ValueError('consumer_version_mismatch')
        from .transcript_semantic_correction import _render_prompt, _render_codex_result_markdown
        output = _inside(root, f'exports/versioned-results/{run_id}/review')
        doc = _render_prompt(pack) + '\n\n' + _render_codex_result_markdown(draft)
        doc += '\n\n## Version and evidence boundary\n\n' + json.dumps(current, ensure_ascii=False, indent=2)
        result = {**current, 'schema': 'lecture_review_session.v1', 'version_binding': pack['version_binding'],
                  'candidate_count': pack['candidate_count'], 'current_zero_candidates': pack['candidate_count'] == 0,
                  'local_draft_decision_count': draft['decision_count'], 'review_only': True,
                  'review_html_path': str(output / 'review.html'), 'review_pack_json_path': str(output / 'review-session.json')}
        write_text_atomic(output / 'review.md', doc)
        write_text_atomic(output / 'review.html', '<!doctype html><meta charset="utf-8"><title>Versioned semantic review</title><pre>' + html.escape(doc) + '</pre>')
        write_json(output / 'review-session.json', result)
        return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('create', 'resume', 'status'))
    p.add_argument('bundle_dir')
    p.add_argument('--request', type=Path)
    p.add_argument('--run-id')
    p.add_argument('--versions', type=Path)
    a = p.parse_args(argv)
    try:
        if a.action == 'create':
            if not a.request: raise ValueError('request_required')
            result = create_run(a.bundle_dir, _read(a.request))
        else:
            if not a.run_id or not a.versions: raise ValueError('explicit_run_and_current_versions_required')
            result = (resume if a.action == 'resume' else status)(a.bundle_dir, a.run_id, _read(a.versions))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result['status'] == 'blocked' else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'blocked', 'error': str(exc), 'semantic_acceptance': False}, ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
