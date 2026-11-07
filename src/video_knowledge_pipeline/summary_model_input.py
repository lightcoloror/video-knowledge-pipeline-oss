"""Hash-bound Reduce request/response receipts using existing Bundle storage.

No Provider, scheduler, retry, or workflow state lives here. A gateway boundary
capture is evidence of the supplied request, not proof of model observation.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

from .canonical_json import canonical_json_sha256
from .courseware_material import safe_path
from .evidence_locator import JsonEvidenceSnapshot
from .storage import bundle_write_lock, write_json

PREFIX = 'exports/summary-input-records/'
REQUEST_SCHEMA = 'video_knowledge_pipeline.summary_model_request.v1'
RESPONSE_SCHEMA = 'video_knowledge_pipeline.summary_model_response.v1'
BINDING_SCHEMA = 'video_knowledge_pipeline.summary_generation_binding.v1'
MAX_BYTES = 16 * 1024 * 1024
MAX_DISCOVERY_ENTRIES = 128
MAX_DISCOVERY_BYTES = 64 * 1024 * 1024
MAX_DISCOVERY_FILES = 136
RECOVERY_DISCOVERY_SCHEMA = 'video_knowledge_pipeline.summary_response_recovery.v1'


def _store(root, payload):
    name = PREFIX + canonical_json_sha256(payload) + '.json'
    path = safe_path(root, name, exists=False)
    with bundle_write_lock(root, operation='summary_input_receipt'):
        if path.exists():
            snap = JsonEvidenceSnapshot.read(root, name, max_bytes=MAX_BYTES)
            if snap.data != payload:
                raise ValueError('immutable summary input receipt drift')
        else:
            write_json(path, payload)
            snap = JsonEvidenceSnapshot.read(root, name, max_bytes=MAX_BYTES)
    return {'path': name, 'sha256': snap.artifact_sha256}


def _load(root, ref, schema, *, read_snapshot=None):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('invalid summary input reference')
    name = ref['path']
    if not isinstance(name, str) or not name.startswith(PREFIX) or '/' in name[len(PREFIX):]:
        raise ValueError('summary input reference outside receipt root')
    snap = (read_snapshot(name) if read_snapshot else
            JsonEvidenceSnapshot.read(root, name, max_bytes=MAX_BYTES))
    data = snap.data
    if (snap.artifact_sha256 != ref['sha256'] or not isinstance(data, dict) or data.get('schema') != schema
            or name != PREFIX + canonical_json_sha256(data) + '.json'):
        raise ValueError('summary input receipt identity/schema drift')
    return data


def _response_pair(root, ref, *, read_snapshot=None):
    """Use the same immutable identities for discovery and explicit recovery."""
    response = _load(root, ref, RESPONSE_SCHEMA, read_snapshot=read_snapshot)
    request = _load(root, response['request'], REQUEST_SCHEMA, read_snapshot=read_snapshot)
    messages = request['messages']
    parameters = request['parameters']
    if (not isinstance(parameters, dict)
            or set(parameters) != {'max_tokens', 'temperature', 'response_format'}
            or type(parameters['max_tokens']) is not int or parameters['max_tokens'] < 1
            or type(parameters['temperature']) not in {int, float}
            or (type(parameters['temperature']) is float and not math.isfinite(parameters['temperature']))
            or parameters['response_format'] != {'type': 'json_object'}
            or not isinstance(request['fact_pack_sha256'], str)
            or len(request['fact_pack_sha256']) != 64
            or any(char not in '0123456789abcdef' for char in request['fact_pack_sha256'])):
        raise ValueError('invalid captured Reduce parameters/fact identity')
    if (request.get('task') != 'smart_summary_global_reduce'
            or not isinstance(messages, list) or not 1 <= len(messages) <= 16
            or any(not isinstance(row, dict) or set(row) != {'role', 'content'}
                   or row['role'] not in {'system', 'user', 'assistant'}
                   or not isinstance(row['content'], str) for row in messages)
            or sum(row['role'] == 'user' for row in messages) != 1
            or sum(len(row['content'].encode('utf-8')) for row in messages) > 2 * 1024 * 1024
            or request['messages_sha256'] != canonical_json_sha256(messages)):
        raise ValueError('invalid captured Reduce request messages/identity')
    raw = response['raw_response']
    if (not isinstance(raw, str) or len(raw.encode('utf-8')) > 4 * 1024 * 1024
            or response['raw_response_sha256'] != hashlib.sha256(raw.encode('utf-8')).hexdigest()
            or type(response['gateway_ok']) is not bool
            or response.get('executor') not in {'model_task_gateway', 'consented_model_connector'}
            or response.get('model_observation') != 'not_assessed'):
        raise ValueError('invalid captured Reduce response content/identity')
    delegated = request['delegate_request']
    if delegated is not None:
        if not isinstance(delegated, dict) or set(delegated) != {'path', 'sha256'}:
            raise ValueError('invalid delegated request reference')
        # The business request path is mutable staging and is reused by the next
        # execution. capture_response verifies it before sealing the response;
        # historical identity is the immutable request above, not today's staging.
    return response, request


def capture_request(bundle_dir, *, messages, fact_pack, max_tokens, temperature, write,
                    delegate_request=None):
    if not write:
        return None
    root = Path(bundle_dir).absolute()
    if (not isinstance(messages, list) or not 1 <= len(messages) <= 16 or
            any(not isinstance(row, dict) or set(row) != {'role', 'content'} or
                row['role'] not in {'system', 'user', 'assistant'} or not isinstance(row['content'], str)
                for row in messages)):
        raise ValueError('invalid bounded text request messages')
    if sum(len(row['content'].encode('utf-8')) for row in messages) > 2 * 1024 * 1024:
        raise ValueError('summary input capture budget exceeded')
    delegated = None
    if delegate_request:
        relative = Path(delegate_request).absolute().relative_to(root).as_posix()
        snap = JsonEvidenceSnapshot.read(root, relative, max_bytes=MAX_BYTES)
        expected_parameters = {'max_tokens': int(max_tokens), 'temperature': float(temperature), 'response_format': 'json_object'}
        if (not isinstance(snap.data, dict) or snap.data.get('messages') != messages
                or snap.data.get('task') != 'smart_summary_global_reduce'
                or snap.data.get('generation_parameters') != expected_parameters):
            raise ValueError('delegate request does not match captured messages')
        delegated = {'path': relative, 'sha256': snap.artifact_sha256}
    return _store(root, {'schema': REQUEST_SCHEMA, 'task': 'smart_summary_global_reduce',
        'messages': messages, 'messages_sha256': canonical_json_sha256(messages),
        'fact_pack_sha256': canonical_json_sha256(fact_pack), 'delegate_request': delegated,
        'parameters': {'max_tokens': max_tokens, 'temperature': temperature, 'response_format': {'type': 'json_object'}}})


def capture_response(bundle_dir, request, *, raw_response, gateway_ok, executor, write):
    if not write:
        return None
    root = Path(bundle_dir).absolute()
    record = _load(root, request, REQUEST_SCHEMA)
    delegated = record['delegate_request']
    if delegated:
        snap = JsonEvidenceSnapshot.read(root, delegated['path'], max_bytes=MAX_BYTES)
        if snap.artifact_sha256 != delegated['sha256']:
            raise ValueError('delegated request changed during execution')
    if not isinstance(raw_response, str) or len(raw_response.encode('utf-8')) > 4 * 1024 * 1024:
        raise ValueError('summary response capture budget exceeded')
    if executor not in {'model_task_gateway', 'consented_model_connector'}:
        raise ValueError('unsupported captured gateway boundary')
    return _store(root, {'schema': RESPONSE_SCHEMA, 'request': request, 'raw_response': raw_response,
        'raw_response_sha256': hashlib.sha256(raw_response.encode('utf-8')).hexdigest(),
        'gateway_ok': gateway_ok is True, 'executor': executor, 'model_observation': 'not_assessed'})


def bind_candidate(bundle_dir, response, *, plan, fact_pack, write):
    if not write or response is None:
        return None
    root = Path(bundle_dir).absolute()
    result, request = _response_pair(root, response)
    if not result['gateway_ok'] or request['fact_pack_sha256'] != canonical_json_sha256(fact_pack):
        raise ValueError('failed response or current facts differ from actual request')
    return _store(root, {'schema': BINDING_SCHEMA, 'response': response,
        'reader_plan_sha256': canonical_json_sha256(plan), 'fact_pack_sha256': canonical_json_sha256(fact_pack)})


def generation_input_status(bundle_dir, *, fact_pack, plan, binding=None):
    root = Path(bundle_dir).absolute()
    report = {'status': 'not_assessed', 'passed': None, 'required': False, 'model_observation': 'not_assessed'}
    try:
        if binding is None:
            path = safe_path(root, 'manifest.json', exists=False)
            if not path.exists():
                return report
            manifest = JsonEvidenceSnapshot.read(root, 'manifest.json', max_bytes=4 * 1024 * 1024).data
            if not isinstance(manifest, dict):
                raise ValueError('invalid manifest for generation binding')
            if 'smart_summary_generation_binding' not in manifest:
                return report
            binding = manifest['smart_summary_generation_binding']
        report['required'] = True
        bound = _load(root, binding, BINDING_SCHEMA)
        result, request = _response_pair(root, bound['response'])
        facts_hash = canonical_json_sha256(fact_pack)
        if (bound['reader_plan_sha256'] != canonical_json_sha256(plan) or bound['fact_pack_sha256'] != facts_hash
                or request['fact_pack_sha256'] != facts_hash or not result['gateway_ok']
                or request['messages_sha256'] != canonical_json_sha256(request['messages'])
                or result['raw_response_sha256'] != hashlib.sha256(result['raw_response'].encode('utf-8')).hexdigest()):
            raise ValueError('current candidate/input identity drift')
        user_messages = [row['content'] for row in request['messages'] if row['role'] == 'user']
        if len(user_messages) != 1:
            raise ValueError('Reduce requires exactly one captured user payload')
        return {**report, 'status': 'bound_gateway_input', 'passed': True, 'prompt': user_messages[0],
                'binding': binding, 'request': result['request'], 'response': bound['response'],
                'executor': result['executor'], 'messages_sha256': request['messages_sha256']}
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return {**report, 'status': 'blocked', 'passed': False, 'errors': [str(exc)]}


def recover_response(bundle_dir, path):
    root = Path(bundle_dir).absolute()
    relative = Path(path).absolute().relative_to(root).as_posix()
    snap = JsonEvidenceSnapshot.read(root, relative, max_bytes=MAX_BYTES)
    ref = {'path': relative, 'sha256': snap.artifact_sha256}
    response, _ = _response_pair(root, ref)
    return {'ok': response['gateway_ok'], 'content': response['raw_response'] if response['gateway_ok'] else '',
            'error': '' if response['gateway_ok'] else 'captured gateway failed', 'input_capture': ref,
            'connector_status': 'recovered_captured_response', 'network_requests_made': False,
            'execution_report': str(path), 'execution_report_sha256': snap.artifact_sha256}


class _DiscoveryBudgetExceeded(ValueError):
    pass


class _CurrentInputUnverified(ValueError):
    pass


def _require_no_indirect_courseware(root, workflow, revisions):
    # Do not invoke build_courseware_material on a registry refresh: even its
    # write=False mode hashes/decodes the entire scan's frames and raw receipts.
    # Existence checks are sufficient to defer, never to assert freshness.
    for name in ('courseware/scan.json', 'courseware/material.json'):
        if safe_path(root, name, exists=False).exists():
            raise _CurrentInputUnverified('courseware freshness requires explicit full recovery validation')
    recorded = [workflow.get('courseware_material', {}), revisions.get('courseware_material', {})]
    for section in workflow['sections']:
        if isinstance(section, dict):
            evidence = section.get('evidence') or {}
            if not isinstance(evidence, dict):
                raise ValueError('invalid current Reduce evidence metadata')
            recorded.append(evidence.get('courseware_material', {}))
    for material in recorded:
        if not isinstance(material, dict):
            raise ValueError('invalid current Reduce courseware metadata')
        if material.get('active') or material.get('pages') or material.get('revision'):
            raise _CurrentInputUnverified('recorded courseware freshness requires explicit full recovery validation')


def _current_fact_pack(root, read_snapshot):
    """Rebuild the existing projection even before the first Reduce report write."""
    from .smart_summary_global_reduce import _chapter_fact_pack, _revision_rows
    from .content_profile import resolve_content_profile
    from .evidence_locator import audit_fact_pack_locators

    paths = {'workflow': 'exports/smart-summary-section-workflow.json',
             'revisions': 'exports/smart-summary-section-llm-revisions.json',
             'course_map': 'exports/course-map.json', 'manifest': 'manifest.json'}
    snapshots = {key: read_snapshot(name) for key, name in paths.items()}
    if any(not isinstance(snap.data, dict) for snap in snapshots.values()):
        raise ValueError('current Reduce inputs must be JSON objects')
    workflow, revisions = snapshots['workflow'].data, snapshots['revisions'].data
    if not isinstance(workflow.get('sections'), list) or not isinstance(revisions.get('rows'), list):
        raise ValueError('current Reduce sections/revisions are incomplete')
    _require_no_indirect_courseware(root, workflow, revisions)
    pack = _chapter_fact_pack(root, _revision_rows(revisions), workflow,
        source_paths={key: root / name for key, name in paths.items() if key != 'manifest'})
    expected_lineage = [{'source': key, 'path': paths[key], 'sha256': snapshots[key].artifact_sha256}
                        for key in sorted(paths) if key != 'manifest']
    manifest = snapshots['manifest'].data
    if (pack['lineage'] != expected_lineage
            or pack['title'] != str(manifest.get('title') or root.name)
            or pack['content_profile'] != resolve_content_profile(root, manifest=manifest)['profile_id']):
        raise ValueError('current Reduce inputs changed during projection')
    for section in pack.get('sections') or []:
        for ref in section.get('evidence_refs') or []:
            context = ref.get('source_context') or {}
            if (ref.get('source_locators') or ref.get('evidence_paths') or ref.get('recognition_provenance')
                    or context.get('frame_region') is not None):
                raise _CurrentInputUnverified('indirect source identity requires explicit full recovery validation')
    # With no locators, frame regions or courseware, this existing audit reads
    # no additional files. Any indirect evidence was deferred before the call.
    audit_fact_pack_locators(root, pack)
    _require_no_indirect_courseware(root, workflow, revisions)
    for key, name in paths.items():
        if read_snapshot(name, refresh=True).artifact_sha256 != snapshots[key].artifact_sha256:
            raise ValueError('current Reduce inputs changed during discovery')
    return pack, manifest


def discover_response_recovery(bundle_dir):
    """Bounded, read-only recovery projection. Never parse/install or call a model.

    The fixed receipt directory is enumerated directly, without glob, recursion,
    or sorting an unbounded directory. An incomplete scan offers no commands.
    Every explicit recovery still passes the existing Reduce gates at execution.
    """
    root = Path(bundle_dir).absolute()
    report = {'schema': RECOVERY_DISCOVERY_SCHEMA, 'status': 'not_assessed',
              'scan_complete': True, 'entries_examined': 0, 'bytes_read': 0,
              'entry_limit': MAX_DISCOVERY_ENTRIES, 'byte_limit': MAX_DISCOVERY_BYTES,
              'file_limit': MAX_DISCOVERY_FILES, 'files_examined': 0,
              'candidate_count': 0, 'unverified_count': 0, 'already_bound_count': 0, 'items': [], 'errors': [],
              'model_observation': 'not_assessed', 'network_requests_made': False,
              'automatic_recovery': False, 'directory': PREFIX.rstrip('/')}
    cache, examined_files = {}, set()

    def read_snapshot(name, *, refresh=False):
        if not refresh and name in cache:
            return cache[name]
        if name not in examined_files:
            if len(examined_files) >= MAX_DISCOVERY_FILES:
                raise _DiscoveryBudgetExceeded('summary response discovery file budget exceeded')
            examined_files.add(name)
            report['files_examined'] = len(examined_files)
        path = safe_path(root, name)
        remaining = MAX_DISCOVERY_BYTES - report['bytes_read']
        if remaining <= 0 or path.stat().st_size > remaining:
            raise _DiscoveryBudgetExceeded('summary response discovery byte budget exceeded')
        budget = min(MAX_BYTES, remaining)
        # Charge failed parses too: malformed files must not evade the aggregate
        # read limit. A failed read conservatively consumes its full allowance.
        report['bytes_read'] += budget
        snap = JsonEvidenceSnapshot.read(root, name, max_bytes=budget)
        report['bytes_read'] -= budget - snap.size_bytes
        cache[name] = snap
        return snap

    responses = []
    try:
        directory = safe_path(root, PREFIX.rstrip('/'), exists=False)
        if not directory.exists():
            return report
        with os.scandir(directory) as entries:
            for index, entry in enumerate(entries):
                if index >= MAX_DISCOVERY_ENTRIES:
                    raise _DiscoveryBudgetExceeded('summary response discovery entry budget exceeded')
                report['entries_examined'] += 1
                # Subdirectories are never traversed. Reparse points are rejected
                # even when their target would stay inside the Bundle.
                name = PREFIX + entry.name
                try:
                    path = safe_path(root, name, exists=False)
                    if not entry.is_file(follow_symlinks=False) or path.suffix != '.json':
                        continue
                    snap = read_snapshot(name)
                    ref = {'path': name, 'sha256': snap.artifact_sha256}
                    schema = snap.data.get('schema') if isinstance(snap.data, dict) else None
                    if schema not in {REQUEST_SCHEMA, RESPONSE_SCHEMA, BINDING_SCHEMA}:
                        raise ValueError('invalid summary input receipt schema')
                    _load(root, ref, schema, read_snapshot=read_snapshot)
                    if schema == RESPONSE_SCHEMA:
                        response, request = _response_pair(root, ref, read_snapshot=read_snapshot)
                        if not response['gateway_ok'] or not response['raw_response'].strip():
                            raise ValueError('captured gateway failed or response is incomplete')
                        responses.append((ref, response, request))
                except _DiscoveryBudgetExceeded:
                    raise
                except (ValueError, OSError, KeyError, TypeError, AttributeError, IndexError, RecursionError) as exc:
                    report['items'].append({'path': name, 'status': 'blocked', 'errors': [str(exc)],
                                            'model_observation': 'not_assessed'})
        if responses:
            pack, manifest = _current_fact_pack(root, read_snapshot)
            facts_hash = canonical_json_sha256(pack)
            active_response = None
            if manifest.get('smart_summary_generation_binding'):
                try:
                    bound = _load(root, manifest['smart_summary_generation_binding'], BINDING_SCHEMA,
                                  read_snapshot=read_snapshot)
                    plan = read_snapshot('exports/smart-summary-reader-plan.json').data
                    if (bound['reader_plan_sha256'] == canonical_json_sha256(plan)
                            and bound['fact_pack_sha256'] == facts_hash):
                        active_response = bound['response']
                except _DiscoveryBudgetExceeded:
                    raise
                except (ValueError, OSError, KeyError, TypeError, AttributeError, IndexError, RecursionError) as exc:
                    report['errors'].append('active generation binding: ' + str(exc))
            for ref, response, request in responses:
                row = {'path': ref['path'], 'response': ref, 'request': response['request'],
                       'status': 'blocked', 'model_observation': 'not_assessed',
                       'validation_scope': 'request_response_and_current_input_identity',
                       'reader_plan_validation': 'pending_existing_recovery_gate'}
                if request['fact_pack_sha256'] != facts_hash:
                    row['errors'] = ['captured request differs from current Reduce input identity']
                elif active_response == ref:
                    report['already_bound_count'] += 1
                    continue
                else:
                    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
                    row.update(status='recovery_available', fact_pack_sha256=facts_hash,
                        recovery_command='.\\scripts\\video-knowledge.ps1 smart-summary-global-reduce '
                            + quote(root) + ' --recovery-execution-report ' + quote(root / ref['path']))
                report['items'].append(row)
    except _CurrentInputUnverified as exc:
        for ref, response, _request in responses:
            report['items'].append({'path': ref['path'], 'response': ref, 'request': response['request'],
                'status': 'unverified_current_input', 'errors': [str(exc)],
                'model_observation': 'not_assessed', 'validation_scope': 'immutable_request_response_only',
                'reader_plan_validation': 'pending_existing_recovery_gate'})
    except _DiscoveryBudgetExceeded as exc:
        report['scan_complete'] = False
        report['errors'].append(str(exc))
    except (ValueError, OSError, KeyError, TypeError, AttributeError, IndexError, RecursionError) as exc:
        report['errors'].append(str(exc))
    if not report['scan_complete']:
        for row in report['items']:
            row.pop('recovery_command', None)
            row['status'] = 'blocked'
    report['candidate_count'] = sum(row['status'] == 'recovery_available' for row in report['items'])
    report['unverified_count'] = sum(row['status'] == 'unverified_current_input' for row in report['items'])
    report['status'] = ('blocked' if report['errors'] or any(row['status'] == 'blocked' for row in report['items'])
                        else 'recovery_available' if report['candidate_count']
                        else 'unverified_current_input' if report['unverified_count']
                        else 'no_recovery_needed' if report['already_bound_count'] else 'not_assessed')
    return report
