"""Local synthetic crash-window discovery; never invoke a model or media tool."""
import json

import pytest

from video_knowledge_pipeline.canonical_json import canonical_json_sha256
from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline import summary_model_input as capture


def fixture(root):
    from test_smart_summary_global_reduce_business_authorization import _bundle
    from video_knowledge_pipeline.smart_summary_global_reduce import _chapter_fact_pack, _revision_rows
    bundle, paths = _bundle(root)
    workflow, revisions, course_map = paths
    pack = _chapter_fact_pack(bundle, _revision_rows(json.loads(revisions.read_text())),
        json.loads(workflow.read_text()), source_paths=dict(workflow=workflow, revisions=revisions, course_map=course_map))
    request = capture.capture_request(bundle, messages=[{'role': 'user', 'content': 'original private prompt'}],
        fact_pack=pack, max_tokens=10, temperature=0, write=True)
    response = capture.capture_response(bundle, request, raw_response='{"original":"response"}',
        gateway_ok=True, executor='model_task_gateway', write=True)
    return bundle, pack, request, response


def test_orphan_discovery_works_before_fact_pack_or_run_was_written(tmp_path):
    root, pack, request, response = fixture(tmp_path)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result = capture.discover_response_recovery(root)
    assert result['status'] == 'recovery_available' and result['scan_complete'] is True
    assert result['candidate_count'] == 1
    row = result['items'][0]
    assert row['status'] == 'recovery_available'
    assert row['request'] == request and row['response'] == response
    assert row['fact_pack_sha256'] == canonical_json_sha256(pack)
    assert row['model_observation'] == 'not_assessed'
    assert '--recovery-execution-report' in row['recovery_command']
    assert str(root / response['path']) in row['recovery_command']
    assert 'original private prompt' not in json.dumps(result)
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('change', ['facts', 'title', 'missing_input', 'malformed_material', 'request', 'response', 'nested_request', 'gateway_failed'])
def test_stale_invalid_and_incomplete_are_not_recovery_candidates(tmp_path, change):
    root, pack, request, response = fixture(tmp_path)
    if change == 'facts':
        write_json(root / 'exports/smart-summary-section-llm-revisions.json', {'rows': []})
    elif change == 'title':
        write_json(root / 'manifest.json', {'title': 'changed'})
    elif change == 'missing_input':
        (root / 'exports/course-map.json').unlink()
    elif change == 'malformed_material':
        path = root / 'exports/smart-summary-section-workflow.json'
        write_json(path, {**json.loads(path.read_text()), 'courseware_material': []})
    elif change in {'request', 'response'}:
        write_json(root / {'request': request, 'response': response}[change]['path'], {'changed': True})
    else:
        value = json.loads((root / response['path']).read_text(encoding='utf-8'))
        (root / response['path']).unlink()
        if change == 'nested_request':
            value['request'] = {'path': capture.PREFIX + '../request.json', 'sha256': 'a' * 64}
        else:
            value['gateway_ok'] = False
        capture._store(root, value)
    result = capture.discover_response_recovery(root)
    assert result['candidate_count'] == 0
    assert result['status'] == 'blocked'
    assert result['items'] or result['errors']
    assert not any(row.get('recovery_command') for row in result['items'])


def test_discovery_does_not_descend_and_fails_closed_when_entry_budget_exhausted(tmp_path, monkeypatch):
    root, _, _, _ = fixture(tmp_path)
    records = root / capture.PREFIX
    write_json(records / 'nested/ignored.json', {'schema': capture.RESPONSE_SCHEMA})
    assert capture.discover_response_recovery(root)['candidate_count'] == 1
    monkeypatch.setattr(capture, 'MAX_DISCOVERY_ENTRIES', 1)
    result = capture.discover_response_recovery(root)
    assert result['scan_complete'] is False and result['status'] == 'blocked'
    assert result['entries_examined'] == 1
    assert result['candidate_count'] == 0
    assert not any(row.get('recovery_command') for row in result['items'])


def test_registry_reuses_existing_console_and_workbench_projection(tmp_path):
    from video_knowledge_pipeline.run_artifact_registry import build_run_artifact_registry
    from video_knowledge_pipeline.task_console import _load_run_registry_for_console, _build_processing_queue, _run_history_html
    from video_knowledge_pipeline.video_workbench import _queue_panel_html
    root, _, _, response = fixture(tmp_path)
    registry = build_run_artifact_registry(root, write=True)
    assert registry['registered_run_count'] == 0 and registry['run_count'] == 1
    row = registry['runs'][0]
    assert row['status'] == 'needs_review' and row['recovery']['status'] == 'recovery_available'
    assert row['run_json'] == ''
    assert not (root / 'runs').exists()
    console = _load_run_registry_for_console(root)
    queue = _build_processing_queue(root, console)
    assert queue['action_required_count'] == 1
    assert '--recovery-execution-report' in _run_history_html(console, root)
    assert '--recovery-execution-report' in _queue_panel_html(queue)
    assert response['path'] in (root / 'run-artifact-registry.md').read_text(encoding='utf-8')
    again = build_run_artifact_registry(root, write=False)
    assert again['run_count'] == 1 and again['runs'][0]['run_id'] == row['run_id']


def test_active_bound_response_is_not_offered_again(tmp_path):
    root, pack, _, response = fixture(tmp_path)
    plan = {'schema': 'fixture'}
    binding = capture.bind_candidate(root, response, plan=plan, fact_pack=pack, write=True)
    write_json(root / 'exports/smart-summary-reader-plan.json', plan)
    write_json(root / 'manifest.json', {'title': 'summary fixture', 'smart_summary_generation_binding': binding})
    result = capture.discover_response_recovery(root)
    assert result['candidate_count'] == 0 and result['already_bound_count'] == 1


def test_semantically_invalid_snapshot_is_refused_even_when_content_addressed(tmp_path):
    root, _, request, response = fixture(tmp_path)
    value = json.loads((root / response['path']).read_text(encoding='utf-8'))
    value['raw_response_sha256'] = 'a' * 64
    ref = capture._store(root, value)
    with pytest.raises(ValueError):
        capture.recover_response(root, root / ref['path'])


def test_malformed_receipts_cannot_evade_aggregate_read_budget(tmp_path, monkeypatch):
    root, _, _, _ = fixture(tmp_path)
    for index in range(5):
        (root / capture.PREFIX / f'incomplete-{index}.json').write_text('{', encoding='utf-8')
    monkeypatch.setattr(capture, 'MAX_DISCOVERY_BYTES', capture.MAX_BYTES * 2)
    result = capture.discover_response_recovery(root)
    assert result['status'] == 'blocked' and result['scan_complete'] is False
    assert result['candidate_count'] == 0
    assert result['bytes_read'] <= result['byte_limit']
    assert any('byte budget' in error for error in result['errors'])


def test_unsafe_receipt_directory_is_rejected_before_enumeration(tmp_path, monkeypatch):
    root, _, _, _ = fixture(tmp_path)
    original = capture.safe_path
    def refuse_directory(bundle, value, **kwargs):
        if value == capture.PREFIX.rstrip('/'):
            raise ValueError('courseware reparse path rejected')
        return original(bundle, value, **kwargs)
    monkeypatch.setattr(capture, 'safe_path', refuse_directory)
    def no_scan(*args):
        raise AssertionError('unsafe directory must never be enumerated')
    monkeypatch.setattr(capture.os, 'scandir', no_scan)
    result = capture.discover_response_recovery(root)
    assert result['candidate_count'] == 0 and result['status'] == 'blocked'
    assert result['entries_examined'] == 0


def test_later_business_request_does_not_invalidate_immutable_response_or_binding(tmp_path):
    from video_knowledge_pipeline.smart_summary_global_reduce import _write_business_reduce_request
    root, pack, _, response = fixture(tmp_path)
    (root / response['path']).unlink()
    messages = [{'role': 'user', 'content': 'delegated input'}]
    delegate = _write_business_reduce_request(root, messages=messages, max_tokens=10, temperature=0)
    request = capture.capture_request(root, messages=messages, fact_pack=pack, max_tokens=10,
        temperature=0, delegate_request=delegate, write=True)
    response = capture.capture_response(root, request, raw_response='{}', gateway_ok=True,
        executor='consented_model_connector', write=True)
    next_messages = [{'role': 'user', 'content': 'second business request'}]
    assert _write_business_reduce_request(root, messages=next_messages, max_tokens=20, temperature=0.1) == delegate
    assert capture.discover_response_recovery(root)['candidate_count'] == 1
    assert capture.recover_response(root, root / response['path'])['content'] == '{}'
    plan = {'schema': 'fixture'}
    binding = capture.bind_candidate(root, response, plan=plan, fact_pack=pack, write=True)
    write_json(root / 'manifest.json', {'title': 'summary fixture', 'smart_summary_generation_binding': binding})
    write_json(root / 'exports/smart-summary-reader-plan.json', plan)
    assert capture.generation_input_status(root, fact_pack=pack, plan=plan)['passed'] is True
    assert capture.discover_response_recovery(root)['already_bound_count'] == 1
    next_request = capture.capture_request(root, messages=next_messages, fact_pack=pack, max_tokens=20,
        temperature=0.1, delegate_request=delegate, write=True)
    _write_business_reduce_request(root, messages=[{'role': 'user', 'content': 'third request'}], max_tokens=30, temperature=0)
    with pytest.raises(ValueError, match='changed during execution'):
        capture.capture_response(root, next_request, raw_response='{}', gateway_ok=True,
            executor='consented_model_connector', write=True)


def test_hard_crash_is_discovered_and_explicitly_recovered_without_second_call(tmp_path, monkeypatch):
    from test_summary_model_input import live_fixture
    from video_knowledge_pipeline import smart_summary_global_reduce as module
    projection = module._chapter_fact_pack
    root, module, calls = live_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(module, '_chapter_fact_pack', projection)
    parse = module._validated_reader_plan_content
    def crash(*args, **kwargs):
        raise SystemExit('synthetic hard crash after immutable response')
    monkeypatch.setattr(module, '_validated_reader_plan_content', crash)
    with pytest.raises(SystemExit, match='synthetic hard crash'):
        module.run_smart_summary_global_reduce(root, execute=True, install=False)
    assert not (root / 'exports/smart-summary-chapter-fact-pack.json').exists()
    report = capture.discover_response_recovery(root)
    assert report['candidate_count'] == 1
    monkeypatch.setattr(module, '_validated_reader_plan_content', parse)
    response = root / report['items'][0]['response']['path']
    recovered = module.run_smart_summary_global_reduce(root, recovery_execution_report=response, install=False)
    assert len(calls) == 1
    assert recovered['generation_input']['status'] == 'bound_gateway_input'
    assert recovered['recovery']['network_requests_made'] is False


def test_registry_no_write_and_blocked_projection_never_create_recovery_state(tmp_path):
    from video_knowledge_pipeline.run_artifact_registry import build_run_artifact_registry
    root, _, _, _ = fixture(tmp_path)
    write_json(root / 'manifest.json', {'title': 'changed since request'})
    registry = build_run_artifact_registry(root, write=False)
    assert registry['registered_run_count'] == 0
    assert all(row['status'] == 'blocked' and not row['retry_command'] for row in registry['runs'])
    assert not (root / 'runs').exists()
    assert not (root / 'run-artifact-registry.json').exists()


def test_request_without_response_is_not_assessed_as_success(tmp_path):
    root, _, _, response = fixture(tmp_path)
    (root / response['path']).unlink()
    result = capture.discover_response_recovery(root)
    assert result['status'] == 'not_assessed' and result['candidate_count'] == 0


def test_large_courseware_scan_is_deferred_before_material_or_frame_validation(tmp_path, monkeypatch):
    from video_knowledge_pipeline import courseware_material, smart_summary_global_reduce
    from video_knowledge_pipeline.run_artifact_registry import build_run_artifact_registry
    root, _, _, response = fixture(tmp_path)
    write_json(root / 'courseware/scan.json', {'candidates': [
        {'page_id': f'p-{index}', 'frame': f'courseware/frames/{index}.png'} for index in range(5609)]})
    def full_validation_forbidden(*args, **kwargs):
        raise AssertionError('registry refresh must not enter full courseware/frame validation')
    monkeypatch.setattr(courseware_material, 'build_courseware_material', full_validation_forbidden)
    monkeypatch.setattr(smart_summary_global_reduce, '_chapter_fact_pack', full_validation_forbidden)
    original_read = capture.JsonEvidenceSnapshot.read
    reads = []
    def bounded_read(bundle, name, **kwargs):
        reads.append(name)
        assert not name.startswith('courseware/')
        return original_read(bundle, name, **kwargs)
    monkeypatch.setattr(capture.JsonEvidenceSnapshot, 'read', bounded_read)
    registry = build_run_artifact_registry(root, write=False)
    result = registry['summary_response_recovery']
    assert result['status'] == 'unverified_current_input' and result['candidate_count'] == 0
    assert result['unverified_count'] == 1 and result['files_examined'] == 6
    row = registry['runs'][0]
    assert row['status'] == 'needs_review' and row['freshness']['passed'] is False
    assert row['recovery']['response'] == response and row['retry_command'] == ''
    assert len(reads) == 6


@pytest.mark.parametrize('metadata', [
    {'source_locators_by_kind': {'asr': [{'artifact_path': 'external-large.json'}]}},
    {'evidence_paths': ['external-large.json']},
])
def test_indirect_locators_are_deferred_before_source_audit(tmp_path, monkeypatch, metadata):
    from video_knowledge_pipeline import evidence_locator
    root, _, _, response = fixture(tmp_path)
    workflow_path = root / 'exports/smart-summary-section-workflow.json'
    workflow = json.loads(workflow_path.read_text())
    workflow['sections'][0]['citations'] = [{'citation_id': 'source-1', 'snippet': 'text', **metadata}]
    write_json(workflow_path, workflow)
    def no_audit(*args, **kwargs):
        raise AssertionError('indirect source audit must be deferred')
    monkeypatch.setattr(evidence_locator, 'audit_fact_pack_locators', no_audit)
    result = capture.discover_response_recovery(root)
    assert result['status'] == 'unverified_current_input' and result['candidate_count'] == 0
    assert result['items'][0]['response'] == response
    assert not result['items'][0].get('recovery_command')


def test_discovery_file_count_is_fail_closed(tmp_path, monkeypatch):
    root, _, _, _ = fixture(tmp_path)
    monkeypatch.setattr(capture, 'MAX_DISCOVERY_FILES', 2)
    result = capture.discover_response_recovery(root)
    assert result['status'] == 'blocked' and result['scan_complete'] is False
    assert result['files_examined'] == 2 and result['candidate_count'] == 0
