"""Bounded recovery of failed Qwen windows using the existing local ASR runner.

Intent: reuse healthy checkpoint text and shrink only exhausted fixed windows.
Decision: retain the old output, create a new contract, and compose references.
Reason: changing chunk size in an old checkpoint would destroy its identity.
Evidence: Qwen's per-window retry commands and media-absolute timestamp contract.
Scope: local derived ASR evidence only; no canonical promotion or model switch.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .asr_adapter import read_asr_segment_dicts
from .asr_execution import run_asr_plan
from .asr_review_audit import build_review_audit
from .file_hash import sha256_file
from .storage import read_json as _read_json, write_json

SCHEMA = 'video_knowledge_pipeline.qwen_failed_window_recovery.v1'


def read_json(path: str | Path) -> Any:
    return _read_json(Path(path))


def _binding(path: Path) -> dict[str, Any]:
    return {'path': str(path.resolve()), 'sha256': sha256_file(path)}


def _checked(binding: dict[str, Any]) -> Path:
    path = Path(binding['path']).resolve()
    if not path.is_file() or sha256_file(path) != binding.get('sha256'):
        raise ValueError('recovery evidence hash drift')
    return path


def _arg(command: list[str], flag: str, default: str = '') -> str:
    return command[command.index(flag) + 1] if flag in command else default


def _set_arg(command: list[str], flag: str, value: str) -> None:
    if flag in command:
        command[command.index(flag) + 1] = value
    else:
        command.extend([flag, value])


def _model_identity(payload: dict[str, Any]) -> dict[str, str]:
    value = {'model': str(payload.get('model') or ''),
             'forced_aligner': str(payload.get('forced_aligner') or '')}
    value['declared_identity_sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def _activity_binding(path: str | Path | None, media: Path) -> dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path).expanduser().resolve()
    payload = read_json(candidate)
    source = payload.get('source_media') if isinstance(payload.get('source_media'), dict) else {}
    expected = payload.get('source_media_sha256') or payload.get('media_sha256') or source.get('sha256')
    if not expected or expected != sha256_file(media):
        raise ValueError('recovery activity evidence requires a matching media hash')
    audit = build_review_audit([], [], [], media=media, speech_activity_json=candidate)
    return {**_binding(candidate), 'evidence_kind': audit['coverage']['evidence_kind'],
            'media_binding': audit['coverage']['binding']['media_binding']}


def _validate_plan(plan: dict[str, Any]) -> tuple[Path, Path]:
    if plan.get('schema') != SCHEMA+'.plan':
        raise ValueError('invalid Qwen recovery plan')
    identity = {key: plan[key] for key in ('partial', 'media', 'model_identity',
        'window_seconds', 'selected_windows', 'remaining_windows', 'activity')}
    if hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() != plan['plan_id']:
        raise ValueError('recovery plan identity drift')
    media, partial = _checked(plan['media']), _checked(plan['partial'])
    retained = Path(plan['retained_partial_json'])
    if retained.is_file() and sha256_file(retained) != plan['partial']['sha256']:
        raise ValueError('retained recovery evidence hash drift')
    raw = read_json(partial)
    if _model_identity(raw) != plan['model_identity']:
        raise ValueError('recovery model identity drift')
    expected = list(raw['failed_chunks'][0]['retry_command']['command'])
    root = Path(plan['output_dir']).resolve()
    child = plan['child_plan']
    output = root / 'child/raw-asr-output.json'
    for flag, value in (('--output', str(output)), ('--chunk-seconds', str(plan['window_seconds'])),
        ('--chunk-indexes', ','.join(str(row['index']) for row in plan['selected_windows']))):
        _set_arg(expected, flag, value)
    if (child.get('command') != expected or child.get('expected_output_json') != str(output)
            or child.get('output_dir') != str(output.parent)
            or child.get('project') != str(output.parent)):
        raise ValueError('recovery child execution contract drift')
    if plan.get('activity'):
        _activity_binding(_checked(plan['activity']), media)
    return media, partial


def plan_qwen_failed_window_recovery(partial_json: str | Path, *, output_dir: str | Path,
                                     asr_plan_json: str | Path | None = None,
                                     media_path: str | Path | None = None,
                                     speech_activity_json: str | Path | None = None,
                                     window_seconds: int = 60, max_windows: int = 24,
                                     write: bool = True) -> dict[str, Any]:
    """Prepare a new fixed-window contract after the original retries expire."""
    if type(window_seconds) is not int or type(max_windows) is not int or min(window_seconds, max_windows) < 1:
        raise ValueError('recovery window_seconds and max_windows must be positive integers')
    partial_path = Path(partial_json).expanduser().resolve()
    raw = read_json(partial_path)
    failed = raw.get('failed_chunks')
    if raw.get('provider') != 'qwen3-asr' or raw.get('status') not in {'degraded', 'failed'}:
        raise ValueError('recovery requires a failed native Qwen output')
    if not isinstance(failed, list) or not failed or not all(isinstance(r, dict) for r in failed):
        raise ValueError('recovery requires explicit failed windows')
    if not all(row.get('retry_exhausted') is True for row in failed):
        raise ValueError('original same-contract retry budget is not exhausted')
    if window_seconds >= int(raw.get('chunk_seconds') or 0):
        raise ValueError('recovery must use a smaller window than the original contract')
    media = Path(media_path or raw.get('input_path') or '').expanduser().resolve()
    if not media.is_file() or sha256_file(media) != raw.get('input_sha256'):
        raise ValueError('recovery media hash differs from original input')
    if raw.get('timestamp_output_basis') != 'media_absolute':
        raise ValueError('recovery requires native media-absolute source timestamps')
    duration = float((raw.get('window_plan') or {}).get('duration_ms') or 0) / 1000
    duration = duration or float((raw.get('window_plan') or {}).get('duration_seconds') or 0)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('recovery requires native media duration in the window plan')
    windows: dict[int, dict[str, Any]] = {}
    for row in failed:
        start, end = float(row['start']), float(row['end'])
        if not math.isfinite(start + end) or start < 0 or end <= start or end > duration + .001:
            raise ValueError('invalid failed window bounds')
        if abs(start / window_seconds - round(start / window_seconds)) > .00001:
            raise ValueError('failed window cannot be partitioned without touching healthy audio')
        if abs(end / window_seconds - round(end / window_seconds)) > .00001 and abs(end-duration) > .001:
            raise ValueError('failed window end is not a fixed-window boundary')
        for index in range(round(start / window_seconds), math.ceil(end / window_seconds)):
            windows[index] = {'index': index, 'start': index * window_seconds,
                              'end': min((index + 1) * window_seconds, end),
                              'original_failed_chunk_index': int(row['chunk_index'])}
    all_windows = [windows[index] for index in sorted(windows)]
    selected, remaining = all_windows[:max_windows], all_windows[max_windows:]
    root = Path(output_dir).expanduser().resolve()
    command = list(failed[0].get('retry_command', {}).get('command') or [])
    if '-m' not in command or _arg(command, '-m') != 'video_knowledge_pipeline.qwen3_asr_python_runner':
        raise ValueError('recovery retry command is not the native Qwen runner')
    if _arg(command, '--model') != raw.get('model') or _arg(command, '--forced-aligner') != str(raw.get('forced_aligner') or ''):
        raise ValueError('recovery model or aligner identity differs')
    if Path(_arg(command, '--input')).resolve() != media:
        raise ValueError('recovery command media differs')
    child_raw = root / 'child/raw-asr-output.json'
    for flag, value in (('--output', str(child_raw)), ('--chunk-seconds', str(window_seconds)),
                        ('--chunk-indexes', ','.join(str(row['index']) for row in selected))):
        _set_arg(command, flag, value)
    base = read_json(asr_plan_json) if asr_plan_json else {}
    base = dict(base.get('asr_plan') or base)
    child = {**base, 'project': str(root / 'child'), 'output_dir': str(child_raw.parent),
             'expected_output_json': str(child_raw), 'command': command,
             'provider': 'qwen3_asr', 'runner': 'qwen3_asr_python',
             'available': base.get('available') is True,
             'media_path': str(media), 'asr_mode': 'full'}
    identity = {'partial': _binding(partial_path), 'media': _binding(media),
                'model_identity': _model_identity(raw), 'window_seconds': window_seconds,
                'selected_windows': selected, 'remaining_windows': remaining,
                'activity': _activity_binding(speech_activity_json, media)}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    plan = {'schema': SCHEMA+'.plan', 'status': 'planned', 'plan_id': fingerprint,
            **identity, 'output_dir': str(root), 'child_plan': child,
            'child_plan_json': str(root / 'child-plan.json'),
            'retained_partial_json': str(root / 'retained-partial.json'),
            'composed_reference_json': str(root / 'secondary-composed-reference.json'),
            'operator_boundary': {'new_contract': True, 'old_checkpoint_immutable': True,
                'model_switch_allowed': False, 'automatic_canonical_promotion_allowed': False,
                'model_binding_is_declared_identity_not_weight_attestation': True}}
    if write:
        root.mkdir(parents=True, exist_ok=True)
        old_plan = root / 'recovery-plan.json'
        if old_plan.is_file() and read_json(old_plan).get('plan_id') != fingerprint:
            raise ValueError('existing recovery contract differs; use a new output directory')
        retained = Path(plan['retained_partial_json'])
        if retained.exists() and retained.read_bytes() != partial_path.read_bytes():
            raise ValueError('immutable retained recovery input drift')
        if not retained.exists():
            retained.write_bytes(partial_path.read_bytes())
        write_json(root / 'child-plan.json', child)
        write_json(root / 'recovery-plan.json', plan)
    return plan


def compose_qwen_failed_window_recovery(plan: dict[str, Any] | str | Path, *,
                                       recovered_json: str | Path | None = None,
                                       write: bool = True) -> dict[str, Any]:
    """Bind prior healthy output and native absolute child times without offsetting."""
    plan = read_json(plan) if not isinstance(plan, dict) else plan
    media, partial_path = _validate_plan(plan)
    partial = read_json(partial_path)
    if _model_identity(partial) != plan['model_identity']:
        raise ValueError('recovery model identity drift')
    child_path = Path(recovered_json or plan['child_plan']['expected_output_json']).resolve()
    child = read_json(child_path)
    if child.get('input_sha256') != plan['media']['sha256'] or _model_identity(child) != plan['model_identity']:
        raise ValueError('recovered output media/model identity differs')
    if child.get('timestamp_output_basis') != 'media_absolute' or child.get('chunk_seconds') != plan['window_seconds']:
        raise ValueError('recovered timestamp/window contract differs')
    selected = {row['index']: row for row in plan['selected_windows']}
    actual = list(child.get('successful_chunk_indexes') or [])
    if len(set(actual)) != len(actual) or not set(actual) <= selected.keys():
        raise ValueError('recovered output contains unrequested or duplicate windows')
    blocks = child.get('results') or []
    if {row.get('chunk_index') for row in blocks} != set(actual):
        raise ValueError('recovered output window identities differ from results')
    for block in blocks:
        expected = selected[block['chunk_index']]
        if (abs(float(block.get('window_start_seconds', -1))-expected['start']) > .001
                or abs(float(block.get('window_end_seconds', -1))-expected['end']) > .001):
            raise ValueError('recovered window bounds differ')
    healthy = read_asr_segment_dicts(partial_path, provider='qwen3-asr')
    recovered = read_asr_segment_dicts(child_path, provider='qwen3-asr')
    for row in recovered:
        if not any(window['start']-.001 <= row['start'] <= row['end'] <= window['end']+.001
                   for index, window in selected.items() if index in actual):
            raise ValueError('recovered timestamp is outside a requested window')
    for row in healthy:
        if any(min(row['end'], window['end']) > max(row['start'], window['start'])
               for window in plan['selected_windows']):
            raise ValueError('recovery would duplicate healthy audio')
    rows=[]
    for label, values in (('retained', healthy), ('recovered', recovered)):
        for index, row in enumerate(values):
            rows.append({**row, 'segment_id': f'{label}-{index:06d}', 'composition_source': label})
    rows.sort(key=lambda row: (row['start'], row['end']))
    remaining = [row for index, row in selected.items() if index not in actual] + plan['remaining_windows']
    complete = (not remaining and child.get('status') == 'completed'
                and not child.get('failed_chunks') and not child.get('failed_chunk_count'))
    payload = {'schema': SCHEMA+'.reference', 'provider': 'qwen3-asr',
        'status': 'completed_composed_reference' if complete else 'partial_composed_reference',
        'ok': complete, 'model': partial['model'], 'forced_aligner': partial.get('forced_aligner', ''),
        'input_path': str(media), 'input_sha256': plan['media']['sha256'],
        'segments': rows, 'text': ''.join(str(row['text']) for row in rows),
        'composition': [{'role': 'retained_healthy',
                         'path': plan['retained_partial_json'] if Path(plan['retained_partial_json']).is_file() else str(partial_path),
                         'sha256': plan['partial']['sha256'], 'original_source_path': str(partial_path)},
                        {'role': 'recovered_missing_windows', **_binding(child_path)}],
        'remaining_windows': remaining, 'speech_activity_json': plan.get('activity', {}).get('path', '') if plan.get('activity') else '',
        'speech_activity_binding': plan.get('activity'), 'timestamp_output_basis': 'media_absolute',
        'single_run_execution_attested': False, 'automatic_canonical_promotion_allowed': False,
        'recovery_plan_id': plan['plan_id']}
    if write:
        write_json(Path(plan['composed_reference_json']), payload)
    return payload


def run_qwen_failed_window_recovery(plan: dict[str, Any] | str | Path, *, execute: bool = False,
                                   timeout_seconds: int = 900, write: bool = True) -> dict[str, Any]:
    plan = read_json(plan) if not isinstance(plan, dict) else plan
    _validate_plan(plan)
    if not execute:
        return {'status': 'planned', 'ok': False, 'plan': plan, 'execute': False}
    if not write:
        raise ValueError('recovery execute requires durable writes')
    if read_json(plan['child_plan_json']) != plan['child_plan']:
        raise ValueError('saved recovery child plan drift')
    run = run_asr_plan(plan['child_plan_json'], execute=True, normalize=False,
                       resume=True, timeout_seconds=timeout_seconds)
    raw = run.get('raw_output_json') or plan['child_plan']['expected_output_json']
    if not Path(raw).is_file():
        result = {'status': 'recovery_failed', 'ok': False, 'run': run, 'plan_id': plan['plan_id']}
    else:
        reference = compose_qwen_failed_window_recovery(plan, recovered_json=raw, write=True)
        result = {'status': reference['status'], 'ok': reference['ok'], 'run': run,
                  'path': plan['composed_reference_json'], 'plan_id': plan['plan_id'],
                  'remaining_windows': reference['remaining_windows'],
                  'speech_activity_json': reference['speech_activity_json']}
    write_json(Path(plan['output_dir']) / 'recovery-result.json', result)
    return result
