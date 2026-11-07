"""Thin RAGChecker-style stage matrix over VKP's existing reference matcher.

This measures declared important-point retention, not NLI, truth, or whole-media
completeness. Sources remain in Bundle artifacts; no workflow or model runtime.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .canonical_json import canonical_json_sha256
from .courseware_material import safe_path, _block_content
from .evidence_locator import JsonEvidenceSnapshot, audit_fact_pack_locators, read_evidence_snapshot, _unique_object
from .smart_summary_keypoint_eval import evaluate_human_key_point_recall, reference_binding_status, _summary_fragments
from .summary_citation_audit import build_citation_index, _review_only

SCHEMA = 'video_knowledge_pipeline.summary_stage_coverage.v1'
REFERENCE_PATHS = ('quality-benchmark-key-points.json', 'exports/human-key-points.json')
MAX_TEXT_CHARS = 2_000_000
REDUCE_PATH = 'exports/smart-summary-global-reduce.json'
PACK_PATH = 'exports/smart-summary-chapter-fact-pack.json'
PROMPT_PATH = 'exports/smart-summary-global-reduce-prompt.md'


def _reference(root):
    for name in REFERENCE_PATHS:
        if safe_path(root, name, exists=False).exists():
            snap = JsonEvidenceSnapshot.read(root, name, max_bytes=1024 * 1024)
            payload = snap.data
            points = payload.get('key_points') if isinstance(payload, dict) else payload
            if not isinstance(points, list) or not 1 <= len(points) <= 128:
                raise ValueError('stage coverage requires 1-128 explicit reference points')
            freshness = reference_binding_status(payload, bundle_root=root)
            if freshness['status'] not in {'fresh', 'legacy_unbound'}:
                raise ValueError('stage reference binding: ' + freshness['status'])
            return payload, snap, freshness
    return None, None, None


def _text(values):
    if len(values) > 4096 or any(not isinstance(value, str) for value in values):
        raise ValueError('stage text units exceed budget or are invalid')
    if sum(len(value) for value in values) > MAX_TEXT_CHARS:
        raise ValueError('stage text budget exceeded; explicit partition required')
    return '\n'.join(values)


def _recall(root, payload, values):
    if values is None:
        return None
    result = evaluate_human_key_point_recall(payload, _text(values), bundle_root=root)
    if not result.get('evaluated'):
        raise ValueError('invalid stage reference: ' + result['detail'])
    ids = [row['id'] for row in result['decisions']]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate important-point identity')
    return result


def _material(root, pack):
    index = build_citation_index(pack)
    if index.errors:
        raise ValueError('stage citation identity invalid: ' + str(index.errors[:3]))
    audit = audit_fact_pack_locators(root, pack)
    # Composite evidence sets are containers, not extra independent snippets.
    names = sorted(index.eligible.intersection(index.snippets))
    units = [index.snippets[name] for name in names]
    _text(units)
    return index, names, units, audit


def _point_payload(payload, decision):
    # Preserve the existing human reference bindings while selecting one point.
    return {**(payload if isinstance(payload, dict) else {}), 'key_points': [{
        key: decision[key] for key in ('id', 'text', 'aliases', 'claim_scope', 'evidence_ids') if key in decision}]}


def _point_names(index, decision, names):
    identifiers = decision['evidence_ids']
    if not identifiers:
        return names
    if any(name not in index.refs for name in identifiers):
        raise ValueError('unknown declared important-point evidence: ' + decision['id'])
    return sorted(index.resolved_ids(identifiers, usable_only=True))


def required_prompt_evidence(bundle_dir, fact_pack):
    """Select original admitted snippets, never insert the reference's answer.

    The existing matcher selects a witness; metadata/group labels and review
    gaps cannot supply text. The global prompt budget still applies unchanged.
    """
    root = Path(bundle_dir).absolute()
    payload, _, _ = _reference(root)
    if payload is None:
        return []
    index, names, units, _ = _material(root, fact_pack)
    matches = _recall(root, payload, units)
    selected = set()
    for point in matches['decisions']:
        parts, owners = [], []
        for name in _point_names(index, point, names):
            fragments = _summary_fragments(index.snippets[name])
            parts.extend(fragments)
            owners.extend([name] * len(fragments))
        row = _recall(root, _point_payload(payload, point), parts)['decisions'][0]
        if row['matched'] and 0 <= row.get('fragment_index', -1) < len(owners):
            name = owners[row['fragment_index']]
            selected.add(name)
    return [{'evidence_id': name, 'snippet': index.snippets[name],
             'fact_status': index.refs[name].get('fact_status', 'candidate_evidence'),
             'source_kind': index.refs[name].get('source_kind', '')} for name in sorted(selected)]


def _input_units(prompt):
    _, marker, raw = prompt.partition('输入 JSON：\n')
    if not marker:
        raise ValueError('stage input payload marker missing')
    body = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(body, dict):
        raise ValueError('stage input payload is not an object')
    units = []
    chapters = body.get('chapters', [])
    required = body.get('required_key_point_evidence', [])
    if not isinstance(chapters, list) or not isinstance(required, list) or len(chapters) + len(required) > 4096:
        raise ValueError('stage input collections invalid or over budget')
    for chapter in chapters:
        if not isinstance(chapter, dict):
            raise ValueError('stage chapter is not an object')
        if chapter.get('evidence_status') not in {'review_only', 'legacy_unbound'}:
            units.append(str(chapter.get('markdown') or ''))
            facts, refs = chapter.get('facts', []), chapter.get('evidence_refs', [])
            if not isinstance(facts, list) or not isinstance(refs, list):
                raise ValueError('stage facts/refs must be lists')
            for row in [*facts, *refs]:
                if not isinstance(row, dict):
                    raise ValueError('stage fact/ref must be an object')
                if not _review_only(row):
                    units.append(str(row.get('text') or row.get('snippet') or ''))
    for row in required:
        if not isinstance(row, dict):
            raise ValueError('required stage evidence must be an object')
        if not _review_only(row):
            units.append(str(row.get('snippet') or ''))
    # This is the existing unabridged courseware channel, not prompt instructions,
    # titles, IDs, or a copied human reference embedded in arbitrary metadata.
    units.append(str(body.get('video_courseware_original_material') or ''))
    return units


def _source_units(root, index, names, *, cache=None):
    cache = {} if cache is None else cache
    values, total_bytes, unknown = [], sum(snap.size_bytes for snap in cache.values()), 0
    seen = set()
    for name in names:
        locators = index.refs[name].get('source_locators') or []
        if not locators:
            unknown += 1
        for loc in locators:
            key = canonical_json_sha256(loc)
            if key in seen:
                continue
            seen.add(key)
            path = loc['artifact_path']
            if path not in cache:
                cache[path] = read_evidence_snapshot(root, path, loc['schema'], max_bytes=64 * 1024 * 1024 - total_bytes)
                total_bytes += cache[path].size_bytes
            snap = cache[path]
            if snap.locator(loc['selector']) != loc:
                raise ValueError('stage source locator drift')
            selected = snap.selected(loc['selector'])
            if index.refs[name].get('observation_kind') in {
                    'visual_understanding', 'temporal_visual_understanding', 'continuous_tag_weak_evidence', 'continuous_tag_frame'}:
                projected = _typed_timeline_source(snap, loc, index.refs[name])
                if projected is None:
                    unknown += 1
                else:
                    values.append(projected)
            elif isinstance(selected, str):
                values.append(selected)
            elif isinstance(selected, dict) and isinstance(selected.get('text'), str):
                values.append(_block_content(selected))
            else:
                unknown += 1  # not a license to flatten arbitrary metadata
    return values if values else None, unknown


def _typed_timeline_source(snapshot, locator, ref):
    """Reproject the bound observation via existing precedence/content contracts.

    This is recognition text retention, not proof of original pixels or truth.
    Never use the fact-pack snippet, model identity, or arbitrary object fields.
    """
    from .visual_evidence import (
        VISUAL_CONTENT_FIELDS, TEMPORAL_CONTENT_FIELDS, visual_evidence_state,
        temporal_evidence_state, understanding_source_selector, continuous_tag_evidence_state,
    )
    from .smart_summary_input_pack import _qualified_evidence_text, _mapping_text
    selector = locator['selector']
    if (locator['artifact_path'] != 'timeline.json' or not isinstance(snapshot.data, list)
            or len(selector) < 2 or type(selector[0]) is not int):
        return None
    item = snapshot.selected([selector[0]])
    if not isinstance(item, dict):
        return None
    kind = ref.get('observation_kind')
    if kind in {'continuous_tag_weak_evidence', 'continuous_tag_frame'}:
        state = continuous_tag_evidence_state(item)
        if not state.get('consumable_weak'):
            return None
        if kind == 'continuous_tag_frame':
            if (len(selector) != 4 or selector[1] != 'temporal_tag_delta'
                    or selector[2] not in {'raw_observations', 'smoothed_observations'} or type(selector[3]) is not int):
                return None
            frame = snapshot.selected(selector)
            if not isinstance(frame, dict):
                return None
            return _mapping_text({key:frame[key] for key in ('tags','ocr_text') if key in frame})
        if selector[1:] not in (['temporal_tag_delta'], ['temporal_tag_delta','coarse_summary']):
            return None
        return state.get('coarse_summary') or None
    if kind not in {'visual_understanding', 'temporal_visual_understanding'}:
        return None
    temporal = kind == 'temporal_visual_understanding'
    state = temporal_evidence_state(item) if temporal else visual_evidence_state(item)
    if selector[1:] != understanding_source_selector(item, temporal=temporal) or not state.get('consumable'):
        return None
    fields = TEMPORAL_CONTENT_FIELDS if temporal else VISUAL_CONTENT_FIELDS
    value = state.get('understanding') or {}
    # Same content-field contract as correction; transport/audit keys are not speech.
    content = {key: value[key] for key in (*fields, 'summary', 'text', 'description') if key in value}
    return _qualified_evidence_text({**state, 'understanding': content}) or None


def evaluate_summary_stage_coverage(bundle_dir, fact_pack, prompt, *, output_text=None):
    root = Path(bundle_dir).absolute()
    report = {'schema': SCHEMA, 'status': 'not_assessed', 'evaluated': False,
              'passed': None, 'input_gate_passed': None, 'points': [],
              'input_basis': 'prepared_payload_not_proof_of_model_observation',
              'assessment_scope': 'declared_important_points_in_supplied_evidence',
              'whole_media_completeness': 'not_assessed', 'factual_truth': 'not_assessed'}
    try:
        report.update(fact_pack_sha256=canonical_json_sha256(fact_pack),
                      prompt_utf8_sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest())
        payload, ref, freshness = _reference(root)
        if payload is None:
            return {**report, 'reason': 'human_reference_missing'}
        index, names, units, audit = _material(root, fact_pack)
        source_cache = {}
        source_units, unknown = _source_units(root, index, names, cache=source_cache)
        texts = {'source_evidence': source_units, 'material': units, 'model_input': _input_units(prompt),
                 'output': None if output_text is None else [output_text]}
        stages = {key: _recall(root, payload, value) for key, value in texts.items()}
        points = []
        for position, row in enumerate(stages['material']['decisions']):
            presence = {key: value['decisions'][position]['matched'] if value else None for key, value in stages.items()}
            if row['evidence_ids']:
                selected = _point_names(index, row, names)
                point_payload = _point_payload(payload, row)
                scoped_source, _ = _source_units(root, index, selected, cache=source_cache)
                for key, values in (('source_evidence', scoped_source),
                                    ('material', [index.snippets[name] for name in selected])):
                    scoped = _recall(root, point_payload, values)
                    presence[key] = scoped['decisions'][0]['matched'] if scoped else None
            loss = next((key for key, value in presence.items() if value is False), None)
            points.append({'id': row['id'], 'stages': presence, 'first_loss': loss})
        input_missing = [row['id'] for row in points if row['stages']['material'] and not row['stages']['model_input']]
        available = sum(row['stages']['material'] for row in points)
        used = sum(bool(row['stages']['material'] and row['stages']['output']) for row in points)
        passed = (freshness['status'] == 'fresh' and not unknown and
                  all(all(value is True for value in row['stages'].values()) for row in points))
        report.update(status='complete_scoped' if passed else 'review_required', evaluated=True, passed=passed,
                      input_gate_passed=not input_missing, input_missing_ids=input_missing,
                      source_unknown_count=unknown, points=points, source_locator_audit=audit,
                      reference_sha256=ref.artifact_sha256, reference_freshness=freshness,
                      fact_pack_sha256=canonical_json_sha256(fact_pack),
                      prompt_utf8_sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                      output_utf8_sha256=hashlib.sha256(output_text.encode('utf-8')).hexdigest() if output_text is not None else None,
                      stage_recall={key: sum(point['stages'][key] is True for point in points) / len(points)
                                    if all(point['stages'][key] is not None for point in points) else None for key in stages},
                      output_recall=stages['output']['recall'] if stages['output'] else None,
                      material_utilization=used / available if available and output_text is not None else None)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        report.update(status='blocked', passed=False, input_gate_passed=False, errors=[str(exc)])
    return report


def stored_summary_stage_coverage(bundle_dir, *, output_text=None):
    """Reproject current evidence, never trust a saved boolean or old output.

    This is deliberately a *prepared-input* audit. A cached/recovered candidate
    does not establish that this prompt was the request that generated it.
    """
    root = Path(bundle_dir).absolute()
    report = {'schema': SCHEMA, 'required': False, 'evaluated': False, 'passed': None,
              'status': 'not_assessed', 'reason': 'legacy_or_no_reduce_stage_contract'}
    try:
        from .transcript_semantic_transaction import summary_transaction_status
        snapshot = summary_transaction_status(root)
        if not snapshot['passed']:
            return {**report, 'required': True, 'passed': False, 'status': 'blocked',
                    'reason': 'interrupted_summary_snapshot', 'snapshot_transaction': snapshot}
        if safe_path(root, 'manifest.json', exists=False).exists():
            manifest = JsonEvidenceSnapshot.read(root, 'manifest.json', max_bytes=4 * 1024 * 1024).data
            report['required'] = isinstance(manifest, dict) and manifest.get('smart_summary_stage_coverage_schema') == SCHEMA
        if not safe_path(root, REDUCE_PATH, exists=False).exists():
            if report['required']:
                raise ValueError('registered stage coverage receipt missing')
            return report
        registered = report['required']
        report['required'] = True  # malformed existing receipt is not legacy success
        saved = JsonEvidenceSnapshot.read(root, REDUCE_PATH, max_bytes=4 * 1024 * 1024).data
        prior = saved.get('stage_coverage') if isinstance(saved, dict) else None
        if not isinstance(prior, dict) or prior.get('schema') != SCHEMA:
            if registered:
                raise ValueError('registered stage coverage contract missing')
            report['required'] = False
            return report
        report['required'] = True
        pack = JsonEvidenceSnapshot.read(root, PACK_PATH, max_bytes=16 * 1024 * 1024).data
        path = safe_path(root, PROMPT_PATH)
        with path.open('rb') as stream:
            raw = stream.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError('saved reduce prompt exceeds audit budget')
        # Existing text writer may use Windows CRLF. Reconstruct the submitted
        # Python string, not an invented byte hash of the platform projection.
        prompt = raw.decode('utf-8-sig').replace('\r\n', '\n')
        if prior.get('prompt_utf8_sha256') != hashlib.sha256(prompt.encode('utf-8')).hexdigest():
            raise ValueError('saved reduce prompt drift')
        if prior.get('fact_pack_sha256') != canonical_json_sha256(pack):
            raise ValueError('saved reduce fact pack drift')
        current = evaluate_summary_stage_coverage(root, pack, prompt, output_text=output_text)
        from .summary_model_input import generation_input_status
        plan_path = 'exports/smart-summary-reader-plan.json'
        plan = JsonEvidenceSnapshot.read(root, plan_path, max_bytes=4 * 1024 * 1024).data if safe_path(root, plan_path, exists=False).exists() else {}
        generation = generation_input_status(root, fact_pack=pack, plan=plan)
        captured = generation.pop('prompt', None)
        if generation.get('passed') is False:
            return {**current, 'required': True, 'status': 'blocked', 'passed': False,
                    'generation_input': generation, 'errors': generation.get('errors', [])}
        if captured is not None:
            prepared_status = current['status']
            current = evaluate_summary_stage_coverage(root, pack, captured, output_text=output_text)
            current.update(input_basis='captured_gateway_input_not_proof_of_model_observation',
                           prepared_stage_status=prepared_status)
        artifacts = {'reduce': REDUCE_PATH, 'fact_pack': PACK_PATH, 'prompt': PROMPT_PATH}
        for key in ('request', 'response', 'binding'):
            if generation.get(key):
                artifacts['generation_' + key] = generation[key]['path']
        return {**current, 'required': True,
                'generation_input_provenance': generation['status'], 'generation_input': generation,
                'artifacts': artifacts}
    except (ValueError, OSError, TypeError, KeyError, UnicodeError) as exc:
        report.update(status='blocked', passed=False, evaluated=True, errors=[str(exc)])
        return report


def current_summary_stage_coverage(bundle_dir):
    """Read-only display projection; use the same preferred output as quality."""
    root = Path(bundle_dir).absolute()
    try:
        from .smart_summary_codex import _quality_summary_body
        from .content_profile import resolve_content_profile
        body, output = None, None
        for name in ('exports/smart-summary.codex.md', 'exports/smart-summary.md'):
            path = safe_path(root, name, exists=False)
            if path.exists():
                with path.open('rb') as stream:
                    raw = stream.read(2 * 1024 * 1024 + 1)
                if len(raw) > 2 * 1024 * 1024:
                    raise ValueError('summary display projection exceeds audit budget')
                profile = resolve_content_profile(root)
                body = _quality_summary_body(raw.decode('utf-8-sig').replace('\r\n', '\n'),
                    interview_profile=profile.get('profile_id') in {'interview-v1', 'medical-insurance-interview-v1'})
                output = name
                break
        return {**stored_summary_stage_coverage(root, output_text=body), 'output_path': output}
    except (ValueError, OSError, TypeError, UnicodeError) as exc:
        return {'schema': SCHEMA, 'required': True, 'evaluated': True, 'passed': False,
                'status': 'blocked', 'errors': [str(exc)]}
