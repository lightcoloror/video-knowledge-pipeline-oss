"""Read-only provenance projection for existing VKP evidence votes.

Adapts W3C PROV derivation semantics, not a truth scorer or a workflow store.
Raw rows remain untouched. Root hashes here are declared identities; admission
and file verification remain the responsibility of the existing import gates.
"""
from __future__ import annotations

from graphlib import CycleError, TopologicalSorter
import re
from typing import Any

from .canonical_json import canonical_json_sha256

ROOT_HASH_FIELDS = ('original_source_sha256', 'source_media_sha256', 'source_audio_sha256')
HASH_FIELDS = ROOT_HASH_FIELDS + ('frame_sha256', 'artifact_sha256')
VISUAL_TYPES = frozenset({'ocr', 'visual_text', 'structured_visual', 'visual_understanding',
                          'temporal_visual', 'tagger', 'qinglong_tagger'})
AUDIO_TYPES = frozenset({'asr_or_subtitle', 'secondary_asr', 'platform_subtitle', 'embedded_subtitle'})


def _hash(value: Any) -> str:
    value = str(value or '').lower()
    return value if re.fullmatch(r'[0-9a-f]{64}', value) else ''


def _parents(row: dict[str, Any]) -> tuple[list[str], bool]:
    value = row.get('derived_from_evidence_ids', [])
    if not isinstance(value, list) or len(value) > 512:
        return [], False
    if any(not isinstance(x, str) or not x or len(x) > 512 for x in value):
        return [], False
    return sorted(set(value)), True


def _recognition_tokens(value: Any) -> tuple[set[str], list[str]]:
    """Only correlate recorded inputs; never establish independent sources."""
    if value is None:
        return set(), []
    if not isinstance(value, dict):
        return set(), ['malformed_recognition_provenance']
    tokens, problems = set(), []
    for key, prefix in (('frame_sha256s', 'digest:'), ('sent_image_sha256s', 'digest:'), ('context_text_sha256s', 'text-input:')):
        hashes = value.get(key, [])
        if (not isinstance(hashes, list) or len(hashes) > 32
                or any(not _hash(digest) for digest in hashes)):
            problems.append('malformed_recognition_' + key)
        else:
            tokens.update(prefix + _hash(digest) for digest in hashes)
    observation = value.get('observation_text_sha256')
    if observation:
        if _hash(observation):
            tokens.add('text-input:' + _hash(observation))
        else:
            problems.append('malformed_recognition_observation_hash')
    if value.get('problems'):
        problems.append('unresolved_recognition_provenance')
    return tokens, problems


def project_recognition_provenance(observation: Any, *, selected_field: list[str], text: str) -> dict[str, Any]:
    """Bounded metadata projection; no file reads, inferred model, or trust promotion.

    Use the actually selected human/model object, not a neighbouring machine
    result. Importers own input validation; consumers only report recorded data.
    """
    source = observation if isinstance(observation, dict) else {}
    recorded = source.get('recognition_provenance')
    _, problems = _recognition_tokens(recorded)
    recorded = recorded if isinstance(recorded, dict) else {}
    human = bool(selected_field and (selected_field[0].startswith('human_')
                                    or selected_field[0] == 'human_review'))
    def scalar(key):
        value = recorded.get(key, source.get(key, ''))
        if not isinstance(value, str) or len(value) > 512:
            problems.append('invalid_recognition_' + key)
            return ''
        return value
    result = dict(schema='video_knowledge_pipeline.recognition_provenance.v1',
        selected_field=list(selected_field),
        executor='human_review' if human else scalar('executor'),
        model_actual=scalar('model_actual'), model_requested=scalar('model_requested'),
        task_pack_sha256=_hash(recorded.get('task_pack_sha256', source.get('task_pack_sha256'))),
        observation_text_sha256=canonical_json_sha256(text) if text else '',
        frame_sha256s=[], context_text_sha256s=[],
        independence_is_verified=False, scope='recognition_inputs_not_independent_factual_sources')
    for key in ('frame_sha256s', 'sent_image_sha256s', 'context_text_sha256s'):
        hashes = recorded.get(key, [])
        if isinstance(hashes, list) and len(hashes) <= 32 and all(_hash(digest) for digest in hashes):
            result[key] = sorted(set(_hash(digest) for digest in hashes))
    result['prompt_sha256'] = _hash(recorded.get('prompt_sha256'))
    result.update(status='invalid' if problems else 'recorded' if recorded else 'legacy_inputs_unrecorded',
                  problems=problems)
    return result


def call_visual_with_provenance(*, root, candidate, original_image_paths, sent_image_paths,
                                prompt, provider_config, invoke):
    """Adapt existing Bundle path/hash gates around an already-authorized call.

    Pre/post snapshots are not transport attestation or proof of observation.
    No retry, Provider, image transform or new writeback state is introduced.
    """
    from pathlib import Path
    from .courseware_material import safe_path
    from .file_hash import sha256_file

    def snapshot():
        groups = {}
        total_bytes = 0
        for key, paths in (('frames', original_image_paths), ('sent', sent_image_paths)):
            if not isinstance(paths, list) or not 1 <= len(paths) <= 32:
                raise ValueError('visual input count outside 1..32')
            rows = []
            for value in paths:
                path = safe_path(Path(root), str(value))
                if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
                    raise ValueError('visual input must be an existing Bundle image')
                size = path.stat().st_size
                total_bytes += size
                if size > 64 * 1024 * 1024 or total_bytes > 256 * 1024 * 1024:
                    raise ValueError('visual input identity budget exceeded')
                rows.append(dict(path=path.relative_to(root).as_posix(), sha256=sha256_file(path)))
            groups[key] = rows
        return groups

    try:
        before = snapshot()
    except (OSError, ValueError) as exc:
        return dict(ok=False, complete=False, executed=False, status='input_identity_blocked',
                    error='input_identity_blocked: ' + str(exc), content='')
    response = dict(invoke())
    provenance = dict(
        schema='video_knowledge_pipeline.recognition_provenance.v1', executor='provider_api',
        model_requested=str(provider_config.get('model') or ''), model_actual='',
        input_identity_basis='pre_post_file_snapshot_not_transport_attestation',
        independence_is_verified=False,
        frame_sha256s=sorted({row['sha256'] for row in before['frames']}),
        sent_image_sha256s=sorted({row['sha256'] for row in before['sent']}),
        context_text_sha256s=sorted({canonical_json_sha256(str(candidate[key]))
                                    for key in ('transcript', 'visual_text') if candidate.get(key)}),
        prompt_sha256=canonical_json_sha256(prompt), input_artifacts=before, problems=[])
    # Only response metadata, never the generated JSON body or singleton route.
    runtime = response.get('runtime_result')
    metadata = runtime.get('response') if isinstance(runtime, dict) else None
    if not isinstance(metadata, dict):
        metadata = response.get('raw_response')
    if isinstance(metadata, dict):
        model = metadata.get('model') or metadata.get('modelVersion') or ''
        if isinstance(model, str) and len(model) <= 512:
            provenance['model_actual'] = model
    try:
        if snapshot() != before:
            raise ValueError('input_identity_drift')
    except (OSError, ValueError):
        provenance['problems'] = ['input_identity_drift']
        response.update(ok=False, complete=False, status='input_identity_drift', error='input_identity_drift')
    return {**response, 'recognition_provenance': provenance,
            'model_actual': provenance['model_actual'], 'model_requested': provenance['model_requested']}


def aggregate_source_votes(evidence: list[dict[str, Any]], votes: list[dict[str, Any]]) -> dict[str, Any]:
    """Cap each side to its strongest vote per correlated provenance component.

    Never collapse contrary observations into a majority. Unknown source families
    cannot establish independence; derived parent cycles/missing IDs force review.
    Output is deterministic and contains metadata, never source text or paths.
    """
    rows = [row for row in evidence if isinstance(row, dict)]
    if len(rows) != len(votes) or len(rows) > 512:
        raise ValueError('evidence lineage requires matched bounded rows/votes (max 512)')
    ids = [str(row.get('evidence_id') or '') for row in rows]
    known = set(ids) - {''}
    duplicate_ids = {name for name in known if ids.count(name) > 1}
    graph: dict[str, set[str]] = {}
    records = []
    for position, row in enumerate(rows):
        name = ids[position]
        parents, valid = _parents(row)
        roots = {_hash(row.get(key)) for key in ROOT_HASH_FIELDS} - {''}
        hashes = {_hash(row.get(key)) for key in HASH_FIELDS} - {''}
        tokens = {'digest:' + value for value in hashes}
        if name:
            tokens.add('id:' + name)
        tokens.update('id:' + parent for parent in parents)
        recognition_tokens, problems = _recognition_tokens(row.get('recognition_provenance'))
        tokens.update(recognition_tokens)
        if not valid:
            problems.append('malformed_parent_list')
        if set(parents) - known:
            problems.append('unresolved_parent')
        if name in duplicate_ids:
            problems.append('duplicate_evidence_id')
        if not roots:
            kind = str(row.get('source_type') or '')
            family = 'visual' if kind in VISUAL_TYPES else 'audio' if kind in AUDIO_TYPES else kind or 'unknown'
            # Conservative fallback: different recognizers/paths are not evidence
            # of different original sources. Preserve raw votes for inspection.
            tokens.add('unverified-family:' + family)
        graph.setdefault(name or f'@anonymous-{position}', set()).update(parents)
        records.append(dict(tokens=tokens, roots=roots, problems=problems, positions={position}))
    sorter = TopologicalSorter(graph)
    try:
        sorter.prepare()
    except CycleError:
        pass  # get_ready remains usable; retain every cycle, not only the first.
    resolved = set()
    while sorter.is_active():
        ready = sorter.get_ready()
        if not ready:
            break
        resolved.update(ready)
        sorter.done(*ready)
    cyclic = set(graph) - resolved

    # Bounded union of aliases/PROV derivation references; no graph database.
    groups: list[dict[str, Any]] = []
    for record in records:
        matching = [group for group in groups if group['tokens'] & record['tokens']]
        for group in matching:
            record['tokens'] |= group['tokens']
            record['roots'] |= group['roots']
            record['positions'] |= group['positions']
            record['problems'] += group['problems']
            groups.remove(group)
        groups.append(record)
    summaries = []
    total = dict(candidate_weight=0, original_weight=0, neutral_weight=0)
    declared_groups = 0
    for group in groups:
        positions = group['positions']
        names = sorted({ids[p] for p in positions if ids[p]})
        problems = set(group['problems'])
        if cyclic & set(names):
            problems.add('cyclic_derivation')
        if group['roots'] and not problems:
            declared_groups += 1
        weights = {}
        for vote_name, field in [('supports_candidate','candidate_weight'),
                                 ('supports_original','original_weight'),('neutral','neutral_weight')]:
            weights[field] = max((int(votes[p]['source_weight']) for p in positions
                                 if votes[p]['vote'] == vote_name), default=0)
            total[field] += weights[field]
        direct_votes = {votes[p]['vote'] for p in positions
                        if rows[p].get('source_type') != 'asr_or_subtitle'}
        summaries.append(dict(
            group_id=canonical_json_sha256(sorted(group['tokens'])),
            evidence_ids=names, source_types=sorted({str(rows[p].get('source_type') or 'unknown') for p in positions}),
            observation_identity_status='declared_root' if group['roots'] and not problems else 'unverified',
            problems=sorted(problems), weights=weights,
            has_conflict=bool(weights['candidate_weight'] and weights['original_weight']),
            has_direct_conflict={'supports_candidate', 'supports_original'} <= direct_votes,
            evidence_count=len(positions)))
    summaries.sort(key=lambda group: group['group_id'])
    return dict(
        **total,
        schema='video_knowledge_pipeline.evidence_lineage.v1',
        correlated_group_count=sum(group['evidence_count'] > 1 for group in summaries),
        declared_source_group_count=declared_groups,
        independent_source_count=0,
        independence_is_verified=False,
        independence_note='Declared root groups only; never an automatic approval or factual probability.',
        unverified_evidence_count=sum(not any(_hash(row.get(key)) for key in ROOT_HASH_FIELDS) for row in rows),
        intra_group_conflict_count=sum(group['has_conflict'] for group in summaries),
        intra_group_direct_conflict_count=sum(group['has_direct_conflict'] for group in summaries),
        unresolved_group_count=sum(bool(group['problems']) for group in summaries),
        groups=summaries)
