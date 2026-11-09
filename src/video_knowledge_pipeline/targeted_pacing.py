"""Read-only, source-bound bridge to the existing MVP v3 pacing candidates.

Audition is reversible. A pending suggestion never becomes approved media EDL.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

PACING_FILENAME = 'targeted-transcript-pacing.candidates.json'

def load_pacing_candidates(root: Path, pack: dict) -> dict | None:
    path = root / PACING_FILENAME
    if not path.is_file(): return None
    value = json.loads(path.read_text(encoding='utf-8'))
    unsigned = {k: v for k, v in value.items() if k != 'sha256'}
    digest = hashlib.sha256(json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if value.get('schema') != 'vkp.targeted-pacing-candidates.v1' or value.get('sha256') != digest or value.get('pack_payload_sha256') != pack['pack_payload_sha256']:
        raise ValueError('transcript_pacing_source_changed')
    sources = {c['number']: c for c in pack['transcript_edit']['chapters']}
    seen = set()
    for row in value.get('candidates', []):
        source = sources.get(row.get('chapter'))
        if not source or row.get('id') in seen or row.get('source_hash') != source['source_text_hash'] or row.get('source_sha256') != source['source_sha256']:
            raise ValueError('transcript_pacing_source_changed')
        seen.add(row['id'])
        times = [row.get(k) for k in ('protect_start', 'pause_start', 'start', 'end', 'pause_end', 'protect_end')]
        if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) for t in times) or not 0 <= times[0] <= times[1] < times[2] < times[3] < times[4] <= times[5] <= source['duration']:
            raise ValueError('transcript_pacing_time_invalid')
        for segment in source['segments']:
            if any(min(row['end'], t['end']) > max(row['start'], t['start']) + 1e-7 for t in segment['timings']):
                raise ValueError('transcript_pacing_cut_overlaps_word')
    return value

def validate_pacing_review(state, packet) -> None:
    if state is None: return
    if not packet or not isinstance(state, dict) or state.get('packet_sha256') != packet['sha256'] or not isinstance(state.get('enabled'), bool) or not isinstance(state.get('decisions'), dict):
        raise ValueError('transcript_pacing_review_invalid')
    ids = {r['id'] for r in packet['candidates']}
    for ident, row in state['decisions'].items():
        if ident not in ids or not isinstance(row, dict) or row.get('action') not in {'keep_pending_review', 'keep', 'compress'}:
            raise ValueError('transcript_pacing_decision_invalid')
        if row['action'] == 'compress' and not (row.get('audio_checked') is True and row.get('context_checked') is True and row.get('semantic_role') == 'ordinary' and isinstance(row.get('evidence'), str) and row['evidence'].strip()):
            raise ValueError('transcript_pacing_audio_context_review_required')
