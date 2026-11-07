from __future__ import annotations

import difflib
import copy
import hashlib
import math
import re
from pathlib import Path
from .asr_comparison_alignment import locate_primary_span, critical_differences
from typing import Any

from .models import now_iso
from .storage import bundle_write_lock, read_json, write_json
from .transcript import format_timestamp, parse_transcript
from .file_hash import sha256_file
from .text_normalization import compact_ascii_cjk


PACK_SCHEMA = "video_knowledge_pipeline.asr_diff_adjudication_pack.v1"
RESULT_SCHEMA = "video_knowledge_pipeline.asr_diff_adjudication_result.v1"
TRANSCRIPT_SCHEMA = "video_knowledge_pipeline.asr_consensus_patched_transcript.v1"
_TOKEN_RE = re.compile(r"[一-鿿]|[A-Za-z]+(?:[-_.][A-Za-z0-9]+)*|\d+(?:\.\d+)?|[^\s]", re.UNICODE)


def build_asr_diff_adjudication(
    bundle_dir: str | Path,
    *,
    consensus_json: str | Path | None = None,
    cluster_token_gap: int = 6,
    write: bool = True,
) -> dict[str, Any]:
    """Build positioned, clustered and source-anonymous ASR disagreements."""

    root = Path(bundle_dir).expanduser().resolve()
    consensus_path = _bundle_path(root, consensus_json or "asr-consensus.json")
    consensus = read_json(consensus_path)
    if not isinstance(consensus, dict):
        raise ValueError("asr-consensus.json must be a JSON object")
    primary_cues = parse_transcript(consensus['primary_transcript'])
    secondary_cues = parse_transcript(consensus['secondary_transcript'])

    differences: list[dict[str, Any]] = []
    clusters: list[dict[str, Any]] = []
    for window in consensus.get("windows") or []:
        if not isinstance(window, dict) or not window.get('review_required', window.get('status') in {'conflict','primary_only','secondary_only'}):
            continue
        primary_text = str(window.get("primary_text") or "")
        secondary_text = str(window.get("secondary_text") or "")
        window_diffs = _positioned_differences(window, primary_text, secondary_text)
        if not window_diffs:
            # Retain surface-only single-sided marks and agreement samples as
            # explicit review units; neither can silently disappear from closure.
            window_diffs = [{"diff_id":f"{window['window_id']}-surface", "window_id":window['window_id'],
                "kind":"surface_or_agreement_review", "operation":"review", "patchable":False,
                "primary_segment_index":window.get('primary_index'),"secondary_segment_index":window.get('secondary_index'),
                "primary_char_start":0,"primary_char_end":len(primary_text),
                "group_char_start":0,"group_char_end":len(primary_text),
                "secondary_char_start":0,"secondary_char_end":len(secondary_text),
                "primary_token_start":0,"primary_token_end":0,
                "primary_text":primary_text,"secondary_text":secondary_text,
                "left_context":"","right_context":""}]
        for row in window_diffs:
            _locate_diff_time(row,window,primary_cues,secondary_cues)
        differences.extend(window_diffs)
        clusters.extend(_cluster_differences(window, window_diffs, max_gap=max(0, int(cluster_token_gap)),clip_dir=root/'asr-consensus-clips'))

    from .smart_summary_input_pack import select_canonical_transcript_path
    manifest = read_json(root/'manifest.json') if (root/'manifest.json').is_file() else {}
    canonical = select_canonical_transcript_path(root,manifest)
    paths = {'consensus':consensus_path,'primary':Path(consensus['primary_transcript']),
             'secondary':Path(consensus['secondary_transcript'])}
    if canonical:
        paths['canonical'] = canonical
    if (root/'dual-asr-review.json').is_file():
        paths['dual_workflow'] = root/'dual-asr-review.json'
    media=Path(str(consensus.get('media_path') or ''))
    media_binding=({'path':str(media.resolve()),'sha256':sha256_file(media),'bytes':media.stat().st_size}
                   if consensus.get('media_path') and media.is_file() else None)

    pack = {
        "schema": PACK_SCHEMA,
        "bundle_dir": str(root),
        "consensus_json": str(consensus_path),
        "primary_transcript": str(consensus.get("primary_transcript") or ""),
        "secondary_transcript": str(consensus.get("secondary_transcript") or ""),
        "status": "ready_for_adjudication" if clusters else "no_positioned_conflicts",
        "ok": True,
        "difference_count": len(differences),
        "lexical_difference_count": sum(row.get('kind') != 'surface_or_agreement_review' for row in differences),
        "surface_review_count": sum(row.get('kind') == 'surface_or_agreement_review' for row in differences),
        "cluster_count": len(clusters),
        "differences": differences,
        "clusters": clusters,
        "cluster_token_gap": max(0,int(cluster_token_gap)),
        "review_window_ids": list(dict.fromkeys(row['window_id'] for row in differences)),
        "source_bindings": {role:{'path':str(path.resolve()),'sha256':sha256_file(path)} for role,path in paths.items()},
        "source_media_binding": media_binding,
        "decision_schema": {
            "rows": [
                {
                    "diff_id": "<diff-id>",
                    "choice": "A | B | keep_primary",
                    "confidence": "0..1",
                    "evidence_refs": ["audio clip or other evidence path"],
                    "reason": "short evidence-based reason",
                }
            ]
        },
        "operator_boundary": {
            "candidate_sources_anonymous": True,
            "raw_hypotheses_preserved": True,
            "does_not_apply_patches": True,
            "no_llm_or_cloud_call": True,
            "changing_patch_requires_evidence": True,
        },
        "artifacts": {
            "json": "asr-consensus-adjudication-pack.json",
            "markdown": "asr-consensus-adjudication-pack.md",
            "todo_json": "asr-consensus-adjudication.todo.json",
        },
        "updated_at": now_iso(),
    }
    if write:
        root.mkdir(parents=True, exist_ok=True)
        with bundle_write_lock(root, operation="asr_diff_adjudication_pack", timeout_seconds=1.0):
            write_json(root / "asr-consensus-adjudication-pack.json", pack)
            (root / "asr-consensus-adjudication-pack.md").write_text(_render_pack_markdown(pack), encoding="utf-8")
            write_json(
                root / "asr-consensus-adjudication.todo.json",
                {
                    "schema": "video_knowledge_pipeline.asr_diff_adjudication_decisions.v1",
                    "consensus_json": str(consensus_path),
                    "rows": [
                        {
                            "diff_id": row["diff_id"],
                            "choice": "",
                            "confidence": None,
                            "evidence_refs": [],
                            "reason": "",
                        }
                        for row in differences
                    ],
                },
            )
            manifest_path = root / "manifest.json"
            manifest = read_json(manifest_path) if manifest_path.exists() else {}
            if isinstance(manifest, dict):
                manifest["asr_consensus_adjudication_pack_json"] = "asr-consensus-adjudication-pack.json"
                manifest["asr_consensus_adjudication_pack_markdown"] = "asr-consensus-adjudication-pack.md"
                manifest["asr_consensus_adjudication_todo_json"] = "asr-consensus-adjudication.todo.json"
                write_json(manifest_path, manifest)
    return pack


def apply_asr_diff_adjudication(
    bundle_dir: str | Path,
    *,
    decisions_json: str | Path,
    pack_json: str | Path | None = None,
    min_confidence: float = 0.75,
    require_evidence: bool = True,
    promote: bool = False,
    review_only: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    """Validate and apply local ASR patches without modifying raw transcripts."""

    root = Path(bundle_dir).expanduser().resolve()
    pack_path = _bundle_path(root, pack_json or "asr-consensus-adjudication-pack.json")
    decisions_path = _bundle_path(root, decisions_json)
    pack = read_json(pack_path)
    decisions = read_json(decisions_path)
    if not isinstance(pack, dict) or not isinstance(decisions, dict):
        raise ValueError("pack and decisions must be JSON objects")
    if review_only:
        if promote or not require_evidence:
            raise ValueError('review-only requires evidence and forbids promotion')
        if not write:
            return _review_only_result(root,pack_path,decisions_path,min_confidence=min_confidence)
        if write:
            with bundle_write_lock(root,operation='asr_adjudication_review',timeout_seconds=1.0):
                result = _review_only_result(root,pack_path,decisions_path,min_confidence=min_confidence)
                pack,decisions=read_json(pack_path),read_json(decisions_path)
                identity=hashlib.sha256((sha256_file(pack_path)+sha256_file(decisions_path)+str(min_confidence)).encode()).hexdigest()
                archive=root/'runs/asr-adjudication-reviews'/identity/'receipt.json'
                if result['ok']:
                    from .codex_vision_session import _write_bytes_atomic
                    snapshots={'pack':{'path':str(pack_path),'sha256':sha256_file(pack_path)},
                        'decisions':{'path':str(decisions_path),'sha256':sha256_file(decisions_path)},
                        **{f'source-{role}':binding for role,binding in pack['source_bindings'].items()},
                        **{f'evidence-{role}':binding for role,binding in decisions['evidence_bindings'].items()}}
                    history=[]
                    for role,binding in snapshots.items():
                        source=_checked_text_binding(root,binding)
                        copy=archive.parent/'evidence'/(binding['sha256']+source.suffix)
                        raw=source.read_bytes()
                        if copy.exists():
                            if copy.read_bytes()!=raw:
                                raise ValueError('immutable review archive drift')
                        else:
                            _write_bytes_atomic(copy,raw)
                        history.append({'role':role,'path':str(copy.relative_to(root)),'sha256':binding['sha256']})
                    write_json(archive.parent/'evidence-index.json',history)
                if not archive.exists():
                    write_json(archive,result)
                write_json(root/'asr-consensus-adjudication-review.json',result)
        return result

    diff_by_id = {
        str(row.get("diff_id") or ""): row
        for row in pack.get("differences") or []
        if isinstance(row, dict) and row.get("diff_id")
    }
    decision_rows = _decision_rows(decisions)
    accepted: list[dict[str, Any]] = []
    no_change: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    occupied: dict[int, list[tuple[int, int]]] = {}
    for decision in decision_rows:
        diff_id = str(decision.get("diff_id") or "")
        diff = diff_by_id.get(diff_id)
        if not diff:
            rejected.append({"diff_id": diff_id, "reason": "unknown_diff_id"})
            continue
        resolved = _resolve_choice(diff, str(decision.get("choice") or ""))
        if resolved is None:
            rejected.append({"diff_id": diff_id, "reason": "invalid_choice"})
            continue
        confidence = _float(decision.get("confidence"))
        evidence_refs = [str(value) for value in decision.get("evidence_refs") or [] if str(value).strip()]
        primary_text = str(diff.get("primary_text") or "")
        chosen_text = str(resolved)
        if chosen_text == primary_text:
            no_change.append({"diff_id": diff_id, "status": "adjudicated_no_change", "choice": decision.get("choice"), "confidence": confidence})
            continue
        if confidence < float(min_confidence):
            rejected.append({"diff_id": diff_id, "reason": "confidence_below_threshold", "confidence": confidence})
            continue
        if require_evidence and not evidence_refs:
            rejected.append({"diff_id": diff_id, "reason": "evidence_required"})
            continue
        if diff.get("patchable") is False or diff.get("primary_segment_index") is None:
            rejected.append({"diff_id": diff_id, "reason": "unlocatable_or_cross_segment_patch"})
            continue
        segment_index = int(diff["primary_segment_index"])
        start = int(diff.get("primary_char_start") or 0)
        end = int(diff.get("primary_char_end") or start)
        if any(not (end <= left or start >= right) for left, right in occupied.setdefault(segment_index, [])):
            rejected.append({"diff_id": diff_id, "reason": "overlapping_patch"})
            continue
        occupied[segment_index].append((start, end))
        accepted.append(
            {
                "diff_id": diff_id,
                "segment_index": segment_index,
                "start": start,
                "end": end,
                "expected_text": primary_text,
                "replacement_text": chosen_text,
                "confidence": confidence,
                "evidence_refs": evidence_refs,
                "reason": str(decision.get("reason") or ""),
            }
        )

    primary_path = Path(str(pack.get("primary_transcript") or "")).expanduser()
    if not primary_path.exists():
        raise FileNotFoundError(f"primary transcript not found: {primary_path}")
    cues = parse_transcript(primary_path)
    by_segment: dict[int, list[dict[str, Any]]] = {}
    for patch in accepted:
        by_segment.setdefault(int(patch["segment_index"]), []).append(patch)
    segments: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    for index, cue in enumerate(cues):
        original = str(cue.text or "")
        text = original
        for patch in sorted(by_segment.get(index, []), key=lambda row: int(row["start"]), reverse=True):
            start = int(patch["start"])
            end = int(patch["end"])
            expected = str(patch["expected_text"])
            if text[start:end] != expected:
                rejected.append({"diff_id": patch["diff_id"], "reason": "base_span_mismatch", "found_text": text[start:end]})
                continue
            text = text[:start] + str(patch["replacement_text"]) + text[end:]
            applied.append(patch)
        segments.append(
            {
                "index": index + 1,
                "start": float(cue.start),
                "end": float(cue.end),
                "text": text,
                "original_text": original,
                "changed": text != original,
                "source": "asr_diff_adjudication",
            }
        )

    transcript_payload = {
        "schema": TRANSCRIPT_SCHEMA,
        "source": "asr_diff_adjudication",
        "base_transcript": str(primary_path.resolve()),
        "updated_at": now_iso(),
        "summary": {
            "segments": len(segments),
            "applied_patches": len(applied),
            "adjudicated_no_change": len(no_change),
            "rejected": len(rejected),
            "promoted": bool(promote),
        },
        "patches": applied,
        "segments": segments,
    }
    result = {
        "schema": RESULT_SCHEMA,
        "bundle_dir": str(root),
        "status": "completed" if not rejected else "completed_with_rejections",
        "ok": bool(applied or no_change) and not rejected,
        "pack_json": str(pack_path),
        "decisions_json": str(decisions_path),
        "accepted_patches": applied,
        "adjudicated_no_change": no_change,
        "rejected": rejected,
        "summary": transcript_payload["summary"],
        "operator_boundary": {
            "raw_hypotheses_preserved": True,
            "only_positioned_patches_applied": True,
            "evidence_required_for_changes": bool(require_evidence),
            "promotion_explicit": True,
        },
        "artifacts": {
            "json": "asr-consensus-adjudication-result.json",
            "markdown": "asr-consensus-adjudication-result.md",
            "patched_transcript": "asr-consensus-patched-transcript.json",
            "promoted_transcript": "source-arbitrated-transcript.json" if promote else "",
        },
        "updated_at": now_iso(),
    }
    promotion = _canonical_promotion(root, pack, cues, applied, transcript_payload) if promote else None
    if promotion:
        result['canonical_promotion'] = promotion['receipt']
    if write:
        with bundle_write_lock(root, operation="apply_asr_diff_adjudication", timeout_seconds=1.0):
            if promotion:
                _assert_canonical_selection(root, promotion['path'])
            if promotion and promotion['path'] and sha256_file(promotion['path']) != promotion['receipt']['base_sha256']:
                raise ValueError('canonical changed before patch promotion')
            if promotion:
                from .codex_vision_session import _write_bytes_atomic
                for retained in (promotion['path'], root/'source-arbitrated-transcript.json'):
                    if retained and retained.is_file():
                        digest=sha256_file(retained)
                        archive=root/'runs/asr-promotion-preimages'/digest/'canonical-before.json'
                        if archive.exists() and archive.read_bytes()!=retained.read_bytes():
                            raise ValueError('immutable canonical preimage drift')
                        if not archive.exists():
                            _write_bytes_atomic(archive,retained.read_bytes())
            write_json(root / "asr-consensus-patched-transcript.json", transcript_payload)
            write_json(root / "asr-consensus-adjudication-result.json", result)
            (root / "asr-consensus-adjudication-result.md").write_text(_render_result_markdown(result), encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest = read_json(manifest_path) if manifest_path.exists() else {}
            if isinstance(manifest, dict):
                manifest["asr_consensus_patched_transcript_json"] = "asr-consensus-patched-transcript.json"
                manifest["asr_consensus_adjudication_result_json"] = "asr-consensus-adjudication-result.json"
                if promote:
                    promoted_payload = promotion['payload']
                    write_json(root / "source-arbitrated-transcript.json", promoted_payload)
                    manifest["source_arbitrated_transcript_json"] = "source-arbitrated-transcript.json"
                    manifest["corrected_transcript_json"] = "source-arbitrated-transcript.json"
                    manifest["corrected_transcript_source"] = "asr_diff_adjudication"
                write_json(manifest_path, manifest)
    return result


def _canonical_promotion(root, pack, primary_cues, patches, fallback):
    """Reuse positioned patches on the current bound canonical, preserving rows.

    Extra silence placeholders, identifiers, metadata and unrelated corrections
    must never disappear when the independently recognized primary has fewer rows.
    A changed source cue fails closed rather than guessing a new character span.
    """
    binding=pack.get('source_bindings',{}).get('canonical')
    if not binding:
        from .smart_summary_input_pack import select_canonical_transcript_path
        manifest=read_json(root/'manifest.json') if (root/'manifest.json').is_file() else {}
        if select_canonical_transcript_path(root,manifest) is not None:
            raise ValueError('existing canonical requires a current source binding')
        return {'path':None,'payload':{**fallback,'schema':'video_knowledge_pipeline.source_arbitrated_transcript.v1'},
                'receipt':{'basis':'primary_without_existing_canonical','base_sha256':'',
                           'preserved_segment_count':len(fallback['segments'])}}
    path=_checked_text_binding(root,binding)
    _assert_canonical_selection(root,path)
    value=read_json(path)
    rows=value.get('segments')
    if not isinstance(rows,list) or any(not isinstance(row,dict) for row in rows):
        raise ValueError('canonical segments must be objects for positioned promotion')
    promoted=copy.deepcopy(value)
    mapped={}
    for patch in patches:
        cue=primary_cues[patch['segment_index']]
        matches=[i for i,row in enumerate(rows) if str(row.get('text',''))==cue.text
                 and abs(float(row.get('start',-1))-cue.start)<=.001
                 and abs(float(row.get('end',-1))-cue.end)<=.001]
        if len(matches)!=1:
            raise ValueError('canonical cue missing, changed or ambiguous; use semantic correction')
        mapped.setdefault(matches[0],[]).append(patch)
    for index,entries in mapped.items():
        text=promoted['segments'][index]['text']
        for patch in sorted(entries,key=lambda r:r['start'],reverse=True):
            if text[patch['start']:patch['end']]!=patch['expected_text']:
                raise ValueError('canonical patch span changed')
            text=text[:patch['start']]+patch['replacement_text']+text[patch['end']:]
        promoted['segments'][index]['text']=text
    receipt={'basis':'hash_bound_canonical_positioned_patches','base_sha256':binding['sha256'],
             'preserved_segment_count':len(rows),'changed_segment_count':len(mapped),
             'all_other_fields_preserved':True}
    promoted.update(schema='video_knowledge_pipeline.source_arbitrated_transcript.v1',
                    asr_patch_promotion=receipt)
    return {'path':path,'payload':promoted,'receipt':receipt}


def _assert_canonical_selection(root, expected):
    """Recheck under the bundle lock, including the legacy no-canonical case."""
    from .smart_summary_input_pack import select_canonical_transcript_path
    manifest=read_json(root/'manifest.json') if (root/'manifest.json').is_file() else {}
    current=select_canonical_transcript_path(root,manifest)
    if ((current is None) != (expected is None)
            or (current is not None and current.resolve()!=expected.resolve())):
        raise ValueError('canonical selection changed before patch promotion')


def _locate_diff_time(row, window, primary, secondary):
    """Use actual provider spans; word clocks only when their text matches."""
    side = 'primary' if row['primary_text'] else 'secondary'
    spans = window.get(f'{side}_spans')
    start = row.get('group_char_start',0) if side == 'primary' else row['secondary_char_start']
    end = row.get('group_char_end',start) if side == 'primary' else row['secondary_char_end']
    projection = locate_primary_span(spans,start,end) if spans else None
    index = projection[0] if projection else window.get(f'{side}_index')
    cues = primary if side == 'primary' else secondary
    if index is None or not 0 <= index < len(cues):
        row['estimated_time']={'start':float(window['group_start']),'end':float(window['group_end'])}
        row['time_basis']='aggregate_bounds_unresolved'
        return
    cue = cues[index]
    row[f'{side}_segment_index']=index
    a,b = (projection[1],projection[2]) if projection else (start,end)
    row['estimated_time']={'start':float(cue.start),'end':float(cue.end)}
    row['time_basis']=f'{side}_provider_segment'
    words = getattr(cue,'metadata',{}).get('words') or []
    if not words or compact_ascii_cjk(''.join(str(w.get('word',w.get('text',''))) for w in words)) != compact_ascii_cjk(cue.text):
        return
    left,right=len(compact_ascii_cjk(cue.text[:a])),len(compact_ascii_cjk(cue.text[:b]))
    position=0
    selected=[]
    for word in words:
        count=len(compact_ascii_cjk(str(word.get('word',word.get('text','')))))
        finish=position+count
        ws,we=float(word['start']),float(word['end'])
        if not math.isfinite(ws) or not math.isfinite(we) or ws < cue.start or we > cue.end or we < ws:
            return
        if min(right,finish)>max(left,position) and we>ws:
            selected.append((ws,we))
        position=finish
    if selected:
        row['estimated_time']={'start':min(t[0] for t in selected),'end':max(t[1] for t in selected)}
        row['time_basis']=f'{side}_native_word_alignment'


def _checked_text_binding(root, binding):
    from .courseware_material import safe_path
    if not isinstance(binding,dict) or not re.fullmatch(r'[0-9a-f]{64}',str(binding.get('sha256',''))):
        raise ValueError('invalid binding')
    path=safe_path(root,str(binding.get('path','')))
    if path.suffix.lower() not in {'.json','.md','.txt','.srt','.vtt'} or path.stat().st_size>64*1024*1024:
        raise ValueError('unbounded text binding')
    if sha256_file(path)!=binding['sha256']:
        raise ValueError('source or evidence hash drift')
    return path


def _review_only_result(root,pack_path,decisions_path,*,min_confidence):
    """Strict closure layered on the existing diff pack; no text promotion."""
    pack,decisions=read_json(pack_path),read_json(decisions_path)
    errors=[]
    resolved=[]
    try:
        if not .75 <= min_confidence <= 1:
            raise ValueError('strict review confidence threshold cannot be lowered')
        bindings=pack['source_bindings']
        paths={role:_checked_text_binding(root,row) for role,row in bindings.items()}
        if not {'consensus','primary','secondary','canonical'} <= paths.keys():
            raise ValueError('missing source bindings')
        rebuilt=build_asr_diff_adjudication(root,consensus_json=paths['consensus'],
            cluster_token_gap=pack.get('cluster_token_gap',6),write=False)
        if rebuilt['source_bindings'] != bindings:
            raise ValueError('current source selection does not match review bindings')
        if rebuilt['source_media_binding'] != pack.get('source_media_binding'):
            raise ValueError('source audio hash drift')
        if rebuilt['differences']!=pack['differences']:
            raise ValueError('diff pack does not match source comparison')
        comparison=read_json(paths['consensus'])
        if len(rebuilt['review_window_ids']) != comparison['review_required_count']:
            raise ValueError('aggregate window coverage does not match comparison')
        if 'dual_workflow' in paths and len(rebuilt['review_window_ids']) != read_json(paths['dual_workflow'])['consensus']['review_required_count']:
            raise ValueError('review does not cover original dual workflow')
        evidence={name:_checked_text_binding(root,row) for name,row in decisions['evidence_bindings'].items()}
        expected={row['diff_id']:row for row in pack['differences']}
        if len(expected)!=len(pack['differences']):
            raise ValueError('duplicate pack IDs')
        canonical=parse_transcript(paths['canonical'])
        primary=parse_transcript(paths['primary'])
        rows=decisions['rows']
        if not isinstance(rows,list) or any(not isinstance(row,dict) for row in rows):
            raise ValueError('decision rows must be objects')
        seen=set()
        for row in rows:
            identity=row.get('diff_id')
            if identity not in expected or identity in seen:
                errors.append({'diff_id':identity,'reason':'unknown_or_duplicate_diff'})
                continue
            seen.add(identity)
            diff=expected[identity]
            if diff.get('time_basis')=='aggregate_bounds_unresolved':
                errors.append({'diff_id':identity,'reason':'fine_timing_not_located'})
                continue
            confidence=row.get('confidence')
            refs=row.get('evidence_refs')
            if (isinstance(confidence,bool) or not isinstance(confidence,(int,float))
                    or not math.isfinite(confidence) or not min_confidence<=confidence<=1
                    or not str(row.get('reason','')).strip()
                    or not isinstance(refs,list) or not refs or any(ref not in evidence for ref in refs)):
                errors.append({'diff_id':identity,'reason':'confidence_reason_or_bound_evidence_required'})
                continue
            choice=str(row.get('choice',''))
            chosen=_resolve_choice(diff,choice)
            if choice=='use_canonical':
                chosen=str(row.get('resolved_text','')).strip()
                # A third value must already have passed the canonical semantic
                # correction path; a review decision cannot invent a new number.
                from .transcript_semantic_correction import _accepted_correction_decisions
                support=[r for r in _accepted_correction_decisions(root)
                         if str(r.get('original_text',''))==diff['primary_text']
                         and str(r.get('corrected_text',''))==chosen]
                if not chosen or not support:
                    errors.append({'diff_id':identity,'reason':'canonical_resolution_not_in_semantic_ledger'})
                    continue
            if chosen is None:
                errors.append({'diff_id':identity,'reason':'invalid_choice'})
                continue
            if compact_ascii_cjk(diff['primary_text']+diff['secondary_text']):
                index=diff.get('primary_segment_index')
                cue=primary[index] if index is not None and 0<=index<len(primary) else None
                context=''.join(c.text for c in canonical if cue and min(c.end,cue.end)>max(c.start,cue.start))
                if chosen and compact_ascii_cjk(chosen) not in compact_ascii_cjk(context):
                    errors.append({'diff_id':identity,'reason':'chosen_text_not_in_current_canonical_context'})
                    continue
                if not chosen and diff['primary_text']:
                    errors.append({'diff_id':identity,'reason':'unproved_deletion'})
                    continue
            resolved.append({'diff_id':identity,'choice':choice,'resolved_text':chosen,'confidence':confidence,
                'reason':row['reason'],'evidence_refs':refs,'reviewer':decisions.get('reviewer','agent'),
                'canonical_modified':False})
        errors.extend({'diff_id':i,'reason':'missing_decision'} for i in sorted(expected.keys()-seen))
    except (OSError,ValueError,TypeError,KeyError,AttributeError) as exc:
        errors.append({'reason':'invalid_or_stale_binding','detail':str(exc)})
    count=len(pack.get('differences') or [])
    complete=not errors and len(resolved)==count
    return {'schema':RESULT_SCHEMA+'.review','status':'completed' if complete else 'review-required','ok':complete,
        'review_only':True,'difference_count':count,'resolved_count':len(resolved),
        'min_confidence':min_confidence,
        'remaining_count':max(count-len(resolved),int(bool(errors))),
        'pack_binding':{'path':str(pack_path),'sha256':sha256_file(pack_path)},
        'decisions_binding':{'path':str(decisions_path),'sha256':sha256_file(decisions_path)},
        'resolved':resolved,'errors':errors,'source_bindings':pack.get('source_bindings',{}),
        'operator_boundary':{'raw_hypotheses_preserved':True,'canonical_modified':False,'human_review_claimed':False,
            'review_scope':'retained_transcript_difference_adjudication_not_independent_fact_verification'},'updated_at':now_iso()}


def asr_diff_review_status(bundle_dir: str | Path) -> dict[str,Any]:
    """Revalidate the full receipt, including decisions and current source bytes."""
    root=Path(bundle_dir).resolve()
    receipt_path=root/'asr-consensus-adjudication-review.json'
    if not receipt_path.is_file():
        return {'ok':False,'status':'missing'}
    try:
        receipt=read_json(receipt_path)
        if receipt.get('status') not in {'completed','review-required'} or receipt.get('review_only') is not True:
            return {'ok':False,'status':'review-required','errors':receipt.get('errors',[])}
        pack=_checked_text_binding(root,receipt['pack_binding'])
        decisions=_checked_text_binding(root,receipt['decisions_binding'])
        result=_review_only_result(root,pack,decisions,min_confidence=receipt.get('min_confidence',.75))
        if any(e.get('reason')=='invalid_or_stale_binding' for e in result.get('errors',[])):
            result['status']='stale_or_invalid'
        return result
    except (OSError,ValueError,TypeError,KeyError,AttributeError) as exc:
        return {'ok':False,'status':'stale_or_invalid','detail':str(exc)}


def _positioned_differences(window: dict[str, Any], primary_text: str, secondary_text: str) -> list[dict[str, Any]]:
    # Keep word/number token boundaries, but do not let omitted punctuation join
    # distant lexical edits into a single replacement.
    def comparison_tokens(text):
        return [row for row in _tokens(text) if re.search(r'\w',row['text']) or
                any(c in row['text'] for c in '+-%％￥¥$€£<>≤≥±≈−–')]
    primary_tokens = comparison_tokens(primary_text)
    secondary_tokens = comparison_tokens(secondary_text)
    matcher = difflib.SequenceMatcher(
        a=[row["normalised"] for row in primary_tokens],
        b=[row["normalised"] for row in secondary_tokens],
        autojunk=False,
    )
    rows: list[dict[str, Any]] = []
    primary_index = window.get("primary_index")
    for ordinal, (operation, a0, a1, b0, b1) in enumerate(matcher.get_opcodes(), start=1):
        if operation == "equal":
            continue
        start_char, end_char = _char_span(primary_tokens, a0, a1, len(primary_text))
        secondary_start, secondary_end = _char_span(secondary_tokens, b0, b1, len(secondary_text))
        diff_id = f"{window.get('window_id')}-diff-{ordinal:03d}"
        primary_piece = primary_text[start_char:end_char]
        secondary_piece = secondary_text[secondary_start:secondary_end]
        if not primary_piece and not secondary_piece:
            continue
        if (_fact_normalise(primary_piece) == _fact_normalise(secondary_piece)
                and not critical_differences(primary_piece, secondary_piece)
                and not any(c in primary_piece + secondary_piece for c in "+-%％￥¥$€£<>≤≥±≈−–")):
            continue
        projection = (locate_primary_span(window["primary_spans"], start_char, end_char)
                      if "primary_spans" in window else
                      (primary_index, start_char, end_char) if primary_index is not None else None)
        rows.append(
            {
                "diff_id": diff_id,
                "window_id": str(window.get("window_id") or ""),
                "operation": operation,
                "patchable": projection is not None,
                "group_char_start": start_char,
                "group_char_end": end_char,
                "primary_segment_index": projection[0] if projection else None,
                "secondary_segment_index": window.get("secondary_index"),
                "primary_token_start": a0,
                "primary_token_end": a1,
                "secondary_token_start": b0,
                "secondary_token_end": b1,
                "primary_char_start": projection[1] if projection else start_char,
                "primary_char_end": projection[2] if projection else end_char,
                "secondary_char_start": secondary_start,
                "secondary_char_end": secondary_end,
                "primary_text": primary_piece,
                "secondary_text": secondary_piece,
                "left_context": primary_text[max(0, start_char - 40):start_char],
                "right_context": primary_text[end_char:end_char + 40],
                "estimated_time": _estimated_diff_time(window, a0, a1, len(primary_tokens)),
            }
        )
    return rows


def _cluster_differences(window: dict[str, Any], differences: list[dict[str, Any]], *, max_gap: int, clip_dir: Path | None = None) -> list[dict[str, Any]]:
    if not differences:
        return []
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    last_end = -1
    for row in differences:
        start = int(row.get("primary_token_start") or 0)
        if current and start - last_end > max_gap:
            groups.append(current)
            current = []
        current.append(row)
        last_end = int(row.get("primary_token_end") or start)
    if current:
        groups.append(current)

    clusters: list[dict[str, Any]] = []
    for ordinal, group in enumerate(groups, start=1):
        cluster_id = f"{window.get('window_id')}-cluster-{ordinal:03d}"
        flip = int(hashlib.sha256(cluster_id.encode("utf-8")).hexdigest()[-1], 16) % 2 == 1
        primary = " … ".join(str(row.get("primary_text") or "∅") for row in group)
        secondary = " … ".join(str(row.get("secondary_text") or "∅") for row in group)
        candidate_a, candidate_b = (secondary, primary) if flip else (primary, secondary)
        for item in group:
            item["cluster_id"] = cluster_id
            item["candidate_a_text"] = str(item.get("secondary_text") if flip else item.get("primary_text") or "")
            item["candidate_b_text"] = str(item.get("primary_text") if flip else item.get("secondary_text") or "")
        start = min(float((row.get("estimated_time") or {}).get("start") or window.get("start") or 0) for row in group)
        end = max(float((row.get("estimated_time") or {}).get("end") or window.get("end") or start) for row in group)
        audio = dict(window.get('audio_review_window') or {})
        audio_start,audio_end=max(0,round(start-2,3)),round(end+2,3)
        audio.update(start=audio_start,end=audio_end,duration_seconds=round(audio_end-audio_start,3),
                     covers_entire_group=True,timing_scope='positioned_difference',execute_required=True)
        if clip_dir is not None:
            audio['clip_path']=str(clip_dir/f'{cluster_id}.wav')
        clusters.append(
            {
                "cluster_id": cluster_id,
                "window_id": str(window.get("window_id") or ""),
                "diff_ids": [str(row.get("diff_id") or "") for row in group],
                "time_range": f"{format_timestamp(start)} - {format_timestamp(end)}",
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
                "context_before": str(group[0].get("left_context") or ""),
                "context_after": str(group[-1].get("right_context") or ""),
                "audio_review_window": audio,
                "instruction": "只根据音频和证据选择 A、B 或 keep_primary；不得改写争议范围外文本。",
            }
        )
    return clusters


def _decision_rows(decisions: dict[str, Any]) -> list[dict[str, Any]]:
    rows = decisions.get("rows") or []
    return [row for row in rows if isinstance(row, dict)]


def _resolve_choice(diff: dict[str, Any], choice: str) -> str | None:
    value = choice.strip().upper()
    if value in {"KEEP_PRIMARY", "PRIMARY", "NO_CHANGE"}:
        return str(diff.get("primary_text") or "")
    if value == "A":
        return str(diff.get("candidate_a_text") or "")
    if value == "B":
        return str(diff.get("candidate_b_text") or "")
    return None


def _fact_normalise(text: str) -> str:
    return re.sub(r"[^\w]+", "", str(text or ""), flags=re.UNICODE).casefold()


def _tokens(text: str) -> list[dict[str, Any]]:
    return [
        {"text": match.group(0), "normalised": match.group(0).casefold(), "start": match.start(), "end": match.end()}
        for match in _TOKEN_RE.finditer(text)
    ]


def _char_span(tokens: list[dict[str, Any]], start: int, end: int, text_length: int) -> tuple[int, int]:
    if start < end and start < len(tokens):
        return int(tokens[start]["start"]), int(tokens[min(end - 1, len(tokens) - 1)]["end"])
    position = int(tokens[start]["start"]) if start < len(tokens) else text_length
    return position, position


def _estimated_diff_time(window: dict[str, Any], start: int, end: int, token_count: int) -> dict[str, float]:
    segment_start = float(window.get("primary_start") if window.get("primary_start") is not None else window.get("start") or 0)
    segment_end = float(window.get("primary_end") if window.get("primary_end") is not None else window.get("end") or segment_start)
    duration = max(0.001, segment_end - segment_start)
    count = max(1, token_count)
    return {
        "start": round(segment_start + duration * start / count, 3),
        "end": round(segment_start + duration * max(start + 1, end) / count, 3),
    }


def _bundle_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(f"artifact not found: {path}")
    return path.resolve()


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _render_pack_markdown(pack: dict[str, Any]) -> str:
    lines = [
        "# ASR Difference Adjudication Pack",
        "",
        f"- Status: `{pack.get('status')}`",
        f"- Positioned differences: `{pack.get('difference_count')}`",
        f"- Clusters: `{pack.get('cluster_count')}`",
        "",
        "## Anonymous Clusters",
        "",
    ]
    for row in pack.get("clusters") or []:
        lines.extend(
            [
                f"### {row.get('cluster_id')} · {row.get('time_range')}",
                "",
                f"- A: {row.get('candidate_a')}",
                f"- B: {row.get('candidate_b')}",
                f"- Context: {row.get('context_before')} **[DIFF]** {row.get('context_after')}",
                f"- Audio: `{(row.get('audio_review_window') or {}).get('clip_path', '')}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_result_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# ASR Difference Adjudication Result",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Applied patches: `{len(result.get('accepted_patches') or [])}`",
        f"- Adjudicated without change: `{len(result.get('adjudicated_no_change') or [])}`",
        f"- Rejected: `{len(result.get('rejected') or [])}`",
        "",
    ]
    for row in result.get("rejected") or []:
        lines.append(f"- `{row.get('diff_id')}`: {row.get('reason')}")
    return "\n".join(lines).rstrip() + "\n"
