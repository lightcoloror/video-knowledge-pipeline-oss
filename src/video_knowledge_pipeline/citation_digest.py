"""Lossless navigation adapter over existing citations, locators and fact refs.

ALCE-inspired identity accounting, not another entailment model or evidence store.
The chapter pack remains the stored projection; navigation != semantic support.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .canonical_json import canonical_json_bytes, canonical_json_sha256 as digest
from .file_hash import sha256_file

SCHEMA = 'video_knowledge_pipeline.citation_digest.v2'
MAX_RECORDS = 20000
MAX_BYTES = 32 * 1024 * 1024
CHAPTERS = 'exports/smart-summary-chapters.json'


def evidence_level(row):
    if row.get('source_type') == 'continuous_tag_weak_evidence' and row.get('evidence_level') in {'confirmed', 'inferred'}:
        raise ValueError('citation_digest:weak_tag_cannot_be_promoted')
    level = row.get('evidence_level')
    if isinstance(level, str) and level.strip():
        return level
    kind = str(row.get('source_type') or '')
    if 'inferred' in kind:
        return 'inferred'
    if kind == 'continuous_tag_weak_evidence':
        return 'candidate_weak'
    if kind == 'review_gap':
        return 'not_seen'
    return 'not_assessed'


def check_record_budget(rows):
    if not isinstance(rows, list) or len(rows) > MAX_RECORDS:
        raise ValueError('citation_digest:record_budget')
    size = 0
    for row in rows:
        size += len(canonical_json_bytes(row))
        if size > MAX_BYTES:
            raise ValueError('citation_digest:byte_budget_split_scope')


def digest_records(rows):
    """Exact full-record dedup: same prefix/time is not the same evidence."""
    check_record_budget(rows)
    seen, identities, result, size = set(), {}, [], 0
    for source in rows:
        if not isinstance(source, dict):
            raise ValueError('citation_digest:invalid_record')
        row = {**source, 'evidence_level': evidence_level(source), 'citation_schema': SCHEMA}
        if not row.get('evidence_id'):
            row['evidence_id'] = 'citation:' + digest(source)
        name = row['evidence_id']
        if not isinstance(name, str) or not name.strip() or len(name) > 512:
            raise ValueError('citation_digest:invalid_evidence_id')
        # Apply CitationIndex's same-ID/same-content rule to navigation records.
        # Reuse in another chapter changes navigation, not source identity.
        source_identity = digest({k: v for k, v in row.items() if k not in {'chapter_index', 'summary_chapter_ref'}})
        if name in identities and identities[name] != source_identity:
            raise ValueError('citation_digest:conflicting_evidence_id')
        identities[name] = source_identity
        encoded = canonical_json_bytes(row)
        size += len(encoded)
        if size > MAX_BYTES:
            raise ValueError('citation_digest:byte_budget_split_scope')
        identity = digest(row)
        if identity not in seen:
            seen.add(identity)
            result.append(row)
    return result


def courseware_records(material, timeline):
    if not material.get('active'):
        return []
    # Reuse the real reducer's native-table/block evidence projection. No second
    # table-to-claim interpretation or independent source identity is invented.
    from .smart_summary_global_reduce import _chapter_fact_section
    refs = _chapter_fact_section({}, {'evidence': {'courseware_material': material}})['evidence_refs']
    pages = {p['page_id']: p for p in material['pages']}
    result = []
    for ref in refs:
        if ref.get('source_kind') == 'evidence_group':
            continue
        context = ref['source_context']
        page = pages[context['page_id']]
        indexes = [item['index'] for item in timeline if isinstance(item, dict)
                   and type(item.get('index')) is int and item['index'] > 0
                   and float(item.get('start', 0)) < page['end']
                   and float(item.get('end', 0)) > page['time']]
        result.append({**ref, 'source_type': 'courseware_table_cell' if 'cell_index' in context else 'courseware_block',
                       'evidence_level': 'unavailable' if ref['fact_status'] == 'review_gap_not_fact' else 'candidate_evidence',
                       'time': ref['time_range'], 'timeline_indexes': sorted(set(indexes)), 'text': ref['snippet'],
                       'evidence_paths': list(dict.fromkeys([page.get('frame', ''),
                            *[loc['artifact_path'] for loc in ref.get('source_locators', [])]]))})
    return digest_records(result)


def _snapshot(pack):
    chapters = pack.get('chapters', [])
    if not isinstance(chapters, list) or len(chapters) > 4096:
        raise ValueError('citation_digest:chapter_budget')
    records = []
    snapshot = []
    for chapter in chapters:
        if not isinstance(chapter, dict):
            raise ValueError('citation_digest:invalid_chapter')
        rows = chapter.get('citation_digest', [])
        normalized = digest_records(rows)
        if normalized != rows:
            raise ValueError('citation_digest:noncanonical_or_duplicate_records')
        records.extend(rows)
        if len(records) > MAX_RECORDS:
            raise ValueError('citation_digest:aggregate_record_budget')
        snapshot.append({key: chapter.get(key) for key in ('index', 'start', 'end', 'citation_digest')})
    if sum(len(canonical_json_bytes(row)) for row in records) > MAX_BYTES:
        raise ValueError('citation_digest:aggregate_byte_budget')
    return snapshot, records


def bind_citation_digest(root, pack):
    root = Path(root)
    snapshot, rows = _snapshot(pack)
    timeline = root / 'timeline.json'
    return {'schema': SCHEMA, 'chapter_records_sha256': digest(snapshot), 'record_count': len(rows),
            'timeline_sha256': sha256_file(timeline) if timeline.is_file() else None,
            'courseware_revision': pack.get('courseware_material', {}).get('revision'),
            'evidence_levels': dict(Counter(evidence_level(r) for r in rows)),
            'semantic_support': 'not_assessed', 'source_completeness': 'not_assessed',
            'scope': 'all_selected_navigation_records_not_all_video_information'}


def audit_citation_digest(bundle_dir):
    """No writes; a recorded navigation snapshot may not silently become legacy."""
    from .courseware_material import safe_path, build_courseware_material
    from .evidence_locator import JsonEvidenceSnapshot, audit_fact_pack_locators
    root = Path(bundle_dir).absolute()
    unknown = {'status': 'not_assessed', 'passed': None, 'evaluated': False,
               'semantic_support': 'not_assessed', 'source_completeness': 'not_assessed'}
    try:
        manifest_path = safe_path(root, 'manifest.json', exists=False)
        manifest = JsonEvidenceSnapshot.read(root, 'manifest.json').data if manifest_path.exists() else {}
        registered = isinstance(manifest, dict) and manifest.get('smart_summary_citation_digest_schema') == SCHEMA
        path = safe_path(root, CHAPTERS, exists=False)
        if not path.exists():
            if registered:
                raise ValueError('registered_citation_inventory_missing')
            return {**unknown, 'detail': 'No chapter citation inventory.'}
        pack = JsonEvidenceSnapshot.read(root, CHAPTERS).data
        if not isinstance(pack, dict):
            raise ValueError('invalid_chapter_pack')
        chapters = pack.get('chapters', [])
        if not isinstance(chapters, list) or any(not isinstance(c, dict) or not isinstance(c.get('citation_digest', []), list) for c in chapters):
            raise ValueError('invalid_chapter_citations')
        binding = pack.get('citation_digest_binding')
        marked = any(r.get('citation_schema') == SCHEMA for c in pack.get('chapters', [])
                     for r in c.get('citation_digest', []) if isinstance(r, dict))
        if binding is None and not marked and not registered and pack.get('citation_digest_schema') != SCHEMA:
            return {**unknown, 'detail': 'Legacy navigation has no verified snapshot.'}
        if not isinstance(binding, dict) or pack.get('citation_digest_schema') != SCHEMA:
            raise ValueError('binding_missing')
        expected = bind_citation_digest(root, pack)
        if binding != expected:
            raise ValueError('navigation_or_timeline_hash_drift')
        material = build_courseware_material(root, write=False)
        if material.get('revision') != binding.get('courseware_revision'):
            raise ValueError('courseware_revision_drift')
        # Chapter-sized locator batches use the existing strict validator;
        # absent locators remain explicitly unverified, not invented identities.
        located, unlocated = 0, 0
        text_hashes = {}
        for chapter in pack.get('chapters', []):
            for row in chapter['citation_digest']:
                if not row.get('source_sha256'):
                    continue
                source = safe_path(root, str(row.get('source_path') or ''))
                if source.suffix.lower() not in {'.json', '.srt', '.vtt', '.txt', '.md'}:
                    raise ValueError('text_source_format_not_allowed')
                relative = source.relative_to(root).as_posix()
                if relative not in text_hashes:
                    if source.stat().st_size > MAX_BYTES:
                        raise ValueError('text_source_byte_budget')
                    text_hashes[relative] = sha256_file(source)
                if text_hashes[relative] != row['source_sha256']:
                    raise ValueError('transcript_source_hash_drift')
            report = audit_fact_pack_locators(root, {'sections': [{'evidence_refs': chapter['citation_digest']}]})
            located += report['verified_count']
            unlocated += sum(not r.get('source_locators') for r in chapter['citation_digest'])
        return {**unknown, 'status': 'verified_navigation', 'passed': True, 'evaluated': True,
                'detail': 'Navigation snapshot and supplied source locators verified; not semantic support.',
                'record_count': binding['record_count'], 'evidence_levels': binding['evidence_levels'],
                'located_object_count': located, 'unlocated_record_count': unlocated,
                'text_source_artifact_count': len(text_hashes),
                'chapter_records_sha256': binding['chapter_records_sha256'], 'artifact': CHAPTERS}
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return {**unknown, 'status': 'blocked', 'passed': False, 'evaluated': True,
                'detail': 'citation_digest:' + str(exc), 'artifact': CHAPTERS}


def preview_note(shown, total):
    return (f'引用预览 {shown}/{total}；完整引用、证据等级和定位见 '
            '[smart-summary-chapters.json](smart-summary-chapters.json)。'
            '预览不代表全部内容已核实。')
