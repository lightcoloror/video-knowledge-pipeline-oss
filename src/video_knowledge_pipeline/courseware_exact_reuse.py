"""Thin, fail-closed exact-frame OCR cache over existing page import receipts.

No perceptual threshold, ROI, guessed slide identity, or second OCR writer.
Identical file bytes remain supported. V2 additionally permits different image
containers only when every decoded RGBA pixel and rendering metadata match.
Approximate plans cannot be supplied to this function.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .courseware_material import (_block_content, _check_frame, _page_id,
                                 _validate_page, import_courseware_page, read_scan, safe_path)
from .file_hash import sha256_file
from .storage import read_json


def _read_page(path):
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('exact reuse source result byte budget exceeded')
    return read_json(path)


def validate_exact_frame_reuse(root, state, value):
    proof = value['exact_frame_reuse']
    method = proof.get('method') if isinstance(proof, dict) else None
    if method not in {'identical_frame_file_sha256', 'identical_decoded_frame_identity_v2'}:
        raise ValueError('invalid exact frame reuse proof')
    source_id = _page_id(proof.get('source_page_id'))
    if source_id == value['page_id'] or proof.get('inference_performed_for_target') is not False:
        raise ValueError('invalid exact reuse source/target')
    path = safe_path(root, f'courseware/pages/{source_id}.json')
    if sha256_file(path) != proof.get('source_result_sha256'):
        raise ValueError('exact reuse source result hash drift')
    source = _read_page(path)
    if source.get('page_id') != source_id:
        raise ValueError('exact reuse source filename identity mismatch')
    if source.get('exact_frame_reuse') is not None or source.get('executor') != 'ebook/RapidOCR':
        raise ValueError('exact reuse requires a direct OCR source, not manual or chained reuse')
    _validate_page(root, state, source)
    if method == 'identical_frame_file_sha256':
        if (source['frame_sha256'] != value['frame_sha256']
                or proof.get('frame_sha256') != value['frame_sha256']):
            raise ValueError('exact reuse frame identity mismatch')
    else:
        from .courseware_dedupe import validate_frame_visual_identity
        source_candidate = next(row for row in state['candidates'] if row['page_id'] == source_id)
        target_candidate = next(row for row in state['candidates'] if row['page_id'] == value['page_id'])
        source_identity = validate_frame_visual_identity(root, source_candidate, source.get('frame_visual_identity'))
        target_identity = validate_frame_visual_identity(root, target_candidate, value.get('frame_visual_identity'))
        digest = source_identity['identity_sha256']
        if (target_identity != source_identity or proof.get('visual_identity_sha256') != digest
                or proof.get('source_frame_sha256') != source['frame_sha256']
                or proof.get('target_frame_sha256') != value['frame_sha256']):
            raise ValueError('exact decoded frame identity mismatch')
    if any(value.get(k) != source.get(k) for k in ('blocks', 'status', 'gaps', 'executor', 'model_actual')):
        raise ValueError('exact reuse content/backend mismatch')
    if source['status'] != 'extracted' or source['gaps']:
        raise ValueError('exact reuse source is incomplete')
    from .visual_structure import _ebook_markdown_quality
    if _ebook_markdown_quality('\n\n'.join(_block_content(b) for b in source['blocks']))['quality'] != 'usable':
        raise ValueError('exact reuse source OCR quality is unusable')


def reuse_exact_courseware_pages(bundle_dir, *, child_indexes, defer_within_batch=False,
                                 visual_source_indexes=None, allow_decoded_identity=False):
    root = Path(bundle_dir).absolute()
    if (not isinstance(child_indexes, (list, tuple)) or not 1 <= len(child_indexes) <= 8
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in child_indexes)
            or len(set(child_indexes)) != len(child_indexes)):
        raise ValueError('exact reuse requires 1..8 unique positive child indexes')
    state = read_scan(root)
    if len(state['candidates']) > 10000:
        raise ValueError('courseware metadata bound exceeded')
    timeline = read_json(safe_path(root, 'courseware/ocr-bundle/timeline.json'))
    by_id = {p['page_id']: p for p in state['candidates']}
    by_index = {r['index']: r for r in timeline}
    if len(by_id) != len(state['candidates']) or len(by_index) != len(timeline):
        raise ValueError('duplicate courseware identity')
    if visual_source_indexes is None:
        visual_source_indexes = []
    if not isinstance(allow_decoded_identity, bool):
        raise ValueError('allow_decoded_identity must be boolean')
    if visual_source_indexes and not allow_decoded_identity:
        raise ValueError('visual identity sources require explicit decoded-identity policy')
    if (not isinstance(visual_source_indexes, (list, tuple)) or len(visual_source_indexes) > 256
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in visual_source_indexes)
            or len(set(visual_source_indexes)) != len(visual_source_indexes)):
        raise ValueError('visual identity source indexes must be bounded unique positive integers')
    if any(n not in by_index for n in visual_source_indexes):
        raise ValueError('visual identity source index missing')
    selected = []
    for n in child_indexes:
        row = by_index.get(n, {})
        page = by_id.get(row.get('courseware_page_id'))
        if (not page or row.get('courseware_frame_sha256') != page.get('frame_sha256')
                or row.get('frame_paths') != [str(root/page['frame'])]):
            raise ValueError('courseware OCR child identity drift')
        _check_frame(root, page)
        from .courseware_dedupe import frame_visual_identity
        selected.append((n, page, frame_visual_identity(root, page)))
    visual_sources = []
    for n in visual_source_indexes:
        page = by_id.get(by_index[n].get('courseware_page_id'))
        if not page:
            raise ValueError('visual identity source page missing')
        path = safe_path(root, f"courseware/pages/{page['page_id']}.json", exists=False)
        if not path.exists():
            continue
        source = _read_page(path)
        if (source.get('page_id') != page['page_id'] or source.get('executor') != 'ebook/RapidOCR'
                or source.get('exact_frame_reuse') or source.get('status') != 'extracted' or source.get('gaps')):
            continue
        _validate_page(root, state, source)
        if not source.get('frame_visual_identity'):
            continue
        visual_sources.append((source, path))
    reused, existing, pending, receipts, deferred = [], [], [], [], []
    pending_hashes = set()
    for n, target, target_identity in selected:
        target_path = safe_path(root, f"courseware/pages/{target['page_id']}.json", exists=False)
        if target_path.exists():
            prior = _read_page(target_path)
            if prior.get('page_id') != target['page_id']:
                raise ValueError('exact reuse target filename identity mismatch')
            _validate_page(root, read_scan(root), prior)
            existing.append(n)
            continue  # Never replace existing manual/model results.
        source = None
        source_path = None
        method = None
        for candidate in state['candidates']:
            if candidate['page_id'] == target['page_id'] or candidate.get('frame_sha256') != target['frame_sha256']:
                continue
            path = safe_path(root, f"courseware/pages/{candidate['page_id']}.json", exists=False)
            if not path.exists():
                continue
            value = _read_page(path)
            if value.get('page_id') != candidate['page_id']:
                raise ValueError('exact reuse source filename identity mismatch')
            if (value.get('executor') != 'ebook/RapidOCR' or value.get('exact_frame_reuse')
                    or value.get('status') != 'extracted' or value.get('gaps')):
                continue
            _validate_page(root, state, value)
            from .visual_structure import _ebook_markdown_quality
            if _ebook_markdown_quality('\n\n'.join(_block_content(b) for b in value['blocks']))['quality'] != 'usable':
                continue
            source, source_path, method = value, path, 'identical_frame_file_sha256'
            break
        if source is None and allow_decoded_identity:
            for candidate_source, path in visual_sources:
                if (candidate_source['page_id'] == target['page_id']
                        or candidate_source['frame_visual_identity']['identity_sha256']
                        != target_identity['identity_sha256']):
                    continue
                source, source_path, method = candidate_source, path, 'identical_decoded_frame_identity_v2'
                break
        if source is None:
            keys = {('file', target['frame_sha256'])}
            if allow_decoded_identity:
                keys.add(('visual', target_identity['identity_sha256']))
            if defer_within_batch and pending_hashes.intersection(keys):
                deferred.append(n)
            else:
                pending.append(n)
                pending_hashes.update(keys)
            continue
        value = {k: deepcopy(source[k]) for k in ('schema', 'scan_identity', 'executor', 'model_actual', 'status', 'blocks', 'gaps')}
        value.update(page_id=target['page_id'], time=target['time'],
                     frame_time=target.get('frame_time', target['time']), frame_sha256=target['frame_sha256'],
                     frame_visual_identity=target_identity,
                     exact_frame_reuse={'method':method,
                         'source_page_id':source['page_id'], 'source_result_sha256':sha256_file(source_path),
                         **({'frame_sha256': target['frame_sha256']} if method == 'identical_frame_file_sha256' else {
                             'visual_identity_sha256': target_identity['identity_sha256'],
                             'source_frame_sha256': source['frame_sha256'],
                             'target_frame_sha256': target['frame_sha256'],
                         }),
                         'inference_performed_for_target':False})
        receipts.append(import_courseware_page(root, value))  # existing lock/raw/validation/atomic write
        reused.append(n)
    return {'ok': True, 'status':'exact_reuse_checked', 'reused_indexes':reused,
            'already_imported_indexes':existing, 'ocr_indexes':pending, 'import_receipts':receipts,
            'deferred_indexes':deferred,
            'visual_source_indexes':list(visual_source_indexes),
            'decoded_identity_reuse_enabled':allow_decoded_identity,
            'approximate_reuse':False, 'original_media_modified':False, 'independent_review_complete':False}
