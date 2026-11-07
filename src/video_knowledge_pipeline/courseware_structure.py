"""Thin, bounded native Docling table-to-evidence projection.

Consumes TableItem/TableCell JSON (half-open offsets and explicit header flags).
No OCR, HTML parser, table inference, model, or alternative document state machine.
Source: Docling html_backend at 6e031d9b3827dfa9be281808c54b485df7de9d21.
The native payload remains in the existing immutable page receipt. This checks
only the admitted table/related-text subset, not the entire Docling schema.
"""
from __future__ import annotations

import math
import re

MAX_CELLS = 2000
MAX_GRID = 100000
DERIVED_FIELDS = {'structural_evidence', 'native_structure'}


def review_context(context):
    """Keep association roles; raw located cells already carry spans and bbox.

    Full provenance stays in the hash-bound fact pack. Do not repeat observation
    hashes, executor labels and prose limitations for every object in an 8KB
    review task. The table footnote limitation remains explicit in both places.
    """
    if context.get('scope') != 'declared_native_table_associations_not_verified_semantics':
        return context
    return {key: context[key] for key in (
        'row_header_indexes', 'column_header_indexes', 'footnote_applicability', 'qualification_refs', 'qualification_roles',
        'frame_region') if key in context}


FRAME_BINDING_SCHEMA = 'video_knowledge_pipeline.docling_frame_binding.v1'


def _frame_region(page, cell):
    """Coordinate-only projection of an explicit native-image binding.

    Docling's image backend uses decoded image dimensions and TOPLEFT crop
    coordinates. Do not guess PDF point scales, rotation, padding or crop transforms.
    """
    binding, box = page.get('docling_frame_binding'), cell.get('bbox')
    if binding is None or box is None:
        return None
    return project_frame_region(binding, box)


def project_frame_region(binding, box):
    """Shared coordinate-only adapter for explicitly bound native pixels."""
    _bbox(box)
    identity = binding['frame_visual_identity']
    width, height = identity['width'], identity['height']
    left, top, right, bottom = (box[k] for k in ('l', 't', 'r', 'b'))
    if box['coord_origin'] == 'BOTTOMLEFT':
        top, bottom = height - top, height - bottom
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        _fail('frame_region_outside_or_empty')
    return {'frame': binding['frame'], 'frame_sha256': binding['frame_sha256'],
            'image_size': [width, height], 'bbox_xyxy': [left, top, right, bottom],
            'coordinate_space': 'decoded_frame_pixels_top_left',
            'pixel_observation': 'not_assessed'}


def validate_native_frame_binding(root, page, candidate):
    if 'docling_frame_binding' not in page:
        return
    binding = page['docling_frame_binding']
    required = {'schema', 'page_no', 'coordinate_space', 'frame', 'frame_sha256', 'frame_visual_identity'}
    if (not isinstance(binding, dict) or set(binding) != required
            or binding['schema'] != FRAME_BINDING_SCHEMA
            or binding['coordinate_space'] != 'decoded_frame_pixels'
            or type(binding['page_no']) is not int
            or not 1 <= binding['page_no'] <= 1000000
            or type(page.get('docling_page_no')) is not int
            or binding['page_no'] != page.get('docling_page_no')
            or binding['frame'] != candidate.get('frame')
            or binding['frame_sha256'] != candidate.get('frame_sha256')
            or binding['frame_sha256'] != page.get('frame_sha256')
            or not page.get('docling_document')):
        _fail('native_frame_binding_invalid')
    validate_frame_binding_identity(root, binding, candidate)
    document = page['docling_document']
    for item in [*document['tables'], *document['texts']]:
        for prov in item.get('prov', []):
            if prov['page_no'] != binding['page_no']:
                _fail('frame_page_mismatch')
            _frame_region(page, prov)
    for table in document['tables']:
        for cell in table['data']['table_cells']:
            _frame_region(page, cell)


def validate_frame_binding_identity(root, binding, candidate):
    from pathlib import Path
    from .courseware_material import safe_path
    frame = safe_path(root, binding['frame'])
    if (frame.relative_to(Path(root).absolute()).as_posix() != binding['frame']
            or frame.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}):
        _fail('bundle_relative_static_frame_required')
    from .courseware_dedupe import validate_frame_visual_identity
    identity = validate_frame_visual_identity(root, candidate, binding['frame_visual_identity'])
    if identity['render_metadata'].get('exif_orientation') not in (None, 1):
        _fail('frame_orientation_requires_explicit_transform')


def audit_frame_regions(root, refs):
    """Reuse native receipt and frame identity gates, not a pixel-seeing claim."""
    from .evidence_locator import JsonEvidenceSnapshot, MAX_BYTES
    from .courseware_material import safe_path
    cache, verified, total = {}, 0, 0
    for ref in refs:
        region = (ref.get('source_context') or {}).get('frame_region')
        locators = ref.get('source_locators') or []
        if region is None:
            if any(loc.get('selector') in (['docling_frame_binding'], ['ocr_geometry', 'frame_binding']) for loc in locators):
                _fail('frame_region_projection_missing')
            continue
        if not locators:
            _fail('frame_region_source_required')
        primary = locators[0]
        path = primary['artifact_path']
        selector = primary['selector']
        is_ocr = isinstance(selector, list) and len(selector) == 3 and selector[:2] == ['ocr_geometry', 'observations']
        binding_selector = ['ocr_geometry', 'frame_binding'] if is_ocr else ['docling_frame_binding']
        cache_key = (path, is_ocr)
        if cache_key not in cache:
            snapshot = JsonEvidenceSnapshot.read(root, path, max_bytes=MAX_BYTES - total)
            page = snapshot.data
            binding = snapshot.selected(binding_selector)
            if not isinstance(binding, dict):
                _fail('frame_region_binding_missing')
            frame = safe_path(root, binding['frame'])
            total += snapshot.size_bytes + frame.stat().st_size
            if len(cache) >= 256 or total > MAX_BYTES:
                _fail('frame_region_audit_budget_split_scope')
            if is_ocr:
                from .courseware_ocr_geometry import validate_geometry
                validate_geometry(root, page, binding)
            else:
                validate_native_structure(page)
                validate_native_frame_binding(root, page, binding)
            cache[cache_key] = snapshot
        snapshot = cache[cache_key]
        if snapshot.locator(binding_selector) not in locators:
            _fail('frame_region_binding_locator_required')
        if not is_ocr and (not isinstance(selector, list) or len(selector) != 6
                or selector[:2] != ['docling_document', 'tables']
                or selector[3:5] != ['data', 'table_cells']):
            _fail('frame_region_target_cell_required')
        cell = snapshot.selected(selector)
        if is_ocr:
            from .courseware_ocr_geometry import observation_region
            expected_region = observation_region(snapshot.data['ocr_geometry'], cell)
        else:
            expected_region = _frame_region(snapshot.data, cell)
        if not isinstance(cell, dict) or expected_region != region:
            _fail('frame_region_projection_drift')
        verified += 1
    return {'verified_region_count': verified, 'source_page_count': len(cache),
            'status': 'bound_navigation_only' if verified else 'not_assessed',
            'pixel_observation': 'not_assessed', 'recognition_accuracy': 'not_assessed'}


def _fail(reason):
    raise ValueError('courseware_structure:' + reason)


def _integer(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        _fail('invalid_integer_or_budget')
    return value


def _bbox(box):
    if box is None:
        return
    if not isinstance(box, dict) or box.get('coord_origin') not in {'TOPLEFT', 'BOTTOMLEFT'}:
        _fail('bbox_coordinate_origin_required')
    values = [box.get(k) for k in ('l', 't', 'r', 'b')]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        _fail('bbox_nonfinite')
    left, top, right, bottom = values
    if left > right or (top > bottom if box['coord_origin'] == 'TOPLEFT' else top < bottom):
        _fail('bbox_orientation')


def _provenance(item, page):
    prov = item.get('prov', [])
    if not isinstance(prov, list) or len(prov) > 32:
        _fail('provenance_budget')
    for entry in prov:
        if not isinstance(entry, dict):
            _fail('invalid_provenance')
        page_no = _integer(entry.get('page_no'), 1, 1000000)
        if type(page.get('docling_page_no')) is not int or page_no != page['docling_page_no']:
            _fail('cross_page_or_unbound_page_number')
        _bbox(entry.get('bbox'))


def _ref(value, kind, rows):
    if not isinstance(value, str) or not re.fullmatch(r'#/' + kind + r'/(0|[1-9][0-9]*)', value):
        _fail('local_native_ref_required')
    index = int(value.rsplit('/', 1)[1])
    if index >= len(rows) or not isinstance(rows[index], dict) or rows[index].get('self_ref') != value:
        _fail('native_ref_missing_or_mismatched')
    return index


def _related(table, texts):
    related, seen = [], set()
    for role in ('footnotes', 'captions', 'references'):
        refs = table.get(role, [])
        if not isinstance(refs, list) or len(refs) > 64:
            _fail('related_text_budget')
        for ref in refs:
            if not isinstance(ref, dict) or set(ref) != {'$ref'}:
                _fail('invalid_related_ref')
            index = _ref(ref['$ref'], 'texts', texts)
            if index in seen:
                _fail('duplicate_related_ref')
            seen.add(index)
            if role in {'footnotes', 'captions'} and texts[index].get('label') != role[:-1]:
                _fail('related_text_label_mismatch')
            related.append((role, index))
    return related


def validate_native_structure(page):
    """Fail closed inside existing page importer, before any formal replacement."""
    blocks = page['blocks']
    if any(DERIVED_FIELDS.intersection(block) for block in blocks):
        _fail('derived_projection_not_importable')
    document = page.get('docling_document')
    if document is None:
        if any('docling_ref' in block for block in blocks):
            _fail('native_document_required')
        return
    if not isinstance(document, dict) or document.get('schema_name') != 'DoclingDocument':
        _fail('native_document_schema')
    if not isinstance(document.get('version'), str) or not document['version'].strip() or len(document['version']) > 64:
        _fail('native_document_version_required')
    tables, texts = document.get('tables'), document.get('texts')
    if not isinstance(tables, list) or not 1 <= len(tables) <= 32 or not isinstance(texts, list) or len(texts) > 256:
        _fail('native_document_budget')
    bound = []
    for block in blocks:
        if 'docling_ref' in block:
            if block['kind'] != 'table':
                _fail('only_table_block_supported')
            bound.append(_ref(block['docling_ref'], 'tables', tables))
    if len(bound) != len(set(bound)) or set(bound) != set(range(len(tables))):
        _fail('unbound_or_duplicate_table')
    for index, text in enumerate(texts):
        _ref(f'#/texts/{index}', 'texts', texts)
        if not isinstance(text.get('text'), str) or len(text['text']) > 16000:
            _fail('related_text_content')
        _provenance(text, page)
    total, related_texts = 0, set()
    for table in tables:
        if table.get('label') != 'table' or not isinstance(table.get('data'), dict):
            _fail('native_table_data_required')
        _provenance(table, page)
        data = table['data']
        rows = _integer(data.get('num_rows'), 1, 10000)
        cols = _integer(data.get('num_cols'), 1, 256)
        if rows * cols > MAX_GRID:
            _fail('grid_budget')
        cells = data.get('table_cells')
        if not isinstance(cells, list) or not cells:
            _fail('cells_required')
        total += len(cells)
        if total > MAX_CELLS:
            _fail('cell_budget_split_scope')
        occupied = set()
        for cell in cells:
            if not isinstance(cell, dict) or not isinstance(cell.get('text'), str) or len(cell['text']) > 16000:
                _fail('cell_text')
            r0 = _integer(cell.get('start_row_offset_idx'), 0, rows - 1)
            r1 = _integer(cell.get('end_row_offset_idx'), r0 + 1, rows)
            c0 = _integer(cell.get('start_col_offset_idx'), 0, cols - 1)
            c1 = _integer(cell.get('end_col_offset_idx'), c0 + 1, cols)
            if _integer(cell.get('row_span'), 1, rows) != r1 - r0 or _integer(cell.get('col_span'), 1, cols) != c1 - c0:
                _fail('native_span_mismatch')
            for key in ('row_header', 'column_header', 'row_section'):
                if type(cell.get(key)) is not bool:
                    _fail('explicit_native_header_flags_required')
            for row in range(r0, r1):
                for col in range(c0, c1):
                    coordinate = (row, col)
                    if coordinate in occupied:
                        _fail('overlapping_cells')
                    occupied.add(coordinate)
            _bbox(cell.get('bbox'))
            if cell.get('ref'):
                # RichTableCell child groups are not plain text; no silent loss.
                _fail('rich_cell_requires_separate_adapter')
        related_texts.update(index for _, index in _related(table, texts))
    if related_texts != set(range(len(texts))):
        _fail('unbound_document_text_scope_required')
    if any(document.get(key) for key in ('pictures', 'key_value_items', 'form_items')):
        _fail('non_table_document_scope_requires_separate_adapter')


def _overlap(a, b, axis):
    return (a[f'start_{axis}_offset_idx'] < b[f'end_{axis}_offset_idx']
            and b[f'start_{axis}_offset_idx'] < a[f'end_{axis}_offset_idx'])


def native_qualification_catalog(document, table_index):
    """Enumerate native context candidates; never infer their row applicability.

    Reuse Docling's explicit related-text links and TableCell.row_section flags.
    All section headings remain visible, including contrary/later sections.
    The existing review budget refuses oversized context instead of truncating.
    """
    table = document['tables'][table_index]
    catalog = {document['texts'][i]['self_ref']: dict(role=role,
               selector=['docling_document', 'texts', i])
               for role, i in _related(table, document['texts'])}
    for i, cell in enumerate(table['data']['table_cells']):
        if cell.get('row_section') is True:
            catalog[f'#/tables/{table_index}/data/table_cells/{i}'] = dict(role='row_section',
                selector=['docling_document', 'tables', table_index, 'data', 'table_cells', i])
    return catalog


def project_native_table(page, block, number, snapshot):
    """Project declared associations, never inferred semantic applicability."""
    if 'docling_ref' not in block:
        return {}
    doc = page['docling_document']
    index = _ref(block['docling_ref'], 'tables', doc['tables'])
    table = doc['tables'][index]
    prefix = ['docling_document', 'tables', index]
    cells = table['data']['table_cells']
    related = _related(table, doc['texts'])
    qualifiers = native_qualification_catalog(doc, index)
    qualifier_roles = {key: value['role'] for key, value in qualifiers.items()}
    section_indexes = [i for i, cell in enumerate(cells) if cell['row_section']]
    notes = [{'role': role, 'item': doc['texts'][i]} for role, i in related]
    notes_locators = [snapshot.locator(['docling_document', 'texts', i]) for _, i in related]
    base_id = f"courseware:{page['page_id']}:{number}"
    refs = []
    for n, cell in enumerate(cells):
        if not cell['text'].strip():
            continue
        roles = [key for key in ('column_header', 'row_header', 'row_section') if cell[key]]
        role = '+'.join(roles) if roles else 'value'
        row_headers = [i for i, h in enumerate(cells) if h['row_header'] and _overlap(h, cell, 'row')
                       and h['end_col_offset_idx'] <= cell['start_col_offset_idx']]
        col_headers = [i for i, h in enumerate(cells) if h['column_header'] and _overlap(h, cell, 'col')
                       and h['end_row_offset_idx'] <= cell['start_row_offset_idx']]
        limitations = ['Native recognition is not independently verified.',
                      'Table-related footnotes do not establish cell-specific applicability.']
        if not row_headers:
            limitations.append('row_header_not_declared')
        if not col_headers:
            limitations.append('column_header_not_declared')
        if any(c['row_section'] for c in cells):
            limitations.append('row_section_applicability_not_assessed')
        identifier = f'{base_id}:cell:{n}'
        context = {'context_id': identifier, 'page_id': page['page_id'], 'native_table_ref': block['docling_ref'],
                   'cell_role': role, 'source_observation_id': page['frame_sha256'],
                   'executor': page['executor'], 'model_actual': page['model_actual'],
                   'frame_sha256': page['frame_sha256'], 'cell_index': n,
                   'row_range': [cell['start_row_offset_idx'], cell['end_row_offset_idx']],
                   'column_range': [cell['start_col_offset_idx'], cell['end_col_offset_idx']],
                   'row_header_indexes': row_headers, 'column_header_indexes': col_headers,
                   'footnote_applicability': 'not_assessed',
                   'qualification_refs': sorted(qualifiers),
                   'bbox': cell.get('bbox'),
                   'scope': 'declared_native_table_associations_not_verified_semantics', 'limitations': limitations}
        if any(role != 'footnotes' for role in qualifier_roles.values()):
            context['qualification_roles'] = qualifier_roles
        region = _frame_region(page, cell)
        if region is not None:
            context['frame_region'] = region
        snippet = '\n'.join([
            'Row: ' + ' | '.join(cells[i]['text'] for i in row_headers),
            'Column: ' + ' | '.join(cells[i]['text'] for i in col_headers),
            f'{role}: ' + cell['text'],
            *[f'row_section [scope unknown]: {cells[i]["text"]}' for i in section_indexes if i != n],
            *[f"{role} [scope unknown]: {doc['texts'][i]['text']}" for role, i in related],
        ])
        refs.append({'evidence_id': identifier, 'source_kind': 'visual', 'source': 'video_displayed_courseware',
                     'snippet': snippet, 'fact_status': 'candidate_evidence', 'source_context': context,
                     'source_locators': [snapshot.locator(prefix + ['data', 'table_cells', i])
                                         for i in dict.fromkeys([n, *row_headers, *col_headers, *section_indexes])] + notes_locators
                                        + ([snapshot.locator(['docling_frame_binding'])] if region else [])})
    # table_cells is authoritative here; do not reinterpret the optional, redundant
    # Docling grid rendering as a second source or count it as corroboration.
    native = {'native_ref': block['docling_ref'], 'native_document_version': doc['version'],
              'qualification_refs': sorted(qualifiers),
              'num_rows': table['data']['num_rows'],
              'num_cols': table['data']['num_cols'], 'table_cells': cells, 'related_texts': notes,
              'prov': table.get('prov', []), 'footnote_applicability': 'not_assessed',
              'evidence_level': 'candidate_evidence', 'recognition_accuracy': 'not_assessed'}
    if any(role != 'footnotes' for role in qualifier_roles.values()):
        native['qualification_roles'] = qualifier_roles
    return {'native_structure': native, 'structural_evidence': refs,
            'source_locators': block['source_locators'] + [snapshot.locator(prefix)]
                               + [snapshot.locator(v['selector']) for v in qualifiers.values()]}


def native_block_review_projection(block):
    """Retain raw OCR plus exact context candidates, without copying a whole table.

    The importer/reporter recomputes this catalog from the immutable native source;
    this compact projection is navigation, not authority or semantic applicability.
    """
    native = block['native_structure']
    qualifiers = native.get('qualification_refs', [])
    context = {'context_id': block['evidence_id'], 'native_table_ref': native['native_ref'],
               'qualification_refs': qualifiers, 'footnote_applicability': 'not_assessed',
               'scope': 'declared_native_table_associations_not_verified_semantics'}
    if 'qualification_roles' in native:
        context['qualification_roles'] = native['qualification_roles']
    locators = block.get('source_locators', [])
    selected = locators[:1] + [loc for loc in locators[1:]
        if loc['selector'][:1] == ['docling_document']
        and '#/' + '/'.join(str(token) for token in loc['selector'][1:]) in qualifiers]
    return {'source_context': context, 'source_locators': selected}
