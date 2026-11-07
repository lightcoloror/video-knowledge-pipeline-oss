"""Synthetic original-frame navigation; never evidence of having viewed pixels."""
import pytest
from PIL import Image

from test_courseware_material import fixture_bundle
from test_courseware_structure import native_page, projected
from video_knowledge_pipeline.courseware_material import read_scan, store_scan, import_courseware_page
from video_knowledge_pipeline.courseware_dedupe import frame_visual_identity
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.evidence_locator import audit_fact_pack_locators


def bound_page(tmp_path):
    root = fixture_bundle(tmp_path)
    return root, bind_native_page(root)


def bind_native_page(root):
    frame = root / 'courseware/frames/0.png'
    Image.new('RGB', (200, 100), 'white').save(frame)
    state = read_scan(root)
    candidate = state['candidates'][0]
    candidate['frame_sha256'] = sha256_file(frame)
    store_scan(root, state)
    page = native_page(root)
    page['docling_page_no'] = 1
    page['docling_frame_binding'] = {
        'schema': 'video_knowledge_pipeline.docling_frame_binding.v1',
        'page_no': 1, 'coordinate_space': 'decoded_frame_pixels',
        'frame': candidate['frame'], 'frame_sha256': candidate['frame_sha256'],
        'frame_visual_identity': frame_visual_identity(root, candidate),
    }
    page['docling_document']['tables'][0]['data']['table_cells'][4]['bbox'] = {
        'l': 20, 'r': 80, 't': 90, 'b': 60, 'coord_origin': 'BOTTOMLEFT'}
    return page


def test_frame_region_reaches_digest_and_real_review_source(tmp_path):
    from video_knowledge_pipeline.citation_digest import courseware_records
    from video_knowledge_pipeline.summary_claim_review import _scope_sources
    from video_knowledge_pipeline.summary_citation_audit import build_citation_index
    root, page = bound_page(tmp_path)
    original = (root / 'courseware/frames/0.png').read_bytes()
    import_courseware_page(root, page)
    material, section = projected(root)
    ref = next(r for r in section['evidence_refs'] if r['evidence_id'].endswith(':cell:4'))
    region = ref['source_context']['frame_region']
    assert region['bbox_xyxy'] == [20, 10, 80, 40]
    assert region['coordinate_space'] == 'decoded_frame_pixels_top_left'
    assert region['pixel_observation'] == 'not_assessed'
    assert region['frame'] == 'courseware/frames/0.png'
    pack = {'sections': [section]}
    report = audit_fact_pack_locators(root, pack)
    assert report['frame_regions']['verified_region_count'] == 1
    rows = courseware_records(material, [])
    assert next(r for r in rows if r['evidence_id'] == ref['evidence_id'])['source_context']['frame_region'] == region
    sources = _scope_sources(root, build_citation_index(pack), pack, [ref['evidence_id']])
    selected = next(r for r in sources if r['evidence_id'] == ref['evidence_id'])
    assert selected['source_context']['frame_region'] == region
    assert (root / 'courseware/frames/0.png').read_bytes() == original


@pytest.mark.parametrize('mutation', ['size', 'path', 'page', 'boolean_page', 'coordinates', 'zero', 'orientation', 'space'])
def test_invalid_frame_binding_rejected_before_formal_page_write(tmp_path, mutation):
    root, page = bound_page(tmp_path)
    binding = page['docling_frame_binding']
    box = page['docling_document']['tables'][0]['data']['table_cells'][4]['bbox']
    if mutation == 'size':
        binding['frame_visual_identity']['width'] = 201
    elif mutation == 'path':
        binding['frame'] = '../outside.png'
    elif mutation == 'page':
        binding['page_no'] = 2
    elif mutation == 'boolean_page':
        page['docling_page_no'] = True
    elif mutation == 'coordinates':
        box['r'] = 201
    elif mutation == 'zero':
        box['r'] = box['l']
    elif mutation == 'orientation':
        binding['frame_visual_identity']['render_metadata']['exif_orientation'] = 6
    else:
        binding['coordinate_space'] = 'pdf_points'
    before = (root / 'timeline.json').read_bytes()
    with pytest.raises(ValueError):
        import_courseware_page(root, page)
    assert not (root / 'courseware/pages/page-0.json').exists()
    assert (root / 'timeline.json').read_bytes() == before


def test_even_inside_bundle_absolute_frame_binding_is_rejected(tmp_path):
    root, page = bound_page(tmp_path)
    state = read_scan(root)
    state['candidates'][0]['frame'] = str(root / 'courseware/frames/0.png')
    store_scan(root, state)
    page['docling_frame_binding']['frame'] = state['candidates'][0]['frame']
    with pytest.raises(ValueError, match='bundle_relative'):
        import_courseware_page(root, page)


@pytest.mark.parametrize('mutation', ['frame', 'region', 'binding_removed', 'region_removed', 'seen'])
def test_frame_drift_or_forged_region_blocks_actual_fact_audit(tmp_path, mutation):
    root, page = bound_page(tmp_path)
    import_courseware_page(root, page)
    _, section = projected(root)
    ref = next(r for r in section['evidence_refs'] if r['evidence_id'].endswith(':cell:4'))
    if mutation == 'frame':
        Image.new('RGB', (200, 100), 'black').save(root / 'courseware/frames/0.png')
    elif mutation == 'region':
        ref['source_context']['frame_region']['bbox_xyxy'][0] = 0
    elif mutation == 'binding_removed':
        ref['source_locators'] = [loc for loc in ref['source_locators'] if loc['selector'] != ['docling_frame_binding']]
    elif mutation == 'region_removed':
        del ref['source_context']['frame_region']
    else:
        ref['source_context']['frame_region']['pixel_observation'] = 'confirmed'
    with pytest.raises(ValueError):
        audit_fact_pack_locators(root, {'sections': [section]})


def test_unbound_legacy_bbox_is_not_silently_promoted(tmp_path):
    root, page = bound_page(tmp_path)
    del page['docling_frame_binding']
    import_courseware_page(root, page)
    _, section = projected(root)
    ref = next(r for r in section['evidence_refs'] if r['evidence_id'].endswith(':cell:4'))
    assert not ref['source_context'].get('frame_region')
    assert audit_fact_pack_locators(root, {'sections': [section]})['frame_regions']['verified_region_count'] == 0


@pytest.mark.parametrize('resolution', [None, 'unresolved', 'not_applicable', 'applies'])
def test_bound_region_uses_existing_bounded_review_import_and_restore(tmp_path, monkeypatch, resolution):
    import test_courseware_structure as existing
    monkeypatch.setattr(existing, 'native_page', bind_native_page)
    existing.test_actual_review_prepare_keeps_cell_context_and_stays_bounded(tmp_path, resolution)


def test_fixed_docling_image_backend_crop_smoke(tmp_path):
    """Run upstream's pure Pillow crop method, not the Docling model runtime."""
    import ast
    import os
    from pathlib import Path
    from types import SimpleNamespace
    from video_knowledge_pipeline.courseware_structure import _frame_region
    checkout = os.environ.get('VKP_DOCLING_SOURCE')
    if not checkout:
        pytest.skip('explicit fixed Docling source not supplied')
    source = Path(checkout) / 'docling/backend/image_backend.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_ImagePageBackend')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_page_image')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
    namespace = {'CoordOrigin': SimpleNamespace(TOPLEFT='TOPLEFT')}
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
    root, page = bound_page(tmp_path)
    box = page['docling_document']['tables'][0]['data']['table_cells'][4]
    region = _frame_region(page, box)
    coordinates = tuple(region['bbox_xyxy'])
    with Image.open(root / 'courseware/frames/0.png') as image:
        crop = namespace['get_page_image'](SimpleNamespace(_image=image), cropbox=SimpleNamespace(
            coord_origin='TOPLEFT', as_tuple=lambda: coordinates))
        assert crop.size == (60, 30)
        assert crop.tobytes() == image.crop(coordinates).tobytes()
        crop.close()
