"""Synthetic RapidOCR final-output geometry through the existing worker."""
import ast
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from test_courseware_incremental_ocr import Engine, picture
from test_courseware_dedupe import image_bundle
from test_courseware_structure import projected
from video_knowledge_pipeline.courseware_incremental_ocr import run_incremental_courseware_ocr
from video_knowledge_pipeline.evidence_locator import audit_fact_pack_locators
from video_knowledge_pipeline.storage import read_json
from types import SimpleNamespace


class GeometryEngine(Engine):
    def __call__(self, path):
        output = super().__call__(path)
        with Image.open(path) as image:
            output.img = np.array(image.convert('RGB'))
        output.boxes = np.array([[[i * 20 + .001, 1], [i * 20 + 19, 0],
                                 [i * 20 + 19.5, 19], [i * 20 + 1, 20]]
                                for i in range(len(output.txts))])
        return output


def ebook_normalizer():
    source_root = os.environ.get('VKP_EBOOK_SOURCE')
    if not source_root:
        pytest.skip('explicit existing ebook source needed; no installation')
    source = Path(source_root) / 'ocr_providers.py'
    names = {'normalize_rapidocr_blocks', 'extract_rapidocr_items', 'rows_from_parallel_values',
             'optional_sequence', 'normalize_rapidocr_item', 'normalize_score', 'normalize_ocr_box'}
    nodes = [n for n in ast.parse(source.read_text(encoding='utf-8-sig')).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
    return namespace['normalize_rapidocr_blocks']


def run_geometry(root, engine=None, normalize=None, **kwargs):
    return run_incremental_courseware_ocr(root, indexes=[1], engine=engine or GeometryEngine(),
        runtime_identity={'model_actual': 'synthetic', 'geometry_contract': 'rapidocr_final_original_pixels.v1'},
        output_factory=SimpleNamespace, normalize_blocks=normalize or ebook_normalizer(), **kwargs)


def test_actual_worker_keeps_original_quad_and_final_citation(tmp_path):
    from video_knowledge_pipeline.citation_digest import courseware_records
    from video_knowledge_pipeline.summary_claim_review import _scope_sources
    from video_knowledge_pipeline.summary_citation_audit import build_citation_index
    root = image_bundle(tmp_path, [picture()])
    before = (root / 'courseware/frames/0.png').read_bytes()
    run_geometry(root)
    page = read_json(root / 'courseware/pages/page-0.json')
    observed = page['ocr_geometry']['observations'][0]
    assert observed['quad'][0][0] == .001  # not the rounded bbox's zero
    assert observed['text'] == page['blocks'][0]['text']
    material, section = projected(root)
    ref = next(r for r in section['evidence_refs'] if r['evidence_id'] == 'courseware:page-0:0')
    assert ref['source_context']['frame_region']['bbox_xyxy'] == [.001, 0, 19.5, 20]
    assert ref['source_context']['frame_region']['pixel_observation'] == 'not_assessed'
    facts = {'sections': [section]}
    assert audit_fact_pack_locators(root, facts)['frame_regions']['verified_region_count'] == 2
    digest = courseware_records(material, [])
    assert digest[0]['source_context']['frame_region'] == ref['source_context']['frame_region']
    sources = _scope_sources(root, build_citation_index(facts), facts, [ref['evidence_id']])
    assert any(o.get('quad') == observed['quad'] for o in sources[0]['original_objects'])
    again = run_geometry(root, resume=True)
    assert again['detector_calls'] == 0
    assert (root / 'courseware/frames/0.png').read_bytes() == before


@pytest.mark.parametrize('kind', ['outside', 'crossed', 'nan', 'missing_boxes', 'missing_image', 'count', 'index', 'text', 'normalized_bbox', 'normalized_score'])
def test_bad_final_geometry_keeps_formal_page_and_timeline_unchanged(tmp_path, kind):
    class Bad(GeometryEngine):
        def __call__(self, path):
            output = super().__call__(path)
            if kind == 'outside': output.boxes[0, 0, 0] = -1
            if kind == 'crossed': output.boxes[0] = output.boxes[0, [0, 2, 1, 3]]
            if kind == 'nan': output.boxes[0, 0, 0] = float('nan')
            if kind == 'missing_boxes': output.boxes = None
            if kind == 'missing_image': output.img = None
            if kind == 'count': output.boxes = output.boxes[:1]
            return output
    normalizer = ebook_normalizer()
    def wrong(raw):
        rows = normalizer(raw)
        if kind == 'index': rows[1]['index'] = rows[0]['index']
        if kind == 'text': rows[0]['text'] = 'a different observation'
        if kind == 'normalized_bbox': rows[0]['bbox'] = [0, 0, 40, 20]
        if kind == 'normalized_score': rows[0]['score'] = .01
        return rows
    root = image_bundle(tmp_path, [picture()])
    before = (root / 'timeline.json').read_bytes()
    with pytest.raises(ValueError):
        run_geometry(root, Bad(), wrong)
    assert not (root / 'courseware/pages/page-0.json').exists()
    assert (root / 'timeline.json').read_bytes() == before


def test_original_quad_drift_or_false_seen_is_not_a_verified_citation(tmp_path):
    root = image_bundle(tmp_path, [picture()])
    run_geometry(root)
    _, section = projected(root)
    ref = next(r for r in section['evidence_refs'] if r['evidence_id'] == 'courseware:page-0:0')
    ref['source_context']['frame_region']['pixel_observation'] = 'confirmed'
    with pytest.raises(ValueError):
        audit_fact_pack_locators(root, {'sections': [section]})


def test_ocr_worker_to_bounded_review_import_and_restore(tmp_path):
    from test_summary_claim_review import submit
    from video_knowledge_pipeline.canonical_json import canonical_json_bytes
    from video_knowledge_pipeline.storage import write_json
    from video_knowledge_pipeline.summary_claim_review import prepare_summary_claim_review, summary_claim_review_report
    from video_knowledge_pipeline.transcript_semantic_transaction import apply_transcript_semantic_restore
    text = '产品甲2026版首次投保等待期90天。'
    class TextEngine(GeometryEngine):
        def recognize_txt(self, images):
            result = super().recognize_txt(images)
            result.txts = tuple(text for _ in images)
            return result
    root = image_bundle(tmp_path, [picture(b=255)])
    before = (root / 'timeline.json').read_bytes()
    run_geometry(root, TextEngine())
    _, section = projected(root)
    ref = next(r for r in section['evidence_refs'] if r['evidence_id'] == 'courseware:page-0:0')
    ref['evidence_id'] = 'r'
    plan = {'overview': {'text': text, 'evidence_ids': ['r']}}
    facts = {'bundle_dir': str(root), 'sections': [{'section_id': 's',
        'time_range': '00:00:00 - 00:00:01', 'evidence_refs': [ref]}]}
    write_json(root / 'exports/smart-summary-reader-plan.json', plan)
    write_json(root / 'exports/smart-summary-chapter-fact-pack.json', facts)
    first = prepare_summary_claim_review(root)
    assert len(canonical_json_bytes(first)) <= 8192
    assert first['sources'][0]['source_context']['frame_region']['pixel_observation'] == 'not_assessed'
    assert submit(root, first)['ok']
    review = prepare_summary_claim_review(root)
    assert len(canonical_json_bytes(review)) <= 8192
    final = submit(root, review, run='separate-ocr-text-review')
    assert final['ok'] and summary_claim_review_report(root, plan, facts)['passed']
    tx = final['transaction']
    assert apply_transcript_semantic_restore(root, tx['restore_plan'], execute=True,
        confirm_transaction_id=tx['transaction_id'])['status'] == 'restored'
    assert not summary_claim_review_report(root, plan, facts)['passed']
    assert (root / 'timeline.json').read_bytes() == before


def test_fixed_rapidocr_original_coordinate_mapping_runs_offline():
    source_root = os.environ.get('VKP_RAPIDOCR_SOURCE')
    if not source_root:
        pytest.skip('explicit fixed RapidOCR source not supplied')
    source = Path(source_root) / 'python/rapidocr/utils/process_img.py'
    node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                if isinstance(n, ast.FunctionDef) and n.name == 'map_boxes_to_original')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    namespace = {'np': np}
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
    boxes = np.array([[[3., 5.], [7., 5.], [7., 9.], [3., 9.]]])
    operations = {'preprocess': {'ratio_h': 2., 'ratio_w': 3.}, 'padding_1': {'top': 1., 'left': 2.}}
    result = namespace['map_boxes_to_original'](boxes, operations, 100, 100)
    assert result.tolist() == [[[3., 8.], [15., 8.], [15., 16.], [3., 16.]]]
    # Upstream clips final coordinates. VKP preserves final output only and
    # does not pretend to recover out-of-frame detector geometry before clipping.
    clipped = namespace['map_boxes_to_original'](np.array([[[-1., -1.], [200., 0.], [200., 200.], [0., 200.]]]), {}, 100, 100)
    assert clipped.tolist() == [[[0., 0.], [100., 0.], [100., 100.], [0., 100.]]]
