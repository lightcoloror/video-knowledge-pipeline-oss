"""Mixed synthetic producer contracts through existing review/export/UI.

No OCR/VLM/tagger model is run. Imported results are explicitly synthetic, and
the authored quote test is not an accuracy benchmark or a human acceptance.
"""
from pathlib import Path

from PIL import Image
import pytest

from test_summary_evidence_end_to_end import (
    seed, close_authored_transcript, exercise_chain, quoted_answer, no_network,
)
from test_smart_summary_reader_plan import _valid_plan
from test_courseware_structure import native_page
from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.evidence_locator import JsonEvidenceSnapshot
from video_knowledge_pipeline.courseware_material import (
    store_scan, import_courseware_page, build_courseware_material, import_courseware_review,
)
from video_knowledge_pipeline.codex_vision_session import prepare_codex_vision_session, import_codex_vision_results
from video_knowledge_pipeline.temporal_tag_delta import run_temporal_tag_delta
from video_knowledge_pipeline.video_moment_index import build_video_moment_index
from video_knowledge_pipeline.smart_summary_section_workflow import _section_citations
from video_knowledge_pipeline import transcript_semantic_correction as semantic
from video_knowledge_pipeline import summary_claim_review as review
from video_knowledge_pipeline.smart_summary_input_pack import build_smart_summary_input_pack
from video_knowledge_pipeline.visual_evidence import evidence_level_counts


def mixed_seed(root, *, reviewed=False):
    plan = _valid_plan()
    plan['overview']['text'] = '本次合成场景中，' + plan['overview']['text']
    plan = seed(root, plan)
    plan['review_items'] = [{'text':'原始视频视觉未执行，合成导入的课件与画面信息待复核。',
        'time_range':'', 'evidence_ids':[], 'missing_evidence':'只有合成导入响应，无真实 OCR/VLM 执行或独立课件人审。'}]
    timeline = read_json(root / 'timeline.json')
    assets = root / 'assets'
    assets.mkdir()
    frame = assets / 'slide.png'
    Image.new('RGB', (32, 24), 'white').save(frame)
    timeline[0].update(visual_route='semantic_frame', midpoint=1.0,
        visual_text=plan['overview']['text'], assets=[{'path': 'assets/slide.png', 'copied': 'true'}])
    frames = []
    for i in range(5):
        path = assets / f'temporal-{i}.png'
        Image.new('RGB', (32, 24), 'white').save(path)
        frames.append({'path': path.relative_to(root).as_posix(),
            'timestamp': timeline[1]['start'] + i * (timeline[1]['end'] - timeline[1]['start']) / 5,
            'frame_id': f'synthetic-frame-{i}'})
    timeline[1].update(visual_route='temporal_sequence', temporal_frame_paths=frames)
    write_json(root / 'timeline.json', timeline)
    vision = prepare_codex_vision_session(root, indexes=[1])
    payload = read_json(Path(vision['result_template_path']))
    payload['model_actual'] = 'synthetic-vision-response-not-model-run'
    for row in payload['items']:
        row.update(evidence_level='confirmed', observed_frame_ids=row['expected_frame_ids'],
            visual_understanding={'objects': ['课件'], 'actions': [],
                'interface_state': '合成课件页面', 'instructor_focus': '课件',
                'spatial_relations': [], 'non_text_information': ['浅色背景'],
                'confidence': 0.9, 'keep_image_reason': '原始布局供复核',
                'evidence_frame_paths': row['frame_paths']})
    answer = root / 'synthetic-vision-result.json'
    write_json(answer, payload)
    imported = import_codex_vision_results(root, input_json=answer)
    assert imported['status'] == 'imported', imported
    prepared = run_temporal_tag_delta(root, indexes='2', frame_count=5)
    payload = read_json(Path(prepared['input_template_path']))
    payload['result_metadata'] = {'executor': 'synthetic-import',
        'model_actual': 'synthetic-tags', 'model_revision': 'fixture-v1'}
    for sequence in payload['sequences']:
        for item in sequence['frames']:
            item.update(tags=['课件', '讲师'], confidence=0.9)
    answer = root / 'synthetic-tag-result.json'
    write_json(answer, payload)
    tagged = run_temporal_tag_delta(**{**prepared['submit_args'], 'input_json': answer})
    assert tagged['status'] == 'completed', tagged
    assert tagged['processed_frame_count'] == 5
    assert tagged['items'][0]['evidence_level'] == 'candidate_weak'
    store_scan(root, {'schema': 'video_knowledge_pipeline.courseware_scan.v1',
        'run_id': 'synthetic-mixed', 'scope': {'start': 0, 'end': 240, 'excluded_intervals': []},
        'source': {'sha256': 'a' * 64}, 'windows': [{'start': 0, 'end': 240}],
        'completed_windows': [0], 'candidates': [{'page_id': 'page-0', 'time': 0., 'end': 240.,
            'frame': 'assets/slide.png', 'frame_sha256': sha256_file(frame)}], 'failure': None})
    page = native_page(root)
    page['blocks'][0]['text'] = plan['overview']['text']
    page['docling_document']['texts'][0]['text'] = '仅指本次合成场景，不用于现实事实判断。'
    import_courseware_page(root, page)
    material = build_courseware_material(root, write=False)
    assert material['extraction_complete'] and not material['independent_review_complete']
    if reviewed:
        # A declared test-double for the external human step, never an actual
        # courseware/media accuracy claim. Use the real import and identity gate.
        recorded = import_courseware_review(root, {
            'material_revision': material['revision'],
            'reviewer': 'synthetic-independent-review-fixture-not-human',
            'method': 'Authored one-page fixture oracle; no playback or real human review performed.',
            'scope_reviewed': True, 'page_ids': ['page-0'], 'gaps': [],
        })
        assert recorded['ok'], recorded
        material = build_courseware_material(root, write=False)
        assert material['complete']
    workflow = read_json(root / 'exports/smart-summary-section-workflow.json')
    snap = JsonEvidenceSnapshot.read(root, 'timeline.json')
    for i, section in enumerate(workflow['sections']):
        section['citations'][0]['source_locators_by_kind'] = {'asr': [snap.locator([i, 'text'])]}
    workflow['sections'][0]['evidence']['courseware_material'] = material
    workflow['sections'][0]['citations'].extend(_section_citations(build_video_moment_index(root), [], 0, 240))
    workflow['courseware_material'] = material
    write_json(root / 'exports/smart-summary-section-workflow.json', workflow)
    revisions = read_json(root / 'exports/smart-summary-section-llm-revisions.json')
    revisions['courseware_material'] = material
    for row in revisions['rows']:
        row['courseware_revision'] = material['revision']
    write_json(root / 'exports/smart-summary-section-llm-revisions.json', revisions)
    plan['overview']['evidence_ids'].append('courseware:page-0:0')
    points = read_json(root / 'quality-benchmark-key-points.json')
    points['source_bindings'][0]['sha256'] = sha256_file(root / 'timeline.json')
    write_json(root / 'quality-benchmark-key-points.json', points)
    semantic.build_transcript_semantic_correction_pack(root, write=True)
    close_authored_transcript(root)
    return plan


def mixed_answer(task):
    result = quoted_answer(task, allow_partition_without_quote=True)
    if task['phase'] == 'annotate' and task['output_text'].startswith('本次合成场景中，'):
        for row in [*result['claims'], *result['assertions']]:
            row['slots']['conditions'] = ['本次合成场景']
        sources = {s['evidence_id']: s for s in task['sources']}
        result['qualification_assessments'] = [
            {'assertion_id': row['assertion_id'], 'qualifier_id': qualifier, 'status': 'applies',
             'reason': 'The authored synthetic footnote limits this quote to the named synthetic scenario.',
             'condition_indexes': [0]}
            for row in result['assertions']
            for qualifier in sources[row['evidence_id']].get('source_context', {}).get('qualification_refs', [])]
        if 'qualification_assessments' not in task['output_schema']['properties']:
            assert not result['qualification_assessments']
            result.pop('qualification_assessments')
    return result


def inspect_mixed(root, phase):
    material = build_courseware_material(root, write=False)
    assert material['complete'] == (root / 'courseware/independent-review.json').exists()
    facts = read_json(root / review.FACTS)
    refs = [r for s in facts['sections'] for r in s['evidence_refs']]
    weak = [r for r in refs if r.get('observation_kind') == 'continuous_tag_weak_evidence']
    assert weak and all(r['evidence_level'] == 'candidate_weak' for r in weak)
    assert any(r.get('observation_kind') == 'visual_understanding' and r['evidence_level'] == 'confirmed' for r in refs)
    assert any(':cell:' in r['evidence_id'] for r in refs)
    block = next(r for r in refs if r['evidence_id'] == 'courseware:page-0:0')
    assert block['source_context']['qualification_refs']
    assert any(r.get('observation_kind') == 'visual_text' for r in refs)
    if phase == 'exported':
        from video_knowledge_pipeline.task_console import export_task_console, _render_task_console_html
        from video_knowledge_pipeline.video_workbench import export_video_workbench, _render_workbench_html
        levels = evidence_level_counts(read_json(root / 'timeline.json'))
        assert levels['continuous_tag']['candidate_weak'] == 1
        for export, render in ((export_task_console, _render_task_console_html),
                (export_video_workbench, lambda v: _render_workbench_html(v, []))):
            view = export(root, write=False)
            assert view['summary_claim_review']['passed'], view['summary_claim_review']
            html = render(view)
            assert '摘要逐条证据审核' in html and '不代表现实事实已证实' in html
        pack = build_smart_summary_input_pack(root, write=False)
        assert pack['evidence_trace']['continuous_tag_items'][0]['evidence_level'] == 'candidate_weak'


@pytest.mark.parametrize('reviewed', [False, True], ids=['review-required', 'synthetic-reviewed'])
def test_mixed_producers_review_export_and_ui(tmp_path, monkeypatch, reviewed):
    from video_knowledge_pipeline import task_console
    # Host settings/health are external to this evidence fixture. No health
    # probe, private configuration or service connection is part of acceptance.
    # Keep actual evidence status, exporters, renderers and quality gates intact.
    monkeypatch.setattr(task_console, 'config_status', lambda: {'service_urls': {}})
    monkeypatch.setattr(task_console, 'model_api_settings_status', lambda: {
        'schema': 'video_knowledge_pipeline.model_api_settings.v1', 'providers': [],
        'fixture': 'offline-host-settings-not-a-health-check'})
    root = tmp_path / 'mixed'
    plan = mixed_seed(root, reviewed=reviewed)
    exercise_chain(root, plan, monkeypatch, answer=mixed_answer, inspect_stage=inspect_mixed,
        expected_install_block=None if reviewed else 'courseware_material_complete_and_consumed')
