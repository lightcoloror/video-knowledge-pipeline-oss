"""Public CLI/MCP recovery and export contracts over existing offline fixtures."""
import json
from pathlib import Path

from test_creator_report_oral_delivery import oral_bundle, snapshot
from test_oral_ocr_candidates_recovery import _case
from test_qwen_failed_window_recovery import _fixture, _child
from video_knowledge_pipeline.cli import main, _mcp_callables
from video_knowledge_pipeline.storage import write_json


def test_cli_saved_recovery_accepts_string_paths_without_model_execution(tmp_path, capsys):
    media, partial, activity = _fixture(tmp_path)
    original = partial.read_bytes()
    output = tmp_path / 'recovery'
    assert main(['plan-qwen-window-recovery', str(partial), str(output),
                 '--speech-activity-json', str(activity), '--verbose']) == 0
    plan = json.loads(capsys.readouterr().out)
    child = _child(plan, tmp_path)
    assert main(['compose-qwen-window-recovery', str(output / 'recovery-plan.json'),
                 str(child), '--verbose']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'completed_composed_reference'
    assert result['input_sha256'] == plan['media']['sha256']
    assert not result['single_run_execution_attested']
    assert not result['automatic_canonical_promotion_allowed']
    assert partial.read_bytes() == original
    assert media.exists()


def test_local_mcp_dispatch_exposes_plan_and_cached_compose(tmp_path):
    _, partial, _ = _fixture(tmp_path)
    tools = _mcp_callables()
    plan = tools['plan_qwen_window_recovery'](str(partial), output_dir=tmp_path / 'recovery')
    child = _child(plan, tmp_path)
    result = tools['compose_qwen_window_recovery'](plan, recovered_json=str(child), write=False)
    assert result['status'] == 'completed_composed_reference'
    assert not Path(plan['composed_reference_json']).exists()


def test_cli_external_interactive_exports_keep_source_bundle_read_only(tmp_path, capsys):
    root = oral_bundle(tmp_path)
    before = snapshot(root)
    output = tmp_path / 'delivery'
    assert main(['export-creator-report', str(root), '--view', 'practical',
        '--formats', 'html', 'srt', 'csv', '--interactive-html', '--output-dir', str(output), '--verbose']) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt['candidate_only'] and not receipt['publication_allowed']
    kinds = {row['format'] for row in receipt['artifacts']}
    assert {'html', 'srt', 'csv', 'subtitle-source-map', 'projection'} <= kinds
    page = Path(next(row['path'] for row in receipt['artifacts'] if row['format'] == 'html')).read_text(encoding='utf-8')
    assert '<video' in page and 'data-seek' in page and 'media-src blob:' in page
    assert snapshot(root) == before
    assert main(['creator-report-status', str(root), '--view', 'practical',
                 '--output-dir', str(output), '--verbose']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'fresh'


def test_cli_oral_ocr_profile_is_plan_only_without_source_writes(tmp_path, capsys):
    root, _, _, result = _case(tmp_path)
    saved = tmp_path / 'ocr-results.json'
    write_json(saved, {'items': [result]})
    before = snapshot(root)
    assert main(['run-screen-text-recovery', str(root), '--profile', 'talking_head',
        '--input-json', str(saved), '--max-regions', '1', '--max-attempts', '1',
        '--no-write', '--verbose']) == 0
    response = json.loads(capsys.readouterr().out)
    plan = response['oral_recovery_queue']
    assert plan['profile'] == 'talking_head' and plan['plan_only']
    assert plan['max_regions'] == plan['max_attempts'] == 1
    assert plan['operator_boundary']['model_calls_made'] == 0
    assert snapshot(root) == before
