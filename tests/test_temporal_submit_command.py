"""Execute the generated offline import command, not reconstructed defaults."""
import json
from pathlib import Path

import pytest

from test_temporal_tag_delta import _bundle_with_sequences
from video_knowledge_pipeline.temporal_tag_delta import run_temporal_tag_delta
from video_knowledge_pipeline.storage import read_json, write_json
from video_knowledge_pipeline.powershell import quote_powershell_literal, run_powershell_command


@pytest.mark.parametrize('frame_count,limit,min_frames', [(5, 1, 5), (8, 2, 7)])
def test_generated_submit_command_preserves_pack_and_quality_parameters(tmp_path, frame_count, limit, min_frames):
    parent = tmp_path / "literal $scope 'quoted'"
    parent.mkdir()
    root = _bundle_with_sequences(parent, count=3, frames=8)
    prepared = run_temporal_tag_delta(root, indexes='1,3', frame_count=frame_count,
                                     limit=limit, min_frames=min_frames)
    command = prepared['submit_command']
    assert f'--frame-count {frame_count}' in command
    assert f'--limit {limit}' in command
    assert f'--min-frames {min_frames}' in command
    assert quote_powershell_literal(root) in command
    assert '--execute-tagger' not in command
    template = Path(prepared['input_template_path'])
    payload = read_json(template)
    payload['result_metadata'] = {'executor': 'synthetic-import',
        'model_actual': 'synthetic-tags', 'model_revision': 'fixture'}
    for sequence in payload['sequences']:
        for frame in sequence['frames']:
            frame.update(tags=['课件'], confidence=0.9)
    write_json(template, payload)
    completed = run_powershell_command(command, cwd=Path(__file__).resolve().parents[1], timeout_seconds=120)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    result = json.loads(completed.stdout)
    assert result['status'] == 'completed', result
    # Large CLI results intentionally expose a persisted-report envelope.
    report_path = result.get('report_json_path') or result['paths']['report_json_path']
    result = read_json(Path(report_path))
    assert result['selected_indexes'] == prepared['selected_indexes']
    assert result['sequence_pack']['task_pack_sha256'] == prepared['sequence_pack']['task_pack_sha256']
    assert result['processed_frame_count'] == frame_count * limit
