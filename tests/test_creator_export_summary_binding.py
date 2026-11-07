"""A historical summary source cannot shadow or authorize a current export."""
import pytest

from video_knowledge_pipeline import creator_report
from video_knowledge_pipeline.storage import write_json
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.workflow_completion import _creator_readable_export_status


@pytest.mark.parametrize('current_binding', ['valid', 'missing', 'wrong_hash'])
def test_readable_export_uses_current_summary_among_historical_sources(tmp_path, monkeypatch, current_binding):
    root = tmp_path / 'bundle'
    root.mkdir()
    write_json(root / 'transcript.json', {'segments': [
        {'start': 0, 'end': 1, 'text': 'synthetic text'},
    ]})
    installed = root / 'validated-summary.md'
    installed.write_text('Current synthetic summary', encoding='utf-8')
    old = root / 'old-summary.md'
    old.write_text('Historical synthetic summary', encoding='utf-8')
    sources = [{'role': 'smart_summary', 'path': str(old), 'sha256': sha256_file(old)}]
    if current_binding != 'missing':
        sources.append({'role': 'validated_smart_summary', 'path': str(installed),
                        'sha256': sha256_file(installed) if current_binding == 'valid' else '0' * 64})
    export = root / 'exports/creator-report'
    export.mkdir(parents=True)
    projection = export / 'projection.json'
    write_json(projection, {'transcript': [{'start': 0.0, 'end': 1.0,
        'text': 'synthetic text', 'speaker': ''}], 'summary': installed.read_text(), 'sources': sources})
    artifacts = [{'format': 'projection', 'path': str(projection.relative_to(root)), 'sha256': sha256_file(projection)}]
    for kind in ('html', 'md', 'pdf'):
        target = export / ('synthetic.' + kind)
        target.write_bytes(b'synthetic artifact for binding check only')
        artifacts.append({'format': kind, 'path': str(target.relative_to(root)), 'sha256': sha256_file(target)})
    write_json(export / 'latest.json', {'artifacts': artifacts})
    # Isolate source-binding validation; renderer/freshness have their own tests.
    monkeypatch.setattr(creator_report, 'creator_report_status',
                        lambda *a, **kw: {'status': 'fresh', 'export_status': 'exported_candidate'})
    result = _creator_readable_export_status(root, {}, installed)
    assert result['passed'] is (current_binding == 'valid')
    if current_binding != 'valid':
        assert result['reason'] == 'summary_source_mismatch'
