"""Regression for copied native assets; public path containment stays strict."""
import pytest
from PIL import Image
from test_creator_report import bundle
from video_knowledge_pipeline.creator_report import build_creator_report
from video_knowledge_pipeline.creator_report_renderers import bundle_file
from video_knowledge_pipeline.storage import read_json, write_json


def test_native_assets_survive_missing_external_workspace_frames(tmp_path):
    root = bundle(tmp_path)
    rows = read_json(root/'timeline.json')
    for row in rows:
        row['frame_paths']=[str(tmp_path/'missing-old-workspace.jpg')]
        row['assets']=[{'path':'frames/1.jpg'}]
    Image.new('RGB',(100,160),'blue').save(root/'frames/1.jpg')
    write_json(root/'timeline.json',rows)
    report=build_creator_report(root)
    assert len(report['previews'])==len(report['shots'])==2
    assert all(p['source']=='frames/1.jpg' for p in report['previews'].values())
    assert all(p['data_uri'].startswith('data:image/jpeg;base64,') for p in report['previews'].values())


def test_external_assets_do_not_become_a_public_preview(tmp_path):
    root=bundle(tmp_path)
    outside=tmp_path/'private.jpg'
    Image.new('RGB',(100,100),'red').save(outside)
    rows=read_json(root/'timeline.json')
    for row in rows:
        row['frame_paths']=[]
        row['assets']=[{'path':str(outside)}]
    write_json(root/'timeline.json',rows)
    report=build_creator_report(root)
    assert report['previews']=={}
    assert report['preview_issues']
    with pytest.raises(ValueError):
        bundle_file(root,outside)
