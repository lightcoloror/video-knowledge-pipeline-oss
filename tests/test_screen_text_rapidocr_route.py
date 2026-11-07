import pytest
from PIL import Image
from video_knowledge_pipeline import screen_text_recovery as recovery
from video_knowledge_pipeline.storage import write_json,read_json
from video_knowledge_pipeline.cli import build_parser


def fixture(tmp_path):
    root=tmp_path/'bundle'
    (root/'assets').mkdir(parents=True)
    image=root/'assets/ui.jpg'
    Image.new('RGB',(160,100),'white').save(image)
    write_json(root/'manifest.json',{'title':'fixture'})
    write_json(root/'timeline.json',[{'index':1,'start':0,'end':3,'visual_route':'semantic_frame',
        'material_types':['ui','text'],'assets':[{'path':'assets/ui.jpg'}],
        'quality_issues':['screen_text_low_confidence']}])
    return root


def test_explicit_rapidocr_crop_route_keeps_review_requirement(tmp_path,monkeypatch):
    root=fixture(tmp_path)
    calls=[]
    def run(rows):
        calls.append(rows)
        return {'available':True,'name':'ebook_rapidocr'},[
            {**r,'executed':True,'ok':True,'text':'可读按钮','source':r['image_path']} for r in rows]
    monkeypatch.setattr(recovery,'_run_rapidocr_candidates',run)
    monkeypatch.setattr(recovery,'_run_captiocr_candidates',lambda *a,**k:pytest.fail('silent engine fallback'))
    result=recovery.run_screen_text_recovery(root,execute_crops=True,execute_ocr=True,engine='rapidocr')
    assert calls and result['engine_requested']=='rapidocr'
    assert result['ocr_summary']['succeeded']>0
    receipt=read_json(root/'runs/screen-text-recovery/run.json')
    assert '--engine rapidocr' in receipt['retry_command']
    assert read_json(root/'mcp-run-screen-text-recovery.args.json')['engine']=='rapidocr'
    row=read_json(root/'timeline.json')[0]
    assert 'screen_text_low_confidence' in row['quality_issues']


def test_engine_validation_is_before_writes(tmp_path):
    root=tmp_path/'missing'
    with pytest.raises(ValueError,match='engine'):
        recovery.run_screen_text_recovery(root,engine='unknown')
    assert not root.exists()
    args=build_parser().parse_args(['run-screen-text-recovery','bundle','--engine','rapidocr'])
    assert args.engine=='rapidocr'
