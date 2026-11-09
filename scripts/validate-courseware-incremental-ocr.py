"""Explicit offline synthetic acceptance using already installed OCR weights."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from video_knowledge_pipeline.courseware_incremental_runtime import preflight, execute
from video_knowledge_pipeline.courseware_material import store_scan
from video_knowledge_pipeline.file_hash import sha256_file
from video_knowledge_pipeline.storage import write_json, read_json

parser=argparse.ArgumentParser()
parser.add_argument('--model-policy',required=True)
parser.add_argument('--output-root',required=True)
parser.add_argument('--execute',action='store_true')
args=parser.parse_args()
status=preflight(args.model_policy)
if not args.execute or not status['ready']:
    print(json.dumps(status))
    raise SystemExit(0 if status['ready'] else 2)
root=Path(args.output_root).absolute()
if root.exists():
    raise SystemExit('Use a new explicit synthetic output directory; prior receipts are retained.')
root.mkdir(parents=True)
write_json(root/'manifest.json',{'title':'Synthetic incremental OCR acceptance'})
write_json(root/'timeline.json',[{'index':1,'start':0,'end':4,'transcript':'synthetic only'}])
font=ImageFont.load_default(size=32)
candidates=[]
for i in range(4):
    image=Image.new('RGB',(1000,600),'white')
    draw=ImageDraw.Draw(image)
    draw.text((100,100),'FIRST LINE REMAINS',font=font,fill='black')
    if i<3:
        draw.text((100,220),'SECOND LINE '+('CHANGED' if i==2 else 'ORIGINAL'),font=font,fill='black')
    if i==1:
        draw.ellipse((920,530,930,540),fill='red')
    path=root/f'courseware/frames/{i}.png'
    path.parent.mkdir(parents=True,exist_ok=True)
    image.save(path)
    candidates.append({'page_id':f'page-synthetic-{i}','time':float(i),'end':float(i+1),
                       'frame':path.relative_to(root).as_posix(),'frame_sha256':sha256_file(path)})
store_scan(root,{'schema':'video_knowledge_pipeline.courseware_scan.v1','run_id':'synthetic-incremental',
    'scope':{'start':0,'end':4,'excluded_intervals':[]},'source':{'sha256':'a'*64},
    'windows':[{'start':0,'end':4}],'completed_windows':[0],'candidates':candidates,'failure':None})
report=execute(root,[1,2,3,4],args.model_policy)
texts=['\n'.join(b['text'] for b in read_json(root/f'courseware/pages/page-synthetic-{i}.json')['blocks']) for i in range(4)]
checks={'same_text_despite_pointer':texts[0]==texts[1],
        'deleted_line_absent':'SECOND' not in texts[3].upper(),
        'changed_line_observed':texts[2]!=texts[0],
        'cache_reused':report['reused_regions']>0,
        'all_four_frames_detected':report['detector_calls']==4}
resumed=execute(root,[1,2,3,4],args.model_policy,resume=True)
checks['resume_no_model_inference']=resumed['detector_calls']==0 and resumed['recognition_calls']==0
receipt={'status':'passed' if all(checks.values()) else 'failed','checks':checks,'run':report,
         'scope':'synthetic only, not real-courseware rollout approval'}
write_json(root/'synthetic-acceptance.json',receipt)
print(json.dumps(receipt))
raise SystemExit(0 if all(checks.values()) else 2)
