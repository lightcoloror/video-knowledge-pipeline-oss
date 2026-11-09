"""Explicit project identity shared by every local review producer.

Reuse the established vkp.review-export-context.v1 contract and atomic writer.
Titles are never inferred from camera filenames or copied from another project.

Dispatch note (STAGE2 rework): a claimed identity is any non-None value. An empty
dict is still a claim (an incomplete one), not permission to fall through to a
different inline/root identity. Malformed claimed shapes fail hard; they are not
silently dropped or converted to None.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
from .storage import read_json, write_json

CONTEXT_FILENAME = 'review-export-context.json'
LEGACY_FILENAME = 'targeted-transcript-export-context.json'
IDENTITY_FIELDS = ('topic_title', 'video_title', 'video_id')

def normalize_project_identity(value: Any, *, allow_incomplete: bool = False) -> dict[str, str]:
    if not isinstance(value, dict) or (not allow_incomplete and any(not isinstance(value.get(k), str) or not value[k].strip() for k in IDENTITY_FIELDS)):
        raise ValueError('导出命名缺少选题、视频名称或视频标识，请先配置项目身份。')
    if value.get('time_zone', 'Asia/Shanghai') != 'Asia/Shanghai':
        raise ValueError('导出命名使用北京时间，请将time_zone设为Asia/Shanghai。')
    return {**{k:value[k].strip() if isinstance(value.get(k),str) else '' for k in IDENTITY_FIELDS}, 'time_zone':'Asia/Shanghai'}

def project_root(context: dict[str, Any]) -> Path | None:
    artifacts=context.get('artifacts');raw=context.get('bundle_dir') or (artifacts.get('bundle_dir') if isinstance(artifacts,dict) else None)
    return Path(raw).expanduser().resolve() if isinstance(raw,str) and raw else None

def configured_identity(root: str | Path | None) -> dict[str, Any] | None:
    if root is None: return None
    root=Path(root)
    for name in (CONTEXT_FILENAME, LEGACY_FILENAME):
        path=root/name
        if path.is_file():
            value=read_json(path)
            if not isinstance(value,dict):raise ValueError('项目导出配置必须是JSON对象。')
            return value
    manifest=root/'manifest.json'
    if manifest.is_file():
        value=read_json(manifest)
        if not isinstance(value,dict):return None
        # Non-None selection: an explicitly present empty/malformed
        # project_identity claim must not fall through to review_export_context.
        if value.get('project_identity') is not None:
            return value.get('project_identity')
        if value.get('review_export_context') is not None:
            return value.get('review_export_context')
        return None
    return None

def source_catalog(context: dict[str, Any]) -> list[dict[str, Any]]:
    sources=context.get('sources')
    if not isinstance(sources,list):sources=[]
    if not sources and isinstance(context.get('media'),dict):sources=[context['media']]
    result=[]
    for i,source in enumerate(sources,1):
        if not isinstance(source,dict):continue
        path=source.get('path') or source.get('source_path') or ''
        row={'source_id':str(source.get('source_id') or source.get('video_id') or f'source-{i:03d}')}
        if path:row['path']=str(path)
        label=source.get('label') or (Path(str(path)).name if path else '')
        if label:row['file']=str(label)
        digest=source.get('sha256') or source.get('source_sha256')
        if digest:row['sha256']=str(digest)
        if path or label or digest:result.append(row)
    return result

def resolve_review_export_context(context: dict[str, Any], explicit: dict[str, Any] | None = None, *, root: str | Path | None = None) -> dict[str, Any] | None:
    # Existing precedence, but with non-None selection: review_export_context,
    # then project_identity, then configured root. An empty claimed dict stops
    # the search instead of falling through to another identity.
    if explicit is not None:
        candidate=explicit
    else:
        candidate=context.get('review_export_context')
        if candidate is None:candidate=context.get('project_identity')
        if candidate is None:candidate=configured_identity(root or project_root(context))
    if candidate is None:return None
    if not isinstance(candidate,dict):
        raise ValueError('项目导出配置必须是JSON对象。')
    pack_hash=context.get('pack_payload_sha256')
    if candidate.get('pack_payload_sha256') and pack_hash and candidate['pack_payload_sha256']!=pack_hash:
        raise ValueError('review_export_context_pack_changed')
    # An old incomplete config must not hide the review UI; export stays blocked
    # by the shared browser validator until the user supplies the missing fields.
    identity=normalize_project_identity(candidate,allow_incomplete=True)
    result={'schema':'vkp.review-export-context.v1',**identity,'sources':source_catalog(context)}
    base=root or project_root(context)
    if base is not None:result['bundle_dir']=str(Path(base).resolve())
    if pack_hash:result['pack_payload_sha256']=pack_hash
    chapters=(context.get('transcript_edit') or {}).get('chapters',[])
    if chapters:
        hashes={c['number']:c['source_text_hash'] for c in chapters};files={c['number']:c['file'] for c in chapters}
        if candidate.get('source_hashes') is not None and candidate['source_hashes']!=hashes:raise ValueError('导出命名上下文与当前视频来源不匹配。')
        if candidate.get('source_files') is not None and candidate['source_files']!=files:raise ValueError('导出命名上下文与当前视频来源不匹配。')
        result.update(source_hashes=hashes,source_files=files)
    return result

def identity_for_preparation(raw: dict[str, Any], input_root: Path, output_root: Path, explicit: dict[str, Any] | None = None) -> dict[str,str] | None:
    # Preserve the existing producer precedence: explicit, project_identity,
    # review_export_context, output config, input config. Non-None selection
    # keeps an empty explicit/request claim from falling through to a root config.
    if explicit is not None:
        candidate=explicit
    else:
        candidate=raw.get('project_identity')
        if candidate is None:candidate=raw.get('review_export_context')
        if candidate is None:candidate=configured_identity(output_root)
        if candidate is None:candidate=configured_identity(input_root)
    return normalize_project_identity(candidate) if candidate is not None else None

def write_bound_review_export_context(root: Path, context: dict[str, Any]) -> dict[str, Any] | None:
    value=resolve_review_export_context(context,root=root)
    if value is not None:write_json(root/CONTEXT_FILENAME,value)
    return value

def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description='Configure explicit topic/video identity for every review export')
    sub=parser.add_subparsers(dest='action',required=True)
    configure=sub.add_parser('configure');configure.add_argument('root')
    for field in IDENTITY_FIELDS:configure.add_argument('--'+field.replace('_','-'),required=True)
    status=sub.add_parser('status');status.add_argument('root')
    args=parser.parse_args(argv);root=Path(args.root).expanduser().resolve()
    if not root.is_dir():raise ValueError('项目目录不存在。')
    if args.action=='configure':
        identity=normalize_project_identity(vars(args));value={'schema':'vkp.review-export-context.v1',**identity}
        write_json(root/CONTEXT_FILENAME,value)
        result={'status':'configured','path':str(root/CONTEXT_FILENAME),'project_identity':identity,'review_decisions_modified':False,'next_step':'按原流程生成或刷新审核页；已打开的静态页面需重新生成。'}
    else:
        identity=configured_identity(root)
        result={'status':'configured' if identity is not None else 'missing_project_identity','project_identity':normalize_project_identity(identity) if identity is not None else None,'required_fields':list(IDENTITY_FIELDS)}
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
