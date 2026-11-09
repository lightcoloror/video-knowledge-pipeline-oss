// Adapted from VKP subtitle_editor_ui._safe_filename and lecture_package's
// trailing-dot guard. Uses built-in Intl; shared by VKP and the legacy page.
export type ReviewExportContext = { topic_title: string; video_title: string; video_id: string; time_zone: 'Asia/Shanghai'; pack_payload_sha256: string; source_hashes: Record<string,string>; source_files: Record<string,string>; provenance?: string[] };
declare global { interface Window { RWE_TRANSCRIPT_EXPORT_CONTEXT?: ReviewExportContext } }
export type ReviewSource = { chapter: string; file: string; source_hash: string };
export type ProjectExportContext = {topic_title:string;video_title:string;video_id:string;time_zone:'Asia/Shanghai';pack_payload_sha256?:string;bundle_dir?:string;sources?:Record<string,string>[]};
export function safeExportPart(value: string, limit: number) {
  const cleaned = String(value||'').trim().replace(/[\u0000-\u001f\u007f<>:"/\\|?*\u202a-\u202e\u2066-\u2069]/g,'_').replace(/[. ]+$/g,'');
  if (!cleaned) throw Error('缺少选题或视频命名标识，不能导出通用文件名。');
  return Array.from(cleaned).slice(0,limit).join('');
}
function projectClock(exported_at:string, context?:ProjectExportContext) {
  if (!context?.topic_title?.trim() || !context?.video_title?.trim() || !context?.video_id?.trim() || context.time_zone!=='Asia/Shanghai') throw Error('缺少选题、视频标识或北京时间设置；请先配置导出上下文。');
  const date = new Date(exported_at);
  if (!Number.isFinite(date.getTime())) throw Error('原导出时间无效，不能用当前时间冒充。');
  const parts = Object.fromEntries(new Intl.DateTimeFormat('en-CA',{timeZone:context.time_zone,year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hourCycle:'h23'}).formatToParts(date).map(p=>[p.type,p.value]));
  const ms=String(date.getUTCMilliseconds()).padStart(3,'0');
  const stamp=`${parts.year}${parts.month}${parts.day}_${parts.hour}${parts.minute}${parts.second}_${ms}_UTC+0800`;
  const local=`${parts.year}-${parts.month}-${parts.day}T${parts.hour}:${parts.minute}:${parts.second}.${ms}+08:00`;
  const identity={schema:'vkp.review-export-identity.v1',topic_title:context.topic_title,video_title:context.video_title,video_id:context.video_id,
    ...(context.pack_payload_sha256?{pack_payload_sha256:context.pack_payload_sha256}:{}),exported_at_utc:date.toISOString(),exported_at_local:local,time_zone:context.time_zone};
  return {prefix:[safeExportPart(context.topic_title,40),safeExportPart(context.video_title,24),safeExportPart(context.video_id,40)],stamp,identity};
}
function prepareIdentity(review:{exported_at:string;chapters:ReviewSource[]}, context?:ReviewExportContext) {
  const clock=projectClock(review.exported_at,context);
  if (!Array.isArray(review.chapters) || !review.chapters.length) throw Error('审核结果格式无效。');
  if (!context?.source_hashes || !context.source_files || new Set(review.chapters.map(c=>c.chapter)).size!==review.chapters.length || review.chapters.some(c=>context.source_hashes[c.chapter]!==c.source_hash||context.source_files[c.chapter]!==c.file) || Object.keys(context.source_hashes).length!==review.chapters.length) throw Error('导出命名上下文与当前视频来源不匹配。');
  const {stamp}=clock,identity={...clock.identity,scope:'all_chapters',chapter_count:review.chapters.length,source_files:review.chapters.map(c=>({chapter:c.chapter,file:c.file,source_hash:c.source_hash}))};
  const prefix=[...clock.prefix,`${review.chapters.length}段口播`];
  return {prefix,stamp,identity};
}
export function prepareReviewExport<T extends {schema:string;exported_at:string;chapters:ReviewSource[]}>(review:T, context?:ReviewExportContext) {
  if (review.schema!=='rwe.transcript-review.v3') throw Error('审核结果格式无效。');
  const {prefix,stamp,identity}=prepareIdentity(review,context);
  const filename=[...prefix,'审核结果','v3','导出'+stamp].join('_')+'.json';
  return {filename,payload:{...review,export_identity:{...identity,filename}}};
}
export function prepareExactReviewExport<T extends {schema_version:string;pack_payload_sha256:string;exported_at:string;decisions:{review_id:string}[]}>(review:T,context:ReviewExportContext,chapters:ReviewSource[],requestIds:string[]) {
  if(!context)throw Error('缺少选题、视频名称或视频标识，请先配置项目身份。');
  if(review.schema_version!=='vkp-targeted-review-decisions.draft.v1'||!context||review.pack_payload_sha256!==context.pack_payload_sha256)throw Error('精确审核结果与项目命名上下文不匹配。');
  if(!Array.isArray(review.decisions)||!review.decisions.length||!Array.isArray(requestIds)||new Set(requestIds).size!==requestIds.length||new Set(review.decisions.map(r=>r.review_id)).size!==review.decisions.length||review.decisions.some(r=>!requestIds.includes(r.review_id)))throw Error('精确审核项来源不匹配，不能导出。');
  if(!chapters.length)return prepareProjectReviewExport('精确审核结果',review,context);
  const {prefix,stamp,identity}=prepareIdentity({exported_at:review.exported_at,chapters},context);
  const filename=[...prefix,'精确审核结果',`${review.decisions.length}项`,'导出'+stamp].join('_')+'.json';
  return {filename,payload:{...review,export_identity:{...identity,scope:review.decisions.length===requestIds.length?'all_requests':'selected_requests',review_count:review.decisions.length,review_ids:review.decisions.map(r=>r.review_id),filename}}};
}
// Scope lookup follows the existing review_export_naming interval/key model.
export function prepareProjectReviewExport<T extends Record<string,any>>(kind:string,review:T,context?:ProjectExportContext,locators:Record<string,any>[]=[],now=new Date()) {
  const exported=review.exported_at||now.toISOString(),clock=projectClock(exported,context);
  if(context?.pack_payload_sha256&&review.pack_payload_sha256&&review.pack_payload_sha256!==context.pack_payload_sha256)throw Error('审核结果与项目来源不匹配。');
  const selected=review.decisions??review.reviews??review.rows??review.items??review.suggestions;
  const rows:Array<Record<string,any>>=Array.isArray(selected)?selected:locators;
  const normalizedPath=(v:any)=>String(v||'').replace(/\\/g,'/').replace(/\/$/,'').toLowerCase();
  const bundles=new Set(rows.map(r=>normalizedPath(r.bundle_dir)).filter(Boolean));
  if(bundles.size>1||(context?.bundle_dir&&[...bundles].some(b=>b!==normalizedPath(context.bundle_dir))))throw Error('审核包含其他项目，请分别导出或先配置正确的项目身份。');
  const seconds=(v:any):number|null=>{if(typeof v==='number')return Number.isFinite(v)&&v>=0?v:null;if(typeof v!=='string'||!v.trim())return null;const p=v.trim().split(':');return p.every(x=>/^\d+(?:\.\d+)?$/.test(x))?p.reduce((n,x)=>n*60+Number(x),0):null;};
  const interval=(r:any):number[]|null=>{let a=seconds(r.review_start_seconds??r.start_seconds??r.start??r.start_time),b=seconds(r.review_end_seconds??r.end_seconds??r.end??r.end_time);if((a===null||b===null)&&(r.time_range||r.time)){const pair=String(r.time_range||r.time).split(/\s*[-–—]\s*/);if(pair.length===2)[a,b]=pair.map(seconds);}return a!==null&&b!==null&&b>=a?[a,b]:null;};
  const keys=['review_id','candidate_id','section_id','index','timeline_index'];
  const spans=rows.map(row=>{if(interval(row))return interval(row);const key=keys.find(k=>row[k]!==undefined&&row[k]!==null);const found=key&&locators.find(r=>String(r[key]??(key==='timeline_index'?r.index:undefined))===String(row[key])&&(!row.bundle_dir||normalizedPath(r.bundle_dir)===normalizedPath(row.bundle_dir)));return found?interval(found):null;});
  let scope=rows.length?`${rows.length}项`:'0项';
  const time=(s:number)=>{const ms=Math.round(s*1000),pad=(v:number)=>String(v).padStart(2,'0');return `${pad(Math.floor(ms/3600000))}h${pad(Math.floor(ms/60000)%60)}m${pad(Math.floor(ms/1000)%60)}s${ms%1000?String(ms%1000).padStart(3,'0'):''}`;};
  if(rows.length&&spans.every(Boolean)){const sorted=(spans as number[][]).sort((a,b)=>a[0]-b[0]),merged:number[][]=[];for(const s of sorted){const last=merged.at(-1);if(last&&s[0]<=last[1])last[1]=Math.max(last[1],s[1]);else merged.push([...s]);}scope=merged.length===1?`${time(merged[0][0])}-${time(merged[0][1])}`:`多片段-${rows.length}项`;}
  if(review.review_scope==='full_video')scope='全视频';
  const filename=[...clock.prefix,safeExportPart(kind,32),safeExportPart(scope,64),'导出'+clock.stamp].join('_')+'.json';
  const identity={...clock.identity,scope,review_count:rows.length,source_files:context?.sources||[],review_ids:rows.map(r=>r.review_id??r.candidate_id??r.section_id??r.index??r.timeline_index).filter(v=>v!==undefined),filename};
  return {filename,payload:{...review,exported_at:exported,export_identity:identity}};
}
export function exportContextHint(context?:ReviewExportContext, count=0) {
  return context?.topic_title&&context.video_title&&context.video_id ? `导出命名：${context.topic_title} / ${context.video_title} / 全部${count}段 / 北京时间（精确到毫秒）` : '导出前需配置选题与视频标识。';
}
