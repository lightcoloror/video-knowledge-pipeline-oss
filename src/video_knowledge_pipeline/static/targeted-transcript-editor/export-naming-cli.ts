// Reissue an existing export using its original timestamp. Never overwrite it.
import {readFileSync,writeFileSync} from 'node:fs';
import {resolve,join,basename} from 'node:path';
import {prepareReviewExport,prepareExactReviewExport} from './export-naming';
const [source,contextFile,destination,packFile]=process.argv.slice(2);
if(!source||!contextFile||!destination)throw Error('Expected source JSON, naming context and existing output directory.');
const old=JSON.parse(readFileSync(source,'utf8')),context=JSON.parse(readFileSync(contextFile,'utf8'));
let result:any;
if(old.schema_version==='vkp-targeted-review-decisions.draft.v1'){
  if(!packFile)throw Error('Exact-review reissue requires the bound review pack.');
  const pack=JSON.parse(readFileSync(packFile,'utf8'));if(pack.pack_payload_sha256!==old.pack_payload_sha256)throw Error('Source review pack does not match.');
  let exported=old.exported_at,provenance='original_exported_at';
  if(!exported){
    const match=basename(source).match(/__(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})(\d{3})(?:-\d+)?\.json$/);
    if(!match)throw Error('Historical export time is missing; refuse to replace it with now.');
    const [,y,m,d,h,minute,s,ms]=match;exported=new Date(`${y}-${m}-${d}T${h}:${minute}:${s}.${ms}+08:00`).toISOString();
    provenance='original_legacy_filename_fixed_UTC+0800';
  }
  result=prepareExactReviewExport({...old,exported_at:exported},context,pack.transcript_edit.chapters.map((c:any)=>({chapter:c.number,file:c.file,source_hash:c.source_text_hash})),pack.requests.map((r:any)=>r.review_id));
  result.payload.export_identity.timestamp_provenance=provenance;
}else result=prepareReviewExport(old,context);
const target=join(resolve(destination),result.filename);
if(target===resolve(source))throw Error('Refusing to overwrite the supplied export.');
writeFileSync(target,JSON.stringify(result.payload,null,2)+'\n',{encoding:'utf8',flag:'wx'});
console.log(JSON.stringify({path:target,exported_at:result.payload.export_identity.exported_at_utc,exported_at_local:result.payload.export_identity.exported_at_local,original_file_preserved:true}));
