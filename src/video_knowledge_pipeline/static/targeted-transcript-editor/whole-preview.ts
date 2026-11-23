// Thin multi-source projection over the existing Hyperaudio kept-section clock.
// Loading/cancellation follows targeted_review_ui.playWholeSource's request token
// and loadedmetadata pattern; no new edit decisions or media engine.
import { editedTime, sourceTime, type PreviewPlan } from './preview';
export type WholeClip = { chapter: string; start: number; end: number; plan: PreviewPlan };
export type WholeTimeline = { clips: WholeClip[]; duration: number };
export function buildWholeTimeline(chapters: { chapter: string; preview: PreviewPlan }[]): WholeTimeline {
  let cursor=0;
  const clips:WholeClip[]=[];
  for(const row of chapters){
    if(!row.preview.sections.length||row.preview.editedDuration<=0)continue;
    const end=Number((cursor+row.preview.editedDuration).toFixed(6));
    clips.push({chapter:row.chapter,start:cursor,end,plan:row.preview});cursor=end;
  }
  return {clips,duration:cursor};
}
export function wholePosition(timeline:WholeTimeline,chapter:string,time:number){
  const clip=timeline.clips.find(c=>c.chapter===chapter);
  return clip ? clip.start+editedTime(clip.plan.sections,time) : 0;
}
export function locateWholeTime(timeline:WholeTimeline,time:number){
  const clamped=Math.max(0,Math.min(timeline.duration,Number.isFinite(time)?time:0));
  const clip=timeline.clips.find(c=>clamped<c.end)||timeline.clips.at(-1);
  return clip ? {chapter:clip.chapter,sourceTime:sourceTime(clip.plan.sections,clamped-clip.start)} : undefined;
}
export function wholeTimeLabel(value:number){
  const ms=Math.max(0,Math.round(value*1000));
  return `${String(Math.floor(ms/60000)).padStart(2,'0')}:${String(Math.floor(ms/1000)%60).padStart(2,'0')}.${String(ms%1000).padStart(3,'0')}`;
}
