import {describe,it,expect,vi} from 'vitest';
import {buildWholeTimeline,wholePosition,locateWholeTime,wholeTimeLabel} from '../whole-preview';
import {CutPreviewController,type PreviewPlan} from '../preview';
const plan=(sections:{start:number;end:number}[]):PreviewPlan=>({sections,cuts:[],editedDuration:sections.reduce((n,s)=>n+s.end-s.start,0),removedSeconds:0,warnings:[],subtitles:{}});
const timeline=()=>buildWholeTimeline([{chapter:'01',preview:plan([{start:2,end:5},{start:7,end:9}])},{chapter:'02',preview:plan([{start:10,end:14}])}]);
describe('whole-program projection over the existing kept-section clock',()=>{
 it('keeps one accumulated clock across source changes',()=>{expect(timeline().duration).toBe(9);expect(wholePosition(timeline(),'02',12)).toBe(7);});
 it('seeks an exact source boundary to the next retained source',()=>{expect(locateWholeTime(timeline(),5)).toEqual({chapter:'02',sourceTime:10});});
 it('seeks across an internal cut without restoring deleted audio',()=>{expect(locateWholeTime(timeline(),3)).toEqual({chapter:'01',sourceTime:7});});
 it('omits an entirely deleted chapter and clamps the global end',()=>{const t=buildWholeTimeline([{chapter:'empty',preview:plan([])},...timeline().clips.map(c=>({chapter:c.chapter,preview:c.plan}))]);expect(t.clips.map(c=>c.chapter)).toEqual(['01','02']);expect(locateWholeTime(t,99)).toEqual({chapter:'02',sourceTime:14});});
 it('supports an empty program without a playback loop',()=>{expect(locateWholeTime(buildWholeTimeline([]),0)).toBeUndefined();});
 it('uses stable millisecond labels with correct minute carry',()=>{expect(wholeTimeLabel(479.978)).toBe('07:59.978');expect(wholeTimeLabel(59.9998)).toBe('01:00.000');});
 it('normalizes accumulated float noise before assigning a range input maximum',()=>{const t=buildWholeTimeline([{chapter:'01',preview:plan([{start:0,end:.1}])},{chapter:'02',preview:plan([{start:0,end:.2}])}]);expect(t.duration).toBe(.3);});
 it('passes source completion to the queue once, then permits replay',()=>{const player={readyState:1,duration:20,currentTime:0,pause:vi.fn(),addEventListener:vi.fn()} as any;const done=vi.fn(()=>true);const p=new CutPreviewController(player,()=>{},done);p.setPlan(plan([{start:2,end:5}]));player.currentTime=5;p.check();p.check();expect(done).toHaveBeenCalledTimes(1);player.currentTime=3;p.check();player.currentTime=5;p.check();expect(done).toHaveBeenCalledTimes(2);});
 it('retains single-chapter stop behavior when no queue is attached',()=>{const player={readyState:1,duration:20,currentTime:0,pause:vi.fn(),addEventListener:vi.fn()} as any;const p=new CutPreviewController(player,()=>{});p.setPlan(plan([{start:2,end:5}]));player.currentTime=5;p.check();expect(player.pause).toHaveBeenCalled();expect(player.currentTime).toBe(20);});
});
