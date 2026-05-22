import React, { useEffect, useRef } from 'react';
import { Renderer, Stave, StaveNote, Voice, Formatter, StaveConnector, Accidental } from 'vexflow';
import type { NoteData } from '../engine/NWCKeyboardEngine';

interface ScoreRendererProps {
  notes: NoteData[];
  cursorPitch: number;
  isDarkMode: boolean;
  width?: number;
  onRenderMap?: (map: Record<string, SVGGElement>) => void;
}

export const ScoreRenderer: React.FC<ScoreRendererProps> = ({ 
  notes, 
  cursorPitch, 
  isDarkMode,
  width = 1400,
  onRenderMap
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    try {
      containerRef.current.innerHTML = '';

      const BEATS_PER_MEASURE = 4.0;
      const lastNote = notes.length > 0 ? notes.reduce((max, n) => n.offset + n.duration > max.offset + max.duration ? n : max) : null;
      const maxMeasureIdx = Math.max(0, Math.floor((lastNote ? lastNote.offset + lastNote.duration - 0.001 : 0) / BEATS_PER_MEASURE));
      
      const dynamicWidth = 100 + 450 + (maxMeasureIdx * 380) + 100; // startX + first measure + rest measures + padding

      const renderer = new Renderer(containerRef.current, Renderer.Backends.SVG);
      renderer.resize(dynamicWidth, 600);
      const context = renderer.getContext();
      
      const themeColor = isDarkMode ? "#ffffff" : "#1e293b";
      context.setFillStyle(themeColor);
      context.setStrokeStyle(themeColor);

      const measures: Record<number, NoteData[]>[] = [];
      
      for (let i = 0; i <= maxMeasureIdx; i++) {
        measures[i] = { 1: [], 2: [], 3: [], 4: [] };
      }

      notes.forEach(note => {
        let currentOffset = note.offset;
        let remainingDuration = note.duration;
        
        while (remainingDuration > 0.001) {
          const mIdx = Math.floor(currentOffset / BEATS_PER_MEASURE);
          if (mIdx < 0 || mIdx > maxMeasureIdx) break;
          
          const measureEnd = (mIdx + 1) * BEATS_PER_MEASURE;
          const durationInThisMeasure = Math.min(remainingDuration, measureEnd - currentOffset);
          
          if (measures[mIdx]) {
            // Find closest DurationType (1=w, 2=h, 4=q, 8=8, 16=16, 32=32)
            let durType: number = 4;
            const rounded = Math.round(durationInThisMeasure * 1000) / 1000;
            if (rounded >= 4.0) durType = 1;
            else if (rounded >= 2.0) durType = 2;
            else if (rounded >= 1.0) durType = 4;
            else if (rounded >= 0.5) durType = 8;
            else if (rounded >= 0.25) durType = 16;
            else durType = 32;

            measures[mIdx][note.voice as 1|2|3|4].push({
              ...note,
              offset: currentOffset,
              duration: durationInThisMeasure,
              durationType: durType as any
            });
          }
          
          currentOffset += durationInThisMeasure;
          remainingDuration -= durationInThisMeasure;
        }
      });

      let startX = 100;
      const yTreble = 80;
      const yBass = 280;
      
      // Store all created notes for the map
      const allRenderedNotes: { note: StaveNote, id: string }[] = [];

      measures.forEach((measure, idx) => {
        const measureWidth = idx === 0 ? 450 : 380;
        const trebleStave = new Stave(startX, yTreble, measureWidth);
        const bassStave = new Stave(startX, yBass, measureWidth);

        if (idx === 0) {
          trebleStave.addClef('treble').addTimeSignature('4/4');
          bassStave.addClef('bass').addTimeSignature('4/4');
        }

        trebleStave.setContext(context).draw();
        bassStave.setContext(context).draw();

        if (idx === 0) {
          new StaveConnector(trebleStave, bassStave).setType(3).setContext(context).draw();
          new StaveConnector(trebleStave, bassStave).setType(1).setContext(context).draw();
        }

        const createStrictVoice = (voiceNotes: NoteData[], clef: string, stemDirection: number, isSoprano: boolean = false) => {
          const vfNotes: StaveNote[] = [];
          let currentPos = idx * BEATS_PER_MEASURE;
          const targetEnd = (idx + 1) * BEATS_PER_MEASURE;

          const sortedNotes = [...voiceNotes].sort((a, b) => a.offset - b.offset);

          sortedNotes.forEach(n => {
            if (n.offset > currentPos) {
              vfNotes.push(...createRests(currentPos, n.offset, clef, stemDirection, isDarkMode));
              currentPos = n.offset;
            }
            
            if (currentPos < targetEnd) {
              const dur = Math.min(n.duration, targetEnd - currentPos);
              const isRest = n.pitch === -1;
              const key = isRest ? (stemDirection === 1 ? 'b/4' : 'd/5') : pitchToVexKey(n.pitch);
              const sn = new StaveNote({
                clef,
                keys: [key],
                duration: isRest ? `${durationToVex(n.durationType)}r` : durationToVex(n.durationType)
              });
              
              sn.setStemDirection(stemDirection);
              sn.setStyle({ fillStyle: themeColor, strokeStyle: themeColor });
              if (!isRest && key.includes("#")) sn.addModifier(new Accidental("#"));
              
              if (n.id) {
                sn.setAttribute('id', `note-${n.id}`);
                allRenderedNotes.push({ note: sn, id: n.id });
              }
              
              vfNotes.push(sn);
              currentPos += dur;
            }
          });

          if (isSoprano && idx === measures.length - 1) {
            const cursorNote = new StaveNote({ clef: 'treble', keys: [pitchToVexKey(cursorPitch)], duration: 'q', stem_direction: 1 });
            cursorNote.setStyle({ fillStyle: '#2dd4bf', strokeStyle: '#2dd4bf' });
            vfNotes.push(cursorNote);
            currentPos += 1.0;
          }

          if (currentPos < targetEnd) {
            vfNotes.push(...createRests(currentPos, targetEnd, clef, stemDirection, isDarkMode));
          }

          const voice = new Voice({ num_beats: 4, beat_value: 4 }).setStrict(false);
          return voice.addTickables(vfNotes);
        };

        const vSoprano = createStrictVoice(measure[1], 'treble', 1, true); 
        const vAlto = createStrictVoice(measure[2], 'treble', -1);        
        const vTenor = createStrictVoice(measure[3], 'bass', 1);          
        const vBassNote = createStrictVoice(measure[4], 'bass', -1);      

        new Formatter().joinVoices([vSoprano, vAlto]).format([vSoprano, vAlto], measureWidth - 100);
        vSoprano.draw(context, trebleStave);
        vAlto.draw(context, trebleStave);

        new Formatter().joinVoices([vTenor, vBassNote]).format([vTenor, vBassNote], measureWidth - 100);
        vTenor.draw(context, bassStave);
        vBassNote.draw(context, bassStave);

        startX += measureWidth;
      });

      // 렌더링 완료 후 모든 노트의 SVG Element를 추출하여 부모로 전달
      if (onRenderMap) {
        const svgMap: Record<string, SVGGElement> = {};
        allRenderedNotes.forEach(item => {
          // VexFlow 4는 사용자가 부여한 id 앞에 'vf-'를 자동으로 붙입니다.
          let domEl = document.getElementById(`vf-note-${item.id}`);
          // 만약 'vf-'가 붙지 않았다면 원래 id로 다시 시도
          if (!domEl) domEl = document.getElementById(`note-${item.id}`);
          
          if (domEl) {
            svgMap[item.id] = domEl as SVGGElement;
          }
        });
        onRenderMap(svgMap);
      }

    } catch (error) {
      console.error('Render Error:', error);
    }
  }, [notes, cursorPitch, isDarkMode, width, onRenderMap]);

  return (
    <div className="score-wrapper">
      <div ref={containerRef} className="score-renderer" />
    </div>
  );
};

function createRests(start: number, end: number, clef: string, stemDirection: number, isDarkMode: boolean): StaveNote[] {
  const rests: StaveNote[] = [];
  let remaining = end - start;
  while (remaining > 0.001) {
    let dur = 1.0;
    let type = 'q';
    if (remaining >= 4.0) { dur = 4.0; type = 'w'; }
    else if (remaining >= 2.0) { dur = 2.0; type = 'h'; }
    else if (remaining >= 1.0) { dur = 1.0; type = 'q'; }
    else if (remaining >= 0.5) { dur = 0.5; type = '8'; }
    else { dur = remaining; type = '16'; }
    
    const r = new StaveNote({ 
      clef, 
      keys: [stemDirection === 1 ? 'b/4' : 'd/5'], 
      duration: type + 'r'
    });
    r.setStemDirection(stemDirection);
    const restColor = isDarkMode ? 'rgba(255, 255, 255, 0.45)' : 'rgba(30, 41, 59, 0.45)';
    r.setStyle({ fillStyle: restColor, strokeStyle: restColor });
    rests.push(r);
    remaining -= dur;
  }
  return rests;
}

function pitchToVexKey(pitch: number): string {
  const notes = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'];
  const octave = Math.floor(pitch / 12) - 1;
  return `${notes[pitch % 12]}/${octave}`;
}

function durationToVex(type: number): string {
  const map: Record<number, string> = { 1: 'w', 2: 'h', 4: 'q', 8: '8', 16: '16', 32: '32' };
  return map[type] || 'q';
}
