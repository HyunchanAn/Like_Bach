import React, { useEffect, useRef, useState } from 'react';
import { Renderer, Stave, StaveNote, Voice, Formatter, StaveConnector, Accidental } from 'vexflow';
import type { NoteData } from '../engine/NWCKeyboardEngine';

interface ScoreRendererProps {
  notes: NoteData[];
  cursorPitch: number;
  isDarkMode: boolean;
  width?: number;
  onRenderMap?: (map: Record<string, SVGGElement>) => void;
  onNoteClick?: (pitch: number) => void;
}

export const ScoreRenderer: React.FC<ScoreRendererProps> = ({ 
  notes, 
  cursorPitch, 
  isDarkMode,
  width = 1400,
  onRenderMap,
  onNoteClick
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const stavesRef = useRef<{ treble?: Stave, bass?: Stave }>({});

  const [hoverState, setHoverState] = useState<{ x: number, y: number, pitchName: string } | null>(null);

  const getPitchFromY = (y: number) => {
    let pitch = 71; 
    
    if (y < 200 && stavesRef.current.treble) {
      // VexFlow's exact line calculation
      // line 0 = top line (F5), line 4 = bottom line (E4)
      const line = stavesRef.current.treble.getLineForY(y);
      const step = Math.round(line * 2);
      const diatonicPitches = [77, 76, 74, 72, 71, 69, 67, 65, 64, 62, 60, 59, 57, 55, 53];
      if (step >= 0 && step < diatonicPitches.length) pitch = diatonicPitches[step];
      else if (step < 0) pitch = 79;
      else pitch = 60;
      
    } else if (y >= 200 && stavesRef.current.bass) {
      // line 0 = top line (A3), line 4 = bottom line (G2)
      const line = stavesRef.current.bass.getLineForY(y);
      const step = Math.round(line * 2);
      const diatonicPitches = [57, 55, 53, 52, 50, 48, 47, 45, 43, 41, 40, 38, 36];
      if (step >= 0 && step < diatonicPitches.length) pitch = diatonicPitches[step];
      else if (step < 0) pitch = 59;
      else pitch = 40;
    }
    
    return pitch;
  };

  const handleScoreClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!onNoteClick) return;
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const y = e.clientY - rect.top;
    onNoteClick(getPitchFromY(y));
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!onNoteClick) return;
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const y = e.clientY - rect.top;
    const x = e.clientX - rect.left;
    const pitch = getPitchFromY(y);
    const pitchName = pitchToVexKey(pitch).replace('/', '').toUpperCase();
    setHoverState({ x, y, pitchName });
  };

  const handleMouseLeave = () => {
    setHoverState(null);
  };

  useEffect(() => {
    if (!containerRef.current) return;

    try {
      containerRef.current.innerHTML = '';

      const BEATS_PER_MEASURE = 4.0;
      
      const maxNoteEnd = notes.length > 0 ? Math.max(...notes.map(n => n.offset + n.duration)) : 0;
      const maxNoteMeasureIdx = Math.max(0, Math.floor(Math.max(0, maxNoteEnd - 0.001) / BEATS_PER_MEASURE));
      
      const v1Notes = notes.filter(n => n.voice === 1);
      const cursorOffset = v1Notes.length > 0 ? Math.max(...v1Notes.map(n => n.offset + n.duration)) : 0;
      const cursorMeasureIdx = Math.floor(cursorOffset / BEATS_PER_MEASURE);
      
      const maxMeasureIdx = Math.max(maxNoteMeasureIdx, cursorMeasureIdx);
      
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
          stavesRef.current = { treble: trebleStave, bass: bassStave };
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

          if (isSoprano && idx === cursorMeasureIdx && Math.abs(currentPos - cursorOffset) < 0.001) {
            const remainingSpace = targetEnd - currentPos;
            if (remainingSpace > 0.001) {
              let cursorDur = 1.0;
              let cursorVexDur = 'q';
              
              if (remainingSpace >= 4.0) { cursorDur = 4.0; cursorVexDur = 'w'; }
              else if (remainingSpace >= 2.0) { cursorDur = 2.0; cursorVexDur = 'h'; }
              else if (remainingSpace >= 1.0) { cursorDur = 1.0; cursorVexDur = 'q'; }
              else if (remainingSpace >= 0.5) { cursorDur = 0.5; cursorVexDur = '8'; }
              else { cursorDur = remainingSpace; cursorVexDur = '16'; }
  
              const cursorNote = new StaveNote({ clef: 'treble', keys: [pitchToVexKey(cursorPitch)], duration: cursorVexDur, stem_direction: 1 });
              cursorNote.setStyle({ fillStyle: '#2dd4bf', strokeStyle: '#2dd4bf' });
              vfNotes.push(cursorNote);
              currentPos += cursorDur;
            }
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

        new Formatter()
          .joinVoices([vSoprano, vAlto])
          .joinVoices([vTenor, vBassNote])
          .format([vSoprano, vAlto, vTenor, vBassNote], measureWidth - 100);

        vSoprano.draw(context, trebleStave);
        vAlto.draw(context, trebleStave);
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
    <div 
      className="score-wrapper" 
      onClick={handleScoreClick} 
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{ cursor: 'crosshair', position: 'relative' }}
    >
      <div ref={containerRef} className="score-renderer" />
      {hoverState && (
        <>
          <div style={{
            position: 'absolute',
            left: hoverState.x + 15,
            top: hoverState.y - 15,
            background: 'rgba(45, 212, 191, 0.9)',
            color: '#fff',
            padding: '2px 6px',
            borderRadius: '4px',
            fontSize: '12px',
            fontWeight: 'bold',
            pointerEvents: 'none',
            boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
            zIndex: 10
          }}>
            {hoverState.pitchName}
          </div>
          <div style={{
            position: 'absolute',
            left: 0,
            right: 0,
            top: hoverState.y,
            height: '1px',
            background: 'rgba(45, 212, 191, 0.4)',
            pointerEvents: 'none',
            zIndex: 9
          }} />
        </>
      )}
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
