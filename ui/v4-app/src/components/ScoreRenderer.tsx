import React, { useEffect, useRef } from 'react';
import { Renderer, Stave, StaveNote, Voice, Formatter, StaveConnector, Accidental } from 'vexflow';
import type { NoteData } from '../engine/NWCKeyboardEngine';

interface ScoreRendererProps {
  notes: NoteData[];
  cursorPitch: number;
  isDarkMode: boolean;
  width?: number;
}

export const ScoreRenderer: React.FC<ScoreRendererProps> = ({ 
  notes, 
  cursorPitch, 
  isDarkMode,
  width = 1400 
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    try {
      containerRef.current.innerHTML = '';

      const renderer = new Renderer(containerRef.current, Renderer.Backends.SVG);
      renderer.resize(width, 600);
      const context = renderer.getContext();
      
      const themeColor = isDarkMode ? "#ffffff" : "#1e293b";
      context.setFillStyle(themeColor);
      context.setStrokeStyle(themeColor);

      const BEATS_PER_MEASURE = 4.0;
      const measures: Record<number, NoteData[]>[] = [];
      
      const lastNote = notes.length > 0 ? notes.reduce((max, n) => n.offset + n.duration > max.offset + max.duration ? n : max) : null;
      const maxMeasureIdx = Math.max(0, Math.floor((lastNote ? lastNote.offset + lastNote.duration - 0.001 : 0) / BEATS_PER_MEASURE));
      
      for (let i = 0; i <= maxMeasureIdx; i++) {
        measures[i] = { 1: [], 2: [], 3: [], 4: [] };
      }

      notes.forEach(note => {
        const mIdx = Math.floor(note.offset / BEATS_PER_MEASURE);
        if (measures[mIdx]) {
          measures[mIdx][note.voice as 1|2|3|4].push(note);
        }
      });

      let startX = 100;
      const yTreble = 80;
      const yBass = 280;

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
              vfNotes.push(...createRests(currentPos, n.offset, clef, stemDirection));
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
              
              // 명시적으로 기둥 방향 강제 적용
              sn.setStemDirection(stemDirection);
              
              sn.setStyle({ fillStyle: themeColor, strokeStyle: themeColor });
              if (!isRest && key.includes("#")) sn.addModifier(new Accidental("#"));
              
              if (n.id) {
                sn.setAttribute('id', `note-${n.id}`);
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
            vfNotes.push(...createRests(currentPos, targetEnd, clef, stemDirection));
          }

          const voice = new Voice({ num_beats: 4, beat_value: 4 }).setStrict(false);
          return voice.addTickables(vfNotes);
        };

        const vSoprano = createStrictVoice(measure[1], 'treble', 1, true); // Soprano: Up
        const vAlto = createStrictVoice(measure[2], 'treble', -1);        // Alto: Down
        const vTenor = createStrictVoice(measure[3], 'bass', 1);          // Tenor: Up
        const vBassNote = createStrictVoice(measure[4], 'bass', -1);      // Bass: Down

        new Formatter().joinVoices([vSoprano, vAlto]).format([vSoprano, vAlto], measureWidth - 100);
        vSoprano.draw(context, trebleStave);
        vAlto.draw(context, trebleStave);

        new Formatter().joinVoices([vTenor, vBassNote]).format([vTenor, vBassNote], measureWidth - 100);
        vTenor.draw(context, bassStave);
        vBassNote.draw(context, bassStave);

        startX += measureWidth;
      });

    } catch (error) {
      console.error('Render Error:', error);
    }
  }, [notes, cursorPitch, isDarkMode, width]);

  return (
    <div className="score-wrapper">
      <div ref={containerRef} className="score-renderer" />
    </div>
  );
};

function createRests(start: number, end: number, clef: string, stemDirection: number): StaveNote[] {
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
    r.setStyle({ fillStyle: 'rgba(128,128,128,0.3)', strokeStyle: 'rgba(128,128,128,0.3)' });
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
