import React, { useEffect, useRef } from 'react';
import { Renderer, Stave, StaveNote, Voice, Formatter, Beam } from 'vexflow';
import type { NoteData } from '../engine/NWCKeyboardEngine';

interface ScoreRendererProps {
  notes: NoteData[];
  cursorPitch: number;
  isDarkMode?: boolean;
  width?: number;
}

export const ScoreRenderer: React.FC<ScoreRendererProps> = ({ 
  notes, 
  cursorPitch,
  isDarkMode = true,
  width = 1200 
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    try {
      // 이전 내용 삭제
      containerRef.current.innerHTML = '';

      const renderer = new Renderer(containerRef.current, Renderer.Backends.SVG);
      renderer.resize(width, 500);
      const context = renderer.getContext();
      
      const themeColor = isDarkMode ? 'rgba(255, 255, 255, 0.9)' : 'rgba(0, 0, 0, 0.8)';
      context.setFillStyle(themeColor);
      context.setStrokeStyle(themeColor);

      // 마디 분할 로직 (4/4 고정)
      const BEATS_PER_MEASURE = 4.0;
      const measures: Record<number, NoteData[]>[] = [];
      let currentMeasure: Record<number, NoteData[]> = { 1: [], 2: [], 3: [], 4: [] };
      let currentMeasureBeats = 0;

      // 성부별로 정렬하여 마디 채우기 (단순화된 코랄 레이아웃)
      // 실제 구현에서는 소프라노(V1)의 오프셋을 기준으로 마디를 나눕니다.
      const sopranoNotes = notes.filter(n => n.voice === 1).sort((a, b) => a.offset - b.offset);
      
      let measureIdx = 0;
      let measureStartOffset = 0;

      // 4성부 전체 노트를 마디별로 배분
      notes.forEach(note => {
        const mIdx = Math.floor(note.offset / BEATS_PER_MEASURE);
        if (!measures[mIdx]) {
          measures[mIdx] = { 1: [], 2: [], 3: [], 4: [] };
        }
        measures[mIdx][note.voice as 1|2|3|4].push(note);
      });

      let startX = 40;
      const yTreble = 50;
      const yBass = 200;

      measures.forEach((measure, idx) => {
        const measureWidth = idx === 0 ? 300 : 250;
        
        // --- 1. 보표(Stave) 생성 ---
        const trebleStave = new Stave(startX, yTreble, measureWidth);
        const bassStave = new Stave(startX, yBass, measureWidth);

        if (idx === 0) {
          trebleStave.addClef('treble').addTimeSignature('4/4');
          bassStave.addClef('bass').addTimeSignature('4/4');
        }

        trebleStave.setContext(context).draw();
        bassStave.setContext(context).draw();

        // --- 2. 성부별 Voice 및 Notes 생성 ---
        const createVoice = (voiceNotes: NoteData[], clef: string, stemDirection: number) => {
          if (voiceNotes.length === 0) {
            // 빈 마디는 온쉼표로 채움
            const ghostNote = new StaveNote({ clef, keys: ['b/4'], duration: 'wr' });
            ghostNote.setStyle({ fillStyle: 'transparent', strokeStyle: 'transparent' });
            return new Voice({ num_beats: 4, beat_value: 4 }).addTickables([ghostNote]);
          }

          const vfNotes = voiceNotes.map(n => {
            const sn = new StaveNote({
              clef: clef,
              keys: [pitchToVexKey(n.pitch)],
              duration: durationToVex(n.durationType),
              stem_direction: stemDirection
            });
            sn.setStyle({ fillStyle: themeColor, strokeStyle: themeColor });
            return sn;
          });

          return new Voice({ num_beats: 4, beat_value: 4 }).setStrict(false).addTickables(vfNotes);
        };

        // Soprano (V1, Treble, Up) / Alto (V2, Treble, Down)
        const vSoprano = createVoice(measure[1], 'treble', 1);
        const vAlto = createVoice(measure[2], 'treble', -1);
        
        // --- Ghost Cursor (소프라노 레이어에 반투명하게 추가) ---
        const isLatestMeasure = idx === measures.length - 1;
        if (isLatestMeasure) {
          const ghostNote = new StaveNote({
            clef: 'treble',
            keys: [pitchToVexKey(cursorPitch)],
            duration: 'q', // 커서는 기본 4분음표 모양
            stem_direction: 1
          });
          ghostNote.setStyle({ fillStyle: isDarkMode ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.1)', strokeStyle: 'transparent' });
          vSoprano.addTickables([ghostNote]);
        }

        // Tenor (V3, Bass, Up) / Bass (V4, Bass, Down)
        const vTenor = createVoice(measure[3], 'bass', 1);
        const vBass = createVoice(measure[4], 'bass', -1);

        // --- 3. 포매팅 및 그리기 ---
        const formatter = new Formatter();
        
        // 상단 보표 정렬
        formatter.joinVoices([vSoprano, vAlto]).format([vSoprano, vAlto], measureWidth - 50);
        vSoprano.draw(context, trebleStave);
        vAlto.draw(context, trebleStave);

        // 하단 보표 정렬
        new Formatter().joinVoices([vTenor, vBass]).format([vTenor, vBass], measureWidth - 50);
        vTenor.draw(context, bassStave);
        vBass.draw(context, bassStave);

        startX += measureWidth;
      });

    } catch (error) {
      console.error('Advanced Render Error:', error);
      if (containerRef.current) {
        containerRef.current.innerHTML = `<div style="color: rgba(255,255,255,0.3); padding: 40px;">Layout Syncing... (${error})</div>`;
      }
    }

  }, [notes, cursorPitch, isDarkMode, width]);

  return (
    <div ref={containerRef} className="score-container glass-card p-4 overflow-x-auto min-h-[500px]" />
  );
};

function pitchToVexKey(pitch: number): string {
  const notes = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'];
  const octave = Math.floor(pitch / 12) - 1;
  const name = notes[pitch % 12];
  return `${name}/${octave}`;
}

function durationToVex(type: number): string {
  const map: Record<number, string> = { 1: 'w', 2: 'h', 4: 'q', 8: '8', 16: '16', 32: '32' };
  return map[type] || 'q';
}
