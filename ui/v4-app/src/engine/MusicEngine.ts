import { Renderer, Stave, StaveNote, Voice, Formatter, BarNote, Barline } from 'vexflow';

export interface NoteData {
  pitch: string; // e.g., "c/4"
  duration: string; // e.g., "q", "h", "w"
  voice: number;
}

export class MusicEngine {
  private renderer: Renderer;
  private container: HTMLDivElement;

  constructor(containerId: string) {
    this.container = document.getElementById(containerId) as HTMLDivElement;
    this.renderer = new Renderer(this.container, Renderer.Backends.SVG);
  }

  render(notes: NoteData[][]) {
    // Clear previous
    this.container.innerHTML = '';
    
    // Get colors from CSS
    const style = getComputedStyle(document.documentElement);
    const staffColor = style.getPropertyValue('--staff-color').trim() || '#000000';
    const scoreBg = style.getPropertyValue('--score-bg').trim() || '#ffffff';

    this.renderer = new Renderer(this.container, Renderer.Backends.SVG);
    this.renderer.resize(1000, 600);
    const context = this.renderer.getContext();
    context.setFont("Arial", 10).setBackgroundFillStyle(scoreBg);

    const staveWidth = 800;
    const voices = [0, 1, 2, 3]; // S, A, T, B
    
    voices.forEach((v, index) => {
      const stave = new Stave(10, index * 120 + 20, staveWidth);
      stave.addClef(v < 2 ? "treble" : "bass");
      
      // Set staff color on context
      stave.setContext(context);
      context.setStrokeStyle(staffColor);
      context.setFillStyle(staffColor);
      stave.draw();

      const voiceNotes = notes[v] || [];
      if (voiceNotes.length > 0) {
        
        let currentMeasureBeats = 0;
        const tickables: (StaveNote | BarNote)[] = [];

        // Helper to get beat value (simplified)
        const getDurValue = (d: string) => {
           const vMap: Record<string, number> = { 'w': 4, 'h': 2, 'q': 1, '8': 0.5, '16': 0.25, '32': 0.125 };
           return vMap[d.replace('r', '')] || 1;
        };

        voiceNotes.forEach((n, i) => {
          const sn = new StaveNote({
            clef: v < 2 ? "treble" : "bass",
            keys: [n.pitch],
            duration: n.duration
          });
          sn.setStyle({ fillStyle: staffColor, strokeStyle: staffColor });
          tickables.push(sn);

          currentMeasureBeats += getDurValue(n.duration);

          // Add Barline if 4 beats reached, except if it's the very last note
          if (currentMeasureBeats >= 4 && i < voiceNotes.length - 1) {
             const bar = new BarNote();
             bar.setType(Barline.type.SINGLE);
             tickables.push(bar);
             currentMeasureBeats = 0;
          }
        });

        const vexVoice = new Voice({ numBeats: 4, beatValue: 4 });
        vexVoice.setStrict(false); 
        vexVoice.addTickables(tickables);

        new Formatter().joinVoices([vexVoice]).format([vexVoice], staveWidth - 100);
        vexVoice.draw(context, stave);
      }
    });
  }
}
