import { useState, useEffect, useRef } from 'react';
import './index.css';
import Toolbar from './components/Toolbar';
import { Activity, Settings, Info, Loader2 } from 'lucide-react';
import { MusicEngine } from './engine/MusicEngine';
import type { NoteData } from './engine/MusicEngine';
import { KeyboardManager } from './engine/KeyboardManager';
import axios from 'axios';

// Pitch conversion helper
const midiToVex = (midi: number): string => {
  const notes = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'];
  const octave = Math.floor(midi / 12) - 1;
  const note = notes[midi % 12];
  return `${note}/${octave}`;
};

const durationToVex = (d: number): string => {
  const map: Record<number, string> = { 1: 'w', 2: 'h', 3: 'q', 4: '8', 5: '16', 6: '32' };
  return map[d] || 'q';
};

function App() {
  const [duration, setDuration] = useState(3);
  const [pitch, setPitch] = useState(72); // C5
  const [notes, setNotes] = useState<NoteData[][]>([[], [], [], []]);
  const [cursorIndex, setCursorIndex] = useState(0); // Tracks current input/edit position in Soprano (voice 0)
  const [status, setStatus] = useState("Ready");
  const [loading, setLoading] = useState(false);
  const [, setTheme] = useState(window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const engineRef = useRef<MusicEngine | null>(null);
  const kbRef = useRef<KeyboardManager | null>(null);

  // Theme Listener
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = (e: MediaQueryListEvent) => {
      setTheme(e.matches ? 'dark' : 'light');
      setStatus(`Theme changed to ${e.matches ? 'dark' : 'light'}`);
    };
    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  // Initialize Engine
  useEffect(() => {
    if (!engineRef.current) {
      engineRef.current = new MusicEngine('score-container');
    }
    // Force re-render on theme change
    if (engineRef.current) {
      engineRef.current.render(notes);
    }
    
    if (!kbRef.current) {
      kbRef.current = new KeyboardManager();
      
      kbRef.current.on('duration', (val) => setDuration(parseInt(val)));
      
      kbRef.current.on('navigate', (dir) => {
        setCursorIndex(prev => {
          const next = dir === 'right' ? prev + 1 : prev - 1;
          return Math.max(0, Math.min(next, notes[0].length)); // Can't go beyond length
        });
      });

      kbRef.current.on('pitch', (dir) => {
        // If cursor is at the end (insert mode), just change pitch state
        if (cursorIndex === notes[0].length) {
          setPitch(p => dir === 'up' ? p + 1 : p - 1);
        } else {
          // Edit existing note if cursor is on it
          setNotes(prev => {
            const next = [...prev];
            const targetNote = next[0][cursorIndex];
            if (targetNote && !targetNote.duration.includes('r')) {
               const currentMidi = pitchToMidi(targetNote.pitch);
               const newMidi = dir === 'up' ? currentMidi + 1 : currentMidi - 1;
               targetNote.pitch = midiToVex(newMidi);
            }
            return next;
          });
        }
      });
      
      kbRef.current.on('octave', (dir) => {
        if (cursorIndex === notes[0].length) {
          setPitch(p => dir === 'up' ? p + 12 : p - 12);
        } else {
           setNotes(prev => {
            const next = [...prev];
            const targetNote = next[0][cursorIndex];
            if (targetNote && !targetNote.duration.includes('r')) {
               const currentMidi = pitchToMidi(targetNote.pitch);
               const newMidi = dir === 'up' ? currentMidi + 12 : currentMidi - 12;
               targetNote.pitch = midiToVex(newMidi);
            }
            return next;
          });
        }
      });

      kbRef.current.on('enter', () => {
        setNotes(prev => {
          const next = [...prev];
          const newNote = { pitch: midiToVex(pitch), duration: durationToVex(duration), voice: 0 };
          
          if (cursorIndex === next[0].length) {
             next[0] = [...next[0], newNote]; // Append
          } else {
             next[0][cursorIndex] = newNote; // Replace existing
          }
          return next;
        });
        setCursorIndex(prev => prev + 1); // Move forward
        setStatus(`Note entered at Soprano: ${midiToVex(pitch)}`);
      });

      kbRef.current.on('space', () => {
        setNotes(prev => {
          const next = [...prev];
          const newRest = { pitch: 'b/4', duration: durationToVex(duration) + 'r', voice: 0 };
          
          if (cursorIndex === next[0].length) {
             next[0] = [...next[0], newRest];
          } else {
             next[0][cursorIndex] = newRest;
          }
          return next;
        });
        setCursorIndex(prev => prev + 1); // Move forward
        setStatus("Rest entered.");
      });

      kbRef.current.on('backspace', () => {
         // Backspace deletes the note BEFORE the cursor. Or if on a note, maybe delete it and shift.
         // Let's implement standard backspace: delete note immediately before cursor and move cursor back.
         if (cursorIndex > 0) {
            setNotes(prev => {
              const next = [...prev];
              next[0].splice(cursorIndex - 1, 1);
              return next;
            });
            setCursorIndex(prev => prev - 1);
            setStatus("Note deleted.");
         }
      });
    }

    return () => {
      // Cleanup if necessary (currently Handled by React strict mode caveats, but kbRef persists)
    };
  }, [pitch, duration, cursorIndex, notes[0].length]); // Ensure callbacks have access to latest state

  // Render when notes change
  useEffect(() => {
    if (engineRef.current) {
      engineRef.current.render(notes);
    }
  }, [notes]);

  const handleCompose = async () => {
    if (notes[0].length === 0) {
      setStatus("Please enter a soprano theme first.");
      return;
    }
    
    setStatus("Neural Engine v4.0: Composing 4-voice fugue...");
    setLoading(true);
    
    try {
      // Convert notes to API format
      const subjectNotes = notes[0].map((n, i) => ({
        pitch: pitchToMidi(n.pitch),
        duration: vexToDur(n.duration),
        offset: i // Simplified offset for now
      }));

      const response = await axios.post('http://localhost:8000/compose', { notes: subjectNotes });
      
      if (response.data.status === 'success') {
        const generated = response.data.notes;
        const newNotes: NoteData[][] = [[], [], [], []];
        
        generated.forEach((gn: any) => {
          const v = (gn.voice || 1) - 1;
          newNotes[v].push({
            pitch: midiToVex(gn.pitch),
            duration: durationToVex(vexToNwcDur(gn.duration)), // Conversion back
            voice: v
          });
        });
        
        setNotes(newNotes);
        setStatus(`Successfully generated ${generated.length} notes.`);
      }
    } catch (err) {
      console.error(err);
      setStatus("Error: Backend not reachable or generation failed.");
    } finally {
      setLoading(false);
    }
  };

  // Helper for API conversion
  const pitchToMidi = (p: string) => {
    const notes = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'];
    const [note, octave] = p.split('/');
    return notes.indexOf(note.toLowerCase()) + (parseInt(octave) + 1) * 12;
  };

  const vexToDur = (v: string) => {
    const map: Record<string, number> = { 'w': 4.0, 'h': 2.0, 'q': 1.0, '8': 0.5, '16': 0.25, '32': 0.125 };
    return map[v.replace('r', '')] || 1.0;
  };

  const vexToNwcDur = (dur: number) => {
    if (dur >= 4.0) return 1;
    if (dur >= 2.0) return 2;
    if (dur >= 1.0) return 3;
    if (dur >= 0.5) return 4;
    if (dur >= 0.25) return 5;
    return 6;
  };
  
  const handlePlay = () => setStatus("Playing (Simulated)...");
  const handleStop = () => setStatus("Stopped.");

  return (
    <div className="glass" style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <div className="top-nav">
        <span style={{ fontWeight: 700, marginRight: 20, color: 'var(--accent-gold)' }}>Like Bach v4.0</span>
        <span style={{ marginRight: 15 }}>File</span>
        <span style={{ marginRight: 15 }}>Edit</span>
        <span style={{ marginRight: 15 }}>View</span>
        <span style={{ marginRight: 15 }}>Staff</span>
        <span style={{ marginRight: 15 }}>Tools</span>
        <span style={{ marginRight: 15 }}>Play</span>
        <span style={{ flex: 1 }} />
        <Activity size={14} style={{ marginRight: 10 }} />
        <Settings size={14} style={{ marginRight: 10 }} />
      </div>

      <Toolbar 
        duration={duration} 
        setDuration={setDuration} 
        onCompose={handleCompose}
        onPlay={handlePlay}
        onStop={handleStop}
      />

      <div className="main-workspace">
        <div className="staff-labels glass">
          {['Soprano', 'Alto', 'Tenor', 'Bass'].map((voice, i) => (
            <div key={voice} style={{ 
              padding: '40px 10px', 
              height: '120px', 
              borderBottom: '1px solid var(--border-color)', 
              fontSize: '12px',
              fontWeight: i === 0 ? 'bold' : 'normal',
              color: i === 0 ? 'var(--text-primary)' : 'var(--text-secondary)'
            }}>
              [{voice}]
            </div>
          ))}
        </div>

        <div className="score-area">
          <div id="score-container" style={{ minWidth: (notes[0].length * 40 + 500) + 'px', height: '600px' }} />
          {/* Cursor is positioned based on cursorIndex */}
          <div className="insertion-cursor" style={{ left: 50 + cursorIndex * 40, top: 40 }} />
        </div>
      </div>

      <div className="status-bar">
        <span style={{ flex: 1 }}>{status}</span>
        {loading && <Loader2 size={12} className="rotate" style={{ marginRight: 10 }} />}
        <span style={{ margin: '0 10px' }}>Pitch: {midiToVex(pitch)}</span>
        <span style={{ margin: '0 10px' }}>Duration: {durationToVex(duration)}</span>
        <span style={{ margin: '0 10px' }}>Voice: 1</span>
        <Info size={12} style={{ marginLeft: 5 }} />
      </div>
    </div>
  );
}

export default App;
