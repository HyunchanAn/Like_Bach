import React, { useState, useEffect, useRef } from 'react';
import { 
  Music, 
  CircleDot, 
  ChevronRight, 
  Cpu, 
  Eraser,
  Keyboard,
  Info,
  Sun,
  Moon
} from 'lucide-react';
import axios from 'axios';
import * as Tone from 'tone';
import { NWCKeyboardEngine } from './engine/NWCKeyboardEngine';
import type { ComposerState, NoteData } from './engine/NWCKeyboardEngine';
import { ScoreRenderer } from './components/ScoreRenderer';
import './App.css';
import './index.css';

const API_BASE_URL = 'http://localhost:8000';

const App: React.FC = () => {
  const [state, setState] = useState<ComposerState>({
    notes: [],
    cursorPitch: 71,
    selectedDuration: 4,
    currentTime: 0
  });

  const [targetMeasures, setTargetMeasures] = useState(8);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);
  const synthRef = useRef<Tone.PolySynth | null>(null);
  const engineRef = useRef<NWCKeyboardEngine | null>(null);

  // 오디오 엔진 초기화
  useEffect(() => {
    synthRef.current = new Tone.PolySynth(Tone.Synth).toDestination();
    synthRef.current.set({
      envelope: { attack: 0.05, decay: 0.1, sustain: 0.3, release: 1 }
    });
  }, []);

  // 엔진 초기화
  useEffect(() => {
    engineRef.current = new NWCKeyboardEngine((newState, item) => {
      setState(newState);
      // 음표 입력 또는 피치 조정 시 피드백 사운드
      if (item && synthRef.current) {
        const pitch = typeof item === 'number' ? item : item.pitch;
        if (pitch === -1) return; // Rests don't play sound
        const freq = Tone.Frequency(pitch, "midi").toFrequency();
        synthRef.current.triggerAttackRelease(freq, "8n");
      }
    });

    const handleKey = (e: KeyboardEvent) => {
      // 폼 입력 중에는 동작 방지
      if (document.activeElement?.tagName === 'INPUT') return;
      engineRef.current?.handleKeyDown(e);
    };

    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, []);

  // Theme effect
  useEffect(() => {
    if (!isDarkMode) {
      document.documentElement.classList.add('light-mode');
    } else {
      document.documentElement.classList.remove('light-mode');
    }
  }, [isDarkMode]);

  // AI 작배 요청
  const handleGenerate = async () => {
    if (state.notes.length === 0) return;
    
    setIsGenerating(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/api/generate`, {
        subject_notes: state.notes.filter(n => n.voice === 1),
        target_measures: targetMeasures,
        temperature: 0.8,
        refine_iters: 3
      });
      if (response.data.success) {
        // 소프라노(V1) 중복 방지: AI 응답 중 V1은 제외하고 V2, V3, V4만 가져오기
        const aiNotes: NoteData[] = response.data.results
          .filter((n: any) => n.voice !== 1)
          .map((n: any) => ({
            id: Math.random().toString(36).substr(2, 9),
            pitch: n.pitch,
            duration: n.duration,
            offset: n.offset,
            voice: n.voice as 1|2|3|4,
            durationType: n.duration === 1.0 ? 4 : (n.duration === 2.0 ? 2 : (n.duration === 4.0 ? 1 : 8))
          }));
        
        // 기존 소프라노 유지 + 생성된 하성 성부 통합
        const newNotes = [...state.notes.filter(n => n.voice === 1), ...aiNotes];
        engineRef.current?.setNotes(newNotes);
      }
    } catch (error) {
      console.error("AI Generation Error:", error);
      alert("백엔드 서버가 켜져 있는지 확인해 주세요.");
    } finally {
      setIsGenerating(false);
    }
  };

  // 재생 기능
  const handlePlay = async () => {
    if (isPlaying) {
      Tone.getTransport().stop();
      Tone.getTransport().cancel();
      Tone.Draw.cancel();
      setIsPlaying(false);
      return;
    }

    await Tone.start();
    setIsPlaying(true);
    
    const transport = Tone.getTransport();
    transport.cancel();
    Tone.Draw.cancel();
    
    const currentBpm = 80;
    transport.bpm.value = currentBpm;

    let scrollRaf: number;
    const startScroll = () => {
      const container = document.querySelector('.score-container') as HTMLElement;
      if (container && transport.state === 'started') {
        const currentBeat = transport.seconds * (currentBpm / 60);
        container.scrollLeft = Math.max(0, (currentBeat * 95) - 300);
        scrollRaf = requestAnimationFrame(startScroll);
      }
    };

    state.notes.forEach(note => {
      if (note.pitch === -1) return; // Skip rests in playback
      const freq = Tone.Frequency(note.pitch, "midi").toFrequency();
      const timeInSec = note.offset * (60 / currentBpm);
      const durInSec = note.duration * (60 / currentBpm);
      
      transport.schedule((t) => {
        synthRef.current?.triggerAttackRelease(freq, note.duration * 0.9, t);
        
        if (note.id) {
          Tone.Draw.schedule(() => {
            const el = document.getElementById(`note-${note.id}`);
            if (el) {
              el.querySelectorAll('path').forEach((p: any) => { 
                p.dataset.origFill = p.style.fill; 
                p.style.fill = '#ef4444'; 
                p.style.stroke = '#ef4444'; 
              });
            }
          }, t);
          
          Tone.Draw.schedule(() => {
            const el = document.getElementById(`note-${note.id}`);
            if (el) {
              el.querySelectorAll('path').forEach((p: any) => { 
                p.style.fill = p.dataset.origFill || ''; 
                p.style.stroke = p.dataset.origFill || ''; 
              });
            }
          }, t + durInSec);
        }
      }, timeInSec); 
    });

    transport.start();
    scrollRaf = requestAnimationFrame(startScroll);
    
    // 마지막 노트 종료 시 정지 처리
    const totalDurationBeats = state.notes.length > 0 ? Math.max(...state.notes.map(n => n.offset + n.duration)) : 0;
    const totalDurationSec = (totalDurationBeats * (60 / currentBpm)) + 1;
    
    transport.schedule(() => {
      setIsPlaying(false);
      transport.stop();
      cancelAnimationFrame(scrollRaf);
    }, totalDurationSec);
  };

  const clearScore = () => {
    engineRef.current?.setNotes([]);
  };

  return (
    <div className={`studio-root ${!isDarkMode ? 'light-theme' : ''}`}>
      {/* Header & Toolbar */}
      <header className="studio-header">
        <div className="header-left">
          <div className="brand">
            <Music size={28} color="var(--accent)" />
            <h1>Like Bach Studio</h1>
          </div>
          
          <div className="nwc-toolbar">
            {[1, 2, 4, 8, 16, 32].map((d) => (
              <button
                key={d}
                onClick={() => engineRef.current?.handleKeyDown({ key: d.toString() } as any)}
                className={`nwc-btn ${state.selectedDuration === d ? 'active' : ''}`}
                title={`1/${d} Note`}
              >
                <span>
                  {d === 1 ? '𝅝' : d === 2 ? '𝅗𝅥' : d === 4 ? '𝅘𝅥' : d === 8 ? '𝅘𝅥𝅮' : d === 16 ? '𝅘𝅥𝅯' : '𝅘𝅥𝅰'}
                </span>
              </button>
            ))}
          </div>
        </div>

        <div className="header-right" style={{ display: 'flex', alignItems: 'center', gap: '32px' }}>
          <div className="slider-group">
            <label>Measures</label>
            <input 
              type="range" min="4" max="32" step="4" 
              value={targetMeasures} 
              onChange={(e) => setTargetMeasures(parseInt(e.target.value))}
              style={{ width: '120px' }}
            />
            <span style={{ fontSize: '18px', fontWeight: '900', color: 'var(--accent)', minWidth: '30px', textAlign: 'center' }}>{targetMeasures}</span>
          </div>

          <div style={{ display: 'flex', gap: '12px' }}>
            <button 
              onClick={handlePlay}
              className="nwc-btn"
              style={{ borderRadius: '16px', background: isPlaying ? '#ef4444' : 'var(--bg-main)', color: isPlaying ? 'white' : 'var(--text-main)', border: '2px solid var(--border)' }}
            >
              <CircleDot size={28} />
            </button>
            <button 
              onClick={handleGenerate}
              disabled={isGenerating}
              className="compose-btn"
            >
              {isGenerating ? 'PROCESSING...' : 'AI COMPOSE'}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="studio-main">
        {/* Sidebar */}
        <aside className="studio-sidebar">
          <div className="sidebar-section">
            <h2><Keyboard size={20} /> Keybindings</h2>
            <div className="key-list">
              <div className="key-row">
                <span className="key-item-label">Pitch Adjust</span>
                <span className="key-item-value">Arrows ↑ ↓</span>
              </div>
              <div className="key-row">
                <span className="key-item-label">Insert Note</span>
                <span className="key-item-value">Enter</span>
              </div>
              <div className="key-row">
                <span className="key-item-label">Insert Rest</span>
                <span className="key-item-value">Space</span>
              </div>
              <div className="key-row">
                <span className="key-item-label">Duration</span>
                <span className="key-item-value">Key 1 - 6</span>
              </div>
            </div>
          </div>

          <div className="info-card">
            <div className="label">Active Note</div>
            <div className="value">
              {['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][state.cursorPitch % 12]}
              <span className="sub-value">{Math.floor(state.cursorPitch / 12) - 1}</span>
            </div>
          </div>

          <div className="sidebar-section" style={{ marginTop: 'auto' }}>
            <button 
              onClick={() => setIsDarkMode(!isDarkMode)}
              className="nwc-btn"
              style={{ width: '100%', fontSize: '15px', fontWeight: '800', gap: '12px', border: '2px solid var(--border)', padding: '0 20px', justifyContent: 'flex-start' }}
            >
              {isDarkMode ? <Sun size={20} /> : <Moon size={20} />} 
              {isDarkMode ? 'LIGHT MODE' : 'DARK MODE'}
            </button>
            <button 
              onClick={clearScore}
              className="nwc-btn"
              style={{ width: '100%', fontSize: '15px', fontWeight: '800', gap: '12px', border: '2px solid var(--border)', marginTop: '12px', color: '#f87171', padding: '0 20px', justifyContent: 'flex-start' }}
            >
              <Eraser size={20} /> RESET PROJECT
            </button>
          </div>
        </aside>

        {/* Workspace */}
        <section className="studio-workspace">
          <div className="score-container">
            <div className="score-paper">
              <ScoreRenderer 
                notes={state.notes} 
                cursorPitch={state.cursorPitch} 
                isDarkMode={isDarkMode}
                width={1200}
              />
            </div>
          </div>

          {/* Footer inside Workspace */}
          <footer className="studio-footer">
            <div style={{ display: 'flex', alignItems: 'center', gap: '32px' }}>
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <div className={`status-dot ${isGenerating ? 'status-busy' : 'status-ready'}`}></div>
                <span style={{ textTransform: 'uppercase' }}>{isGenerating ? 'AI Engine Processing' : 'Engine Ready'}</span>
              </div>
              <span style={{ opacity: 0.2 }}>|</span>
              <span>PROGRESS: {Math.floor(state.currentTime / 4) + 1} / {targetMeasures} MEASURES</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '32px' }}>
              <span>TEMPO: 120 BPM</span>
              <span style={{ opacity: 0.2 }}>|</span>
              <span style={{ color: 'var(--accent)' }}>LIKE BACH STUDIO v4.6.0</span>
            </div>
          </footer>
        </section>
      </main>
    </div>
  );
};

export default App;
