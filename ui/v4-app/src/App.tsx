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

const API_BASE_URL = 'http://localhost:8000';

const App: React.FC = () => {
  const [state, setState] = useState<ComposerState>({
    notes: [],
    cursorPitch: 71,
    selectedDuration: 4,
    currentTime: 0
  });

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
        target_measures: 16,
        temperature: 0.8,
        refine_iters: 3
      });

      if (response.data.success) {
        // 기존 소프라노 유지 + 생성된 하성 성부 통합
        const aiNotes: NoteData[] = response.data.results.map((n: any) => ({
          id: Math.random().toString(36).substr(2, 9),
          pitch: n.pitch,
          duration: n.duration,
          durationType: 4, // API에서 durationType 정보는 보완 필요
          offset: n.offset,
          voice: n.voice
        }));
        
        // 소프라노(V1)만 남기고 새로 생성된 결과 합치기
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
      setIsPlaying(false);
      return;
    }

    await Tone.start();
    setIsPlaying(true);
    
    const transport = Tone.getTransport();
    transport.cancel();
    transport.bpm.value = 80;

    state.notes.forEach(note => {
      const time = note.offset * (60 / transport.bpm.value) * 4; // 대략적인 싱크
      const freq = Tone.Frequency(note.pitch, "midi").toFrequency();
      
      transport.schedule((t) => {
        synthRef.current?.triggerAttackRelease(freq, note.duration * 0.9, t);
      }, note.offset); 
    });

    transport.start();
    
    // 마지막 노트 종료 시 정지 처리
    const totalDuration = state.notes.length > 0 ? Math.max(...state.notes.map(n => n.offset + n.duration)) : 0;
    transport.schedule(() => {
      setIsPlaying(false);
      transport.stop();
    }, totalDuration + 1);
  };

  const clearScore = () => {
    engineRef.current?.setNotes([]);
  };

  return (
    <div className="flex flex-col h-screen text-white">
      {/* Top Header */}
      <header className="h-16 px-8 flex items-center justify-between border-bottom border-white/10 glass-panel mx-4 mt-4 mb-4">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-4">
            <Music className="w-8 h-8 text-primary glow-text" />
            <div>
              <h1 className="text-2xl font-bold tracking-tight gradient-text">Like Bach Studio</h1>
              <p className="text-xs text-white/40 flex items-center gap-2">
                Premium AI Harmonic Engine v4.5
                <button 
                  onClick={() => setIsDarkMode(!isDarkMode)}
                  className="ml-2 p-1 rounded-md hover:bg-white/10 transition-colors inline-flex items-center"
                  title={isDarkMode ? "Light Mode" : "Dark Mode"}
                >
                  {isDarkMode ? <Sun className="w-3 h-3 text-secondary" /> : <Moon className="w-3 h-3 text-slate-700" />}
                </button>
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10">
            <CircleDot size={14} className="text-green-400 animate-pulse" />
            <span className="text-xs font-medium text-white/60">AI Engine Ready</span>
          </div>
          <button 
            onClick={handleGenerate}
            disabled={isGenerating}
            className={`btn-action ${isGenerating ? 'opacity-50' : ''}`}
          >
            <Cpu size={18} className={isGenerating ? 'animate-spin' : ''} />
            {isGenerating ? 'Generating...' : 'AI Compose'}
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 workspace-container">
        {/* Left Sidebar: Controls & Stats */}
        <aside className="sidebar">
          <section className="glass-panel p-6 flex-1 flex flex-direction-column gap-6">
            <div>
              <h2 className="text-sm font-semibold text-white/40 uppercase tracking-wider mb-4 flex items-center gap-2">
                <Keyboard size={16} /> Keyboard Input (NWC Mode)
              </h2>
              <ul className="space-y-3 text-sm text-white/70">
                <li className="flex justify-between items-center bg-white/5 p-2 rounded">
                  <span>Pitch Adjust</span>
                  <div className="flex gap-1"><span className="key-label">↑</span><span className="key-label">↓</span></div>
                </li>
                <li className="flex justify-between items-center bg-white/5 p-2 rounded">
                  <span>Enter Note</span>
                  <span className="key-label">Enter</span>
                </li>
                <li className="flex justify-between items-center bg-white/5 p-2 rounded">
                  <span>Duration Select</span>
                  <span className="key-label">1-6</span>
                </li>
                <li className="flex justify-between items-center bg-white/5 p-2 rounded">
                  <span>Delete Last</span>
                  <span className="key-label">Back</span>
                </li>
              </ul>
            </div>

            <div className="mt-auto pt-6 border-t border-white/10">
              <div className="flex items-center gap-4 mb-4">
                <button 
                  onClick={() => setIsDarkMode(!isDarkMode)}
                  className="p-2 rounded-full hover:bg-white/10 transition-colors"
                  title={isDarkMode ? "Light Mode" : "Dark Mode"}
                >
                  {isDarkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5 text-slate-700" />}
                </button>
                <div className="flex items-center gap-2 px-3 py-1 glass-panel text-xs text-secondary animate-pulse">
                  <span className="w-2 h-2 rounded-full bg-secondary"></span>
                  AI Engine Ready
                </div>
              </div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs text-white/40">Current Duration</span>
                <span className="text-sm font-bold text-violet-400">{1 / state.selectedDuration} Note</span>
              </div>
              <button onClick={clearScore} className="w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-white/5 hover:bg-red-500/20 text-white/60 hover:text-red-400 transition-all border border-white/10">
                <Eraser size={16} /> Clear Canvas
              </button>
            </div>
          </section>

          <section className="glass-panel p-4 h-32 flex items-center justify-center relative overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-to-br from-violet-600/20 to-fuchsia-600/20 opacity-0 group-hover:opacity-100 transition-opacity" />
            <div className="text-center">
              <p className="text-[10px] text-white/40 uppercase tracking-widest mb-1">Total Notes</p>
              <h3 className="text-3xl font-bold gradient-text">{state.notes.length}</h3>
            </div>
          </section>
        </aside>

        {/* Center: Score Board */}
        <div className="score-editor">
          <div className="canvas-area glass-panel">
            <ScoreRenderer 
              notes={state.notes} 
              cursorPitch={state.cursorPitch} 
              isDarkMode={isDarkMode}
            />
            
            {state.notes.length === 0 && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-20">
                <p className="text-2xl font-light italic">Start composing with keyboard...</p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer / Status Bar */}
      <footer className="status-bar">
        <div className="flex items-center gap-1">
          <ChevronRight size={14} className="text-violet-400" />
          <span>Status: <strong>{isGenerating ? 'AI Generating Harmony...' : 'Ready'}</strong></span>
        </div>
        <div className="flex items-center gap-1">
          <Info size={14} className="text-white/30" />
          <span>Project: Like Bach v4.5</span>
        </div>
        <div className="ml-auto flex gap-4">
          <span>Tempo: 120 BPM</span>
          <span>Time: {state.currentTime.toFixed(2)} Beats</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
