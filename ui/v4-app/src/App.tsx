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
  Moon,
  Download,
  TerminalSquare
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
  const [creativitySlider, setCreativitySlider] = useState(0.32); // 0 to 1 scale
  const [generatingMode, setGeneratingMode] = useState<'none'|'choral'|'fugue'>('none');
  const [debugMode, setDebugMode] = useState(false);
  const [debugLogs, setDebugLogs] = useState<Record<number, string[]>>({});
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);
  const [midiData, setMidiData] = useState<string | null>(null);
  
  const synthRef = useRef<Tone.PolySynth | null>(null);
  const engineRef = useRef<NWCKeyboardEngine | null>(null);
  const originalSubjectRef = useRef<NoteData[] | null>(null);

  // 사용자가 수동으로 악보를 편집하여 하성 3성부가 사라지면 최초 테마 보존 상태 리셋
  useEffect(() => {
    const hasAIGenerated = state.notes.some(n => n.voice > 1);
    if (!hasAIGenerated) {
      originalSubjectRef.current = null;
    }
  }, [state.notes]);

  // 오디오 엔진 초기화
  useEffect(() => {
    synthRef.current = new Tone.Sampler({
      urls: {
        A0: "A0.mp3",
        C1: "C1.mp3",
        "D#1": "Ds1.mp3",
        "F#1": "Fs1.mp3",
        A1: "A1.mp3",
        C2: "C2.mp3",
        "D#2": "Ds2.mp3",
        "F#2": "Fs2.mp3",
        A2: "A2.mp3",
        C3: "C3.mp3",
        "D#3": "Ds3.mp3",
        "F#3": "Fs3.mp3",
        A3: "A3.mp3",
        C4: "C4.mp3",
        "D#4": "Ds4.mp3",
        "F#4": "Fs4.mp3",
        A4: "A4.mp3",
        C5: "C5.mp3",
        "D#5": "Ds5.mp3",
        "F#5": "Fs5.mp3",
        A5: "A5.mp3",
        C6: "C6.mp3",
        "D#6": "Ds6.mp3",
        "F#6": "Fs6.mp3",
        A6: "A6.mp3",
        C7: "C7.mp3",
        "D#7": "Ds7.mp3",
        "F#7": "Fs7.mp3",
        A7: "A7.mp3",
        C8: "C8.mp3"
      },
      release: 1,
      baseUrl: "https://tonejs.github.io/audio/salamander/"
    }).toDestination();
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

  // AI 작곡 요청
  const handleGenerate = async (mode: 'choral' | 'fugue') => {
    if (state.notes.length === 0) {
      alert("먼저 오선지를 마우스로 클릭하여 한 개 이상의 음표(주제)를 입력해 주세요!");
      return;
    }
    
    setGeneratingMode(mode);
    setDebugLogs({});

    // 1. 백엔드 상태 검사 및 자동 기동
    let isBackendReady = false;
    try {
      const healthRes = await axios.get(`${API_BASE_URL}/api/health`, { timeout: 1000 });
      if (healthRes.data && healthRes.data.status === "ok") {
        isBackendReady = true;
      }
    } catch (e) {
      isBackendReady = false;
    }

    if (!isBackendReady) {
      console.log("Backend not responding. Attempting to start backend server...");
      try {
        // Vite dev server의 start-backend 미들웨어 호출
        await axios.get(`/api-start-backend`);
        
        // 최대 10회 (5초간) 헬스체크 폴링 진행
        for (let i = 0; i < 10; i++) {
          await new Promise(resolve => setTimeout(resolve, 500));
          try {
            const healthRes = await axios.get(`${API_BASE_URL}/api/health`, { timeout: 500 });
            if (healthRes.data && healthRes.data.status === "ok") {
              isBackendReady = true;
              break;
            }
          } catch (err) {}
        }
      } catch (startErr) {
        console.error("Failed to trigger start backend middleware:", startErr);
      }
    }

    if (!isBackendReady) {
      alert("백엔드 서버 기동에 실패했습니다. 수동으로 initial.bat를 실행해 주세요.");
      setGeneratingMode('none');
      return;
    }

    try {
      const actualTemperature = 0.1 + (creativitySlider * 1.4); // Scale 0~1 to 0.1~1.5
      
      const hasAIGenerated = state.notes.some(n => n.voice > 1);
      let subjectNotesToSend: NoteData[] = [];
      
      if (!hasAIGenerated) {
        subjectNotesToSend = state.notes.filter(n => n.voice === 1);
        originalSubjectRef.current = subjectNotesToSend;
      } else {
        subjectNotesToSend = originalSubjectRef.current || state.notes.filter(n => n.voice === 1);
      }
      
      if (mode === 'choral') {
        const response = await axios.post(`${API_BASE_URL}/api/generate`, {
          subject_notes: subjectNotesToSend,
          target_measures: targetMeasures,
          temperature: actualTemperature,
          refine_iters: 3
        });
        if (response.data.success) {
          const aiNotes: NoteData[] = response.data.results.map((n: any) => ({
            id: Math.random().toString(36).substr(2, 9),
            pitch: n.pitch,
            duration: n.duration,
            offset: n.offset,
            voice: n.voice as 1|2|3|4,
            durationType: n.duration >= 4.0 ? 1 : (n.duration >= 2.0 ? 2 : (n.duration >= 1.0 ? 4 : 8))
          }));
          const engineRefInstance = engineRef.current;
          if (engineRefInstance) {
            engineRefInstance.setNotes(aiNotes);
          }
          if (response.data.midi_base64) setMidiData(response.data.midi_base64);
        }
      } else {
        // SSE Streaming for Fugue
        const response = await fetch(`${API_BASE_URL}/api/stream_fugue`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            subject_notes: subjectNotesToSend,
            target_measures: targetMeasures,
            temperature: actualTemperature,
            refine_iters: 3
          })
        });
        
        if (!response.body) return;
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split('\n\n');
          buffer = parts.pop() || "";
          
          for (const part of parts) {
            if (part.startsWith('data: ')) {
              const jsonStr = part.slice(6);
              try {
                const data = JSON.parse(jsonStr);
                if (data.type === 'chunk' || data.type === 'done') {
                  const aiNotes: NoteData[] = data.notes.map((n: any) => ({
                    id: Math.random().toString(36).substr(2, 9),
                    pitch: n.pitch,
                    duration: n.duration,
                    offset: n.offset,
                    voice: n.voice as 1|2|3|4,
                    durationType: n.duration >= 4.0 ? 1 : (n.duration >= 2.0 ? 2 : (n.duration >= 1.0 ? 4 : 8))
                  }));
                  const engineRefInstance = engineRef.current;
                  if (engineRefInstance) {
                    engineRefInstance.setNotes(aiNotes);
                  }
                  if (data.debug) setDebugLogs(data.debug);
                  if (data.type === 'done' && data.midi_base64) {
                    setMidiData(data.midi_base64);
                    try {
                      await Tone.start();
                      const synth = new Tone.PolySynth(Tone.Synth).toDestination();
                      synth.volume.value = -10;
                      synth.triggerAttackRelease(["C4", "E4", "G4", "C5"], "8n");
                    } catch (e) {
                      console.error("Audio play error", e);
                    }
                  }
                } else if (data.type === 'debug') {
                  setDebugLogs(data.debug);
                } else if (data.type === 'retry') {
                  // Retry animation: Keep only the subject and clear the rest
                  const engineRefInstance = engineRef.current;
                  if (engineRefInstance) {
                    engineRefInstance.setNotes(state.notes.filter(n => n.voice === 1));
                  }
                } else if (data.type === 'error') {
                  console.error("Stream Error:", data.message);
                }
              } catch (e) {
                console.error("Stream JSON Parse Error:", e);
              }
            }
          }
        }
      }
    } catch (error) {
      console.error("AI Generation Error:", error);
      alert("백엔드 서버가 켜져 있는지 확인해 주세요.");
    } finally {
      setGeneratingMode('none');
    }
  };

  const noteElementsRef = useRef<Record<string, SVGGElement>>({});

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
    const themeColor = isDarkMode ? "#ffffff" : "#1e293b";

    state.notes.forEach(note => {
      if (note.pitch === -1) return; 
      const freq = Tone.Frequency(note.pitch, "midi").toFrequency();
      const timeInSec = note.offset * (60 / currentBpm);
      const durationInSec = note.duration * (60 / currentBpm);
      
      transport.schedule((t) => {
        synthRef.current?.triggerAttackRelease(freq, durationInSec * 0.85, t);
      }, timeInSec); 
    });

    // 60FPS 부드러운 애니메이션 및 스크롤루프
    let animationFrameId: number;
    let prevActiveIds = new Set<string>();

    const updateVisuals = () => {
      if (Tone.getTransport().state !== "started") return;

      const currentBeat = Tone.getTransport().seconds * (currentBpm / 60);
      
      // 현재 프레임에서 활성화된 노트 식별
      const currentActiveIds = new Set<string>();
      state.notes.forEach(note => {
        if (note.pitch === -1) return;
        if (currentBeat >= note.offset && currentBeat < note.offset + note.duration && note.id) {
          currentActiveIds.add(note.id);
        }
      });

      // 새롭게 활성화된 노트 ON (빨간색)
      currentActiveIds.forEach(id => {
        if (!prevActiveIds.has(id)) {
          const el = noteElementsRef.current[id];
          if (el) {
            el.style.fill = '#ff0000';
            el.style.stroke = '#ff0000';
            el.querySelectorAll('*').forEach((p: any) => { 
              p.style.fill = '#ff0000'; 
              p.style.stroke = '#ff0000'; 
            });
          }
        }
      });

      // 더 이상 활성화되지 않은 노트 OFF (원래 색상 복구)
      prevActiveIds.forEach(id => {
        if (!currentActiveIds.has(id)) {
          const el = noteElementsRef.current[id];
          if (el) {
            el.style.fill = themeColor;
            el.style.stroke = themeColor;
            el.querySelectorAll('*').forEach((p: any) => { 
              p.style.fill = themeColor; 
              p.style.stroke = themeColor; 
            });
          }
        }
      });

      prevActiveIds = currentActiveIds;

      // 60FPS 초당 60회 횡스크롤 계산 및 이동 (극강의 부드러움)
      const container = document.querySelector('.score-container') as HTMLElement;
      if (container) {
        const scrollX = Math.max(0, (currentBeat * 95) - (container.clientWidth / 3));
        container.scrollLeft = scrollX;
      }

      animationFrameId = requestAnimationFrame(updateVisuals);
    };

    transport.start();
    animationFrameId = requestAnimationFrame(updateVisuals);
    
    // 마지막 노트 종료 시 정지 처리
    const totalDurationBeats = state.notes.length > 0 ? Math.max(...state.notes.map(n => n.offset + n.duration)) : 0;
    const totalDurationSec = (totalDurationBeats * (60 / currentBpm)) + 1;
    
    transport.schedule(() => {
      setIsPlaying(false);
      transport.stop();
      cancelAnimationFrame(animationFrameId);
      
      // 종료 시 뷰 및 색상 원상 복구
      Object.values(noteElementsRef.current).forEach(el => {
        el.style.fill = themeColor;
        el.style.stroke = themeColor;
        el.querySelectorAll('*').forEach((p: any) => { 
          p.style.fill = themeColor; 
          p.style.stroke = themeColor; 
        });
      });
      const container = document.querySelector('.score-container') as HTMLElement;
      if (container) container.scrollLeft = 0;
    }, totalDurationSec);
  };

  const clearScore = () => {
    engineRef.current?.setNotes([]);
    setMidiData(null);
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
          <div className="slider-group" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <label style={{ fontWeight: 600 }}>Measures</label>
            <input 
              type="range" min="4" max="64" step="4" 
              value={targetMeasures} 
              onChange={(e) => setTargetMeasures(parseInt(e.target.value))}
              style={{ width: '120px' }}
            />
            <span style={{ fontSize: '18px', fontWeight: '900', color: 'var(--accent)', minWidth: '30px', textAlign: 'center' }}>{targetMeasures}</span>
          </div>

          <div className="slider-group" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600 }}>
              Creativity
              <div className="tooltip-wrapper" style={{ position: 'relative', display: 'flex' }}>
                <Info 
                  size={14} 
                  color="var(--text-main)" 
                  style={{ opacity: 0.6, cursor: 'help' }} 
                />
                <div className="tooltip-content">
                  0에 가까울수록 안정성/규칙준수, 1에 가까울수록 창의적이고 예측 불가능한 전개
                </div>
              </div>
            </label>
            <input 
              type="range" min="0" max="1" step="0.01" 
              value={creativitySlider} 
              onChange={(e) => setCreativitySlider(parseFloat(e.target.value))}
              style={{ width: '120px' }}
            />
            <span style={{ fontSize: '18px', fontWeight: '900', color: '#ec4899', minWidth: '40px', textAlign: 'center' }}>{creativitySlider.toFixed(2)}</span>
          </div>

          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <button 
              onClick={() => {
                if (midiData) {
                  const link = document.createElement('a');
                  link.href = `data:audio/midi;base64,${midiData}`;
                  link.download = "fugue_output.mid";
                  link.click();
                }
              }}
              className="nwc-btn"
              disabled={!midiData}
              title="Download MIDI File"
              style={{ borderRadius: '16px', background: 'var(--bg-main)', color: midiData ? 'var(--text-main)' : 'gray', border: '2px solid var(--border)', opacity: midiData ? 1 : 0.5 }}
            >
              <Download size={24} />
            </button>
            <button 
              onClick={handlePlay}
              className="nwc-btn"
              style={{ borderRadius: '16px', background: isPlaying ? '#ef4444' : 'var(--bg-main)', color: isPlaying ? 'white' : 'var(--text-main)', border: '2px solid var(--border)' }}
            >
              <CircleDot size={28} />
            </button>
            <button 
              onClick={() => handleGenerate('choral')}
              disabled={generatingMode !== 'none'}
              className="compose-btn"
              style={{ background: 'linear-gradient(135deg, #6366f1, #a855f7)', boxShadow: '0 4px 15px rgba(99, 102, 241, 0.4)', transition: 'transform 0.2s, box-shadow 0.2s' }}
            >
              {generatingMode === 'choral' ? 'PROCESSING...' : 'CHORAL'}
            </button>
            <button 
              onClick={() => handleGenerate('fugue')}
              disabled={generatingMode !== 'none'}
              className="compose-btn"
              style={{ background: 'linear-gradient(135deg, #ec4899, #f43f5e)', boxShadow: '0 4px 15px rgba(236, 72, 153, 0.4)', transition: 'transform 0.2s, box-shadow 0.2s' }}
            >
              {generatingMode === 'fugue' ? 'PROCESSING...' : 'FUGUE'}
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
              onClick={() => setDebugMode(!debugMode)}
              className={`nwc-btn ${debugMode ? 'active' : ''}`}
              style={{ width: '100%', fontSize: '15px', fontWeight: '800', gap: '12px', border: '2px solid var(--border)', padding: '0 20px', justifyContent: 'flex-start', backgroundColor: debugMode ? '#334155' : 'transparent' }}
            >
              <TerminalSquare size={20} /> 
              {debugMode ? 'HIDE DEBUG' : 'SHOW DEBUG'}
            </button>
            <button 
              onClick={() => setIsDarkMode(!isDarkMode)}
              className="nwc-btn"
              style={{ width: '100%', fontSize: '15px', fontWeight: '800', gap: '12px', border: '2px solid var(--border)', marginTop: '12px', padding: '0 20px', justifyContent: 'flex-start' }}
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
        <section className="studio-workspace" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="score-container" style={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
            <div className="score-paper">
              <ScoreRenderer 
                notes={state.notes} 
                cursorPitch={state.cursorPitch} 
                isDarkMode={isDarkMode}
                width={1200}
                onRenderMap={(map) => { noteElementsRef.current = map; }}
                onNoteClick={(pitch) => {
                  engineRef.current?.addNoteAtPitch(pitch);
                  document.querySelector<HTMLElement>('.studio-root')?.focus();
                }}
              />
            </div>
          </div>

          {/* Debug Console Panel */}
          {debugMode && (
            <div className="debug-console-panel" style={{
              height: '35%',
              minHeight: '250px',
              backgroundColor: '#0f172a',
              borderTop: '2px solid #334155',
              overflowY: 'auto',
              padding: '16px',
              fontFamily: 'Consolas, Monaco, monospace',
              fontSize: '13px',
              boxShadow: 'inset 0 4px 6px -1px rgba(0, 0, 0, 0.5)'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', marginBottom: '16px', color: '#38bdf8', fontWeight: 'bold', fontSize: '14px' }}>
                <TerminalSquare size={16} style={{ marginRight: '8px' }}/> 
                AI Engine Neural Activity Console
              </div>
              
              {Object.keys(debugLogs).length === 0 ? (
                <div style={{ opacity: 0.5, color: '#94a3b8' }}>&gt; 대기 중... FUGUE 작곡을 시작하면 AI의 실시간 추론(Reasoning) 및 마스킹 로그가 이곳에 표시됩니다.</div>
              ) : (
                Object.entries(debugLogs).map(([measure, logs]) => (
                  <div key={measure} style={{ marginBottom: '20px' }}>
                    <div style={{ color: '#fbbf24', marginBottom: '8px', borderBottom: '1px solid #334155', paddingBottom: '4px', fontWeight: 'bold' }}>
                      === Measure {measure} ===
                    </div>
                    {logs.map((log, idx) => {
                      const isHighlight = log.includes('마스킹') || log.includes('페널티') || log.includes('강제') || log.includes('생성 호출') || log.includes('정상 종료');
                      const isToken = log.includes('토큰:');
                      return (
                        <div key={idx} style={{ 
                          padding: '3px 0', 
                          color: isHighlight ? '#f43f5e' : (isToken ? '#e2e8f0' : '#10b981'),
                          fontWeight: isHighlight ? 'bold' : 'normal',
                          paddingLeft: isToken ? '8px' : '0'
                        }}>
                          <span style={{ color: '#64748b', marginRight: '12px' }}>[{new Date().toISOString().substring(11, 23)}]</span>
                          <span style={{ color: isToken ? '#a78bfa' : 'inherit', marginRight: '6px' }}>{isToken ? '→' : '>'}</span>
                          {log}
                        </div>
                      );
                    })}
                  </div>
                ))
              )}
            </div>
          )}

          {/* Footer inside Workspace */}
          <footer className="studio-footer">
            <div style={{ display: 'flex', alignItems: 'center', gap: '32px' }}>
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <div className={`status-dot ${generatingMode !== 'none' ? 'status-busy' : 'status-ready'}`}></div>
                <span style={{ textTransform: 'uppercase' }}>{generatingMode !== 'none' ? 'AI Engine Processing' : 'Engine Ready'}</span>
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
