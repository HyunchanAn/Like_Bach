import { 
  File, FolderOpen, Save, Play, Square, 
  Music, Undo, Redo, 
  Scissors, Copy, Clipboard,
  Circle
} from 'lucide-react';

interface ToolbarProps {
  duration: number;
  setDuration: (d: number) => void;
  onCompose: () => void;
  onPlay: () => void;
  onStop: () => void;
}

const Toolbar: React.FC<ToolbarProps> = ({ duration, setDuration, onCompose, onPlay, onStop }) => {
  return (
    <div className="toolbar-container">
      {/* Standard Toolbar */}
      <div className="toolbar glass">
        <button className="toolbar-btn"><File size={16} /></button>
        <button className="toolbar-btn"><FolderOpen size={16} /></button>
        <button className="toolbar-btn"><Save size={16} /></button>
        <div className="toolbar-separator" />
        <button className="toolbar-btn"><Undo size={16} /></button>
        <button className="toolbar-btn"><Redo size={16} /></button>
        <div className="toolbar-separator" />
        <button className="toolbar-btn"><Scissors size={16} /></button>
        <button className="toolbar-btn"><Copy size={16} /></button>
        <button className="toolbar-btn"><Clipboard size={16} /></button>
        <div className="toolbar-separator" />
        <button className="toolbar-btn" onClick={onPlay}><Play size={16} /></button>
        <button className="toolbar-btn" onClick={onStop}><Square size={16} /></button>
        <div className="toolbar-separator" />
        <button 
          className="toolbar-btn" 
          onClick={onCompose}
          style={{ 
            background: 'rgba(212, 175, 55, 0.2)', 
            color: '#d4af37', 
            fontWeight: 'bold', 
            padding: '0 10px',
            fontSize: '12px'
          }}
        >
          GENERATE (Master Bach v4.0)
        </button>
      </div>

      {/* Notation Toolbar (NWC Style) */}
      <div className="toolbar glass" style={{ borderTop: 'none' }}>
        <button className={`toolbar-btn ${duration === 1 ? 'active' : ''}`} onClick={() => setDuration(1)}>
          <span style={{ fontWeight: 800 }}>1</span>
        </button>
        <button className={`toolbar-btn ${duration === 2 ? 'active' : ''}`} onClick={() => setDuration(2)}>
          <span style={{ fontWeight: 800 }}>2</span>
        </button>
        <button className={`toolbar-btn ${duration === 3 ? 'active' : ''}`} onClick={() => setDuration(3)}>
          <span style={{ fontWeight: 800 }}>3</span>
        </button>
        <button className={`toolbar-btn ${duration === 4 ? 'active' : ''}`} onClick={() => setDuration(4)}>
          <span style={{ fontWeight: 800 }}>4</span>
        </button>
        <button className={`toolbar-btn ${duration === 5 ? 'active' : ''}`} onClick={() => setDuration(5)}>
          <span style={{ fontWeight: 800 }}>5</span>
        </button>
        <button className={`toolbar-btn ${duration === 6 ? 'active' : ''}`} onClick={() => setDuration(6)}>
          <span style={{ fontWeight: 800 }}>6</span>
        </button>
        <div className="toolbar-separator" />
        <button className="toolbar-btn"><Circle size={14} fill="currentColor" /></button> {/* Dot */}
        <div className="toolbar-separator" />
        <button className="toolbar-btn">#</button> {/* Sharp */}
        <button className="toolbar-btn">b</button> {/* Flat */}
        <button className="toolbar-btn">♮</button> {/* Natural */}
        <div className="toolbar-separator" />
        <button className="toolbar-btn"><Music size={16} /></button> {/* Tie */}
      </div>
    </div>
  );
};

export default Toolbar;
