/**
 * NWCKeyboardEngine.ts
 * NWC2 스타일의 키보드 조작 이벤트를 처리하고 음표 상태 데이터를 관리하는 코어 엔진
 */

export type DurationType = 1 | 2 | 4 | 8 | 16 | 32;

export interface NoteData {
  id: string;
  pitch: number; // MIDI Pitch
  duration: number; // Quarter note = 1.0
  durationType: DurationType;
  offset: number;
  voice: number;
}

export interface ComposerState {
  notes: NoteData[];
  cursorPitch: number;
  selectedDuration: DurationType;
  currentTime: number;
}

const DURATION_MAP: Record<DurationType, number> = {
  1: 4.0,
  2: 2.0,
  4: 1.0,
  8: 0.5,
  16: 0.25,
  32: 0.125
};

export class NWCKeyboardEngine {
  private state: ComposerState;
  private onChange: (state: ComposerState, lastNote?: NoteData) => void;

  constructor(onChange: (state: ComposerState, lastNote?: NoteData) => void) {
    this.state = {
      notes: [],
      cursorPitch: 71, // B4 (Middle Line of Treble Clef)
      selectedDuration: 4, // Default: Quarter note
      currentTime: 0
    };
    this.onChange = onChange;
  }

  public getState() {
    return { ...this.state };
  }

  public handleKeyDown(e: KeyboardEvent) {
    // 1. Pitch 조정 (방향키)
    if (e.key === 'ArrowUp') {
      this.state.cursorPitch += 1;
      this.notify(this.state.cursorPitch);
    } else if (e.key === 'ArrowDown') {
      this.state.cursorPitch -= 1;
      this.notify(this.state.cursorPitch);
    }

    // 2. Duration 선택 (숫자키 1~6)
    const durationKeys: Record<string, DurationType> = {
      '1': 1, '2': 2, '3': 4, '4': 8, '5': 16, '6': 32
    };
    if (durationKeys[e.key]) {
      this.state.selectedDuration = durationKeys[e.key];
      this.notify();
    }

    // 3. 음표 입력 (Enter)
    if (e.key === 'Enter') {
      this.addNote(false);
    }

    // 4. 쉼표 입력 (Space)
    if (e.code === 'Space') {
      e.preventDefault();
      this.addNote(true);
    }

    // 5. 삭제 (Backspace)
    if (e.key === 'Backspace') {
      this.removeLastNote();
    }
  }

  private addNote(isRest: boolean = false) {
    const duration = DURATION_MAP[this.state.selectedDuration];
    const newNote: NoteData = {
      id: Math.random().toString(36).substr(2, 9),
      pitch: isRest ? -1 : this.state.cursorPitch, // -1 represents a rest
      duration: duration,
      durationType: this.state.selectedDuration,
      offset: this.state.currentTime,
      voice: 1
    };

    this.state.notes = [...this.state.notes, newNote];
    this.state.currentTime += duration;
    this.notify(isRest ? undefined : this.state.cursorPitch);
  }

  private removeLastNote() {
    if (this.state.notes.length === 0) return;
    const lastNote = this.state.notes[this.state.notes.length - 1];
    this.state.notes = this.state.notes.slice(0, -1);
    this.state.currentTime -= lastNote.duration;
    this.notify();
  }

  private notify(item?: NoteData | number) {
    this.onChange({ ...this.state }, item);
  }

  public setNotes(notes: NoteData[]) {
    this.state.notes = notes;
    // 모든 성부의 길이를 합산하는 대신, 소프라노(V1)의 마지막 위치를 기준으로 현재 시간을 설정
    const v1Notes = notes.filter(n => n.voice === 1);
    this.state.currentTime = v1Notes.length > 0 
      ? Math.max(...v1Notes.map(n => n.offset + n.duration))
      : 0;
    this.notify();
  }
}
