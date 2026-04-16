type ShortcutCallback = (key: string) => void;

export class KeyboardManager {
  private listeners: Map<string, ShortcutCallback> = new Map();

  constructor() {
    window.addEventListener('keydown', this.handleKeyDown.bind(this));
  }

  on(key: string, callback: ShortcutCallback) {
    this.listeners.set(key, callback);
  }

  private handleKeyDown(event: KeyboardEvent) {
    // Duration shortcuts: 1-6
    if (/^[1-6]$/.test(event.key)) {
      this.trigger('duration', event.key);
      return;
    }

    // Pitch shortcuts: Arrows
    if (event.key === 'ArrowUp') {
      this.trigger('pitch', 'up');
      event.preventDefault();
      return;
    }
    if (event.key === 'ArrowDown') {
      this.trigger('pitch', 'down');
      event.preventDefault();
      return;
    }
    if (event.key === 'ArrowLeft') {
      this.trigger('navigate', 'left');
      event.preventDefault();
      return;
    }
    if (event.key === 'ArrowRight') {
      this.trigger('navigate', 'right');
      event.preventDefault();
      return;
    }

    // Entry shortcuts: Enter, Space, Backspace
    if (event.key === 'Enter') {
      this.trigger('enter', '');
      return;
    }
    if (event.key === ' ') {
      this.trigger('space', '');
      event.preventDefault();
      return;
    }
    if (event.key === 'Backspace') {
      this.trigger('backspace', '');
      return;
    }

    // Octave: Ctrl + Arrows
    if (event.ctrlKey && event.key === 'ArrowUp') {
      this.trigger('octave', 'up');
      return;
    }
    if (event.ctrlKey && event.key === 'ArrowDown') {
      this.trigger('octave', 'down');
      return;
    }
  }

  private trigger(action: string, value: string) {
    const callback = this.listeners.get(action);
    if (callback) callback(value);
  }

  destroy() {
    window.removeEventListener('keydown', this.handleKeyDown.bind(this));
  }
}
