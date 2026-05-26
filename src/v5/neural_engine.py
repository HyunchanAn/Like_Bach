import torch
import torch.nn.functional as F
import pickle
import os
import sys

sys.path.append(os.getcwd())
from src.v5.models import FugueTransformerV5, BLOCK_SIZE

class HybridFugueEngine:
    def __init__(self, model_path='data/processed/v5/fugue_model_v5.pt', tokenizer_path='data/processed/v5/fugue_vocab_v5.pkl'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        with open(tokenizer_path, 'rb') as f:
            vocab_dict = pickle.load(f)
            class MockTokenizer: pass
            self.tokenizer = MockTokenizer()
            self.tokenizer.stoi = vocab_dict['stoi']
            self.tokenizer.itos = vocab_dict['itos']
            self.tokenizer.vocab_size = len(self.tokenizer.stoi)
            self.tokenizer.encode = lambda seq: [self.tokenizer.stoi.get(t, self.tokenizer.stoi.get("[UNK]", 0)) for t in seq]
            
        self.model = FugueTransformerV5(self.tokenizer.vocab_size, device=self.device).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.eval()
            print(f">>> V5 Hybrid Fugue Engine loaded successfully.")
            
        # Cache token mappings for fast NumPy validation
        self.token_pitches = {}
        self.token_durations = {}
        self.token_voices = {}
        for token, tid in self.tokenizer.stoi.items():
            if token.startswith("P"):
                try: self.token_pitches[tid] = int(token[1:])
                except: pass
            elif token.startswith("D"):
                try: self.token_durations[tid] = float(token[1:])
                except: pass
            elif token.startswith("[VOICE_"):
                try: self.token_voices[tid] = int(token[7])
                except: pass

    def generate_fugue(self, subject_notes, target_measures=16, temperature=0.55, refine_iters=3, stream_queue=None):
        import random
        if not subject_notes:
            return []
        
        subject_end_beat = max([n['offset'] + n['duration'] for n in subject_notes])
        subject_measures = int((subject_end_beat + 3.99) // 4)
        current_seq = []
        all_debug_logs = {}
        
        # 1. Exposition (V1 -> V2 -> V3 -> V4)
        last_pitches = {1: None, 2: None, 3: None, 4: None}
        
        for m in range(1, 4 * subject_measures + 1):
            current_seq.append(f"[BAR_{m}]")
            debug_data_exp = {str(m): [f"=== Measure {m} Exposition ===", "4성부 강제 스케줄링 진행 중"]}
            
            for v in range(1, 5):
                current_seq.append(f"[VOICE_{v}]")
                
                entry_start_measure = (v - 1) * subject_measures + 1
                entry_end_measure = v * subject_measures
                
                if m >= entry_start_measure and m <= entry_end_measure:
                    # Forced Entry (Subject or Answer)
                    # Zero-shot 호환성을 위해 [SUBJECT_START], [ANSWER_START] 주입 보류 (Phase 2에서 적용)
                            
                    start_beat = (m - entry_start_measure) * 4.0
                    end_beat = start_beat + 4.0
                    notes_in_measure = [n for n in subject_notes if n['offset'] < end_beat and n['offset'] + n['duration'] > start_beat]
                    
                    if not notes_in_measure:
                        current_seq.extend(["[REST]", "D4.0"])
                    else:
                        transposition = 0
                        if v == 2: transposition = -7
                        elif v == 3: transposition = -12
                        elif v == 4: transposition = -19
                        
                        current_beat = start_beat
                        for n in sorted(notes_in_measure, key=lambda x: x['offset']):
                            if n['offset'] > current_beat:
                                rest_dur = n['offset'] - current_beat
                                current_seq.extend(["[REST]", f"D{round(rest_dur, 3)}"])
                                current_beat = n['offset']
                                
                            note_start = max(n['offset'], start_beat)
                            note_end = min(n['offset'] + n['duration'], end_beat)
                            note_dur = note_end - note_start
                            
                            if note_dur > 0:
                                p = int(n['pitch']) + transposition
                                current_seq.extend([f"P{p}", f"D{round(note_dur, 3)}"])
                                current_beat += note_dur
                                last_pitches[v] = p
                                
                        if current_beat < end_beat:
                            rest_dur = end_beat - current_beat
                            current_seq.extend(["[REST]", f"D{round(rest_dur, 3)}"])
                            
                elif m < entry_start_measure:
                    # Voice hasn't entered yet -> Rest
                    current_seq.extend(["[REST]", "D4.0"])
                    
                else:
                    # Voice has already entered -> AI generated Counterpoint
                    idx = torch.tensor([self.tokenizer.encode(current_seq)], dtype=torch.long, device=self.device)
                    for _ in range(60):
                        logits, _ = self.model(idx[:, -BLOCK_SIZE:])
                        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                        token = self.tokenizer.itos.get(idx_next.item(), "[UNK]")
                        
                        if token.startswith("[VOICE_") or token.startswith("[BAR_") or token == "[FINAL]":
                            break
                            
                        if token.startswith("P"):
                            try:
                                p = int(token[1:])
                                if last_pitches[v] is not None:
                                    if abs(p - last_pitches[v]) > 15:
                                        p = last_pitches[v] + random.choice([3, 4, -3, -4])
                                        token = f"P{p}"
                                        idx_next[0, 0] = self.tokenizer.stoi.get(token, idx_next.item())
                                last_pitches[v] = p
                            except: pass
                            
                        idx = torch.cat((idx, idx_next), dim=1)
                        current_seq.append(token)
                        
            if stream_queue:
                stream_queue.put({"type": "chunk", "notes": self._parse_v5_tokens(current_seq), "debug": debug_data_exp})
            
            for measure, logs in debug_data_exp.items():
                if measure not in all_debug_logs:
                    all_debug_logs[measure] = []
                all_debug_logs[measure].extend(logs)
                
        # 2. Continuation (Let AI generate freely, but ENFORCE scaffolding)
        for m in range(4 * subject_measures + 1, target_measures + 1):
            current_seq.append(f"[BAR_{m}]")
            debug_data_cont = {str(m): [f"=== Measure {m} Generation Start ===", "AI 자유 대위법 전개 중..."]}
            
            for v in range(1, 5):
                current_seq.append(f"[VOICE_{v}]")
                idx = torch.tensor([self.tokenizer.encode(current_seq)], dtype=torch.long, device=self.device)
                
                # Generate notes for this voice until it tries to start a new voice or bar
                for _ in range(60): # Max 60 tokens per voice per measure
                    logits, _ = self.model(idx[:, -BLOCK_SIZE:])
                    probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    token = self.tokenizer.itos.get(idx_next.item(), "[UNK]")
                    
                    if token.startswith("[BAR_") or token.startswith("[VOICE_") or token == "[FINAL]":
                        break
                        
                    if token.startswith("P"):
                        try:
                            p = int(token[1:])
                            if last_pitches[v] is not None:
                                prev_p = last_pitches[v]
                                if abs(p - prev_p) > 15:
                                    p = prev_p + random.choice([3, 4, -3, -4])
                                    token = f"P{p}"
                                    idx_next[0, 0] = self.tokenizer.stoi.get(token, idx_next.item())
                                    debug_data_cont[str(m)].append(f"[Fail-Safe] 성부 {v} 화음 강제 교정 (도약 차단)")
                            last_pitches[v] = p
                        except: pass
                        
                    idx = torch.cat((idx, idx_next), dim=1)
                    current_seq.append(token)
                    
            if stream_queue:
                stream_queue.put({"type": "chunk", "notes": self._parse_v5_tokens(current_seq), "debug": debug_data_cont})
                
            # Save to debug document
            for measure, logs in debug_data_cont.items():
                if measure not in all_debug_logs:
                    all_debug_logs[measure] = []
                all_debug_logs[measure].extend(logs)
                
        # Write the final debug document to disk
        try:
            import os, datetime
            os.makedirs("debug_logs", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            with open(f"debug_logs/DEBUG-V5-{timestamp}.txt", "w", encoding="utf-8") as f:
                f.write("=== V5 Hybrid Fugue Engine Neural Activity Console ===\n")
                for measure, logs in all_debug_logs.items():
                    f.write(f"=== Measure {measure} ===\n")
                    for log in logs:
                        f.write(f"{log}\n")
        except Exception as e:
            print("Failed to save debug document:", e)
            
        return self._parse_v5_tokens(current_seq)

    def _parse_v5_tokens(self, tokens):
        # Convert V5 tokens [BAR_x] [VOICE_y] Pxx Dxx back to absolute MIDI notes for the frontend
        notes = []
        current_bar = 0
        current_voice = 1
        voice_offsets = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        
        for i, t in enumerate(tokens):
            if t.startswith("[BAR_"):
                try: 
                    current_bar = int(t[5:-1])
                    bar_start = (current_bar - 1) * 4.0
                    for v in voice_offsets:
                        if voice_offsets[v] < bar_start:
                            voice_offsets[v] = bar_start
                except: pass
            elif t.startswith("[VOICE_"):
                try: 
                    current_voice = int(t[7:-1])
                    if current_voice not in voice_offsets:
                        voice_offsets[current_voice] = voice_offsets.get(1, 0.0)
                except: pass
            elif t == "[REST]":
                dur = 1.0
                if i + 1 < len(tokens) and tokens[i+1].startswith("D"):
                    dur = float(tokens[i+1][1:])
                elif i + 2 < len(tokens) and tokens[i+2].startswith("D"):
                    dur = float(tokens[i+2][1:])
                voice_offsets[current_voice] += dur
            elif t.startswith("P"):
                p = int(t[1:])
                dur = 1.0
                if i + 1 < len(tokens) and tokens[i+1].startswith("D"):
                    dur = float(tokens[i+1][1:])
                elif i + 2 < len(tokens) and tokens[i+2].startswith("D"):
                    dur = float(tokens[i+2][1:])
                    
                notes.append({
                    "pitch": p,
                    "duration": dur,
                    "offset": voice_offsets[current_voice],
                    "voice": current_voice
                })
                voice_offsets[current_voice] += dur
                
        return notes
