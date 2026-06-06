import torch
import torch.nn.functional as F
import pickle
import os
import sys

sys.path.append(os.getcwd())
from src.v5.models import UnifiedTransformerV5, BLOCK_SIZE

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
            
        self.model = UnifiedTransformerV5(self.tokenizer.vocab_size, device=self.device, is_causal=True).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.eval()
            print(">>> V5 Hybrid Fugue Engine loaded successfully.")
            
        # Cache token mappings for fast NumPy validation
        self.token_pitches = {}
        self.token_durations = {}
        self.token_voices = {}
        for token, tid in self.tokenizer.stoi.items():
            if token.startswith("P"):
                try: self.token_pitches[tid] = int(token[1:])
                except Exception: pass
            elif token.startswith("D"):
                try: self.token_durations[tid] = float(token[1:])
                except Exception: pass
            elif token.startswith("[VOICE_"):
                try: self.token_voices[tid] = int(token[7])
                except Exception: pass

    def _filter_logits(self, logits, voice, last_pitch, already_generated_notes, current_offset, current_seq):
        # 1. 복제본 생성
        masked_logits = logits.clone()
        
        # 성부별 정석 음역대 정의
        VOICE_RANGES = {
            1: (60, 81), # Soprano: C4 ~ A5
            2: (55, 76), # Alto: G3 ~ E5
            3: (48, 69), # Tenor: C3 ~ A4
            4: (40, 64)  # Bass: E2 ~ E4
        }
        
        # 2. 현재 성부의 음역대 가져오기
        v_min, v_max = VOICE_RANGES.get(voice, (40, 81))
        
        # 3. 현재 오프셋에 울리고 있는 다른 성부들의 (성부, 피치) 파악
        active_other_voices = []
        for note in already_generated_notes:
            if note['voice'] != voice:
                # 쉼표가 아닌 실음이 울리고 있는 구간 확인
                if note['offset'] <= current_offset < note['offset'] + note['duration']:
                    active_other_voices.append((note['voice'], note['pitch']))
                    
        # 4. 각 토큰에 대해 검사
        for tid, p in self.token_pitches.items():
            # 4-1. 음역대 이탈 검사
            if p < v_min or p > v_max:
                masked_logits[0, tid] = -1e9
                continue
                
            # 4-2. 수평적 도약 제한
            # 1옥타브(12반음) 초과 도약 금지
            # 또한 감5도(6반음), 단7도(10반음), 장7도(11반음) 등의 불협화음 도약 방지
            if last_pitch is not None:
                interval = abs(p - last_pitch)
                if interval > 12: # 1옥타브 초과 도약 차단
                    masked_logits[0, tid] = -1e9
                    continue
                if interval in [6, 10, 11]: # 트라이톤, 단7도, 장7도 도약 제한
                    masked_logits[0, tid] = -1e9
                    continue
            
            # 4-3. 수직적 화성 협화 및 Voice Crossing 방지 (현재 오프셋에 다른 성부가 울리고 있을 때)
            for other_v, other_p in active_other_voices:
                harm_interval = abs(p - other_p)
                # 단2도(1), 장7도(11) 같은 거친 불협화음 원천 배제
                if harm_interval in [1, 11]:
                    masked_logits[0, tid] = -1e9
                    break
                # 증4도/감5도(트라이톤=6) 배제
                if harm_interval == 6:
                    masked_logits[0, tid] = -1e9
                    break
                # 성부 간 유니즌(음 겹침) 배제 (단, 옥타브는 허용하되 같은 피치 0은 차단)
                if harm_interval == 0:
                    masked_logits[0, tid] = -1e9
                    break
                    
                # Voice Crossing 방지: 성부 간의 수직적 순서 보장 (Soprano 1 > Alto 2 > Tenor 3 > Bass 4)
                if other_v < voice: # other_v가 더 높은 성부 (예: 1 < 2)
                    if p >= other_p: # 내가 더 높은 성부의 피치보다 같거나 높게 불면 Voice Crossing
                        masked_logits[0, tid] = -1e9
                        break
                elif other_v > voice: # other_v가 더 낮은 성부 (예: 4 > 3)
                    if p <= other_p: # 내가 더 낮은 성부의 피치보다 같거나 낮게 불면 Voice Crossing
                        masked_logits[0, tid] = -1e9
                        break
                                
        # 5. 과도한 쉼표 제약 및 듀레이션 제약
        rest_tid = self.tokenizer.stoi.get("[REST]", None)
        
        # 5-1. 현재 성부의 최근 생성 토큰들을 역순으로 추적
        voice_tokens = []
        i = len(current_seq) - 1
        while i >= 0:
            t = current_seq[i]
            if t == f"[VOICE_{voice}]":
                j = i + 1
                while j < len(current_seq):
                    nt = current_seq[j]
                    if nt.startswith("[VOICE_") or nt.startswith("[BAR_") or nt == "[FINAL]":
                        break
                    voice_tokens.append(nt)
                    j += 1
                if voice_tokens:
                    break
            i -= 1

        if len(voice_tokens) >= 1:
            last_t = voice_tokens[-1]
            if last_t == "[REST]":
                # REST 직후 -> 2.0박 이하의 듀레이션만 허용하고, 피치 및 쉼표 토큰 차단
                for tid, dur in self.token_durations.items():
                    if dur > 2.0:
                        masked_logits[0, tid] = -1e9
                for tid in self.token_pitches:
                    masked_logits[0, tid] = -1e9
                if rest_tid is not None:
                    masked_logits[0, rest_tid] = -1e9
            elif last_t.startswith("P"):
                # 피치 직후 -> 듀레이션만 허용
                for tid in self.token_pitches:
                    masked_logits[0, tid] = -1e9
                if rest_tid is not None:
                    masked_logits[0, rest_tid] = -1e9
            elif last_t.startswith("D"):
                # 듀레이션 직후 -> 다음 음(피치 또는 쉼표) 선택
                # 만약 직전 음표가 쉼표였다면 연속 쉼표 방지
                if len(voice_tokens) >= 2:
                    prev_t = voice_tokens[-2]
                    if prev_t == "[REST]":
                        if rest_tid is not None:
                            masked_logits[0, rest_tid] = -1e9
                            
        # 5-2. 다른 성부들이 이 오프셋에서 모두 쉬고 있다면, 이 성부는 쉴 수 없음 (침묵 구간 방지)
        if rest_tid is not None and masked_logits[0, rest_tid] > -1e8:
            active_voices_count = 0
            for note in already_generated_notes:
                if note['offset'] <= current_offset < note['offset'] + note['duration']:
                    active_voices_count += 1
            if active_voices_count == 0 and len(already_generated_notes) > 0:
                masked_logits[0, rest_tid] = -1e9

        # 5-3. 쉼표 자체에 대한 페널티 부여 (쉼표가 과도하게 자주 선택되는 현상 방지)
        if rest_tid is not None and masked_logits[0, rest_tid] > -1e8:
            masked_logits[0, rest_tid] -= 3.5

        # 6. 세이프가드 (데드락 방지): 화성 규칙 등으로 인해 유효한 피치 후보가 극단적으로 적을 때, 쉼표가 강제되는 버그 방지
        num_active_pitches = sum([1 for tid in self.token_pitches if masked_logits[0, tid] > -1e8])
        if num_active_pitches < 3:
            # 쉼표를 제외한 피치 후보군이 고갈되면, 원래 logits로 복원하되 정석 음역대 이탈만 필터링
            masked_logits = logits.clone()
            for tid, p in self.token_pitches.items():
                if p < v_min or p > v_max:
                    masked_logits[0, tid] = -1e9
            # 복원 후에도 쉼표 페널티는 유지
            if rest_tid is not None and masked_logits[0, rest_tid] > -1e8:
                masked_logits[0, rest_tid] -= 3.5
                    
        return masked_logits

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
                        
                        current_notes, voice_offsets = self._parse_v5_tokens_with_offsets(current_seq)
                        current_offset = voice_offsets[v]
                        
                        filtered_logits = self._filter_logits(
                            logits=logits[:, -1, :],
                            voice=v,
                            last_pitch=last_pitches[v],
                            already_generated_notes=current_notes,
                            current_offset=current_offset,
                            current_seq=current_seq
                        )
                        
                        probs = F.softmax(filtered_logits / temperature, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                        token = self.tokenizer.itos.get(idx_next.item(), "[UNK]")
                        
                        if token.startswith("[VOICE_") or token.startswith("[BAR_") or token == "[FINAL]":
                            break
                            
                        if token.startswith("P"):
                            try:
                                p = int(token[1:])
                                last_pitches[v] = p
                            except Exception: pass
                            
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
            if m % 8 == 0:
                current_seq.append("[EPISODE_MODULATION]")
            current_seq.append(f"[BAR_{m}]")
            debug_data_cont = {str(m): [f"=== Measure {m} Generation Start ===", "AI 자유 대위법 전개 중..."]}
            
            for v in range(1, 5):
                current_seq.append(f"[VOICE_{v}]")
                idx = torch.tensor([self.tokenizer.encode(current_seq)], dtype=torch.long, device=self.device)
                
                # Generate notes for this voice until it tries to start a new voice or bar
                for _ in range(60): # Max 60 tokens per voice per measure
                    logits, _ = self.model(idx[:, -BLOCK_SIZE:])
                    
                    current_notes, voice_offsets = self._parse_v5_tokens_with_offsets(current_seq)
                    current_offset = voice_offsets[v]
                    
                    filtered_logits = self._filter_logits(
                        logits=logits[:, -1, :],
                        voice=v,
                        last_pitch=last_pitches[v],
                        already_generated_notes=current_notes,
                        current_offset=current_offset,
                        current_seq=current_seq
                    )
                    
                    # Prevent generating structural tokens in the middle of a voice
                    for tid, tok in self.tokenizer.stoi.items():
                        if tok in ["[SUBJECT]", "[ANSWER]", "[EPISODE]", "[EPISODE_MODULATION]", "[TS_4/4]", "[KEY_C]"]:
                            filtered_logits[0, tid] = -1e9
                    
                    probs = F.softmax(filtered_logits / temperature, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    token = self.tokenizer.itos.get(idx_next.item(), "[UNK]")
                    
                    if token.startswith("[BAR_") or token.startswith("[VOICE_") or token == "[FINAL]":
                        break
                        
                    if token.startswith("P"):
                        try:
                            p = int(token[1:])
                            last_pitches[v] = p
                        except Exception: pass
                        
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
            import os
            import datetime
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
            
        # 추론 연산 종료 후 VRAM 캐시 비우기 (메모리 단편화 및 누수 억제)
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
            
        return self._parse_v5_tokens(current_seq)

    def _parse_v5_tokens(self, tokens):
        notes, _ = self._parse_v5_tokens_with_offsets(tokens)
        return notes

    def _parse_v5_tokens_with_offsets(self, tokens):
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
                except Exception: pass
            elif t.startswith("[VOICE_"):
                try: 
                    current_voice = int(t[7:-1])
                    if current_voice not in voice_offsets:
                        voice_offsets[current_voice] = voice_offsets.get(1, 0.0)
                except Exception: pass
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
                
        return notes, voice_offsets
