import torch
import torch.nn.functional as F
import pickle
import os
import sys
import random

# Project root 추가
sys.path.append(os.getcwd())

from src.v4.models import BachTransformer, BachTokenizer, BLOCK_SIZE

class NeuralBachEngine:
    def __init__(self, model_path='data/processed/v4/bach_model.pt', tokenizer_path='data/processed/v4/tokenizer.pkl'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        token_path = tokenizer_path if os.path.exists(tokenizer_path) else 'data/processed/v4/tokenizer.pkl'
        fugue_token_path = 'models/v4/fugue_vocab.pkl'
        
        if os.path.exists(token_path):
            with open(token_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
        else:
            self.tokenizer = BachTokenizer()
            
        if os.path.exists(fugue_token_path):
            with open(fugue_token_path, 'rb') as f:
                self.fugue_tokenizer = pickle.load(f)
        else:
            self.fugue_tokenizer = self.tokenizer
            
        self.model = BachTransformer(self.tokenizer.vocab_size, device=self.device).to(self.device)
        self.fugue_model = BachTransformer(self.fugue_tokenizer.vocab_size, device=self.device).to(self.device)
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
            except: pass
            
        fugue_model_path = 'models/v4/fugue_model.pt'
        if os.path.exists(fugue_model_path):
            try:
                self.fugue_model.load_state_dict(torch.load(fugue_model_path, map_location=self.device))
                self.fugue_model.eval()
                print(f"Dual Neural Engine (Chorale & Fugue) loaded.")
            except: pass

        # Pre-build a map of token IDs to pitch and duration values for fast logit masking
        self.token_pitches = {}
        self.token_voices = {}
        self.token_durations = {}
        self.bad_pitch_indices = []
        
        # 성부별 허용 음역대를 더 엄격하게 제한 (시각적으로 너무 높은 덧줄이 생기지 않도록)
        # Soprano(V1): C4(60) ~ A5(81)
        # Alto(V2): F3(53) ~ D5(74)
        # Tenor(V3): A2(45) ~ G4(67)
        # Bass(V4): E2(40) ~ C4(60)
        allowed_ranges = {
            "[V1]": (60, 81),
            "[V2]": (53, 74),
            "[V3]": (45, 67),
            "[V4]": (40, 60)
        }
        
        for token, token_id in self.tokenizer.stoi.items():
            if token.startswith("[V"):
                self.token_voices[token_id] = token[:4]
                v = token[:4]
                try:
                    parts = token.split()
                    if len(parts) >= 2 and parts[1].startswith("P"):
                        pitch_val = int(parts[1][1:])
                        self.token_pitches[token_id] = pitch_val
                        if v in allowed_ranges:
                            min_p, max_p = allowed_ranges[v]
                            if pitch_val < min_p or pitch_val > max_p:
                                self.bad_pitch_indices.append(token_id)
                    if len(parts) >= 3 and parts[2].startswith("D"):
                        self.token_durations[token_id] = float(parts[2][1:])
                except: pass

        self.fugue_token_pitches = {}
        self.fugue_token_voices = {}
        self.fugue_token_durations = {}
        self.fugue_bad_pitch_indices = []
        self.fugue_token_times = {}
        self.fugue_token_remains = []
        
        for token, token_id in self.fugue_tokenizer.stoi.items():
            if token.startswith("[TIME_"):
                try:
                    self.fugue_token_times[token_id] = float(token[6:-1])
                except: pass
            elif token.startswith("[REMAIN_"):
                self.fugue_token_remains.append(token_id)
            elif token.startswith("[V"):
                self.fugue_token_voices[token_id] = token[:4]
                v = token[:4]
                try:
                    parts = token.split()
                    if len(parts) >= 2 and parts[1].startswith("P"):
                        pitch_val = int(parts[1][1:])
                        self.fugue_token_pitches[token_id] = pitch_val
                        if v in allowed_ranges:
                            min_p, max_p = allowed_ranges[v]
                            if pitch_val < min_p or pitch_val > max_p:
                                self.fugue_bad_pitch_indices.append(token_id)
                    if len(parts) >= 3 and parts[2].startswith("D"):
                        self.fugue_token_durations[token_id] = float(parts[2][1:])
                except: pass

    def _get_active_pitches_at_current_time(self, current_seq):
        active_pitches = set()
        for token in reversed(current_seq):
            if token.startswith("[TIME_"):
                break
            if token.startswith("[V"):
                try:
                    parts = token.split()
                    if len(parts) >= 2 and parts[1].startswith("P"):
                        active_pitches.add(int(parts[1][1:]))
                except:
                    pass
        return active_pitches

    def _get_active_voices_at_current_time(self, current_seq):
        active_voices = set()
        for token in reversed(current_seq):
            if token.startswith("[TIME_"):
                break
            if token.startswith("[V"):
                active_voices.add(token[:4])
        return active_voices

    def generate_response(self, subject_notes, target_measures=16, temperature=0.8, refine_iters=3):
        best_notes = None
        best_score = -1.0
        max_attempts = 5
        threshold = 90.0  # 화성 점수 문턱값 (정상 악곡은 보통 89~93점 수준)
        
        for attempt in range(max_attempts):
            notes = self._generate_single_response(subject_notes, target_measures, temperature, refine_iters)
            score = self._evaluate_harmony(notes)
            
            if score >= threshold:
                return notes
                
            if score > best_score:
                best_score = score
                best_notes = notes
                
        return best_notes

    def generate_fugue(self, subject_notes, target_measures=16, temperature=0.8, refine_iters=3, stream_queue=None):
        best_notes = None
        best_score = -1.0
        max_attempts = 5
        threshold = 90.0
        
        for attempt in range(max_attempts):
            if stream_queue and attempt > 0:
                stream_queue.put({"type": "retry", "attempt": attempt + 1})
                
            notes = self._generate_single_fugue_response(subject_notes, target_measures, temperature, refine_iters, stream_queue)
            score = self._evaluate_harmony(notes)
            
            if score >= threshold:
                return notes
                
            if score > best_score:
                best_score = score
                best_notes = notes
                
        return best_notes

    def _generate_single_fugue_response(self, subject_notes, target_measures=16, temperature=0.8, refine_iters=3, stream_queue=None):
        raw_key = "C"  
        current_seq = [f"[KEY_{raw_key}]", "[TS_4/4]", "[SUBJECT]"]
        last_measure_idx = -1
        
        debug_logs = {}
        def add_debug(m_idx, msg):
            if m_idx not in debug_logs:
                debug_logs[m_idx] = []
            debug_logs[m_idx].append(msg)
            
        if not subject_notes:
            return []
            
        subject_end_off = max([float(n['offset']) + float(n['duration']) for n in subject_notes])
        answer_start_off = float(int((subject_end_off + 3.99) // 4) * 4) 
        
        forced_notes = []
        for n in subject_notes:
            off = round(float(n['offset']), 3)
            dur = round(float(n['duration']), 3)
            p = int(n['pitch'])
            
            forced_notes.append({"offset": off, "pitch": p, "duration": dur, "voice": "[V1]"})
            
            ans_off = round(off + answer_start_off, 3)
            forced_notes.append({"offset": ans_off, "pitch": p - 7, "duration": dur, "voice": "[V2]"})
            
            sub2_off = round(off + answer_start_off * 2, 3)
            forced_notes.append({"offset": sub2_off, "pitch": p - 12, "duration": dur, "voice": "[V3]"})
            
            ans2_off = round(off + answer_start_off * 3, 3)
            forced_notes.append({"offset": ans2_off, "pitch": p - 19, "duration": dur, "voice": "[V4]"})
            
        forced_notes.sort(key=lambda x: (x['offset'], x['voice']))
        
        curr_measure = 0
        current_offset = 0.0
        
        # 1. Phase 3: 강제 5도 이조 (Rule-based Exposition) + 푸가 대위법 결합
        for fn in forced_notes:
            off = fn['offset']
            if off >= target_measures * 4.0: break
            curr_measure = int(off // 4)
            
            if curr_measure > last_measure_idx:
                remain = max(0, target_measures - curr_measure)
                current_seq.append(f"[REMAIN_{remain}]")
                last_measure_idx = curr_measure
            
            if off > current_offset or current_offset == 0.0:
                current_seq.append(f"[TIME_{off}]")
                current_offset = off
            
            v = fn['voice']
            p = fn['pitch']
            d = fn['duration']
            
            # [SUBJECT] 및 [ANSWER] 마커 주입
            if off == 0.0 and v == "[V1]":
                current_seq.append("[SUBJECT]")
                add_debug(curr_measure, "토큰: [SUBJECT] (주제부 시작 - 소프라노)")
            elif off == answer_start_off and v == "[V2]":
                current_seq.append("[ANSWER]")
                add_debug(curr_measure, f"토큰: [ANSWER] ({answer_start_off}박자 - 알토 응답)")
            elif off == answer_start_off * 2.0 and v == "[V3]":
                current_seq.append("[SUBJECT]")
                add_debug(curr_measure, f"토큰: [SUBJECT] ({answer_start_off * 2.0}박자 - 테너 옥타브 하강 주제)")
            elif off == answer_start_off * 3.0 and v == "[V4]":
                current_seq.append("[ANSWER]")
                add_debug(curr_measure, f"토큰: [ANSWER] ({answer_start_off * 3.0}박자 - 베이스 12도 하강 응답)")
                
            current_seq.append(f"{v} P{p} D{d}")
            add_debug(curr_measure, f"토큰: {v} P{p} D{d} (규칙 기반 강제 주입)")
            
            # 대주제(Countersubject) AI 생성 호출
            if v == "[V2]":
                add_debug(curr_measure, f"-> [V1] 대주제(CS1) AI 생성 호출")
                self._fill_voices(current_seq, target_measures, curr_measure, temperature, ["[V1]"], p, use_fugue_model=True)
            elif v == "[V3]":
                add_debug(curr_measure, f"-> [V1, V2] 대주제(CS2, CS1) AI 생성 호출")
                self._fill_voices(current_seq, target_measures, curr_measure, temperature, ["[V1]", "[V2]"], p, use_fugue_model=True)
            elif v == "[V4]":
                add_debug(curr_measure, f"-> [V1, V2, V3] 대주제(CS3, CS2, CS1) AI 생성 호출")
                self._fill_voices(current_seq, target_measures, curr_measure, temperature, ["[V1]", "[V2]", "[V3]"], p, use_fugue_model=True)

        current_seq.append("[EPISODE]")
        add_debug(curr_measure, "토큰: [EPISODE] (자유 대위법 전개 시작)")
                
        # 2. AI 자유 작곡 (Continuation - Episode)
        max_new_tokens = 2000
        for _ in range(max_new_tokens):
            if curr_measure >= target_measures:
                break
                
            try:
                current_idx = torch.tensor([self.fugue_tokenizer.encode(current_seq)], dtype=torch.long, device=self.device)
                logits, _ = self.fugue_model(current_idx[:, -BLOCK_SIZE:])
                
                if curr_measure >= target_measures - 1:
                    f_idx = self.fugue_tokenizer.stoi.get("[FINAL]", -1)
                    if f_idx != -1: logits[0, -1, f_idx] += 2.0
                elif curr_measure < target_measures - 1:
                    f_idx = self.fugue_tokenizer.stoi.get("[FINAL]", -1)
                    if f_idx != -1: logits[0, -1, f_idx] = -1e9
                
                # 1. 2옥타브 범위 이탈 토큰 원천 차단
                if self.fugue_bad_pitch_indices:
                    logits[0, -1, self.fugue_bad_pitch_indices] = -1e9
                
                # REMAIN 토큰 원천 차단 (UI 로직과 맞지 않음)
                if self.fugue_token_remains:
                    logits[0, -1, self.fugue_token_remains] = -1e9
                    
                # 시간 역행 및 비정상적 점프 차단 (현재 offset보다 작거나 4.0박자 초과 시 억제)
                bad_time_indices = [tid for tid, t_val in self.fugue_token_times.items() if t_val <= current_offset or t_val > current_offset + 4.0]
                if bad_time_indices:
                    logits[0, -1, bad_time_indices] = -1e9
                
                active_pitches = self._get_active_pitches_at_current_time(current_seq)
                if active_pitches:
                    bad_indices = [tid for tid, p in self.fugue_token_pitches.items() if p in active_pitches]
                    if bad_indices: logits[0, -1, bad_indices] = -1e9
                
                active_voices = self._get_active_voices_at_current_time(current_seq)
                if active_voices:
                    bad_voice_indices = [tid for tid, voice in self.fugue_token_voices.items() if voice in active_voices]
                    if bad_voice_indices: logits[0, -1, bad_voice_indices] = -1e9
                    
                    # 바로크 폴리포니(대위법) 텍스처 보존: 한 오프셋에 이미 2성부 이상 존재하면
                    # 추가적인 성부 쌓기를 강력 억제하여 듬성듬성하고 역동적인 선율선을 유도합니다.
                    if len(active_voices) >= 2:
                        all_voice_indices = [tid for tid in self.fugue_token_voices.keys()]
                        logits[0, -1, all_voice_indices] -= 15.0
                
                # 질질 끄는 음표 방지: 온음표(D4.0) 및 점2분음표(D3.0) 억제, 그리고 직전과 완전히 동일한 노트 반복 시 강력한 페널티
                if curr_measure < target_measures - 1:
                    bad_dur_indices = [tid for tid, dur in self.fugue_token_durations.items() if dur >= 3.0]
                    if bad_dur_indices: logits[0, -1, bad_dur_indices] = -1e9
                    
                    # 32분음표 차단 및 16분음표 억제 (VexFlow 겹침 방지)
                    too_short_indices = [tid for tid, dur in self.fugue_token_durations.items() if dur < 0.25]
                    if too_short_indices: logits[0, -1, too_short_indices] = -1e9
                    
                    short_dur_indices = [tid for tid, dur in self.fugue_token_durations.items() if dur == 0.25]
                    if short_dur_indices: logits[0, -1, short_dur_indices] -= 8.0
                    
                # Repetition Penalty
                last_few_tokens = current_seq[-10:]
                for tid in range(logits.shape[-1]):
                    tok = self.fugue_tokenizer.itos.get(tid, "")
                    if tok.startswith("[V") and tok in last_few_tokens:
                        logits[0, -1, tid] -= 5.0
                    
                probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                token = self.fugue_tokenizer.itos.get(idx_next.item(), "[UNK]")
                
                # 디버그 로그 작성
                logic_notes = []
                if token.startswith("[TIME_") and active_voices and len(active_voices) >= 2:
                    logic_notes.append("다성부 화음 밀집 억제(-15.0)로 인한 시간 진행")
                if token.startswith("[V") and token in last_few_tokens:
                    logic_notes.append("반복 페널티(-5.0) 극복하고 출력")
                if token.startswith("[V") and active_pitches and int(token.split()[1][1:]) in active_pitches:
                    logic_notes.append("유니즌 방지 페널티 극복")
                if token.startswith("[V") and self.fugue_token_durations.get(idx_next.item(), 1.0) <= 0.25:
                    logic_notes.append("초단기 음표 억제(-8.0) 페널티 극복")
                    
                log_msg = f"토큰: {token}"
                if logic_notes:
                    log_msg += f" ({', '.join(logic_notes)})"
                add_debug(curr_measure, log_msg)
                
                if token == "[FINAL]" and curr_measure >= target_measures:
                    current_seq.append(token)
                    add_debug(curr_measure, "토큰: [FINAL] (목표 마디 도달, 생성 정상 종료)")
                    break
 
                current_seq.append(token)
                
                if stream_queue and len(current_seq) % 5 == 0:
                    stream_queue.put({
                        "type": "chunk", 
                        "notes": self._parse_tokens_to_notes(current_seq),
                        "debug": debug_logs
                    })
                
                if token.startswith("[TIME_"):
                    try:
                        off = float(token[6:-1])
                        curr_measure = int(off // 4)
                        current_offset = off
                    except: pass
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                break
 
        if refine_iters > 0:
            current_seq = self._refine_sequence(current_seq, refine_iters, temperature)
 
        notes = self._parse_tokens_to_notes(current_seq)
        if stream_queue:
            stream_queue.put({"type": "debug", "debug": debug_logs})
            
        try:
            import os
            import datetime
            os.makedirs("debug_logs", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            with open(f"debug_logs/DEBUG-{timestamp}.txt", "w", encoding="utf-8") as f:
                for measure, logs in debug_logs.items():
                    f.write(f"=== Measure {measure} ===\n")
                    for log in logs:
                        f.write(f"{log}\n")
        except Exception as e:
            print("Failed to save debug log:", e)
            
        return self._fix_silence_gaps(notes, target_measures * 4.0)

    def _generate_single_response(self, subject_notes, target_measures=16, temperature=0.8, refine_iters=3):
        raw_key = "C" 
        current_seq = [f"[KEY_{raw_key}]", "[TS_4/4]"]
        last_measure_idx = -1
        
        # 1. 고정된 주제 노트를 딕셔너리로 관리 (오프셋 정렬)
        subject_map = {}
        for n in subject_notes:
            off = round(float(n['offset']), 3)
            subject_map[off] = n
  
        sorted_offsets = sorted(subject_map.keys())
        current_offset = 0.0
        curr_measure = 0

        # 2. 사용자 입력(Soprano) 기반의 1차 생성 (Guided Generation)
        for off in sorted_offsets:
            if off >= target_measures * 4.0: break
            curr_measure = int(off // 4)
            
            # 마디 변경 제어
            if curr_measure > last_measure_idx:
                remain = max(0, target_measures - curr_measure)
                current_seq.append(f"[REMAIN_{remain}]")
                last_measure_idx = curr_measure
                self._inject_roman_token(current_seq, temperature)
            
            current_seq.append(f"[TIME_{off}]")
            
            # 해당 지점에 주제(V1)가 있으면 그대로 사용
            n = subject_map[off]
            current_seq.append(f"[V1] P{int(n['pitch'])} D{round(float(n['duration']), 3)}")
            # 하성 3성부만 해당 오프셋에서 동일 리듬으로 생성
            self._fill_voices(current_seq, target_measures, curr_measure, temperature, ["[V2]", "[V3]", "[V4]"], int(n['pitch']))
            current_offset = off
            
        # 3. AI 자유 작곡 (Continuation)
        # 사용자의 입력이 끝난 시점부터 목표 마디 수까지 자유롭게 생성 (Auto-regressive)
        max_new_tokens = 2500
        for _ in range(max_new_tokens):
            if curr_measure >= target_measures:
                break
                
            try:
                current_idx = torch.tensor([self.tokenizer.encode(current_seq)], dtype=torch.long, device=self.device)
                logits, _ = self.model(current_idx[:, -BLOCK_SIZE:])
                
                # 목표 마디에 가까워지면 FINAL 유도
                if curr_measure >= target_measures - 1:
                    f_idx = self.tokenizer.stoi.get("[FINAL]", -1)
                    if f_idx != -1: logits[0, -1, f_idx] += 2.0
                
                # 1. 2옥타브 범위 이탈 토큰 원천 차단
                if self.bad_pitch_indices:
                    logits[0, -1, self.bad_pitch_indices] = -1e9
                
                # 유니즌 방지 마스킹
                active_pitches = self._get_active_pitches_at_current_time(current_seq)
                if active_pitches:
                    bad_indices = [tid for tid, p in self.token_pitches.items() if p in active_pitches]
                    if bad_indices:
                        logits[0, -1, bad_indices] = -1e9
                
                # 성부 중복 방지 마스킹
                active_voices = self._get_active_voices_at_current_time(current_seq)
                if active_voices:
                    bad_voice_indices = [tid for tid, v in self.token_voices.items() if v in active_voices]
                    if bad_voice_indices:
                        logits[0, -1, bad_voice_indices] = -1e9
                
                # 질질 끄는 음표 방지: 온음표(D4.0) 및 점2분음표(D3.0) 억제
                if curr_measure < target_measures - 1:
                    bad_dur_indices = [tid for tid, d in self.token_durations.items() if d >= 3.0]
                    if bad_dur_indices:
                        logits[0, -1, bad_dur_indices] = -1e9
                    
                    # 32분음표 등 지나치게 짧은 음표 차단 (0.25 미만)
                    short_dur_indices = [tid for tid, d in self.token_durations.items() if d < 0.25]
                    if short_dur_indices:
                        logits[0, -1, short_dur_indices] = -1e9
                        
                # Repetition Penalty
                last_few_tokens = current_seq[-10:]
                for tid in range(logits.shape[-1]):
                    tok = self.tokenizer.itos.get(tid, "")
                    if tok.startswith("[V") and tok in last_few_tokens:
                        logits[0, -1, tid] -= 5.0
                    
                probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                token = self.tokenizer.itos.get(idx_next.item(), "[UNK]")
                
                if token == "[FINAL]":
                    if curr_measure >= target_measures:
                        current_seq.append(token)
                        break
                    else:
                        probs[0, idx_next.item()] = 0.0
                        idx_next = torch.multinomial(probs, num_samples=1)
                        token = self.tokenizer.itos.get(idx_next.item(), "[UNK]")
 
                current_seq.append(token)
                
                if token.startswith("[TIME_"):
                    try:
                        off = float(token[6:-1])
                        curr_measure = int(off // 4)
                    except: pass
                    
            except Exception as e:
                break
 
        if refine_iters > 0:
            current_seq = self._refine_sequence(current_seq, refine_iters, temperature)
 
        notes = self._parse_tokens_to_notes(current_seq)
        return self._fix_silence_gaps(notes, target_measures * 4.0)

    def _inject_roman_token(self, current_seq, temperature):
        try:
            current_idx = torch.tensor([self.tokenizer.encode(current_seq)], dtype=torch.long, device=self.device)
            logits, _ = self.model(current_idx[:, -BLOCK_SIZE:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            for _ in range(5):
                idx_next = torch.multinomial(probs, num_samples=1)
                token = self.tokenizer.itos.get(idx_next.item(), "[UNK]")
                if token.startswith("[ROMAN_"):
                    current_seq.append(token)
                    break
        except: pass

    def _fill_voices(self, current_seq, target_measures, curr_measure, temperature, required_voices, base_pitch, use_fugue_model=False):
        found_voices = []
        model = self.fugue_model if use_fugue_model else self.model
        tokenizer = self.fugue_tokenizer if use_fugue_model else self.tokenizer
        
        t_pitches = self.fugue_token_pitches if use_fugue_model else self.token_pitches
        t_voices = self.fugue_token_voices if use_fugue_model else self.token_voices
        t_durations = self.fugue_token_durations if use_fugue_model else self.token_durations
        t_bad_pitch = self.fugue_bad_pitch_indices if use_fugue_model else self.bad_pitch_indices
        
        for retry in range(15):
            try:
                current_idx = torch.tensor([tokenizer.encode(current_seq)], dtype=torch.long, device=self.device)
                logits, _ = model(current_idx[:, -BLOCK_SIZE:])
                
                if curr_measure >= target_measures:
                    f_idx = tokenizer.stoi.get("[FINAL]", -1)
                    if f_idx != -1: logits[0, -1, f_idx] += 2.0
                
                if t_bad_pitch:
                    logits[0, -1, t_bad_pitch] = -1e9
                
                active_pitches = self._get_active_pitches_at_current_time(current_seq)
                if active_pitches:
                    bad_indices = [tid for tid, p in t_pitches.items() if p in active_pitches]
                    if bad_indices: logits[0, -1, bad_indices] = -1e9
                
                active_voices = self._get_active_voices_at_current_time(current_seq)
                if active_voices:
                    bad_voice_indices = [tid for tid, v in t_voices.items() if v in active_voices]
                    if bad_voice_indices: logits[0, -1, bad_voice_indices] = -1e9
                
                if curr_measure < target_measures - 1:
                    bad_dur_indices = [tid for tid, d in t_durations.items() if d >= 3.0]
                    if bad_dur_indices: logits[0, -1, bad_dur_indices] = -1e9
                
                probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                token = tokenizer.itos.get(idx_next.item(), "[UNK]")
                
                prefix = token[:4]
                if prefix in required_voices and prefix not in found_voices:
                    current_seq.append(token)
                    found_voices.append(prefix)
                elif token == "[FINAL]" or token.startswith("[ROMAN_"):
                    current_seq.append(token)
                
                if len(found_voices) == len(required_voices): break
            except: break
                
        default_pitches = {
            "[V1]": 70, # Soprano
            "[V2]": 63, # Alto
            "[V3]": 56, # Tenor
            "[V4]": 50  # Bass
        }
        for v in required_voices:
            if v not in found_voices:
                active_pitches = self._get_active_pitches_at_current_time(current_seq)
                candidate_pitch = default_pitches.get(v, 60)
                while candidate_pitch in active_pitches:
                    candidate_pitch -= 12
                current_seq.append(f"{v} P{candidate_pitch} D0.5")

    def _refine_sequence(self, full_seq, iters, temp):
        return full_seq # Refinement temporarily disabled for stability # Refinement temporarily disabled for stability

    def _parse_tokens_to_notes(self, tokens):
        raw_notes = []
        current_offset = 0.0
        seen_notes = set() # (voice, offset) 중복 방지
        
        for token in tokens:
            if token.startswith("[TIME_"):
                try: current_offset = float(token[6:-1])
                except: pass
            elif token.startswith("[V"):
                try:
                    parts = token.split()
                    v_num = int(parts[0][2])
                    p = int(parts[1][1:])
                    d = float(parts[2][1:])
                    
                    key = (v_num, round(current_offset, 3))
                    if key not in seen_notes:
                        raw_notes.append({"pitch": p, "duration": d, "offset": current_offset, "voice": v_num})
                        seen_notes.add(key)
                except: pass
                
        # 성부별 오버랩 방지 (단성부 타임라인 정돈)
        voice_groups = {1: [], 2: [], 3: [], 4: []}
        for n in raw_notes:
            voice_groups[n["voice"]].append(n)
            
        clean_raw_notes = []
        for v_num, v_notes in voice_groups.items():
            # 오프셋 기준으로 정렬
            v_notes.sort(key=lambda x: x["offset"])
            for i in range(len(v_notes)):
                curr_note = v_notes[i]
                if i < len(v_notes) - 1:
                    next_note = v_notes[i+1]
                    # 이전 노트의 끝나는 시간이 다음 노트의 시작 시간보다 뒤에 있으면
                    if curr_note["offset"] + curr_note["duration"] > next_note["offset"] + 0.001:
                        # 듀레이션을 겹치지 않게 단축
                        new_dur = next_note["offset"] - curr_note["offset"]
                        curr_note["duration"] = max(0.0, round(new_dur, 3))
                
                # 듀레이션이 유효한 음표만 추가
                if curr_note["duration"] > 0.01:
                    clean_raw_notes.append(curr_note)

        notes = []
        BEATS_PER_MEASURE = 4.0
        # 마디(Barline)를 넘어가는 긴 음표를 두 개의 독립된 음표로 강제 분리(Split)하여 
        # 타이가 아닌 각각 타건(Articulated)되도록 수정합니다.
        for n in clean_raw_notes:
            off = n["offset"]
            rem_dur = n["duration"]
            p = n["pitch"]
            v = n["voice"]
            
            while rem_dur > 0.001:
                measure_idx = int(off // BEATS_PER_MEASURE)
                measure_end = (measure_idx + 1) * BEATS_PER_MEASURE
                dur_in_measure = min(rem_dur, measure_end - off)
                
                notes.append({
                    "pitch": p, 
                    "duration": round(dur_in_measure, 3), 
                    "offset": round(off, 3), 
                    "voice": v
                })
                
                off += dur_in_measure
                rem_dur -= dur_in_measure
                
        return notes

    def _fix_silence_gaps(self, notes, max_time):
        # 무한 지속음 버그의 원인이 된 로직을 비활성화하고, 자연스러운 쉼표(Rests)를 허용합니다.
        # 음표를 늘리는 대신 원본 노트 그대로 반환하여 악보 상에서 빈 공간이 Rests로 올바르게 표시되도록 합니다.
        return notes

    def _evaluate_harmony(self, notes):
        if not notes:
            return 0.0
            
        # 1. 사용자 요구사항 검증: 한 성부 안에 있는 음이 2옥타브(24반음)를 초과하는지 검사
        voice_pitches = {1: [], 2: [], 3: [], 4: []}
        for n in notes:
            voice_pitches[n["voice"]].append(n["pitch"])
            
        for v, pitches in voice_pitches.items():
            if pitches:
                if max(pitches) - min(pitches) > 24:
                    # 2옥타브를 벗어나면 무조건 탈락 (점수 0점 반환하여 재작곡 유도)
                    return 0.0
                    
        offsets = sorted(list(set(n["offset"] for n in notes)))
        if not offsets:
            return 100.0
            
        total_score = 0.0
        evaluation_points = 0
        
        for off in offsets:
            active_pitches = []
            for n in notes:
                if n["offset"] - 0.001 <= off < n["offset"] + n["duration"] - 0.001:
                    active_pitches.append(n["pitch"])
                    
            if len(active_pitches) < 2:
                total_score += 100.0
                evaluation_points += 1
                continue
                
            chord_score = 100.0
            dissonant_count = 0
            strong_dissonant_count = 0
            
            n_pitches = len(active_pitches)
            for i in range(n_pitches):
                for j in range(i + 1, n_pitches):
                    diff = abs(active_pitches[i] - active_pitches[j]) % 12
                    if diff in [1, 11]:  # Minor 2nd, Major 7th
                        strong_dissonant_count += 1
                    elif diff in [2, 10, 6]:  # Major 2nd, Minor 7th, Tritone
                        dissonant_count += 1
                        
            chord_score -= (strong_dissonant_count * 25.0 + dissonant_count * 12.0)
            chord_score = max(0.0, chord_score)
            
            total_score += chord_score
            evaluation_points += 1
            
        average_score = total_score / evaluation_points if evaluation_points > 0 else 100.0
        return average_score

