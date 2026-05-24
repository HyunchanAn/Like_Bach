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
        for token, token_id in self.tokenizer.stoi.items():
            if token.startswith("[V"):
                self.token_voices[token_id] = token[:4]
                try:
                    parts = token.split()
                    if len(parts) >= 2 and parts[1].startswith("P"):
                        pitch_val = int(parts[1][1:])
                        self.token_pitches[token_id] = pitch_val
                    if len(parts) >= 3 and parts[2].startswith("D"):
                        dur_val = float(parts[2][1:])
                        self.token_durations[token_id] = dur_val
                except:
                    pass

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

    def generate_fugue(self, subject_notes, target_measures=16, temperature=0.8, refine_iters=3):
        best_notes = None
        best_score = -1.0
        max_attempts = 5
        threshold = 90.0
        
        for attempt in range(max_attempts):
            notes = self._generate_single_fugue_response(subject_notes, target_measures, temperature, refine_iters)
            score = self._evaluate_harmony(notes)
            
            if score >= threshold:
                return notes
                
            if score > best_score:
                best_score = score
                best_notes = notes
                
        return best_notes

    def _generate_single_fugue_response(self, subject_notes, target_measures=16, temperature=0.8, refine_iters=3):
        raw_key = "C" 
        current_seq = [f"[KEY_{raw_key}]", "[TS_4/4]", "[SUBJECT]"]
        last_measure_idx = -1
        
        if not subject_notes:
            return []
            
        subject_end_off = max([float(n['offset']) + float(n['duration']) for n in subject_notes])
        answer_start_off = float(int((subject_end_off + 3.99) // 4) * 4) 
        
        forced_notes = []
        for n in subject_notes:
            off = round(float(n['offset']), 3)
            forced_notes.append({"offset": off, "pitch": int(n['pitch']), "duration": round(float(n['duration']), 3), "voice": "[V1]"})
            ans_off = round(off + answer_start_off, 3)
            ans_pitch = int(n['pitch']) - 7
            forced_notes.append({"offset": ans_off, "pitch": ans_pitch, "duration": round(float(n['duration']), 3), "voice": "[V4]"})
            
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
            elif off == answer_start_off and v == "[V4]":
                current_seq.append("[ANSWER]")
                
            current_seq.append(f"{v} P{p} D{d}")
            
            # Answer가 연주될 때 AI가 빈 성부(Countersubject)를 채우도록 유도
            if v == "[V4]":
                self._fill_voices(current_seq, target_measures, curr_measure, temperature, ["[V1]", "[V2]", "[V3]"], p, use_fugue_model=True)

        current_seq.append("[EPISODE]")
                
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
                
                active_pitches = self._get_active_pitches_at_current_time(current_seq)
                if active_pitches:
                    bad_indices = [tid for tid, p in self.token_pitches.items() if p in active_pitches]
                    if bad_indices: logits[0, -1, bad_indices] = -1e9
                
                active_voices = self._get_active_voices_at_current_time(current_seq)
                if active_voices:
                    bad_voice_indices = [tid for tid, voice in self.token_voices.items() if voice in active_voices]
                    if bad_voice_indices: logits[0, -1, bad_voice_indices] = -1e9
                
                if curr_measure < target_measures - 1:
                    bad_dur_indices = [tid for tid, dur in self.token_durations.items() if abs(dur - 4.0) < 0.001]
                    if bad_dur_indices: logits[0, -1, bad_dur_indices] = -1e9
                    
                probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                token = self.fugue_tokenizer.itos.get(idx_next.item(), "[UNK]")
                
                if token == "[FINAL]":
                    if curr_measure >= target_measures:
                        current_seq.append(token)
                        break
                    else:
                        probs[0, idx_next.item()] = 0.0
                        idx_next = torch.multinomial(probs, num_samples=1)
                        token = self.fugue_tokenizer.itos.get(idx_next.item(), "[UNK]")
 
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
                
                # 온음표(Whole note, D4.0) 억제 (종지부 제외)
                if curr_measure < target_measures - 1:
                    bad_dur_indices = [tid for tid, d in self.token_durations.items() if abs(d - 4.0) < 0.001]
                    if bad_dur_indices:
                        logits[0, -1, bad_dur_indices] = -1e9
                    
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

        # 누락 성부 강제 할당 (Collision 방지 및 유니즌 방지)
        for v in required_voices:
            if v not in found_voices:
                active_pitches = self._get_active_pitches_at_current_time(current_seq)
                candidate_pitch = base_pitch - (12 if v != "[V1]" else 0)
                while candidate_pitch in active_pitches:
                    candidate_pitch -= 12
                current_seq.append(f"{v} P{candidate_pitch} D0.5")

    def _fill_voices(self, current_seq, target_measures, curr_measure, temperature, required_voices, base_pitch, use_fugue_model=False):
        found_voices = []
        model = self.fugue_model if use_fugue_model else self.model
        tokenizer = self.fugue_tokenizer if use_fugue_model else self.tokenizer
        
        for retry in range(15):
            try:
                current_idx = torch.tensor([tokenizer.encode(current_seq)], dtype=torch.long, device=self.device)
                logits, _ = model(current_idx[:, -BLOCK_SIZE:])
                
                if curr_measure >= target_measures:
                    f_idx = tokenizer.stoi.get("[FINAL]", -1)
                    if f_idx != -1: logits[0, -1, f_idx] += 2.0
                
                active_pitches = self._get_active_pitches_at_current_time(current_seq)
                if active_pitches:
                    bad_indices = [tid for tid, p in self.token_pitches.items() if p in active_pitches]
                    if bad_indices: logits[0, -1, bad_indices] = -1e9
                
                active_voices = self._get_active_voices_at_current_time(current_seq)
                if active_voices:
                    bad_voice_indices = [tid for tid, v in self.token_voices.items() if v in active_voices]
                    if bad_voice_indices: logits[0, -1, bad_voice_indices] = -1e9
                
                if curr_measure < target_measures - 1:
                    bad_dur_indices = [tid for tid, d in self.token_durations.items() if abs(d - 4.0) < 0.001]
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
                
        for v in required_voices:
            if v not in found_voices:
                active_pitches = self._get_active_pitches_at_current_time(current_seq)
                candidate_pitch = base_pitch - (12 if v != "[V1]" else 0)
                while candidate_pitch in active_pitches:
                    candidate_pitch -= 12
                current_seq.append(f"{v} P{candidate_pitch} D0.5")

    def _refine_sequence(self, full_seq, iters, temp):
        return full_seq # Refinement temporarily disabled for stability # Refinement temporarily disabled for stability

    def _parse_tokens_to_notes(self, tokens):
        notes = []
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
                        notes.append({"pitch": p, "duration": d, "offset": current_offset, "voice": v_num})
                        seen_notes.add(key)
                except: pass
        return notes

    def _fix_silence_gaps(self, notes, max_time):
        # 무한 지속음 버그의 원인이 된 로직을 비활성화하고, 자연스러운 쉼표(Rests)를 허용합니다.
        # 음표를 늘리는 대신 원본 노트 그대로 반환하여 악보 상에서 빈 공간이 Rests로 올바르게 표시되도록 합니다.
        return notes

    def _evaluate_harmony(self, notes):
        if not notes:
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

