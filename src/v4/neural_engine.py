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
        if os.path.exists(token_path):
            with open(token_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
        else:
            self.tokenizer = BachTokenizer()
            
        self.model = BachTransformer(self.tokenizer.vocab_size, device=self.device).to(self.device)
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                print(f"Neural Engine loaded.")
            except: pass

        # Pre-build a map of token IDs to pitch values for fast logit masking (preventing unisons)
        self.token_pitches = {}
        for token, token_id in self.tokenizer.stoi.items():
            if token.startswith("[V"):
                try:
                    parts = token.split()
                    if len(parts) >= 2 and parts[1].startswith("P"):
                        pitch_val = int(parts[1][1:])
                        self.token_pitches[token_id] = pitch_val
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
                    
                probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                token = self.tokenizer.itos.get(idx_next.item(), "[UNK]")
                
                if token == "[FINAL]":
                    if curr_measure >= target_measures - 1:
                        current_seq.append(token)
                        break
                    else:
                        # 아직 목표 길이에 도달하지 않았는데 끝나려고 하면 무시하고 다음 확률 높은 토큰 선택
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

    def _fill_voices(self, current_seq, target_measures, curr_measure, temperature, required_voices, base_pitch):
        found_voices = []
        for retry in range(15):
            try:
                current_idx = torch.tensor([self.tokenizer.encode(current_seq)], dtype=torch.long, device=self.device)
                logits, _ = self.model(current_idx[:, -BLOCK_SIZE:])
                
                # 목표 마디 근처에서만 FINAL 유도
                if curr_measure >= target_measures - 1:
                    f_idx = self.tokenizer.stoi.get("[FINAL]", -1)
                    if f_idx != -1: logits[0, -1, f_idx] += 2.0
                
                # 유니즌 방지 마스킹
                active_pitches = self._get_active_pitches_at_current_time(current_seq)
                if active_pitches:
                    bad_indices = [tid for tid, p in self.token_pitches.items() if p in active_pitches]
                    if bad_indices:
                        logits[0, -1, bad_indices] = -1e9
                
                probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                token = self.tokenizer.itos.get(idx_next.item(), "[UNK]")
                
                prefix = token[:4]
                if prefix in required_voices and prefix not in found_voices:
                    current_seq.append(token)
                    found_voices.append(prefix)
                elif token == "[FINAL]" or token.startswith("[ROMAN_"):
                    current_seq.append(token)
                
                if len(found_voices) == len(required_voices): break
            except: break
                
        # 누락 성부 강제 할당 (Collision 방지 및 유니즌 방지)
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
        if not notes:
            return notes
            
        step = 0.125
        t = 0.0
        
        voice_notes = {1: [], 2: [], 3: [], 4: []}
        for n in notes:
            voice_notes[n["voice"]].append(n)
            
        for v in [1, 2, 3, 4]:
            voice_notes[v].sort(key=lambda x: x["offset"])
            
        while t < max_time:
            active_count = 0
            for v in [1, 2, 3, 4]:
                for n in voice_notes[v]:
                    if n["offset"] - 0.001 <= t < n["offset"] + n["duration"] - 0.001:
                        active_count += 1
                        break
            
            if active_count == 0:
                # Silence detected at time t! Extend the last played note for each voice.
                for v in [1, 2, 3, 4]:
                    last_note = None
                    for n in voice_notes[v]:
                        if n["offset"] + n["duration"] - 0.001 <= t:
                            last_note = n
                    
                    if last_note is not None:
                        new_dur = t + step - last_note["offset"]
                        last_note["duration"] = round(new_dur, 3)
                        
            t += step
            
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

