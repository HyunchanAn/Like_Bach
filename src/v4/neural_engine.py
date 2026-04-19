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
        
        # Load tokenizer
        token_path = tokenizer_path if os.path.exists(tokenizer_path) else 'data/processed/v4/tokenizer.pkl'
        if os.path.exists(token_path):
            with open(token_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
        else:
            # Fallback for development if file not present yet
            self.tokenizer = BachTokenizer()
            print("Warning: Tokenizer dummy initialized for algorithm development.")
            
        # Load model
        self.model = BachTransformer(self.tokenizer.vocab_size, device=self.device).to(self.device)
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                print(f"Neural Engine loaded from {model_path}")
            except:
                print("Warning: Model architecture mismatch or corrupt file.")
        else:
            print("Note: Model file not found. Algorithm implemented, waiting for training.")

    def generate_response(self, subject_notes, target_measures=16, temperature=0.8, refine_iters=3):
        """
        [V4.5 Advanced] 고도화된 4성부 생성 엔진
        - Measure Countdown 제어
        - Functional Harmony (Roman Numeral) 반영
        - Iterative Refinement (Gibbs Sampling)을 통한 화성 정밀도 향상
        """
        # 1. 초기 컨텍스트 설정 (조성, 박자, 전체 마디 수)
        raw_key = "C" 
        key_token = f"[KEY_{raw_key}]"
        ts_token = "[TS_4/4]"
        
        current_seq = [key_token, ts_token]
        
        # 2. 1차 생성 (Guided Generation with measure awareness)
        last_measure_idx = -1
        
        # 주제(V1) 정보를 기반으로 순차 생성
        for n in subject_notes:
            off = round(float(n['offset']), 3)
            curr_measure = int(off // 4)
            
            # 마디 변경 시 카운트다운 토큰 주입
            if curr_measure > last_measure_idx:
                remain = max(0, target_measures - curr_measure)
                current_seq.append(f"[REMAIN_{remain}]")
                last_measure_idx = curr_measure
            
            # 현재 시점의 주제(Soprano) 입력
            current_seq.append(f"[TIME_{off}]")
            current_seq.append(f"[V1] P{int(n['pitch'])} D{round(float(n['duration']), 3)}")
            
            # 하성 성부(V2, V3, V4) 및 화성 기호(Roman) 생성을 위한 샘플링
            current_idx = torch.tensor([self.tokenizer.encode(current_seq)], dtype=torch.long, device=self.device)
            
            # 각 타임스텝에서 하성 성부 3개가 다 나올 때까지 최대 샘플링 시도
            for _ in range(12): 
                idx_cond = current_idx[:, -BLOCK_SIZE:]
                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :] / temperature
                
                # [Optimization] 만약 마지막에 가까워지면 [FINAL] 가중치 부여
                if curr_measure >= target_measures - 1:
                    final_idx = self.tokenizer.stoi.get("[FINAL]", -1)
                    if final_idx != -1: logits[0, final_idx] += 2.0 

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                token = self.tokenizer.itos.get(idx_next.item(), "[UNK]")
                
                current_seq.append(token)
                current_idx = torch.cat([current_idx, idx_next], dim=1)
                
                if token.startswith("[TIME_") or token == "[FINAL]":
                    break
        
        # 3. Iterative Refinement (Gibbs Sampling 스타일)
        # 생성된 결과에서 하성 성부들만 선택적으로 다시 샘플링하여 화성적 정밀도 상향
        if refine_iters > 0:
            current_seq = self._refine_sequence(current_seq, refine_iters, temperature)

        # 4. 최종 결과 파싱
        response_notes = self._parse_tokens_to_notes(current_seq)
        return response_notes

    def _refine_sequence(self, full_seq, iters, temp):
        """
        Gibbs Sampling: 확률이 낮은 성부나 특정 구간을 주변 문맥에 맞춰 다시 생성
        """
        for _ in range(iters):
            # 시퀀스 내에서 [V2], [V3], [V4] 토큰 위치 찾기
            v_indices = [i for i, t in enumerate(full_seq) if any(t.startswith(prefix) for prefix in ["[V2]", "[V3]", "[V4]", "[ROMAN_"])]
            
            if not v_indices: break
            
            # 무작위로 일부 성부 토큰을 마스킹하고 다시 샘플링 (간이 Gibbs)
            sample_count = max(1, len(v_indices) // 5)
            targets = random.sample(v_indices, sample_count)
            targets.sort()
            
            for idx in targets:
                # 해당 토큰 이전까지의 컨텍스트
                prefix = full_seq[:idx]
                prefix_idx = torch.tensor([self.tokenizer.encode(prefix)], dtype=torch.long, device=self.device)
                
                logits, _ = self.model(prefix_idx[:, -BLOCK_SIZE:])
                logits = logits[:, -1, :] / temp
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                full_seq[idx] = self.tokenizer.itos.get(idx_next.item(), "[UNK]")
                
        return full_seq

    def _parse_tokens_to_notes(self, tokens):
        notes = []
        current_offset = 0.0
        for token in tokens:
            if token.startswith("[TIME_"):
                try: current_offset = float(token[6:-1])
                except: pass
            elif token.startswith("[V") and not token.startswith("[V1]"):
                try:
                    parts = token.split()
                    v_num = int(parts[0][2])
                    p = int(parts[1][1:])
                    d = float(parts[2][1:])
                    notes.append({"pitch": p, "duration": d, "offset": current_offset, "voice": v_num})
                except: pass
        return notes

if __name__ == "__main__":
    try:
        engine = NeuralBachEngine()
        # Dummy subject for algorithm verification
        test_subject = [{"pitch": 67, "duration": 1.0, "offset": 0.0}, {"pitch": 65, "duration": 1.0, "offset": 1.0}]
        results = engine.generate_response(test_subject, target_measures=8)
        print(f"Algorithm executed. Generated {len(results)} notes.")
    except Exception as e:
        print(f"Algorithm check: {e}")
