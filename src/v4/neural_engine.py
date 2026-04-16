import torch
import pickle
import os
from src.v4.models import BachTransformer, BachTokenizer, BLOCK_SIZE

class NeuralBachEngine:
    def __init__(self, model_path='data/processed/v4/bach_model.pt', tokenizer_path='data/processed/v4/tokenizer.pkl'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
        else:
            raise FileNotFoundError("Tokenizer not found. Please run training first.")
            
        # Load model
        self.model = BachTransformer(self.tokenizer.vocab_size, device=self.device).to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Neural Engine loaded from {model_path}")
        else:
            print("Warning: Model not found. Neural generation will not work until training completes.")

    def generate_response(self, subject_notes, max_tokens=2000, temperature=0.8):
        """
        [V4.0] 4성부 가이드형 생성 (Guided Generation)
        주제(V1)의 각 시점 뒤에 모델이 알토(V2), 테너(V3), 베이스(V4)를 차례로 생성합니다.
        """
        # 1. 조성 분석 및 토큰 매핑
        raw_key = "C" # Default
        key_token = f"[KEY_{raw_key}]"
        
        if key_token not in self.tokenizer.stoi:
            key_token = list(self.tokenizer.stoi.keys())[0] if self.tokenizer.stoi else "[SOS]"

        current_idx = torch.tensor([self.tokenizer.encode([key_token])], dtype=torch.long, device=self.device)
        
        response_notes = []
        
        # 주제의 각 노트를 순회하며 하성 성부들(V2, V3, V4) 생성
        for n in subject_notes:
            off = round(float(n['offset']), 3)
            # 현재 시점의 주제(Soprano) 토큰 추가
            v1_tokens = [f"[TIME_{off}]", f"[V1] P{int(n['pitch'])} D{round(float(n['duration']), 3)}"]
            
            v1_encoded = []
            for t in v1_tokens:
                if t in self.tokenizer.stoi:
                    v1_encoded.append(self.tokenizer.stoi[t])
                else:
                    v1_encoded.append(self.tokenizer.stoi["[PAD]"])

            v1_idx = torch.tensor([v1_encoded], dtype=torch.long, device=self.device)
            current_idx = torch.cat([current_idx, v1_idx], dim=1)
            
            # 모델이 V2, V3, V4를 모두 생성할 때까지 시도
            # 최대 15개 토큰 정도면 3개 성부 생성에 충분함
            for _ in range(15): 
                idx_cond = current_idx[:, -BLOCK_SIZE:]
                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :] / temperature
                
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                token = self.tokenizer.itos.get(idx_next.item(), "[UNK]")
                current_idx = torch.cat([current_idx, idx_next], dim=1)
                
                # V2, V3, V4 성부 정보 추출 (통합 토큰 형식: [V2] Ppitch Ddur)
                if token.startswith("[V") and token != "[V1]":
                    try:
                        parts = token.split()
                        if len(parts) >= 3:
                            v_num = int(parts[0][2])
                            p = int(parts[1][1:])
                            d = float(parts[2][1:])
                            response_notes.append({
                                "pitch": p, 
                                "duration": d, 
                                "offset": off, 
                                "voice": v_num
                            })
                    except:
                        continue
                
                if token.startswith("[TIME_"): # 이미 다음 시간으로 넘어가 버리면 해당 시점 종료
                    break
                        
        return response_notes

if __name__ == "__main__":
    try:
        engine = NeuralBachEngine()
        test_subject = [{"pitch": 60, "duration": 1.0, "offset": 0.0}]
        resp = engine.generate_response(test_subject)
        print(f"Generated {len(resp)} notes.")
    except Exception as e:
        print(f"Test failed: {e}")
