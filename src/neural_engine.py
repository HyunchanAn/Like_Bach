import torch
import pickle
import os
from src.models import BachTransformer, BachTokenizer, BLOCK_SIZE

class NeuralBachEngine:
    def __init__(self, model_path='data/processed/bach_model.pt', tokenizer_path='data/processed/tokenizer.pkl'):
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

    def generate_response(self, subject_notes, max_tokens=1000, temperature=0.8):
        """
        [NEW] 가이드형 생성(Guided Generation)
        주제(V1)의 각 시점 뒤에 모델이 대위선율(V2)을 생성하도록 유도합니다.
        """
        # 1. 조성 분석 (현재는 기본 C Major, 추후 입력값에 따라 매핑)
        # music21 Key -> Vocab Key Mapping (CMaj -> [KEY_C], Am -> [KEY_Am])
        raw_key = "C" # Default
        key_token = f"[KEY_{raw_key}]"
        
        if key_token not in self.tokenizer.stoi:
            # Fallback to similar or SOS
            key_token = list(self.tokenizer.stoi.keys())[0] if self.tokenizer.stoi else "[SOS]"

        current_idx = torch.tensor([self.tokenizer.encode([key_token])], dtype=torch.long, device=self.device)
        
        response_notes = []
        
        # 주제의 각 노트를 순회하며 대위선율 생성
        for n in subject_notes:
            off = round(float(n['offset']), 3)
            # 현재 시점의 주제 토큰 추가
            v1_tokens = [f"[TIME_{off}]", f"[V1] P{int(n['pitch'])} D{float(n['duration'])}"]
            
            # 토큰이 어휘 사전에 있는지 확인 (없으면 패딩 혹은 건너뜀)
            v1_encoded = []
            for t in v1_tokens:
                if t in self.tokenizer.stoi:
                    v1_encoded.append(self.tokenizer.stoi[t])
                else:
                    # TIME_X.X 가 없을 경우 가장 가까운 값 혹은 보간 필요하나 일단 PAD
                    v1_encoded.append(self.tokenizer.stoi["[PAD]"])

            v1_idx = torch.tensor([v1_encoded], dtype=torch.long, device=self.device)
            current_idx = torch.cat([current_idx, v1_idx], dim=1)
            
            # 모델이 [V2]를 생성할 때까지 혹은 최대 토큰까지 생성
            # 2성부이므로 [V2] P_ D_ 형식을 기대함
            for _ in range(3): # [V2], P_ D_ 생성 시도
                idx_cond = current_idx[:, -BLOCK_SIZE:]
                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :] / temperature
                
                # [PAD], [SOS], [EOS], [V1], [TIME_...] 토큰 제외 (화성 음표 생성을 위해)
                # logits[:, self.tokenizer.stoi["[PAD]"]] = -1e9 # 필요시 강제 제한
                
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                token = self.tokenizer.itos.get(idx_next.item(), "[UNK]")
                current_idx = torch.cat([current_idx, idx_next], dim=1)
                
                if token == "[V2]":
                    continue 
                
                if token.startswith("P"):
                    try:
                        parts = token.split()
                        p = int(parts[0][1:])
                        d = float(parts[1][1:])
                        response_notes.append({"pitch": p, "duration": d, "offset": off})
                        break 
                    except:
                        continue
                
                if token.startswith("[TIME_"): # 이미 다음 시간으로 넘어가 버리면 종료
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
