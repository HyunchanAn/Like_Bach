import torch
import pickle
import os
from src.v3.models import BachTransformer, BachTokenizer, BLOCK_SIZE

class NeuralBachEngine:
    def __init__(self, model_path='data/processed/v3/bach_model.pt', tokenizer_path='data/processed/v3/tokenizer.pkl'):
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
        [NEW] 媛?대뱶???앹꽦(Guided Generation)
        二쇱젣(V1)??媛??쒖젏 ?ㅼ뿉 紐⑤뜽????꾩꽑??V2)???앹꽦?섎룄濡??좊룄?⑸땲??
        """
        # 1. 議곗꽦 遺꾩꽍 (?꾩옱??湲곕낯 C Major, 異뷀썑 ?낅젰媛믪뿉 ?곕씪 留ㅽ븨)
        # music21 Key -> Vocab Key Mapping (CMaj -> [KEY_C], Am -> [KEY_Am])
        raw_key = "C" # Default
        key_token = f"[KEY_{raw_key}]"
        
        if key_token not in self.tokenizer.stoi:
            # Fallback to similar or SOS
            key_token = list(self.tokenizer.stoi.keys())[0] if self.tokenizer.stoi else "[SOS]"

        current_idx = torch.tensor([self.tokenizer.encode([key_token])], dtype=torch.long, device=self.device)
        
        response_notes = []
        
        # 二쇱젣??媛??명듃瑜??쒗쉶?섎ŉ ??꾩꽑???앹꽦
        for n in subject_notes:
            off = round(float(n['offset']), 3)
            # ?꾩옱 ?쒖젏??二쇱젣 ?좏겙 異붽?
            v1_tokens = [f"[TIME_{off}]", f"[V1] P{int(n['pitch'])} D{float(n['duration'])}"]
            
            # ?좏겙???댄쐶 ?ъ쟾???덈뒗吏 ?뺤씤 (?놁쑝硫??⑤뵫 ?뱀? 嫄대꼫?)
            v1_encoded = []
            for t in v1_tokens:
                if t in self.tokenizer.stoi:
                    v1_encoded.append(self.tokenizer.stoi[t])
                else:
                    # TIME_X.X 媛 ?놁쓣 寃쎌슦 媛??媛源뚯슫 媛??뱀? 蹂닿컙 ?꾩슂?섎굹 ?쇰떒 PAD
                    v1_encoded.append(self.tokenizer.stoi["[PAD]"])

            v1_idx = torch.tensor([v1_encoded], dtype=torch.long, device=self.device)
            current_idx = torch.cat([current_idx, v1_idx], dim=1)
            
            # 紐⑤뜽??[V2]瑜??앹꽦???뚭퉴吏 ?뱀? 理쒕? ?좏겙源뚯? ?앹꽦
            # 2?깅??대?濡?[V2] P_ D_ ?뺤떇??湲곕???            for _ in range(3): # [V2], P_ D_ ?앹꽦 ?쒕룄
                idx_cond = current_idx[:, -BLOCK_SIZE:]
                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :] / temperature
                
                # [PAD], [SOS], [EOS], [V1], [TIME_...] ?좏겙 ?쒖쇅 (?붿꽦 ?뚰몴 ?앹꽦???꾪빐)
                # logits[:, self.tokenizer.stoi["[PAD]"]] = -1e9 # ?꾩슂??媛뺤젣 ?쒗븳
                
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
                
                if token.startswith("[TIME_"): # ?대? ?ㅼ쓬 ?쒓컙?쇰줈 ?섏뼱媛 踰꾨━硫?醫낅즺
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
