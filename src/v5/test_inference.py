import os
import sys
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from src.v5.models import UnifiedTransformerV5, BLOCK_SIZE

class FugueEngineV5:
    def __init__(self, model_path, vocab_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading V5 Fugue Engine on {self.device}...")
        
        # Load vocab
        with open(vocab_path, 'rb') as f:
            data = pickle.load(f)
            self.stoi = data['stoi']
            self.itos = data['itos']
            self.vocab_size = len(self.stoi)
            
        # Load model
        self.model = UnifiedTransformerV5(vocab_size=self.vocab_size, device=self.device).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("Model loaded successfully.")
        
    def encode(self, sequence):
        return [self.stoi.get(t, self.stoi["[UNK]"]) for t in sequence]
        
    def decode(self, indices):
        return [self.itos.get(i, "[UNK]") for i in indices]

    def test_generate(self, start_tokens, max_new_tokens=100, temperature=0.8):
        print(f"Starting generation with tokens: {start_tokens}")
        idx = torch.tensor([self.encode(start_tokens)], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            generated_idx = self.model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
            
        generated_tokens = self.decode(generated_idx[0].tolist())
        return generated_tokens

if __name__ == "__main__":
    engine = FugueEngineV5('data/processed/v5/fugue_model_v5.pt', 'data/processed/v5/fugue_vocab_v5.pkl')
    
    # 테마 시작 토큰 주입 (마디 1, 성부 1, 주제 시작)
    prompt = ["[BAR_1]", "[VOICE_1]", "[SUBJECT_START]", "P60", "D1.0", "P62", "D1.0"]
    
    output = engine.test_generate(prompt, max_new_tokens=50, temperature=0.8)
    print("=== Generated Output ===")
    print(" ".join(output))
