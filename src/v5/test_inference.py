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

from src.v5.neural_engine import HybridFugueEngine

if __name__ == "__main__":
    engine = HybridFugueEngine()
    
    # 테마(주제) 정의
    subject = [
        {"pitch": 60, "duration": 1.0, "offset": 0.0, "voice": 1},
        {"pitch": 62, "duration": 1.0, "offset": 1.0, "voice": 1},
        {"pitch": 64, "duration": 1.0, "offset": 2.0, "voice": 1},
        {"pitch": 65, "duration": 1.0, "offset": 3.0, "voice": 1}
    ]
    
    print("Testing generate_fugue (Target 4 measures)...")
    # 온도값을 낮게 줘서 고의로 제약 조건 충돌을 일으키기 쉽게 만듦 (옵션)
    output = engine.generate_fugue(subject_notes=subject, target_measures=4, temperature=0.01)
    print("=== Generated Output (MIDI Base64) ===")
    print(len(output), "notes generated.")
