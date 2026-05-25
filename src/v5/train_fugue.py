import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from src.v5.models import FugueTransformerV5, BLOCK_SIZE
import torch.optim as optim

class FugueDatasetV5(Dataset):
    def __init__(self, sequences, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = []
        
        for seq in sequences:
            # Encode tokens to indices
            encoded = tokenizer.encode(seq)
            # Create sliding windows
            for i in range(0, len(encoded) - block_size):
                chunk = encoded[i:i+block_size+1]
                if len(chunk) == block_size + 1:
                    self.data.append(chunk)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Data
    vocab_path = 'data/processed/v5/fugue_vocab_v5.pkl'
    data_path = 'data/processed/v5/fugue_dataset_v5.pkl'
    
    if not os.path.exists(vocab_path) or not os.path.exists(data_path):
        print("Data files not found. Run preprocess_fugue.py first.")
        return
        
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
        
    # Reconstruct tokenizer
    class TokenizerStub:
        def __init__(self, d):
            self.stoi = d['stoi']
            self.itos = d['itos']
            self.vocab_size = len(self.stoi)
        def encode(self, seq):
            return [self.stoi.get(t, self.stoi["[UNK]"]) for t in seq]
            
    tokenizer = TokenizerStub(vocab_data)
    
    with open(data_path, 'rb') as f:
        sequences = pickle.load(f)
        
    print(f"Loaded {len(sequences)} sequences.")
    
    # 2. Split train/val
    import random
    random.shuffle(sequences)
    split_idx = int(len(sequences) * 0.9)
    train_seqs = sequences[:split_idx]
    val_seqs = sequences[split_idx:]
    
    print(f"Train sequences: {len(train_seqs)}, Val sequences: {len(val_seqs)}")
    
    train_dataset = FugueDatasetV5(train_seqs, tokenizer, BLOCK_SIZE)
    val_dataset = FugueDatasetV5(val_seqs, tokenizer, BLOCK_SIZE)
    
    print(f"Train chunks: {len(train_dataset)}, Val chunks: {len(val_dataset)}")
    
    batch_size = 4 # Reduced batch size to prevent CUDA OOM with 4096 BLOCK_SIZE on 16GB VRAM
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Model
    model = FugueTransformerV5(vocab_size=tokenizer.vocab_size, device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    
    epochs = 10
    
    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 50 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")
                
        print(f"=== Epoch {epoch} Average Train Loss: {total_loss/len(train_loader):.4f} ===")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, loss = model(x, y)
                val_loss += loss.item()
        
        print(f"=== Epoch {epoch} Val Loss: {val_loss/len(val_loader):.4f} ===")
        
        # Save model
        save_path = 'data/processed/v5/fugue_model_v5.pt'
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
