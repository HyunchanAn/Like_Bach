import os
import pickle
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.v5.models import BLOCK_SIZE, UnifiedTransformerV5


class FugueDatasetV5(Dataset):
    def __init__(self, sequences, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = []

        pad_idx = tokenizer.stoi.get("[PAD]", 0)

        for seq in sequences:
            encoded = tokenizer.encode(seq)

            if len(encoded) <= block_size:
                pad_len = block_size + 1 - len(encoded)
                chunk = encoded + [pad_idx] * pad_len
                self.data.append(chunk)
            else:
                stride = block_size // 4
                for i in range(0, len(encoded) - block_size, stride):
                    chunk = encoded[i : i + block_size + 1]
                    if len(chunk) == block_size + 1:
                        self.data.append(chunk)

                if (len(encoded) - block_size) % stride != 0:
                    chunk = encoded[-block_size - 1 :]
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
    print(f"Using device: {device}", flush=True)

    vocab_path = "data/processed/v5/fugue_vocab_v5.pkl"
    data_path = "data/processed/v5/fugue_dataset_v5.pkl"

    if not os.path.exists(vocab_path) or not os.path.exists(data_path):
        print("Data files not found. Run preprocess_fugue.py first.")
        return

    with open(vocab_path, "rb") as f:
        vocab_data = pickle.load(f)

    class TokenizerStub:
        def __init__(self, d):
            self.stoi = d["stoi"]
            self.itos = d["itos"]
            self.vocab_size = len(self.stoi)

        def encode(self, seq):
            return [self.stoi.get(t, self.stoi["[UNK]"]) for t in seq]

    tokenizer = TokenizerStub(vocab_data)

    with open(data_path, "rb") as f:
        sequences = pickle.load(f)

    print(f"Loaded {len(sequences)} sequences.", flush=True)

    random.shuffle(sequences)
    split_idx = int(len(sequences) * 0.9)
    train_seqs = sequences[:split_idx]
    val_seqs = sequences[split_idx:]

    print(f"Train sequences: {len(train_seqs)}, Val sequences: {len(val_seqs)}", flush=True)

    train_dataset = FugueDatasetV5(train_seqs, tokenizer, BLOCK_SIZE)
    val_dataset = FugueDatasetV5(val_seqs, tokenizer, BLOCK_SIZE)

    print(f"Train chunks: {len(train_dataset)}, Val chunks: {len(val_dataset)}", flush=True)

    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = UnifiedTransformerV5(
        vocab_size=tokenizer.vocab_size, device=device, is_causal=True
    ).to(device)

    save_path = "data/processed/v5/fugue_model_v5.pt"
    if os.path.exists(save_path):
        print(f"Loading existing model weights from {save_path}...", flush=True)
        try:
            model.load_state_dict(torch.load(save_path, weights_only=True))
        except Exception as e:
            print(f"Failed to load weights, starting fresh: {e}", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    epochs = 10
    use_amp = device == "cuda"

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits, loss = model(x, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:
                print(
                    f"Epoch {epoch}, Step {i}/{len(train_loader)}, Loss: {loss.item():.4f}",
                    flush=True,
                )

        print(
            f"=== Epoch {epoch} Average Train Loss: {total_loss / len(train_loader):.4f} ===",
            flush=True,
        )

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits, loss = model(x, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        print(f"=== Epoch {epoch} Val Loss: {avg_val_loss:.4f} ===", flush=True)

        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}", flush=True)


if __name__ == "__main__":
    train()
