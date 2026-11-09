import json
import os
import pickle
import random
import subprocess
import sys
import time

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.v5.models import BLOCK_SIZE, BachTransformerConfig, UnifiedTransformerV5


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


def _get_git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


@torch.no_grad()
def evaluate_loader(model, loader, device, use_amp):
    if len(loader) == 0:
        return 0.0
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            _logits, loss = model(x, y)
        total_loss += loss.item()
    return total_loss / len(loader)


def train():
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
        raw_payload = pickle.load(f)

    if isinstance(raw_payload, dict) and "train" in raw_payload:
        train_seqs = raw_payload["train"]
        val_seqs = raw_payload["val"]
        test_seqs = raw_payload.get("test", [])
        split_manifest = raw_payload.get("split_manifest", {})
        print(
            f"Loaded piece-level split dataset -> "
            f"Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}",
            flush=True,
        )
    else:
        sequences = list(raw_payload)
        random.shuffle(sequences)
        split_idx = int(len(sequences) * 0.9)
        train_seqs = sequences[:split_idx]
        val_seqs = sequences[split_idx:]
        test_seqs = []
        split_manifest = {}

    train_dataset = FugueDatasetV5(train_seqs, tokenizer, BLOCK_SIZE)
    val_dataset = FugueDatasetV5(val_seqs, tokenizer, BLOCK_SIZE)
    test_dataset = FugueDatasetV5(test_seqs, tokenizer, BLOCK_SIZE)

    print(
        f"Chunks -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}",
        flush=True,
    )

    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model_scale = os.environ.get("LIKE_BACH_MODEL_SCALE", "small")
    config = BachTransformerConfig(model_scale=model_scale, is_causal=True)
    model = UnifiedTransformerV5(
        vocab_size=tokenizer.vocab_size, device=device, config=config
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model scale={model_scale}, total parameters={num_params:,}", flush=True)

    save_path = "data/processed/v5/fugue_model_v5.pt"
    lr = 3e-4
    epochs = int(os.environ.get("LIKE_BACH_EPOCHS", "10"))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    use_amp = device == "cuda"
    best_val_loss = float("inf")
    history = []
    start_wall = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                _logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if i % 20 == 0:
                print(
                    f"Epoch {epoch}, Step {i}/{len(train_loader)}, Loss: {loss.item():.4f}",
                    flush=True,
                )

        avg_train_loss = total_loss / max(1, len(train_loader))
        avg_val_loss = evaluate_loader(model, val_loader, device, use_amp)
        avg_test_loss = evaluate_loader(model, test_loader, device, use_amp)

        is_best = avg_val_loss <= best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 4),
                "val_loss": round(avg_val_loss, 4),
                "test_loss": round(avg_test_loss, 4),
                "best_checkpoint": is_best,
            }
        )
        print(
            f"=== Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Test Loss: {avg_test_loss:.4f} | "
            f"Best: {is_best} ===",
            flush=True,
        )

    training_log = {
        "git_commit_sha": _get_git_sha(),
        "seed": seed,
        "model_scale": model_scale,
        "parameters": num_params,
        "vocab_size": tokenizer.vocab_size,
        "block_size": BLOCK_SIZE,
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": epochs,
        "best_val_loss": round(best_val_loss, 4),
        "elapsed_seconds": round(time.time() - start_wall, 2),
        "split_manifest": split_manifest,
        "history": history,
    }
    log_path = "data/processed/v5/training_log.json"
    with open(log_path, "w", encoding="utf-8") as lf:
        json.dump(training_log, lf, indent=2, ensure_ascii=False)
    print(f"Saved training log to {log_path}", flush=True)


if __name__ == "__main__":
    train()
