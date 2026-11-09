import os
import pickle
import sys

sys.path.append(os.getcwd())

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.v4.models import BLOCK_SIZE, BachTokenizer, BachTransformer

# --- Configuration for RTX 5080 (16GB VRAM) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
MAX_ITERS = 3000
EVAL_INTERVAL = 500


class BachDataset(Dataset):
    def __init__(self, data_path, block_size, device="cuda"):
        with open(data_path, "rb") as f:
            sequences = pickle.load(f)

        self.tokenizer = BachTokenizer(sequences)
        print("Flattening and pre-tokenizing dataset for maximum speed...", flush=True)

        all_ids = []
        for seq in tqdm(sequences):
            encoded = (
                [self.tokenizer.stoi["[SOS]"]]
                + self.tokenizer.encode(seq)
                + [self.tokenizer.stoi["[EOS]"]]
            )
            all_ids.extend(encoded)

        self.data_tensor = torch.tensor(all_ids, dtype=torch.long).to(device)
        self.block_size = block_size
        print(
            f"Total tokens in dataset: {len(self.data_tensor)} (VRAM-Direct Loaded)",
            flush=True,
        )

    def __len__(self):
        return 200000

    def __getitem__(self, idx):
        return None


if __name__ == "__main__":
    data_path = "data/processed/v4/bach_tokens_advanced.pkl"
    dataset = BachDataset(data_path, BLOCK_SIZE)
    vocab_size = dataset.tokenizer.vocab_size
    print(
        f"Vocab size: {vocab_size}, Tokens: {len(dataset.data_tensor)}",
        flush=True,
    )

    m = BachTransformer(
        vocab_size, device=DEVICE, ignore_index=dataset.tokenizer.stoi["[PAD]"]
    ).to(DEVICE)

    model_save_path = "data/processed/v4/bach_model.pt"
    if os.path.exists(model_save_path):
        print(f"Loading existing model from {model_save_path} for resumption...", flush=True)
        try:
            m.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting from scratch.", flush=True)

    with open("data/processed/v4/tokenizer.pkl", "wb") as f:
        pickle.dump(dataset.tokenizer, f)

    optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_ITERS, eta_min=1e-5
    )

    def get_batch():
        ix = torch.randint(len(dataset.data_tensor) - BLOCK_SIZE, (BATCH_SIZE,), device=DEVICE)
        offsets = torch.arange(BLOCK_SIZE, device=DEVICE)
        x = dataset.data_tensor[ix.unsqueeze(1) + offsets]
        y = dataset.data_tensor[ix.unsqueeze(1) + offsets + 1]
        return x, y

    print(f"Starting Hyper-Fast training on {DEVICE}...", flush=True)
    use_amp = DEVICE == "cuda"
    for step in range(MAX_ITERS):
        x, y = get_batch()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits, loss = m(x, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step < 10 or step % 200 == 0:
            print(f"step {step}/{MAX_ITERS}: loss {loss.item():.4f}", flush=True)

        if step > 0 and step % EVAL_INTERVAL == 0:
            torch.save(m.state_dict(), "data/processed/v4/bach_model.pt")
            print(f"Model saved at step {step}", flush=True)

    torch.save(m.state_dict(), "data/processed/v4/bach_model.pt")
    print("Training Complete. Final model saved.", flush=True)
