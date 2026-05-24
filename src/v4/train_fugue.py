import os
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from models import BachTransformer, BachTokenizer, BLOCK_SIZE

def train_fugue_model():
    print("Loading Fugue tokens...")
    try:
        with open('data/processed/v4/fugue_tokens.pkl', 'rb') as f:
            all_sequences = pickle.load(f)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Loaded {len(all_sequences)} sequences.")

    # Initialize Tokenizer and save it
    tokenizer = BachTokenizer(all_sequences)
    os.makedirs('models/v4', exist_ok=True)
    with open('models/v4/fugue_vocab.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Prepare data tensors
    encoded_data = []
    for seq in all_sequences:
        encoded_data.extend(tokenizer.encode(seq))
    
    data = torch.tensor(encoded_data, dtype=torch.long)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}, Total tokens: {len(data)}")

    # Hyperparameters for training
    batch_size = 16
    learning_rate = 3e-4
    max_iters = 5000  # Prototyping duration
    eval_interval = 500

    def get_batch():
        ix = torch.randint(len(data) - BLOCK_SIZE, (batch_size,))
        x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
        return x.to(device), y.to(device)

    model = BachTransformer(vocab_size=vocab_size, device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("Starting Phase 2 Fugue Training...")
    
    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            model.eval()
            with torch.no_grad():
                X, Y = get_batch()
                _, loss = model(X, Y)
                print(f"Iter {it}: Loss {loss.item():.4f}")
            model.train()

        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Training complete. Saving fugue_model.pt...")
    torch.save(model.state_dict(), 'models/v4/fugue_model.pt')
    print("Saved successfully.")

if __name__ == '__main__':
    train_fugue_model()
