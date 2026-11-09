from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class BachTransformerConfig:
    model_scale: str = "v5_pro"
    block_size: int = 4096
    dropout: float = 0.0625
    ignore_index: int = -1
    is_causal: bool = True

    def __post_init__(self):
        if self.model_scale == "small":
            self.n_embd = 384
            self.n_head = 6
            self.n_layer = 6
        elif self.model_scale == "base":
            self.n_embd = 768
            self.n_head = 12
            self.n_layer = 12
        elif self.model_scale == "large":
            self.n_embd = 1024
            self.n_head = 16
            self.n_layer = 24
        else:
            self.n_embd = 768
            self.n_head = 12
            self.n_layer = 16  # Default v5-Pro 113M config


BLOCK_SIZE = 4096
DROPOUT = 0.0625


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) module for attention operations."""

    def __init__(self, dim=64, max_seq_len=4096, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=True)
        self.register_buffer("sin_cached", emb.sin(), persistent=True)

    def forward(self, seq_len, device):
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)


def rotate_half(x):
    """Split input in half and rotate for RoPE calculation."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Head(nn.Module):
    """Single causal/non-causal self-attention head with RoPE."""

    def __init__(self, config: BachTransformerConfig, head_size: int):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.is_causal = config.is_causal
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos=None, sin=None):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        if cos is not None and sin is not None:
            cos_u = cos[:T].to(dtype=q.dtype).unsqueeze(0)
            sin_u = sin[:T].to(dtype=q.dtype).unsqueeze(0)
            q = (q * cos_u) + (rotate_half(q) * sin_u)
            k = (k * cos_u) + (rotate_half(k) * sin_u)

        dropout_p = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q.unsqueeze(1),
            k.unsqueeze(1),
            v.unsqueeze(1),
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=self.is_causal,
        ).squeeze(1)
        return out


class MultiHeadAttention(nn.Module):
    """Multi-head attention block with 4D batched FlashAttention dispatch."""

    def __init__(self, config: BachTransformerConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos=None, sin=None):
        B, T, C = x.shape
        q = torch.stack([h.query(x) for h in self.heads], dim=1)
        k = torch.stack([h.key(x) for h in self.heads], dim=1)
        v = torch.stack([h.value(x) for h in self.heads], dim=1)

        if cos is not None and sin is not None:
            cos_u = cos[:T].to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)
            sin_u = sin[:T].to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)
            q = (q * cos_u) + (rotate_half(q) * sin_u)
            k = (k * cos_u) + (rotate_half(k) * sin_u)

        dropout_p = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=self.heads[0].is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Standard MLP feed-forward block."""

    def __init__(self, config: BachTransformerConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Standard Transformer block with LayerNorm and residual connections."""

    def __init__(self, config: BachTransformerConfig):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x, cos=None, sin=None):
        x = x + self.sa(self.ln1(x), cos=cos, sin=sin)
        x = x + self.ffwd(self.ln2(x))
        return x


class UnifiedTransformerV5(nn.Module):
    """Unified counterpoint Transformer neural network compatible with V5 checkpoints."""

    def __init__(
        self,
        vocab_size,
        device="cuda",
        config: BachTransformerConfig = None,
        ignore_index=-1,
        is_causal=True,
    ):
        super().__init__()
        self.device = device
        self.config = (
            config
            if config is not None
            else BachTransformerConfig(ignore_index=ignore_index, is_causal=is_causal)
        )
        self.ignore_index = self.config.ignore_index
        self.is_causal = self.config.is_causal

        self.token_embedding_table = nn.Embedding(vocab_size, self.config.n_embd)
        self.pitch_class_embedding_table = nn.Embedding(13, self.config.n_embd)
        self.octave_embedding_table = nn.Embedding(11, self.config.n_embd)
        self.voice_embedding_table = nn.Embedding(6, self.config.n_embd)

        self.rope = RotaryEmbedding(dim=64, max_seq_len=self.config.block_size)
        self.blocks = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)])
        self.ln_f = nn.LayerNorm(self.config.n_embd)
        self.lm_head = nn.Linear(self.config.n_embd, vocab_size)

        pitch_class_map = torch.zeros(vocab_size, dtype=torch.long)
        octave_map = torch.zeros(vocab_size, dtype=torch.long)
        voice_map = torch.zeros(vocab_size, dtype=torch.long)

        pitch_class_map.fill_(12)
        octave_map.fill_(10)
        voice_map.fill_(0)

        import os
        import pickle

        vocab_path = "data/processed/v5/fugue_vocab_v5.pkl"
        if os.path.exists(vocab_path):
            with open(vocab_path, "rb") as f:
                vocab_data = pickle.load(f)
                itos = vocab_data["itos"]
            for i in range(vocab_size):
                token = itos.get(i, "[UNK]")
                if token.startswith("P"):
                    try:
                        p = int(token[1:])
                        pitch_class_map[i] = p % 12
                        octave_map[i] = p // 12
                    except Exception:
                        pass
                elif token.startswith("[VOICE_"):
                    try:
                        v = int(token[7])
                        voice_map[i] = v
                    except Exception:
                        pass

        self.register_buffer("pitch_class_map", pitch_class_map, persistent=False)
        self.register_buffer("octave_map", octave_map, persistent=False)
        self.register_buffer("voice_map", voice_map, persistent=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pitch_class_idx = self.pitch_class_map[idx]
        octave_idx = self.octave_map[idx]
        voice_idx = self.voice_map[idx]

        pitch_class_emb = self.pitch_class_embedding_table(pitch_class_idx)
        octave_emb = self.octave_embedding_table(octave_idx)
        voice_emb = self.voice_embedding_table(voice_idx)

        x = tok_emb + pitch_class_emb + octave_emb + voice_emb

        cos, sin = self.rope(T, x.device)

        for block in self.blocks:
            x = block(x, cos=cos, sin=sin)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class BachTokenizerV5:
    """Fallback tokenizer for V5 fugue engine when vocab pickle is unavailable."""

    def __init__(self):
        tokens = [
            "[PAD]",
            "[UNK]",
            "[REST]",
            "[FINAL]",
            "[SUBJECT]",
            "[ANSWER]",
            "[EPISODE]",
            "[EPISODE_MODULATION]",
            "[SUBJECT_START]",
            "[SUBJECT_END]",
            "[ANSWER_START]",
            "[ANSWER_END]",
            "[COUNTERSUBJECT_START]",
            "[COUNTERSUBJECT_END]",
            "[TS_4/4]",
            "[KEY_C]",
        ]
        for b in range(1, 65):
            tokens.append(f"[BAR_{b}]")
        for v in range(1, 6):
            tokens.append(f"[VOICE_{v}]")
        for p in range(36, 85):
            tokens.append(f"P{p}")
        for d in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
            tokens.append(f"D{d}")

        unique_tokens = sorted(set(tokens))
        self.itos = {i: tok for i, tok in enumerate(unique_tokens)}
        self.stoi = {tok: i for i, tok in self.itos.items()}
        self.vocab_size = len(self.itos)

    def encode(self, seq):
        unk_id = self.stoi.get("[UNK]", 0)
        return [self.stoi.get(t, unk_id) for t in seq]

    def decode(self, indices):
        return [self.itos.get(int(i), "[UNK]") for i in indices]

