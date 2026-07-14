import torch
import torch.nn as nn
from torch.nn import functional as F

from dataclasses import dataclass

@dataclass
class BachTransformerConfig:
    model_scale: str = 'v5_pro'
    block_size: int = 4096
    dropout: float = 0.1
    ignore_index: int = -1
    is_causal: bool = True
    
    def __post_init__(self):
        if self.model_scale == 'small':
            self.n_embd = 384
            self.n_head = 6
            self.n_layer = 6
        elif self.model_scale == 'base':
            self.n_embd = 768
            self.n_head = 12
            self.n_layer = 12
        elif self.model_scale == 'large':
            self.n_embd = 1024
            self.n_head = 16
            self.n_layer = 24
        else:
            self.n_embd = 768
            self.n_head = 12
            self.n_layer = 16 # Default v5-Pro 113M config
            
# Backward compatibility variables if needed (though removed internally)
BLOCK_SIZE = 4096
DROPOUT = 0.1

class RotaryEmbedding(nn.Module):
    """어텐션 연산에 적용할 Rotary Position Embedding (RoPE) 클래스입니다."""
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
    """RoPE 회전 연산을 위해 입력을 절반으로 나누어 반전 결합합니다."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

class Head(nn.Module):
    """RoPE가 통합된 단일 인과적/비인과적 셀프 어텐션 헤드 모듈입니다."""
    def __init__(self, config: BachTransformerConfig, head_size: int):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.is_causal = config.is_causal
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)), persistent=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos=None, sin=None):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        if cos is not None and sin is not None:
            cos_u = cos[:T].unsqueeze(0)
            sin_u = sin[:T].unsqueeze(0)
            q = (q * cos_u) + (rotate_half(q) * sin_u)
            k = (k * cos_u) + (rotate_half(k) * sin_u)
            
        dropout_p = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=self.is_causal
        )
        return out

class MultiHeadAttention(nn.Module):
    """다중 헤드 어텐션 연산 블록입니다."""
    def __init__(self, config: BachTransformerConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos=None, sin=None):
        out = torch.cat([h(x, cos=cos, sin=sin) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """전형적인 다층 퍼셉트론 피드포워드 모듈입니다."""
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
    """LayerNorm과 Residual Connection이 포함된 표준 트랜스포머 블록입니다."""
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
    """v5 모델 가중치와 호환되는 최신 통합 대위법 트랜스포머 신경망입니다."""
    def __init__(self, vocab_size, device="cuda", config: BachTransformerConfig = None, ignore_index=-1, is_causal=True):
        super().__init__()
        self.device = device
        self.config = config if config is not None else BachTransformerConfig(ignore_index=ignore_index, is_causal=is_causal)
        self.ignore_index = self.config.ignore_index
        self.is_causal = self.config.is_causal
        
        # 다중 속성 임베딩
        self.token_embedding_table = nn.Embedding(vocab_size, self.config.n_embd)
        self.pitch_class_embedding_table = nn.Embedding(13, self.config.n_embd)
        self.octave_embedding_table = nn.Embedding(11, self.config.n_embd)
        self.voice_embedding_table = nn.Embedding(6, self.config.n_embd)
        
        # RoPE 모듈
        self.rope = RotaryEmbedding(dim=64, max_seq_len=self.config.block_size)
        
        # 레이어 블록 구성
        self.blocks = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)])
        
        self.ln_f = nn.LayerNorm(self.config.n_embd)
        self.lm_head = nn.Linear(self.config.n_embd, vocab_size) # bias=True 디폴트
        
        # 토큰별 속성 룩업 맵 초기화 및 로드
        pitch_class_map = torch.zeros(vocab_size, dtype=torch.long)
        octave_map = torch.zeros(vocab_size, dtype=torch.long)
        voice_map = torch.zeros(vocab_size, dtype=torch.long)
        
        # 디폴트 값 설정 (피치 없는 상태=12, 옥타브 없음=10, 일반/기본 성부=0)
        pitch_class_map.fill_(12)
        octave_map.fill_(10)
        voice_map.fill_(0)
        
        import pickle
        import os
        vocab_path = 'data/processed/v5/fugue_vocab_v5.pkl'
        if os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)
                itos = vocab_data['itos']
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
        
        # 임베딩 룩업 및 병합
        tok_emb = self.token_embedding_table(idx)
        pitch_class_idx = self.pitch_class_map[idx]
        octave_idx = self.octave_map[idx]
        voice_idx = self.voice_map[idx]
        
        pitch_class_emb = self.pitch_class_embedding_table(pitch_class_idx)
        octave_emb = self.octave_embedding_table(octave_idx)
        voice_emb = self.voice_embedding_table(voice_idx)
        
        x = tok_emb + pitch_class_emb + octave_emb + voice_emb
        
        # RoPE 코사인/사인 캐시 얻기
        cos, sin = self.rope(T, x.device)
        
        # 트랜스포머 블록 순차 전파
        for block in self.blocks:
            x = block(x, cos=cos, sin=sin)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
