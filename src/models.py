import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Tuple, Optional

# --- Constants Shared Across Train/Infer ---
BLOCK_SIZE = 1024
N_EMBD = 512
N_HEAD = 8
N_LAYER = 8
DROPOUT = 0.1

class BachTokenizer:
    """바흐 코랄 토큰 시퀀스를 정수 인덱스 벡터로 인코딩 및 디코딩하는 토크나이저 클래스입니다."""

    def __init__(self, all_sequences: Optional[List[List[str]]] = None) -> None:
        """토크나이저의 어휘집(Vocabulary)을 구축하고 인덱스 매핑 사전을 초기화합니다.
        
        Args:
            all_sequences: 학습 코러스 코퍼스로부터 추출된 전체 토큰 시퀀스 리스트.
        """
        if all_sequences:
            self.vocab = sorted(list(set([t for seq in all_sequences for t in seq])))
            self.vocab.append("[PAD]")
            self.vocab.append("[SOS]")
            self.vocab.append("[EOS]")
            self.stoi = {s: i for i, s in enumerate(self.vocab)}
            self.itos = {i: s for i, s in enumerate(self.vocab)}
            self.vocab_size = len(self.vocab)
        else:
            self.vocab = []
            self.stoi = {}
            self.itos = {}
            self.vocab_size = 0
        
    def encode(self, s_list: List[str]) -> List[int]:
        """텍스트 기반의 음표 및 화성 토큰 리스트를 정수 인덱스 목록으로 변환합니다.
        
        Args:
            s_list: 인코딩할 텍스트 토큰 리스트.
            
        Returns:
            List[int]: 매핑된 정수 인덱스 리스트.
        """
        return [self.stoi.get(s, self.stoi["[PAD]"]) for s in s_list]
        
    def decode(self, i_list: List[int]) -> List[str]:
        """정수 인덱스 리스트를 원래의 텍스트 토큰 리스트로 복원합니다.
        
        Args:
            i_list: 디코딩할 정수 인덱스 리스트.
            
        Returns:
            List[str]: 디코딩 완료된 텍스트 토큰 리스트.
        """
        return [self.itos.get(i, "[UNK]") for i in i_list]

class Head(nn.Module):
    """단일 어텐션 헤드(Single Attention Head) 모듈입니다."""

    def __init__(self, head_size: int) -> None:
        """어텐션 연산에 필요한 선형 변환 레이어와 룩업 캐시 버퍼를 초기화합니다.
        
        Args:
            head_size: 헤드의 차원 크기 (N_EMBD // N_HEAD).
        """
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """인풋 텐서에 인과적 마스크를 적용하여 어텐션 가중치를 계산하고 소프트맥스 출력을 반환합니다.
        
        Args:
            x: 배치 크기(B), 컨텍스트 길이(T), 채널 수(C)를 지닌 입력 텐서.
            
        Returns:
            torch.Tensor: 셀프 어텐션 연산이 완료된 텐서.
        """
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """다중 헤드 어텐션(Multi-Head Attention) 블록입니다."""

    def __init__(self, num_heads: int, head_size: int) -> None:
        """여러 개의 단일 어텐션 헤드를 병렬 생성하고 최종 투영 레이어를 초기화합니다.
        
        Args:
            num_heads: 병렬 작동할 헤드 수.
            head_size: 각 헤드의 내부 차원 크기.
        """
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """병렬 어텐션 결과를 채널 차원으로 결합하고 선형 투영을 수행합니다.
        
        Args:
            x: 입력 텐서 (B, T, N_EMBD).
            
        Returns:
            torch.Tensor: 다중 헤드 어텐션이 융합된 출력 텐서 (B, T, N_EMBD).
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """피드포워드 신경망(MLP) 블록입니다."""

    def __init__(self, n_embd: int) -> None:
        """입력 차원을 4배로 확장하고 비선형 활성화를 거쳐 다시 환원하는 선형 신경망을 구성합니다.
        
        Args:
            n_embd: 모델 임베딩 차원 차원 수.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """피드포워드 연산을 처리합니다.
        
        Args:
            x: 입력 텐서 (B, T, N_EMBD).
            
        Returns:
            torch.Tensor: MLP 출력 텐서 (B, T, N_EMBD).
        """
        return self.net(x)

class Block(nn.Module):
    """LayerNorm과 Residual Connection이 탑재된 표준 트랜스포머 블록입니다."""

    def __init__(self, n_embd: int, n_head: int) -> None:
        """셀프 어텐션 및 MLP 연산자와 레이어 정규화 블록을 정의합니다.
        
        Args:
            n_embd: 임베딩 차원 수.
            n_head: 어텐션 헤드 개수.
        """
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """프리-레이어 정규화 구조를 거치며 잔차 연결 연산을 수행합니다.
        
        Args:
            x: 입력 텐서.
            
        Returns:
            torch.Tensor: 블록 출력 텐서.
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BachTransformer(nn.Module):
    """바흐 4성부 코랄 작곡을 담당하는 25M 파라미터 생성형 트랜스포머 백본 신경망입니다."""

    def __init__(self, vocab_size: int, device: str = "cuda", ignore_index: int = -1) -> None:
        """토큰/포지셔널 임베딩, 스택형 트랜스포머 블록 및 출력 투영 헤드를 빌드합니다.
        
        Args:
            vocab_size: 어휘집 크기.
            device: 연산 장비 명칭 (예: 'cuda' 혹은 'cpu').
            ignore_index: 손실 함수 계산 시 무시할 마스크 인덱스 번호.
        """
        super().__init__()
        self.device = device
        self.ignore_index = ignore_index
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """토큰 인덱스 배치를 인풋으로 받아 어텐션을 전개하고 예측 로짓과 교차 엔트로피 손실을 계산합니다.
        
        Args:
            idx: 입력 정수 텐서 (B, T).
            targets: 손실 계산용 정답 라벨 텐서 (B, T) (Optional).
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 각 토큰별 예측 로짓(Logits) 텐서 및 계산된 Loss 값.
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
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

    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """인풋 컨텍스트 선율 텐서를 받아 다항 분포 확률적 샘플링에 따라 미래 화성 토큰 시퀀스를 연장 생성합니다.
        
        Args:
            idx: 시작 인덱스 텐서 (B, T).
            max_new_tokens: 최대 새로 생성할 음표 및 화성 토큰 수.
            temperature: 샘플링 다양성을 제어하는 온도 하이퍼파라미터.
            
        Returns:
            torch.Tensor: 생성된 결과가 누적 덧붙여진 정수 시퀀스 텐서 (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
