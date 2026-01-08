# PatchTST: A Time Series is Worth 64 Words

## 논문 정보
- **저자**: Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam
- **소속**: Princeton University, IBM Research
- **발표**: ICLR 2023
- **arXiv**: 2211.14730v2
- **주제**: Patch 기반 Transformer로 장기 시계열 예측 및 표현 학습

---

## 핵심 아이디어

### 문제 정의
- 기존 Transformer 기반 시계열 모델들의 한계:
  - Point-wise attention → 단일 시점은 의미론적 정보 부족
  - O(N²) 복잡도 → 긴 시퀀스 처리 어려움
  - 최근 연구에서 단순 선형 모델(DLinear)이 복잡한 Transformer를 능가
- **의문**: Transformer가 정말 시계열 예측에 효과적인가?

### 핵심 기여
1. **Patching**: 시계열을 subseries-level patch로 분할
2. **Channel-Independence**: 각 채널(변수) 독립 처리, 가중치 공유

### PatchTST 장점
| 장점 | 설명 |
|------|------|
| 복잡도 감소 | O(N²) → O((L/S)²), 최대 22배 속도 향상 |
| 긴 히스토리 활용 | Look-back window 확장 가능 |
| 표현 학습 | Self-supervised pre-training 지원 |
| 성능 향상 | MSE 21% 감소, MAE 16.7% 감소 |

---

## 모델 구조

### 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                     PatchTST Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│  Input: x ∈ ℝ^(M×L)  (M channels, L timesteps)                  │
│           ↓                                                      │
│  Channel Split → x^(i) ∈ ℝ^(1×L), i = 1,...,M                   │
│           ↓                                                      │
│  Instance Normalization                                          │
│           ↓                                                      │
│  Patching → x_p^(i) ∈ ℝ^(P×N)  (P: patch length, N: patches)   │
│           ↓                                                      │
│  Linear Projection + Position Embedding                          │
│           ↓                                                      │
│  Transformer Encoder (shared weights)                            │
│           ↓                                                      │
│  Flatten + Linear Head                                           │
│           ↓                                                      │
│  Output: x̂^(i) ∈ ℝ^(1×T)                                        │
│           ↓                                                      │
│  Concatenate → x̂ ∈ ℝ^(M×T)                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 1. Patching

**개념**: 시계열을 subseries-level patch로 분할

```
시계열: [x₁, x₂, ..., x_L]
           ↓ Patching (P=16, S=8)
Patches: [[x₁...x₁₆], [x₉...x₂₄], [x₁₇...x₃₂], ...]
         N = ⌊(L-P)/S⌋ + 2 개의 패치
```

**파라미터**:
- P: Patch length (권장: 16)
- S: Stride (권장: 8, overlap 허용)
- N: Number of patches ≈ L/S

**장점**:
1. **Local semantic information** 보존
2. **토큰 수 감소**: L → L/S (복잡도 제곱으로 감소)
3. **긴 히스토리 학습** 가능

### 2. Channel-Independence

**개념**: 각 변수를 독립적으로 처리

```
Channel-Mixing (기존):          Channel-Independence (PatchTST):
[x₁, x₂, ..., x_M] → Embed     x^(1) → Transformer → ŷ^(1)
     ↓                          x^(2) → Transformer → ŷ^(2)  (같은 가중치)
  Transformer                   ...
     ↓                          x^(M) → Transformer → ŷ^(M)
[ŷ₁, ŷ₂, ..., ŷ_M]
```

**장점**:
1. **적응성**: 각 시계열이 자체 attention map 학습
2. **과적합 방지**: 적은 데이터에서도 효과적
3. **확장성**: 학습/테스트 시 변수 개수 다를 수 있음

### 3. Transformer Encoder

**구성요소**:
```python
# Patch Embedding
x_d = W_p @ x_p + W_pos  # W_p ∈ ℝ^(D×P), W_pos ∈ ℝ^(D×N)

# Multi-Head Attention
Q = x_d @ W_Q, K = x_d @ W_K, V = x_d @ W_V
Attention(Q,K,V) = Softmax(QK^T / √d_k) @ V

# Feed Forward Network
FFN(x) = Linear(GELU(Linear(x)))

# Encoder Block (n layers)
z = LayerNorm(x + Attention(x))
z = LayerNorm(z + FFN(z))
```

**기본 설정**:
- Layers: 3
- Heads: 16
- Dimension: D = 128
- FFN dimension: F = 256
- Dropout: 0.2

### 4. Instance Normalization

**목적**: Distribution shift 완화

```python
x_norm = (x - mean(x)) / std(x)  # 입력 정규화
output = x_pred * std(x) + mean(x)  # 출력 역정규화
```

---

## Self-Supervised Learning

### Masked Patch Modeling

**개념**: BERT/MAE 스타일의 masked autoencoding

```
┌─────────────────────────────────────────────────────────────┐
│               Masked Self-Supervised Learning                │
├─────────────────────────────────────────────────────────────┤
│  1. Non-overlapping patches 생성 (S = P)                     │
│  2. 40% patches 랜덤 선택 → 0으로 마스킹                      │
│  3. Transformer로 마스킹된 패치 복원                          │
│  4. MSE Loss로 학습                                          │
└─────────────────────────────────────────────────────────────┘
```

### Point-wise vs Patch-wise Masking

| 방식 | 문제점 |
|------|--------|
| Point-wise | 인접값으로 보간 가능 → 고수준 표현 불필요 |
| Patch-wise | 전체 시퀀스 이해 필요 → 추상적 표현 학습 |

### Fine-tuning 전략

```
Pre-training (100 epochs)
         ↓
Linear Probing (10 epochs) - 헤드만 학습
         ↓
End-to-end Fine-tuning (20 epochs) - 전체 모델 학습
```

---

## 실험 결과

### 데이터셋

| Dataset | Features | Timesteps | 특징 |
|---------|----------|-----------|------|
| Weather | 21 | 52,696 | 기상 지표 |
| Traffic | 862 | 17,544 | 도로 점유율 |
| Electricity | 321 | 26,304 | 전력 소비 |
| ILI | 7 | 966 | 인플루엔자 |
| ETTh1/h2 | 7 | 17,420 | 변압기 온도 (시간) |
| ETTm1/m2 | 7 | 69,680 | 변압기 온도 (15분) |

### 장기 예측 성능 (MSE)

| Model | Weather | Traffic | Electricity | ETTh1 |
|-------|---------|---------|-------------|-------|
| PatchTST/64 | **0.149** | **0.360** | **0.129** | 0.370 |
| PatchTST/42 | 0.152 | 0.367 | 0.130 | **0.375** |
| DLinear | 0.176 | 0.410 | 0.140 | 0.375 |
| FEDformer | 0.238 | 0.576 | 0.186 | 0.376 |
| Autoformer | 0.249 | 0.597 | 0.196 | 0.435 |
| Informer | 0.354 | 0.733 | 0.304 | 0.941 |

**성능 향상**:
- vs Transformer 기반: MSE 21.0% ↓, MAE 16.7% ↓
- vs DLinear: 대형 데이터셋에서 우수

### Self-Supervised vs Supervised

| Dataset | Fine-tuning | Lin. Prob. | Supervised |
|---------|-------------|------------|------------|
| Weather (96) | **0.144** | 0.158 | 0.152 |
| Traffic (96) | **0.352** | 0.399 | 0.367 |
| Electricity (96) | **0.126** | 0.138 | 0.130 |

→ Self-supervised pre-training이 supervised보다 우수

### Transfer Learning

| Source → Target | MSE | 비교 |
|-----------------|-----|------|
| Electricity → Weather | 0.145 | vs Supervised 0.152 |
| Electricity → Traffic | 0.388 | vs Supervised 0.367 |

### 다른 Self-supervised 방법과 비교

| Method | MSE (ETTh1-96) | 향상률 |
|--------|----------------|--------|
| **PatchTST (transferred)** | **0.312** | - |
| **PatchTST (self-sup)** | 0.322 | - |
| BTSF | 0.541 | 42.3% ↓ |
| TS2Vec | 0.599 | 47.9% ↓ |
| TNC | 0.632 | 50.6% ↓ |
| TS-TCC | 0.653 | 52.2% ↓ |

---

## Ablation Study

### Patching + Channel-Independence 효과

| 설정 | Weather | Traffic | Electricity |
|------|---------|---------|-------------|
| P + CI (PatchTST) | **0.152** | **0.367** | **0.130** |
| CI only | 0.164 | 0.397 | 0.136 |
| P only | 0.168 | 0.595 | 0.196 |
| Original TST | 0.177 | 0.576 | 0.186 |

### Look-back Window 효과

```
L=96  → L=336 → L=512
MSE: 0.178 → 0.152 → 0.149  (Weather)
MSE: 0.477 → 0.367 → 0.365  (Traffic)
```

→ PatchTST는 긴 look-back window에서 일관된 성능 향상

### Patch Length 효과

```
P = [4, 8, 16, 24, 32, 40]
→ P ∈ {8, 16}이 일반적으로 좋은 성능
→ MSE 변동이 작음 (robust)
```

---

## 구현 가이드

### 1. PatchTST 모델 구현

```python
import torch
import torch.nn as nn

class PatchTST(nn.Module):
    def __init__(self, c_in, seq_len, pred_len,
                 patch_len=16, stride=8,
                 d_model=128, n_heads=16, n_layers=3,
                 d_ff=256, dropout=0.2):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride

        # Number of patches
        self.n_patches = (seq_len - patch_len) // stride + 2

        # Patch embedding
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.position_encoding = nn.Parameter(
            torch.randn(1, self.n_patches, d_model)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Output head
        self.head = nn.Linear(d_model * self.n_patches, pred_len)

        self.c_in = c_in

    def forward(self, x):
        # x: [B, L, M] → [B, M, L]
        x = x.permute(0, 2, 1)
        B, M, L = x.shape

        # Instance normalization
        means = x.mean(dim=-1, keepdim=True)
        stds = x.std(dim=-1, keepdim=True) + 1e-5
        x = (x - means) / stds

        # Patching: [B, M, L] → [B*M, N, P]
        x = x.reshape(B * M, L)
        x = self._create_patches(x)  # [B*M, N, P]

        # Patch embedding + position
        x = self.patch_embedding(x)  # [B*M, N, D]
        x = x + self.position_encoding

        # Transformer
        x = self.transformer(x)  # [B*M, N, D]

        # Flatten and predict
        x = x.reshape(B * M, -1)  # [B*M, N*D]
        x = self.head(x)  # [B*M, T]

        # Reshape: [B*M, T] → [B, M, T] → [B, T, M]
        x = x.reshape(B, M, -1)

        # Denormalize
        x = x * stds + means

        return x.permute(0, 2, 1)  # [B, T, M]

    def _create_patches(self, x):
        """Create patches with padding"""
        B, L = x.shape

        # Pad with last value
        pad_len = self.stride
        x_padded = torch.cat([x, x[:, -1:].repeat(1, pad_len)], dim=1)

        # Unfold to patches
        patches = x_padded.unfold(dimension=1,
                                   size=self.patch_len,
                                   step=self.stride)
        return patches  # [B, N, P]
```

### 2. Masked Self-Supervised Learning

```python
class PatchTST_SSL(PatchTST):
    def __init__(self, *args, mask_ratio=0.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_ratio = mask_ratio

        # Reconstruction head (instead of prediction head)
        self.recon_head = nn.Linear(self.d_model, self.patch_len)

    def forward(self, x, mask=None):
        # Patching (non-overlapping for SSL)
        x = x.permute(0, 2, 1)
        B, M, L = x.shape

        # Instance norm
        means = x.mean(dim=-1, keepdim=True)
        stds = x.std(dim=-1, keepdim=True) + 1e-5
        x = (x - means) / stds

        # Non-overlapping patches
        x = x.reshape(B * M, L)
        patches = self._create_patches_no_overlap(x)  # [B*M, N, P]

        # Random masking
        if mask is None:
            mask = self._random_mask(B * M, patches.shape[1])

        # Mask patches (set to zero)
        masked_patches = patches.clone()
        masked_patches[mask] = 0

        # Embed and encode
        x = self.patch_embedding(masked_patches)
        x = x + self.position_encoding
        x = self.transformer(x)

        # Reconstruct masked patches
        recon = self.recon_head(x)  # [B*M, N, P]

        # Loss only on masked patches
        loss = ((recon[mask] - patches[mask]) ** 2).mean()

        return loss, recon

    def _random_mask(self, batch_size, n_patches):
        """Random mask selection"""
        mask = torch.zeros(batch_size, n_patches, dtype=torch.bool)
        n_mask = int(n_patches * self.mask_ratio)

        for i in range(batch_size):
            mask_idx = torch.randperm(n_patches)[:n_mask]
            mask[i, mask_idx] = True

        return mask.to(self.device)

    def _create_patches_no_overlap(self, x):
        """Non-overlapping patches"""
        B, L = x.shape
        n_patches = L // self.patch_len
        x = x[:, :n_patches * self.patch_len]
        return x.reshape(B, n_patches, self.patch_len)
```

### 3. Training Pipeline

```python
def train_patchtst(model, train_loader, val_loader,
                   epochs=100, lr=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            pred = model(batch_x)
            loss = criterion(pred, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred = model(batch_x)
                val_loss += criterion(pred, batch_y).item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        scheduler.step()

        print(f'Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, '
              f'Val Loss={val_loss:.4f}')

    return model
```

### 4. Self-Supervised Pre-training + Fine-tuning

```python
def pretrain_and_finetune(model, pretrain_loader, finetune_loader,
                          val_loader):
    """Two-stage training: pre-training → fine-tuning"""

    # Stage 1: Self-supervised pre-training
    print("Stage 1: Pre-training...")
    ssl_model = PatchTST_SSL.from_supervised(model)

    optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=1e-4)

    for epoch in range(100):
        ssl_model.train()
        total_loss = 0

        for batch_x, _ in pretrain_loader:
            optimizer.zero_grad()
            loss, _ = ssl_model(batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Pretrain Epoch {epoch}: Loss={total_loss/len(pretrain_loader):.4f}')

    # Transfer weights to supervised model
    model.load_state_dict(ssl_model.state_dict(), strict=False)

    # Stage 2a: Linear probing (freeze encoder)
    print("Stage 2a: Linear Probing...")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
    train_patchtst(model, finetune_loader, val_loader, epochs=10)

    # Stage 2b: Fine-tuning (all parameters)
    print("Stage 2b: Fine-tuning...")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    train_patchtst(model, finetune_loader, val_loader, epochs=20)

    return model
```

### 5. Channel-Independence 구현

```python
class ChannelIndependentWrapper(nn.Module):
    """Channel-independence wrapper for any time series model"""

    def __init__(self, base_model_class, n_channels, **kwargs):
        super().__init__()
        # Single model, shared across all channels
        self.model = base_model_class(c_in=1, **kwargs)
        self.n_channels = n_channels

    def forward(self, x):
        # x: [B, L, M]
        B, L, M = x.shape

        # Process each channel independently
        outputs = []
        for i in range(M):
            channel_input = x[:, :, i:i+1]  # [B, L, 1]
            channel_output = self.model(channel_input)  # [B, T, 1]
            outputs.append(channel_output)

        return torch.cat(outputs, dim=-1)  # [B, T, M]
```

---

## 금융 시계열 적용

### 가격 예측

```python
def prepare_financial_data(prices, seq_len=336, pred_len=96):
    """금융 데이터 전처리"""
    # Log returns
    returns = np.log(prices / prices.shift(1)).dropna()

    # Features: returns, volatility, momentum
    features = pd.DataFrame({
        'returns': returns,
        'volatility': returns.rolling(20).std(),
        'momentum': prices.pct_change(20)
    }).dropna()

    # Create sequences
    X, y = [], []
    for i in range(len(features) - seq_len - pred_len):
        X.append(features.iloc[i:i+seq_len].values)
        y.append(features['returns'].iloc[i+seq_len:i+seq_len+pred_len].values)

    return np.array(X), np.array(y)
```

### 변동성 예측

```python
class VolatilityPatchTST(PatchTST):
    """변동성 예측 특화 모델"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Positive output for volatility
        self.softplus = nn.Softplus()

    def forward(self, x):
        pred = super().forward(x)
        return self.softplus(pred)  # 항상 양수
```

### Regime 기반 포지션 조절

```python
def regime_aware_forecasting(model, data, regime_states):
    """
    Regime 상태에 따른 예측 가중치 조절
    """
    forecasts = model(data)

    # Regime별 신뢰도 가중치
    weights = np.where(
        regime_states == 'stable',
        1.0,  # 안정 시장: 예측 신뢰
        0.5   # 불안정 시장: 예측 할인
    )

    # 가중 평균 예측
    weighted_forecast = forecasts * weights[:, None, None]

    return weighted_forecast
```

---

## 핵심 인사이트

### Channel-Independence가 효과적인 이유

1. **적응성**: 각 시계열이 자체 attention pattern 학습
2. **데이터 효율성**: Channel-mixing보다 적은 데이터로 수렴
3. **과적합 방지**: 학습 곡선이 안정적

### Patching의 효과

1. **Local semantics**: 단일 시점보다 풍부한 정보
2. **Computational efficiency**: 토큰 수 L → L/S
3. **긴 히스토리 활용**: 더 많은 과거 데이터 참조 가능

### 기존 Transformer 대비 개선점

| 문제 | 기존 모델 | PatchTST |
|------|----------|----------|
| Point-wise attention | 의미 부족 | Patch-wise semantic |
| 긴 시퀀스 | 메모리 폭발 | 효율적 처리 |
| 다변량 | Channel-mixing | Channel-independence |

---

## 한계점 및 향후 연구

### 한계점
1. **Cross-channel 관계 미고려**: 변수 간 상관관계 무시
2. **고정된 patch 크기**: 데이터 특성에 따른 적응 부족
3. **Point forecast만 제공**: 불확실성 정량화 없음

### 향후 연구 방향
1. **Cross-channel modeling**: Graph attention 등 활용
2. **Adaptive patching**: 데이터 특성에 따른 동적 패치
3. **Probabilistic forecasting**: 예측 분포 제공
4. **Foundation model**: 더 큰 규모의 pre-trained 모델

---

## 참고문헌

1. Nie et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. ICLR 2023
2. Zeng et al. (2022). Are Transformers Effective for Time Series Forecasting?
3. Zhou et al. (2022). FEDformer: Frequency Enhanced Decomposed Transformer
4. Wu et al. (2021). Autoformer: Decomposition Transformers with Auto-Correlation
5. He et al. (2021). Masked Autoencoders Are Scalable Vision Learners
