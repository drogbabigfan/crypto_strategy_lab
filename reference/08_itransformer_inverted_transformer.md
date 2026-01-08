# iTransformer: Inverted Transformers Are Effective for Time Series Forecasting

## 논문 정보
- **저자**: Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, Mingsheng Long
- **소속**: Tsinghua University, Ant Group
- **발표**: ICLR 2024
- **arXiv**: 2310.06625v4
- **GitHub**: https://github.com/thuml/iTransformer
- **주제**: 차원을 뒤집은 Transformer로 다변량 시계열 예측

---

## 핵심 아이디어

### 문제 정의
기존 Transformer 기반 시계열 예측 모델의 한계:

1. **Temporal Token 문제**: 같은 시점의 다변량 포인트를 하나의 토큰으로 임베딩
   - 서로 다른 물리적 의미와 측정 단위를 가진 변수들이 혼합됨
   - 변수 간 상관관계가 희석됨

2. **지역적 수용장**: 단일 시점 토큰은 정보가 너무 국소적
   - 시계열의 전역적 패턴을 포착하기 어려움

3. **시간 지연 문제**: 동시 기록된 시점이 실제로는 같은 이벤트를 반영하지 않을 수 있음
   - 센서 간 시스템적 시간 지연 존재

4. **Permutation Invariance**: Attention은 순서에 불변 → 시계열에 부적합

### 핵심 통찰

```
"Transformer가 시계열 예측에 효과가 없는 것이 아니라,
 부적절하게 사용되고 있는 것이다."
```

### 해결책: Inverted Transformer

| 기존 Transformer | iTransformer |
|-----------------|--------------|
| Temporal Token (시점 → 토큰) | Variate Token (변수 → 토큰) |
| Attention: 시간 의존성 | Attention: 다변량 상관관계 |
| FFN: 다변량 혼합 표현 | FFN: 시계열 표현 학습 |

---

## 모델 구조

### 아키텍처 비교

```
┌─────────────────────────────────────────────────────────────────┐
│                    Transformer (기존)                            │
├─────────────────────────────────────────────────────────────────┤
│  Input: X ∈ ℝ^(T×N)                                             │
│           ↓                                                      │
│  Embedding: 각 시점 t의 N개 변수 → 1개 Temporal Token             │
│           ↓                                                      │
│  [Token₁, Token₂, ..., Token_T]  (T개 토큰)                       │
│           ↓                                                      │
│  Attention: 시간 의존성 모델링                                    │
│           ↓                                                      │
│  FFN: 다변량 혼합 표현                                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    iTransformer (제안)                           │
├─────────────────────────────────────────────────────────────────┤
│  Input: X ∈ ℝ^(T×N)                                             │
│           ↓ Transpose                                            │
│  X^T ∈ ℝ^(N×T)                                                  │
│           ↓                                                      │
│  Embedding: 각 변수 n의 T개 시점 → 1개 Variate Token               │
│           ↓                                                      │
│  [Token₁, Token₂, ..., Token_N]  (N개 토큰)                       │
│           ↓                                                      │
│  Attention: 다변량 상관관계 모델링                                │
│           ↓                                                      │
│  FFN: 시계열 표현 학습 (각 변수 독립)                              │
└─────────────────────────────────────────────────────────────────┘
```

### iTransformer 전체 구조

```python
# iTransformer Forward Process
h⁰_n = Embedding(X_{:,n})           # 각 변수의 시계열 → variate token
H^{l+1} = TrmBlock(H^l), l=0,...,L-1  # L개 블록 통과
Ŷ_{:,n} = Projection(h^L_n)          # 예측 시계열 출력
```

**핵심 수식**:
```
Embedding: ℝ^T → ℝ^D   (시계열 → 토큰 임베딩)
Projection: ℝ^D → ℝ^S  (토큰 → 예측 시계열)
```

### 구성 요소별 역할 변화

| 구성요소 | Transformer | iTransformer |
|---------|-------------|--------------|
| **Layer Norm** | 같은 시점의 다변량 정규화 → 변수 혼합 | 각 변수의 시계열 정규화 → 비정상성 완화 |
| **Self-Attention** | 시간 의존성 (부적절) | 다변량 상관관계 (적절) |
| **Feed-Forward** | 혼합된 다변량 표현 | 시계열 표현 학습 (공유) |

---

## 핵심 구성요소

### 1. Layer Normalization

**기존 문제**:
- 같은 시점의 다변량을 정규화 → 관련 없는 변수 간 상호작용 노이즈

**iTransformer에서**:
```python
LayerNorm(H) = {(h_n - Mean(h_n)) / sqrt(Var(h_n)) | n = 1, ..., N}
```
- 각 변수의 시계열 표현을 개별 정규화
- **효과**: 비정상성 문제 완화, 변수 간 측정 단위 차이 감소

### 2. Feed-Forward Network

**기존 역할**: 다변량이 혼합된 temporal token 표현

**iTransformer에서**:
- 각 variate token에 동일하게 적용 (가중치 공유)
- **역할**: 시계열의 본질적 특성 학습 (진폭, 주기성, 주파수 스펙트럼)
- **해석**: 뉴런이 시계열 필터 역할

```
MLP 뉴런 = 시계열 패턴 필터
→ 다양한 변수에 전이 가능한 표현 학습
```

### 3. Self-Attention

**기존 문제**:
- Permutation invariant attention이 시계열 순서 정보 무시
- 시간 지연된 시점들 간의 무의미한 attention

**iTransformer에서**:
```python
Q, K, V = Linear(H)  # H ∈ ℝ^(N×D)
A = Softmax(QK^T / sqrt(d_k))  # A ∈ ℝ^(N×N): 다변량 상관관계 맵
Output = A @ V
```

**해석**:
- A_{i,j} ∝ q_i^T k_j : 변수 i와 j 간의 상관관계
- 고상관 변수는 더 높은 가중치로 상호작용
- **향상된 해석 가능성**: attention map = 다변량 상관관계

---

## 실험 결과

### 데이터셋

| Dataset | Variates | Timesteps | 도메인 |
|---------|----------|-----------|--------|
| ETT (4 subsets) | 7 | 8,545~69,680 | 전력 |
| ECL | 321 | 26,304 | 전력 소비 |
| Traffic | 862 | 17,544 | 교통 |
| Weather | 21 | 52,696 | 기상 |
| Solar-Energy | 137 | 52,560 | 태양광 |
| PEMS (4 subsets) | 170~883 | 10,172~16,911 | 교통 |

### 주요 예측 성능 (MSE, 평균)

| Model | ECL | Traffic | Weather | Solar |
|-------|-----|---------|---------|-------|
| **iTransformer** | **0.178** | **0.428** | **0.258** | **0.233** |
| PatchTST | 0.205 | 0.481 | 0.259 | 0.270 |
| TimesNet | 0.192 | 0.620 | 0.259 | 0.301 |
| DLinear | 0.212 | 0.625 | 0.265 | 0.330 |
| Crossformer | 0.244 | 0.550 | 0.259 | 0.641 |
| FEDformer | 0.214 | 0.610 | 0.309 | 0.291 |

### iTransformers 일반화 (다양한 Attention)

| Base Model | Original MSE | +Inverted MSE | 향상률 |
|------------|--------------|---------------|--------|
| Transformer | 0.277 | 0.178 | **35.6%** |
| Reformer | 0.338 | 0.208 | **38.4%** |
| Informer | 0.311 | 0.216 | **30.5%** |
| Flowformer | 0.267 | 0.210 | **21.3%** |
| Flashformer | 0.285 | 0.206 | **27.8%** |

→ 모든 Transformer 변형에서 일관된 성능 향상

### Lookback Window 증가 효과

```
기존 Transformer: lookback 증가 → 성능 저하 (attention 분산)
iTransformer:    lookback 증가 → 성능 향상 (더 많은 정보 활용)

ECL Dataset (T=96 예측):
Lookback:  48    96    192   336   720
MSE:      0.20  0.15  0.14  0.14  0.14  (iTransformer)
MSE:      0.26  0.28  0.30  0.32  0.35  (Transformer)
```

### 변수 일반화 능력

```
학습: 20% 변수만 사용
테스트: 100% 변수 예측

ECL Dataset:
CI-Transformer (100%→20%): +0.074 MSE 증가
iTransformer (100%→20%):   +0.048 MSE 증가  → 더 나은 일반화
```

---

## Ablation Study

### 구성요소별 효과

| Design | Variate Dim | Temporal Dim | ECL MSE | Traffic MSE |
|--------|-------------|--------------|---------|-------------|
| **iTransformer** | Attention | FFN | **0.178** | **0.428** |
| Attention both | Attention | Attention | 0.193 | 0.913 |
| Transformer | FFN | Attention | 0.202 | 0.863 |
| FFN both | FFN | FFN | 0.182 | 0.599 |
| w/o Attention | w/o | FFN | 0.193 | 0.461 |
| w/o FFN | Attention | w/o | 0.189 | 0.456 |

**핵심 발견**:
- Variate dim에 Attention + Temporal dim에 FFN이 최적
- 기존 Transformer (FFN on variate, Attention on temporal)는 최악

### CKA 유사도 분석

```
높은 CKA = 더 나은 시계열 예측 표현

iTransformers: CKA ≈ 0.85-0.95  → 좋은 표현
Transformers:  CKA ≈ 0.60-0.70  → 덜 적합한 표현
```

### Attention Map 해석

```
Layer 1 (얕은 층): Lookback 시계열의 상관관계와 유사
Layer L (깊은 층): Future 시계열의 상관관계와 유사

→ 층이 깊어질수록 과거 인코딩 → 미래 디코딩 진행
```

---

## 구현 가이드

### 1. iTransformer 모델

```python
import torch
import torch.nn as nn

class iTransformer(nn.Module):
    def __init__(self, seq_len, pred_len, n_vars,
                 d_model=512, n_heads=8, n_layers=4,
                 d_ff=2048, dropout=0.1):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.d_model = d_model

        # Embedding: 시계열 → variate token
        self.embedding = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            iTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Projection: variate token → 예측 시계열
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, pred_len)
        )

    def forward(self, x):
        # x: [B, T, N] → [B, N, T]
        x = x.permute(0, 2, 1)
        B, N, T = x.shape

        # Embedding: [B, N, T] → [B, N, D]
        h = self.embedding(x)

        # Transformer blocks
        for block in self.blocks:
            h = block(h)

        # Projection: [B, N, D] → [B, N, S]
        out = self.projection(h)

        # [B, N, S] → [B, S, N]
        return out.permute(0, 2, 1)


class iTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()

        # Layer Norm (on variate token features)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Multi-Head Attention (on variate tokens)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Feed-Forward (on each variate token)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, N, D] - N variate tokens

        # Self-attention on variate tokens
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out

        # FFN on each variate token
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x
```

### 2. 효율적 학습 전략 (변수 샘플링)

```python
class EfficientTrainer:
    """
    고차원 다변량 시계열을 위한 효율적 학습
    일부 변수만 샘플링하여 학습, 전체 변수로 추론
    """
    def __init__(self, model, sample_ratio=0.4):
        self.model = model
        self.sample_ratio = sample_ratio

    def train_step(self, x, y, optimizer):
        B, T, N = x.shape

        # 변수 랜덤 샘플링
        n_sample = int(N * self.sample_ratio)
        indices = torch.randperm(N)[:n_sample]

        x_sampled = x[:, :, indices]
        y_sampled = y[:, :, indices]

        # Forward
        optimizer.zero_grad()
        pred = self.model(x_sampled)
        loss = F.mse_loss(pred, y_sampled)

        # Backward
        loss.backward()
        optimizer.step()

        return loss.item()

    def predict(self, x):
        """전체 변수로 추론 (attention 유연성 활용)"""
        return self.model(x)
```

### 3. 다변량 상관관계 시각화

```python
def visualize_multivariate_correlations(model, x):
    """
    iTransformer의 attention map 시각화
    → 다변량 상관관계 해석
    """
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        # Embedding
        x_t = x.permute(0, 2, 1)
        h = model.embedding(x_t)

        # 각 층의 attention weights 추출
        attention_maps = []
        for block in model.blocks:
            h_norm = block.norm1(h)
            _, attn_weights = block.attention(
                h_norm, h_norm, h_norm,
                need_weights=True
            )
            attention_maps.append(attn_weights.squeeze(0).cpu().numpy())
            h = block(h)

    # 시각화
    n_layers = len(attention_maps)
    fig, axes = plt.subplots(1, n_layers+2, figsize=(4*(n_layers+2), 4))

    # Raw series correlations
    x_np = x.squeeze(0).numpy()
    lookback_corr = np.corrcoef(x_np.T)
    axes[0].imshow(lookback_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title('Lookback Correlations')

    # Attention maps per layer
    for i, attn in enumerate(attention_maps):
        axes[i+1].imshow(attn, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[i+1].set_title(f'Layer {i+1} Attention')

    plt.tight_layout()
    return fig
```

### 4. Efficient Attention 플러그인

```python
class iTransformerWithFlashAttention(iTransformer):
    """
    FlashAttention을 사용한 효율적 iTransformer
    변수 수가 많을 때 유용
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # FlashAttention 블록으로 교체
        from flash_attn.modules.mha import FlashMHA

        for i, block in enumerate(self.blocks):
            self.blocks[i].attention = FlashMHA(
                embed_dim=self.d_model,
                num_heads=kwargs.get('n_heads', 8),
                dropout=kwargs.get('dropout', 0.1),
                use_flash_attn=True
            )


class iTransformerWithLinearAttention(iTransformer):
    """
    Linear Attention을 사용한 iTransformer
    O(N²) → O(N) 복잡도 감소
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Linear Attention으로 교체
        for i, block in enumerate(self.blocks):
            self.blocks[i].attention = LinearAttention(
                dim=self.d_model,
                heads=kwargs.get('n_heads', 8)
            )


class LinearAttention(nn.Module):
    """Performer 스타일 Linear Attention"""
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, *args, **kwargs):
        B, N, D = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # Feature map (kernel approximation)
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Linear attention: O(N)
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        qkv = torch.einsum('bhnd,bhde->bhne', q, kv)
        z = torch.einsum('bhnd,bhd->bhn', q, k.sum(dim=2))

        out = qkv / (z.unsqueeze(-1) + 1e-6)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)

        return self.to_out(out), None
```

### 5. 금융 시계열 적용

```python
class FinancialForecaster:
    """금융 시계열을 위한 iTransformer 래퍼"""

    def __init__(self, n_assets, seq_len=96, pred_len=24):
        self.model = iTransformer(
            seq_len=seq_len,
            pred_len=pred_len,
            n_vars=n_assets,
            d_model=256,
            n_heads=8,
            n_layers=3
        )
        self.scaler = None

    def prepare_features(self, prices):
        """
        가격 데이터에서 예측용 특성 생성
        """
        features = {}

        # Returns
        features['returns'] = prices.pct_change()

        # Volatility (rolling)
        features['volatility'] = features['returns'].rolling(20).std()

        # Momentum
        features['momentum_5'] = prices.pct_change(5)
        features['momentum_20'] = prices.pct_change(20)

        # Volume-weighted features (if available)
        # ...

        return pd.concat(features, axis=1).dropna()

    def get_correlation_insights(self, x):
        """
        iTransformer의 학습된 다변량 상관관계 분석
        → 자산 간 동적 상관관계 파악
        """
        self.model.eval()
        with torch.no_grad():
            # 최종 층의 attention weights
            x_t = x.permute(0, 2, 1)
            h = self.model.embedding(x_t)

            for block in self.model.blocks[:-1]:
                h = block(h)

            # 마지막 층의 attention
            h_norm = self.model.blocks[-1].norm1(h)
            _, attn_weights = self.model.blocks[-1].attention(
                h_norm, h_norm, h_norm, need_weights=True
            )

        return attn_weights.squeeze(0).cpu().numpy()

    def regime_aware_forecast(self, x, regime):
        """
        Regime 정보를 활용한 예측
        """
        pred = self.model(x)

        # Regime별 신뢰도 조정
        if regime == 'high_volatility':
            # 변동성 높은 시기: 예측 불확실성 반영
            pred = pred * 0.7
        elif regime == 'trending':
            # 추세 시장: 예측 신뢰
            pred = pred * 1.0

        return pred
```

---

## 핵심 인사이트

### 왜 Inverted가 효과적인가?

1. **Variate Token의 전역적 수용장**
   - 전체 시계열을 하나의 토큰으로 → 충분한 정보
   - Patching의 극단적 케이스 (patch = 전체 시퀀스)

2. **Attention의 적절한 역할**
   - 다변량 상관관계는 학습 가능한 복잡한 관계
   - 시간 의존성은 FFN의 dense connection으로 충분

3. **FFN의 표현 학습**
   - Universal approximation theorem에 의해 시계열 패턴 학습 가능
   - 공유 가중치 → 변수 간 전이 가능한 표현

4. **유연성**
   - 학습/추론 시 변수 개수 변경 가능
   - 효율적 attention 메커니즘 플러그인 가능

### PatchTST vs iTransformer

| 측면 | PatchTST | iTransformer |
|------|----------|--------------|
| 토큰화 | Patch Token | Variate Token |
| 변수 처리 | Channel Independence | Multivariate Correlation |
| Attention | Temporal Patterns | Variate Correlations |
| 장점 | Local semantic | Global representation |
| 단점 | 급격한 변동 취약 | 변수 수 많으면 비효율 |

### 언제 iTransformer를 사용할까?

**적합한 경우**:
- 다변량 간 상관관계가 중요한 경우
- 고차원 다변량 시계열 (많은 변수)
- 변수 간 시간 지연이 존재하는 경우

**덜 적합한 경우**:
- 단변량 시계열 (attention 퇴화)
- 변수 간 독립적인 경우

---

## 한계점 및 향후 연구

### 한계점
1. **Univariate에서 퇴화**: 변수가 1개면 attention 무의미
2. **변수 수에 따른 복잡도**: O(N²) attention
3. **Temporal Inductive Bias 부족**: FFN만으로는 복잡한 시간 패턴 한계

### 향후 연구 방향
1. **Large-scale Pre-training**: 다양한 시계열 데이터로 사전학습
2. **Hybrid Approach**: Temporal + Variate attention 결합
3. **Fine-grained Tokenization**: TCN 기반 임베딩 등

---

## 참고문헌

1. Liu et al. (2024). iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. ICLR 2024
2. Nie et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. ICLR 2023
3. Zeng et al. (2023). Are Transformers Effective for Time Series Forecasting? AAAI 2023
4. Wu et al. (2023). TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. ICLR 2023
5. Zhang & Yan (2023). Crossformer: Transformer Utilizing Cross-Dimension Dependency. ICLR 2023
