# Blueprint v4.9: Sniper Classification System

**Version**: 4.9
**Date**: 2026-01-01
**Status**: Design Phase
**Breaking Change**: SSL Pre-training 폐기, End-to-End Supervised Learning 전환

---

## Changelog from v4.8

| Component | v4.8 | v4.9 |
|-----------|------|------|
| Learning | SSL (Masked Reconstruction) → Fine-tuning | **Direct Supervised Learning** |
| Labeling | TBM (timeout=100, loose) | **Sniper TBM (timeout=12~24, tight)** |
| Neutral Ratio | 0.03% (거의 없음) | **≥30% (강제)** |
| Model | PatchTST (Channel Independence) | **PatchTST + Channel Mixing Head** |
| Signal | argmax | **Confidence Threshold (≥0.6)** |

---

## 1. Goal

### 1.1 Core Philosophy: "Sniper, Not Machine Gun"

> **"확실한 기회만 잡고, 불확실하면 쉰다"**

- 모든 bar에서 진입하려 하지 않음
- 높은 확신(confidence ≥ 0.6)이 있을 때만 진입
- 수익 기대값이 수수료를 충분히 커버할 때만 진입
- 나머지는 Neutral (관망)

### 1.2 Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Neutral Prediction Ratio** | 40~60% | 충분히 선별적이어야 함 |
| **Entry Accuracy** | ≥55% | Long/Short 예측 시 방향 정확도 |
| **Profit Factor** | ≥1.3 | 총이익/총손실 |
| **Sharpe Ratio** | ≥0.5 | Risk-adjusted return |

### 1.3 Anti-Goals

- ❌ 모든 움직임 예측 시도
- ❌ 노이즈 구간에서 강제 진입
- ❌ 과적합된 복잡한 모델
- ❌ 수수료 대비 작은 수익 추구

---

## 2. Architecture

### 2.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────┤
│  [Raw Trades] → [Dollar Bars] → [Features] → [Sniper Labels]       │
│                                                    │                 │
│                                    ┌───────────────┴───────────────┐│
│                                    │  Long | Short | Neutral       ││
│                                    │  ~35% | ~35%  | ~30%          ││
│                                    └───────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MODEL PIPELINE (Per Fold)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Features   │    │   PatchTST   │    │   Channel    │          │
│  │  (N, C, L)   │───▶│   Encoder    │───▶│  Mixing Head │──▶ P(y)  │
│  │              │    │              │    │              │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                      │
│  C = 29 features                                                     │
│  L = 128 context length (reduced from 512)                          │
│  P(y) = [P(Short), P(Neutral), P(Long)]                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SIGNAL GENERATION                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  if max(P(y)) ≥ 0.6:                                                │
│      signal = argmax(P(y)) - 1    # {-1, 0, +1}                     │
│  else:                                                               │
│      signal = 0                   # Neutral (skip)                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     BACKTEST (Go)                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **SSL 폐기** | Masked reconstruction은 "연속성"을 학습하지만, 트레이딩에 필요한 "변곡점/방향성"과 무관 |
| **Context 128** | 512는 과도하게 김. Dollar bar 기준 128개면 충분한 context (약 1-2일) |
| **Channel Mixing** | 가격-거래량 상호작용이 핵심 alpha. Independence는 이를 포착 못함 |
| **Confidence 0.6** | argmax만으로는 51% vs 49%도 진입. 확신 없으면 관망 |

---

## 3. Pipeline Steps

### Stage 1: ETL (기존 유지)

```
[Binance Trades] → [Dollar Bar Aggregation] → [Feature Engineering]
```

- **Input**: Raw trade data
- **Output**: `features/*.parquet` (29 features per bar)
- **Status**: ✅ 완료 (Go 기반 ETL)

### Stage 2: Sniper Labeling (신규)

```
[Features + OHLCV] → [Sniper TBM] → [Balanced Labels]
```

#### 2.1 Sniper TBM Configuration

```python
@dataclass
class SniperTBMConfig:
    """Tight labeling for high-confidence entries only."""

    # Barrier multipliers (volatility-based)
    sl_mult: float = 1.5          # Stop Loss = 1.5σ
    pt_mult: float = 2.0          # Profit Target = 2.0σ (asymmetric)

    # SHORT timeout - 핵심 변경
    max_hold_bars: int = 18       # 12~24 bars (약 반나절)

    # Minimum expected return filter
    min_expected_return: float = 0.0045  # 0.45% (수수료 0.15% × 3)

    # Target neutral ratio
    target_neutral_ratio: float = 0.30   # 최소 30% neutral
```

#### 2.2 Labeling Logic

```python
def sniper_label(bar_idx, ohlcv, volatility):
    """
    Returns:
        +1 (Long): PT hit first within max_hold_bars
        -1 (Short): SL hit first within max_hold_bars
         0 (Neutral): Timeout OR low expected return
    """
    sigma = volatility[bar_idx]
    entry = ohlcv[bar_idx].close

    # Filter 1: Minimum Return Check
    expected_return = pt_mult * sigma
    if expected_return < min_expected_return:
        return 0  # Neutral - 수익 기대값 부족

    # Calculate barriers
    pt_level = entry * (1 + pt_mult * sigma)
    sl_level = entry * (1 - sl_mult * sigma)

    # Scan forward (SHORT horizon)
    for i in range(1, max_hold_bars + 1):
        future_bar = ohlcv[bar_idx + i]

        pt_hit = future_bar.high >= pt_level
        sl_hit = future_bar.low <= sl_level

        if pt_hit and not sl_hit:
            return +1  # Long
        if sl_hit and not pt_hit:
            return -1  # Short
        if pt_hit and sl_hit:
            # Same bar - check which hit first using high_time/low_time
            return +1 if future_bar.high_time < future_bar.low_time else -1

    # Filter 2: Timeout = Neutral
    return 0  # No clear direction within horizon
```

#### 2.3 Expected Label Distribution

| Class | Target Ratio | Meaning |
|-------|-------------|---------|
| Long (+1) | 30~40% | PT hit first (clear uptrend) |
| Short (-1) | 30~40% | SL hit first (clear downtrend) |
| **Neutral (0)** | **≥30%** | Timeout OR low volatility |

### Stage 3: Model Training (신규)

```
[Labeled Dataset] → [PatchTST + Mixing Head] → [Trained Classifier]
```

#### 3.1 Model Architecture

```python
class SniperClassifier(nn.Module):
    """
    PatchTST Encoder + Channel Mixing Head for 3-class classification.

    Key difference from vanilla PatchTST:
    - Vanilla: Each channel predicts independently
    - Ours: Channels are mixed before final classification
    """

    def __init__(
        self,
        n_features: int = 29,
        context_len: int = 128,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        n_classes: int = 3,
    ):
        super().__init__()

        # Number of patches
        n_patches = (context_len - patch_len) // stride + 1

        # Patch Embedding (per channel)
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # Positional Encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, n_patches, d_model) * 0.02
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ============================================
        # CHANNEL MIXING HEAD (핵심 변경)
        # ============================================
        # Encoder output: (B, n_features, n_patches, d_model)
        # After flatten: (B, n_features * n_patches * d_model)

        flat_dim = n_features * n_patches * d_model

        self.mixing_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        # Store config
        self.n_features = n_features
        self.context_len = context_len
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x):
        """
        Args:
            x: (B, n_features, context_len) - channel first

        Returns:
            logits: (B, n_classes)
        """
        B, C, L = x.shape

        # Create patches for each channel
        # (B, C, L) -> (B, C, n_patches, patch_len)
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)

        # Embed patches
        # (B, C, n_patches, patch_len) -> (B, C, n_patches, d_model)
        embedded = self.patch_embedding(patches)

        # Add positional encoding
        embedded = embedded + self.pos_encoding

        # Process each channel through encoder
        # (B, C, n_patches, d_model) -> (B*C, n_patches, d_model)
        B, C, N, D = embedded.shape
        embedded = embedded.view(B * C, N, D)

        encoded = self.encoder(embedded)

        # Reshape back
        # (B*C, n_patches, d_model) -> (B, C, n_patches, d_model)
        encoded = encoded.view(B, C, N, D)

        # Channel Mixing Head
        # (B, C, N, D) -> (B, n_classes)
        logits = self.mixing_head(encoded)

        return logits
```

#### 3.2 Training Configuration

```python
@dataclass
class SniperTrainingConfig:
    """Training config optimized for classification."""

    # Data
    context_len: int = 128
    batch_size: int = 128

    # Training
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Loss
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    use_class_weights: bool = True  # Balance Long/Short/Neutral

    # Early Stopping
    patience: int = 10
    min_delta: float = 0.001

    # Validation
    val_ratio: float = 0.15

    # Device
    device: str = "cuda"
```

#### 3.3 Loss Function

```python
class BalancedFocalLoss(nn.Module):
    """
    Focal Loss with:
    - Class weights for Long/Short/Neutral balance
    - Label smoothing for regularization
    - Focal modulation for hard example mining
    """

    def __init__(self, class_weights, gamma=2.0, smoothing=0.05):
        super().__init__()
        self.class_weights = class_weights  # (3,) tensor
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits, targets):
        # Apply label smoothing
        n_classes = logits.size(-1)
        smooth_targets = torch.full_like(logits, self.smoothing / n_classes)
        smooth_targets.scatter_(1, targets.unsqueeze(1),
                                1 - self.smoothing + self.smoothing / n_classes)

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)

        # Focal modulation
        pt = (probs * smooth_targets).sum(dim=-1)
        focal_weight = (1 - pt) ** self.gamma

        # Cross entropy
        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss = -(smooth_targets * log_probs).sum(dim=-1)

        # Apply class weights
        sample_weights = self.class_weights[targets]

        # Final loss
        loss = focal_weight * ce_loss * sample_weights

        return loss.mean()
```

### Stage 4: WFA Execution (수정)

```
For each fold:
    [Train Data] → [Sniper Labels] → [Train Model] → [Generate Signals] → [Backtest]
```

#### 4.1 Signal Generation with Confidence

```python
class ConfidenceSignalGenerator:
    """Generate signals only when confidence exceeds threshold."""

    def __init__(
        self,
        model: nn.Module,
        context_len: int = 128,
        confidence_threshold: float = 0.6,
        device: str = "cuda",
    ):
        self.model = model.to(device).eval()
        self.context_len = context_len
        self.threshold = confidence_threshold
        self.device = device

    @torch.no_grad()
    def generate(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            signals: (N,) int8 array {-1, 0, +1}
            confidences: (N,) float32 array [0, 1]
        """
        N = len(features)
        signals = np.zeros(N, dtype=np.int8)
        confidences = np.zeros(N, dtype=np.float32)

        for i in range(self.context_len, N):
            # Get context window
            window = features[i - self.context_len:i]
            x = torch.tensor(window.T, dtype=torch.float32, device=self.device)
            x = x.unsqueeze(0)  # (1, C, L)

            # Forward pass
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1).squeeze()  # (3,)

            # Get prediction and confidence
            confidence, pred = probs.max(dim=-1)
            confidence = confidence.item()
            pred = pred.item()

            # Apply threshold
            if confidence >= self.threshold:
                signals[i] = pred - 1  # {0,1,2} -> {-1,0,+1}
                confidences[i] = confidence
            else:
                signals[i] = 0  # Not confident enough
                confidences[i] = confidence

        return signals, confidences
```

#### 4.2 WFA Configuration

```python
@dataclass
class WFAConfig:
    """Walk-Forward Analysis configuration."""

    # Fold structure
    train_bars: int = 10000       # ~1-2 months of dollar bars
    test_bars: int = 2000         # ~1 week
    step_bars: int = 1000         # Roll forward

    # Early stopping
    max_cumulative_loss: float = -0.30  # Stop at -30%

    # Backtest
    sl_mult: float = 1.5
    pt_mult: float = 2.0
    max_hold_bars: int = 18
```

### Stage 5: Backtest (기존 Go Backtester 활용)

```
[Signals] + [OHLCV] → [Go Backtester] → [Metrics]
```

- 기존 Go backtester 재사용
- SL/PT/Timeout 파라미터를 Sniper TBM과 일치시킴

---

## 4. Implementation Details

### 4.1 Directory Structure

```
dl_rl_btc/
├── research/
│   ├── sniper/                    # NEW: v4.9 components
│   │   ├── __init__.py
│   │   ├── labeler.py             # Sniper TBM labeling
│   │   ├── model.py               # SniperClassifier
│   │   ├── dataset.py             # Dataset with balanced sampling
│   │   ├── trainer.py             # Training loop
│   │   └── signal_generator.py    # Confidence-based signals
│   ├── wfa/
│   │   ├── engine.py              # WFA orchestration
│   │   └── go_bridge.py           # Go backtester interface
│   └── ssl/                       # DEPRECATED in v4.9
│       └── ...
├── scripts/
│   ├── run_sniper_wfa.py          # NEW: Main entry point
│   └── run_btc_wfa.py             # DEPRECATED
└── artifacts/
    └── sniper/                    # Trained models per fold
```

### 4.2 Implementation Order

| Phase | Task | Estimated Effort |
|-------|------|------------------|
| **1** | Sniper TBM Labeler | 2-3 hours |
| **2** | SniperClassifier Model | 2-3 hours |
| **3** | Dataset with Balanced Sampling | 1-2 hours |
| **4** | Training Loop | 2-3 hours |
| **5** | Confidence Signal Generator | 1 hour |
| **6** | WFA Integration | 2-3 hours |
| **7** | Testing & Validation | 2-3 hours |

### 4.3 Key Implementation Notes

#### Sniper Labeler

```python
# sniper/labeler.py

class SniperLabeler:
    """
    CRITICAL: Label distribution must have ≥30% Neutral.

    If natural timeout rate is too low, increase min_expected_return
    or decrease max_hold_bars until target is met.
    """

    def label(self, df: pd.DataFrame) -> LabelResult:
        # ... labeling logic ...

        # Post-check: Verify neutral ratio
        neutral_ratio = (labels == 0).sum() / len(labels)
        if neutral_ratio < self.config.target_neutral_ratio:
            logger.warning(
                f"Neutral ratio {neutral_ratio:.1%} < target {self.config.target_neutral_ratio:.0%}. "
                "Consider tightening parameters."
            )

        return LabelResult(...)
```

#### Channel Mixing Head

```python
# sniper/model.py

# WHY Channel Mixing?
#
# Channel Independence (vanilla PatchTST):
#   - Price features processed separately from volume features
#   - Cannot learn: "High volume + price spike = momentum"
#   - Cannot learn: "Low volume + price change = fake breakout"
#
# Channel Mixing:
#   - All features combined before classification
#   - CAN learn cross-feature patterns
#   - Essential for alpha that depends on P-V interaction
```

#### Confidence Threshold

```python
# sniper/signal_generator.py

# WHY 0.6 threshold?
#
# Probability distribution example:
#   Case 1: [0.51, 0.25, 0.24] -> argmax = Short, but low confidence
#   Case 2: [0.65, 0.20, 0.15] -> argmax = Short, HIGH confidence
#
# With threshold 0.6:
#   - Case 1: Signal = 0 (Neutral) - not confident enough
#   - Case 2: Signal = -1 (Short) - confident entry
#
# Expected outcome:
#   - Fewer trades, but higher win rate
#   - Matches "Sniper" philosophy
```

### 4.4 Validation Checklist

Before running WFA, verify:

- [ ] Label distribution has ≥30% Neutral
- [ ] Model outputs balanced predictions (not all one class)
- [ ] Val F1 > 0.40 (better than random)
- [ ] Confidence threshold produces 40-60% Neutral signals
- [ ] No look-ahead bias in labeling

---

## 5. Expected Results

### 5.1 Baseline Comparison

| Metric | v4.8 (Failed) | v4.9 (Target) |
|--------|---------------|---------------|
| Val F1 | 0.33 (random) | ≥0.45 |
| Neutral Ratio (labels) | 0.03% | ≥30% |
| Neutral Ratio (signals) | 0% (all Short) | 40-60% |
| Entry Accuracy | ~50% | ≥55% |
| Sharpe | -30 | ≥0 (breakeven+) |

### 5.2 Success Criteria

**Minimum Viable Result:**
- Val F1 ≥ 0.40
- Sharpe ≥ 0 over full WFA
- Win rate ≥ 52%

**Good Result:**
- Val F1 ≥ 0.50
- Sharpe ≥ 0.5
- Win rate ≥ 55%
- Profit Factor ≥ 1.3

---

## 6. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Model still predicts single class | Monitor per-class prediction ratio during training |
| Overfitting | Early stopping + dropout + weight decay |
| Label noise | Minimum return filter removes ambiguous labels |
| Distribution shift | Walk-forward ensures out-of-sample testing |

---

## 7. Future Extensions (v5.0+)

After v4.9 validates the Sniper approach:

1. **Multi-timeframe fusion** - Combine bar-level and higher-TF signals
2. **Regime detection** - Separate models for trending vs ranging
3. **Dynamic confidence** - Adjust threshold based on recent performance
4. **Position sizing** - Kelly criterion based on confidence

---

## Appendix A: Migration from v4.8

```python
# What to DELETE:
- research/ssl/                    # SSL pre-training (not needed)
- scripts/run_btc_wfa.py          # Old WFA script

# What to KEEP:
- etl/                            # Go ETL pipeline
- research/wfa/go_bridge.py       # Go backtester interface
- research/wfa/engine.py          # WFA orchestration (modify)

# What to CREATE:
- research/sniper/                # New v4.9 components
- scripts/run_sniper_wfa.py       # New entry point
```

---

## Appendix B: Quick Reference

```python
# Sniper TBM Defaults
SL_MULT = 1.5
PT_MULT = 2.0
MAX_HOLD_BARS = 18
MIN_EXPECTED_RETURN = 0.0045

# Model Defaults
CONTEXT_LEN = 128
PATCH_LEN = 16
D_MODEL = 128
N_LAYERS = 3

# Signal Defaults
CONFIDENCE_THRESHOLD = 0.6

# WFA Defaults
TRAIN_BARS = 10000
TEST_BARS = 2000
MAX_CUMULATIVE_LOSS = -0.30
```
