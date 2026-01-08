# Deep Learning Bitcoin Trading Strategy Blueprint v4.6

## 변경 이력

| 버전 | 날짜 | 주요 변경 |
|------|------|----------|
| v1 | - | 초기 설계 (TIB 기반) |
| v2 | - | 실제 구현 반영 (Adaptive Dollar Bar 기반) |
| v3 | - | ETL + Feature Engineering 완료 (Go 통합 파이프라인) |
| v4 | - | SSL Pre-training + Dynamic Label Optimization 도입 |
| v4.1 | - | Fee Trap, Dirty Zero, Overlap, Dynamic Slippage 수정 |
| v4.2 | - | Profitability Proxy, Nonlinear Cost, Execution Lag 추가 |
| v4.3 | - | Multi-Asset Support 추가 |
| v4.5 | - | 복잡도 제거, Plateau Search, Fail-Fast, One-off SSL |
| **v4.6** | 2026-01 | **Re-structured** - 데이터 의존성 기반 Stage 재정렬 |

---

## 1. Goal Description

### 1.1 목표
**Crypto Futures Mid-Frequency Trading Strategy 개발**

- **전략**: Trend Following (Long & Short)
- **핵심 제약**: Avg Profit per Trade > 0.15% (수수료/슬리피지 후)
- **설계 철학**: Modular, Config-Driven, File-Centric, TDD, Fail-Fast

### 1.2 대상 자산

| 자산 | 용도 | 우선순위 |
|------|------|----------|
| BTCUSDT | SSL + Fine-tuning + Trading | Primary |
| ETHUSDT | SSL Pre-training | Foundation |
| SOLUSDT | SSL Pre-training | Foundation |
| XRPUSDT | SSL Pre-training | Foundation |
| DOGEUSDT | SSL Pre-training | Foundation |
| BNBUSDT | SSL Pre-training | Foundation |
| LTCUSDT | SSL Pre-training | Foundation |

### 1.3 v4.6 핵심 원칙

```
1. One-off SSL: WFA 진입 전 1회만 Foundation Model 학습
2. Plateau Search: Grid Search에서 Peak 금지, 안정적 영역 중심 선택
3. Fail-Fast: 데이터 부족/학습 실패 시 즉시 중단, Fallback 없음
4. Next Bar Entry: 신호(t) → 진입(t+1 Open), 미래 정보 사용 금지
5. Sample-level Drop: Context Window 내부가 아닌 샘플 단위로 제외
```

---

## 2. Design Philosophy

### 2.1 Test-Driven Development (TDD)
- Go: 테이블 기반 테스트 + 엣지케이스 테스트
- Python: pytest + synthetic data fixtures
- 커버리지: 핵심 경로 > 90%

### 2.2 File-Centric Architecture
- 모든 단계는 파일 입력 → 파일 출력
- 단계 간 인메모리 상태 없음
- "Resume from anywhere" 지원

### 2.3 Streaming-First
- 전체 데이터셋 단일 for-loop 처리
- 모든 정규화는 `Update(value)` 재귀적/누적 방식
- 수치 안정성: Kahan Summation, Welford's Algorithm

### 2.4 Fail-Fast Policy
```python
# v4.6: Fallback 제거, 명시적 실패
class FailFastPolicy:
    MIN_SAMPLES = 30000
    DIVERGENCE_CHECK_EPOCHS = 5
    MIN_LOSS_REDUCTION = 0.05  # 5%

    @staticmethod
    def check_data(n_samples: int):
        if n_samples < FailFastPolicy.MIN_SAMPLES:
            raise InsufficientDataError(
                f"Need {FailFastPolicy.MIN_SAMPLES}, got {n_samples}"
            )

    @staticmethod
    def check_convergence(loss_history: list):
        if len(loss_history) < FailFastPolicy.DIVERGENCE_CHECK_EPOCHS:
            return
        initial = loss_history[0]
        current = loss_history[-1]
        reduction = (initial - current) / (initial + 1e-10)
        if reduction < FailFastPolicy.MIN_LOSS_REDUCTION:
            raise DivergenceError(
                f"Loss reduction {reduction:.2%} < {FailFastPolicy.MIN_LOSS_REDUCTION:.0%}"
            )
```

---

## 3. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     v4.6 Pipeline Architecture (Re-structured)                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  STAGE 1: ETL & Feature Engineering (Go) - 데이터 준비                     │ │
│  │                                                                            │ │
│  │  Raw Trades (ZIP) ──▶ Dollar Bars ──▶ Feature Parquet (30 features)       │ │
│  │  (Multi-Asset)         Streaming        data/features/futures/            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                        │                                         │
│                                        ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  STAGE 2: SSL Pre-training (Python - One-off, WFA 진입 전 1회)            │ │
│  │                                                                            │ │
│  │  Multi-Asset Features ──▶ Patch Masking (40%) ──▶ Foundation Encoder      │ │
│  │  (BTC,ETH,SOL,XRP,         Reconstruction          artifacts/ssl/         │ │
│  │   DOGE,BNB,LTC)            Loss                    foundation_encoder.pt  │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                        │                                         │
│                                        ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  STAGE 3: Walk-Forward Analysis Loop (Fold별 반복)                         │ │
│  │  ┌──────────────────────────────────────────────────────────────────────┐  │ │
│  │  │  For each (Train Window, Test Window):                               │  │ │
│  │  │                                                                      │  │ │
│  │  │  [STEP 3.1]         [STEP 3.2]          [STEP 3.3]                  │  │ │
│  │  │  Label          ──▶  Fine-tuning   ──▶   Execution &                │  │ │
│  │  │  Optimization        (Python)            Backtest                   │  │ │
│  │  │  (Python)                                (Python)                   │  │ │
│  │  │                                                                      │  │ │
│  │  │  Grid Search    ──▶  Load Encoder  ──▶  Next Bar Entry              │  │ │
│  │  │  Plateau Search      + Classifier       Square Root Cost            │  │ │
│  │  │  Best Params         Focal Loss         Trade Execution             │  │ │
│  │  └──────────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 단계별 요약

| Stage | 이름 | 언어 | 입력 | 출력 |
|-------|------|------|------|------|
| 1 | ETL & Features | Go | Raw Trades | Feature Parquet |
| 2 | SSL Pre-training | Python | Multi-Asset Features | Foundation Encoder |
| 3 | WFA Loop | Python | Encoder + Features | Trades + Metrics |
| 3.1 | └ Label Optimizer | Python | Train Features | Best Params + Labels |
| 3.2 | └ Fine-tuning | Python | Encoder + Labels | Classifier Model |
| 3.3 | └ Execution | Python | Model + Test Data | Trades + Metrics |

---

## 4. Stage 1: ETL & Feature Engineering (Go)

> **상태**: ✅ 구현 완료 (v3에서 완성)

### 4.1 개요

- **입력**: Binance Raw Trades (ZIP)
- **출력**: Feature Parquet (30개 피처)
- **Dollar Bar**: `Threshold_t = EMA_14(DailyDollarVolume) / 50`

### 4.2 Feature 목록 (30개)

| 카테고리 | Feature | 설명 |
|---------|---------|------|
| **L1 Stateless** | log_volume, log_tick_count, log_duration | 기본 바 통계 |
| | log_trade_intensity, vwap_deviation | 거래 강도 |
| | volume_imbalance, bar_range, bar_body | 가격 구조 (8개) |
| **Regime** | garman_klass_vol, realized_vol | 변동성 |
| | shannon_entropy, vol_ratio, vol_zscore | 시장 상태 |
| | entropy_zscore, skewness, kurtosis | 분포 특성 (8개) |
| **Stationarity** | frac_diff_close, detrended_log_price, returns | 정상성 (3개) |
| **Momentum** | vw_momentum | 거래량 가중 모멘텀 |
| | momentum_zscore_10/50/250/1000 | 다중 윈도우 (5개) |
| **Technical** | connors_rsi | 복합 RSI (1개) |
| **Cyclical Time** | sin_time, cos_time | 시간 주기성 (Hour of Day) |
| | sin_week, cos_week | 요일 주기성 (Day of Week) (4개) |
| **Meta** | is_primed | Warmup 완료 플래그 (1개) |

### 4.3 정규화 방식

```go
// Go ETL: Streaming EWM Z-Score
// 자산별 독립 인스턴스 → 자산 간 Scale 차이 자동 처리

func (e *EWM) Update(value float64) float64 {
    e.count++
    if e.count == 1 {
        e.mean = value
        e.var_ = 0
        return 0
    }

    delta := value - e.mean
    e.mean += e.alpha * delta
    e.var_ = (1 - e.alpha) * (e.var_ + e.alpha*delta*delta)

    return e.zscore(value)  // Soft clipping [-5, 5]
}
```

### 4.4 파일 구조

```
data/
├── features/futures/
│   ├── BTCUSDT/
│   │   ├── BTCUSDT-features-2024-01.parquet
│   │   └── ...
│   ├── ETHUSDT/
│   └── ...
└── state/futures/
    ├── BTCUSDT/
    │   └── feature_state.json
    └── ...
```

---

## 5. Stage 2: SSL Pre-training (One-off)

### 5.1 개요

WFA 루프 진입 **전**, 단 1회 실행되는 Foundation Model 학습.

```
목적: 다양한 자산의 차트 패턴에서 "보편적 시장 구조"를 학습
효과: Data Hunger 해결, Transfer Learning 기반 확보
```

### 5.2 데이터 구성

```python
@dataclass
class SSLDataConfig:
    """SSL Pre-training 데이터 설정"""
    assets: list = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "SOLUSDT",
        "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LTCUSDT"
    ])
    context_len: int = 512
    min_samples_per_asset: int = 10000

    # 자산 간 상관관계 무시, 독립 샘플로 취급
    # → Shuffle하여 섞음 (시간순 아님)
    shuffle: bool = True


def load_ssl_data(config: SSLDataConfig) -> np.ndarray:
    """
    모든 자산의 Feature를 로드하여 하나의 데이터셋으로 통합.

    핵심: 각 자산은 Go ETL에서 이미 독립적으로 Z-Score 정규화됨.
          추가 정규화 불필요.
    """
    all_windows = []

    for asset in config.assets:
        features = load_features(asset)  # (N, 30 features)

        # Sliding window로 context 생성
        for i in range(len(features) - config.context_len + 1):
            window = features[i:i + config.context_len]  # (512, 30)
            all_windows.append(window)

    # 전체 섞기 (자산/시간 순서 무시)
    all_windows = np.array(all_windows)  # (Total, 512, 30)
    if config.shuffle:
        np.random.shuffle(all_windows)

    return all_windows
```

### 5.3 Patch-wise Random Masking

```python
class PatchMaskingDataset(Dataset):
    """
    v4.6: Patch-wise Random Masking (40%)

    PatchTST 표준 방식 채택.
    - Hybrid Masking 폐기 (복잡도 제거)
    - 샘플 단위로 독립적인 마스크 생성
    """

    def __init__(
        self,
        data: np.ndarray,           # (N, context_len, n_features)
        patch_len: int = 16,
        mask_ratio: float = 0.4     # 40% 마스킹
    ):
        self.data = data
        self.patch_len = patch_len
        self.mask_ratio = mask_ratio
        self.n_patches = data.shape[1] // patch_len

        # Learnable mask token (학습 시 초기화)
        self.mask_token = None

    def __getitem__(self, idx: int) -> dict:
        # (context_len, n_features) → (n_features, context_len)
        x = self.data[idx].T.copy()  # Channel-first: (28, 512)

        # 마스킹할 패치 수
        n_mask = int(self.n_patches * self.mask_ratio)

        # 랜덤 패치 인덱스 선택 (샘플마다 다름)
        mask_indices = np.random.choice(
            self.n_patches, n_mask, replace=False
        )

        # 마스킹 적용
        x_masked = x.copy()
        for patch_idx in mask_indices:
            start = patch_idx * self.patch_len
            end = start + self.patch_len
            # Zero masking (또는 learnable token)
            x_masked[:, start:end] = 0

        return {
            "input": torch.tensor(x_masked, dtype=torch.float32),
            "target": torch.tensor(x, dtype=torch.float32),
            "mask_indices": mask_indices
        }

    def __len__(self) -> int:
        return len(self.data)
```

### 5.4 SSL 모델 구조

```
Input: (Batch, 28 channels, 512 timesteps)
    ↓
Patching: (Batch, 28, 32 patches, 16 patch_len)
    ↓
Patch Embedding: Linear(16 → 128)
    ↓
Positional Embedding: Learnable (32 positions)
    ↓
Transformer Encoder: 4 heads, 2 layers, d_model=128
    ↓
Reconstruction Head: Linear(128 → 16)
    ↓
Output: Reconstructed patches (masked positions only)
```

### 5.5 SSL 학습 설정

```python
@dataclass
class SSLTrainingConfig:
    # Architecture
    n_features: int = 30
    context_len: int = 512
    patch_len: int = 16
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1

    # Masking
    mask_ratio: float = 0.4

    # Training
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Early stopping
    patience: int = 5
    min_delta: float = 1e-4


def ssl_loss(pred: torch.Tensor, target: torch.Tensor,
             mask_indices: np.ndarray, patch_len: int) -> torch.Tensor:
    """
    Masked Patch Reconstruction Loss.
    마스킹된 패치에 대해서만 MSE 계산.
    """
    total_loss = 0
    n_samples = pred.shape[0]

    for i in range(n_samples):
        for patch_idx in mask_indices[i]:
            start = patch_idx * patch_len
            end = start + patch_len
            loss = F.mse_loss(
                pred[i, :, start:end],
                target[i, :, start:end]
            )
            total_loss += loss

    return total_loss / n_samples
```

### 5.6 출력

```
artifacts/
└── ssl/
    └── foundation_encoder.pt    # Pre-trained encoder weights
```

---

## 6. Stage 3: Walk-Forward Analysis Loop

### 6.1 개요

**Rolling Window 방식**으로 파이프라인을 반복 실행하여 Out-of-Sample 성능 검증.

```
Time ────────────────────────────────────────────────────────▶

Fold 1:  [======= Train 12M =======][= Test 1M =]
Fold 2:       [======= Train 12M =======][= Test 1M =]
Fold 3:            [======= Train 12M =======][= Test 1M =]
...
```

### 6.2 WFA 설정

```python
@dataclass
class WFAConfig:
    train_months: int = 12
    test_months: int = 1
    step_months: int = 1
    min_train_bars: int = 30000  # Fail-Fast 기준
```

### 6.3 WFA 메인 루프

```python
class WalkForwardEngine:
    """Walk-Forward Analysis 엔진"""

    def __init__(
        self,
        config: WFAConfig,
        encoder_path: str,
        cost_model: SquareRootCostModel
    ):
        self.config = config
        self.encoder_path = encoder_path
        self.cost_model = cost_model
        self.results = []

    def run(self, features_path: str) -> list[dict]:
        """전체 WFA 실행"""

        # 1. Feature 로드 (is_primed=True만)
        features = load_all_features(features_path)
        features = features[features['is_primed'] == True]

        # 2. Fold 생성
        folds = self._generate_folds(features)

        for fold_id, (train_start, train_end, test_start, test_end) in enumerate(folds):
            print(f"\n{'='*60}")
            print(f"Fold {fold_id}: Train [{train_start} ~ {train_end}]")
            print(f"           Test  [{test_start} ~ {test_end}]")
            print(f"{'='*60}")

            # 3. 데이터 분할
            train_data = features[
                (features.index >= train_start) &
                (features.index < train_end)
            ]
            test_data = features[
                (features.index >= test_start) &
                (features.index < test_end)
            ]

            # 4. Step 3.1: Label Optimization
            opt_result = optimize_labels(train_data, GRID_CONFIG, self.cost_model)
            best_params = opt_result.best_params
            print(f"Best Params: {best_params}")

            # 5. Step 3.2: Fine-tuning
            train_dataset = FineTuningDataset(
                train_data[FEATURE_COLS].values,
                opt_result.valid_indices,
                opt_result.labels
            )

            # Train/Val split (90/10)
            train_ds, val_ds = random_split(train_dataset, [0.9, 0.1])

            model = finetune(
                self.encoder_path,
                train_ds,
                val_ds,
                FINETUNE_CONFIG
            )

            # 6. Step 3.3: Inference + Backtest
            signals = self._generate_signals(model, test_data)

            backtester = RealisticBacktester(self.cost_model)
            result = backtester.run(
                test_data,
                signals,
                sl_mult=best_params["sl"],
                pt_mult=best_params["pt"]
            )

            self.results.append({
                "fold_id": fold_id,
                "train_period": (train_start, train_end),
                "test_period": (test_start, test_end),
                "best_params": best_params,
                "metrics": result.metrics
            })

            print(f"Metrics: {result.metrics}")

        return self.results

    def _generate_signals(self, model: nn.Module, data: pd.DataFrame) -> np.ndarray:
        """모델 추론으로 신호 생성"""
        model.eval()
        signals = []

        features = data[FEATURE_COLS].values
        context_len = 512

        with torch.no_grad():
            for i in range(context_len, len(features)):
                window = features[i-context_len:i]  # (512, 28)
                x = torch.tensor(window.T, dtype=torch.float32).unsqueeze(0)

                logits = model(x)
                pred = torch.argmax(logits, dim=-1).item()

                # {0, 1, 2} → {-1, 0, +1}
                signal = pred - 1
                signals.append(signal)

        # 앞부분 패딩 (context_len만큼)
        signals = [0] * context_len + signals

        return np.array(signals)
```

---

## 7. Step 3.1: Label Optimization

### 7.1 개요

**목적**: 현재 시장 국면에 최적화된 Triple Barrier 파라미터 탐색

```
핵심 원칙:
1. Per-Bar Fee Trap Guard: 바별로 수익 가능성 검증
2. Sample-level Drop: Context Window가 아닌 샘플 단위 제외
3. Plateau Search: Peak 대신 안정적 영역 선택
```

### 7.2 Triple Barrier Method (TBM)

```python
@dataclass
class TBMConfig:
    sl_mult: float    # Stop Loss σ 배수
    pt_mult: float    # Profit Target σ 배수
    vertical_bars: int  # 최대 보유 기간 (바 수)


class TripleBarrierLabeler:
    """
    Triple Barrier 라벨링.

    라벨:
      +1: Profit Target 먼저 도달 (Long)
      -1: Stop Loss 먼저 도달 (Short)
       0: Vertical Barrier 도달 (Timeout)
    """

    def __init__(self, config: TBMConfig, cost_model: CostModel):
        self.config = config
        self.cost_model = cost_model

    def label(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            valid_indices: 학습에 사용할 샘플 인덱스
            labels: 해당 인덱스의 라벨
        """
        valid_indices = []
        labels = []

        for i in range(len(data) - self.config.vertical_bars):
            row = data.iloc[i]
            sigma = row['realized_vol']
            entry_price = row['close']

            # Fee Trap Check (Per-Bar)
            if not self.cost_model.is_viable(
                pt_mult=self.config.pt_mult,
                sigma=sigma,
                volume_ratio=row['volume'] / data['volume'].rolling(100).mean().iloc[i]
            ):
                continue  # 이 샘플은 학습에서 제외

            # Barrier 계산
            pt = entry_price * (1 + self.config.pt_mult * sigma)
            sl = entry_price * (1 - self.config.sl_mult * sigma)

            # Forward path
            path = data.iloc[i+1:i+1+self.config.vertical_bars]['close']

            # 먼저 도달한 barrier 확인
            pt_hit = (path >= pt).idxmax() if (path >= pt).any() else None
            sl_hit = (path <= sl).idxmax() if (path <= sl).any() else None

            if pt_hit and (not sl_hit or pt_hit < sl_hit):
                label = 1
            elif sl_hit and (not pt_hit or sl_hit < pt_hit):
                label = -1
            else:
                label = 0

            valid_indices.append(i)
            labels.append(label)

        return np.array(valid_indices), np.array(labels)
```

### 7.3 Grid Search 공간

| 파라미터 | 탐색 범위 | 단위 | 설명 |
|----------|-----------|------|------|
| SL | [1.0, 2.0, 3.0] | σ | 손절 폭 |
| PT | [1.5, 2.0, 2.5, 3.0, 3.5] | σ | 익절 폭 |
| Time | [50, 100, 200, 500, 1000] | Bars | 만기 |

**총 조합**: 3 × 5 × 5 = **75개**

### 7.4 Scoring Metrics

#### 7.4.1 Entropy (가중치: 0.2)

```python
def compute_entropy(labels: np.ndarray) -> float:
    """
    클래스 분포의 Shannon Entropy.
    최대값 = log(3) ≈ 1.585 (완전 균형)
    """
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return -np.sum(probs * np.log(probs + 1e-10))
```

#### 7.4.2 Mutual Information (가중치: 0.4)

```python
from sklearn.feature_selection import mutual_info_classif

def compute_mi(features: np.ndarray, labels: np.ndarray) -> float:
    """
    Feature-Label 상호 정보량.
    높을수록 Feature가 Label 예측에 유용.
    """
    mi_scores = mutual_info_classif(features, labels, discrete_features=False)
    return np.mean(mi_scores)
```

#### 7.4.3 Rank IC (가중치: 0.4)

```python
from scipy.stats import spearmanr

def compute_rank_ic(features: np.ndarray, labels: np.ndarray) -> float:
    """
    v4.6: Profitability Proxy 대체.

    Spearman Correlation의 절대값 평균.
    Label이 {-1, 0, +1}인 Ordinal Data이므로 적합.

    장점:
    - Outlier에 강건 (Rank 기반)
    - Selection Bias 방지
    """
    n_features = features.shape[1]
    ics = []

    for i in range(n_features):
        ic, _ = spearmanr(features[:, i], labels)
        if not np.isnan(ic):
            ics.append(abs(ic))  # 절대값 (방향 무관)

    return np.mean(ics) if ics else 0.0
```

#### 7.4.4 Composite Score

```python
def compute_composite_score(
    entropy: float,
    mi: float,
    rank_ic: float,
    weights: dict = {"entropy": 0.2, "mi": 0.4, "rank_ic": 0.4}
) -> float:
    """
    정규화 후 가중 합산.
    """
    # Min-Max 정규화는 Grid 전체에서 수행
    return (
        weights["entropy"] * norm_entropy +
        weights["mi"] * norm_mi +
        weights["rank_ic"] * norm_rank_ic
    )
```

### 7.5 Plateau Search (Connected Components)

```python
from scipy import ndimage

def find_plateau_center(
    scores: np.ndarray,
    grid_shape: tuple,  # (n_sl, n_pt, n_time) = (3, 5, 5)
    threshold_percentile: float = 90,
    fallback_percentile: float = 80
) -> tuple:
    """
    v4.6: Peak 대신 Plateau 중심 선택.

    알고리즘:
    1. 상위 10% 점수 영역을 Binary Mask로 변환
    2. Connected Components 찾기
    3. 가장 큰 Component의 Center of Mass 반환

    Args:
        scores: 1D array of composite scores (flattened grid)
        grid_shape: Original 3D grid shape
        threshold_percentile: 상위 N% 기준 (기본 90 = 상위 10%)
        fallback_percentile: Component 없을 시 fallback (기본 80)

    Returns:
        (sl_idx, pt_idx, time_idx): Grid 인덱스
    """
    scores_3d = scores.reshape(grid_shape)

    # 1차 시도: 상위 10%
    threshold = np.percentile(scores, threshold_percentile)
    binary_mask = (scores_3d >= threshold).astype(int)

    labeled, n_components = ndimage.label(binary_mask)

    # Component가 없으면 threshold 완화
    if n_components == 0:
        threshold = np.percentile(scores, fallback_percentile)
        binary_mask = (scores_3d >= threshold).astype(int)
        labeled, n_components = ndimage.label(binary_mask)

    # 여전히 없으면 최고점 반환 (최후 수단)
    if n_components == 0:
        flat_idx = np.argmax(scores)
        return np.unravel_index(flat_idx, grid_shape)

    # 가장 큰 Component 찾기
    component_sizes = ndimage.sum(
        binary_mask, labeled, range(1, n_components + 1)
    )
    largest_label = np.argmax(component_sizes) + 1

    # Center of Mass 계산
    center = ndimage.center_of_mass(
        binary_mask, labeled, largest_label
    )

    # 가장 가까운 정수 인덱스로 반올림
    return tuple(int(round(c)) for c in center)


def grid_idx_to_params(
    idx: tuple,
    sl_range: list,
    pt_range: list,
    time_range: list
) -> dict:
    """Grid 인덱스를 실제 파라미터로 변환"""
    return {
        "sl": sl_range[idx[0]],
        "pt": pt_range[idx[1]],
        "time": time_range[idx[2]]
    }
```

### 7.6 Label Optimizer 메인 루프

```python
def optimize_labels(
    features: pd.DataFrame,
    grid_config: dict,
    cost_model: CostModel
) -> OptimizationResult:
    """
    Grid Search + Plateau Search로 최적 파라미터 탐색.
    """
    sl_range = grid_config["sl_range"]      # [1.0, 2.0, 3.0]
    pt_range = grid_config["pt_range"]      # [1.5, 2.0, 2.5, 3.0, 3.5]
    time_range = grid_config["time_range"]  # [50, 100, 200, 500, 1000]

    grid_shape = (len(sl_range), len(pt_range), len(time_range))
    results = []

    # Grid Search
    for sl in sl_range:
        for pt in pt_range:
            for time in time_range:
                config = TBMConfig(sl_mult=sl, pt_mult=pt, vertical_bars=time)
                labeler = TripleBarrierLabeler(config, cost_model)

                valid_idx, labels = labeler.label(features)

                if len(labels) < grid_config["min_samples"]:
                    results.append({"score": 0})
                    continue

                valid_features = features.iloc[valid_idx][FEATURE_COLS].values

                entropy = compute_entropy(labels)
                mi = compute_mi(valid_features, labels)
                rank_ic = compute_rank_ic(valid_features, labels)

                results.append({
                    "sl": sl, "pt": pt, "time": time,
                    "entropy": entropy, "mi": mi, "rank_ic": rank_ic,
                    "n_samples": len(labels)
                })

    # Composite Score 계산 (정규화 포함)
    df = pd.DataFrame(results)
    df = normalize_and_score(df)

    # Plateau Search
    plateau_idx = find_plateau_center(
        df["score"].values,
        grid_shape
    )
    best_params = grid_idx_to_params(
        plateau_idx, sl_range, pt_range, time_range
    )

    # 최종 라벨 생성
    final_config = TBMConfig(**best_params)
    final_labeler = TripleBarrierLabeler(final_config, cost_model)
    final_idx, final_labels = final_labeler.label(features)

    return OptimizationResult(
        best_params=best_params,
        valid_indices=final_idx,
        labels=final_labels,
        grid_scores=df
    )
```

---

## 8. Step 3.2: Fine-tuning

### 8.1 개요

Pre-trained Encoder 위에 Classification Head를 부착하여 지도 학습.

```
입력: Pre-trained Encoder (Stage 2) + Labels (Step 3.1)
출력: Trained Classifier
```

### 8.2 모델 구조

```
[Pre-trained PatchTST Encoder] ← foundation_encoder.pt 로드
              ↓
      Flatten (32 patches × 128 dim = 4096)
              ↓
      Residual Bottleneck Head:
        Linear(4096 → 256) → GELU → Dropout(0.3)
        Linear(256 → 4096)
        Residual Connection
              ↓
      Classification: Linear(4096 → 3)
              ↓
      Output: Logits [Short, Neutral, Long]
```

### 8.3 Sample-level Drop (시계열 연속성 보존)

```python
class FineTuningDataset(Dataset):
    """
    v4.6: Sample-level Drop.

    핵심: Context Window 내부의 바를 삭제하지 않음.
          Fee Trap인 샘플 전체를 학습에서 제외.

    예시:
      Sample = (bars[100:612], label[612])
      - label[612]가 Fee Trap이면 → 이 샘플 제외
      - bars[100:612]는 항상 연속 512개 유지
    """

    def __init__(
        self,
        features: np.ndarray,    # (N_total, 28)
        valid_indices: np.ndarray,  # Step 3.1에서 생성
        labels: np.ndarray,
        context_len: int = 512
    ):
        self.features = features
        self.valid_indices = valid_indices
        self.labels = labels
        self.context_len = context_len

        # valid_indices 중 context window를 만들 수 있는 것만 필터
        self.usable_indices = valid_indices[valid_indices >= context_len]
        self.usable_labels = labels[valid_indices >= context_len]

    def __getitem__(self, idx: int) -> tuple:
        target_idx = self.usable_indices[idx]

        # 연속 512개 바 (항상 연속, Drop 없음)
        window = self.features[target_idx - self.context_len:target_idx]
        label = self.usable_labels[idx]

        # (512, 28) → (28, 512) Channel-first
        x = torch.tensor(window.T, dtype=torch.float32)
        y = torch.tensor(label + 1, dtype=torch.long)  # {-1,0,1} → {0,1,2}

        return x, y

    def __len__(self) -> int:
        return len(self.usable_indices)
```

### 8.4 Focal Loss

```python
class FocalLoss(nn.Module):
    """
    Class Imbalance 처리를 위한 Focal Loss.
    FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)

        # Label smoothing
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.label_smoothing / n_classes)
            smooth_targets.scatter_(
                1, targets.unsqueeze(1),
                1 - self.label_smoothing + self.label_smoothing / n_classes
            )

        # Focal weight
        probs = F.softmax(logits, dim=-1)
        pt = (probs * smooth_targets).sum(dim=-1)
        focal_weight = (1 - pt) ** self.gamma

        # Cross entropy
        ce_loss = -torch.sum(smooth_targets * F.log_softmax(logits, dim=-1), dim=-1)

        return (focal_weight * ce_loss).mean()
```

### 8.5 Fine-tuning 학습

```python
def finetune(
    encoder_path: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: FinetuningConfig
) -> nn.Module:
    """
    2-Phase Fine-tuning:
    1. Encoder Frozen (5 epochs): Classification Head만 학습
    2. Full Unfrozen (나머지): 전체 미세조정
    """

    # 1. 모델 로드
    encoder = load_encoder(encoder_path)
    model = PatchTSTClassifier(encoder, n_classes=3)

    # 2. Data check
    FailFastPolicy.check_data(len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    loss_fn = FocalLoss(gamma=2.0, label_smoothing=0.1)
    loss_history = []

    for epoch in range(config.epochs):
        # Phase 1: Encoder Frozen
        if epoch < config.frozen_epochs:
            for param in model.encoder.parameters():
                param.requires_grad = False
            optimizer = torch.optim.AdamW(
                model.classifier.parameters(), lr=config.head_lr
            )
        # Phase 2: Full Unfrozen
        else:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": model.encoder.parameters(), "lr": config.encoder_lr},
                {"params": model.classifier.parameters(), "lr": config.head_lr}
            ])

        # Training loop
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        # Fail-Fast: Divergence Check
        FailFastPolicy.check_convergence(loss_history)

        # Validation
        val_metrics = validate(model, val_loader)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, val_f1={val_metrics['f1']:.4f}")

    return model
```

---

## 9. Step 3.3: Execution & Backtest

### 9.1 Square Root Cost Model

```python
@dataclass
class SquareRootCostModel:
    """
    v4.6: 비선형 Market Impact 모델.

    Impact = η × σ × √(Q / V)

    - η: Impact coefficient
    - σ: Volatility
    - Q: Order size
    - V: Average volume
    """

    base_fee: float = 0.001        # 0.10% 거래소 수수료
    base_slippage: float = 0.0001  # 0.01% 최소 슬리피지
    impact_coeff: float = 0.1      # Impact 계수
    slippage_cap: float = 0.005    # 최대 0.50%
    min_profit_buffer: float = 1.5 # 비용의 1.5배 이상 수익 필요

    def get_slippage(
        self,
        volatility: float,
        trade_size: float,
        avg_volume: float
    ) -> float:
        """
        Square Root Law 기반 슬리피지 계산.

        slippage = η × σ × √(trade_size / volume)
        """
        if avg_volume <= 0:
            return self.slippage_cap

        volume_ratio = trade_size / avg_volume
        impact = self.impact_coeff * volatility * np.sqrt(volume_ratio)

        total_slippage = self.base_slippage + impact
        return min(total_slippage, self.slippage_cap)

    def get_total_cost(
        self,
        volatility: float,
        trade_size: float = 0.01,
        avg_volume: float = 1.0
    ) -> float:
        """진입 + 청산 총 비용 (Round-trip)"""
        slippage = self.get_slippage(volatility, trade_size, avg_volume)
        return 2 * (self.base_fee + slippage)  # 2배 (진입 + 청산)

    def is_viable(
        self,
        pt_mult: float,
        sigma: float,
        volume_ratio: float = 1.0
    ) -> bool:
        """
        Fee Trap Guard: 이 거래가 수익 가능한가?

        Expected Profit = PT × σ
        Required = Total Cost × Buffer
        """
        expected_profit = pt_mult * sigma
        total_cost = self.get_total_cost(sigma, volume_ratio=volume_ratio)
        required = total_cost * self.min_profit_buffer

        return expected_profit > required
```

### 9.2 Next Bar Entry Logic

```python
class RealisticBacktester:
    """
    v4.6: Realistic Execution.

    핵심 규칙:
    1. Signal은 Close[t] 확정 후 생성
    2. Entry는 Open[t+1]에서만 가능
    3. Square Root Cost 적용
    """

    def __init__(
        self,
        cost_model: SquareRootCostModel,
        initial_capital: float = 100000,
        risk_per_trade: float = 0.02
    ):
        self.cost_model = cost_model
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade

    def run(
        self,
        data: pd.DataFrame,
        signals: np.ndarray,
        sl_mult: float,
        pt_mult: float
    ) -> BacktestResult:
        """
        Args:
            data: OHLCV + Features
            signals: [-1, 0, +1] array (same length as data)
        """
        equity = self.initial_capital
        equity_curve = [equity]
        trades = []
        position = None
        pending_signal = None

        for i in range(len(data)):
            row = data.iloc[i]

            # ===== 1. 대기 중인 신호 실행 (t+1 시점) =====
            if pending_signal is not None and position is None:
                # Next Bar Open 진입
                entry_price = row['open']  # ← 핵심: open 가격 사용
                sigma = row['realized_vol']

                # Cost 계산
                avg_volume = data['volume'].iloc[max(0,i-100):i].mean()
                slippage = self.cost_model.get_slippage(
                    sigma, self.risk_per_trade, avg_volume
                )

                # 슬리피지 적용된 실제 진입가
                if pending_signal == 1:  # Long
                    actual_entry = entry_price * (1 + slippage)
                    tp = entry_price * (1 + pt_mult * sigma)
                    sl = entry_price * (1 - sl_mult * sigma)
                else:  # Short
                    actual_entry = entry_price * (1 - slippage)
                    tp = entry_price * (1 - pt_mult * sigma)
                    sl = entry_price * (1 + sl_mult * sigma)

                position = {
                    "direction": pending_signal,
                    "entry": actual_entry,
                    "tp": tp,
                    "sl": sl,
                    "entry_bar": i
                }
                pending_signal = None

            # ===== 2. 포지션 청산 체크 =====
            if position is not None:
                exit_price, exit_reason = self._check_exit(position, row)

                if exit_price is not None:
                    # 슬리피지 적용
                    sigma = row['realized_vol']
                    avg_volume = data['volume'].iloc[max(0,i-100):i].mean()
                    slippage = self.cost_model.get_slippage(
                        sigma, self.risk_per_trade, avg_volume
                    )

                    if position["direction"] == 1:
                        actual_exit = exit_price * (1 - slippage)
                        pnl = (actual_exit - position["entry"]) / position["entry"]
                    else:
                        actual_exit = exit_price * (1 + slippage)
                        pnl = (position["entry"] - actual_exit) / position["entry"]

                    # 수수료 차감
                    pnl -= 2 * self.cost_model.base_fee

                    trades.append({
                        "entry_bar": position["entry_bar"],
                        "exit_bar": i,
                        "direction": position["direction"],
                        "pnl": pnl,
                        "reason": exit_reason
                    })

                    equity *= (1 + pnl * self.risk_per_trade)
                    position = None

            # ===== 3. 새 신호 대기열 등록 =====
            if i < len(signals) and position is None and pending_signal is None:
                if signals[i] != 0:
                    pending_signal = signals[i]
                    # 다음 바에서 실행됨

            equity_curve.append(equity)

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            metrics=self._compute_metrics(trades, equity_curve)
        )

    def _check_exit(self, position: dict, row: pd.Series) -> tuple:
        """TP/SL 체크"""
        if position["direction"] == 1:  # Long
            if row["high"] >= position["tp"]:
                return position["tp"], "TP"
            if row["low"] <= position["sl"]:
                return position["sl"], "SL"
        else:  # Short
            if row["low"] <= position["tp"]:
                return position["tp"], "TP"
            if row["high"] >= position["sl"]:
                return position["sl"], "SL"
        return None, None

    def _compute_metrics(self, trades: list, equity: list) -> dict:
        if not trades:
            return {"total_trades": 0}

        pnls = [t["pnl"] for t in trades]

        return {
            "total_trades": len(trades),
            "win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
            "avg_pnl": np.mean(pnls),
            "sharpe": np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252 * 50),
            "max_drawdown": self._calc_max_drawdown(equity),
            "profit_factor": (
                sum(p for p in pnls if p > 0) /
                (-sum(p for p in pnls if p < 0) + 1e-10)
            )
        }

    def _calc_max_drawdown(self, equity: list) -> float:
        peak = equity[0]
        max_dd = 0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd
```

### 9.3 성공 기준

```python
SUCCESS_CRITERIA = {
    "avg_pnl_per_trade": 0.0015,  # > 0.15%
    "win_rate": 0.45,             # > 45%
    "sharpe_ratio": 1.5,          # > 1.5
    "max_drawdown": 0.20          # < 20%
}
```

---

## 10. Configuration Schema

```yaml
# config/config_v46.yaml

experiment_name: "btc_trading_v46"

# ============================================================
# Assets
# ============================================================
assets:
  foundation:  # SSL Pre-training용
    - BTCUSDT
    - ETHUSDT
    - SOLUSDT
    - XRPUSDT
    - DOGEUSDT
    - BNBUSDT
    - LTCUSDT
  trading:     # 실제 매매 대상
    - BTCUSDT

# ============================================================
# Data Paths
# ============================================================
paths:
  features: "./data/features/futures"
  ssl_encoder: "./artifacts/ssl/foundation_encoder.pt"
  models: "./artifacts/classifier"
  results: "./results/wfa"

# ============================================================
# Stage 1: ETL (Go - 외부 실행)
# ============================================================
etl:
  dollar_bar_divisor: 50
  ewm_alpha: 0.02

# ============================================================
# Stage 2: SSL Pre-training
# ============================================================
ssl:
  context_len: 512
  patch_len: 16
  mask_ratio: 0.4

  model:
    d_model: 128
    n_heads: 4
    n_layers: 2
    dropout: 0.1

  training:
    epochs: 30
    batch_size: 64
    learning_rate: 1.0e-4
    weight_decay: 1.0e-5
    patience: 5

# ============================================================
# Stage 3: WFA Loop
# ============================================================
wfa:
  train_months: 12
  test_months: 1
  step_months: 1

# Step 3.1: Label Optimizer
label_optimizer:
  grid:
    sl_range: [1.0, 2.0, 3.0]
    pt_range: [1.5, 2.0, 2.5, 3.0, 3.5]
    time_range: [50, 100, 200, 500, 1000]

  scoring:
    entropy_weight: 0.2
    mi_weight: 0.4
    rank_ic_weight: 0.4

  plateau:
    threshold_percentile: 90
    fallback_percentile: 80

  min_samples: 1000
  min_entropy: 1.0

# Step 3.2: Fine-tuning
finetuning:
  frozen_epochs: 5
  total_epochs: 30
  batch_size: 64
  encoder_lr: 1.0e-5
  head_lr: 1.0e-4

  focal_loss:
    gamma: 2.0
    label_smoothing: 0.1

# Step 3.3: Execution & Backtest
cost_model:
  base_fee: 0.001
  base_slippage: 0.0001
  impact_coeff: 0.1
  slippage_cap: 0.005
  min_profit_buffer: 1.5

backtest:
  initial_capital: 100000
  risk_per_trade: 0.02

# ============================================================
# Fail-Fast
# ============================================================
fail_fast:
  min_samples: 30000
  divergence_check_epochs: 5
  min_loss_reduction: 0.05
```

---

## 11. Directory Structure

```
/home/kimhoyeon/dev/dl_rl_btc/
├── config/
│   ├── config.yaml              # 기존 설정
│   └── config_v46.yaml          # v4.6 설정
│
├── data/
│   ├── features/futures/        # Stage 1 출력: Feature Parquet
│   │   ├── BTCUSDT/
│   │   ├── ETHUSDT/
│   │   └── ...
│   └── state/futures/           # ETL 상태
│
├── etl/                         # Stage 1: Go ETL (구현 완료)
│   ├── cmd/main.go
│   └── internal/
│       ├── bars/
│       ├── features/
│       └── ...
│
├── research/                    # Python ML
│   │
│   ├── ssl/                     # Stage 2: SSL Pre-training
│   │   ├── __init__.py
│   │   ├── dataset.py           # PatchMaskingDataset
│   │   ├── model.py             # PatchTST Encoder
│   │   └── train.py             # SSL 학습 루프
│   │
│   ├── label_optimizer/         # Step 3.1: Label Optimization
│   │   ├── __init__.py
│   │   ├── cost_model.py        # Square Root Cost Model
│   │   ├── tbm.py               # Triple Barrier
│   │   ├── scoring.py           # Entropy, MI, Rank IC
│   │   ├── plateau.py           # Plateau Search
│   │   └── optimizer.py         # 메인 오케스트레이터
│   │
│   ├── trainer/                 # Step 3.2: Fine-tuning
│   │   ├── __init__.py
│   │   ├── dataset.py           # FineTuningDataset
│   │   ├── model.py             # Classifier
│   │   ├── loss.py              # Focal Loss
│   │   └── train.py             # Fine-tuning 루프
│   │
│   ├── wfa/                     # Stage 3: WFA Engine
│   │   ├── __init__.py
│   │   ├── engine.py            # WFA 메인 루프
│   │   ├── backtester.py        # Realistic Backtester (Step 3.3)
│   │   └── metrics.py           # 성과 지표
│   │
│   ├── models/                  # 공통 모델
│   │   ├── patchtst.py
│   │   └── classifier.py
│   │
│   └── utils/
│       ├── fail_fast.py         # Fail-Fast Policy
│       └── data_loader.py
│
├── artifacts/                   # 학습 결과물
│   ├── ssl/
│   │   └── foundation_encoder.pt    # Stage 2 출력
│   └── classifier/
│       └── fold_{id}_model.pt       # Step 3.2 출력
│
├── results/                     # 실험 결과
│   └── wfa/
│       ├── fold_{id}/
│       └── summary.json
│
├── tests/                       # pytest
│   ├── test_ssl.py              # Stage 2 테스트
│   ├── test_label_optimizer.py  # Step 3.1 테스트
│   ├── test_trainer.py          # Step 3.2 테스트
│   └── test_backtester.py       # Step 3.3 테스트
│
└── docs/
    ├── blueprint_v4.md          # 이전 버전
    └── blueprint_v4.6.md        # 현재 버전 (Re-structured)
```

---

## 12. Implementation Checklist

### Stage 1: ETL & Feature Engineering (Go)
- [x] `etl/internal/bars/` - Adaptive Dollar Bar
- [x] `etl/internal/features/` - 30개 Feature 계산
- [x] `etl/internal/normalize/` - Streaming EWM Z-Score
- [x] Parquet 출력
- [x] 단위 테스트

### Stage 2: SSL Pre-training (Python)
- [x] `ssl/dataset.py` - PatchMaskingDataset
- [x] `ssl/model.py` - PatchTST Encoder (reconstruction head 포함)
- [x] `ssl/train.py` - SSL 학습 루프
- [x] `test_ssl.py` - 단위 테스트 (25개)

### Stage 3: Walk-Forward Analysis Loop

#### Step 3.1: Label Optimization
- [x] `label_optimizer/cost_model.py` - Square Root Cost Model
- [x] `label_optimizer/tbm.py` - Triple Barrier Labeler
- [x] `label_optimizer/scoring.py` - Entropy, MI, Rank IC
- [x] `label_optimizer/plateau.py` - Connected Components 기반 Plateau Search
- [x] `label_optimizer/optimizer.py` - Grid Search 오케스트레이터
- [x] `test_label_optimizer.py` - 단위 테스트 (48개)

#### Step 3.2: Fine-tuning
- [x] `trainer/dataset.py` - FineTuningDataset (Sample-level Drop)
- [x] `trainer/model.py` - PatchTSTClassifier
- [x] `trainer/loss.py` - Focal Loss
- [x] `trainer/train.py` - 2-Phase Fine-tuning
- [x] `test_trainer.py` - 단위 테스트 (37개)

#### Step 3.3: Execution & Backtest
- [ ] `wfa/backtester.py` - Realistic Backtester (Next Bar Entry)
- [ ] `wfa/engine.py` - Walk-Forward 메인 루프
- [ ] `wfa/metrics.py` - 성과 지표 계산
- [ ] `test_backtester.py` - 단위 테스트

### Utils
- [ ] `utils/fail_fast.py` - FailFastPolicy
- [ ] `utils/data_loader.py` - Feature 로더

### Integration
- [ ] End-to-End 테스트
- [ ] 성능 벤치마크

---

## 13. Success Metrics

| 지표 | 기준 | 의미 |
|------|------|------|
| Avg PnL/Trade | > 0.15% | 순수익 (비용 후) |
| Win Rate | > 45% | 승률 |
| Sharpe Ratio | > 1.5 | 위험 조정 수익 |
| Max Drawdown | < 20% | 최대 손실폭 |
| Profit Factor | > 1.5 | 총이익/총손실 |

---

## 부록 A: 핵심 알고리즘 요약

### A.1 Plateau Search

```python
def find_plateau_center(scores, grid_shape, threshold_pct=90, fallback_pct=80):
    scores_3d = scores.reshape(grid_shape)
    threshold = np.percentile(scores, threshold_pct)
    mask = (scores_3d >= threshold).astype(int)
    labeled, n = ndimage.label(mask)

    if n == 0:  # Fallback
        threshold = np.percentile(scores, fallback_pct)
        mask = (scores_3d >= threshold).astype(int)
        labeled, n = ndimage.label(mask)

    if n == 0:  # 최후 수단
        return np.unravel_index(np.argmax(scores), grid_shape)

    sizes = ndimage.sum(mask, labeled, range(1, n+1))
    largest = np.argmax(sizes) + 1
    center = ndimage.center_of_mass(mask, labeled, largest)

    return tuple(int(round(c)) for c in center)
```

### A.2 Rank IC

```python
def compute_rank_ic(features, labels):
    ics = [abs(spearmanr(features[:,i], labels)[0])
           for i in range(features.shape[1])
           if not np.isnan(spearmanr(features[:,i], labels)[0])]
    return np.mean(ics) if ics else 0.0
```

### A.3 Square Root Cost

```python
def get_slippage(volatility, trade_size, avg_volume, coeff=0.1):
    impact = coeff * volatility * np.sqrt(trade_size / (avg_volume + 1e-10))
    return min(0.0001 + impact, 0.005)  # base + impact, capped
```

### A.4 Next Bar Entry

```python
# Signal at bar t (using close[t])
# Entry at bar t+1 (using open[t+1])
if pending_signal is not None:
    entry_price = current_bar['open']  # NOT close
    # ... execute trade
```

---

**Document Version**: v4.6 (Re-structured)
**Last Updated**: 2026-01
**Status**: Ready for Implementation
