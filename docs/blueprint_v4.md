# Deep Learning Bitcoin Trading Strategy Blueprint v4

## 변경 이력
- v1: 초기 설계 (TIB 기반)
- v2: 실제 구현 반영 (Adaptive Dollar Bar 기반)
- v3: ETL + Feature Engineering 완료 반영 (Go 통합 파이프라인)
- **v4: Self-Supervised Learning + Dynamic Label Optimization 체제 전환**
- **v4.1: Critical Bug Fixes (Fee Trap, Dirty Zero, Overlap, Dynamic Slippage)**
- **v4.2: Advanced Fixes (Profitability Proxy, Nonlinear Cost, Execution Lag, Parameter Smoothing)**
- **v4.3: Multi-Asset Support (ETH, XRP, SOL, DOGE, BNB, LTC 추가)**

---

## 1. Goal Description

**목표**: Crypto 자동 매매 연구 시스템 개발
- **전략**: Mid-Frequency Trend Following (Long & Short)
- **핵심 제약**: Avg Profit per Trade > 0.15% (수수료/슬리피지 후)
- **설계 철학**: Modular, Config-Driven, File-Centric, TDD
- **대상 자산** (v4.3): BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT, DOGEUSDT, BNBUSDT, LTCUSDT

### v4.3 Multi-Asset 설계 원칙
| 항목 | 설명 |
|------|------|
| 데이터 수집 | 7개 심볼 동시 수집 |
| Dollar Bar | 심볼별 독립 Threshold (유동성 반영) |
| Feature | 심볼별 독립 Feature Parquet |
| 학습 | 심볼별 독립 모델 (Transfer Learning 고려) |
| 백테스트 | 심볼별 독립 실행 → 통합 리포트 |

### v4 핵심 변경점
| 항목 | v3 | v4 |
|------|----|----|
| 학습 방식 | 단순 지도학습 | **SSL Pre-training → Fine-tuning** |
| 라벨링 | 고정 파라미터 | **Dynamic Label Optimization (Grid Search)** |
| 파이프라인 | 단일 패스 | **Walk-Forward Adaptive Loop** |
| 파라미터 | 전역 고정 | **구간별 최적화** |

---

## 2. Design Philosophy

### Test-Driven Development (TDD)
- Go: 테이블 기반 테스트 + 엣지케이스 테스트
- Python: pytest + synthetic data fixtures
- 커버리지: 핵심 경로 > 90%

### File-Centric Architecture
- 모든 단계는 파일 입력 → 파일 출력
- 단계 간 인메모리 상태 없음
- "Resume from anywhere" 지원

### Config-Driven
- 모든 파라미터는 `config.yaml`에 정의
- 하드코딩 금지

### Streaming-First
- 전체 데이터셋 단일 for-loop 처리
- 모든 정규화는 `Update(value)` 재귀적/누적 방식
- Python에서는 `model.fit()`만 수행, 정규화 코드 없음
- 수치 안정성: Kahan Summation, Welford's Algorithm

### Adaptive Learning (v4 신규)
- **시장 국면 적응**: 각 Walk-Forward 윈도우에서 최적 라벨링 파라미터 재탐색
- **Self-Supervised Foundation**: 라벨 없이 시장 구조/문법 사전 학습
- **Transfer Learning**: SSL 가중치를 분류 태스크로 전이

---

## 3. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Walk-Forward Analysis Loop                           │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  For each (Train Window, Test Window) pair:                           │  │
│  │                                                                       │  │
│  │  [Stage 1]        [Stage 2]           [Stage 3]         [Stage 4]    │  │
│  │    ETL     →    Label Optimizer  →  Hybrid Trainer  →  WFA Engine    │  │
│  │    (Go)          (Python)            (Python)          (Python)      │  │
│  │                                                                       │  │
│  │  Trades →        Grid Search         SSL Pre-train     Signal Gen    │  │
│  │  Dollar Bars →   Best Params →       Fine-tune →       Execution     │  │
│  │  Features        TBM Labels          Classifier        Backtest      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 단계별 역할

| Stage | 이름 | 언어 | 입력 | 출력 |
|-------|------|------|------|------|
| 1 | ETL & Features | Go | Raw Trades | Feature Parquet |
| 2 | Label Optimizer | Python | Features + Grid Config | Best Params + Labels |
| 3 | Hybrid Trainer | Python | Features + Labels | Trained Model |
| 4 | WFA Engine | Python | Model + Test Data | Signals + Backtest Results |

---

## 4. Stage 1: ETL & Feature Engineering (Go) ✅ COMPLETED

> v3와 동일. 상세 내용은 `docs/blueprint_v3.md` Section 4 참조.

### 요약
- **입력**: Binance Raw Trades (ZIP)
- **출력**: Feature Parquet (28개 피처)
- **Dollar Bar**: `Threshold_t = EMA_14(DailyDollarVolume) / 50`
- **State Management**: Monthly Checkpoint System

### v4.3 Multi-Asset 지원

```yaml
# 지원 심볼
symbols:
  - symbol: "BTCUSDT"
    market: "futures"
    priority: 1        # 주력 자산 (가장 유동성 높음)
  - symbol: "ETHUSDT"
    market: "futures"
    priority: 1        # 주력 자산
  - symbol: "XRPUSDT"
    market: "futures"
    priority: 2        # 보조 자산
  - symbol: "SOLUSDT"
    market: "futures"
    priority: 2
  - symbol: "DOGEUSDT"
    market: "futures"
    priority: 3        # 변동성 높음
  - symbol: "BNBUSDT"
    market: "futures"
    priority: 2
  - symbol: "LTCUSDT"
    market: "futures"
    priority: 3
```

**심볼별 Dollar Bar Threshold 자동 조정:**
```
BTCUSDT: EMA_14(DailyDollarVolume) / 50   # 기준 (일 ~$30B)
ETHUSDT: EMA_14(DailyDollarVolume) / 50   # 유사 유동성
SOLUSDT: EMA_14(DailyDollarVolume) / 50   # 자동 조정
XRPUSDT: EMA_14(DailyDollarVolume) / 50   # 자동 조정
DOGEUSDT: EMA_14(DailyDollarVolume) / 50  # 자동 조정
BNBUSDT: EMA_14(DailyDollarVolume) / 50   # 자동 조정
LTCUSDT: EMA_14(DailyDollarVolume) / 50   # 자동 조정

# 모든 심볼은 동일한 공식 사용 → 유동성 자동 반영
```

### 피처 목록 (28개)
- L1 Stateless (9개): log_volume, log_tick_count, log_duration, log_trade_intensity, vwap_deviation, volume_imbalance, bar_range, bar_body, kyle_lambda
- Regime (8개): garman_klass_vol, realized_vol, shannon_entropy, vol_ratio, vol_zscore, entropy_zscore, skewness, kurtosis
- Stationarity (3개): frac_diff_close, detrended_log_price, returns
- Momentum (5개): vw_momentum, momentum_zscore_10/50/250/1000
- Technical (1개): connors_rsi
- Meta (1개): is_primed

### 파일 위치 (Multi-Asset)
- Bar 출력: `data/bars/{market}/{symbol}/{symbol}-bars-{year}-{month}.parquet`
- Feature 출력: `data/features/{market}/{symbol}/{symbol}-features-{year}-{month}.parquet`
- State 출력: `data/state/{market}/{symbol}/generator_state.json`

```
data/
├── raw/futures/
│   ├── BTCUSDT/           # 다운로드 임시 파일 (처리 후 삭제)
│   ├── ETHUSDT/
│   └── ...
├── bars/futures/
│   ├── BTCUSDT/
│   │   ├── BTCUSDT-bars-2024-01.parquet
│   │   ├── BTCUSDT-bars-2024-02.parquet
│   │   └── ...
│   ├── ETHUSDT/
│   │   ├── ETHUSDT-bars-2024-01.parquet
│   │   └── ...
│   └── ...
├── features/futures/
│   ├── BTCUSDT/
│   │   ├── BTCUSDT-features-2024-01.parquet
│   │   └── ...
│   ├── ETHUSDT/
│   │   └── ...
│   └── ...
└── state/futures/
    ├── BTCUSDT/
    │   ├── generator_state.json    # Bar generator state
    │   └── feature_state.json      # Feature generator state
    ├── ETHUSDT/
    │   ├── generator_state.json
    │   └── feature_state.json
    └── ...
```

---

## 5. Stage 2: Label Optimizer (Python) 📝 NEW

### 5.1 개요

학습 전 **"어떤 Triple Barrier 파라미터가 현재 구간에서 가장 학습하기 좋은 라벨을 생성하는가?"** 를 수학적으로 탐색하는 단계.

```
┌─────────────────────────────────────────────────────────────────┐
│                  Label Optimizer (v4.1)                         │
│                                                                 │
│  Features ──┬──▶ Grid Search ──▶ Per-Bar Fee Trap Filter       │
│             │         │                │                        │
│  Grid       │    ┌────┴────┐     ┌─────┴─────┐                 │
│  Config ────┘    │ For each│     │ Dynamic   │                 │
│                  │ (SL,PT, │     │ Cost Model│                 │
│                  │  Time)  │     └─────┬─────┘                 │
│                  └────┬────┘           │                        │
│                       │                │                        │
│                       ▼                ▼                        │
│              ┌────────────────────────────────┐                │
│              │  TBM Labeling (Clean Labels)   │                │
│              │  +1: Long PT, 0: Timeout,      │                │
│              │  -1: Short PT                  │                │
│              │  DROP: Fee Trap (학습 제외)    │                │
│              └───────────┬────────────────────┘                │
│                          │                                      │
│                          ▼                                      │
│              ┌────────────────────────────────┐                │
│              │ Purged K-Fold + Embargo        │                │
│              │ (Overlap 제거)                 │                │
│              └───────────┬────────────────────┘                │
│                          │                                      │
│                          ▼                                      │
│              Select Best Params + Valid Mask                    │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Critical Design Principles (v4.1)

#### ⚠️ Principle 1: Feature Continuity (시계열 연속성 보존)

**절대 규칙: Feature Parquet는 수정하지 않는다**

```
❌ 잘못된 순서 (시계열 연속성 파괴):
   Raw Bars → Fee Trap Drop → Feature Generation

   문제: t, t+1, t+2, [t+3 DROP], [t+4 DROP], t+5...
   - RSI(14) 계산 시 중간 바 누락으로 오류
   - Lag Features가 실제로는 다른 시점을 참조
   - Rolling Window 내 시간 간격 불균일

✅ 올바른 순서 (Feature 연속성 보존):
   Stage 1 (Go): Raw Bars → ALL Features (연속 시계열)
   Stage 2 (Python): Features → TBM Labeling → Valid Mask 생성
   Stage 3 (Python): Features[valid_mask] + Labels[valid_mask]

   핵심: Drop은 "인덱스 마스크"로만 처리, 원본 불변
```

#### ⚠️ Principle 2: Clean Labels (Dirty Zero 방지)

```
❌ 오염된 Label 0 (모델 혼란 유발):
   Label 0 = {
       Timeout: 가격 횡보 → "기다려"
       Fee Trap: 변동성 부족 → "시장 떠나"
       Whipsaw: PT/SL 동시 터치 → "조심해"
   }
   → 서로 다른 의미가 같은 클래스로 학습됨 → Decision Boundary 왜곡

✅ Clean Labels (의미 분리):
   +1: Long PT 먼저 도달 (Long 추천)
    0: Timeout만 (순수 방향성 실패)
   -1: Short PT 먼저 도달 (Short 추천)

   DROP (학습 데이터에서 완전 제외):
   - Fee Trap: 거래 가치 없음 (수익 < 비용)
   - Duration Exceeded: 비정상 바 (72시간 초과)
```

#### ⚠️ Principle 3: Per-Bar Fee Trap Guard (평균의 함정 방지)

```
❌ 평균 σ 기반 필터링 (치명적 오류):
   avg_sigma = data['realized_vol'].mean()
   if pt * avg_sigma > cost: viable  # WRONG!

   문제: 변동성은 군집(Cluster) 현상
   - 2021-05 σ = 8% (폭락장)
   - 2023-09 σ = 1% (횡보장)
   - 평균 4.5%로 통과했지만, 저변동성 구간에서 손실

✅ Per-Bar 필터링 (현재 σ 기반):
   for each bar:
       current_sigma = bar.realized_vol
       if pt * current_sigma < dynamic_cost(current_sigma):
           DROP this sample  # 이 바는 학습 제외
```

#### ⚠️ Principle 4: Overlap Handling (Serial Correlation 제거)

```
❌ 중복 데이터 문제:
   vertical_bars = 100일 때
   t=0 라벨: bars[1:100] 참조
   t=1 라벨: bars[2:101] 참조
   → 99% 겹침 → CV Score 과대평가

✅ 해결책:
   Option A: Purged K-Fold + Embargo
   - Train/Test 경계에서 중복 제거
   - Test 직후 embargo_pct만큼 추가 제거

   Option B: Non-Overlapping Sampling
   - vertical_bars 간격으로만 샘플 생성
   - 데이터 손실 있지만 완전한 독립성
```

### 5.3 Nonlinear Cost Model (v4.2 Updated)

> ⚠️ **v4.2 변경**: 선형 모델 → 비선형 모델 (Square Root Law)
> - 실제 Market Impact는 Square Root Law를 따름
> - Volume/Orderbook Depth 고려

```python
@dataclass
class NonlinearCostModel:
    """
    v4.2: 비선형 Market Impact 모델

    참고: Almgren-Chriss, "Optimal Execution of Portfolio Transactions"

    v4.1 문제점:
    - slippage = base + α × σ (선형) → 실제와 괴리
    - Volume 고려 없음 → 저거래량 구간 과소평가

    v4.2 개선:
    - Square Root Law 적용
    - Volume 기반 페널티 추가
    """

    # 기본 비용
    base_fee: float = 0.001        # 0.10% (거래소 수수료)
    base_slippage: float = 0.0001  # 0.01% (최소 슬리피지)

    # 비선형 파라미터
    impact_exponent: float = 0.5       # Square Root Law (0.5)
    volatility_multiplier: float = 0.3 # σ 계수
    volume_penalty: float = 0.1        # 저거래량 페널티
    slippage_cap: float = 0.005        # 최대 0.50%

    # Fee Trap 판정
    min_profit_buffer: float = 1.5

    def get_total_cost(
        self,
        sigma: float,
        volume_ratio: float = 1.0,  # 현재 거래량 / 평균 거래량
        position_pct: float = 0.01  # 포지션이 일 거래량의 몇 %인지
    ) -> float:
        """
        Market Impact = η × σ × (Q / V)^0.5

        Q: Order Size
        V: Average Daily Volume
        η: Impact coefficient
        σ: Volatility

        예시:
        - 정상 거래량 (ratio=1.0): 0.10% + 0.03% = 0.13%
        - 거래량 급감 (ratio=0.3): 0.10% + 0.03% + 0.07% = 0.20%
        - 고변동성 + 저거래량: 최대 0.50%
        """
        # 1. 기본 거래소 수수료
        fee = self.base_fee

        # 2. 변동성 기반 슬리피지 (Square Root Law)
        vol_slip = self.volatility_multiplier * sigma * (position_pct ** self.impact_exponent)

        # 3. 저거래량 페널티
        if volume_ratio < 0.5:
            vol_penalty = self.volume_penalty * (1 - volume_ratio * 2)
        else:
            vol_penalty = 0

        # 4. 총 비용 (capped)
        total = fee + self.base_slippage + vol_slip + vol_penalty
        return min(total, self.slippage_cap)

    def get_cost_for_bar(self, bar: pd.Series, avg_volume: float) -> float:
        """단일 바에 대한 비용 계산"""
        volume_ratio = bar['volume'] / (avg_volume + 1e-10)
        return self.get_total_cost(
            sigma=bar['realized_vol'],
            volume_ratio=volume_ratio
        )

    def is_viable_trade(
        self,
        pt_mult: float,
        sigma: float,
        volume_ratio: float = 1.0
    ) -> bool:
        """
        Per-Bar Fee Trap Guard

        Expected Profit = PT × σ
        Required = Total Cost × Buffer (1.5x)
        """
        expected_profit = pt_mult * sigma
        required = self.get_total_cost(sigma, volume_ratio) * self.min_profit_buffer
        return expected_profit > required
```

**v4.1 → v4.2 비교:**
```
v4.1 (선형):
  Cost(σ=2%, vol=normal) = 0.10% + 0.01% + 0.5×2% = 0.21%
  Cost(σ=2%, vol=low)    = 0.21% (동일 - 문제!)

v4.2 (비선형):
  Cost(σ=2%, vol=normal) = 0.10% + 0.01% + 0.06% = 0.17%
  Cost(σ=2%, vol=low)    = 0.10% + 0.01% + 0.06% + 0.07% = 0.24%
```

### 5.4 파라미터 탐색 공간 (Grid Search Space)

| 파라미터 | 탐색 범위 | 단위 | 설명 |
|----------|-----------|------|------|
| **SL (Stop Loss)** | `[1.0, 2.0, 3.0]` | σ (realized_vol) | 손절 폭 |
| **PT (Profit Target)** | `[1.5, 2.0, 2.5, 3.0, 3.5]` | σ (realized_vol) | 익절 폭 |
| **Time (Vertical Barrier)** | `[50, 100, 200, 500, 1000, 1500, 2000]` | Bars | 만기 |

**총 조합 수**: 3 × 5 × 7 = **105개**

### 5.5 Triple Barrier Method (TBM)

```
         PT (Profit Target)
          ↑
    ──────┼──────  ← Entry
          ↓
         SL (Stop Loss)

    |←─────────→|
    Vertical Barrier
```

**Barrier 계산:**
```python
sigma = realized_vol[t]  # 이미 Feature로 계산됨
PT = entry_price * (1 + n_pt * sigma)
SL = entry_price * (1 - n_sl * sigma)
vertical_barrier = t + n_bars  # 바 수 기준
```

**레이블:**
- **+1 (Long)**: PT 먼저 도달
- **-1 (Short)**: SL 먼저 도달
- **0 (Neutral)**: Vertical barrier 도달 (타임아웃)

**HighTime/LowTime 활용:**
```python
# 바 내에서 PT/SL 도달 순서 정확히 판단
if bar.high_time < bar.low_time:
    # 가격이 올랐다가 내려감 → Long TP 먼저 체결 가능성
else:
    # 가격이 내려갔다가 올라감 → Short TP 먼저 체결 가능성
```

### 5.6 Purged K-Fold with Embargo

중복 데이터 문제 해결을 위한 검증 분할 전략.

```python
class PurgedKFold:
    """
    Purging: Train/Test 경계에서 라벨이 참조하는 미래 데이터 중복 제거
    Embargo: Test 직후 N개 샘플 추가 제거 (안전 마진)

    Reference: de Prado, "Advances in Financial Machine Learning", Ch.7
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,  # 전체의 1%
        vertical_bars: int = 100    # TBM 만기
    ):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.vertical_bars = vertical_bars

    def split(self, X, y, timestamps):
        """
        Args:
            timestamps: 각 샘플의 이벤트 시작 시점
        """
        n = len(X)
        embargo_size = int(n * self.embargo_pct)
        fold_size = n // self.n_splits

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n

            # Test 구간
            test_idx = np.arange(test_start, test_end)

            # ===== Purging =====
            # Train 샘플 중, 라벨이 Test 구간을 침범하는 것 제거
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_start:test_end] = False

            test_min_t = timestamps[test_start]
            for j in range(test_start):
                label_end_t = timestamps[min(j + self.vertical_bars, n - 1)]
                if label_end_t >= test_min_t:
                    train_mask[j] = False  # 이 샘플의 라벨이 Test와 겹침

            # ===== Embargo =====
            # Test 직후 embargo_size 만큼 추가 제거
            embargo_end = min(test_end + embargo_size, n)
            train_mask[test_end:embargo_end] = False

            train_idx = np.where(train_mask)[0]

            yield train_idx, test_idx
```

### 5.7 검증 지표 (Scoring Metrics) - v4.2 Updated

각 파라미터 조합에 대해 3가지 지표를 계산하고 종합 점수를 산출한다.

> ⚠️ **v4.2 변경**: Stationarity 제거, Profitability Proxy 추가
> - Stationarity 높음 = Trend 과적합 위험
> - Profitability = 실제 수익 가능성 검증

#### 5.7.1 Entropy (클래스 균형도)

```python
def compute_entropy(labels: np.ndarray) -> float:
    """
    클래스 분포의 Shannon Entropy 계산.
    최대값 = log(3) ≈ 1.585 (완전 균형)
    """
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return -np.sum(probs * np.log(probs + 1e-10))
```

- **목표**: 최대화 (균형 잡힌 분포)
- **탈락 조건**: `entropy < 1.0` (한쪽으로 심하게 쏠림)
- **주의**: Entropy만 높으면 Random Walk 선택 위험 → MI와 함께 봐야 함

#### 5.7.2 Mutual Information (Feature-Label 연관성)

```python
from sklearn.feature_selection import mutual_info_classif

def compute_mi(X: np.ndarray, y: np.ndarray) -> float:
    """
    Feature와 Label 간의 Mutual Information.
    높을수록 Feature가 Label 예측에 유용함.
    """
    mi_scores = mutual_info_classif(X, y, discrete_features=False)
    return np.mean(mi_scores)
```

- **목표**: 최대화
- **의미**: Feature가 Label을 잘 설명하는 정도
- **핵심**: Entropy 높음 + MI 낮음 = 노이즈 (탈락)

#### 5.7.3 Profitability Proxy (v4.2 NEW - Stationarity 대체)

```python
def compute_profitability_proxy(
    data: pd.DataFrame,
    labels: np.ndarray,
    valid_indices: np.ndarray,
    forward_bars: int = 50
) -> float:
    """
    라벨이 맞았을 때 실제 수익률 계산 (Sharpe-like)

    v4.2 NEW: Stationarity 대신 실제 수익 가능성 측정
    - 아무리 학습하기 좋은 분포라도 수익이 안 나면 무의미
    - Label 방향대로 가격이 움직였을 때의 평균 수익률

    Returns:
        profit_ratio: mean(profit) / std(profit) (높을수록 좋음)
    """
    profits = []

    for i, idx in enumerate(valid_indices):
        label = labels[i]
        if label == 0:  # Neutral은 제외
            continue

        entry = data.iloc[idx]['close']
        future_idx = min(idx + forward_bars, len(data) - 1)
        exit_price = data.iloc[future_idx]['close']

        if label == 1:  # Long
            ret = (exit_price - entry) / entry
        else:  # Short (label == -1)
            ret = (entry - exit_price) / entry

        profits.append(ret)

    if len(profits) < 10:
        return 0.0

    mean_profit = np.mean(profits)
    std_profit = np.std(profits) + 1e-10

    # Sharpe-like ratio
    return mean_profit / std_profit
```

- **목표**: 최대화
- **의미**: 라벨 방향대로 진입했을 때 실제 수익 가능성
- **v4.2 핵심**: "학습하기 좋음" + "수익 가능성" 동시 만족

#### 5.7.4 종합 점수 (Composite Score) - v4.2 Updated

```python
def compute_composite_score(
    entropy: float,
    mi: float,
    profitability: float,  # v4.2: Stationarity → Profitability
    weights: dict = {"entropy": 0.2, "mi": 0.4, "profitability": 0.4}
) -> float:
    """
    v4.2 Updated: Profitability Proxy 추가, Stationarity 제거

    가중치 변경:
    - entropy: 0.3 → 0.2 (보조 지표로 격하)
    - mi: 0.5 → 0.4
    - profitability: 0.0 → 0.4 (신규, 핵심 지표)
    - stationarity: 0.2 → 0.0 (제거)
    """
    # Min-Max 정규화 (Grid 내에서)
    score = (
        weights["entropy"] * norm_entropy +
        weights["mi"] * norm_mi +
        weights["profitability"] * norm_profitability
    )
    return score
```

**v4.2 점수 해석:**
```
High Entropy + High MI + High Profitability = 최적
High Entropy + Low MI = Random Walk (탈락)
High MI + Low Profitability = 예측은 되지만 수익 불가 (탈락)
```

### 5.8 알고리즘 의사 코드 (Corrected)

```python
def optimize_labels(
    features: pd.DataFrame,
    config: dict,
    cost_model: DynamicCostModel
) -> OptimizationResult:
    """
    Grid Search를 통한 최적 TBM 파라미터 탐색.
    v4.1: Per-Bar Fee Trap Guard + Clean Labels
    """
    grid = product(
        config["sl_range"],      # [1.0, 2.0, 3.0]
        config["pt_range"],      # [1.5, 2.0, 2.5, 3.0, 3.5]
        config["time_range"]     # [50, 100, 200, 500, 1000, 1500, 2000]
    )

    results = []
    for sl, pt, time in grid:
        # 1. TBM 라벨링 (Per-Bar Fee Trap 적용)
        labeler = TripleBarrierLabeler(
            TBMConfig(sl_mult=sl, pt_mult=pt, vertical_bars=time),
            cost_model=cost_model
        )
        valid_indices, labels = labeler.get_events_and_labels(features)

        # 유효 샘플이 너무 적으면 스킵
        if len(labels) < config["min_valid_samples"]:
            continue

        # 2. Valid 샘플에 대해서만 점수 계산
        valid_features = features.iloc[valid_indices][FEATURE_COLS].values
        entropy = compute_entropy(labels)

        # Early rejection: 극단적 불균형
        if entropy < config["min_entropy"]:
            continue

        mi = compute_mi(valid_features, labels)
        stationarity = compute_stationarity(labels)

        # Valid ratio도 기록 (참고용)
        valid_ratio = len(labels) / len(features)

        results.append({
            "sl": sl, "pt": pt, "time": time,
            "entropy": entropy, "mi": mi, "stationarity": stationarity,
            "valid_ratio": valid_ratio, "n_samples": len(labels)
        })

    # 3. 종합 점수 계산 및 정렬
    df = pd.DataFrame(results)
    df["score"] = compute_composite_score(df)
    best = df.sort_values("score", ascending=False).iloc[0]

    # 4. 최적 파라미터로 최종 라벨 생성
    final_labeler = TripleBarrierLabeler(
        TBMConfig(sl_mult=best["sl"], pt_mult=best["pt"], vertical_bars=best["time"]),
        cost_model=cost_model
    )
    final_valid_indices, final_labels = final_labeler.get_events_and_labels(features)

    return OptimizationResult(
        best_params={"sl": best["sl"], "pt": best["pt"], "time": best["time"]},
        valid_indices=final_valid_indices,  # 학습에 사용할 인덱스
        labels=final_labels,
        scores=df
    )
```

### 5.9 출력

```
data/labels/
├── optimization_results/
│   └── {fold_id}_grid_search.parquet    # 모든 조합의 점수
├── labeled/
│   └── {fold_id}_labels.parquet         # 최종 라벨 (valid만)
└── masks/
    └── {fold_id}_valid_mask.npy         # Valid 인덱스 (원본 Feature와 매핑)
```

**중요**: `valid_mask`를 통해 원본 Feature Parquet의 어떤 행이 학습에 사용되는지 추적

---

## 6. Stage 3: Hybrid Trainer (Python) 📝 NEW

### 6.1 개요

2단계 하이브리드 학습 파이프라인:
1. **Step A: SSL Pre-training** - 라벨 없이 시장 구조 학습
2. **Step B: Fine-tuning** - 분류 헤드 부착 후 지도 학습

```
┌─────────────────────────────────────────────────────────────────┐
│                     Hybrid Trainer                              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Step A: SSL Pre-training (Self-Supervised)             │   │
│  │                                                         │   │
│  │  Input: X (Features only, no labels)                    │   │
│  │                                                         │   │
│  │  ┌─────────┐    ┌──────────┐    ┌─────────────────┐    │   │
│  │  │ Masking │ -> │ PatchTST │ -> │ Reconstruction  │    │   │
│  │  │ (15%)   │    │ Encoder  │    │ Loss (MSE)      │    │   │
│  │  └─────────┘    └──────────┘    └─────────────────┘    │   │
│  │                                                         │   │
│  │  Output: Pre-trained Encoder Weights                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Step B: Fine-tuning (Supervised)                       │   │
│  │                                                         │   │
│  │  Input: X (Features) + Y (Labels from Stage 2)          │   │
│  │                                                         │   │
│  │  ┌──────────┐    ┌────────────┐    ┌──────────────┐    │   │
│  │  │ PatchTST │ -> │ Classifier │ -> │ Focal Loss   │    │   │
│  │  │ Encoder  │    │ Head       │    │ (3-class)    │    │   │
│  │  │(frozen/  │    │            │    │              │    │   │
│  │  │ unfrozen)│    │            │    │              │    │   │
│  │  └──────────┘    └────────────┘    └──────────────┘    │   │
│  │                                                         │   │
│  │  Output: Trained Classifier                             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Step A: SSL Pre-training

#### 목적
라벨 없이 Feature 데이터($X$)만으로 시장의 구조적 패턴과 변동성 문법을 학습.

#### Task: Masked Time Series Modeling

```python
class MaskedTimeSeriesDataset(Dataset):
    def __init__(self, features: np.ndarray, mask_ratio: float = 0.15):
        self.features = features  # (N, C, L) - N samples, C channels, L length
        self.mask_ratio = mask_ratio

    def __getitem__(self, idx):
        x = self.features[idx].copy()  # (C, L)

        # 패치 단위 마스킹
        n_patches = x.shape[1] // PATCH_LEN
        n_mask = int(n_patches * self.mask_ratio)
        mask_indices = np.random.choice(n_patches, n_mask, replace=False)

        # 마스크 토큰으로 대체 (learnable)
        x_masked = x.copy()
        for idx in mask_indices:
            start = idx * PATCH_LEN
            end = start + PATCH_LEN
            x_masked[:, start:end] = MASK_TOKEN

        return {
            "input": torch.tensor(x_masked, dtype=torch.float32),
            "target": torch.tensor(x, dtype=torch.float32),
            "mask_indices": mask_indices
        }
```

#### 모델 구조 (Pre-training)

```
Input (C=28, L=512)
    ↓
Patching (patch_len=16, stride=16) → (C, N_patches=32, patch_len)
    ↓
Linear Embedding (patch_len → d_model=128)
    ↓
Positional Embedding (learnable)
    ↓
Transformer Encoder (heads=4, layers=2)
    ↓
Reconstruction Head: Linear(d_model → patch_len)
    ↓
Output: Reconstructed patches
```

#### Loss: MSE (Masked Patches Only)

```python
def ssl_loss(pred, target, mask_indices):
    """마스킹된 패치에 대해서만 reconstruction loss 계산"""
    loss = 0
    for i, indices in enumerate(mask_indices):
        for idx in indices:
            start = idx * PATCH_LEN
            end = start + PATCH_LEN
            loss += F.mse_loss(pred[i, :, start:end], target[i, :, start:end])
    return loss / len(mask_indices)
```

### 6.3 Step B: Fine-tuning

#### 목적
SSL로 학습된 Encoder 가중치를 활용하여 3-Class Classification 수행.

#### 모델 구조 (Fine-tuning)

```
Input (C=28, L=512)
    ↓
[Pre-trained PatchTST Encoder]  ← Load SSL weights
    ↓
Flatten (N_patches × d_model)
    ↓
Residual Bottleneck Head:
    residual = x
    x = Dense(flatten_dim → 256) → GELU → Dropout(0.3)
    x = Dense(256 → flatten_dim)
    x = residual + x
    ↓
Classification: Dense(flatten_dim → 3) → Softmax
    ↓
Output: Class probabilities [Short, Neutral, Long]
```

#### Fine-tuning 전략

```python
class FineTuningStrategy:
    """
    2-Phase Fine-tuning:
    1. Encoder Frozen: 분류 헤드만 학습 (5 epochs)
    2. Full Unfrozen: 전체 모델 미세조정 (나머지 epochs)
    """

    def __init__(self, model, frozen_epochs: int = 5):
        self.model = model
        self.frozen_epochs = frozen_epochs

    def configure_optimizers(self, epoch: int):
        if epoch < self.frozen_epochs:
            # Phase 1: Encoder frozen
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            params = self.model.classifier.parameters()
            lr = 1e-3
        else:
            # Phase 2: All unfrozen
            for param in self.model.parameters():
                param.requires_grad = True
            params = [
                {"params": self.model.encoder.parameters(), "lr": 1e-5},
                {"params": self.model.classifier.parameters(), "lr": 1e-4}
            ]
            lr = None

        return torch.optim.AdamW(params, lr=lr)
```

#### Loss: Focal Loss

```python
class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.
    FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # Label smoothing
        n_classes = logits.size(-1)
        smooth_targets = (1 - self.label_smoothing) * F.one_hot(targets, n_classes)
        smooth_targets += self.label_smoothing / n_classes

        # Focal weight
        probs = F.softmax(logits, dim=-1)
        pt = (probs * smooth_targets).sum(dim=-1)
        focal_weight = (1 - pt) ** self.gamma

        # Cross entropy
        ce_loss = -torch.sum(smooth_targets * F.log_softmax(logits, dim=-1), dim=-1)

        return (focal_weight * ce_loss).mean()
```

### 6.4 학습 파이프라인

```python
def hybrid_training(
    features: np.ndarray,
    labels: np.ndarray,
    config: TrainingConfig
) -> TrainedModel:
    """
    2-Stage Hybrid Training Pipeline.
    """
    # ============== Step A: SSL Pre-training ==============
    print("Step A: SSL Pre-training...")

    ssl_dataset = MaskedTimeSeriesDataset(features, mask_ratio=0.15)
    ssl_loader = DataLoader(ssl_dataset, batch_size=config.batch_size, shuffle=True)

    encoder = PatchTSTEncoder(
        n_channels=28,
        context_len=512,
        patch_len=16,
        d_model=128,
        n_heads=4,
        n_layers=2
    )

    ssl_model = SSLModel(encoder)
    ssl_trainer = Trainer(
        max_epochs=config.ssl_epochs,
        precision="16-mixed",
        callbacks=[EarlyStopping(monitor="val_loss", patience=5)]
    )
    ssl_trainer.fit(ssl_model, ssl_loader)

    # ============== Step B: Fine-tuning ==============
    print("Step B: Fine-tuning...")

    # Load pre-trained weights
    classifier = PatchTSTClassifier(encoder, n_classes=3)
    classifier.encoder.load_state_dict(ssl_model.encoder.state_dict())

    # Prepare supervised dataset
    ft_dataset = SupervisedDataset(features, labels)
    train_loader, val_loader = create_loaders(ft_dataset, config)

    ft_strategy = FineTuningStrategy(classifier, frozen_epochs=5)
    ft_trainer = Trainer(
        max_epochs=config.ft_epochs,
        precision="16-mixed",
        callbacks=[
            EarlyStopping(monitor="val_f1", patience=10, mode="max"),
            ModelCheckpoint(monitor="val_f1", mode="max")
        ]
    )
    ft_trainer.fit(classifier, train_loader, val_loader)

    return classifier
```

### 6.5 Lightweight Model Alternative (v4.2 NEW)

> ⚠️ **v4.2 추가**: 데이터 부족 시 경량 모델 대안
> - Transformer는 데이터가 많이 필요 (Data Hungry)
> - Purge + Drop + Embargo로 유효 데이터 감소
> - Walk-Forward 윈도우에서 Transformer 수렴 불충분 가능

#### 문제: Data Hunger vs. Purged CV

```
Walk-Forward 12개월 윈도우 분석:

원본: ~18,000 bars (50 bars/day × 365)
- Fee Trap Drop: -20~40% → ~12,000 bars
- Purge: -5~10% → ~11,000 bars
- Embargo: -1% → ~10,900 bars

유효 학습 데이터: ~10,000 bars
Context Window: 512 bars
실제 샘플 수: ~10,000 - 512 = ~9,500

Transformer 학습에 충분한가? → 데이터 부족 시 과적합 위험
```

#### 대안: TCN-Attention 경량 모델

```python
class TCNAttentionClassifier(nn.Module):
    """
    v4.2: 데이터 부족 시 대안 모델

    - Temporal Convolutional Network + Attention
    - Transformer 대비 파라미터 수 1/10
    - 적은 데이터에서도 안정적 수렴
    """

    def __init__(
        self,
        n_features: int = 28,
        seq_len: int = 512,
        tcn_channels: list = [64, 64, 64],
        kernel_size: int = 7,
        n_heads: int = 4,
        dropout: float = 0.2,
        n_classes: int = 3
    ):
        super().__init__()

        # 1. Temporal Convolutional Network
        self.tcn = TemporalConvNet(
            num_inputs=n_features,
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # 2. Self-Attention over TCN output
        hidden_dim = tcn_channels[-1]
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # 3. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )

    def forward(self, x):
        # x: (B, C, L) - Batch, Channels, Length
        out = self.tcn(x)  # (B, hidden, L)
        out = out.permute(0, 2, 1)  # (B, L, hidden)

        # Self-Attention
        out, _ = self.attention(out, out, out)

        # Global Average Pooling
        out = out.mean(dim=1)  # (B, hidden)

        return self.classifier(out)


class TemporalConvNet(nn.Module):
    """Dilated Causal Convolution Network"""

    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                dilation=dilation, dropout=dropout
            ))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
```

#### 모델 자동 선택

```python
def select_model(n_samples: int, config: dict) -> nn.Module:
    """
    데이터 크기에 따라 자동으로 모델 선택

    - 50,000+ samples: PatchTST (Transformer)
    - 10,000~50,000: TCN-Attention (경량)
    - <10,000: 학습 불가 경고
    """
    MIN_SAMPLES_TRANSFORMER = 50000
    MIN_SAMPLES_TCN = 10000

    if n_samples >= MIN_SAMPLES_TRANSFORMER:
        print(f"Using PatchTST (n_samples={n_samples})")
        return PatchTSTClassifier(
            n_channels=config['n_channels'],
            context_len=config['context_window'],
            **config['patchtst']
        )
    elif n_samples >= MIN_SAMPLES_TCN:
        print(f"Using TCN-Attention (n_samples={n_samples})")
        return TCNAttentionClassifier(
            n_features=config['n_channels'],
            seq_len=config['context_window'],
            **config['tcn_attention']
        )
    else:
        raise ValueError(
            f"Insufficient data: {n_samples} < {MIN_SAMPLES_TCN}. "
            "Consider expanding training window."
        )
```

**모델 비교:**
```
| 항목 | PatchTST | TCN-Attention |
|------|----------|---------------|
| 파라미터 | ~2M | ~200K |
| 최소 데이터 | 50,000 | 10,000 |
| 학습 속도 | 느림 | 빠름 |
| 장기 패턴 | 우수 | 보통 |
| 단기 반전 | 보통 | 우수 |
```

### 6.6 출력

```
artifacts/
├── ssl/
│   └── {fold_id}_encoder.pt          # SSL Pre-trained encoder
└── classifier/
    └── {fold_id}_model.pt            # Fine-tuned classifier (PatchTST or TCN)
```

---

## 7. Stage 4: Walk-Forward Analysis Engine (Python) 📝 NEW

### 7.1 개요

전체 파이프라인을 **Rolling Window 방식**으로 반복 실행하여 시장 변화에 적응하는 시스템.

```
Time ────────────────────────────────────────────────────────────────▶

Fold 1:  [====== Train Window ======][= Test =]
Fold 2:       [====== Train Window ======][= Test =]
Fold 3:            [====== Train Window ======][= Test =]
Fold 4:                 [====== Train Window ======][= Test =]
...

각 Fold에서:
1. Train Window로 Grid Search → Best Params 도출
2. Best Params로 라벨링
3. SSL Pre-training → Fine-tuning
4. Test Window에서 신호 생성 및 백테스트
```

### 7.2 Walk-Forward 구성

```python
@dataclass
class WalkForwardConfig:
    train_months: int = 12      # 학습 윈도우 크기
    test_months: int = 1        # 테스트 윈도우 크기
    step_months: int = 1        # 슬라이딩 간격
    min_train_bars: int = 10000 # 최소 학습 데이터 수
```

### 7.3 메인 루프

```python
class WalkForwardEngine:
    """
    Walk-Forward Analysis를 제어하는 메인 엔진.
    """

    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.results = []

    def run(self, features_path: str, output_path: str):
        """
        전체 Walk-Forward Loop 실행.
        """
        # 1. 전체 기간의 Feature 로드
        features = load_features(features_path, filter_primed=True)

        # 2. Fold 경계 생성
        folds = self._generate_folds(features)

        for fold_id, (train_start, train_end, test_start, test_end) in enumerate(folds):
            print(f"\n{'='*60}")
            print(f"Fold {fold_id}: Train [{train_start} ~ {train_end}], Test [{test_start} ~ {test_end}]")
            print(f"{'='*60}")

            # 3. 데이터 분할
            train_data = features[(features.index >= train_start) & (features.index < train_end)]
            test_data = features[(features.index >= test_start) & (features.index < test_end)]

            if len(train_data) < self.config.min_train_bars:
                print(f"Skipping fold {fold_id}: insufficient training data")
                continue

            # 4. Stage 2: Label Optimization
            opt_result = label_optimizer.optimize_labels(train_data, self.config.grid_config)
            best_params = opt_result.best_params
            train_labels = opt_result.labels

            print(f"Best Params: SL={best_params['sl']}, PT={best_params['pt']}, Time={best_params['time']}")

            # 5. Stage 3: Hybrid Training
            model = hybrid_trainer.train(
                features=train_data[FEATURE_COLS].values,
                labels=train_labels,
                config=self.config.training_config
            )

            # 6. Stage 4a: Signal Generation (Test Window)
            test_features = test_data[FEATURE_COLS].values
            signals = self._generate_signals(model, test_features)

            # 7. Stage 4b: Backtest with Best Params
            backtest_result = self._backtest(
                test_data,
                signals,
                sl_mult=best_params['sl'],
                pt_mult=best_params['pt']
            )

            # 8. 결과 저장
            self.results.append({
                "fold_id": fold_id,
                "train_period": (train_start, train_end),
                "test_period": (test_start, test_end),
                "best_params": best_params,
                "metrics": backtest_result.metrics
            })

            # Checkpoint 저장
            self._save_checkpoint(fold_id, model, opt_result, backtest_result)

        # 9. 종합 결과 분석
        self._summarize_results()

    def _generate_signals(self, model: nn.Module, features: np.ndarray) -> np.ndarray:
        """
        모델을 사용하여 신호 생성.
        Output: -1 (Short), 0 (Neutral), +1 (Long)
        """
        model.eval()
        with torch.no_grad():
            # Sliding window로 context_len만큼씩 입력
            signals = []
            for i in range(len(features) - self.config.context_len + 1):
                x = features[i:i + self.config.context_len]
                x = torch.tensor(x.T, dtype=torch.float32).unsqueeze(0)  # (1, C, L)

                logits = model(x)
                pred = torch.argmax(logits, dim=-1).item()

                # 0, 1, 2 → -1, 0, +1
                signal = pred - 1
                signals.append(signal)

            return np.array(signals)

    def _backtest(
        self,
        data: pd.DataFrame,
        signals: np.ndarray,
        sl_mult: float,
        pt_mult: float
    ) -> BacktestResult:
        """
        신호 기반 백테스트 실행.
        Train에서 찾은 SL/PT 배수를 적용.
        """
        backtester = Backtester(
            fee_rate=self.config.fee_rate,
            slippage=self.config.slippage,
            risk_per_trade=self.config.risk_per_trade
        )

        return backtester.run(
            data=data,
            signals=signals,
            sl_multiplier=sl_mult,
            pt_multiplier=pt_mult,
            volatility_col="realized_vol"
        )
```

### 7.4 백테스트 로직 (v4.2 Updated - Execution Lag)

> ⚠️ **v4.2 핵심 수정**: Execution Lag 반영
> - Signal은 Close[t] 확정 후 생성됨
> - 진입은 Open[t+1] 이후에만 가능
> - Dollar Bar 특성상 시간 간격이 불규칙 → 물리적 시간 지연 고려

```python
class RealisticBacktester:
    """
    v4.2: Execution Lag 반영 백테스터

    주요 변경:
    1. entry_delay: 신호 후 N바 대기
    2. entry_price: Close[t] → Open[t+entry_delay]
    3. 물리적 시간 최소 지연 옵션
    """

    def __init__(
        self,
        cost_model: NonlinearCostModel,
        initial_capital: float = 100000,
        risk_per_trade: float = 0.02,
        entry_delay: int = 1,           # v4.2: 진입 지연 (바 수)
        entry_price_type: str = "open", # "open", "twap", "vwap"
        min_physical_delay: int = 60    # v4.2: 최소 물리적 지연 (초)
    ):
        self.cost_model = cost_model
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.entry_delay = entry_delay
        self.entry_price_type = entry_price_type
        self.min_physical_delay = min_physical_delay

    def run(
        self,
        data: pd.DataFrame,
        signals: np.ndarray,
        sl_multiplier: float,
        pt_multiplier: float,
        volatility_col: str = "realized_vol"
    ) -> BacktestResult:

        equity = [self.initial_capital]
        trades = []
        position = None

        # v4.2: 대기 중인 신호
        pending_signal = None
        pending_bar_idx = None
        pending_time = None

        for i, (idx, row) in enumerate(data.iterrows()):
            current_time = row.get('end_time', 0)

            # ===== 1. 대기 중인 신호 실행 (v4.2 NEW) =====
            if pending_signal is not None:
                bars_waited = i - pending_bar_idx
                time_waited = current_time - pending_time

                # 지연 조건 충족 확인
                if bars_waited >= self.entry_delay and time_waited >= self.min_physical_delay:
                    entry_price = self._get_entry_price(row)
                    sigma = row[volatility_col]

                    # Fee Trap 체크 (진입 시점의 σ로!)
                    volume_ratio = self._get_volume_ratio(data, i)
                    if self.cost_model.is_viable_trade(pt_multiplier, sigma, volume_ratio):
                        position = self._open_position(
                            pending_signal, entry_price, sigma,
                            sl_multiplier, pt_multiplier, equity[-1]
                        )

                    pending_signal = None
                    pending_bar_idx = None
                    pending_time = None

            # ===== 2. 포지션 관리 =====
            if position is not None:
                result = self._check_exit(position, row)
                if result:
                    trades.append(result)
                    position = None

            # ===== 3. 새 신호 대기열에 추가 (v4.2: 즉시 진입 X) =====
            if i < len(signals):
                signal = signals[i]
            else:
                signal = 0

            if position is None and pending_signal is None and signal != 0:
                pending_signal = signal
                pending_bar_idx = i
                pending_time = current_time
                # 주의: 여기서 진입하지 않음! 다음 루프에서 delay 후 진입

            # 자본금 업데이트
            if trades and trades[-1].get("close_bar") == i:
                equity.append(equity[-1] + trades[-1]["pnl"])
            else:
                equity.append(equity[-1])

        return BacktestResult(
            equity=equity,
            trades=trades,
            metrics=self._calculate_metrics(trades, equity)
        )

    def _get_entry_price(self, bar: pd.Series) -> float:
        """v4.2: 진입 가격 결정"""
        if self.entry_price_type == "open":
            return bar['open']
        elif self.entry_price_type == "twap":
            return (bar['open'] + bar['high'] + bar['low'] + bar['close']) / 4
        elif self.entry_price_type == "vwap":
            return bar['dollar_value'] / (bar['volume'] + 1e-10)
        return bar['open']

    def _get_volume_ratio(self, data: pd.DataFrame, idx: int, window: int = 100) -> float:
        """현재 거래량 / 평균 거래량"""
        start = max(0, idx - window)
        avg_vol = data.iloc[start:idx]['volume'].mean()
        return data.iloc[idx]['volume'] / (avg_vol + 1e-10)

    def _open_position(
        self, signal, entry_price, sigma,
        sl_mult, pt_mult, current_equity
    ) -> dict:
        """포지션 생성"""
        sigma_price = sigma * entry_price

        if signal == 1:  # Long
            return {
                "direction": 1,
                "entry": entry_price,
                "tp": entry_price + pt_mult * sigma_price,
                "sl": entry_price - sl_mult * sigma_price,
                "size": self._calc_position_size(current_equity, sl_mult * sigma_price)
            }
        else:  # Short
            return {
                "direction": -1,
                "entry": entry_price,
                "tp": entry_price - pt_mult * sigma_price,
                "sl": entry_price + sl_mult * sigma_price,
                "size": self._calc_position_size(current_equity, sl_mult * sigma_price)
            }

        return BacktestResult(
            equity=equity,
            trades=trades,
            metrics=self._calculate_metrics(trades, equity)
        )

    def _calculate_metrics(self, trades: list, equity: list) -> dict:
        """핵심 성과 지표 계산"""
        if not trades:
            return {}

        pnls = [t["pnl_pct"] for t in trades]

        return {
            "total_trades": len(trades),
            "win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
            "avg_profit_per_trade": np.mean(pnls),  # 핵심 지표!
            "sharpe_ratio": np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252 * 50),
            "max_drawdown": self._calc_max_drawdown(equity),
            "profit_factor": sum(p for p in pnls if p > 0) / (-sum(p for p in pnls if p < 0) + 1e-10)
        }
```

### 7.5 Parameter Smoothing (v4.2 NEW)

> ⚠️ **v4.2 추가**: Walk-Forward에서 파라미터 불안정성 방지
> - Fold마다 최적 파라미터가 급변하면 실전에서 불안정
> - EMA 기반 스무딩 + 변동 페널티로 안정화

```python
class SmoothParameterOptimizer:
    """
    v4.2: 파라미터 스무딩

    문제:
    - Fold 1: SL=1.0, PT=2.5, Time=200
    - Fold 2: SL=3.0, PT=1.5, Time=1000
    → 급격한 변화 = 실전에서 리스크

    해결:
    - EMA 기반 스무딩: new_param = α × grid_best + (1-α) × prev_param
    - 변동 페널티: score -= penalty × |param_change|
    """

    def __init__(
        self,
        ema_alpha: float = 0.3,       # 반영 속도 (0=고정, 1=즉시반영)
        instability_penalty: float = 0.1
    ):
        self.ema_alpha = ema_alpha
        self.instability_penalty = instability_penalty
        self.prev_params = None

    def smooth_params(
        self,
        grid_best: dict,
        grid_scores: pd.DataFrame
    ) -> dict:
        """
        Grid Search 결과에 스무딩 적용

        Args:
            grid_best: Grid Search에서 찾은 최적 파라미터
            grid_scores: 전체 Grid 점수 DataFrame

        Returns:
            smoothed_params: 스무딩된 파라미터
        """
        if self.prev_params is None:
            # 첫 Fold는 그대로 사용
            self.prev_params = grid_best.copy()
            return grid_best

        # 1. 변동 페널티 적용
        adjusted_scores = self._apply_instability_penalty(
            grid_scores, self.prev_params
        )

        # 2. 페널티 적용 후 최적 파라미터 재선정
        penalized_best = adjusted_scores.sort_values(
            "adjusted_score", ascending=False
        ).iloc[0]

        # 3. EMA 스무딩
        smoothed = {}
        for key in ["sl", "pt", "time"]:
            grid_val = penalized_best[key]
            prev_val = self.prev_params[key]

            if key == "time":
                # 정수 파라미터는 반올림
                smoothed[key] = int(round(
                    self.ema_alpha * grid_val + (1 - self.ema_alpha) * prev_val
                ))
            else:
                smoothed[key] = (
                    self.ema_alpha * grid_val + (1 - self.ema_alpha) * prev_val
                )

        self.prev_params = smoothed
        return smoothed

    def _apply_instability_penalty(
        self,
        scores: pd.DataFrame,
        prev_params: dict
    ) -> pd.DataFrame:
        """변동 페널티 계산"""
        df = scores.copy()

        # 정규화된 변화량 계산
        sl_range = df["sl"].max() - df["sl"].min()
        pt_range = df["pt"].max() - df["pt"].min()
        time_range = df["time"].max() - df["time"].min()

        df["sl_change"] = abs(df["sl"] - prev_params["sl"]) / (sl_range + 1e-10)
        df["pt_change"] = abs(df["pt"] - prev_params["pt"]) / (pt_range + 1e-10)
        df["time_change"] = abs(df["time"] - prev_params["time"]) / (time_range + 1e-10)

        df["total_change"] = (df["sl_change"] + df["pt_change"] + df["time_change"]) / 3
        df["adjusted_score"] = df["score"] - self.instability_penalty * df["total_change"]

        return df
```

**스무딩 효과 예시:**
```
Without Smoothing (v4.1):
  Fold 1: SL=1.0, PT=3.5, Time=100
  Fold 2: SL=3.0, PT=1.5, Time=2000  ← 급변!
  Fold 3: SL=1.0, PT=2.0, Time=500

With Smoothing (v4.2, α=0.3):
  Fold 1: SL=1.0, PT=3.5, Time=100   (기준)
  Fold 2: SL=1.6, PT=2.9, Time=670   (스무딩)
  Fold 3: SL=1.4, PT=2.6, Time=619   (스무딩)

→ 파라미터 변동폭 감소 → 실전 안정성 향상
```

### 7.6 성공 기준

```python
SUCCESS_CRITERIA = {
    "avg_profit_per_trade": 0.0015,  # > 0.15% (Net of Fees)
    "win_rate": 0.45,                # > 45%
    "sharpe_ratio": 1.5,             # > 1.5
    "max_drawdown": -0.20            # < 20%
}
```

### 7.7 출력

```
results/
├── walk_forward/
│   ├── fold_{id}/
│   │   ├── best_params.json
│   │   ├── model.pt
│   │   ├── signals.parquet
│   │   ├── trades.parquet
│   │   └── equity.parquet
│   └── summary.json               # 전체 결과 요약
└── backtest/
    └── final_report.html          # 시각화 리포트
```

---

## 8. Directory Structure

```
/home/kimhoyeon/dev/dl_rl_btc/
├── config/
│   └── config.yaml                    # 중앙 설정
├── data/
│   ├── raw/                           # 원본 ZIP (다운로드 후 삭제)
│   └── labels/                        # Labeled Dataset
│       ├── optimization_results/      # Grid Search 결과
│       └── labeled/                   # 최종 라벨
├── etl/                               # Go ETL + Features ✅
│   ├── cmd/main.go
│   ├── internal/
│   │   ├── bars/                      # Dollar Bar 생성
│   │   ├── binance/                   # 데이터 파싱/다운로드
│   │   ├── config/                    # ETL 설정
│   │   ├── features/                  # Feature Engineering ✅
│   │   │   ├── generator.go
│   │   │   ├── models.go
│   │   │   ├── l1.go
│   │   │   ├── regime.go
│   │   │   ├── stationarity.go
│   │   │   ├── momentum.go
│   │   │   ├── connors_rsi.go
│   │   │   └── normalizer/
│   │   └── storage/
│   └── data/
│       ├── bars/                      # Dollar Bar 출력
│       └── features/                  # Feature 출력
├── research/                          # Python ML 📝
│   ├── label_optimizer/               # Stage 2: Label Optimization (v4 NEW)
│   │   ├── __init__.py
│   │   ├── grid_search.py             # Grid Search 로직
│   │   ├── triple_barrier.py          # TBM 라벨링
│   │   ├── scoring.py                 # Entropy, MI, Stationarity
│   │   └── optimizer.py               # 메인 오케스트레이터
│   ├── hybrid_trainer/                # Stage 3: Hybrid Training (v4 NEW)
│   │   ├── __init__.py
│   │   ├── ssl_pretraining.py         # Step A: SSL
│   │   ├── finetuning.py              # Step B: Fine-tuning
│   │   ├── focal_loss.py              # Focal Loss 구현
│   │   └── trainer.py                 # 메인 학습 루프
│   ├── models/                        # 모델 정의
│   │   ├── __init__.py
│   │   ├── patchtst.py                # PatchTST Encoder
│   │   ├── classifier.py              # Classification Head
│   │   └── ssl_model.py               # SSL Wrapper
│   ├── wfa_engine/                    # Stage 4: Walk-Forward (v4 NEW)
│   │   ├── __init__.py
│   │   ├── engine.py                  # 메인 WFA 루프
│   │   ├── backtester.py              # 백테스트 로직
│   │   ├── signal_generator.py        # 신호 생성
│   │   └── metrics.py                 # 성과 지표
│   └── utils/                         # Utilities
│       ├── data_loader.py
│       └── visualization.py
├── tests/                             # pytest 테스트
│   ├── test_label_optimizer.py
│   ├── test_hybrid_trainer.py
│   └── test_wfa_engine.py
├── artifacts/                         # 모델 체크포인트
│   ├── ssl/                           # SSL Pre-trained
│   └── classifier/                    # Fine-tuned
├── results/                           # 실험 결과
│   ├── walk_forward/
│   └── backtest/
└── docs/                              # 문서
    ├── blueprint_v3.md                # 이전 버전 (참조용)
    └── blueprint_v4.md                # 현재 버전
```

---

## 9. Configuration Schema (config.yaml)

```yaml
experiment_name: "hybrid_ssl_wfa_v4.3"

# ============================================================
# Multi-Asset Configuration (v4.3 NEW)
# ============================================================
assets:
  # Primary Assets (BTC, ETH - 높은 유동성, 안정적인 데이터)
  primary:
    - symbol: "BTCUSDT"
      market: "futures"
      enabled: true
    - symbol: "ETHUSDT"
      market: "futures"
      enabled: true

  # Secondary Assets (중간 유동성)
  secondary:
    - symbol: "XRPUSDT"
      market: "futures"
      enabled: true
    - symbol: "SOLUSDT"
      market: "futures"
      enabled: true
    - symbol: "BNBUSDT"
      market: "futures"
      enabled: true

  # Tertiary Assets (높은 변동성, 낮은 유동성)
  tertiary:
    - symbol: "DOGEUSDT"
      market: "futures"
      enabled: true
    - symbol: "LTCUSDT"
      market: "futures"
      enabled: true

# ============================================================
# Data Paths (Multi-Asset)
# ============================================================
data:
  bar_path: "./etl/data/bars"
  feature_path: "./etl/data/features"
  state_path: "./etl/data/state"
  label_output: "./data/labels"
  results_output: "./results"

# ============================================================
# ETL (Go) - 기존 유지
# ============================================================
etl:
  bars_per_day: 50
  warmup_days: 14

# ============================================================
# Stage 2: Label Optimization (v4.1 UPDATED)
# ============================================================
label_optimizer:
  # Grid Search Space
  grid:
    sl_range: [1.0, 2.0, 3.0]                         # σ multiplier
    pt_range: [1.5, 2.0, 2.5, 3.0, 3.5]              # σ multiplier
    time_range: [50, 100, 200, 500, 1000, 1500, 2000] # bars

  # Nonlinear Cost Model (v4.2 UPDATED - Square Root Law)
  cost_model:
    base_fee: 0.001            # 0.10% (거래소 수수료)
    base_slippage: 0.0001      # 0.01% (최소 슬리피지)
    impact_exponent: 0.5       # Square Root Law
    volatility_multiplier: 0.3 # σ 계수
    volume_penalty: 0.1        # 저거래량 페널티
    slippage_cap: 0.005        # 최대 0.50%
    min_profit_buffer: 1.5     # 비용 대비 최소 수익 배수

  # Drop Conditions (v4.1 NEW)
  drop_conditions:
    max_physical_duration: 259200  # 72시간 (초)
    min_valid_samples: 1000        # 최소 유효 샘플 수

  # Scoring Weights (v4.2 UPDATED - Profitability 추가)
  scoring:
    entropy_weight: 0.2      # v4.2: 0.3 → 0.2 (보조 지표로 격하)
    mi_weight: 0.4           # v4.2: 0.5 → 0.4
    profitability_weight: 0.4  # v4.2 NEW: Stationarity 대체

  # Profitability Proxy (v4.2 NEW)
  profitability:
    forward_bars: 50         # 수익 계산 구간
    min_samples: 10          # 최소 샘플 수

  # Rejection Criteria
  min_entropy: 1.0           # 최소 엔트로피 (이하 탈락)

  # Overlap Handling (v4.1 NEW)
  overlap_handling:
    method: "purged_kfold"     # "purged_kfold" or "non_overlapping"
    embargo_pct: 0.01          # 전체의 1%
    n_splits: 5                # K-Fold 분할 수

  # Feature for volatility
  volatility_feature: "realized_vol"

# ============================================================
# Stage 3: Hybrid Training (v4 NEW)
# ============================================================
training:
  # Common
  context_window: 512
  batch_size: 64
  mixed_precision: true

  # SSL Pre-training (Step A)
  ssl:
    mask_ratio: 0.15
    epochs: 30
    learning_rate: 1e-4
    early_stopping_patience: 5

  # Fine-tuning (Step B)
  finetuning:
    epochs: 50
    frozen_epochs: 5           # Encoder frozen phase
    encoder_lr: 1e-5           # Unfrozen phase encoder LR
    head_lr: 1e-4              # Classifier head LR
    early_stopping_patience: 10
    early_stopping_metric: "val_f1"

  # Loss
  focal_loss:
    gamma: 2.0
    label_smoothing: 0.1

# ============================================================
# Model Architecture (v4.2 UPDATED - Auto Selection)
# ============================================================
model:
  # Auto Selection Thresholds (v4.2 NEW)
  auto_select:
    enabled: true
    min_samples_transformer: 50000  # PatchTST 사용 기준
    min_samples_tcn: 10000          # TCN 사용 기준

  # PatchTST (Primary - Transformer)
  patchtst:
    architecture: "PatchTST"
    n_channels: 28               # 피처 수
    patch_len: 16
    stride: 16
    d_model: 128
    n_heads: 4
    n_layers: 2
    dropout: 0.1

  # TCN-Attention (Alternative - 데이터 부족 시) (v4.2 NEW)
  tcn_attention:
    architecture: "TCN-Attention"
    n_features: 28
    tcn_channels: [64, 64, 64]
    kernel_size: 7
    n_heads: 4
    dropout: 0.2

  # Classification Head (공통)
  head:
    type: "residual_bottleneck"
    hidden_dim: 256
    dropout: 0.3
    n_classes: 3

# ============================================================
# Stage 4: Walk-Forward Analysis (v4 NEW)
# ============================================================
walk_forward:
  train_months: 12             # 학습 윈도우
  test_months: 1               # 테스트 윈도우
  step_months: 1               # 슬라이딩 간격
  min_train_bars: 10000        # 최소 학습 데이터

  # Backtest (v4.2 UPDATED - Execution Lag)
  backtest:
    initial_capital: 100000
    fee_rate: 0.001            # 0.10% (거래소 수수료)
    use_dynamic_slippage: true # label_optimizer.cost_model 사용
    risk_per_trade: 0.02       # 2% of capital

    # Execution Lag (v4.2 NEW)
    execution_lag:
      entry_delay: 1           # 신호 후 N바 대기
      entry_price_type: "open" # "open", "twap", "vwap"
      min_physical_delay: 60   # 최소 물리적 지연 (초)

  # Parameter Smoothing (v4.2 NEW)
  parameter_smoothing:
    enabled: true
    ema_alpha: 0.3             # EMA 계수 (0=변화없음, 1=즉시반영)
    instability_penalty: 0.1   # 파라미터 변동 페널티

# ============================================================
# Success Criteria
# ============================================================
success_criteria:
  avg_profit_per_trade: 0.0015  # > 0.15%
  win_rate: 0.45                # > 45%
  sharpe_ratio: 1.5             # > 1.5
  max_drawdown: -0.20           # < 20%
```

---

## 10. Implementation Status

| Stage | 모듈 | 상태 | 완료도 | 비고 |
|-------|------|------|--------|------|
| 1 | ETL - Bars (Go) | ✅ Done | 100% | Dollar Bar 구현 완료 |
| 1 | ETL - Features (Go) | ✅ Done | 100% | 28개 피처, State 직렬화 완료 |
| 2 | Label Optimizer | 📝 TODO | 0% | Grid Search, TBM, Scoring |
| 3 | SSL Pre-training | 📝 TODO | 0% | Masked Time Series Modeling |
| 3 | Fine-tuning | 📝 Skeleton | 30% | PatchTST, Focal Loss 기존 존재 |
| 4 | WFA Engine | 📝 TODO | 0% | Walk-Forward Loop |
| 4 | Backtester | 📝 Skeleton | 30% | 기본 로직 존재 |

---

## 11. Implementation Roadmap

### Phase 1: Label Optimizer (우선순위 1)
1. `triple_barrier.py` - TBM 라벨링 구현
2. `scoring.py` - Entropy, MI, Stationarity 계산
3. `grid_search.py` - 병렬 Grid Search
4. `optimizer.py` - 메인 오케스트레이터
5. 테스트: `test_label_optimizer.py`

### Phase 2: Hybrid Trainer (우선순위 2)
1. `ssl_pretraining.py` - Masked TSM 구현
2. `ssl_model.py` - SSL Wrapper 모델
3. `finetuning.py` - 2-Phase Fine-tuning
4. `trainer.py` - 통합 학습 파이프라인
5. 테스트: `test_hybrid_trainer.py`

### Phase 3: WFA Engine (우선순위 3)
1. `engine.py` - Walk-Forward 메인 루프
2. `signal_generator.py` - 신호 생성
3. `backtester.py` - 백테스트 고도화
4. `metrics.py` - 성과 지표 계산
5. 테스트: `test_wfa_engine.py`

### Phase 4: Integration & Evaluation
1. 전체 파이프라인 통합 테스트
2. 하이퍼파라미터 튜닝
3. 결과 분석 및 리포트 생성

---

## 12. Verification Checklist

### Stage 1: ETL (v4.3 Multi-Asset)
- [x] Dollar Bar 생성 검증
- [x] Feature 생성 검증 (28개 피처)
- [x] State 저장/복원 테스트
- [x] 월 경계 처리 테스트
- [ ] **ETHUSDT 데이터 수집/처리 검증** (v4.3)
- [ ] **XRPUSDT 데이터 수집/처리 검증** (v4.3)
- [ ] **SOLUSDT 데이터 수집/처리 검증** (v4.3)
- [ ] **DOGEUSDT 데이터 수집/처리 검증** (v4.3)
- [ ] **BNBUSDT 데이터 수집/처리 검증** (v4.3)
- [ ] **LTCUSDT 데이터 수집/처리 검증** (v4.3)
- [ ] **심볼별 독립 State 관리 검증** (v4.3)

### Stage 2: Label Optimizer (v4.2 업데이트)
- [ ] Grid Search 전체 조합 탐색 확인
- [ ] TBM 라벨링 정확성 (HighTime/LowTime 활용)
- [ ] Entropy 계산 검증
- [ ] Mutual Information 계산 검증
- [ ] **Profitability Proxy 계산 검증** (v4.2, Stationarity 대체)
- [ ] 종합 점수 정규화 및 가중 합산 검증
- [ ] 최적 파라미터 선정 로직 검증
- [ ] **Per-Bar Fee Trap Guard 동작 확인** (v4.1)
- [ ] **Nonlinear Cost Model (Square Root Law) 검증** (v4.2)
- [ ] **Clean Labels 검증 (Fee Trap은 DROP, Label 0은 Timeout만)** (v4.1)
- [ ] **Feature Continuity: 원본 Parquet 불변 확인** (v4.1)
- [ ] **Valid Mask 정확성 검증** (v4.1)
- [ ] **Purged K-Fold + Embargo 동작 확인** (v4.1)

### Stage 3: Hybrid Training (v4.2 업데이트)
- [ ] SSL Masking 로직 검증 (패치 단위)
- [ ] SSL Reconstruction Loss 검증
- [ ] Pre-trained Weight 로딩 검증
- [ ] Encoder Frozen Phase 동작 확인
- [ ] Focal Loss 계산 검증
- [ ] Label Smoothing 적용 확인
- [ ] Gradient Flow 검증
- [ ] **TCN-Attention 모델 학습 검증** (v4.2)
- [ ] **Model Auto Selection 로직 검증** (v4.2)

### Stage 4: WFA Engine (v4.2 업데이트)
- [ ] Walk-Forward Fold 생성 검증
- [ ] Train/Test 데이터 분리 (미래 누수 없음)
- [ ] Best Params가 Test에 적용되는지 확인
- [ ] Signal 생성 정확성
- [ ] 백테스트 PnL 계산 검증
- [ ] 수수료 반영 확인
- [ ] **Nonlinear Cost Model 적용 확인** (v4.2)
- [ ] **Execution Lag 적용 확인 (entry_delay)** (v4.2)
- [ ] **Entry Price Type 검증 (Open[t+1])** (v4.2)
- [ ] **Parameter Smoothing 동작 확인** (v4.2)
- [ ] 성과 지표 계산 검증

### Integration (v4.3 Multi-Asset)
- [ ] 전체 파이프라인 End-to-End 테스트
- [ ] 메모리 사용량 모니터링
- [ ] 재현성 검증 (동일 seed → 동일 결과)
- [ ] **Multi-Asset 병렬 처리 검증** (v4.3)
- [ ] **심볼별 독립 모델 학습 검증** (v4.3)
- [ ] **통합 리포트 생성 검증** (v4.3)

---

## 13. Key Differences from v3

| 항목 | v3 | v4.3 |
|------|----|----|
| **대상 자산** | BTCUSDT only | **7개 심볼 (BTC, ETH, XRP, SOL, DOGE, BNB, LTC)** |
| **라벨링** | 고정 파라미터 (PT=2σ, SL=1σ) | Grid Search + **Per-Bar Fee Trap Guard** |
| **Label 0** | 혼합 (Timeout+Fee Trap) | **Clean: Timeout만 (Fee Trap은 DROP)** |
| **비용 모델** | 고정 슬리피지 | **Nonlinear Cost (Square Root Law)** |
| **검증 지표** | - | **Profitability Proxy (Stationarity 대체)** |
| **중복 처리** | 없음 (Serial Correlation) | **Purged K-Fold + Embargo** |
| **학습** | 단일 지도학습 | SSL Pre-training → Fine-tuning |
| **모델 선택** | 단일 모델 | **Auto Selection (PatchTST/TCN-Attention)** |
| **백테스트** | 즉시 진입 (Close) | **Execution Lag (Open[t+1])** |
| **파라미터** | Fold별 독립 | **Parameter Smoothing (EMA)** |
| **적응성** | 전역 모델 | Walk-Forward 구간별 재학습 |
| **리스크 관리** | 고정 SL/TP | 구간별 최적 SL/TP 적용 |
| **파이프라인** | 선형 (ETL→Label→Train→Backtest) | 반복 루프 (WFA Loop) |
| **검증** | 단순 Train/Test Split | Rolling Walk-Forward |
| **Feature 처리** | 불명확 | **Feature Continuity (원본 불변)** |

---

## 14. References

- PatchTST: "A Time Series is Worth 64 Words" (Nie et al., 2023)
- Triple Barrier Method: "Advances in Financial Machine Learning" (de Prado, 2018)
- Focal Loss: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
- Walk-Forward Analysis: "Evidence-Based Technical Analysis" (Aronson, 2006)
