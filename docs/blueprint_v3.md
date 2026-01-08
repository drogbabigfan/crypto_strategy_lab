# Deep Learning Bitcoin Trading Strategy Blueprint v3

## 변경 이력
- v1: 초기 설계 (TIB 기반)
- v2: 실제 구현 반영 (Adaptive Dollar Bar 기반)
- v3: ETL + Feature Engineering 완료 반영 (Go 통합 파이프라인)

---

## 1. Goal Description

**목표**: Bitcoin 자동 매매 연구 시스템 개발
- **전략**: Mid-Frequency Trend Following (Long & Short)
- **핵심 제약**: Avg Profit per Trade > 0.15% (수수료/슬리피지 후)
- **설계 철학**: Modular, Config-Driven, File-Centric, TDD

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

### Streaming-First (v3 신규)
- 전체 데이터셋 단일 for-loop 처리
- 모든 정규화는 `Update(value)` 재귀적/누적 방식
- Python에서는 `model.fit()`만 수행, 정규화 코드 없음
- 수치 안정성: Kahan Summation, Welford's Algorithm

---

## 3. Pipeline Architecture

```
[Stage 1: ETL]           → [Stage 2: Labels]    → [Stage 3: Training]
      Go                        Python                 Python
 Trades → Dollar Bars      Triple Barrier         PatchTST + Focal
 → Features (28개)

                                    ↓

                           [Stage 4: Backtest]
                                 Python
                            Walk-Forward +
                            Money Management
```

**v2 → v3 변경점:**
- Feature Engineering이 Go로 통합 (기존 Python Stage 제거)
- 파이프라인 단순화: 4단계 → 실질 3단계

---

## 4. Stage 1: ETL & Feature Engineering (Go) ✅ COMPLETED

### 4.1 데이터 소스

| 타입 | 심볼 | 기간 | URL |
|------|------|------|-----|
| Spot | BTCUSDT | 2017-08 ~ Present | data.binance.vision/data/spot/monthly/trades |
| Futures | BTCUSDT | 2019-09 ~ Present | data.binance.vision/data/futures/um/monthly/trades |

### 4.2 Two-Stage Go Pipeline

| Stage | Input | Output | Command |
|-------|-------|--------|---------|
| 1 | Raw Trades (ZIP) | Dollar Bars (Parquet) | `./bin/etl bars` |
| 2 | Dollar Bars | Features (Parquet) | `./bin/etl features` |

### 4.3 Adaptive Dynamic Dollar Bar

**Dollar Bar 생성 로직:**
```
Threshold_t = EMA_14(DailyDollarVolume) / 50
```
- 하루 약 50개 바 목표 (~29분 간격)
- 14일 워밍업 기간 필요
- EMA 사용으로 최근 거래량 변화에 빠르게 적응

**Look-Ahead Bias 방지:**
- Threshold는 오직 과거 데이터로만 계산
- 당일 데이터는 다음날 threshold 계산에 사용

**Dollar Bar 출력 스키마:**
```go
type DynamicDollarBar struct {
    StartTime, EndTime   int64
    Open, High, Low, Close float64
    HighTime, LowTime    int64    // 고가/저가 발생 시간 (백테스트 정밀도용)
    Volume               float64
    DollarValue          float64  // 바의 총 달러 거래량
    ThresholdUsed        float64  // 해당 바 생성 시 사용된 임계값
    Duration             float64  // 바 지속 시간 (초)
    TickCount            int64    // 바 내 틱 수
    BuyDollarVol         float64  // 매수 측 달러 거래량
    SellDollarVol        float64  // 매도 측 달러 거래량
    NetImbalance         float64  // BuyDollarVol - SellDollarVol
}
```

**HighTime/LowTime 활용 (백테스트):**
```python
# 바 내에서 고점/저점 발생 순서 판단
if bar.high_time < bar.low_time:
    # 가격이 올랐다가 내려감 → 롱 TP 먼저 체결
else:
    # 가격이 내려갔다가 올라감 → 숏 TP 먼저 체결
```

### 4.4 Feature Engineering (Go Streaming)

총 **28개 피처** 생성, 4개 카테고리:

#### L1 Features (Stateless) - 9개

단일 바에서 계산되는 순수 함수형 피처:

| Feature | Formula | Description |
|---------|---------|-------------|
| `log_volume` | `log1p(volume)` | 로그 거래량 |
| `log_tick_count` | `log1p(tick_count)` | 로그 체결 수 |
| `log_duration` | `log1p(duration)` | 로그 바 형성 시간 |
| `log_trade_intensity` | `log1p(dollar_value / duration)` | 거래 강도 |
| `vwap_deviation` | `(close - VWAP) / close` | VWAP 이탈도 |
| `volume_imbalance` | `net_imbalance / dollar_value` | 매수/매도 불균형 [-1, 1] |
| `bar_range` | `(high - low) / close` | 바 변동폭 |
| `bar_body` | `(close - open) / close` | 바 몸통 크기 |
| `kyle_lambda` | `log1p(\|return\| / dollar_value)` | Amihud 비유동성 |

#### Regime Features (Stateful) - 8개

시장 상태 감지용 롤링/EWM 피처:

| Feature | Algorithm | Description |
|---------|-----------|-------------|
| `garman_klass_vol` | `sqrt(0.5*ln(H/L)² - (2ln2-1)*ln(C/O)²)` | GK 변동성 |
| `realized_vol` | Rolling std of returns | 실현 변동성 |
| `shannon_entropy` | Histogram-based entropy | 수익률 분포 엔트로피 |
| `vol_ratio` | `GK_vol / realized_vol` | 변동성 비율 |
| `vol_zscore` | EWM z-score of GK vol | 변동성 z-점수 |
| `entropy_zscore` | EWM z-score of entropy | 엔트로피 z-점수 |
| `skewness` | Welford 3rd moment | 수익률 비대칭성 |
| `kurtosis` | Welford 4th moment (excess) | 꼬리 두께 |

#### Stationarity Features - 3개

정상성 확보를 위한 변환:

| Feature | Algorithm | Description |
|---------|-----------|-------------|
| `frac_diff_close` | Fractional Differentiation (d=0.4) | 메모리 보존 차분 |
| `detrended_log_price` | `log(price) - EWM(log(price))` | 추세 제거 로그 가격 |
| `returns` | `(close - prev_close) / prev_close` | 단순 수익률 |

#### Momentum Features - 5개

다중 시간 지평 모멘텀:

| Feature | Algorithm | Description |
|---------|-----------|-------------|
| `vw_momentum` | `return × volume` | 거래량 가중 모멘텀 |
| `momentum_zscore_10` | EWM z-score (halflife=10) | 단기 모멘텀 신호 |
| `momentum_zscore_50` | EWM z-score (halflife=50) | 중기 모멘텀 신호 |
| `momentum_zscore_250` | EWM z-score (halflife=250) | 장기 모멘텀 신호 |
| `momentum_zscore_1000` | EWM z-score (halflife=1000) | 초장기 모멘텀 신호 |

#### Technical Indicators - 1개

| Feature | Algorithm | Description |
|---------|-----------|-------------|
| `connors_rsi` | `(RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3` | Connors RSI 복합 지표 |

#### Meta - 1개

| Feature | Type | Description |
|---------|------|-------------|
| `is_primed` | BOOL | 모든 normalizer 워밍업 완료 여부 |

### 4.5 수치 안정성

#### Kahan Summation
부동소수점 누적 오차 방지:
```go
type KahanSum struct {
    sum        float64
    correction float64
}
```

#### Welford's Algorithm
스트리밍 평균/분산/왜도/첨도 계산:
```go
// 단일 패스, O(1) 메모리
func (w *Welford) Update(value float64) {
    delta := value - w.mean
    w.mean += delta / n
    w.m2 += delta * (value - w.mean)
    // m3, m4 for skewness/kurtosis
}
```

#### Periodic Recalculation
장기 실행 시 drift 방지:
```go
if updateCount % recalcInterval == 0 {
    recalculate()  // 버퍼 전체 재계산
}
```

### 4.6 State Management

**Monthly Checkpoint System:**
```
1. State 로드
   ├─ State 없음 → Fresh start
   └─ State 있음 → Checkpoint 검사

2. Partial Month 감지 (current_processing_month != "")
   ├─ 해당 월 parquet 파일 삭제
   ├─ 이전 월 checkpoint로 상태 복원
   └─ 해당 월부터 재처리

3. 각 월 처리 시
   ├─ Complete 상태 + 파일 존재 → Skip
   ├─ Complete 상태 + 파일 없음 → Reprocess
   └─ Partial/없음 → Process
```

**Walk-Forward 백테스트 지원:**
- 각 fold 경계에서 상태 저장
- 새 fold 시작 시 상태 복원
- 정규화 파라미터 연속성 보장

### 4.7 파일 위치

- 코드: `etl/internal/bars/`, `etl/internal/features/`, `etl/cmd/main.go`
- Bar 출력: `etl/data/bars/{market}/{symbol}-bars-{year}-{month}.parquet`
- Feature 출력: `etl/data/features/{market}/{symbol}-features-{year}-{month}.parquet`

---

## 5. Stage 2: Labeling (Python) 📝 SKELETON

### Triple Barrier Method

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
sigma = RealizedVol(window=24)  # 24바 변동성
PT = entry_price * (1 + n_pt * sigma)  # n_pt = 2.0
SL = entry_price * (1 - n_sl * sigma)  # n_sl = 1.0
```

**Vertical Barrier (Hybrid):**
```python
vertical_hit = (bars_since_entry > 20) OR (time_since_entry > 72h)
```

**레이블:**
- +1: PT 먼저 도달 (Long 성공)
- -1: SL 먼저 도달 (Short 성공)
- 0: Vertical barrier 도달 (타임아웃)

**HighTime/LowTime 활용:**
- 바 내에서 PT/SL 도달 순서 정확히 판단
- 백테스트 정밀도 향상

### 출력
- `data/labels/labeled_{config_hash}.parquet`

---

## 6. Stage 3: Network Training (Python) 📝 SKELETON

### Model: PatchTST Classifier

**입력:**
- Shape: `(Batch, Channels, 512)` - 28개 피처 채널

**구조:**
```
Input (28, 512)
    ↓
Patching (patch_len=16, stride=16)
    ↓
Linear Embedding (patch_len → d_model=128)
    ↓
Positional Embedding (learnable)
    ↓
Transformer Encoder (heads=4, layers=2)
    ↓
Flatten
    ↓
Residual Bottleneck Head:
    residual = x
    x = Dense(flatten_dim→256) → ReLU → Dense(256→flatten_dim)
    out = Softmax(Dense(residual + Dropout(x) → 3))
```

**Loss: Focal Loss**
```python
FL = (1 - p_t)^gamma * CE_loss  # gamma=2.0
```
- Label Smoothing: 0.1
- 클래스 불균형 처리 (Easy example downweight)

**학습:**
- Mixed Precision (FP16)
- Walk-Forward Validation (subprocess isolation for OOM 방지)

### 출력
- `artifacts/model_{fold}_{epoch}.pt`

---

## 7. Stage 4: Backtest & Money Management (Python) 📝 SKELETON

### 실행 로직

**포지션 사이징:**
- Fixed Risk: 자본의 n% (기본 2%)
- Half-Kelly (옵션)

**피라미딩:**
```python
if unrealized_profit > n * sigma:
    add_position(0.5 * unit)
    move_sl_to(avg_entry - 0.5 * sigma)  # Buffered BEP
```

**동적 Exit:**
- TP: 변동성 증가 시 확장 (Let profits run)
- SL: 절대 확장 금지, 오직 트레일링 또는 고정
- Trailing Stop: `price - n_trail * sigma`

### 성공 기준
```
Avg Profit per Trade > 0.150% (Net of Fees)
```

### 수수료 모델
- Fee Rate: 0.05% (maker) ~ 0.1% (taker)
- Slippage: 0.01%

---

## 8. Directory Structure

```
/home/kimhoyeon/dev/dl_rl_btc/
├── config/
│   └── config.yaml              # 중앙 설정
├── data/
│   ├── raw/                     # 원본 ZIP (다운로드 후 삭제)
│   └── labels/                  # Labeled Dataset
├── etl/                         # Go ETL + Features ✅
│   ├── cmd/main.go
│   ├── internal/
│   │   ├── bars/                # Dollar Bar 생성
│   │   ├── binance/             # 데이터 파싱/다운로드
│   │   ├── config/              # ETL 설정
│   │   ├── features/            # Feature Engineering ✅
│   │   │   ├── generator.go     # 피처 오케스트레이터
│   │   │   ├── models.go        # FeatureRow, Config
│   │   │   ├── l1.go            # L1 피처 (Stateless)
│   │   │   ├── regime.go        # Regime 피처
│   │   │   ├── stationarity.go  # Stationarity 피처
│   │   │   ├── momentum.go      # Momentum 피처
│   │   │   ├── connors_rsi.go   # Connors RSI
│   │   │   └── normalizer/      # 정규화 유틸리티
│   │   │       ├── kahan.go     # Kahan Summation
│   │   │       ├── welford.go   # Welford Algorithm
│   │   │       ├── ewm.go       # EWM Normalizer
│   │   │       └── rolling.go   # Rolling Normalizer
│   │   └── storage/             # Parquet 저장
│   ├── data/
│   │   ├── bars/                # Dollar Bar 출력
│   │   └── features/            # Feature 출력
│   └── docs/
│       └── ETL_PIPELINE.md      # ETL 상세 문서
├── research/                    # Python ML
│   ├── labels/                  # Labeling
│   ├── models/                  # PatchTST, Loss
│   ├── backtest/                # Backtester
│   └── utils/                   # Utilities
├── tests/                       # pytest 테스트
├── artifacts/                   # 모델 체크포인트
└── docs/                        # 문서
```

---

## 9. Implementation Status

| Stage | 상태 | 완료도 | 비고 |
|-------|------|--------|------|
| ETL - Bars (Go) | ✅ Done | 100% | Dollar Bar 구현 완료 |
| ETL - Features (Go) | ✅ Done | 100% | 28개 피처, State 직렬화 완료 |
| Labels (Python) | 📝 Skeleton | 30% | 기본 로직 존재 |
| Training (Python) | 📝 Skeleton | 50% | PatchTST, Focal Loss 구현됨 |
| Backtest (Python) | 📝 Skeleton | 30% | 기본 로직 존재 |

---

## 10. Next Steps: Labeling

### 즉시 작업 필요 항목

1. **Feature Parquet 로드**
   - Go에서 생성된 Feature Parquet 읽기
   - `is_primed=True`인 데이터만 사용

2. **Triple Barrier 구현**
   - `realized_vol` 피처 활용 (이미 계산됨)
   - HighTime/LowTime 활용한 정밀 체결 판단

3. **Class Balance 분석**
   - 레이블 분포 확인
   - Focal Loss gamma 조정

4. **Walk-Forward Split**
   - 시간 순서 기반 분할
   - 각 fold에서 Feature Generator 상태 복원

---

## 11. Configuration Schema (config.yaml)

```yaml
experiment_name: "dollarbar_patchtst_v2"

data:
  symbol: "BTCUSDT"
  bar_path: "./etl/data/bars"
  feature_path: "./etl/data/features"
  label_output: "./data/labels"

# Go ETL에서 사용 (이미 구현됨)
etl:
  bars_per_day: 50
  warmup_days: 14

features:
  # L1 (Stateless)
  l1_enabled: true

  # Regime
  regime:
    gk_window: 24
    entropy_window: 24
    entropy_bins: 10

  # Stationarity
  stationarity:
    frac_diff_d: 0.4
    detrend_halflife: 100

  # Momentum
  momentum:
    halflives: [10, 50, 250, 1000]

  # Technical
  connors_rsi:
    rsi_period: 3
    streak_period: 2
    rank_period: 100

labeling:
  volatility_feature: "realized_vol"  # 이미 계산된 피처 사용
  pt_multiplier: 2.0
  sl_multiplier: 1.0
  vertical_bars: 20
  vertical_hours: 72

model:
  architecture: "PatchTST"
  context_window: 512
  n_channels: 28  # 피처 수
  patch_len: 16
  d_model: 128
  n_heads: 4
  n_layers: 2
  head_type: "residual_bottleneck"

training:
  loss: "focal_loss"
  focal_gamma: 2.0
  label_smoothing: 0.1
  mixed_precision: true
  batch_size: 64
  epochs: 50

execution:
  fee_rate: 0.001
  slippage: 0.0001
  risk_per_trade: 0.02
  pyramiding_enabled: true
  trailing_stop_multiplier: 2.5
```

---

## 12. Verification Checklist

- [x] ETL: Dollar Bar 생성 검증
- [x] ETL: Feature 생성 검증 (28개 피처)
- [x] ETL: State 저장/복원 테스트
- [x] ETL: 월 경계 처리 테스트
- [ ] Labels: is_primed 필터링 확인
- [ ] Labels: HighTime/LowTime 활용 검증
- [ ] Labels: Train/Test 시간 순서 검증
- [ ] Model: 28 채널 입력 처리 확인
- [ ] Model: Gradient flow 검증
- [ ] Backtest: 수수료 반영 확인

---

## 13. Performance Reference

### ETL 처리량 예시 (2020-01 Futures)

```
Input:  122.87 MB ZIP (10,281,248 trades)
Output: 814.40 KB Parquet (6,491 bars)
Time:   ~7 seconds
Ratio:  ~150:1 compression
```

### 메모리 최적화

- Streaming 처리 (전체 데이터 메모리 로드 없음)
- Circular buffer (14개 float64만 유지)
- Sliding Window: 최대 2개월치 bars만 메모리 유지
- 단일 BarAccumulator 재사용
