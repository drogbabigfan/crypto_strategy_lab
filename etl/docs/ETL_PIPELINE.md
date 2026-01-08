# BTC Trading Data ETL Pipeline

Binance 거래 데이터를 **Adaptive Dynamic Dollar Bars**로 변환하는 ETL 파이프라인.
Mid-Frequency Trading (MFT) 딥러닝 모델 학습용 데이터 생성을 목적으로 설계됨.

---

## 목차

1. [개요](#개요)
2. [아키텍처](#아키텍처)
3. [Adaptive Dynamic Dollar Bar](#adaptive-dynamic-dollar-bar)
4. [Feature Generation](#feature-generation)
5. [데이터 구조](#데이터-구조)
6. [패키지 구조](#패키지-구조)
7. [사용법](#사용법)
8. [State Management](#state-management)
9. [Risk Mitigation](#risk-mitigation)

---

## 개요

### 왜 Dollar Bar인가?

기존 시간 기반 바(1분봉, 5분봉)의 문제점:
- 거래량이 적은 시간대에는 노이즈가 많음
- 거래량이 많은 시간대에는 정보 손실 발생
- 변동성과 무관하게 일정한 간격으로 샘플링

**Dollar Bar의 장점:**
- 일정 달러 거래량마다 바 생성 → 정보량 균등화
- 거래량이 많을 때 더 많은 바 생성 → 중요 구간 세밀한 포착
- 머신러닝 모델에 더 stationary한 입력 제공

### 왜 "Adaptive"인가?

고정된 임계값 대신 **14일 EMA 기반 동적 임계값** 사용:
- 시장 상황에 따라 자동 조정
- 2017년과 2024년의 거래량 차이를 자동 보상
- 목표: 하루 약 **50개 바** 생성 (약 29분 간격)
- EMA 사용으로 최근 거래량 변화에 빠르게 적응

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                      Binance Data Vision                         │
│    https://data.binance.vision/data/{spot|futures}/um/monthly   │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP Download
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Downloader (binance)                        │
│                   - Monthly ZIP download                         │
│                   - Checksum verification                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │ ZIP → CSV
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Parser (binance)                            │
│                   - CSV → Trade struct                           │
│                   - Validation (Fail-Fast)                       │
│                   - Whitespace trimming                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Trade stream
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Generator (bars)                            │
│                   - 14-day warmup period                         │
│                   - Adaptive threshold (EMA-based)               │
│                   - Dollar bar accumulation                      │
│                   - Crash recovery support                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │ DynamicDollarBar
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Feature Generator (features)                │
│   ┌─────────────┬──────────────┬──────────────┬───────────────┐ │
│   │     L1      │    Regime    │ Stationarity │   Momentum    │ │
│   │ (stateless) │  (GK, Skew)  │  (FracDiff)  │  (VW, ZScore) │ │
│   └─────────────┴──────────────┴──────────────┴───────────────┘ │
│                   - Streaming normalization                      │
│                   - State serialization (Walk-Forward)           │
│                   - Numerical stability (Kahan, Welford)         │
└─────────────────────────┬───────────────────────────────────────┘
                          │ FeatureRow
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Storage (storage)                           │
│                   - Parquet output                               │
│                   - Monthly partitioning                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Adaptive Dynamic Dollar Bar

### 알고리즘

#### 1. Warmup Period (14일)

```
Day 1-14: 바 생성 없음, 일별 거래량만 수집
          DailyVolume[i] = Σ(QuoteQty) for day i
```

#### 2. Threshold Calculation (매일 UTC 00:00)

```
Threshold_t = EMA_14(DailyVolume) / 50

예시:
  - 14일 EMA 일거래량: $1.4B
  - Threshold = $1.4B / 50 ≈ $28M
  - 결과: 약 $28M 거래될 때마다 바 1개 생성
```

**EMA vs SMA:**
- EMA(Exponential Moving Average)는 최근 데이터에 더 큰 가중치 부여
- Alpha = 2 / (14 + 1) ≈ 0.133
- 급격한 거래량 변화에 더 빠르게 반응

#### 3. Bar Generation

```python
for each trade:
    accumulator.add(trade)

    if accumulator.dollar_value >= threshold:
        emit_bar(accumulator)
        accumulator.reset()
```

### Look-Ahead Bias 방지

```
Timeline:
─────────────────────────────────────────────────────────────►
Day 1   Day 2   ...   Day 14  │  Day 15   Day 16   ...
◄──── Warmup (수집만) ────────►│◄─── Bar Generation ────────►
                               │
                    Threshold 계산 시점 (UTC 00:00)
                    사용 데이터: Day 1-14 (과거 데이터만)
```

- Threshold는 **오직 과거 데이터**로만 계산
- 당일 데이터는 다음날 threshold 계산에 사용
- 미래 정보 누출 Zero

---

## Feature Generation

Bar 데이터를 딥러닝 모델 입력용 피처로 변환하는 **순수 스트리밍 파이프라인**.

### 설계 원칙

1. **No Batch Processing**: 단일 for-loop으로 2020-2024 전체 처리
2. **Stream Only**: 모든 정규화는 `Update(value)` 재귀적/누적 방식
3. **Python Read-Only**: Python에서는 `model.fit()`만 수행, 정규화 코드 없음
4. **State Persistence**: 파일 경계에서 상태 유지 (Walk-Forward Analysis 지원)

### 피처 그룹

#### L1 Features (Stateless)

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

#### Regime Features (Stateful)

시장 상태 감지용 롤링/EWM 피처:

| Feature | Algorithm | Description |
|---------|-----------|-------------|
| `garman_klass_vol` | `sqrt(0.5*ln(H/L)² - (2ln2-1)*ln(C/O)²)` | GK 변동성 (Parkinson 대체) |
| `realized_vol` | Rolling std of returns | 실현 변동성 |
| `shannon_entropy` | Histogram-based entropy | 수익률 분포 엔트로피 |
| `vol_ratio` | `GK_vol / realized_vol` | 변동성 비율 |
| `vol_zscore` | EWM z-score of GK vol | 변동성 z-점수 |
| `entropy_zscore` | EWM z-score of entropy | 엔트로피 z-점수 |
| `skewness` | Welford 3rd moment | 수익률 비대칭성 |
| `kurtosis` | Welford 4th moment (excess) | 꼬리 두께 |

#### Stationarity Features

정상성 확보를 위한 변환:

| Feature | Algorithm | Description |
|---------|-----------|-------------|
| `frac_diff_close` | Fractional Differentiation (d=0.4) | 메모리 보존 차분 |
| `detrended_log_price` | `log(price) - EWM(log(price))` | 추세 제거 로그 가격 |
| `returns` | `(close - prev_close) / prev_close` | 단순 수익률 |

#### Momentum Features

다중 시간 지평 모멘텀:

| Feature | Algorithm | Description |
|---------|-----------|-------------|
| `vw_momentum` | `return × volume` | 거래량 가중 모멘텀 |
| `momentum_zscore_10` | EWM z-score (halflife=10) | 단기 모멘텀 신호 |
| `momentum_zscore_50` | EWM z-score (halflife=50) | 중기 모멘텀 신호 |
| `momentum_zscore_250` | EWM z-score (halflife=250) | 장기 모멘텀 신호 |
| `momentum_zscore_1000` | EWM z-score (halflife=1000) | 초장기 모멘텀 신호 |

#### Technical Indicators

기술적 지표:

| Feature | Algorithm | Description |
|---------|-----------|-------------|
| `connors_rsi` | `(RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3` | Connors RSI 복합 지표 |

### 수치 안정성

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

### State Serialization

모든 피처 생성기는 상태 직렬화 지원:

```go
// 상태 저장
state := generator.State()
json.Marshal(state)

// 상태 복원
generator.LoadState(state)
```

**Walk-Forward 백테스트 지원:**
- 각 fold 경계에서 상태 저장
- 새 fold 시작 시 상태 복원
- 정규화 파라미터 연속성 보장

---

## 데이터 구조

### DynamicDollarBar

```go
type DynamicDollarBar struct {
    // 1. 기본 OHLCV
    StartTime   int64   // Bar 시작 시간 (ms)
    EndTime     int64   // Bar 종료 시간 (ms)
    Open        float64
    High        float64
    HighTime    int64   // High 발생 시간 (ms) - 백테스트 정밀도용
    Low         float64
    LowTime     int64   // Low 발생 시간 (ms) - 백테스트 정밀도용
    Close       float64
    Volume      float64 // 총 거래량 (BTC)

    // 2. Bar 메타데이터
    DollarValue   float64 // 총 달러 거래량 (Quote Qty)
    TickCount     int64   // 거래 수
    ThresholdUsed float64 // 이 바 생성에 사용된 임계값
    Duration      float64 // 바 형성 시간 (초)

    // 3. Imbalance Features (Alpha Source)
    BuyDollarVol  float64 // Taker Buy 달러 거래량
    SellDollarVol float64 // Taker Sell 달러 거래량
    NetImbalance  float64 // Buy - Sell (순매수 압력)
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

### Trade (입력)

```go
type Trade struct {
    ID             int64   // Binance Trade ID
    Price          float64
    Quantity       float64
    QuoteQuantity  float64 // Price * Quantity
    Time           int64   // Timestamp (ms)
    IsBuyerMaker   bool    // true: 매수자가 Maker
    IsAggressorBuy bool    // true: Taker가 매수 (derived)
}
```

### Parquet Schema

| Column | Type | Description |
|--------|------|-------------|
| start_time | INT64 | Bar 시작 시간 (ms) |
| end_time | INT64 | Bar 종료 시간 (ms) |
| open | DOUBLE | 시가 |
| high | DOUBLE | 고가 |
| high_time | INT64 | 고가 발생 시간 (ms) |
| low | DOUBLE | 저가 |
| low_time | INT64 | 저가 발생 시간 (ms) |
| close | DOUBLE | 종가 |
| volume | DOUBLE | 거래량 (BTC) |
| dollar_value | DOUBLE | 달러 거래량 |
| tick_count | INT64 | 거래 수 |
| threshold_used | DOUBLE | 사용된 임계값 |
| duration | DOUBLE | 바 형성 시간 (초) |
| buy_dollar_vol | DOUBLE | Taker Buy 거래량 |
| sell_dollar_vol | DOUBLE | Taker Sell 거래량 |
| net_imbalance | DOUBLE | 순매수 압력 |

### FeatureRow Parquet Schema

피처 생성 후 출력되는 최종 데이터 스키마:

| Column | Type | Description |
|--------|------|-------------|
| start_time | INT64 | Bar 시작 시간 (ms) |
| end_time | INT64 | Bar 종료 시간 (ms) |
| open | DOUBLE | 시가 |
| high | DOUBLE | 고가 |
| low | DOUBLE | 저가 |
| close | DOUBLE | 종가 |
| **L1 Features** | | |
| log_volume | DOUBLE | 로그 거래량 |
| log_tick_count | DOUBLE | 로그 체결 수 |
| log_duration | DOUBLE | 로그 바 형성 시간 |
| log_trade_intensity | DOUBLE | 로그 거래 강도 |
| vwap_deviation | DOUBLE | VWAP 이탈도 |
| volume_imbalance | DOUBLE | 매수/매도 불균형 |
| bar_range | DOUBLE | 바 변동폭 |
| bar_body | DOUBLE | 바 몸통 |
| kyle_lambda | DOUBLE | Amihud 비유동성 |
| **Regime Features** | | |
| garman_klass_vol | DOUBLE | GK 변동성 |
| realized_vol | DOUBLE | 실현 변동성 |
| shannon_entropy | DOUBLE | 수익률 엔트로피 |
| vol_ratio | DOUBLE | 변동성 비율 |
| vol_zscore | DOUBLE | 변동성 z-점수 |
| entropy_zscore | DOUBLE | 엔트로피 z-점수 |
| skewness | DOUBLE | 왜도 (3차 모멘트) |
| kurtosis | DOUBLE | 첨도 (4차 모멘트) |
| **Stationarity Features** | | |
| frac_diff_close | DOUBLE | 분수 차분 종가 |
| detrended_log_price | DOUBLE | 추세 제거 로그 가격 |
| returns | DOUBLE | 수익률 |
| **Momentum Features** | | |
| vw_momentum | DOUBLE | 거래량 가중 모멘텀 |
| momentum_zscore_10 | DOUBLE | 단기 모멘텀 z-점수 |
| momentum_zscore_50 | DOUBLE | 중기 모멘텀 z-점수 |
| momentum_zscore_250 | DOUBLE | 장기 모멘텀 z-점수 |
| momentum_zscore_1000 | DOUBLE | 초장기 모멘텀 z-점수 |
| **Technical Indicators** | | |
| connors_rsi | DOUBLE | Connors RSI |
| **Meta** | | |
| is_primed | BOOL | 모든 normalizer 워밍업 완료 여부 |

---

## 패키지 구조

```
etl/
├── cmd/
│   └── main.go              # ETL 진입점
│
├── internal/
│   ├── bars/
│   │   ├── generator.go     # Adaptive Dollar Bar 생성기
│   │   ├── models.go        # DynamicDollarBar, BarAccumulator
│   │   └── state.go         # State 저장/로드/검증
│   │
│   ├── binance/
│   │   ├── downloader.go    # HTTP 다운로더
│   │   ├── parser.go        # Trade CSV 파서
│   │   └── url.go           # URL 생성, 시작 날짜
│   │
│   ├── config/
│   │   └── config.go        # ETL 설정
│   │
│   ├── features/
│   │   ├── generator.go     # 피처 오케스트레이터 (메인 진입점)
│   │   ├── models.go        # FeatureRow, Config 정의
│   │   ├── l1.go            # L1 피처 (Stateless)
│   │   ├── regime.go        # Regime 피처 (GK Vol, Entropy, Skew/Kurt)
│   │   ├── stationarity.go  # Stationarity 피처 (FracDiff, Detrend)
│   │   ├── momentum.go      # Momentum 피처 (VW Momentum, Z-scores)
│   │   ├── connors_rsi.go   # Connors RSI 지표
│   │   └── normalizer/
│   │       ├── kahan.go     # Kahan Summation (부동소수점 정밀도)
│   │       ├── welford.go   # Welford Algorithm (스트리밍 통계)
│   │       ├── ewm.go       # EWM Normalizer (지수가중 Z-score)
│   │       └── rolling.go   # Rolling Normalizer (고정 윈도우)
│   │
│   └── storage/
│       └── parquet.go       # Parquet 읽기/쓰기
│
├── data/
│   ├── raw/                 # 다운로드된 ZIP (처리 후 삭제)
│   └── bars/                # 출력 Parquet 파일
│
└── docs/
    └── ETL_PIPELINE.md      # 이 문서
```

---

## 사용법

### CLI 명령어

```bash
cd etl
go build -o bin/etl ./cmd/

# Bar 생성 (Stage 1: Trades → Dollar Bars)
./bin/etl bars

# Feature 생성 (Stage 2: Dollar Bars → Features)
./bin/etl features

# 도움말
./bin/etl help
```

### Two-Stage Pipeline

| Stage | Input | Output | Command |
|-------|-------|--------|---------|
| 1 | Raw Trades (ZIP) | Dollar Bars (Parquet) | `./bin/etl bars` |
| 2 | Dollar Bars | Features (Parquet) | `./bin/etl features` |

### 설정 변경

`cmd/main.go`에서 설정:

```go
markets := []config.ETLConfig{
    config.DefaultFuturesConfig(),  // Futures 처리
    // config.DefaultSpotConfig(),  // Spot 처리 (주석 해제)
}
```

### 출력 예시

```
=== Processing futures market ===
[futures] Restored generator state from checkpoint
[futures] Detected partial month: 2024-06 - will reprocess
[futures] Threshold: $28000000.00
[futures] 2024-01: Already complete (1550 bars), skipping
[futures] 2024-02: Already complete (1423 bars), skipping
[futures] Processing 2024-06...
[futures] 2024-06: Processed 15234567 trades -> 1550 bars (threshold: $28000000.00)
  Saved 1550 bars to data/bars/futures/BTCUSDT-bars-2024-06.parquet
ETL Pipeline Finished.
```

### 데이터 읽기 (Python)

```python
import pandas as pd

df = pd.read_parquet('data/bars/futures/BTCUSDT-bars-2024-01.parquet')
print(df.head())
```

---

## State Management

### 체크포인트 저장

매월 처리 후 자동 저장:

```
data/bars/futures/generator_state.json
```

### State 구조

```json
{
  "daily_volumes": [1.4e9, 1.5e9, ...],  // 14일 일별 거래량
  "daily_volume_head": 5,                 // Circular buffer head
  "daily_volume_count": 14,               // 유효 데이터 수
  "ema_volume": 1452000000,               // EMA 일별 거래량
  "current_day_start": 1704067200000,     // 현재 날짜 시작 (ms)
  "current_day_volume": 250000000,        // 당일 누적 거래량
  "current_threshold": 29040000,          // 현재 임계값
  "accumulator": {...},                   // 진행 중인 바 상태
  "is_warmup": false,                     // Warmup 여부
  "last_processed_trade_id": 987654321,   // 마지막 처리 Trade ID
  "monthly_checkpoints": {                // 월별 체크포인트
    "2024-01": {
      "status": "complete",
      "bar_count": 1550,
      "first_trade_id": 123456789,
      "last_trade_id": 234567890,
      "first_bar_time": 1704067200000,
      "last_bar_time": 1706745599000,
      "ema_volume_snapshot": 1452000000,
      "threshold_snapshot": 29040000,
      "day_start_snapshot": 1706745600000,
      "day_volume_snapshot": 0,
      "is_warmup_snapshot": false
    }
  },
  "current_processing_month": ""          // 현재 처리 중인 월 (partial 감지용)
}
```

### Monthly Checkpoint System

ETL 재시작 시 안전한 복구를 위한 월별 체크포인트 시스템:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ETL 재시작 시 동작 흐름                        │
└─────────────────────────────────────────────────────────────────┘

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

4. 월 처리 완료 시
   └─ MarkMonthComplete() → 상태 스냅샷 저장
```

**Checkpoint 상태:**
| Status | 의미 | 동작 |
|--------|------|------|
| `complete` | 완전히 처리됨 | 파일 존재 시 Skip |
| `partial` | 처리 중 중단됨 | 이전 상태로 롤백 후 재처리 |

### Crash Recovery

```
1. 프로세스 크래시 발생 (current_processing_month = "2024-06")
2. 재시작 시 state 로드
3. Partial month 감지 → "2024-06"
4. 2024-05 checkpoint에서 상태 복원
5. 2024-06.parquet 삭제
6. 2024-06 처음부터 재처리
7. 데이터 정합성 보장
```

**Trade ID 기반 중복 방지:**
- `last_processed_trade_id` 이후 거래만 처리
- 동일 거래 중복 카운팅 방지

---

## Risk Mitigation

### 1. Input Validation (Fail-Fast)

```go
// parser.go
func ValidateTrade(t *Trade) error {
    if t.Price <= 0 {
        return fmt.Errorf("price must be positive")
    }
    if math.IsNaN(t.Price) || math.IsInf(t.Price, 0) {
        return fmt.Errorf("price is NaN or Inf")
    }
    // ... 추가 검증
}
```

**거부 조건:**
- 가격/수량이 음수, NaN, Inf
- Timestamp가 0 이하
- Trade ID가 0 이하

### 2. Timestamp Out-of-Order 처리

```go
// models.go - BarAccumulator
type BarAccumulator struct {
    MinTimestamp int64  // 최소 타임스탬프
    MaxTimestamp int64  // 최대 타임스탬프
}

func (a *BarAccumulator) Finalize() DynamicDollarBar {
    // Duration = (Max - Min) / 1000
    // 순서가 뒤섞여도 정확한 duration 계산
}
```

### 3. State Corruption 감지

```go
// state.go
func (s *GeneratorState) Validate() error {
    if s.DailyVolumeCount < 0 || s.DailyVolumeCount > WarmupDays {
        return fmt.Errorf("corrupted: DailyVolumeCount out of range")
    }
    if s.Accumulator.High < s.Accumulator.Low {
        return fmt.Errorf("corrupted: High < Low")
    }
    // ... 추가 검증
}
```

**손상 감지 시:**
```go
// main.go
savedState, err := bars.LoadState(cfg.OutDir)
if err != nil {
    log.Printf("CORRUPTED STATE DETECTED: %v", err)
    bars.DeleteState(cfg.OutDir)  // 손상된 state 삭제
    // Fresh start
}
```

### 4. Duration 최소값 클리핑

```go
const MinDuration = 0.001  // 1ms

func (a *BarAccumulator) Finalize() DynamicDollarBar {
    duration := float64(a.MaxTimestamp - a.MinTimestamp) / 1000.0
    if duration < MinDuration {
        duration = MinDuration  // Division by zero 방지
    }
}
```

---

## 데이터 품질

### 지원 시장

| 시장 | 시작일 | URL Pattern |
|------|--------|-------------|
| Spot | 2017-08 | `data/spot/monthly/trades` |
| Futures | 2019-09 | `data/futures/um/monthly/trades` |

### 데이터 포맷 차이

| 필드 | Spot | Futures |
|------|------|---------|
| id | ✓ | ✓ |
| price | ✓ | ✓ |
| qty | ✓ | ✓ |
| quote_qty | ✓ | ✓ |
| time | ✓ | ✓ |
| is_buyer_maker | ✓ | ✓ |
| is_best_match | ✓ | ✗ (없음) |

파서는 두 포맷 모두 지원 (6필드 또는 7필드).

---

## 테스트

```bash
# 전체 테스트
go test ./...

# 통합 테스트 (실제 데이터 다운로드)
go test -v ./cmd/ -run Integration

# 단위 테스트만
go test -short ./...
```

### 테스트 커버리지

- Edge case 테스트 (음수, NaN, Inf, 빈 필드)
- State 저장/로드/검증 테스트
- Crash recovery 테스트
- URL 생성 테스트
- Parquet 읽기/쓰기 테스트
- **Sliding Window 테스트**:
  - 월 경계 bar 처리 (`TestSlidingWindowMonthBoundaryBar`)
  - 저장 타이밍 검증 (`TestSlidingWindowSavesAtCorrectTime`)
  - 빈 월 처리 (`TestSlidingWindowEmptyMonth`)
  - 잔여 bars 저장 (`TestSlidingWindowRemainingBars`)
  - HighTime/LowTime 추적 (`TestSlidingWindowHighLowTimestamps`)
  - 월 경계 bar 분류 (`TestSlidingWindowCrossMonthBarAttribution`)

---

## 성능 참고

### 처리량 예시 (2020-01 Futures)

```
Input:  122.87 MB ZIP (10,281,248 trades)
Output: 814.40 KB Parquet (6,491 bars)
Time:   ~7 seconds
Ratio:  ~150:1 compression
```

### 메모리 사용

- Streaming 처리 (전체 데이터 메모리 로드 없음)
- Circular buffer (14개 float64만 유지)
- 단일 BarAccumulator 재사용
- **Sliding Window 저장**: 최대 2개월치 bars만 메모리 유지

### Sliding Window 메모리 최적화

월 경계에 걸친 bar 처리를 위해 2개월 슬라이딩 윈도우 사용:

```
9월 처리 → barsByMonth["2020-09"]에 추가
10월 처리 → barsByMonth에 추가 (경계 bar는 StartTime 기준 분류)
11월 처리 → 9월 bars 확정 → 9월.parquet 저장 → 메모리 해제
12월 처리 → 10월 bars 확정 → 10월.parquet 저장 → 메모리 해제
...
마지막 → 남은 2개월치 저장
```

**왜 2개월인가?**
- 월 경계 bar: 9월 말 trades + 10월 초 trades → 하나의 bar
- 이 bar는 10월 처리 시 완성됨 (StartTime은 9월)
- 11월 처리 시점에 9월 bars는 더 이상 추가될 일 없음 → 저장 가능

---

## 향후 개선 가능 사항

1. **병렬 처리**: 월별 데이터 동시 다운로드/처리
2. **Incremental Update**: 신규 데이터만 처리
3. **다른 심볼 지원**: ETHUSDT, SOLUSDT 등
4. **실시간 스트리밍**: WebSocket 기반 실시간 바 생성
