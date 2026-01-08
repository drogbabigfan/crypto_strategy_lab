# Deep Learning Bitcoin Trading Strategy Blueprint v2

## 변경 이력
- v1: 초기 설계 (TIB 기반)
- v2: 실제 구현 반영 (Adaptive Dollar Bar 기반)

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

---

## 3. Pipeline Architecture

```
[Stage 1: ETL]     → [Stage 2: Features]  → [Stage 3: Labels]
     Go                   Python                 Python
 Adaptive Dollar Bar   L1 + FracDiff +       Triple Barrier
                       Stitching

                              ↓

[Stage 4: Training]  → [Stage 5: Backtest]
     Python                 Python
 PatchTST + Focal       Walk-Forward +
                        Money Management
```

---

## 4. Stage 1: Ingestion & ETL (Go) ✅ COMPLETED

### 데이터 소스
| 타입 | 심볼 | 기간 | URL |
|------|------|------|-----|
| Spot | BTCUSDT | 2017-08 ~ Present | data.binance.vision/data/spot/monthly/trades |
| Futures | BTCUSDT | 2019-09 ~ Present | data.binance.vision/data/futures/um/monthly/trades |

### 구현 방식: Adaptive Dynamic Dollar Bar

**v1 설계 (TIB)와의 차이점:**
| 항목 | v1 (TIB) | v2 (Dollar Bar) |
|------|----------|-----------------|
| 샘플링 기준 | Tick Imbalance | Dollar Volume |
| 임계값 | Dual EMA 기반 | 14일 EMA 기반 |
| 목표 바/일 | N/A | 50개 (~29분 간격) |
| 복잡도 | 높음 (확률 추정 필요) | 낮음 (직관적) |

**Dollar Bar 생성 로직:**
```
Threshold_t = EMA_14(DailyDollarVolume) / 50
```
- 하루 약 50개 바 목표 (~29분 간격)
- 14일 워밍업 기간 필요
- EMA 사용으로 최근 거래량 변화에 빠르게 적응

**출력 스키마 (Parquet):**
```go
type DynamicDollarBar struct {
    StartTime, EndTime   int64
    Open, High, Low, Close, Volume float64
    DollarValue          float64  // 바의 총 달러 거래량
    ThresholdUsed        float64  // 해당 바 생성 시 사용된 임계값
    Duration             float64  // 바 지속 시간 (초)
    TickCount            int64    // 바 내 틱 수
    BuyDollarVol         float64  // 매수 측 달러 거래량
    SellDollarVol        float64  // 매도 측 달러 거래량
    NetImbalance         float64  // BuyDollarVol - SellDollarVol
}
```

**크래시 복구:**
- Trade ID 기반 중복 방지
- 월별 체크포인트 저장
- 상태 검증 (NaN/Inf, OHLC 정합성)

### 파일 위치
- 코드: `etl/internal/bars/`, `etl/cmd/main.go`
- 출력: `etl/data/bars/{market}/{symbol}-bars-{year}-{month}.parquet`

---

## 5. Stage 2: Feature Engineering (Python) 🔄 IN PROGRESS

### 입력
- Dollar Bar Parquet 파일들

### 구조

#### 5.1 L1 Features (기본 피처)

| 피처 | 계산 | 목적 |
|------|------|------|
| log_volume | log(1 + Volume) | 정규화 |
| log_duration | log(1 + Duration) | 불규칙 시간 반영 |
| volume_imbalance | (BuyDollarVol - SellDollarVol) / DollarValue | 매수/매도 압력 |
| vwap_deviation | (Close - VWAP) / Close | 가격 이탈 |
| log_tick_count | log(1 + TickCount) | 활동량 |

**VWAP 계산:**
```python
VWAP = DollarValue / Volume
```

#### 5.2 Regime Features (시장 상태)

| 피처 | 계산 | 목적 |
|------|------|------|
| parkinson_vol | sqrt(sum(log(H/L)^2) / (4*ln(2)*N)) | 고효율 변동성 |
| shannon_entropy | -sum(p * log(p)) | 노이즈 vs 트렌드 감지 |

#### 5.3 Stationarity (정상성 변환)

**Dual Input Strategy:**
- **Input A (Dynamics)**: FracDiff 적용된 가격 (단기 패턴용)
- **Input B (Context)**: Detrended Log Price (장기 트렌드 컨텍스트)

```python
# Input A: Fractional Differentiation
frac_diff_close = FracDiff(close, d=0.4)  # ADF p < 0.05

# Input B: Detrended Log Price
detrended = log(close) - EMA_100(log(close))
```

#### 5.4 Stitching (파일 간 연속성)

- **버퍼 크기**: 2000 bars
- **Adaptive Halflife**: 변동성 급변 시 decay 가속
- **Z-score 정규화**: EWM 기반

### 출력
- `data/features/features_{config_hash}.parquet`

---

## 6. Stage 3: Labeling (Python)

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

### 출력
- `data/labels/labeled_{config_hash}.parquet`

---

## 7. Stage 4: Network Training (Python)

### Model: PatchTST Classifier

**입력:**
- Shape: `(Batch, 2, 512)` - Dual Input (Dynamics + Context)

**구조:**
```
Input (2, 512)
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

## 8. Stage 5: Backtest & Money Management (Python)

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

## 9. Directory Structure

```
/home/kimhoyeon/dev/dl_rl_btc/
├── config/
│   └── config.yaml              # 중앙 설정
├── data/
│   ├── raw/                     # 원본 ZIP (다운로드 후 삭제)
│   ├── features/                # Feature Parquet
│   └── labels/                  # Labeled Dataset
├── etl/                         # Go ETL ✅
│   ├── cmd/main.go
│   ├── internal/
│   │   ├── bars/                # Dollar Bar 생성
│   │   ├── binance/             # 데이터 파싱/다운로드
│   │   └── storage/             # Parquet 저장
│   └── data/bars/               # ETL 출력
├── research/                    # Python ML
│   ├── features/                # Feature Engineering
│   ├── labels/                  # Labeling
│   ├── models/                  # PatchTST, Loss
│   ├── backtest/                # Backtester
│   └── utils/                   # Utilities
├── tests/                       # pytest 테스트
├── artifacts/                   # 모델 체크포인트
└── docs/                        # 문서
```

---

## 10. Implementation Status

| Stage | 상태 | 완료도 | 비고 |
|-------|------|--------|------|
| ETL (Go) | ✅ Done | 100% | Dollar Bar 구현 완료, 테스트 완료 |
| Features (Python) | 🔄 In Progress | 40% | 스켈레톤 존재, Dollar Bar 연동 필요 |
| Labels (Python) | 📝 Skeleton | 30% | 기본 로직 존재 |
| Training (Python) | 📝 Skeleton | 50% | PatchTST, Focal Loss 구현됨 |
| Backtest (Python) | 📝 Skeleton | 30% | 기본 로직 존재 |

---

## 11. Next Steps: Feature Engineering

### 즉시 작업 필요 항목

1. **L1 피처 업데이트**
   - Dollar Bar 컬럼명에 맞게 수정
   - VWAP 계산 추가 (DollarValue / Volume)
   - 누락 피처 추가 (log_tick_count, log_dollar_value)

2. **파이프라인 메인 엔트리포인트**
   - Parquet 로드 → L1 → Regime → Stationarity → Stitching → 저장
   - Config 연동

3. **Regime 피처 구현**
   - Parkinson Volatility
   - Shannon Entropy

4. **Stitching 검증**
   - 월별 파일 간 연속성 테스트
   - Adaptive halflife 동작 확인

5. **ADF 검증 리포트**
   - FracDiff 후 정상성 검증 자동화

---

## 12. Configuration Schema (config.yaml)

```yaml
experiment_name: "dollarbar_patchtst_v1"

data:
  symbol: "BTCUSDT"
  bar_path: "./etl/data/bars"       # Dollar Bar 위치
  feature_output: "./data/features"
  label_output: "./data/labels"

features:
  l1_factors:
    - "log_volume"
    - "log_duration"
    - "volume_imbalance"
    - "vwap_deviation"
    - "log_tick_count"
  regime:
    parkinson_window: 24
    entropy_window: 24
  stationarity:
    frac_diff_d: 0.4
    adf_threshold: 0.05
    detrend_ema_span: 100
  stitching:
    buffer_size: 2000
    base_halflife: 50

labeling:
  volatility_window: 24
  pt_multiplier: 2.0
  sl_multiplier: 1.0
  vertical_bars: 20
  vertical_hours: 72

model:
  architecture: "PatchTST"
  context_window: 512
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

## 13. Verification Checklist

- [ ] ETL: Go-Python 바 일치 검증 (작은 데이터셋)
- [ ] Features: ADF 리포트 자동 생성
- [ ] Labels: Train/Test 시간 순서 검증
- [ ] Model: Gradient flow 검증
- [ ] Backtest: 수수료 반영 확인
