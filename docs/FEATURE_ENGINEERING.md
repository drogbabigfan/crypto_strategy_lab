# Feature Engineering Pipeline

> BTC/USDT 트레이딩을 위한 피처 엔지니어링 파이프라인 문서

## 개요

본 파이프라인은 ETL에서 생성된 Dynamic Dollar Bar 데이터를 머신러닝 모델 입력용 피처로 변환합니다.

### 아키텍처 (Go Streaming)

```
Dollar Bars → Feature Generator → FeatureRow (Parquet)
                    │
    ┌───────────────┼───────────────┐
    │               │               │
   L1          Regime         Stationarity
(stateless)   (GK, Skew)     (FracDiff)
                    │
              ┌─────┴─────┐
              │           │
          Momentum    ConnorsRSI
         (VW, ZScore)
```

### 설계 원칙

1. **No Batch Processing**: 단일 for-loop으로 전체 데이터 처리
2. **Stream Only**: 모든 정규화는 `Update(value)` 재귀적/누적 방식
3. **Python Read-Only**: Python에서는 `model.fit()`만 수행
4. **State Persistence**: 파일 경계에서 상태 유지 (Walk-Forward 지원)
5. **Numerical Stability**: Kahan Summation, Welford Algorithm

**총 피처 수:** 27개

---

## 입력 데이터 (Dollar Bar)

ETL에서 생성된 Dollar Bar 컬럼:

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `open`, `high`, `low`, `close` | float64 | OHLC 가격 |
| `volume` | float64 | BTC 거래량 |
| `dollar_value` | float64 | USD 거래대금 |
| `tick_count` | int64 | 체결 건수 |
| `duration` | float64 | 바 지속시간 (초) |
| `buy_dollar_vol` | float64 | 매수 체결 금액 |
| `sell_dollar_vol` | float64 | 매도 체결 금액 |
| `net_imbalance` | float64 | 순 불균형 (buy - sell) |

---

## 1. L1 Features (9개)

**파일:** `etl/internal/features/l1.go`

단일 바에서 계산되는 순수 함수형 피처 (Stateless).

### 1.1 Log Transforms

| 피처 | 수식 | 설명 |
|------|------|------|
| `log_volume` | $\log(1 + V)$ | 거래량 로그 변환 |
| `log_tick_count` | $\log(1 + N_{tick})$ | 체결 건수 로그 변환 |
| `log_duration` | $\log(1 + \Delta t)$ | 바 지속시간 로그 변환 |
| `log_trade_intensity` | $\log(1 + \frac{DollarValue}{Duration})$ | 거래 강도 로그 변환 |

### 1.2 Price Features

| 피처 | 수식 | 설명 |
|------|------|------|
| `vwap_deviation` | $\frac{Close - VWAP}{Close}$ | VWAP 대비 종가 편차 |

### 1.3 Imbalance Features

| 피처 | 수식 | 설명 |
|------|------|------|
| `volume_imbalance` | $\frac{NetImbalance}{DollarValue}$ | 매수/매도 불균형 (-1 ~ 1) |

### 1.4 Bar Shape Features

| 피처 | 수식 | 설명 |
|------|------|------|
| `bar_range` | $\frac{High - Low}{Close}$ | 정규화된 바 범위 (변동성) |
| `bar_body` | $\frac{Close - Open}{Close}$ | 정규화된 바 몸통 (방향성) |

### 1.5 Microstructure Features

| 피처 | 수식 | 설명 |
|------|------|------|
| `kyle_lambda` | $\log(1 + \frac{\|Return\|}{DollarValue})$ | Amihud 비유동성 (가격 충격) |

**Kyle's Lambda 해석:**
- 높은 값: 낮은 유동성, 높은 가격 충격
- 낮은 값: 높은 유동성, 낮은 가격 충격

---

## 2. Regime Features (8개)

**파일:** `etl/internal/features/regime.go`

시장 레짐(변동성/추세 상태) 감지. Stateful - 롤링/EWM 계산.

### 2.1 Volatility Features

| 피처 | 수식 | 설명 |
|------|------|------|
| `garman_klass_vol` | $\sqrt{0.5 \ln^2\frac{H}{L} - (2\ln2-1)\ln^2\frac{C}{O}}$ | Garman-Klass 변동성 |
| `realized_vol` | $\sigma_R = std(returns, window)$ | 실현 변동성 (rolling std) |
| `vol_ratio` | $\frac{\sigma_{GK}}{\sigma_{Realized}}$ | 변동성 비율 (점프/갭 감지) |
| `vol_zscore` | $\frac{\sigma_t - \mu_{\sigma}}{\sigma_{\sigma}}$ | 변동성 Z-score (EWM) |

**Garman-Klass vs Parkinson:**
- GK는 OHLC 전체 사용 (Parkinson은 H/L만 사용)
- 더 효율적인 추정량 (lower variance)
- Open-Close 정보 활용으로 점프 감지 향상

### 2.2 Entropy Features

| 피처 | 수식 | 설명 |
|------|------|------|
| `shannon_entropy` | $H = -\frac{\sum_{i=1}^{k} p_i \ln p_i}{\ln k}$ | 정규화된 Shannon 엔트로피 |
| `entropy_zscore` | $z_H = \frac{H_t - \mu_H}{\sigma_H}$ | 엔트로피 Z-score (EWM) |

**해석:**
- $H \approx 0$: 예측 가능 (추세 시장)
- $H \approx 1$: 무작위 (노이즈/횡보)

### 2.3 Higher Moments

| 피처 | 수식 | 설명 |
|------|------|------|
| `skewness` | $\gamma_1 = E\left[\left(\frac{X-\mu}{\sigma}\right)^3\right]$ | 수익률 비대칭성 (3차 모멘트) |
| `kurtosis` | $\gamma_2 = E\left[\left(\frac{X-\mu}{\sigma}\right)^4\right] - 3$ | 꼬리 두께 (4차 모멘트, excess) |

**Welford Algorithm:**
- 스트리밍 방식으로 고차 모멘트 계산
- 수치적 안정성 보장 (catastrophic cancellation 방지)

**해석:**
- Skewness > 0: 오른쪽 꼬리 (극단적 상승)
- Skewness < 0: 왼쪽 꼬리 (극단적 하락)
- Kurtosis > 0: Fat tails (극단값 빈번)
- Kurtosis < 0: Thin tails (정규분포보다 안정)

---

## 3. Stationarity Features (3개)

**파일:** `etl/internal/features/stationarity.go`

시계열 정상성 변환. 비정상 시계열은 ML 학습에 부적합.

### 3.1 Fractional Differentiation

| 피처 | 수식 | 설명 |
|------|------|------|
| `frac_diff_close` | $(1-L)^d \cdot \log(P_t) = \sum_{k=0}^{\infty} w_k \cdot \log(P_{t-k})$ | 분수 차분된 로그 가격 |

**가중치 계산:**
$$w_k = -w_{k-1} \cdot \frac{d - k + 1}{k}, \quad w_0 = 1$$

**파라미터:**
- $d = 0.4$ (기본값): 정상성과 메모리 균형
- $d = 0$: 원본 (비정상)
- $d = 1$: 완전 차분 (메모리 손실)

**출처:** Lopez de Prado, "Advances in Financial Machine Learning" (2018)

### 3.2 Detrended Log Price

| 피처 | 수식 | 설명 |
|------|------|------|
| `detrended_log_price` | $\log(P_t) - EWM(\log(P), halflife)$ | 추세 제거된 로그 가격 |

### 3.3 Returns

| 피처 | 수식 | 설명 |
|------|------|------|
| `returns` | $r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$ | 단순 수익률 |

---

## 4. Momentum Features (5개)

**파일:** `etl/internal/features/momentum.go`

다중 시간 지평 모멘텀 신호.

### 4.1 Volume-Weighted Momentum

| 피처 | 수식 | 설명 |
|------|------|------|
| `vw_momentum` | $Return \times Volume$ | 거래량 가중 모멘텀 |

**해석:**
- 수익률과 거래량을 결합
- 높은 거래량 + 양수 수익률 = 강한 상승 신호
- 거래량을 conviction proxy로 사용

### 4.2 Momentum Z-Score (Multi-Horizon)

| 피처 | 수식 | 설명 |
|------|------|------|
| `momentum_zscore_10` | EWM z-score (halflife=10) | 단기 모멘텀 신호 |
| `momentum_zscore_50` | EWM z-score (halflife=50) | 중기 모멘텀 신호 |
| `momentum_zscore_250` | EWM z-score (halflife=250) | 장기 모멘텀 신호 |
| `momentum_zscore_1000` | EWM z-score (halflife=1000) | 초장기 모멘텀 신호 |

**Multi-Horizon 설계:**
- 짧은 halflife: 반응 빠름, 노이즈 많음
- 긴 halflife: 안정적, 지연 있음
- 다중 시간 지평으로 다양한 시장 상황 포착

---

## 5. Technical Indicators (1개)

**파일:** `etl/internal/features/connors_rsi.go`

### 5.1 Connors RSI

| 피처 | 수식 | 설명 |
|------|------|------|
| `connors_rsi` | $\frac{RSI(3) + RSI_{Streak}(2) + PercentRank(100)}{3}$ | Connors RSI 복합 지표 |

**구성 요소:**
1. **RSI(3)**: 3기간 RSI (Wilder's smoothing)
2. **RSI_Streak(2)**: 연속 상승/하락 일수의 2기간 RSI
3. **PercentRank(100)**: 현재 수익률의 100일 백분위

**해석:**
- 0-100 범위
- < 20: 과매도 (매수 기회)
- > 80: 과매수 (매도 기회)

---

## 6. 전체 피처 요약

### 카테고리별 개수

| 카테고리 | 개수 | 피처 |
|---------|:----:|------|
| L1 Transforms | 4 | log_volume, log_tick_count, log_duration, log_trade_intensity |
| L1 Price | 1 | vwap_deviation |
| L1 Imbalance | 1 | volume_imbalance |
| L1 Bar Shape | 2 | bar_range, bar_body |
| L1 Microstructure | 1 | kyle_lambda |
| Regime Volatility | 4 | garman_klass_vol, realized_vol, vol_ratio, vol_zscore |
| Regime Entropy | 2 | shannon_entropy, entropy_zscore |
| Regime Moments | 2 | skewness, kurtosis |
| Stationarity | 3 | frac_diff_close, detrended_log_price, returns |
| Momentum | 5 | vw_momentum, momentum_zscore_10/50/250/1000 |
| Technical | 1 | connors_rsi |
| **Meta** | 1 | is_primed |
| **Total** | **27** | |

### 전체 피처 테이블

| # | 피처명 | 수식 | 범위 | Stateful |
|---|--------|------|:----:|:--------:|
| 1 | `log_volume` | $\log(1+V)$ | $[0, \infty)$ | ✗ |
| 2 | `log_tick_count` | $\log(1+N)$ | $[0, \infty)$ | ✗ |
| 3 | `log_duration` | $\log(1+\Delta t)$ | $[0, \infty)$ | ✗ |
| 4 | `log_trade_intensity` | $\log(1+TI)$ | $[0, \infty)$ | ✗ |
| 5 | `vwap_deviation` | $\frac{C-VWAP}{C}$ | $(-1, 1)$ | ✗ |
| 6 | `volume_imbalance` | $\frac{Net}{DV}$ | $[-1, 1]$ | ✗ |
| 7 | `bar_range` | $\frac{H-L}{C}$ | $[0, \infty)$ | ✗ |
| 8 | `bar_body` | $\frac{C-O}{C}$ | $(-1, 1)$ | ✗ |
| 9 | `kyle_lambda` | $\log(1+\frac{\|r\|}{DV})$ | $[0, \infty)$ | ✗ |
| 10 | `garman_klass_vol` | GK formula | $[0, \infty)$ | ✓ |
| 11 | `realized_vol` | $std(r, n)$ | $[0, \infty)$ | ✓ |
| 12 | `vol_ratio` | $\frac{\sigma_{GK}}{\sigma_R}$ | $(0, \infty)$ | ✓ |
| 13 | `vol_zscore` | EWM z-score | $(-\infty, \infty)$ | ✓ |
| 14 | `shannon_entropy` | $-\frac{\sum p\ln p}{\ln k}$ | $[0, 1]$ | ✓ |
| 15 | `entropy_zscore` | EWM z-score | $(-\infty, \infty)$ | ✓ |
| 16 | `skewness` | 3rd moment | $(-\infty, \infty)$ | ✓ |
| 17 | `kurtosis` | 4th moment (excess) | $(-\infty, \infty)$ | ✓ |
| 18 | `frac_diff_close` | $(1-L)^{0.4}\log P$ | $(-\infty, \infty)$ | ✓ |
| 19 | `detrended_log_price` | $\log P - EWM(\log P)$ | $(-\infty, \infty)$ | ✓ |
| 20 | `returns` | $\frac{P_t - P_{t-1}}{P_{t-1}}$ | $(-\infty, \infty)$ | ✓ |
| 21 | `vw_momentum` | $r \times V$ | $(-\infty, \infty)$ | ✗ |
| 22 | `momentum_zscore_10` | EWM z-score | $(-\infty, \infty)$ | ✓ |
| 23 | `momentum_zscore_50` | EWM z-score | $(-\infty, \infty)$ | ✓ |
| 24 | `momentum_zscore_250` | EWM z-score | $(-\infty, \infty)$ | ✓ |
| 25 | `momentum_zscore_1000` | EWM z-score | $(-\infty, \infty)$ | ✓ |
| 26 | `connors_rsi` | Composite RSI | $[0, 100]$ | ✓ |
| 27 | `is_primed` | All normalizers ready | $\{0, 1\}$ | ✓ |

---

## 7. 설정 (Go Config)

```go
type Config struct {
    // Regime
    ParkinsonWindow int     // GK volatility window (default: 24)
    EntropyWindow   int     // Shannon entropy window (default: 24)
    EntropyBins     int     // Entropy histogram bins (default: 10)
    VolHalflife     float64 // EWM halflife for vol z-score (default: 50)
    EntropyHalflife float64 // EWM halflife for entropy z-score (default: 50)

    // Stationarity
    FracDiffD       float64 // Fractional diff order (default: 0.4)
    FracDiffThresh  float64 // Weight threshold (default: 1e-5)
    DetrendHalflife float64 // EWM halflife for detrending (default: 100)

    // Momentum
    MomentumWindows  []int   // Windows: [10, 50, 250, 1000]
    MomentumHalflife float64 // EWM halflife (default: 50)
}
```

---

## 8. 사용법 (Go)

```go
import "dl-rl-btc-etl/internal/features"

// 기본 설정으로 Generator 생성
gen := features.NewGenerator(features.DefaultConfig())

// 바 처리
for _, bar := range bars {
    row := gen.Process(bar)

    // is_primed가 true일 때부터 유효한 피처
    if row.IsPrimed {
        // 모델 입력으로 사용
    }
}

// 상태 저장 (Walk-Forward용)
state := gen.State()
json.Marshal(state)

// 상태 복원
gen.LoadState(state)
```

---

## 9. 수치 안정성

### Kahan Summation

부동소수점 누적 오차 방지:

```go
type KahanSum struct {
    sum        float64
    correction float64
}

func (k *KahanSum) Add(value float64) {
    y := value - k.correction
    t := k.sum + y
    k.correction = (t - k.sum) - y
    k.sum = t
}
```

### Welford's Algorithm

스트리밍 통계 (mean, variance, skewness, kurtosis):

```go
func (w *Welford) Update(value float64) {
    n1 := w.count
    w.count++
    delta := value - w.mean
    deltaN := delta / float64(w.count)

    w.mean += deltaN
    // m2, m3, m4 업데이트 (Terriberry 방식)
}
```

### Periodic Recalculation

장기 실행 시 drift 방지:

```go
if updateCount % recalcInterval == 0 {
    recalculate()  // 버퍼 전체 재계산
}
```

---

## 10. State Serialization

모든 피처 생성기는 JSON 직렬화 지원:

```go
// GeneratorState 구조
type GeneratorState struct {
    Regime       RegimeState
    Stationarity StationarityState
    Momentum     MomentumState
    ConnorsRSI   ConnorsRSIState
    Count        int64
}
```

**Walk-Forward 백테스트 지원:**
- 각 fold 경계에서 상태 저장
- 새 fold 시작 시 상태 복원
- 정규화 파라미터 연속성 보장

---

## 11. 테스트

```bash
# 전체 테스트
cd etl && go test ./internal/features/... -v

# 특정 패키지
go test ./internal/features/normalizer/... -v
```

**테스트 커버리지:**
- L1, Regime, Stationarity, Momentum, ConnorsRSI 개별 테스트
- Generator 통합 테스트
- State 직렬화/복원 테스트
- 수치 안정성 테스트 (Kahan, Welford)
- Edge case 테스트 (NaN, Inf, zero values)

---

## 12. 참고 문헌

1. **Lopez de Prado, M. (2018)**. *Advances in Financial Machine Learning*. Wiley.
   - Fractional Differentiation (Chapter 5)
   - Dollar Bars (Chapter 2)

2. **Garman, M. B., & Klass, M. J. (1980)**. "On the Estimation of Security Price Volatilities from Historical Data". *Journal of Business*.
   - Garman-Klass Volatility Estimator

3. **Shannon, C. E. (1948)**. "A Mathematical Theory of Communication". *Bell System Technical Journal*.
   - Shannon Entropy

4. **Welford, B. P. (1962)**. "Note on a method for calculating corrected sums of squares and products". *Technometrics*.
   - Welford's Online Algorithm

5. **Terriberry, T. (2007)**. "Computing Higher-Order Moments Online".
   - Online Skewness/Kurtosis

6. **Amihud, Y. (2002)**. "Illiquidity and stock returns: cross-section and time-series effects". *Journal of Financial Markets*.
   - Kyle's Lambda / Amihud Illiquidity

7. **Connors, L. & Alvarez, C. (2009)**. *Short Term Trading Strategies That Work*.
   - Connors RSI
