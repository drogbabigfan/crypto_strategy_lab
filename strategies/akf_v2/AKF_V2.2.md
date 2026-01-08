# AKF V2.2 Strategy

## Overview
Adaptive Kalman Filter 기반 추세추종 전략 V2.2
- V2.1 대비 변경: **KF Residual 기반 Dynamic Stop-Loss (3σ)** 추가

## Changes from V2.1
| 항목 | V2.1 | V2.2 |
|------|------|------|
| Dynamic Stop | 없음 | KF 3σ Stop |
| Sharpe | 1.02 | 1.10 |
| PnL (2020-2025) | 1521% | 1657% |
| MDD | 19.6% | 19.6% |

---

## 1. Kalman Filter 설정

### State Model
```
상태 벡터: [log_trend, velocity]
관측 모델: log(close) = log_trend + noise
```

### Process Noise (PWNA)
```python
dt = 1  # bar 단위
q = 0.01  # base process noise

Q = [[dt³/3 * q,  dt²/2 * q],
     [dt²/2 * q,  dt * q    ]]
```

### Adaptive Q (NIS 기반)
```python
NIS = (innovation)² / innovation_variance
smooth_NIS = EMA(NIS, span=10)

if smooth_NIS > 2.0:
    Q_scale = 2.0  # 모델 불확실 → Q 증가
elif smooth_NIS < 0.5:
    Q_scale = 0.5  # 모델 확실 → Q 감소
else:
    Q_scale = 1.0
```

---

## 2. Signal Generation

### Parameters (고정)
| Parameter | Value | 설명 |
|-----------|-------|------|
| zscore_window | 42 bars | 7일 (6 bars/day) |
| uncertainty_window | 210 bars | 35일 |
| warmup | 210 bars | 초기 학습 기간 |
| uncertainty_pct_threshold | 0.5 | 하위 50% 불확실성만 진입 |

### Entry/Exit Thresholds
| Signal | Condition |
|--------|-----------|
| Long Entry | vel_zscore > +2.0 AND uncertainty_pct < 0.5 |
| Long Exit | vel_zscore < -2.0 OR **KF 3σ Stop** |
| Short Entry | vel_zscore < -2.0 AND uncertainty_pct < 0.5 |
| Short Exit | vel_zscore > +2.0 OR **KF 3σ Stop** |

### Velocity Z-score
```python
vel_mean = velocity.rolling(42).mean()
vel_std = velocity.rolling(42).std()
vel_zscore = (velocity - vel_mean) / vel_std
```

### Uncertainty Percentile
```python
uncertainty_pct = uncertainty.rolling(210).rank(pct=True)
low_uncertainty = uncertainty_pct < 0.5
```

---

## 3. Dynamic Stop-Loss (NEW in V2.2)

### Concept
KF Trend에서 3σ 이상 이탈 시 손절
- 추세 전환 조기 감지
- 일시적 노이즈에는 반응 안함 (KF trend가 smooth)

### Implementation
```python
# 잔차 계산 (log space)
log_residuals = log(price) - log(kf_trend)

# 잔차의 rolling std (30일 = 180 bars)
resid_std = log_residuals.rolling(180).std()

# Long Stop: price가 trend 대비 3σ 아래로 이탈
if log_residuals[i] < -3.0 * resid_std[i]:
    exit_long()

# Short Stop: price가 trend 대비 3σ 위로 이탈
if log_residuals[i] > +3.0 * resid_std[i]:
    exit_short()
```

### Statistics
- 평균 resid_std: 1.21%
- 3σ 손절선: ±3.62%
- Stop 발동 횟수: 16회 / 88 trades (18%)

### Why KF-based > Entry-based
| 방식 | Sharpe | PnL | 문제점 |
|------|--------|-----|--------|
| 평단 기준 3σ | 0.88 | 706% | 진입 직후 노이즈에 손절 |
| KF Trend 3σ | 1.10 | 1657% | 추세 따라 trailing |

---

## 4. Position Sizing

### V2.2 기본 설정
```python
method = "fixed"
position_size = 1.0  # 100% 고정
```

### 선택적: KF Probability Sizing
```python
method = "kf_prob"
min_size = 0.0
max_size = 2.0

# 불확실성 낮을수록 큰 사이즈
confidence = 1 - uncertainty_pct
size = min_size + (max_size - min_size) * confidence
```

---

## 5. Backtest Results

### Full Period (2020-2025)
| Metric | V2.1 | V2.2 | 변화 |
|--------|------|------|------|
| Sharpe Ratio | 1.02 | 1.10 | +7.8% |
| Total PnL | 1521% | 1657% | +136%p |
| Max Drawdown | 19.6% | 19.6% | 동일 |
| Total Trades | 87 | 88 | +1 |
| Win Rate | 56.3% | 55.7% | -0.6%p |
| Stop Hits | 0 | 16 | - |

### Configuration
```python
initial_capital = 100,000
fee_rate = 0.1% (10bp)
slippage_rate = 0.01% (1bp)
compounding = True
```

---

## 6. Full Pipeline

```
1. Bar 생성 (4시간봉, 6 bars/day)
     ↓
2. Adaptive Kalman Filter
   - Input: log(close)
   - Output: kf_trend, kf_velocity, kf_uncertainty
     ↓
3. Feature 계산
   - vel_zscore = zscore(velocity, window=42)
   - unc_pct = percentile_rank(uncertainty, window=210)
   - log_residuals = log(price) - log(kf_trend)
   - resid_std = rolling_std(log_residuals, window=180)
     ↓
4. Signal 생성
   - Entry: |vel_zscore| > 2.0 AND unc_pct < 0.5
   - Exit: vel_zscore 반전 OR 3σ Stop
     ↓
5. Position Sizing (fixed 1.0)
     ↓
6. Backtest
```

---

## 7. V2.2 Parameters Summary

```python
V22_PARAMS = {
    # Kalman Filter
    "process_noise_q": 0.01,
    "measurement_noise_r": 1.0,
    "nis_threshold_high": 2.0,
    "nis_threshold_low": 0.5,

    # Signal Generation
    "zscore_window": 42,           # 7일
    "uncertainty_window": 210,     # 35일
    "warmup": 210,
    "uncertainty_pct_threshold": 0.5,

    # Entry/Exit
    "long_entry": 2.0,
    "long_exit": -2.0,
    "short_entry": -2.0,
    "short_exit": 2.0,

    # Dynamic Stop (NEW)
    "stop_mult": 3.0,              # 3σ
    "resid_window": 180,           # 30일

    # Sizing
    "sizing_method": "fixed",
    "position_size": 1.0,
}
```
