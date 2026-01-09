# AKF V3.5 - Dual-Eye Kalman + Rolling MAE Stops

## Overview

V3.5는 스탑 시스템을 전면 개편한 버전입니다. 기존의 그리드 서치로 찾은 `7×σ_hybrid` 대신, 이론적 근거가 있는 Kalman Innovation Covariance와 Rolling MAE를 사용합니다.

## Version History

| Version | Key Changes | CAGR/MDD |
|---------|-------------|----------|
| V3.3 | Intensity Filter 추가 | 1.50 |
| V3.4 | Gaussian Sizing, low/high stops, T=7/H=7 | 1.66 |
| **V3.5** | **Dual-Eye Kalman Trail + MAE Hard Stop** | **2.38** |

## Stop System

### Trail Stop: Dual-Eye Kalman

```
Trail Stop = highest × exp(-k × √S_risk)
```

- **k = 5**: 5σ 이탈 시 청산
- **S_risk = P_pred + R_parkinson**: Innovation Covariance

#### Dual-Eye 개념

하나의 Kalman Filter가 두 개의 "눈"을 가짐:

1. **Trend Eye (R_close)**: Close-to-close variance
   - 부드러운 추세 추적용
   - State update에 사용

2. **Risk Eye (R_parkinson)**: Parkinson volatility
   - 봉 내 변동폭 (high-low range) 반영
   - Stop 계산에 사용
   - 휩소 방어에 효과적

```python
# Parkinson Volatility
σ² = (1/4ln2) × mean[(ln(H) - ln(L))²]
```

### Hard Stop: Rolling MAE

```
Hard Stop = entry_price × (1 - MAE_Q95)
```

- **MAE (Maximum Adverse Excursion)**: 진입 후 최대 역행폭
- **H=6**: 진입 후 6봉 (1일) 동안의 최저가 확인
- **W=180**: 과거 180봉 (30일) 참고
- **Q=95%**: 95th percentile 사용

```python
MAE_t = (Open_t - min(Low_{t+1}...Low_{t+H})) / Open_t
```

"지금 진입하면 향후 1일 동안 재수 없으면 어디까지 빠질까?"

## Entry Conditions

```python
# Long Entry
vel_zscore > 2.0 AND unc_pct < 0.5

# Short Entry
vel_zscore < -2.0 AND unc_pct < 0.5
```

### Entry Threshold 민감도

| 조건 | CAGR | MDD | C/M | Trades |
|------|------|-----|-----|--------|
| Z>2.5 | 40.5% | 20.3% | 1.99 | 166 |
| **Z>2.0** | **66.3%** | **27.8%** | **2.38** | **439** |
| Z>1.5 | 56.3% | 65.5% | 0.86 | 877 |
| Z>1.0 | 27.2% | 78.1% | 0.35 | 1444 |
| Z>0.0 | -15.3% | 90.4% | -0.17 | 3347 |

→ Z>2.0이 최적. 조건 완화 시 품질 급락.

## Exit Conditions (우선순위)

1. **Hard Stop**: `low < entry × (1 - MAE)` - 장중 터치 시 청산
2. **Trail Stop**: `low < highest × exp(-5 × √S_risk)` - Kalman 기반 동적
3. **Innovation Breaker**: `|residual| > innov_mult × resid_std` (v_ratio ≥ 1.0)
4. **Signal Exit**: vel_zscore 반전 (v_ratio < 1.0 AND Intensity < 4.0)

## Position Sizing

```python
# Base Size (MAE 기반)
base_size = risk_target / MAE_stop_distance

# Gaussian Scaling
M = 2.0 × exp(-unc_pct² / (2 × 0.5²))

# Final Size
final_size = clip(base_size × M, 0.1, 3.0)
```

## Parameters

```python
V35_PARAMS = {
    # Entry
    "entry_z": 2.0,
    "unc_pct_max": 0.5,

    # Trail Stop (Kalman)
    "trail_k": 5.0,

    # Hard Stop (MAE)
    "mae_horizon": 6,      # 1일
    "mae_window": 180,     # 30일
    "mae_quantile": 0.95,  # 95th percentile

    # Sizing
    "risk_target": 0.03,
    "gauss_max_mult": 2.0,
    "gauss_sigma": 0.5,
}
```

## Performance

### 4-Asset Portfolio (BTC, ETH, XRP, SOL)

| Metric | V3.4 | V3.5 | Change |
|--------|------|------|--------|
| CAGR | 60.2% | 66.3% | +10% |
| MDD | 36.2% | 27.8% | -23% |
| CAGR/MDD | 1.66 | 2.38 | **+43%** |
| Sharpe | 1.20 | 1.23 | +3% |
| Leverage | 2.41x | 2.61x | +8% |

### Individual Assets

| Symbol | CAGR | MDD | C/M | Sharpe | Trades | WinRate | Hold |
|--------|------|-----|-----|--------|--------|---------|------|
| BTCUSDT | 25.4% | 21.3% | 1.19 | 0.94 | 105 | 47.6% | 4.6d |
| ETHUSDT | 7.4% | 39.0% | 0.19 | 0.46 | 112 | 40.2% | 4.3d |
| XRPUSDT | 10.5% | 33.4% | 0.31 | 0.50 | 119 | 39.5% | 4.1d |
| SOLUSDT | 13.9% | 32.1% | 0.43 | 0.61 | 103 | 44.7% | 4.8d |

### Trading Statistics

- **거래 빈도**: 월 1.5회 (자산당)
- **평균 보유**: 4-5일
- **시장 참여율**: ~22%

## Why This Works

### Trail Stop: Kalman S_risk

1. **이론적 근거**: Innovation Covariance = 예측 오차의 분산
2. **동적 적응**: 매 봉 업데이트로 현재 변동성 반영
3. **Parkinson R**: 봉 내 wicks 반영 → 휩소 방어

### Hard Stop: Rolling MAE

1. **통계적 근거**: 실제 과거 데이터 기반 worst-case
2. **안정적**: 180봉 rolling → 노이즈에 덜 민감
3. **직관적**: "최악의 경우 얼마나 빠질 수 있나?"

### Hybrid Approach

- **Trail (Kalman)**: 빠른 적응, 현재 상황 반영
- **Hard (MAE)**: 통계적 안전망, 극단적 손실 방지

## Files

- `strategy_v35.py`: Main strategy implementation
- `test_dual_eye_backtest.py`: Backtest with different k values
- `test_kalman_stops.py`: Kalman stop distance analysis
- `research/features/kalman.py`: Dual-Eye Kalman implementation

## Usage

```python
from strategies.akf_v2.strategy_v35 import run_backtest, load_data

df = load_data("BTCUSDT")
metrics, df_result = run_backtest(df)

print(f"CAGR: {metrics['cagr']*100:.1f}%")
print(f"MDD: {metrics['max_drawdown']*100:.1f}%")
```
