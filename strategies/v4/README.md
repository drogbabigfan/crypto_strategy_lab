# AKF V4 Strategy

Adaptive Kalman Filter 기반 트렌드 팔로잉 전략

## Overview

- **Python**: 시그널 생성 (진입/청산 로직)
- **Go**: 백테스터 실행 (mark-to-market equity curve)
- **Mode**: `custom_stop` (Python이 모든 청산 로직 제어)

---

## Strategy Logic

### 1. Entry Conditions

```
Long Entry:
  - vel_zscore > entry_z (2.0)      # Velocity Z-score 상위
  - velocity > 0                     # 방향 필터
  - unc_pct < unc_pct_max (0.5)     # Uncertainty 하위 50%
  - rolling_k_long, rolling_mae_long available

Short Entry:
  - vel_zscore < -entry_z (-2.0)    # Velocity Z-score 하위
  - velocity < 0                     # 방향 필터
  - unc_pct < unc_pct_max (0.5)     # Uncertainty 하위 50%
  - rolling_k_short, rolling_mae_short available
```

### 2. Exit Conditions (4가지)

| Exit Type | Condition | Description |
|-----------|-----------|-------------|
| **Hard Stop** | price < entry × exp(-MAE_P99) | Rolling MAE 기반 손절 |
| **Trail Stop** | price < highest × exp(-K × sqrt(S)) | Dynamic trailing stop |
| **Innovation Breaker** | v_ratio ≥ 1.0 & log_resid breaches | 급변동 시 청산 |
| **Signal Exit** | v_ratio < 1.0 & 반대 신호 & intensity < 4.0 | 시그널 반전 청산 |

### 3. Position Sizing (Gaussian)

```python
base_size = risk_target / stop_distance
gauss_scale = gauss_max_mult × exp(-unc_pct² / (2 × sigma²))
final_size = clamp(base_size × gauss_scale, size_min, size_max)
```

- 낮은 uncertainty → 큰 포지션
- 높은 uncertainty → 작은 포지션

---

## Key Parameters

```python
V4_PARAMS = {
    # Entry
    "entry_z": 2.0,
    "unc_pct_max": 0.5,
    "warmup": 210,

    # Trail Stop - Rolling K (P99)
    "trail_k_horizon": 6,
    "trail_k_window": 180,
    "trail_k_quantile": 0.99,

    # Hard Stop - Rolling MAE (P99)
    "mae_horizon": 6,
    "mae_window": 180,
    "mae_quantile": 0.99,

    # Innovation Breaker
    "v_ratio_threshold": 1.0,
    "innov_base_mult": 2.5,
    "innov_mult_min": 1.5,
    "innov_mult_max": 4.0,

    # Position Sizing
    "risk_target": 0.03,        # 3%
    "gauss_max_mult": 4.0,
    "gauss_sigma": 0.5,
    "size_min": 0.1,
    "size_max": 3.0,
}
```

---

## Features Used

| Feature | Source | Description |
|---------|--------|-------------|
| `kf_trend` | Kalman Filter | Smoothed price trend |
| `kf_velocity` | Kalman Filter | Trend velocity (1st derivative) |
| `kf_uncertainty` | Kalman Filter | State uncertainty |
| `kf_innovation_cov_risk` | Dual-Eye Kalman | Risk-adjusted innovation covariance |
| `vel_zscore` | Rolling normalization | Velocity z-score (42-bar window) |
| `unc_pct` | Rolling percentile | Uncertainty percentile (210-bar window) |
| `v_ratio` | σ_hybrid / baseline | Volatility ratio |
| `sigma_hybrid` | max(sqrt(unc), RV) | Hybrid volatility |
| `resid_std` | Rolling std | Residual standard deviation |
| `intensity` | Duration-based | Flow intensity |

---

## Rolling Metrics (No Lookahead)

### MAE (Maximum Adverse Excursion)
```
MAE_long[t] = ln(Open_t / min(Low_{t+1:t+H}))   # 하방 리스크
MAE_short[t] = ln(max(High_{t+1:t+H}) / Open_t) # 상방 리스크

Rolling_MAE[t] = P99(MAE[t-window-horizon : t-horizon])
```

### K (Trail Stop Coefficient)
```
K[t] = MAE[t] / avg(sqrt(S_risk_{t+1:t+H}))

Rolling_K[t] = P99(K[t-window-horizon : t-horizon])
```

---

## Backtest Results

### Individual Assets (2020-2025)

| Symbol | Sharpe | CAGR | MDD | CAGR/MDD | Trades | WinRate | AvgSize |
|--------|--------|------|-----|----------|--------|---------|---------|
| BTCUSDT | 13.84 | 69.4% | 28.6% | 2.43 | 100 | 51.0% | 1.24 |
| ETHUSDT | 10.51 | 29.2% | 39.0% | 0.75 | 112 | 46.4% | 1.01 |
| XRPUSDT | 10.55 | 32.0% | 38.5% | 0.83 | 114 | 43.9% | 0.82 |
| SOLUSDT | 27.05 | 38.8% | 43.0% | 0.90 | 105 | 52.4% | 0.73 |
| BNBUSDT | 6.29 | 2.3% | 43.8% | 0.05 | 108 | 36.1% | 1.05 |
| DOGEUSDT | -123.18 | -9.2% | 52.6% | -0.17 | 24 | 20.8% | 0.75 |
| LTCUSDT | 0.13 | -1.9% | 47.8% | -0.04 | 106 | 38.7% | 0.92 |

### 4-Asset Portfolio (BTC, ETH, XRP, SOL)

| Metric | Value |
|--------|-------|
| CAGR | 43.3% |
| MDD | 19.0% |
| CAGR/MDD | 2.28 |
| Sharpe | 1.46 |
| Leverage | 3.80x |

### Yearly Performance (Portfolio)

| Year | Return | MDD | Ret/MDD |
|------|--------|-----|---------|
| 2020 | 22.8% | 11.9% | 1.91 |
| 2021 | 197.8% | 8.4% | 23.43 |
| 2022 | -1.0% | 21.9% | -0.05 |
| 2023 | 12.4% | 8.6% | 1.44 |
| 2024 | 48.8% | 11.1% | 4.38 |
| 2025 | 25.6% | 18.0% | 1.42 |
| **TOTAL** | **681.4%** | **21.9%** | **31.18** |

### Yearly Performance (Individual Assets)

| Year | BTC | ETH | XRP | SOL |
|------|-----|-----|-----|-----|
| 2020 | 45.1% | -7.0% | 3.9% | 2.3% |
| 2021 | 245.7% | 24.4% | 98.2% | 7.5% |
| 2022 | 0.2% | -18.0% | 14.2% | -20.2% |
| 2023 | 18.3% | -20.4% | -18.1% | 122.2% |
| 2024 | 28.6% | 22.9% | 93.8% | 67.9% |
| 2025 | 16.8% | 103.8% | -13.0% | 71.4% |

---

## Known Limitations

### 1. 2022년 베어마켓 성과 부진 (-1%)

**원인**: `vel_zscore`가 rolling 정규화되어 베어마켓에서도 평균 ≈ 0
- 숏 진입 조건 `vel_zscore < -2.0` 충족률: 3.3%
- 대부분의 하락이 "평범한 하락"으로 분류됨

**잠재적 개선 방향**:
- `entry_z` 하향 조정 (2.0 → 1.5)
- 절대 velocity 기준 추가
- 시장 레짐 필터 도입

### 2. 전략 수익률 상관관계

| Pair | Correlation |
|------|-------------|
| BTC-ETH | 0.220 |
| BTC-XRP | 0.040 |
| BTC-SOL | 0.189 |
| ETH-XRP | 0.121 |
| ETH-SOL | 0.116 |
| XRP-SOL | 0.131 |
| **Average** | **0.136** |

낮은 상관관계 → 높은 분산 효과 (개별 MDD 28~43% → 포트폴리오 MDD 19%)

---

## File Structure

```
strategies/v4/
├── README.md                 # This file
├── strategy_v35_go.py        # Main strategy implementation
├── yearly_analysis.py        # Yearly breakdown analysis
├── portfolio_equity.py       # Portfolio analysis
└── result/
    ├── BTCUSDT_equity.csv
    ├── ETHUSDT_equity.csv
    ├── XRPUSDT_equity.csv
    ├── SOLUSDT_equity.csv
    └── portfolio_equity.csv
```

---

## Dependencies

- Python: numpy, pandas, numba
- Go: etl/bin/backtester
- Data: etl/data/features-6/futures/{symbol}/

---

## Usage

```bash
# Run backtest
python strategies/v4/strategy_v35_go.py

# Yearly analysis
python strategies/v4/yearly_analysis.py
```
