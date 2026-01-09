# AKF V4 Strategy

Adaptive Kalman Filter 기반 트레이딩 전략 V4 (최종 버전)

## Overview

V3.5에서 발전한 버전으로, 다음 핵심 개선사항 포함:
- **미래참조 완전 제거**: Rolling 지표 계산 시 horizon만큼 추가 지연
- **롱/숏 분리 MAE**: 방향별 최대역행폭 별도 계산
- **로그 스케일 통일**: 모든 가격 기반 계산에 로그 스케일 적용
- **가격 변동성 기반 Dynamic K**: 트레이드 없이 순수 변동성으로 Trail K 추정

## Performance (4-Asset Portfolio: BTC, ETH, XRP, SOL)

| Metric | Value |
|--------|-------|
| CAGR | 110.0% |
| MDD | 20.7% |
| CAGR/MDD | 5.31 |
| Sharpe | 2.01 |
| Avg Leverage | 1.89x |

## Architecture

```
Python (Signal Generation)          Go (Execution)
┌─────────────────────────┐        ┌──────────────────┐
│ Kalman Filter           │        │                  │
│ ├─ kf_trend             │        │ custom_stop mode │
│ ├─ kf_velocity          │   →    │ ├─ signal: 1=롱  │
│ └─ kf_innovation_cov    │        │ ├─ signal: -1=숏 │
│                         │        │ └─ signal: 0=청산│
│ Rolling MAE (P99)       │        │                  │
│ Rolling K (P99)         │        │ Stop Price 검증  │
│                         │        │ Equity Tracking  │
│ Signal + Stop Price     │   →    │                  │
└─────────────────────────┘        └──────────────────┘
```

## Key Components

### 1. Entry Signal
- Velocity Z-score > entry_z (2.0)
- Uncertainty Percentile < unc_pct_max (0.5)
- Rolling K, MAE가 유효할 때만 진입

### 2. Trail Stop (Dynamic K)
```python
# 로그 스케일 K 계산
K_t = ln(Open_t / MinLow_{t+1:t+H}) / avg(sqrt(S_risk_{t+1:t+H}))
rolling_k = P99 of K over window

# Trail Stop Price
trail_stop_long = highest * exp(-k * sqrt(S_risk))
trail_stop_short = lowest * exp(k * sqrt(S_risk))
```

### 3. Hard Stop (Rolling MAE)
```python
# 로그 스케일 MAE 계산
MAE_long = ln(Open_t / MinLow_{t+1:t+H})
MAE_short = ln(MaxHigh_{t+1:t+H} / Open_t)
rolling_mae = P99 over window

# Hard Stop Price
hard_stop_long = entry * exp(-mae)
hard_stop_short = entry * exp(mae)
```

### 4. 미래참조 방지
```python
# t 시점에서 사용 가능한 데이터: t-horizon-1까지
# mae[t-horizon]은 prices[t-horizon+1:t+1] 사용 → t 포함!
# mae[t-horizon-1]은 prices[t-horizon:t] 사용 → OK

rolling_value[t] = percentile(raw[t-window-horizon-1 : t-horizon-1])
first_valid_idx = window + horizon + 1  # = 187
```

### 5. Position Sizing
```python
# MAE를 raw 퍼센트로 변환하여 risk_target과 일관성 유지
mae_pct = 1 - exp(-mae_log)  # for long
mae_pct = exp(mae_log) - 1   # for short
base_size = risk_target / mae_pct
```

## Parameters (V4 Final)

```python
V4_PARAMS = {
    # Entry
    "entry_z": 2.0,
    "unc_pct_max": 0.5,
    "warmup": 210,

    # Trail Stop - Rolling K
    "trail_k_horizon": 6,      # 6봉 = 1일
    "trail_k_window": 180,     # 180봉 = 30일
    "trail_k_quantile": 0.99,  # P99

    # Hard Stop - Rolling MAE
    "mae_horizon": 6,
    "mae_window": 180,
    "mae_quantile": 0.99,      # P99

    # Innovation Breaker
    "v_ratio_threshold": 1.0,
    "innov_base_mult": 2.5,
    "innov_mult_min": 1.5,
    "innov_mult_max": 4.0,

    # Gaussian Position Sizing
    "risk_target": 0.03,       # 3%
    "gauss_max_mult": 2.0,
    "gauss_sigma": 0.5,
    "size_min": 0.1,
    "size_max": 3.0,
}
```

## Files

| File | Description |
|------|-------------|
| `strategy_v35_go.py` | V4 메인 전략 (Go 백테스터 연동) |
| `test_go_backtester.py` | Go 백테스터 동작 검증 테스트 |
| `strategy.py` | 이전 버전 (참고용) |
| `backtester.py` | Python 백테스터 (참고용) |

## Usage

```bash
# 백테스트 실행
python strategies/v4/strategy_v35_go.py

# 테스트 실행
python strategies/v4/test_go_backtester.py
```

## Changelog

### V4 (2025-01-09)
- 미래참조 버그 수정 (horizon 지연 추가)
- 롱/숏 MAE 분리
- 로그 스케일 통일
- 가격 변동성 기반 Dynamic K (트레이드 기반 → 가격 기반)
- Position Sizing에서 MAE를 raw 퍼센트로 변환

### V3.5
- Dual-Eye Kalman Trail Stop
- Rolling MAE Hard Stop
- Gaussian Position Sizing

### V3.4
- Gaussian sigma 기반 포지션 사이징
- Low/High stops
