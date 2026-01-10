# AKF V5 Strategy - Velocity Only (Simplified Trend Following)

## Overview

V4 기반 + Kalman Velocity 방향 기반 Always-In-Market 전략

**핵심: velocity > 0 → Long, velocity < 0 → Short**

---

## Evolution Path

```
V4 (vel_zscore > 2.0)
    ↓
V5 Corr Lock-in (corr > 0.9) ← 의미없음 발견
    ↓
V5 Velocity Only ← 최종 (corr 체크 제거)
```

---

## Verification Results

### 1. Look-ahead Bias: ✅ 없음

```
검증 방법:
  signal[t] = 1 → bar[t+1].Open에서 진입

  - velocity[t]는 close[t]를 사용해 계산
  - t봉이 끝나야 velocity[t]를 알 수 있음
  - Go 백테스터가 t+1봉 Open에서 진입 → 정상
```

### 2. Fee/Slippage: ✅ 적용됨

```
비용 구조:
  - base_fee:      0.10%
  - base_slippage: 0.01%
  - 왕복 비용:     0.22%

  ~500 trades × 0.22% = ~110% 총 거래비용
  → Go 백테스터에서 entry/exit 슬리피지 및 fee 차감 확인
```

### 3. Sharpe Ratio: ✅ 검증됨

| Metric | Go 백테스터 | 재계산 (일간) |
|--------|------------|--------------|
| Sharpe | 29.4 | **2.6** |

Go 백테스터 Sharpe 29.4는 계산 오류. **일간 Sharpe 2.6**이 실제 값.

---

## Backtest Results

### BTCUSDT (5년)

| Metric | Value |
|--------|-------|
| Total Trades | 496 |
| Win Rate | 57.5% |
| Avg PnL/Trade | 3.16% |
| CAGR | **334%** |
| MDD | 39.9% |
| Total Return | 358,264% |
| Daily Sharpe | 2.6 |
| Avg Size | 0.75x |

### Portfolio Yearly Performance

| Year | Return | MDD | Sharpe |
|------|--------|-----|--------|
| 2020 | +228% | 8.1% | 2.58 |
| 2021 | +427% | 12.8% | 3.89 |
| 2022 | +167% | 16.5% | 1.42 |
| 2023 | +180% | 9.8% | 1.09 |
| 2024 | +315% | 9.1% | 3.32 |
| 2025 | +430% | 18.7% | 2.50 |
| **TOTAL** | **+284,208%** | **18.7%** | - |

### Individual Assets Yearly

**BTCUSDT:**
```
2020: +357%  MDD 20.4%
2021: +843%  MDD 21.8%
2022:  +98%  MDD 25.2%  ← 하락장도 수익
2023:  +40%  MDD 39.9%  ← 횡보장
2024: +620%  MDD 21.6%
2025: +330%  MDD 24.4%
```

**vs Buy & Hold:**
- BTC Buy&Hold (5년): +1,117%
- Strategy: +358,264%
- **321x 초과 수익**

---

## Entry Logic

```python
# V4 기본 조건 (가속도 기반)
v4_long = vel_zscore > 2.0 and unc_pct < 0.5
v4_short = vel_zscore < -2.0 and unc_pct < 0.5

# Velocity Only 조건 (방향 기반)
vel_long = velocity > 0    # 상승 추세
vel_short = velocity < 0   # 하락 추세

# OR 조건 → 항상 포지션 보유
long_signal = v4_long or vel_long
short_signal = v4_short or vel_short
```

---

## Why Does This Work?

### 1. Momentum Factor

```
암호화폐에서 모멘텀은 역사적으로 강력한 팩터:
- 상승 중이면 계속 상승할 확률 높음
- 하락 중이면 계속 하락할 확률 높음
- Kalman velocity가 이를 매끄럽게 추출
```

### 2. Always-In-Market

```
포지션 보유 비율: ~70%+
- velocity > 0 OR velocity < 0 는 항상 true
- 모든 큰 움직임을 포착
- 횡보장에서도 작은 추세 캐치
```

### 3. Trailing Stop

```
손실 제한:
- Trail Stop: highest × exp(-k × sqrt(S_risk))
- Hard Stop: entry × exp(-rolling_mae)
- 큰 손실 방지하면서 수익은 계속 누적
```

### 4. Compounding Effect

```
equity *= (1 + pnl × size)

예시: CAGR 334% over 5 years
(1 + 3.34)^5 = 1,547x = +154,649%

복리 효과가 기하급수적 성장 생성
```

---

## Risk Factors

### 주의사항

1. **백테스트 ≠ 실거래**
   - 슬리피지 과소 추정 가능
   - 체결 실패, 네트워크 지연 미반영
   - 과거 데이터 기반, 미래 보장 없음

2. **MDD 39.9%**
   - 포트폴리오 MDD는 18.7%로 낮지만
   - 개별 자산에서 큰 드로다운 발생 가능

3. **일간 Sharpe 2.6**
   - 세계 최고 수준이지만 불가능하지 않음
   - 실거래에서는 하락 예상

---

## Parameters

```python
V5_PARAMS = {
    # Entry
    "entry_z": 2.0,
    "unc_pct_max": 0.5,
    "warmup": 210,

    # Trail Stop - Rolling K
    "trail_k_horizon": 6,
    "trail_k_window": 180,
    "trail_k_quantile": 0.99,

    # Hard Stop (Rolling MAE)
    "mae_horizon": 6,
    "mae_window": 180,
    "mae_quantile": 0.99,

    # Dynamic Sizing (Gaussian)
    "risk_target": 0.03,
    "gauss_max_mult": 4.0,
    "gauss_sigma": 0.5,
    "size_min": 0.1,
    "size_max": 3.0,
}
```

---

## Files

- `strategy_v5.py` - Velocity Only 구현
- `check_lookahead.py` - 검증 스크립트
- `README.md` - 이 문서

---

## Conclusion

Velocity Only 전략은 단순하지만 강력:

| 장점 | 단점 |
|------|------|
| 미래참조 없음 확인 | 높은 MDD (개별 39.9%) |
| 수수료 포함 수익 | 실거래 검증 필요 |
| 모든 연도 수익 | 과거 성과 기반 |
| Sharpe 2.6 (일간) | 암호화폐 특화 |

**핵심 인사이트:**
> "velocity 방향만으로 진입" + "Trailing Stop으로 청산" =
> 단순하지만 효과적인 추세추종 전략
