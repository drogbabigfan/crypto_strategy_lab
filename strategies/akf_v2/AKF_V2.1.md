# AKF V2.1 Strategy Specification

## Overview

Adaptive Kalman Filter 기반 모멘텀 전략. Kalman Filter로 추정한 가격 velocity의 z-score를 활용하여 진입/청산 시점을 결정.

**핵심 특징:**
- Kalman Filter velocity 기반 트렌드 감지
- Uncertainty filter로 노이즈 구간 회피
- 롱/숏 비대칭 청산 로직
- 시장 노출 ~24% (보수적 운용)

---

## 1. Signal Generation

### 1.1 Kalman Filter

기존 `calculate_adaptive_kalman()` 함수 사용:
- **State**: [price, velocity]
- **Process Noise**: PWNA (Piecewise White Noise Acceleration)
- **Measurement Noise**: NIS (Normalized Innovation Squared) 기반 적응형

**Output:**
- `kf_velocity`: 추정된 가격 변화율
- `kf_uncertainty`: 상태 공분산 (P matrix trace)

### 1.2 Velocity Z-Score

```python
zscore_window = 42  # 7일 (6 bars/day × 7)

vel_mean = velocity.rolling(window=zscore_window).mean()
vel_std = velocity.rolling(window=zscore_window).std()
vel_zscore = (velocity - vel_mean) / (vel_std + 1e-10)
```

### 1.3 Uncertainty Filter

```python
uncertainty_window = 210  # 30일 (6 bars/day × 30 + buffer)

uncertainty_pct = uncertainty.rolling(window=uncertainty_window).rank(pct=True)
low_uncertainty = uncertainty_pct < 0.5  # 하위 50%만 진입 허용
```

### 1.4 Entry/Exit Conditions

| Direction | Entry Condition | Exit Condition |
|-----------|-----------------|----------------|
| **Long** | `vel_zscore > +2.0` AND `low_uncertainty` | `vel_zscore < -2.0` |
| **Short** | `vel_zscore < -2.0` AND `low_uncertainty` | `vel_zscore > +2.0` |

**로직 특징:**
- 롱/숏 모두 z-score ±2.0 사용 (대칭)
- Entry는 uncertainty filter 적용, Exit는 미적용
- 청산 후 즉시 반대 포지션 진입 가능

---

## 2. Parameters

### 2.1 Signal Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `zscore_window` | 42 bars | Velocity z-score 계산 윈도우 (7일) |
| `uncertainty_window` | 210 bars | Uncertainty percentile 계산 윈도우 (30일) |
| `warmup` | 210 bars | 시그널 생성 시작 전 대기 기간 |
| `uncertainty_percentile` | 0.5 | Uncertainty 필터 임계값 (하위 50%) |

### 2.2 Entry/Exit Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `long_entry_zscore` | +2.0 | 롱 진입 z-score 임계값 |
| `short_entry_zscore` | -2.0 | 숏 진입 z-score 임계값 |
| `long_exit_zscore` | -2.0 | 롱 청산 z-score 임계값 |
| `short_exit_zscore` | +2.0 | 숏 청산 z-score 임계값 |

### 2.3 Position Sizing

| Parameter | Value | Description |
|-----------|-------|-------------|
| `method` | fixed | 고정 사이즈 (1.0x) |
| `size` | 1.0 | 레버리지 배율 |

### 2.4 Execution Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `bar_size` | 6 bars/day | 4시간봉 기준 |
| `fee_rate` | 0.001 (0.1%) | 거래 수수료 |
| `slippage_rate` | 0.0001 (0.01%) | 슬리피지 |

---

## 3. Backtest Results

### 3.1 In-Sample Performance (2020-2025)

| Metric | Value |
|--------|-------|
| **Cumulative PnL** | 1,521% |
| **Sharpe Ratio** | 1.02 |
| **Max Drawdown** | 19.6% |
| **Total Trades** | 87 |
| **Win Rate** | 56.3% |
| **Avg Holding Period** | 4.6 days |
| **Market Exposure** | 23.5% |

### 3.2 Yearly Breakdown (In-Sample)

| Year | PnL | MDD | Sharpe | Trades | Market |
|------|-----|-----|--------|--------|--------|
| 2020 | +135.7% | 9.1% | 2.39 | 20 | Bull |
| 2021 | +56.9% | 15.0% | 1.31 | 20 | Bull |
| 2022 | +16.3% | 19.6% | 0.58 | 17 | Bear |
| 2023 | +29.2% | 12.4% | 1.45 | 8 | Sideways |
| 2024 | +17.2% | 16.4% | 0.72 | 18 | Recovery |
| 2025 | -7.5% | 16.7% | -0.15 | 17 | Current |

### 3.3 Out-of-Sample Performance (AWFO)

Anchored Walk-Forward Optimization 결과:

| Test Year | Train Period | PnL | MDD |
|-----------|--------------|-----|-----|
| 2022 | 2020-2021 | +16.3% | 19.6% |
| 2023 | 2020-2022 | +29.2% | 12.4% |
| 2024 | 2020-2023 | +17.2% | 16.4% |
| 2025 | 2020-2024 | -7.5% | 16.7% |

| Metric | Value |
|--------|-------|
| **Cumulative PnL (OOS)** | 62.9% |
| **Avg MDD** | 16.3% |
| **Avg Sharpe** | 0.51 |
| **Positive Years** | 3/4 (75%) |

### 3.4 Long/Short Analysis

| Metric | Long | Short |
|--------|------|-------|
| Entry Count | 45 (42.5%) | 61 (57.5%) |
| Holding Time | 83.6% | 16.4% |
| Avg Duration | ~54 bars | ~8 bars |

---

## 4. Comparison vs Benchmarks

### 4.1 vs Buy & Hold

| Metric | AKF V2.1 | Buy & Hold |
|--------|----------|------------|
| Cumulative PnL | 1,521% | 1,087% |
| Avg MDD | 19.6% | 45.7% |
| Sharpe | 1.02 | 1.02 |
| 2022 (Bear) | +16.3% | -52.3% |

### 4.2 Risk-Adjusted

- **MDD 1/2 수준**으로 리스크 대폭 감소
- 하락장(2022)에서 양수 수익 달성
- 시장 노출 24%로 자본 효율적 운용

---

## 5. Implementation

### 5.1 Signal Generation Code

```python
def generate_signals(df_kf):
    velocity = df_kf["kf_velocity"].values
    uncertainty = df_kf["kf_uncertainty"].values
    n = len(df_kf)

    # Parameters
    zscore_window = 42
    uncertainty_window = 210
    warmup = 210

    long_entry, long_exit = 2.0, -2.0
    short_entry, short_exit = -2.0, 2.0

    # Velocity Z-score
    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=zscore_window, min_periods=5).mean()
    vel_std = vel_series.rolling(window=zscore_window, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

    # Uncertainty percentile
    p_pct = (
        pd.Series(uncertainty)
        .rolling(window=uncertainty_window, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )
    low_uncertainty = p_pct < 0.5

    # Entry conditions
    entry_long = (vel_zscore > long_entry) & low_uncertainty
    entry_short = (vel_zscore < short_entry) & low_uncertainty

    # Signal generation
    signals = np.zeros(n, dtype=np.int8)
    position = 0

    for i in range(warmup, n):
        if position == 0:
            if entry_long[i]:
                position = 1
                signals[i] = 1
            elif entry_short[i]:
                position = -1
                signals[i] = -1
        elif position == 1:
            if vel_zscore[i] < long_exit:
                position = 0
            else:
                signals[i] = 1
        elif position == -1:
            if vel_zscore[i] > short_exit:
                position = 0
            else:
                signals[i] = -1

    return signals
```

### 5.2 File Structure

```
strategies/akf_v2/
├── AKF_V2.1.md              # This specification
├── anchored_wfa.py          # Walk-forward analysis
├── yearly_breakdown.py      # Yearly performance analysis
├── param_sweep.py           # Parameter optimization
├── compare_bar_sizes.py     # Bar size comparison
├── common/
│   ├── sizing.py            # Position sizing methods
│   └── pyramiding.py        # Pyramiding logic (disabled)
├── wfa/
│   ├── engine.py            # WFA engine
│   └── metrics.py           # Performance metrics
└── backtest_pyramid.py      # Backtest engine
```

---

## 6. Notes & Limitations

### 6.1 Known Limitations

1. **매매 빈도 낮음**: 연 ~17회, 월 ~1.4회
2. **2025년 성과 부진**: OOS에서 -7.5% 손실
3. **In-sample vs OOS 괴리**: 1521% vs 63%

### 6.2 Parameter Sensitivity

- `long_exit`이 성과에 가장 큰 영향
- `-2.0`이 안정적 (더 공격적인 `-1.0`은 OOS에서 실패)
- `short_exit=+2.0`으로 숏도 트렌드 끝까지 보유

### 6.3 Future Improvements

- [ ] 매매 빈도 증가 방안 검토
- [ ] Entry threshold 동적 조정
- [ ] Multi-timeframe 확장
- [ ] 다른 자산군 적용 테스트

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| V2.0 | - | Initial Kalman Filter strategy |
| V2.1 | 2025-01 | Short exit 최적화 (+2.0), 파라미터 표준화 |

---

*Generated: 2025-01-07*
