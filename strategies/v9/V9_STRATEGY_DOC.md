# V9 Strategy - Dual Kalman Long Only

## Final Strategy

**Dual Kalman Filter 기반 Long Only + Trail Pyramid**

---

## Performance Summary

| Metric | Value |
|--------|-------|
| **CAGR** | **48.2%** |
| **MDD** | **21.2%** |
| **Sharpe** | **1.19** |
| **Calmar** | **2.27** |
| Trades | 204 (5년, 4심볼) |
| Win Rate | 34.5% |
| Avg Entries | 2.59 |

---

## Configuration

### Entry Signal: VelTrend

```python
def long_signal(bar):
    return (fast_zscore[bar] > 2.0 and
            duration[bar] < duration_ma[bar] * 0.5 and
            slow_zscore[bar] > 1.0)
```

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| fast_window | 20 | Fast Kalman window |
| slow_window | 120 | Slow Kalman window |
| fast_zscore_threshold | 2.0 | Fast Z-score 진입 임계값 |
| slow_zscore_threshold | 1.0 | Slow Z-score 필터 |
| duration_ratio | 0.5 | Duration 필터 (< MA × 0.5) |
| K | 1.5 | Stop/TP 거리 배수 |
| risk_per_trade | **0.07** | 트레이드당 리스크 (7%) |
| pyramid_threshold | **0.75** | 피라미드 트리거 (0.75×SD) |
| max_pyramids | **5** | 최대 피라미드 횟수 |
| pyramid_size_ratio | **0.5** | 피라미드 사이즈 (초기의 50%) |
| max_total_size | **2.5** | 최대 총 레버리지 |

---

## Entry Logic

### Dual Kalman Filter

```python
# Fast Kalman (window=20) - 빠른 반응
fast_zscore = fast_velocity / sqrt(fast_velocity_var)

# Slow Kalman (window=120) - 추세 확인
slow_zscore = slow_velocity / sqrt(slow_velocity_var)

# Duration Filter - 거래량 급증 확인
duration = exp(log_duration)
duration_ma = rolling_mean(duration, 50)
```

### VelTrend Signal

```
진입 조건 (AND):
1. fast_zscore > 2.0      (빠른 모멘텀 확인)
2. duration < ma × 0.5    (거래량 급증)
3. slow_zscore > 1.0      (추세 방향 확인)
```

---

## Exit Logic

### Stop/TP Distance

```python
stop_distance = K × Parkinson × √horizon × √compression
# K = 1.5, horizon = 10

initial_stop = entry_price × (1 - stop_distance)
initial_tp = entry_price × (1 + stop_distance)
```

### Trail Pyramid

```python
# 피라미드 트리거: 가격이 0.75×SD 상승시
if high >= next_pyramid_price:
    add_position(size=initial_size × 0.5)

    # Stop/TP 트레일
    trail_amount = 0.75 × initial_sd
    current_stop *= (1 + trail_amount)
    current_tp *= (1 + trail_amount)

    # 다음 피라미드 가격 설정
    next_pyramid_price *= (1 + 0.75 × initial_sd)
```

---

## Position Sizing

```python
# 초기 사이즈
initial_size = risk_per_trade / stop_distance
initial_size = clip(initial_size, 0.1, 1.0)

# 피라미드 사이즈
pyramid_size = initial_size × 0.5

# 최대 총 사이즈
total_size = min(sum(all_sizes), 2.5)
```

---

## Individual Asset Performance

| Symbol | Trades | WR | CAGR | MDD | Calmar |
|--------|--------|-----|------|-----|--------|
| BTCUSDT | 51 | 37.3% | 30.7% | 45.0% | 0.68 |
| **ETHUSDT** | 45 | 42.2% | **73.2%** | 42.1% | **1.74** |
| XRPUSDT | 64 | 26.6% | 7.2% | 77.1% | 0.09 |
| SOLUSDT | 44 | 31.8% | 44.6% | 36.6% | 1.22 |

---

## Tested Alternatives (Not Recommended)

### 1. Short Strategy
- 숏 승률 28%로 엣지 없음
- Long Only가 더 나은 성과

### 2. OR Signal Combinations
- AND 조합이 노이즈 필터링에 효과적
- OR 조합은 Calmar 저하

### 3. Threshold Relaxation
- 조건 완화시 노이즈 진입 증가
- 현재 설정이 최적

### 4. Signal-based Pyramid
- 시그널 기반 피라미딩은 성과 저하
- Price-based만 사용

### 5. Parameter Variations

| Variation | CAGR | MDD | Calmar |
|-----------|------|-----|--------|
| **Original 7%/5pyr** | **48.2%** | **21.2%** | **2.27** |
| Risk 5%/5pyr | 31.4% | 17.5% | 1.79 |
| Risk 10%/5pyr | 55.7% | 24.7% | 2.25 |
| Pyramid 0.5 threshold | 44.6% | 23.9% | 1.87 |

---

## Files

| File | Description |
|------|-------------|
| `V9_STRATEGY_DOC.md` | 이 문서 |
| `strategy_v9_long.py` | 전략 코드 |

---

## Usage

```bash
# 시그널 비교 (No Pyramid)
python strategies/v9/strategy_v9_long.py

# 피라미딩 비교
python strategies/v9/strategy_v9_long.py pyramid

# 최종 전략 실행
python strategies/v9/strategy_v9_long.py pyramid
```

---

## Changelog

- 2025-01-25: 최종 전략 확정 (VelTrend + Pyramid, 7%/5pyr)
- 2025-01-25: 파라미터 최적화 테스트 완료
- 2025-01-25: Phase 4 완료 - 숏 전략 테스트 (비권장)
- 2025-01-25: Phase 3 완료 - 피라미딩 적용
- 2025-01-25: Phase 1-2 완료 - VelTrend 최적 시그널 확정
