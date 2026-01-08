# AKF (Adaptive Kalman Filter) Trading Strategies

## Overview

BTC/USDT 선물 시장을 위한 Kalman Filter 기반 트레이딩 전략 연구 및 구현.

- **백테스트 기간**: 2020-01 ~ 2025-11 (71개월, 12,237 bars)
- **데이터**: 5min Dollar Bars (BTCUSDT Futures)
- **Backtester**: Go 기반 signal-mode (look-ahead bias 수정 완료)

---

## 🏆 전략 성능 비교

### Look-ahead Bias 수정 후 결과

| 전략 | 파일 | 거래수 | PnL | Sharpe | MDD | PF | Status |
|------|------|--------|-----|--------|-----|-----|--------|
| **DualTrack** | `strategies_dual_track.py` | 168 | **+101.3%** | **0.95** | **12.4%** | **1.88** | ⭐ Best |
| Advanced | `strategies_advanced.py` | 168 | +31.6% | 0.45 | 18.3% | 1.30 | Good |
| DKF Crossover | `strategies_dkf.py` | 440 | +13.2% | 0.18 | - | 1.03 | Fair |
| Model 4 | `strategies_model4.py` | - | -50% | - | - | - | ❌ Failed |
| V1-V3 | `strategies*.py` | - | -50% | - | - | - | ❌ Failed |
| Mean Reversion | `strategies_mr.py` | - | -50% | - | - | - | ❌ Failed |

> ⚠️ **2026-01-05**: Look-ahead bias 수정 후 기존 단순 밴드 돌파 전략들은 전부 실패.
> KF 파생 지표 + Regime Detection + Exit Buffer 조합만 수익 달성.

---

## 전략 상세

### 1. AKFDualTrackStrategy ⭐ (권장)

**파일**: `strategies_dual_track.py`

#### 성능
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Trades:   168 (월 2.4회)
Win Rate:       52.4%
Total PnL:      +101.3%
Sharpe Ratio:   0.95
Max Drawdown:   12.4%
Profit Factor:  1.88
Annualized PnL: 17.1%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### 로직 흐름
```
Input: OHLCV Data
    │
    ▼
┌─────────────────────────────────┐
│  1. Kalman Filter               │
│  → trend, velocity, uncertainty │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  2. Feature Engineering         │
│  • kf_rsi = RSI(trend, 14)      │
│  • z_score = (close-pred)/std   │
│  • efficiency = |vel|/unc       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  3. Regime Detection            │
│  Trending = eff>0.4 & unc<95%   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  4. Entry (Breakout Only)       │
│  Long:  regime=1 & vel>0 &      │
│         z>1.75 & rsi<75         │
│  Short: regime=1 & vel<0 &      │
│         z<-1.75 & rsi>25        │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  5. Exit (With Buffer) ⭐        │
│  Long:  close < trend - 1.5σ    │
│  Short: close > trend + 1.5σ    │
└─────────────────────────────────┘
    │
    ▼
Output: signal (1=Long, -1=Short, 0=Flat)
```

#### 최적 파라미터
```python
DualTrackConfig(
    efficiency_threshold=0.4,
    z_score_breakout=1.75,
    rsi_max_breakout=75,
    rsi_min_breakout=25,
    exit_buffer_mult=1.5,  # ⭐ 핵심!
)
```

---

### 2. AKFAdvancedStrategy

**파일**: `strategies_advanced.py`

#### 성능
```
Total Trades:   168
Win Rate:       47.6%
Total PnL:      +31.6%
Sharpe Ratio:   0.45
```

#### DualTrack과의 차이
- Exit buffer 없음 (`close < trend`로 청산)
- 동일 entry 파라미터
- **Exit buffer가 +69.7% PnL 차이**

---

### 3. DKFStrategy (Dual Kalman Filter Crossover)

**파일**: `strategies_dkf.py`

#### 개념
두 개의 Kalman Filter를 다른 민감도로 실행:
- **Fast KF** (R=0.5): 빠른 반응
- **Slow KF** (R=2.0): 안정적 추세

#### 로직
```python
# Indicators
trend_fast = KalmanFilter(price, R=0.5)
trend_slow = KalmanFilter(price, R=2.0)
spread = trend_fast - trend_slow  # K-MACD

# Entry
Long:  spread crosses above 0 (Golden Cross)
Short: spread crosses below 0 (Dead Cross)

# Filter
velocity_slow > -0.05 for Long
velocity_slow < 0.05 for Short
```

#### 성능
```
Total Trades:   440
Win Rate:       34.8%
Total PnL:      +13.2%
Sharpe Ratio:   0.18
```

#### 분석
- DualTrack 대비 성능 열등 (PnL 13% vs 101%)
- Regime Detection 없음 → 노이즈에 취약
- 단순 교차 신호의 한계

---

### 4. 실패한 전략들 (Deprecated)

#### V1: 단순 밴드 돌파
```python
Long: close > trend + k * sqrt(uncertainty)
Exit: close < trend
```
**결과**: -50% 조기 종료

#### V2: Velocity 필터 추가
```python
Long: close > upper_band AND velocity > 0
```
**결과**: -50% 조기 종료

#### V3: Always-in-Market (Reversal)
```python
# 청산 대신 반대 포지션
if close > trend: position = -1
```
**결과**: -50% 조기 종료

#### Model 4: Benhamou + Oscillator
```python
Long: kf_trend_pred > prev_close + delta AND stoch < 80
```
**결과**: -50% 조기 종료

#### Mean Reversion
```python
Long: close < lower_band (oversold → bounce)
```
**결과**: -50% 조기 종료

---

## 핵심 발견사항

### 1. Exit Buffer의 중요성 ⭐
```
기존: close < trend → 청산
개선: close < trend - 1.5σ → 청산

결과:
  PnL:    +31.6% → +101.3% (+69.7%)
  Sharpe: 0.45 → 0.95 (+0.50)
  MDD:    18.3% → 12.4% (-5.9%)
```

### 2. KF 직접 사용 vs 파생 지표
```
❌ 실패: close > upper_band → Long
✅ 성공: z_score > 1.75 AND regime=1 AND rsi<75 → Long
```

### 3. Regime Detection 필수
```
Without Regime Filter: DKF +13% PnL
With Regime Filter:    DualTrack +101% PnL
```

### 4. Pullback 진입 역효과
```
Breakout Only:  +101% PnL
Breakout + Pullback: +22% PnL
```

### 5. Z-Score 임계값
```
z = 1.0:  노이즈, 손실
z = 1.75: ⭐ Sweet Spot, +101%
z = 2.5:  거래 감소, +6%
```

---

## Look-ahead Bias 수정

### 발견된 버그 (2026-01-05)
```
Go Backtester에서:
- Signal[i]는 close[i]로 생성 (bar 끝에서 확정)
- Exit는 bar[i].Open에서 실행 (bar 시작)
→ 미래 정보 사용
```

### 수정 내용
```go
// executor.go
type Executor struct {
    prevSignal Signal  // 이전 bar의 signal 저장
}

func (e *Executor) ProcessBar(...) {
    // Exit 시 prevSignal 사용 (current가 아닌)
    e.checkSignalExit(barIdx, bar, e.prevSignal, avgVolume)

    // 현재 signal 저장
    e.prevSignal = signal
}
```

### 수정 후 영향
- 기존 전략들: 전부 -50% 손실
- 새 전략 (DualTrack): +101% 수익

---

## 사용법

### 권장 전략 (DualTrack)
```python
from strategies.akf.strategies_dual_track import (
    AKFDualTrackStrategy,
    prepare_kalman_data
)
from research.features.kalman import calculate_adaptive_kalman

# 1. Kalman Filter 적용
df_kf = calculate_adaptive_kalman(df)

# 2. 컬럼 매핑
df_prepared = prepare_kalman_data(df_kf)

# 3. 전략 실행
strategy = AKFDualTrackStrategy()
df_signals = strategy.run_strategy(df_prepared)

# 4. 신호 추출
signals = df_signals['signal']  # 1=Long, -1=Short, 0=Flat
```

### DKF 전략
```python
from strategies.akf.strategies_dkf import DKFStrategy, DKFConfig

config = DKFConfig(
    fast_r_scale=0.5,
    slow_r_scale=2.0,
    velocity_filter_long=-0.05
)
strategy = DKFStrategy(config)
df_signals = strategy.run_strategy(df)
```

---

## 파일 구조

```
strategies/akf/
├── README.md                     # 이 문서
├── __init__.py
│
├── strategies_dual_track.py      # ⭐ Best (PnL +101%, Sharpe 0.95)
├── strategies_advanced.py        # Good (PnL +31%, Sharpe 0.45)
├── strategies_dkf.py             # Fair (PnL +13%, Sharpe 0.18)
│
├── strategies.py                 # V1 (deprecated)
├── strategies_v2.py              # V2 (deprecated)
├── strategies_v3.py              # V3 (deprecated)
├── strategies_mr.py              # Mean Reversion (deprecated)
├── strategies_model4.py          # Benhamou Model 4 (deprecated)
│
├── test_*.py                     # 테스트 스크립트
└── results/                      # 백테스트 결과
```

---

## 의존성

### Python
```
research/features/kalman.py  # Adaptive Kalman Filter
  - calculate_adaptive_kalman()
  - kf_trend, kf_velocity, kf_uncertainty, kf_trend_pred
```

### Go Backtester
```
etl/internal/backtest/
  - executor.go   # Signal processing (prevSignal 수정)
  - runner.go     # Backtest runner
  - metrics.go    # Performance metrics
```

---

## 향후 연구

- [ ] Walk-forward validation (rolling window)
- [ ] 다른 자산 (ETH, SOL) 적용
- [ ] 거래 비용 민감도 분석
- [ ] Position sizing 최적화 (Kelly, Risk Parity)
- [ ] 실시간 시그널 생성 파이프라인

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-01-05 | Look-ahead bias 발견 및 수정 |
| 2026-01-05 | DualTrack 전략 구현 (+101% PnL) |
| 2026-01-05 | DKF Crossover 전략 구현 (+13% PnL) |
| 2026-01-05 | 전략 문서화 완료 |

---

**Author**: Claude Code
**Last Updated**: 2026-01-05
