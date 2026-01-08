# Kalman Filter Model Improvements

## 개요

AKF V2 전략의 칼만필터 모델을 개선하여 성능을 크게 향상시켰습니다.

| 지표 | 기존 | 개선 | 변화 |
|------|------|------|------|
| **PnL** | 282.8% | **382.6%** | +35% |
| **Sharpe** | 1.07 | **1.35** | +26% |
| **MDD** | 17.3% | **14.5%** | -16% |

---

## 1. 핵심 개선 사항

### 1.1 Q Scale 최적화

```python
# 기존
q_scale = 0.1

# 개선
q_scale = 0.4
```

**효과**: 필터가 모멘텀 변화에 더 빠르게 반응

---

### 1.2 PWNA Q Matrix (Piecewise White Noise Acceleration)

#### 문제: 기존 Diagonal Q의 한계

기존 방식은 위치(position)와 속도(velocity)의 process noise를 독립적으로 설정:

```
Q = [[Q_pos, 0    ],
     [0,     Q_vel]]
```

이는 **물리적으로 불가능한 모순**을 야기합니다:
- Constant Velocity 모델에서 `Position_{t+1} = Position_t + Velocity_t`
- 속도가 변하면 위치도 **반드시** 변해야 함
- 그러나 Diagonal Q는 이 커플링을 무시

#### 해결: PWNA 모델

가속도 노이즈(σ_a²) 하나로 물리적으로 정확한 Q 행렬 생성:

$$Q = \sigma_a^2 \begin{bmatrix} \frac{\Delta t^4}{4} & \frac{\Delta t^3}{2} \\ \frac{\Delta t^3}{2} & \Delta t^2 \end{bmatrix}$$

Δt = 1 (1 bar) 일 때:

$$Q = \sigma_a^2 \begin{bmatrix} 0.25 & 0.5 \\ 0.5 & 1.0 \end{bmatrix}$$

**핵심 특징**:
- Off-diagonal terms ≠ 0 → 위치-속도 오차가 자동으로 커플링
- 튜닝 파라미터가 2개(Q_pos, Q_vel) → **1개(σ_a²)**로 단순화
- 물리 법칙에 부합하는 오차 전파

```python
# 구현 (kalman.py)
def _get_q_matrix(self, base_q, nis_boost=1.0):
    sigma_a_sq = base_q * self.config.sigma_a_scale * nis_boost

    if self.config.use_pwna_q:
        dt = 1.0
        q_matrix = sigma_a_sq * np.array([
            [dt**4 / 4, dt**3 / 2],
            [dt**3 / 2, dt**2]
        ])
    else:
        q_matrix = np.diag([sigma_a_sq, sigma_a_sq])

    return np.maximum(q_matrix, self.config.min_q)
```

---

### 1.3 NIS 기반 적응형 Q 조정

#### NIS (Normalized Innovation Squared)

$$\alpha = \frac{\text{Innovation}^2}{\text{Innovation Covariance}} = \frac{y^2}{S}$$

**해석**:
- NIS ≈ 1: 필터가 잘 튜닝됨
- NIS > 1: 예측이 틀림 → Q를 높여서 빠르게 적응
- NIS < 1: 필터가 과민반응

#### 적응 로직

```python
def _get_nis_boost(self, nis_history):
    window = min(len(nis_history), self.config.nis_window)
    recent_nis = np.array(nis_history[-window:])
    mean_nis = np.mean(recent_nis)

    # NIS > 1이면 Q를 boost (최대 nis_boost_factor까지)
    boost = max(1.0, min(mean_nis, self.config.nis_boost_factor))
    return boost
```

**효과**: 추세 급변 시 지연(Lag) 없이 즉각 반응

---

## 2. 파라미터 설정

### 최적화된 기본값 (KalmanConfig)

```python
@dataclass
class KalmanConfig:
    # Noise estimation
    r_window: int = 20
    q_window: int = 20

    # Scale factors
    r_scale: float = 1.0
    q_scale: float = 0.4        # 최적화됨 (기존 0.1)

    # NIS adaptive
    use_nis_adaptive: bool = True
    nis_window: int = 10
    nis_boost_factor: float = 2.0

    # PWNA model
    use_pwna_q: bool = True
    sigma_a_scale: float = 0.4  # 최적화됨
```

### 파라미터 탐색 결과

| σ_a | PnL% | Sharpe | MDD% |
|-----|------|--------|------|
| 0.2 | 278.6% | 1.10 | 17.9% |
| 0.3 | 317.2% | 1.20 | 17.0% |
| **0.4** | **382.6%** | **1.35** | **14.5%** |
| 0.5 | 370.8% | 1.35 | 14.5% |
| 0.6 | 354.1% | 1.26 | 16.9% |

---

## 3. 연도별 성과

| Year | Strategy | Buy&Hold | Alpha | Sharpe |
|------|----------|----------|-------|--------|
| 2020 | +124.7% | +305.5% | -180.8% | 2.31 |
| 2021 | +143.6% | +59.2% | **+84.4%** | 2.02 |
| 2022 | +23.6% | -52.3% | **+75.8%** | 0.52 |
| 2023 | +44.4% | +71.8% | -27.4% | 1.15 |
| 2024 | +49.3% | +141.3% | -92.0% | 1.25 |
| 2025 | -22.2% | -7.1% | -15.0% | -0.65 |

**특징**:
- 하락장(2022)에서 **+75.8% alpha** 생성
- 강한 상승장(2020, 2024)에서는 Buy & Hold 대비 underperform
- 전반적으로 리스크 조정 수익률(Sharpe) 우수

---

## 4. 실패한 시도들

### 4.1 3-State Model (Acceleration 추가)

State: [position, velocity, acceleration]

**결과**: 성능 악화
- 추세 MAE: $427 → $805 (2배 악화)
- 반전 감지: 98.8% → 83.2%

**원인**: 금융 시계열은 constant acceleration 모델을 따르지 않음

### 4.2 Diagonal Q 분리 (Q_pos ≠ Q_vel)

```python
Q = [[Q_pos, 0    ],
     [0,     Q_vel]]
```

**결과**: 성능 악화 (366.3% → 337.7%)

**원인**: 물리적 커플링 무시로 필터 최적화 실패

### 4.3 동적밴드 청산 (P 기반)

Exit: `close < kf_trend * exp(-k * sqrt(P))`

**결과**: 효과 미미
- k < 6: 성능 악화 (조기 청산)
- k ≥ 6: Z-score 청산과 동일 (발동 안 함)

**결론**: Z-score 기반 청산으로 충분

---

## 5. 전략 설정 요약

### Entry Conditions
```python
entry_long:  velocity_zscore > 1.0   # 1σ 상승
entry_short: velocity_zscore < -2.0  # 2σ 하락 (더 엄격)
```

### Exit Conditions
```python
exit_long:  velocity_zscore < -2.0 OR std_innovation < -3.0
exit_short: velocity_zscore > 0      # 빠른 청산
```

### Filters
```python
uncertainty_filter: P < 50th percentile  # 확신 높을 때만 진입
```

---

## 6. 동적 포지션 사이징

### 6.1 개요

칼만필터의 P(불확실성)를 활용한 동적 포지션 사이징으로 성능을 추가 향상.

**원리:**
- P 낮음 (확신 높음) → 포지션 증가 (최대 200%)
- P 높음 (불확실) → 포지션 감소 (최소 50%)

### 6.2 구현 방식

#### P-inverse 방식 (Percentile)

```python
# P의 역수를 percentile로 변환
p_inv = 1.0 / sqrt(P)
p_inv_pct = rolling_percentile(p_inv, window=100)

# 선형 매핑: 0~1 → min_size ~ max_size
position_size = min_size + (max_size - min_size) * p_inv_pct
```

#### Kelly Criterion

```python
def kelly_sizing(returns, window=250):
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_rate = len(wins) / len(returns)
    avg_win = wins.mean()
    avg_loss = -losses.mean()

    b = avg_win / avg_loss
    kelly_raw = win_rate - (1 - win_rate) / b

    # P(불확실성)로 조정
    p_ratio = p_mean / p_current  # P 낮으면 ratio 높음
    kelly_adjusted = kelly_raw * np.clip(p_ratio, 0.5, 2.0)

    return np.clip(kelly_adjusted, min_size, max_size)
```

### 6.3 성능 비교

| Config | PnL% | Sharpe | MDD% | PnL/MDD |
|--------|------|--------|------|---------|
| **Fixed 100%** | 382.6% | 1.35 | 14.5% | 26.33 |
| P-inv 25~150% | 466.0% | **1.38** | 15.2% | 30.57 |
| P-inv 50~200% | 635.7% | 1.37 | 17.3% | **36.71** |
| Kelly w=500 | ~400% | 1.21 | ~30% | ~13 |

### 6.4 추천 설정

| 목표 | 추천 설정 | 결과 |
|------|----------|------|
| **최고 Sharpe** | P-inv 25~150% | 1.38 |
| **최고 PnL/MDD** | P-inv 50~200% | 36.71 |
| **보수적** | Kelly w=500 | MDD 낮음 |

---

## 7. Go 백테스터

### 7.1 동적 포지션 사이징 지원

**수정된 파일:**

1. **`etl/internal/backtest/runner.go`**
   - `SignalRecord`에 `Size` 필드 추가 (optional parquet column)
   - `loadSignals()`: size 배열 반환

2. **`etl/internal/backtest/types.go`**
   - `Position`, `Trade` 구조체에 `Size` 필드 추가

3. **`etl/internal/backtest/executor.go`**
   - `ProcessBar()`: size 파라미터 추가
   - Entry/Exit 시 size를 Position/Trade에 저장

4. **`etl/internal/backtest/metrics.go`**
   - `buildEquityCurve()`: 각 거래의 `Size`를 equity 계산에 반영

### 7.2 Parquet 스키마

**signals.parquet:**
```python
schema = pa.schema([
    ('timestamp', pa.int64()),
    ('signal', pa.int8()),      # -1, 0, 1
    ('size', pa.float64()),     # Position size ratio (optional)
    ('sl_price', pa.float64()), # Stop loss price (optional, 미구현)
])
```

**features.parquet:**
```python
schema = pa.schema([
    ('timestamp', pa.int64()),
    ('open', pa.float64()),
    ('high', pa.float64()),
    ('high_time', pa.int64()),
    ('low', pa.float64()),
    ('low_time', pa.int64()),
    ('close', pa.float64()),
    ('volume', pa.float64()),
    ('realized_vol', pa.float64()),
])
```

### 7.3 사용법

```bash
./bin/backtester \
    -signals signals.parquet \
    -features features.parquet \
    -exit-mode signal \
    -compounding \
    -quiet
```

---

## 8. 코드 구조

### 핵심 파일

**Kalman Filter:**
- `research/features/kalman.py` - PWNA + NIS 적응형 칼만필터

**Strategy:**
- `strategies/akf_v2/final_strategy_comparison.py` - 전략 비교 테스트
- `strategies/akf_v2/base.py` - 기본 전략 클래스

**Backtester:**
- `etl/internal/backtest/` - Go 백테스터
- `etl/cmd/backtester/main.go` - CLI

---

## 9. 향후 TODO

### 9.1 Risk-based Stop Loss

진입 시점의 가격과 수량 기준으로 계좌 총손실 n%가 되는 가격을 스탑로스로 설정.

**공식:**
```python
# Long position
sl_price = entry_price * (1 - max_account_loss / position_size)

# Short position
sl_price = entry_price * (1 + max_account_loss / position_size)

# 예: entry=50000, size=2.0, max_loss=2%
# Long SL = 50000 * (1 - 0.02 / 2.0) = 50000 * 0.99 = 49500
```

**파라미터:**
- `max_account_loss`: 거래당 최대 계좌 손실 (default: 2%)

**구현 필요 사항:**
- [ ] Go 백테스터에 sl_price 처리 로직 추가
- [ ] 인트라바 SL 체크 (high_time/low_time 활용)
- [ ] Python 시그널 생성 시 sl_price 계산

### 9.2 Regime Detection

추세/횡보 구간별 다른 전략 파라미터 적용.

### 9.3 Multi-timeframe

여러 시간대 칼만필터 결합.

### 9.4 GARCH 기반 R

변동성 클러스터링을 반영한 측정 노이즈 추정.

---

*최종 수정: 2025-01*
