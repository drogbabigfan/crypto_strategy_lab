# Adaptive Kalman Filter

적응형 칼만 필터 구현 및 검증 문서.

## 목차

1. [개요](#1-개요)
2. [이론적 배경](#2-이론적-배경)
3. [구현 상세](#3-구현-상세)
4. [출력 피처](#4-출력-피처)
5. [검증 결과](#5-검증-결과)
6. [사용법](#6-사용법)
7. [파라미터 가이드](#7-파라미터-가이드)

---

## 1. 개요

### 1.1 목적

금융 시계열에서 노이즈를 제거하고 실제 추세(Trend)를 추정하기 위한 적응형 칼만 필터.

**핵심 특징:**
- **적응형 R_t**: 시장 변동성에 따라 측정 잡음 동적 조정
- **적응형 Q_t**: 모멘텀 강도에 따라 프로세스 잡음 동적 조정
- **포지션 사이징**: 오차 공분산(P)을 활용한 불확실성 기반 사이징
- **Typical Price**: (H+L+C)/3로 미시적 노이즈 감소

### 1.2 파일 구조

```
research/features/
├── kalman.py              # 핵심 구현
│   ├── KalmanConfig       # 설정 데이터클래스
│   ├── AdaptiveKalmanFilter  # 메인 필터 클래스
│   ├── KalmanFeatureGenerator  # 파이프라인 통합용
│   └── calculate_adaptive_kalman()  # 편의 함수

scripts/
├── verify_kalman_filter.py      # 기본 검증 (7개 테스트)
└── verify_kalman_advanced.py    # 고급 검증 (7개 테스트)

tests/
└── test_kalman_filter.py        # Unit 테스트 (26개)
```

---

## 2. 이론적 배경

### 2.1 상태 공간 모델

칼만 필터는 두 가지 방정식으로 정의:

**상태 방정식 (Process Equation):**
```
x_t = F * x_{t-1} + w_t,  w_t ~ N(0, Q_t)
```

**관측 방정식 (Measurement Equation):**
```
z_t = H * x_t + v_t,  v_t ~ N(0, R_t)
```

### 2.2 2-상태 모델

| 상태 | 의미 | 설명 |
|------|------|------|
| x[0] | Trend | 필터링된 가격 수준 |
| x[1] | Velocity | 추세의 변화율 (모멘텀) |

**상태 전이 행렬 (Constant Velocity Model):**
```
F = [[1, 1],    # trend_new = trend + velocity
     [0, 1]]    # velocity_new = velocity
```

**관측 행렬:**
```
H = [[1, 0]]    # 관측값 = trend만
```

### 2.3 칼만 이득 (Kalman Gain)

```
K_t = P_{t|t-1} * H^T * (H * P_{t|t-1} * H^T + R_t)^(-1)
```

- **R_t 증가** (변동성 ↑) → K_t 감소 → 스무딩 강화
- **Q_t 증가** (모멘텀 ↑) → P 증가 → K_t 증가 → 추적 민감도 증가

### 2.4 적응형 메커니즘

**R_t (측정 잡음) 추정:**
```python
# 최근 N개 봉의 가격 변화 분산
returns = np.diff(prices[t-window:t+1])
R_t = np.var(returns) * r_scale
```

**Q_t (프로세스 잡음) 추정:**
```python
# Velocity 분산 + 모멘텀 가속도 반영
vel_var = np.var(velocities[t-window:t+1])
accel = abs(velocity_diff).mean()
momentum_factor = 1 + q_momentum_weight * accel / abs_vel_mean
Q_t = vel_var * q_scale * momentum_factor
```

---

## 3. 구현 상세

### 3.1 KalmanConfig

```python
@dataclass
class KalmanConfig:
    r_window: int = 20          # R 추정 윈도우
    q_window: int = 20          # Q 추정 윈도우
    initial_p: float = 1.0      # 초기 오차 공분산
    r_scale: float = 1.0        # R 스케일 팩터
    q_scale: float = 0.1        # Q 스케일 팩터
    q_momentum_weight: float = 2.0  # 모멘텀 가중치
    use_typical_price: bool = True  # (H+L+C)/3 사용
    min_r: float = 1e-8         # 최소 R 값
    min_q: float = 1e-10        # 최소 Q 값
```

### 3.2 필터 알고리즘

```python
for each bar t:
    # === PREDICTION ===
    x_pred = F @ x
    Q_t = estimate_q(prices, velocities, t)
    P_pred = F @ P @ F.T + Q_matrix

    # === UPDATE ===
    R_t = estimate_r(prices, t)
    y = z - H @ x_pred  # Innovation
    S = H @ P_pred @ H.T + R_t  # Innovation covariance
    K = P_pred @ H.T / S  # Kalman gain

    x = x_pred + K * y
    P = (I - K @ H) @ P_pred
```

### 3.3 포지션 사이징

```python
def get_position_size_factor(uncertainty, min_unc, max_unc):
    """
    불확실성 기반 포지션 크기 계산.

    - 불확실성 높음 → 포지션 작게
    - 불확실성 낮음 → 포지션 크게
    """
    # Log-scale 정규화
    log_unc = log(clamp(uncertainty, min_unc, max_unc))
    normalized = 1 - (log_unc - log_min) / (log_max - log_min)
    return clamp(normalized, 0, 1)
```

---

## 4. 출력 피처

| 피처 | 설명 | 용도 |
|------|------|------|
| `kf_trend` | 필터링된 추세 가격 | 기준 가격, 지지/저항 |
| `kf_velocity` | 추세 속도 (모멘텀) | 추세 방향/강도 |
| `kf_deviation` | 가격 - 추세 | Mean reversion 신호 |
| `kf_deviation_pct` | 편차 백분율 | 상대적 편차 |
| `kf_gain` | 칼만 이득 | 필터 반응성 모니터링 |
| `kf_uncertainty` | 오차 공분산 P | 포지션 사이징 |
| `kf_signal` | 편차 z-score | 정규화된 매매 신호 |

### 4.1 신호 해석

**kf_deviation / kf_signal:**
- 양수: 가격이 추세 위 → 과매수 / Short 신호
- 음수: 가격이 추세 아래 → 과매도 / Long 신호

**kf_velocity:**
- 양수 & 증가: 상승 추세 가속
- 양수 & 감소: 상승 추세 둔화
- 음수: 하락 추세

**kf_uncertainty:**
- 높음: 추세 불확실 → 포지션 축소
- 낮음: 추세 명확 → 포지션 확대

---

## 5. 검증 결과

### 5.1 기본 검증 (7/7 통과)

| 테스트 | 결과 | 핵심 지표 |
|--------|------|----------|
| Basic Functionality | PASS | 모든 컬럼 존재, inf/NaN 없음 |
| Trend Tracking | PASS | Correlation: 0.9984 |
| Deviation Statistics | PASS | ADF p-value: 0.000 (Stationary) |
| Signal Distribution | PASS | Mean: 0.002, Std: 1.071 |
| Adaptive Behavior | PASS | High Vol Uncertainty 3.2x 증가 |
| Position Sizing | PASS | Size-Unc Corr: -0.742 |
| Typical vs Close | PASS | Noise 23% 감소 |

### 5.2 고급 검증 (7/7 통과)

| 테스트 | 결과 | 핵심 지표 |
|--------|------|----------|
| **Step Function** | PASS | Convergence: 22 bars, Spike: 6.5x |
| **Outlier Robustness** | PASS | Attenuation: 99.4% |
| **Parameter Sensitivity** | PASS | Min Corr: 0.965 (100x 범위) |
| **Innovation Whiteness** | PASS | ACF(1): 0.651 < 0.8 |
| **Lag vs Smoothness** | PASS | 8x smoother than SMA |
| **Warm-up Period** | PASS | 87 bars (권장: 130) |
| **Regime Transitions** | PASS | High/Low Vol Ratio: 4.27x |

### 5.3 SMA/EMA 대비 성능

```
                 Lag    Smoothness  Efficiency
Kalman           0      30.5        0.8135
SMA(20)          0      242.5       0.3400
EMA(20)          0      460.1       0.2592

Kalman vs SMA Smoothness: 0.13x (8배 더 부드러움)
Kalman vs EMA Efficiency: 3.14x (3배 더 효율적)
```

### 5.4 스트레스 테스트 결과

**Step Function (100→200 점프):**
```
Convergence Time (95%): 22 bars
Uncertainty Spike: 6.5x
Peak Kalman Gain: 0.96
Overshoot: 1.0%
```

**Outlier Injection (10σ spike):**
```
Outlier Magnitude: 70.9
Trend Deviation: 0.42
Attenuation: 99.4%
```

---

## 6. 사용법

### 6.1 기본 사용

```python
from research.features.kalman import AdaptiveKalmanFilter, KalmanConfig

# 설정
config = KalmanConfig(
    r_window=20,
    q_window=20,
    use_typical_price=True
)

# 필터 적용
kf = AdaptiveKalmanFilter(config)
result = kf.filter(df)  # df: OHLC DataFrame

# 결과 사용
trend = result['kf_trend']
signal = result['kf_signal']
uncertainty = result['kf_uncertainty']
```

### 6.2 포지션 사이징

```python
# 데이터 기반 범위 설정
uncertainties = result['kf_uncertainty'].iloc[100:]
min_unc = uncertainties.quantile(0.05)
max_unc = uncertainties.quantile(0.95)

# 현재 포지션 크기 계산
current_unc = result['kf_uncertainty'].iloc[-1]
size_factor = kf.get_position_size_factor(current_unc, min_unc, max_unc)

# 적용
position_size = base_size * size_factor
```

### 6.3 파이프라인 통합

```python
from research.features.kalman import KalmanFeatureGenerator

# Feature Pipeline에서 자동 사용
gen = KalmanFeatureGenerator(r_window=20, q_window=20)
features = gen.generate(df)

# 핵심 피처만 선택
core_features = gen.get_core_features()
# ['kf_deviation', 'kf_velocity', 'kf_signal', 'kf_uncertainty']
```

### 6.4 config.yaml 설정

```yaml
features:
  kalman:
    r_window: 20
    q_window: 20
    use_typical_price: true
    q_scale: 0.1
    r_scale: 1.0
```

---

## 7. 파라미터 가이드

### 7.1 권장 설정

| 파라미터 | 권장값 | 범위 | 설명 |
|----------|--------|------|------|
| `r_window` | 20 | 10-50 | R 추정 윈도우 |
| `q_window` | 20 | 10-50 | Q 추정 윈도우 |
| `q_scale` | 0.1 | 0.01-1.0 | 작을수록 smooth |
| `r_scale` | 1.0 | 0.1-10 | 클수록 smooth |
| `q_momentum_weight` | 2.0 | 1-5 | 모멘텀 민감도 |
| `use_typical_price` | True | - | 노이즈 감소 |

### 7.2 튜닝 가이드

**더 부드러운 추세 원할 때:**
- `r_scale` 증가 (1.0 → 2.0)
- `q_scale` 감소 (0.1 → 0.05)

**더 빠른 반응 원할 때:**
- `r_scale` 감소 (1.0 → 0.5)
- `q_scale` 증가 (0.1 → 0.2)
- `q_momentum_weight` 증가 (2.0 → 3.0)

**변동성 높은 시장:**
- `r_window` 증가 (20 → 30)
- `r_scale` 증가

### 7.3 주의사항

1. **Warm-up Period**: 최소 130 bars 필요 (안정적 추정)
2. **Look-ahead Bias**: 모든 계산이 현재 바까지만 사용
3. **초기값 의존성**: 100x 파라미터 범위에서 0.96+ 상관관계로 수렴

---

## 8. 테스트 실행

```bash
# Unit 테스트 (26개)
python -m pytest tests/test_kalman_filter.py -v

# 기본 검증 (실제 데이터)
python scripts/verify_kalman_filter.py

# 고급 검증 (스트레스 테스트)
python scripts/verify_kalman_advanced.py
```

---

## 9. 참고 자료

- Alpha Architect: Noise-Adaptive Kalman Filter
- Benhamou: Kalman Filter in Finance
- Marcos Lopez de Prado: Advances in Financial Machine Learning

---

## 10. 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2025-01-05 | 초기 구현 및 검증 완료 |
