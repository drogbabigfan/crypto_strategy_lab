# Adaptive Kalman Filter API Reference

빠른 참조를 위한 API 문서.

---

## 1. KalmanConfig

설정 데이터클래스.

```python
from research.features.kalman import KalmanConfig

config = KalmanConfig(
    r_window=20,           # R 추정 윈도우
    q_window=20,           # Q 추정 윈도우
    initial_p=1.0,         # 초기 오차 공분산
    r_scale=1.0,           # R 스케일 팩터
    q_scale=0.1,           # Q 스케일 팩터
    q_momentum_weight=2.0, # 모멘텀 가중치
    use_typical_price=True,# (H+L+C)/3 사용
    min_r=1e-8,            # 최소 R
    min_q=1e-10,           # 최소 Q
)
```

### 파라미터

| 이름 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `r_window` | int | 20 | 측정 잡음(R) 추정 윈도우 |
| `q_window` | int | 20 | 프로세스 잡음(Q) 추정 윈도우 |
| `initial_p` | float | 1.0 | 초기 오차 공분산 |
| `r_scale` | float | 1.0 | R 스케일 팩터 (클수록 smooth) |
| `q_scale` | float | 0.1 | Q 스케일 팩터 (클수록 반응 빠름) |
| `q_momentum_weight` | float | 2.0 | 모멘텀 기반 Q 조정 가중치 |
| `use_typical_price` | bool | True | Typical Price 사용 여부 |
| `min_r` | float | 1e-8 | 최소 R 값 (0 방지) |
| `min_q` | float | 1e-10 | 최소 Q 값 (0 방지) |

---

## 2. AdaptiveKalmanFilter

메인 필터 클래스.

```python
from research.features.kalman import AdaptiveKalmanFilter, KalmanConfig

kf = AdaptiveKalmanFilter(config)
```

### 메서드

#### filter(df) → DataFrame

칼만 필터 적용.

```python
result = kf.filter(df)
```

**Parameters:**
- `df` (DataFrame): OHLC 데이터 (`close` 필수, `high`/`low` 선택)

**Returns:**
- DataFrame: 원본 + Kalman 피처

**출력 컬럼:**

| 컬럼 | 설명 |
|------|------|
| `kf_trend` | 필터링된 추세 가격 |
| `kf_velocity` | 추세 속도 (모멘텀) |
| `kf_deviation` | 가격 - 추세 |
| `kf_deviation_pct` | 편차 백분율 |
| `kf_gain` | 칼만 이득 |
| `kf_uncertainty` | 오차 공분산 P |
| `kf_signal` | 편차 z-score |

#### get_position_size_factor(uncertainty, min_unc, max_unc) → float

불확실성 기반 포지션 크기 계산.

```python
size = kf.get_position_size_factor(
    uncertainty=result['kf_uncertainty'].iloc[-1],
    min_uncertainty=100,
    max_uncertainty=1000
)
```

**Parameters:**
- `uncertainty` (float): 현재 오차 공분산 P
- `min_uncertainty` (float, optional): 최소 불확실성 (기본: initial_p * 0.1)
- `max_uncertainty` (float, optional): 최대 불확실성 (기본: initial_p * 100)

**Returns:**
- float: 포지션 크기 팩터 (0.0 ~ 1.0)

---

## 3. KalmanFeatureGenerator

파이프라인 통합용 래퍼.

```python
from research.features.kalman import KalmanFeatureGenerator

gen = KalmanFeatureGenerator(
    r_window=20,
    q_window=20,
    use_typical_price=True,
    q_scale=0.1,
    r_scale=1.0,
)
```

### 메서드

#### generate(df) → DataFrame

피처 생성.

```python
features = gen.generate(df)
```

#### get_feature_names() → List[str]

생성되는 피처 이름 목록.

```python
names = gen.get_feature_names()
# ['kf_trend', 'kf_velocity', 'kf_deviation',
#  'kf_deviation_pct', 'kf_gain', 'kf_uncertainty', 'kf_signal']
```

#### get_core_features() → List[str]

핵심 피처 (모델 입력용).

```python
core = gen.get_core_features()
# ['kf_deviation', 'kf_velocity', 'kf_signal', 'kf_uncertainty']
```

---

## 4. 편의 함수

### calculate_adaptive_kalman()

한 줄로 칼만 필터 적용.

```python
from research.features.kalman import calculate_adaptive_kalman

result = calculate_adaptive_kalman(
    df,
    r_window=20,
    q_window=20,
    use_typical_price=True,
)
```

---

## 5. 사용 예시

### 5.1 기본 사용

```python
import pandas as pd
from research.features.kalman import AdaptiveKalmanFilter, KalmanConfig

# 데이터 로드
df = pd.read_parquet('data.parquet')

# 필터 적용
config = KalmanConfig(r_window=20, q_window=20)
kf = AdaptiveKalmanFilter(config)
result = kf.filter(df)

# 결과 사용
print(result[['close', 'kf_trend', 'kf_deviation', 'kf_signal']].tail())
```

### 5.2 매매 신호

```python
# 과매수/과매도 신호
result['signal'] = 0
result.loc[result['kf_signal'] < -2, 'signal'] = 1   # Long
result.loc[result['kf_signal'] > 2, 'signal'] = -1   # Short

# 추세 필터
result['trend_up'] = result['kf_velocity'] > 0
```

### 5.3 포지션 사이징

```python
# 불확실성 범위 계산
uncertainties = result['kf_uncertainty'].iloc[130:]  # After warmup
min_unc = uncertainties.quantile(0.05)
max_unc = uncertainties.quantile(0.95)

# 포지션 크기 계산
result['position_size'] = result['kf_uncertainty'].apply(
    lambda u: kf.get_position_size_factor(u, min_unc, max_unc)
)

# 최종 포지션
base_position = 1.0
result['actual_position'] = result['signal'] * result['position_size'] * base_position
```

### 5.4 파이프라인 통합

```python
from research.features.pipeline import FeaturePipeline

# config.yaml
"""
features:
  kalman:
    r_window: 20
    q_window: 20
    use_typical_price: true
    q_scale: 0.1
    r_scale: 1.0
"""

# 파이프라인 실행 시 자동으로 Kalman 피처 생성
pipeline = FeaturePipeline(config)
output = pipeline.run()
```

---

## 6. 파라미터 튜닝 가이드

### 더 부드러운 추세

```python
config = KalmanConfig(
    r_scale=2.0,    # 증가
    q_scale=0.05,   # 감소
)
```

### 더 빠른 반응

```python
config = KalmanConfig(
    r_scale=0.5,           # 감소
    q_scale=0.2,           # 증가
    q_momentum_weight=3.0, # 증가
)
```

### 변동성 높은 시장

```python
config = KalmanConfig(
    r_window=30,    # 증가
    r_scale=2.0,    # 증가
)
```

---

## 7. 주의사항

1. **Warm-up**: 최소 130 bars 데이터 필요
2. **입력 데이터**: `close` 컬럼 필수, Typical Price 사용 시 `high`/`low` 필요
3. **NaN 처리**: 첫 몇 개 바는 NaN (rolling 계산)
4. **메모리**: 전체 데이터를 메모리에 로드

---

## 8. 에러 처리

```python
# 데이터 부족
if len(df) < config.r_window:
    raise ValueError("Insufficient data for Kalman filter")

# 필수 컬럼 체크
required = ['close']
if config.use_typical_price:
    required.extend(['high', 'low'])

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")
```
