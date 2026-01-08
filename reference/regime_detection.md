# Regime Detection System

GMM-HMM 기반 시장 국면(Regime) 판단 시스템

## 1. 개요

### 1.1 목적
시장을 3가지 국면으로 분류하여 각 국면에 최적화된 전략 적용:
- **Bear (0)**: 하락 추세 → Module A (Short Only)
- **Sideways (1)**: 횡보 → Module B (Mean Reversion)
- **Bull (2)**: 상승 추세 → Module A (Long Only)

### 1.2 핵심 구성요소
```
┌─────────────────────────────────────────────────────────────────┐
│                     Regime Detection Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────────┐  │
│  │   Feature    │────▶│   GMM-HMM    │────▶│  State Order    │  │
│  │  Generator   │     │  (3 States)  │     │  (by Return)    │  │
│  └──────────────┘     └──────────────┘     └─────────────────┘  │
│         │                    │                      │            │
│         ▼                    ▼                      ▼            │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────────┐  │
│  │ 12 Features  │     │ Raw States   │     │ 0=Bear          │  │
│  │ (normalized) │     │ (0, 1, 2)    │     │ 1=Sideways      │  │
│  │              │     │              │     │ 2=Bull          │  │
│  └──────────────┘     └──────────────┘     └─────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Feature Engineering

### 2.1 입력 피쳐 (12개)

`RegimeFeatureGenerator` (`python/models/regime/sjm.py`)에서 생성:

| # | Feature | Window | Description | 역할 |
|---|---------|--------|-------------|------|
| 1 | `trend_zscore` | 70 bars (~1.4일) | (Price - MA) / StdDev | 단기 추세 |
| 2 | `trend_zscore_mid` | 300 bars (~6일) | 중기 Z-Score | 중기 추세 |
| 3 | `trend_zscore_long` | 1000 bars (~20일) | 장기 Z-Score | 장기 추세 |
| 4 | `ppo_histogram` | 15/30/15 | PPO - Signal | 모멘텀 가속도 |
| 5 | `signed_return` | 12 bars (~5시간) | Rolling return sum | 단기 민감도 |
| 6 | `rsi` | 35 bars (~17시간) | Relative Strength Index | 과매수/과매도 |
| 7 | `adx` | 80 bars (~1.6일) | Average Directional Index | 추세 강도 |
| 8 | `bar_duration` | - | Dollar bar 생성 시간 (분) | 시장 활동성 |
| 9 | `efficiency_ratio` | 30 bars (~14시간) | Kaufman ER | 방향성/노이즈 |
| 10 | `choppiness` | 35 bars (~17시간) | Choppiness Index | 추세 vs 횡보 |
| 11 | `volatility` | 90 bars (~1.8일) | Rolling std of returns | 변동성 |
| 12 | `ffd` | - | Fractional Differentiation | 정상성 가격 |

### 2.2 피쳐 설계 원칙

1. **Multi-timeframe**: 단기(5-17시간), 일간(1-2일), 주간(6일), 월간(20일)
2. **Trend Indicators**: Z-Score, PPO, ADX로 추세 판단
3. **Regime Indicators**: Efficiency Ratio, Choppiness로 횡보 판단
4. **Look-ahead Bias 방지**: 모든 피쳐에 `lag=1` 적용

### 2.3 피쳐별 Regime 분리 기여도

검증 결과 (Bull - Bear Spread):

| Feature | Spread | 기여도 |
|---------|--------|--------|
| `rsi` | 1.92 | ★★★★★ |
| `trend_zscore` | 1.92 | ★★★★★ |
| `signed_return` | 1.27 | ★★★★ |
| `trend_zscore_mid` | 1.27 | ★★★★ |
| `ppo_histogram` | 0.97 | ★★★ |
| `volatility` | 0.73 | ★★★ |
| `trend_zscore_long` | 0.67 | ★★ |
| `ffd` | 0.56 | ★★ |
| `efficiency_ratio` | 0.45 | ★★ |
| `choppiness` | -0.49 | ★★ (역방향) |
| `bar_duration` | -0.26 | ★ |
| `adx` | 0.21 | ★ |

---

## 3. GMM-HMM Model

### 3.1 모델 구조

```python
GMMHMMRegimeDetector(
    n_states=3,        # Bear, Sideways, Bull
    n_mix=2,           # Gaussian Mixture components per state
    covariance_type='diag'  # Diagonal covariance
)
```

### 3.2 State Ordering

HMM의 원시 상태는 의미 없는 숫자이므로, 평균 수익률로 재정렬:

```python
# 각 State의 평균 수익률 계산
state_means = [(k, returns[regimes == k].mean()) for k in range(3)]

# 수익률 오름차순 정렬
sorted_states = sorted(state_means, key=lambda x: x[1])

# Mapping: 최저 수익률 → 0 (Bear), 최고 수익률 → 2 (Bull)
state_mapping = {old: new for new, (old, _) in enumerate(sorted_states)}
```

### 3.3 State 전환 매트릭스

검증 결과:

```
         To Bear  To Side  To Bull
From Bear   94.6%     5.1%     0.3%
From Side    4.6%    91.5%     3.9%
From Bull    0.3%     5.5%    94.2%
```

- 각 State에서 90% 이상 체류 (안정적)
- Bear ↔ Bull 직접 전환 < 1% (급격한 전환 없음)
- 평균 체류 기간: 15-16 bars (~7시간)

---

## 4. 검증 결과

### 4.1 In-Sample 검증 (전체 데이터)

| Test | 결과 | 상세 |
|------|------|------|
| State 분리 | ✓ PASS | 3개 State 정상 분리 |
| 수익률 정렬 | ✓ PASS | Bear < Sideways < Bull |
| 전환 빈도 | ✓ PASS | 평균 체류 15 bars |
| 시장 특성 | ✓ PASS | Regime별 통계 차이 유의미 |
| 피쳐 기여도 | ✓ PASS | 10/12 피쳐 유의미 |
| 예측력 | ✓ PASS | 미래 수익률 예측 성공 |

**결과: 6/6 통과 (100%)**

### 4.2 Out-of-Sample 검증 (Walk-Forward)

설정:
- Train: 10,000 bars (~200일)
- Test: 2,500 bars (~50일)
- Step: 2,500 bars
- 총 40 Folds

| 지표 | In-Sample | Out-of-Sample | 변화 |
|------|-----------|---------------|------|
| Bear 수익률 | -0.0264% | -0.0197% | - |
| Sideways 수익률 | 0.0107% | 0.0043% | - |
| Bull 수익률 | 0.0303% | 0.0299% | - |
| Bull-Bear Spread | 0.0567% | 0.0496% | -12.5% |
| 정렬 성공률 | 100% | 90.0% | -10% |

**결과: 3/3 기준 통과**
- 수익률 정렬 성공률: 90.0% ✓
- Bull-Bear Spread 양수: 0.0496% ✓
- Spread Positive 비율: 95.0% ✓

### 4.3 Regime 분포

| Regime | 비율 | 평균 수익률 | Volatility |
|--------|------|------------|------------|
| Bear | 34.4% | -0.0264% | 0.4456% |
| Sideways | 38.1% | 0.0107% | 0.3868% |
| Bull | 27.5% | 0.0303% | 0.6510% |

---

## 5. 사용법

### 5.1 Python API

```python
from python.models.regime.hmm import GMMHMMRegimeDetector
from python.models.regime.sjm import RegimeFeatureGenerator

# 1. Feature 생성
feature_gen = RegimeFeatureGenerator()
features = feature_gen.generate_normalized(dollarbar, ffd_series, lag=1)

# 2. HMM 학습
hmm = GMMHMMRegimeDetector(n_states=3, n_mix=2, covariance_type='diag')
hmm.fit(features.values)

# 3. Regime 예측
raw_regimes = hmm.predict(features.values)

# 4. State 재정렬 (수익률 기준)
returns = dollarbar['close'].pct_change().values
state_means = [(k, returns[raw_regimes == k].mean()) for k in range(3)]
sorted_states = sorted(state_means, key=lambda x: x[1])
state_mapping = {old: new for new, (old, _) in enumerate(sorted_states)}
regimes = np.array([state_mapping[s] for s in raw_regimes])
```

### 5.2 Walk-Forward 사용

```python
# Train 구간에서 학습
train_features = features.iloc[:train_end]
hmm.fit(train_features.values)

# Train에서 State 순서 결정
train_regimes = hmm.predict(train_features.values)
# ... state_mapping 계산 ...

# Test 구간에서 예측 (동일한 mapping 사용)
test_features = features.iloc[train_end:test_end]
test_regimes_raw = hmm.predict(test_features.values)
test_regimes = np.array([state_mapping[s] for s in test_regimes_raw])
```

---

## 6. 파일 구조

```
python/models/regime/
├── hmm.py              # GMMHMMRegimeDetector
└── sjm.py              # RegimeFeatureGenerator, StatisticalJumpModel

scripts/
├── verify_regime_detection.py      # In-sample 검증
├── verify_regime_walkforward.py    # Out-of-sample 검증
└── analyze_regime_model.py         # Regime 분석

outputs/
├── regime_verification.png         # 시각화
└── regime_walkforward_results.csv  # Walk-forward 결과
```

---

## 7. 주의사항

### 7.1 Look-ahead Bias
- 모든 피쳐에 `lag=1` 적용 필수
- State ordering은 Train 데이터의 수익률만 사용

### 7.2 State Ordering
- Walk-Forward에서 각 Fold마다 Train 데이터로 재정렬 필요
- Test 데이터의 수익률을 ordering에 사용하면 look-ahead bias 발생

### 7.3 HMM 수렴
- 데이터가 적으면 수렴 경고 발생 가능
- `n_mix=2`, `covariance_type='diag'`로 파라미터 수 제한

---

## 8. 결론

### 8.1 검증 완료
- **In-Sample**: 6/6 테스트 통과
- **Out-of-Sample**: 3/3 기준 통과, 성능 저하 12.5%로 허용 가능

### 8.2 현재 상태
**Regime Detection은 정상 작동 중**

문제는 Regime이 아니라 Primary Signal의 Win Rate (39.7%):
- Module A (Trend): 38.7%
- Module B (Reversion): 38.3%

### 8.3 개선 방향
1. Primary Signal 개선 (Kalman threshold, RSI 조건 완화)
2. Meta-labeling으로 저품질 신호 필터링
3. Module B가 Sideways에서 더 많은 신호 생성하도록 조건 완화

---

## 9. 관련 문서

- [Dual Module Strategy](./dual_module.md) - 전체 전략 파이프라인
- [Regime Theory](./regime_theory.md) - HMM 이론 및 고급 아키텍처 (RegimeNAS, Mamba 등)

---

## 10. 참고 자료

- Nystrup, P., Kolm, P. N., & Lindström, E. (2020). "Greedy online classification of persistent market states using realized intraday volatility features."
- Hidden Markov Models for Regime Detection
- Advances in Financial Machine Learning (Marcos López de Prado)

---

## 11. 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2024-12-27 | 최초 작성, In-sample/Out-of-sample 검증 완료 |
