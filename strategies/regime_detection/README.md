# Regime Detection Module

HMM(Hidden Markov Model)과 Hurst 지수를 활용한 시장 국면 탐지 모듈입니다.

## 개요

금융 시장은 단일한 상태가 아니라 Bull(상승), Bear(하락), Sideways(횡보) 등 다양한 국면(Regime)을 오가며 진화합니다. 이 모듈은 시장의 숨겨진 상태를 확률적으로 추론하고, 각 국면에 최적화된 트레이딩 전략을 선택하는 프레임워크를 제공합니다.

## 구성 요소

```
strategies/regime_detection/
├── __init__.py          # 모듈 exports
├── hurst.py             # Hurst 지수 계산기
├── hmm.py               # HMM 기반 regime detection
├── labeling.py          # 후처리 라벨링 (ZigZag, Trend Scanning)
├── hybrid.py            # HMM + Hurst 하이브리드 모델
├── strategy.py          # Regime 기반 트레이딩 전략
├── walk_forward.py      # Walk-Forward 분석
└── README.md            # 이 문서
```

## 핵심 개념

### 1. Hurst 지수 (Hurst Exponent)

시계열의 장기 기억(Long Memory)과 지속성을 측정합니다.

| Hurst 값 | 해석 | 추천 전략 |
|----------|------|-----------|
| H > 0.55 | Trending (추세 지속) | 모멘텀 전략 |
| H ≈ 0.50 | Random Walk | 예측 어려움 |
| H < 0.45 | Mean-Reverting (평균 회귀) | 역추세 전략 |

```python
from strategies.regime_detection import HurstCalculator

calculator = HurstCalculator()

# DFA (Detrended Fluctuation Analysis) - 권장
result = calculator.calculate_dfa(returns)
print(f"Hurst: {result.hurst:.4f}")
print(f"Regime: {result.regime.value}")

# Rolling Hurst (실시간 분석용)
rolling = calculator.calculate_rolling(returns_series, window=100)
```

### 2. HMM (Hidden Markov Model)

관측 가능한 데이터(수익률, 변동성) 뒤에 숨겨진 시장 상태를 확률적으로 추론합니다.

```python
from strategies.regime_detection import HMMRegimeDetector, HMMConfig

config = HMMConfig(
    n_states=3,           # Bull, Bear, Sideways
    n_iter=100,           # Baum-Welch 반복
    vol_window=20,        # 변동성 계산 윈도우
)

detector = HMMRegimeDetector(config)
detector.fit(train_data)

# Viterbi decoding (사후 분석용)
regimes = detector.predict(test_data)

# Forward algorithm (실시간 필터링용)
filtered = detector.filter(test_data)

# 현재 상태
current = detector.get_current_regime(recent_data)
print(f"Regime: {current.regime.value}")
print(f"Confidence: {current.confidence:.2%}")
```

### 3. 라벨링 방법

HMM 학습 및 검증을 위한 후처리 라벨링 기법입니다.

#### ZigZag 라벨링
가격의 주요 방향만 추출하여 추세 구간을 정의합니다.

```python
from strategies.regime_detection import ZigZagLabeler, ZigZagConfig

config = ZigZagConfig(
    threshold=0.05,       # 5% 반전 기준
    use_atr=True,         # ATR 기반 동적 임계값
    atr_multiplier=2.0,
)

labeler = ZigZagLabeler(config)
labels = labeler.label(data)
pivots = labeler.get_pivots()
```

#### Trend Scanning
통계적 유의성(t-statistic)을 기반으로 추세를 정의합니다.

```python
from strategies.regime_detection import TrendScanningLabeler, TrendScanConfig

config = TrendScanConfig(
    min_window=20,
    max_window=100,
    t_threshold=2.0,      # t-stat 임계값
)

labeler = TrendScanningLabeler(config)
labels = labeler.label(data)
```

### 4. 하이브리드 모델

HMM과 Hurst를 결합하여 전략 추천을 제공합니다.

```python
from strategies.regime_detection import HybridRegimeDetector, HybridConfig

config = HybridConfig(
    hmm_n_states=3,
    hurst_window=100,
    trending_threshold=0.55,
    mean_revert_threshold=0.45,
    min_confidence=0.6,
    vol_scaling=True,
)

hybrid = HybridRegimeDetector(config)
hybrid.fit(train_data)

# 현재 상태 및 전략 추천
state = hybrid.get_current_state(data)
print(f"Market Regime: {state.market_regime.value}")
print(f"Hurst: {state.hurst_value:.3f} ({state.hurst_regime.value})")
print(f"Strategy: {state.strategy_mode.value}")
print(f"Position Scale: {state.position_scale:.2f}")
```

#### 전략 매트릭스

|  | Trending (H>0.55) | Mean-Rev (H<0.45) | Random (H≈0.5) |
|--|-------------------|-------------------|----------------|
| **Bull** | MOMENTUM_LONG | MEAN_REVERSION | CAUTIOUS |
| **Bear** | MOMENTUM_SHORT | MEAN_REVERSION | DEFENSIVE |
| **Sideways** | CAUTIOUS | MEAN_REVERSION | NEUTRAL |

### 5. 트레이딩 전략

Hurst 기반 regime-adaptive 전략입니다.

```python
from strategies.regime_detection import RegimeStrategy, RegimeStrategyConfig

config = RegimeStrategyConfig(
    hurst_window=100,
    momentum_entry_zscore=1.5,
    mean_revert_entry_zscore=2.0,
    stop_loss_atr_mult=2.0,
    take_profit_atr_mult=3.0,
)

strategy = RegimeStrategy(config)
signals = strategy.generate_signals(data)

# 백테스트
from strategies.regime_detection import RegimeBacktester

backtester = RegimeBacktester(initial_capital=100000)
results = backtester.run(signals)
backtester.print_summary()
```

## Walk-Forward 분석

과적합을 방지하기 위한 Walk-Forward 분석을 지원합니다.

```bash
# 실행
python strategies/regime_detection/walk_forward.py \
    --start 2020-01 \
    --end 2024-12 \
    --train-months 12 \
    --test-months 3

# 옵션
#   --anchored: 확장 윈도우 (기본값)
#   --rolling: 롤링 윈도우
```

### WFA 구성

```python
from strategies.regime_detection.walk_forward import WFAConfig

config = WFAConfig(
    train_months=12,
    test_months=3,
    anchored=True,

    # 최적화할 파라미터
    hurst_windows=[50, 100, 150],
    momentum_entry_zscores=[1.0, 1.5, 2.0],
    mean_revert_entry_zscores=[1.5, 2.0, 2.5],
    stop_loss_atr_mults=[1.5, 2.0, 2.5],

    optimization_metric='sharpe',
)
```

## 주요 파라미터 가이드

### Hurst 계산
| 파라미터 | 권장값 | 설명 |
|----------|--------|------|
| `window` | 100-200 | 롤링 윈도우 크기 |
| `method` | 'dfa' | DFA가 더 robust |
| `trending_threshold` | 0.55 | 추세 지속 판단 |
| `mean_revert_threshold` | 0.45 | 평균 회귀 판단 |

### HMM
| 파라미터 | 권장값 | 설명 |
|----------|--------|------|
| `n_states` | 2-3 | 상태 수 (2=Bull/Bear, 3=+Sideways) |
| `n_iter` | 100 | Baum-Welch 반복 횟수 |
| `vol_window` | 20 | 변동성 계산 윈도우 |

### 트레이딩 전략
| 파라미터 | 권장값 | 설명 |
|----------|--------|------|
| `momentum_entry_zscore` | 1.0-2.0 | 모멘텀 진입 Z-score |
| `mean_revert_entry_zscore` | 1.5-2.5 | 평균회귀 진입 Z-score |
| `stop_loss_atr_mult` | 1.5-2.5 | ATR 기반 손절 배수 |

## 의존성

```bash
pip install numpy pandas scipy
pip install hmmlearn  # HMM용 (선택)
```

## 참고 문헌

- Lopez de Prado, "Advances in Financial Machine Learning"
- Hamilton, "A New Approach to the Economic Analysis of Nonstationary Time Series"
- Mandelbrot, "The Fractal Geometry of Nature"

자세한 연구 내용은 `docs/regime_detection/` 참조.
