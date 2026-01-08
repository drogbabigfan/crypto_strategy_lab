# LPPLS (Log-Periodic Power Law Singularity) Trading Strategy

LPPLS 모델을 활용한 금융 버블 탐지 및 트레이딩 전략 구현.

## Overview

LPPLS(Log-Periodic Power Law Singularity) 모델은 Johansen-Ledoit-Sornette(JLS) 모델의 핵심 구현체로,
금융 시장의 버블과 붕괴를 예측하기 위한 통계물리학 기반 접근법입니다.

### 핵심 개념

- **임계 현상(Critical Phenomena)**: 버블 붕괴는 외부 충격이 아닌 시장 내부의 불안정성이 임계점을 넘어설 때 발생
- **로그 주기적 진동(Log-Periodic Oscillation)**: 붕괴 시점에 다가갈수록 진동 주기가 로그 스케일로 감소
- **이산적 척도 불변성(Discrete Scale Invariance)**: 다양한 시간 지평의 투자자 집단이 위계적으로 상호작용

## LPPLS 방정식

```
E[ln p(t)] = A + B(tc - t)^m + C(tc - t)^m * cos(ω * ln(tc - t) - φ)
```

### 파라미터

| 파라미터 | 의미 | 유효 범위 |
|---------|------|----------|
| tc | 임계 시간 (예상 붕괴 시점) | t_current < tc < t_current + 60일 |
| m | 멱지수 (가속도 결정) | 0.1 < m < 0.9 (권장: 0.3~0.7) |
| ω | 각진동수 (진동 빈도) | 6 < ω < 13 |
| A | tc 시점의 예상 로그 가격 | A > 0 |
| B | 멱법칙 성장 진폭 | B < 0 (양의 버블) |
| C | 진동 크기 | C ≠ 0 |
| φ | 위상 | 0 ≤ φ < 2π |

### 필터링 조건

1. **감쇠 조건**: D = m|B| / (ω|C|) ≥ 1
2. **진동 횟수**: (ω/2π) * ln((tc-t1)/(tc-t2)) ≥ 2.5

## 모듈 구조

```
strategies/lppls/
├── __init__.py       # 패키지 exports
├── lppls.py          # 핵심 LPPLS 모델 (Filimonov-Sornette 최적화)
├── filter.py         # 파라미터 필터링
├── indicator.py      # Confidence Indicator, CLIP
├── strategy.py       # 트레이딩 전략
├── walk_forward.py   # Walk-forward 최적화
├── test_lppls.py     # 테스트
└── README.md         # 문서
```

## Quick Start

### 1. 단일 LPPLS 피팅

```python
from strategies.lppls import fit_lppls
import numpy as np

prices = np.array([...])  # 가격 데이터
result = fit_lppls(prices)

if result:
    print(f"tc = {result.params.tc:.1f}")
    print(f"m = {result.params.m:.3f}")
    print(f"ω = {result.params.omega:.2f}")
    print(f"R² = {result.r_squared:.3f}")
```

### 2. Confidence Indicator

```python
from strategies.lppls import LPPLSIndicator, WindowConfig
import numpy as np

prices = np.array([...])
log_prices = np.log(prices)

# 다중 윈도우 분석
indicator = LPPLSIndicator(
    window_config=WindowConfig(
        min_window=120,
        max_window=500,
        step=10,
    )
)
result = indicator.calculate(log_prices)

print(f"Confidence: {result.confidence:.2%}")
print(f"tc mean: {result.tc_mean:.1f}")
print(f"tc std: {result.tc_std:.1f}")
```

### 3. 백테스트

```python
from strategies.lppls import LPPLSStrategy, LPPLSStrategyConfig, load_btc_data

# 데이터 로드
df = load_btc_data()
prices = df['close'].values

# 전략 설정
config = LPPLSStrategyConfig(
    min_window=60,
    max_window=150,
    confidence_entry=0.6,
    confidence_exit=0.2,
    profit_target_pct=0.20,
    stop_loss_pct=0.12,
)

# 백테스트 실행
strategy = LPPLSStrategy(config)
result = strategy.backtest(prices)

print(f"Total PnL: {result.total_pnl_pct*100:.2f}%")
print(f"Win Rate: {result.win_rate*100:.1f}%")
print(f"Sharpe: {result.sharpe_ratio:.2f}")
```

### 4. Walk-Forward 최적화

```python
from strategies.lppls import run_walk_forward_backtest, WalkForwardConfig

config = WalkForwardConfig(
    train_window=1500,
    test_window=500,
    step_size=500,
    param_grid={
        'confidence_entry': [0.4, 0.5, 0.6],
        'confidence_exit': [0.15, 0.2],
        'profit_target_pct': [0.15, 0.20],
        'stop_loss_pct': [0.10, 0.15],
    },
)

result = run_walk_forward_backtest(
    start_date='2020-01-01',
    end_date='2024-12-31',
    config=config,
)
```

## 트레이딩 신호

| Confidence | 신호 | 해석 |
|------------|------|------|
| ≥ 0.8 + tc 수렴 | STRONG | 강한 버블 신호, 숏 포지션 고려 |
| ≥ 0.8 | MODERATE | 버블 신호, 주의 깊게 관찰 |
| ≥ 0.5 | WEAK | 일부 버블 징후 |
| < 0.5 | NONE | 명확한 버블 패턴 없음 |

### 전략 A: 버블 붕괴 베팅 (Contrarian Short)

**진입 조건:**
- Confidence ≥ 0.8
- tc 표준편차 ≤ 5일 (수렴)
- 기술적 지표 과매수 (선택사항)

**청산 조건:**
- 목표 수익: 고점 대비 20% 하락
- 손절: 10-12% 손실
- Confidence < 0.1 (신호 소멸)

### 전략 B: 저점 매수 (Anti-bubble)

- B > 0 조건의 "음의 버블" 탐지
- 급락 후 반등 타이밍 포착

## 주요 클래스

### LPPLSModel

```python
class LPPLSModel:
    """핵심 LPPLS 모델."""

    def fit(
        self,
        log_prices: NDArray,
        method: str = 'nelder-mead',
        n_starts: int = 10,
    ) -> Optional[LPPLSResult]:
        """모델 피팅."""

    def predict(
        self,
        params: LPPLSParams,
        t: NDArray,
    ) -> NDArray:
        """가격 예측."""
```

### LPPLSIndicator

```python
class LPPLSIndicator:
    """다중 윈도우 Confidence Indicator."""

    def calculate(
        self,
        log_prices: NDArray,
        n_jobs: int = 1,
    ) -> ConfidenceResult:
        """Confidence 지표 계산."""
```

### LPPLSStrategy

```python
class LPPLSStrategy:
    """LPPLS 기반 트레이딩 전략."""

    def backtest(
        self,
        prices: NDArray,
        verbose: bool = True,
    ) -> BacktestResult:
        """백테스트 실행."""
```

## 설정 파라미터

### LPPLSStrategyConfig

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| min_window | 60 | LPPLS 최소 윈도우 (일) |
| max_window | 180 | LPPLS 최대 윈도우 (일) |
| window_step | 10 | 다중 윈도우 간격 |
| recalc_interval | 5 | 재계산 주기 (바) |
| confidence_entry | 0.6 | 진입 Confidence 임계값 |
| confidence_exit | 0.2 | 청산 Confidence 임계값 |
| profit_target_pct | 0.15 | 목표 수익률 |
| stop_loss_pct | 0.10 | 손절 수익률 |

## 성능 고려사항

### 계산 비용

LPPLS 피팅은 계산 집약적입니다:

- **단일 피팅**: ~100ms (Nelder-Mead, 10 starts)
- **다중 윈도우 (100개)**: ~10초
- **전체 백테스트 (5년)**: 수십 분

### 최적화 팁

1. **recalc_interval 증가**: 재계산 빈도 감소 (예: 20-30)
2. **윈도우 수 감소**: window_step 증가 (예: 20-30)
3. **병렬 처리**: n_jobs > 1 사용

## 한계점 및 주의사항

1. **신호 희소성**: LPPLS 버블 신호는 드물게 발생 (연간 수 회)
2. **거짓 양성**: 버블 신호 후 추가 상승 가능 (타이밍 어려움)
3. **파라미터 민감도**: 결과가 윈도우 크기, 필터 설정에 민감
4. **계산 비용**: 실시간 적용 시 지연 고려 필요

### 개선 방향

1. **Deep LPPLS**: 신경망 기반 파라미터 추정 (P-LNN)
2. **하이브리드 모델**: LSTM/Transformer와 결합
3. **감성 분석 결합**: 뉴스/소셜 미디어 데이터 활용

## Deep LPPLS (신경망 기반)

### 개요

신경망을 사용한 빠른 파라미터 추정으로, 전통적 최적화 대비 **2700배 빠른** 추론 속도를 달성합니다.

### 속도 비교

| 방법 | 시간/샘플 | 속도 향상 |
|------|----------|----------|
| Traditional (Nelder-Mead) | 52.4 ms | 1x |
| Deep LPPLS (단일) | 0.34 ms | 154x |
| Deep LPPLS (배치) | 0.019 ms | **2770x** |

### 사용법

```python
from strategies.lppls import DeepLPPLSPredictor, DeepLPPLSConfig

# Create and train predictor
config = DeepLPPLSConfig(input_length=252, epochs=50)
predictor = DeepLPPLSPredictor(config=config, use_conv=True)
predictor.train(n_samples=30000, epochs=50)

# Fast prediction
log_prices = np.log(prices[-252:])
pred = predictor.predict(log_prices)
print(f"tc={pred['tc']:.1f}, m={pred['m']:.3f}, omega={pred['omega']:.2f}")

# Batch prediction (최대 속도)
results = predictor.predict_batch([series1, series2, series3])
```

### 아키텍처

- **PLNNWithConv**: 1D CNN + MLP 하이브리드
  - Conv layers: 로컬 패턴 추출
  - MLP head: 파라미터 예측
  - Output constraints: Sigmoid로 유효 범위 보장

### 활용 시나리오

1. **대규모 스크리닝**: 수천 개 종목을 실시간 모니터링
2. **하이브리드 파이프라인**: Deep LPPLS로 후보 선별 → 전통적 방법으로 확인
3. **백테스트 가속**: 대량의 히스토리컬 분석

### 주의사항

- 합성 데이터로 학습하므로 실제 시장과 분포 차이 존재
- 실전 사용 전 해당 자산의 데이터로 파인튜닝 권장
- 신호의 "존재 여부" 판단보다는 "스크리닝" 목적에 적합

## 파라미터 최적화 결과

### 최적화된 파라미터 (BTC 2020-2023)

```python
from strategies.lppls import OPTIMIZED_PARAMS

# 파라미터 내용
OPTIMIZED_PARAMS = {
    'tc_max_days': 70,
    'tc_min_days': 5,
    'm_min': 0.2,
    'm_max': 0.7,
    'omega_min': 5.0,
    'omega_max': 15.0,
    'signal_lookback': 5,
    'entry_threshold': 0.5,
    'profit_target_pct': 0.10,
    'stop_loss_pct': 0.06,
    'max_hold_days': 20,
}
```

### 성능 비교

| 기간 | LPPLS 전략 | Buy & Hold |
|------|-----------|------------|
| COVID 폭락 (2020.02-04) | **+14.27%** | -10.27% |
| In-Sample (2020-2023) | +5.06% | +410.21% |
| Out-of-Sample (2024) | 0% (거래 없음) | +113.81% |

### 핵심 인사이트

1. **LPPLS는 버블 탐지 도구**: 추세 추종 전략이 아님
2. **폭락 구간에서 우수**: COVID 폭락 시 Buy & Hold 대비 +24.5% 초과 수익
3. **보수적 진입**: 2024년 상승장에서 신호 없어 거래 안 함 (손실 회피)
4. **낮은 최대 낙폭**: 16.6% (vs 일반적 40%+)

### 권장 사용법

- **단독 전략 X**: 추세 추종 전략과 **병행** 사용
- **포지션 오버레이**: 버블 신호 시 기존 롱 포지션 축소
- **위험 관리**: 신호 강도에 따른 포지션 크기 조절

### 평가 실행

```bash
python -m strategies.lppls.evaluate_strategy
```

## 참고 문헌

1. Johansen, A., Ledoit, O., & Sornette, D. (2000). "Crashes as Critical Points"
2. Filimonov, V., & Sornette, D. (2013). "A Stable and Robust Calibration Scheme"
3. Sornette, D. (2003). "Why Stock Markets Crash"
4. Nielsen, A., Sornette, D., & Raissi, M. (2024). "Deep LPPLS"

## 테스트 실행

```bash
# 기본 테스트
python -m pytest strategies/lppls/test_lppls.py -v

# Deep LPPLS 속도 비교
python -c "from strategies.lppls import compare_speed; compare_speed()"
```
