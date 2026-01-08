# Feature Engineering 작업 목록

## 현재 상태 분석

### 구현됨 (스켈레톤)
| 파일 | 상태 | 문제점 |
|------|------|--------|
| `l1.py` | 50% | TIB 컬럼명 사용, Dollar Bar와 불일치 |
| `stationarity.py` | 80% | FracDiff, Detrend 구현됨 |
| `stitching.py` | 70% | 기본 로직 있음, close만 처리 |

### 미구현
- Regime Features (Parkinson Vol, Shannon Entropy)
- 파이프라인 메인 엔트리포인트
- Config 연동
- ADF 검증 자동화

---

## 작업 목록 (우선순위순)

### Task 1: L1 피처 Dollar Bar 연동 [높음]

**문제:**
현재 `l1.py`는 TIB 컬럼명(`taker_buy_volume`, `vwap`)을 기대하지만, Dollar Bar는 다른 컬럼을 제공:

```
Dollar Bar 컬럼:
- StartTime, EndTime
- Open, High, Low, Close, Volume
- DollarValue, ThresholdUsed, Duration
- TickCount, BuyDollarVol, SellDollarVol, NetImbalance
```

**작업:**
```python
# 수정 필요 항목:
1. volume_imbalance = NetImbalance / DollarValue  # 이미 계산됨
   또는 = (BuyDollarVol - SellDollarVol) / DollarValue

2. vwap = DollarValue / Volume  # 새로 계산 필요
   vwap_deviation = (Close - vwap) / Close

3. log_duration = log(1 + Duration)  # Duration 컬럼 직접 사용

4. log_tick_count = log(1 + TickCount)  # 새 피처 추가

5. log_dollar_value = log(1 + DollarValue)  # 새 피처 추가
```

**테스트:**
- Dollar Bar Parquet 로드 후 L1 피처 생성 확인
- 결측값 처리 확인

---

### Task 2: Regime Features 구현 [높음]

**Parkinson Volatility:**
```python
def parkinson_volatility(high, low, window=24):
    """
    High-efficiency volatility estimator.
    More efficient than close-to-close volatility.
    """
    log_hl = np.log(high / low)
    return np.sqrt(
        (log_hl ** 2).rolling(window).sum() / (4 * np.log(2) * window)
    )
```

**Shannon Entropy:**
```python
def shannon_entropy(returns, window=24, bins=10):
    """
    Measure of randomness in price movements.
    Low entropy = trending, High entropy = noisy.
    """
    def calc_entropy(x):
        hist, _ = np.histogram(x, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zeros
        return -np.sum(hist * np.log(hist))

    return returns.rolling(window).apply(calc_entropy)
```

**테스트:**
- 합성 데이터로 트렌드/노이즈 감지 확인
- 엣지 케이스 (모든 값 동일) 처리

---

### Task 3: Stitching 다중 피처 지원 [중간]

**현재:**
- `close` 컬럼만 Z-score 계산

**목표:**
- 모든 L1 피처에 대해 연속성 보장
- 월별 파일 간 상태 유지

```python
class AdaptiveStitcher:
    def __init__(self, columns, buffer_size=2000, base_halflife=50):
        self.columns = columns  # ['log_volume', 'volume_imbalance', ...]
        ...

    def process(self, df):
        # 각 컬럼에 대해 Z-score 계산
        for col in self.columns:
            zscore = self._calc_zscore(col, df)
            df[f"{col}_zscore"] = zscore
        return df
```

---

### Task 4: 파이프라인 메인 엔트리포인트 [높음]

**새 파일:** `research/features/pipeline.py`

```python
import yaml
import glob
import pandas as pd
from l1 import L1FeatureGenerator
from stationarity import StationarityEngine
from stitching import AdaptiveStitcher
from regime import RegimeFeatures

def run_feature_pipeline(config_path: str):
    """
    Main entry point for feature engineering pipeline.
    """
    # 1. Config 로드
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 2. Bar 파일 목록 가져오기
    bar_pattern = f"{config['data']['bar_path']}/**/*.parquet"
    bar_files = sorted(glob.glob(bar_pattern))

    # 3. 초기화
    l1_gen = L1FeatureGenerator()
    stat_engine = StationarityEngine()
    stitcher = AdaptiveStitcher(
        columns=config['features']['l1_factors'],
        buffer_size=config['features']['stitching']['buffer_size']
    )
    regime = RegimeFeatures(window=config['features']['regime']['parkinson_window'])

    all_features = []

    # 4. 파일별 처리
    for bar_file in bar_files:
        df = pd.read_parquet(bar_file)

        # L1 Features
        df = l1_gen.generate(df)

        # Regime Features
        df = regime.generate(df)

        # Stationarity
        df['frac_diff_close'] = stat_engine.frac_diff_ffd(
            df['Close'], d=config['features']['stationarity']['frac_diff_d']
        )
        df['detrended_log_price'] = stat_engine.detrend_log_price(
            df['Close'], window=config['features']['stationarity']['detrend_ema_span']
        )

        # Stitching (월간 연속성)
        df = stitcher.process(df)

        all_features.append(df)

    # 5. 합치기 및 저장
    result = pd.concat(all_features, ignore_index=True)

    # Config hash로 파일명 생성
    import hashlib
    config_str = yaml.dump(config['features'])
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    output_path = f"{config['data']['feature_output']}/features_{config_hash}.parquet"
    result.to_parquet(output_path)

    return output_path
```

---

### Task 5: ADF 검증 리포트 [중간]

**새 파일:** `research/features/validation.py`

```python
from statsmodels.tsa.stattools import adfuller

def generate_adf_report(df, columns, threshold=0.05):
    """
    Generate ADF test report for stationarity verification.
    """
    report = []
    for col in columns:
        if col in df.columns:
            result = adfuller(df[col].dropna())
            report.append({
                'column': col,
                'adf_statistic': result[0],
                'p_value': result[1],
                'is_stationary': result[1] < threshold
            })

    return pd.DataFrame(report)
```

---

### Task 6: 테스트 업데이트 [높음]

**수정 필요:**
- `test_l1_features.py`: Dollar Bar 컬럼으로 테스트 데이터 변경
- `test_stitching.py`: 다중 컬럼 지원 테스트 추가

**새 테스트:**
- `test_regime_features.py`: Parkinson, Entropy 테스트
- `test_pipeline.py`: E2E 파이프라인 테스트
- `test_validation.py`: ADF 리포트 테스트

---

## 작업 순서 권장

```
1. [Task 1] L1 피처 Dollar Bar 연동
   └── 테스트 업데이트 포함

2. [Task 2] Regime Features 구현
   └── Parkinson Vol + Shannon Entropy

3. [Task 4] 파이프라인 엔트리포인트
   └── 기본 흐름 완성

4. [Task 3] Stitching 다중 피처 지원
   └── 월간 연속성 검증

5. [Task 5] ADF 검증 리포트
   └── 자동 검증 추가

6. [Task 6] 통합 테스트
   └── E2E 검증
```

---

## 예상 산출물

작업 완료 후:
```
research/features/
├── l1.py              # 수정됨
├── regime.py          # 새 파일
├── stationarity.py    # 유지
├── stitching.py       # 수정됨
├── pipeline.py        # 새 파일
└── validation.py      # 새 파일

tests/
├── test_l1_features.py    # 수정됨
├── test_regime_features.py # 새 파일
├── test_pipeline.py       # 수정됨
└── test_validation.py     # 새 파일

data/features/
└── features_{hash}.parquet  # 출력
```
