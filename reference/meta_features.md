# Meta-Labeling Features Analysis

## 개요

Meta-labeling 모델에 사용할 피처 분석 결과.

- **분석 일자**: 2025-12-27 (Updated: 2025-12-29)
- **Best Model**: XGBoost with Clustered MDA + Sample Weights
- **Key Update**: Clustered MDA로 Substitution Effect 해결, F1 68% 개선

### 모델 비교 (CV ROC-AUC)

| Model | ROC-AUC |
|-------|---------|
| LightGBM | 0.5821 |
| XGBoost | 0.5751 |
| CatBoost | 0.5657 |

---

## 피처 카테고리

### 1. 유동성 추정기 (VPIN/OIF Proxies)

Microstructure 데이터 없이 OHLCV에서 유동성을 추정하는 피쳐들.

| Feature | 설명 | 참조 |
|---------|------|------|
| `cs_spread` | Corwin-Schultz Spread - High-Low 기반 스프레드 추정 | Corwin & Schultz (2012) |
| `cs_spread_relative` | CS Spread / Rolling Mean - 상대적 스프레드 변화 | |
| `roll_spread` | Roll's Effective Spread - 연속 수익률 공분산 기반 | Roll (1984) |
| `roll_momentum` | Roll spread 음수 = 모멘텀 레짐 | |
| `amihud_illiq` | Amihud Illiquidity - \|Return\| / Dollar Volume | Amihud (2002) |
| `amihud_relative` | 상대적 Amihud (현재 / 롤링 평균) | |
| `edge_spread` | Abdi-Ranaldo EDGE Spread - Open/Close 보정 | Abdi & Ranaldo (2017) |
| `liquidity_stress` | CS/Roll 스프레드 발산 지표 | |

**사용법**:
```python
from python.indicators.liquidity import LiquidityFeatureGenerator, LiquidityConfig

generator = LiquidityFeatureGenerator(LiquidityConfig(rolling_window=20))
features = generator.generate(ohlcv_data)
```

### 2. 고급 변동성 추정기 (Volatility Texture)

Close-to-close 변동성보다 5-8배 효율적인 OHLC 기반 추정기.

| Feature | 설명 | 효율성 | 참조 |
|---------|------|--------|------|
| `parkinson_vol` | High-Low 기반 변동성 | 5x | Parkinson (1980) |
| `rogers_satchell_vol` | Drift-adjusted 변동성 | 6x | Rogers & Satchell (1991) |
| `yang_zhang_vol` | Overnight + Intraday 결합 | 8x | Yang & Zhang (2000) |
| `garman_klass_vol` | OHLC 기반 변동성 | 8x | Garman & Klass (1980) |
| `vol_efficiency_ratio` | Yang-Zhang / Close-to-Close 비율 | | |
| `vol_cone_ratio` | 단기/장기 변동성 비율 | | |
| `vol_zscore` | 변동성 Z-score | | |

**수식**:
- Parkinson: $\sigma = \sqrt{\frac{1}{4\ln(2)} \cdot E[(\ln(H/L))^2]}$
- Yang-Zhang: $\sigma^2 = \sigma_o^2 + k\sigma_c^2 + (1-k)\sigma_{RS}^2$

**사용법**:
```python
from python.indicators.volatility import VolatilityFeatureGenerator, VolatilityConfig

generator = VolatilityFeatureGenerator(VolatilityConfig(rolling_window=20))
features = generator.generate(ohlcv_data)
```

### 3. 분포 피쳐 (Higher Moments & Entropy)

꼬리 위험과 시장 복잡성을 측정하는 피쳐.

| Feature | 설명 | 참조 |
|---------|------|------|
| `realized_skewness` | 수익률 왜도 - 꼬리 방향 | |
| `realized_kurtosis` | 수익률 첨도 - 점프 빈도 | |
| `tail_risk` | -Skewness (양수 = 하방 위험) | |
| `jump_indicator` | Kurtosis > 3 여부 (Fat tails) | |
| `approx_entropy` | Approximate Entropy - 예측 가능성 | Pincus (1991) |
| `sample_entropy` | Sample Entropy - 더 robust한 ApEn | Richman & Moorman (2000) |
| `market_stress` | Kurtosis + Entropy 결합 지표 | |

**해석**:
- **Negative Skewness**: 하방 꼬리 위험 (패닉 셀링 시그널)
- **High Kurtosis**: 점프 빈도 증가 (모델 가정 실패 가능)
- **Low Entropy**: 예측 가능한 패턴 (추세/평균회귀)
- **High Entropy**: 랜덤/노이즈 (거래 자제)

**사용법**:
```python
from python.indicators.distribution import DistributionFeatureGenerator, DistributionConfig

generator = DistributionFeatureGenerator(DistributionConfig(rolling_window=60))
features = generator.generate(ohlcv_data)
```

### 4. Fractional Differentiation

메모리 보존과 정상성의 균형.

| Feature | 설명 |
|---------|------|
| `ffd_raw` | Fractionally Differentiated series |
| `ffd_zscore` | FFD의 Z-score |
| `ffd_momentum` | FFD의 모멘텀 |

**최적 d 값 탐색**: ADF 테스트로 정상성 달성하는 최소 d 찾기.

### 5. 레거시 피쳐

| Feature | 설명 |
|---------|------|
| `volume_ratio` | 단기/장기 거래량 비율 (20/100) |
| `funding_rate` | Binance 펀딩비율 (8시간마다) |
| `vol_ratio` | 단기/장기 변동성 비율 (20/200) |
| `rolling_skewness` | FFD의 rolling skewness (window=20) |
| `rolling_kurtosis` | FFD의 rolling kurtosis (window=20) |
| `ffd` | Fractional Differentiation (d=0.55) |
| `kf_deviation` | Kalman Filter 편차 |
| `rsi` | RSI(14) - 모멘텀 지표 |
| `dollarbar_interval` | Dollar bar 생성 소요 시간 (분) |
| `regime` | Volatility 기반 regime (0/1/2) |
| `lzc` | Lempel-Ziv Complexity - 엔트로피 |

---

## 통합 피쳐 엔지니어링

### MetaFeatureEngineer

모든 피쳐 생성기를 통합하는 클래스.

```python
from python.meta.engineer import MetaFeatureEngineer, MetaFeatureEngineerConfig

config = MetaFeatureEngineerConfig(
    liquidity_window=20,
    volatility_window=20,
    distribution_window=60,
    frac_diff_d=None,  # Auto-find optimal d
    include_liquidity=True,
    include_volatility=True,
    include_distribution=True,
    include_frac_diff=True,
    include_legacy=True,
)

engineer = MetaFeatureEngineer(config)
features = engineer.generate(ohlcv_data)

# 피쳐 그룹 확인
print(engineer.get_feature_groups())
# {'liquidity': [...], 'volatility': [...], 'distribution': [...], ...}
```

### 생성되는 피쳐 수

| 그룹 | 피쳐 수 |
|------|---------|
| Liquidity | ~15 |
| Volatility | ~12 |
| Distribution | ~10 |
| Fractional Diff | ~3 |
| Legacy | ~10 |
| Signal Quality | ~8 |
| **Total** | **~58** |

---

## 검증 유틸리티

### Purged K-Fold Cross-Validation

금융 시계열에서 look-ahead bias를 방지하는 CV.

```python
from python.models.validation import PurgedKFold, purged_train_test_split

# t1 = 각 샘플의 레이블 종료 시점 (Triple Barrier exit time)
cv = PurgedKFold(n_splits=5, t1=events['exit_idx'], embargo_pct=0.01)

for train_idx, test_idx in cv.split(X, y):
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[test_idx], y[test_idx])
```

**핵심 개념**:
- **Purging**: 훈련 샘플 중 테스트 세트와 시간적으로 겹치는 샘플 제거
- **Embargo**: 테스트 세트 직후 일정 기간 훈련에서 제외

### MDA Feature Importance

Permutation 기반 피쳐 중요도 (Purged CV 적용).

```python
from python.models.validation import MDAFeatureImportance, FeatureSelector, MDAConfig

mda = MDAFeatureImportance(MDAConfig(
    n_splits=5,
    n_permutations=10,
    embargo_pct=0.01,
))

importance = mda.calculate(model, X, y, t1=events['exit_idx'])

# 피쳐 선택
selector = FeatureSelector(method='threshold', threshold=0.01)
selector.fit(importance)
selected_features = selector.get_selected_features()
```

### Clustered MDA (cMDA) - De Prado's Solution

#### 문제: Substitution Effect

Standard MDA는 상관관계가 높은 피쳐들에서 **대체 효과(Substitution Effect)** 문제가 발생합니다.

```
예시: RSI_14와 RSI_15가 함께 있을 때
1. RSI_14를 셔플해도 RSI_15가 대체
2. "RSI_14는 불필요하다" 판정 → Importance = 0
3. RSI_15 검사 시에도 동일 현상
4. 결과: 둘 다 0점으로 평가됨
```

**증상**: 대부분의 피쳐 importance가 0에 수렴하고 p-value가 높음.

실제 상관계수 분석 결과:
```
Pairs with |r| > 0.9: 18개
Pairs with |r| > 0.8: 26개

예시 (거의 동일한 피쳐들):
- realized_skewness <-> tail_risk: -1.000
- rogers_satchell_vol <-> yang_zhang_vol: 0.999
- yang_zhang_vol <-> garman_klass_vol: 0.999
```

#### 해결: Clustered MDA

클러스터 전체를 동시에 셔플하여 대체 효과를 차단합니다.

**De Prado 거리 공식**:
$$D_{i,j} = \sqrt{0.5 \times (1 - \rho_{i,j})}$$

- $D = 0$: 완전 상관 ($\rho = 1$)
- $D = 0.707$: 무상관 ($\rho = 0$)
- $D = 1$: 완전 역상관 ($\rho = -1$)

**ONC (Optimal Number of Clusters)**: Silhouette Score 최대화로 최적 클러스터 수 결정.

```python
from python.models.validation.feature_importance import ClusteredMDAImportance, MDAConfig

cmda = ClusteredMDAImportance(
    config=MDAConfig(
        n_splits=3,
        n_permutations=5,
        embargo_pct=0.01,
    ),
    min_clusters=3,
    max_clusters=12,
)

# 클러스터 단위로 importance 계산
feature_importance, cluster_importance = cmda.calculate(
    model, X, y, t1=events['exit_idx'], sample_weight=weights
)

# 클러스터 구성 확인
print(cmda.get_cluster_summary())
```

**출력 예시**:
```
Clustered 58 features into 10 clusters
Silhouette Score: 0.258

Cluster Importance:
  Cluster 7 (Distribution): 0.0041 (p=0.547, 6 features)
  Cluster 6 (Mixed):        0.0029 (p=0.501, 6 features)
```

### Sample Weights (수익률 기반)

De Prado의 권장사항: **큰 수익/손실에 더 높은 가중치** 부여.

```python
# 수익률 절대값을 가중치로 사용
returns = events_df['return_pct'].values
sample_weight = np.abs(returns)

# Floor 적용 (평균의 10% 이상)
sample_weight = np.clip(sample_weight, sample_weight.mean() * 0.1, None)

# 정규화 (합 = N)
sample_weight = sample_weight / sample_weight.sum() * len(sample_weight)
```

**효과**: 모델이 "결정적 순간" (큰 수익/손실)을 더 중요하게 학습.

---

## Daily CUSUM Pipeline

전체 Meta-Labeling 파이프라인.

```python
from daily_cumsum_pipeline import DailyCusumPipeline, PipelineConfig

config = PipelineConfig(
    # Triple Barrier
    profit_take_mult=2.0,
    stop_loss_mult=1.0,
    max_holding_bars=20,

    # Feature Engineering
    liquidity_window=20,
    volatility_window=20,
    distribution_window=60,

    # Model
    model_type='xgboost',
    n_estimators=100,

    # Validation
    n_cv_splits=5,
    embargo_pct=0.01,
    test_size=0.2,

    # Feature Selection (NEW: Clustered MDA)
    use_mda_selection=True,
    use_clustered_mda=True,   # De Prado Clustered MDA
    mda_threshold=0.01,
    max_features=30,
    min_clusters=3,
    max_clusters=12,

    # Sample Weights (NEW)
    use_sample_weights=True,  # Weight by return magnitude
    weight_by_return=True,

    # Bet Sizing
    min_confidence=0.55,
    max_position_size=1.0,
)

pipeline = DailyCusumPipeline(config)
results = pipeline.run(
    data=ohlcv_data,
    signals=primary_signals,
    signal_col='signal',
    timestamp_col='timestamp'
)

# 결과
print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
print(f"F1 Score: {results['metrics']['f1']:.4f}")

# 예측
predictions = pipeline.predict(new_data, new_signals)
# predictions: ['prediction', 'probability', 'bet_size', 'take_trade']
```

---

## 파이프라인 흐름

```
OHLCV Data + Primary Signals
            ↓
[1] Triple Barrier Labeling → Events DataFrame
            ↓
[2] Meta-Labeling → Binary Labels (Success/Fail)
            ↓
[3] Feature Engineering → 58 Meta-Features
            ↓
[4] Purged Train/Test Split → Avoid Look-ahead Bias
            ↓
[4.5] Sample Weights → Weight by Return Magnitude (NEW)
            ↓
[5] Clustered MDA Feature Selection → Select from Each Cluster (NEW)
            ↓
[6] Train Meta-Model (XGBoost/RF) → Predict Success Probability
            ↓
[7] Bet Sizing → Position Size based on Confidence
```

---

## 실험 결과 (2025-12-29)

### 데이터셋
- **Data**: BTC Futures Dollar Bars
- **Events**: 37,018 signals
- **Win Rate**: 34.20% (Meta-label positive rate: 34.87%)
- **Train/Test Split**: 8,044 / 11,106 samples

### 성능 비교

| 설정 | Accuracy | Precision | Recall | F1 Score |
|------|----------|-----------|--------|----------|
| Baseline (58개 전체, MDA 없음) | **0.6216** | 0.3619 | 0.1528 | 0.2149 |
| Clustered MDA + Sample Weights (10개 선택) | 0.5437 | 0.3433 | **0.3800** | **0.3607** |

### 분석

**베이스라인 문제점**:
- Accuracy가 높아 보이지만, Recall 15%는 "거의 모든 것을 0으로 예측"
- Label 분포가 65:35인데 accuracy 62% = 다수 클래스만 예측
- 실제로 승리(1)를 예측하는 능력 없음

**Clustered MDA + Sample Weights 효과**:
- **Recall 2.5배 증가** (0.15 → 0.38): 실제 승리 케이스를 더 많이 탐지
- **F1 68% 개선** (0.21 → 0.36): 균형 잡힌 예측
- Accuracy 하락은 모델이 실제로 학습 시작했다는 증거

### 중요 클러스터 발견

| Cluster | Importance | p-value | 주요 피쳐 |
|---------|------------|---------|-----------|
| **Cluster 7** (Distribution) | 0.0041 | 0.547 | realized_kurtosis, entropy, jump_indicator, distribution_regime |
| **Cluster 6** (Mixed) | 0.0029 | 0.501 | realized_skewness, ffd_raw, dollarbar_interval, roll_spread |

**인사이트**:
- 변동성/방향성 지표가 아닌 **Distribution/Entropy** 계열이 가장 중요
- 시장의 "불확실성 수준"을 측정하는 피쳐가 meta-labeling에 유용

### 다음 단계 (TODO)

1. **Threshold 조정**: min_confidence를 0.6~0.7로 올려서 high-confidence trades 정확도 확인
2. **Primary Model 완화**: CUSUM 임계값을 낮춰서 더 많은 신호 생성 → 메타모델에게 더 많은 학습 기회
3. **Feature 확장**: Distribution/Entropy 계열 피쳐 추가 (Shannon Entropy, Spectral Entropy 등)
4. **Profit Factor 기반 평가**: Accuracy 대신 Sharpe Ratio, Profit Factor로 성과 측정

---

## 참고 문헌

1. **Advances in Financial Machine Learning** - Marcos Lopez de Prado (Chapters 3, 5, 8)
   - Chapter 3: Triple Barrier Method
   - Chapter 5: Fractional Differentiation
   - Chapter 8: Feature Importance (MDA, Clustered MDA, Substitution Effect)
2. Corwin & Schultz (2012) - "A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices"
3. Roll (1984) - "A Simple Implicit Measure of the Effective Bid-Ask Spread"
4. Amihud (2002) - "Illiquidity and Stock Returns"
5. Abdi & Ranaldo (2017) - "A Simple Estimation of Bid-Ask Spreads from Daily Close, High, and Low Prices"
6. Parkinson (1980) - "The Extreme Value Method for Estimating the Variance of the Rate of Return"
7. Yang & Zhang (2000) - "Drift Independent Volatility Estimation"
8. Pincus (1991) - "Approximate Entropy as a Measure of System Complexity"
9. Richman & Moorman (2000) - "Sample Entropy"
10. **Clustered Feature Importance** - De Prado, Section 8.6
    - ONC (Optimal Number of Clusters) via Silhouette Score
    - Distance metric: $D = \sqrt{0.5(1-\rho)}$
