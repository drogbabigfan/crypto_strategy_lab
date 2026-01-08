# Label Optimizer Implementation (v4.6)

## 개요

Blueprint v4.6 Section 6 (Stage 2: Label Optimizer) 구현 완료.

**목적**: Triple Barrier 파라미터 최적화 + Fee Trap 필터링 + Plateau Search

---

## 파일 구조

```
research/label_optimizer/
├── __init__.py          # Package exports
├── cost_model.py        # Square Root Cost Model
├── tbm.py               # Triple Barrier Method
├── scoring.py           # Scoring metrics
├── plateau.py           # Plateau Search
└── optimizer.py         # Main orchestrator

tests/
└── test_label_optimizer.py  # 48 test cases (basic + edge cases)
```

---

## 모듈별 상세

### 1. cost_model.py - Square Root Cost Model

**클래스**: `SquareRootCostModel`

```python
@dataclass
class SquareRootCostModel:
    base_fee: float = 0.001        # 0.10% 거래소 수수료
    base_slippage: float = 0.0001  # 0.01% 최소 슬리피지
    impact_coeff: float = 0.1      # η (impact coefficient)
    slippage_cap: float = 0.005    # 0.50% 최대 슬리피지
    min_profit_buffer: float = 1.5 # 최소 수익 배수
```

**핵심 메서드**:

| 메서드 | 설명 | 수식 |
|--------|------|------|
| `get_slippage()` | 슬리피지 계산 | `base + η × σ × √(Q/V)` |
| `get_total_cost()` | Round-trip 비용 | `2 × (fee + slippage)` |
| `is_viable()` | Fee Trap Guard | `PT × σ > cost × buffer` |
| `get_breakeven_pt()` | 손익분기 PT 계산 | `(cost × buffer) / σ` |

**테스트 케이스** (10개):
- `test_base_slippage`: 최소 슬리피지 확인
- `test_slippage_increases_with_volatility`: 변동성 증가 → 슬리피지 증가
- `test_slippage_cap`: 슬리피지 상한선 적용
- `test_is_viable_with_low_pt`: 낮은 PT → 거래 불가
- `test_is_viable_with_high_pt`: 높은 PT → 거래 가능
- `test_total_cost_is_round_trip`: Round-trip = 2 × One-way
- `test_get_breakeven_pt`: 손익분기점 PT 계산
- `test_get_breakeven_pt_zero_sigma`: σ=0일 때 ∞ 반환
- `test_slippage_with_zero_volume`: 거래량 0일 때 cap 적용
- `test_slippage_with_negative_volume`: 음수 거래량 처리

---

### 2. tbm.py - Triple Barrier Method

**클래스**: `TBMConfig`, `TripleBarrierLabeler`, `LabelResult`

```python
@dataclass
class TBMConfig:
    sl_mult: float      # Stop Loss σ 배수
    pt_mult: float      # Profit Target σ 배수
    vertical_bars: int  # 최대 보유 기간 (바 수)
```

**라벨링 로직**:
```
+1: Profit Target 먼저 도달 (Long)
-1: Stop Loss 먼저 도달 (Short)
 0: Vertical Barrier 도달 (Timeout)
```

**Fee Trap Guard 적용**:
- Per-bar 단위로 `cost_model.is_viable()` 체크
- 불가능한 샘플은 `n_filtered`로 집계 후 제외

**테스트 케이스** (10개):
- `test_basic_labeling`: 기본 라벨링 동작
- `test_label_distribution`: 라벨 분포 다양성
- `test_fee_trap_filtering`: Fee Trap 필터링 동작
- `test_no_filtering_without_cost_model`: Cost Model 없으면 필터링 안함
- `test_missing_columns_raises_error`: 필수 컬럼 누락 시 에러
- `test_nan_volatility_skipped`: NaN volatility 행 스킵
- `test_zero_volatility_skipped`: 0 volatility 행 스킵
- `test_same_bar_pt_sl_hit`: 동일 바에서 PT/SL 동시 도달 처리
- `test_get_label_distribution`: 라벨 분포 helper 테스트
- `test_returns_calculated_correctly`: 수익률 계산 정확성

---

### 3. scoring.py - Scoring Metrics

**함수**:

| 함수 | 설명 | 가중치 (기본) |
|------|------|---------------|
| `compute_entropy()` | Shannon Entropy (클래스 균형) | 0.2 |
| `compute_mi()` | Mutual Information (Feature-Label 의존성) | 0.4 |
| `compute_rank_ic()` | Rank IC (Spearman 상관) | 0.4 |
| `compute_composite_score()` | 정규화 후 가중합 | - |

**Rank IC 특징**:
- Ordinal Label {-1, 0, +1}에 적합
- Outlier에 강건 (Rank 기반)
- 절대값 사용 (방향 무관)

**테스트 케이스** (13개):
- `test_entropy_balanced`: 균형 라벨 → 높은 엔트로피
- `test_entropy_imbalanced`: 불균형 라벨 → 낮은 엔트로피
- `test_rank_ic_with_correlation`: 상관 Feature → 높은 IC
- `test_rank_ic_no_correlation`: 무상관 Feature → 낮은 IC
- `test_scoring_metrics_compute`: ScoringMetrics 통합 테스트
- `test_mi_downsampling`: MI 다운샘플링 동작
- `test_entropy_empty_labels`: 빈 라벨 → 0 반환
- `test_entropy_single_class`: 단일 클래스 → 0 엔트로피
- `test_mi_empty_data`: 빈 데이터 → 0 반환
- `test_mi_with_nan_features`: NaN Feature 처리
- `test_rank_ic_empty_data`: 빈 데이터 → 0 반환
- `test_rank_ic_constant_feature`: 상수 Feature 스킵
- `test_composite_score_normalization`: 정규화 범위 테스트

---

### 4. plateau.py - Plateau Search

**알고리즘** (Connected Components):

```
1. 상위 N% 점수 → Binary Mask 생성
2. scipy.ndimage.label() → Connected Components 탐색
3. 가장 큰 Component 선택
4. Center of Mass 계산 → 가장 안정적인 파라미터
```

**함수**:

| 함수 | 설명 |
|------|------|
| `find_plateau_center()` | Plateau 중심점 탐색 |
| `grid_idx_to_params()` | Grid 인덱스 → 실제 파라미터 변환 |
| `visualize_plateau()` | ASCII 시각화 (2D/3D) |

**Fallback 전략**:
1. threshold_percentile (90%) 시도
2. 실패 시 fallback_percentile (80%)
3. 그래도 실패 시 argmax (Peak)

**테스트 케이스** (8개):
- `test_find_single_peak`: 단일 Peak 탐색
- `test_find_plateau`: Plateau 영역에서 중심 선택
- `test_grid_idx_to_params`: 인덱스→파라미터 변환
- `test_fallback_to_argmax`: 컴포넌트 없을 때 argmax 폴백
- `test_min_component_size_filter`: 최소 컴포넌트 크기 필터
- `test_visualize_plateau_2d`: 2D ASCII 시각화
- `test_visualize_plateau_3d`: 3D 슬라이스 시각화
- `test_grid_idx_to_params_length_mismatch`: 인덱스 길이 불일치 에러

---

### 5. optimizer.py - Main Orchestrator

**클래스**: `GridConfig`, `LabelOptimizer`, `OptimizationResult`

```python
@dataclass
class GridConfig:
    sl_range: list[float]   # [1.0, 2.0, 3.0]
    pt_range: list[float]   # [1.5, 2.0, 2.5, 3.0, 3.5]
    time_range: list[int]   # [50, 100, 200, 500, 1000]

    # Scoring weights
    entropy_weight: float = 0.2
    mi_weight: float = 0.4
    rank_ic_weight: float = 0.4

    # Thresholds
    min_samples: int = 1000
    min_entropy: float = 1.0
```

**파이프라인**:

```
1. Grid Search: 모든 (SL, PT, Time) 조합 평가
   ↓
2. Scoring: 각 조합에 대해 Entropy, MI, Rank IC 계산
   ↓
3. Normalization: Grid 전체에서 Min-Max 정규화
   ↓
4. Plateau Search: Connected Components로 안정 영역 탐색
   ↓
5. Final Labeling: Best params로 최종 라벨 생성
```

**테스트 케이스** (6개):
- `test_basic_optimization`: 기본 최적화 파이프라인
- `test_optimizer_summary`: 결과 요약 출력
- `test_grid_config_properties`: GridConfig 속성 테스트
- `test_missing_feature_columns_warning`: 누락 컬럼 경고 처리
- `test_all_grid_points_invalid`: 모든 그리드 포인트 무효 시 폴백
- `test_mi_sample_ratio_applied`: MI 샘플링 비율 적용

**통합 테스트** (1개):
- `test_end_to_end`: 전체 파이프라인 통합 테스트

---

## 테스트 요약

| 카테고리 | Basic | Edge Cases | 합계 | 상태 |
|----------|-------|------------|------|------|
| Cost Model | 6 | 4 | 10 | ✅ Pass |
| Triple Barrier | 4 | 6 | 10 | ✅ Pass |
| Scoring | 6 | 7 | 13 | ✅ Pass |
| Plateau Search | 3 | 5 | 8 | ✅ Pass |
| Optimizer | 2 | 4 | 6 | ✅ Pass |
| Integration | 1 | 0 | 1 | ✅ Pass |
| **Total** | **22** | **26** | **48** | **✅ All Pass** |

---

## 사용 예시

```python
from research.label_optimizer import (
    LabelOptimizer,
    GridConfig,
    SquareRootCostModel,
)

# 1. 기본 설정 (3×5×5 = 75 grid)
grid_config = GridConfig(
    sl_range=[1.0, 2.0, 3.0],
    pt_range=[1.5, 2.0, 2.5, 3.0, 3.5],
    time_range=[50, 100, 200, 500, 1000],
    min_samples=1000,
    mi_sample_ratio=0.1,  # MI 계산 시 10%만 샘플링 (속도 최적화)
)

# 1-1. Dense Grid (5×9×7 = 315 grid, 더 정밀한 Plateau 탐색)
dense_config = GridConfig(
    sl_range=[1.0, 1.5, 2.0, 2.5, 3.0],
    pt_range=[1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5],
    time_range=[50, 75, 100, 150, 200, 300, 500],
    mi_sample_ratio=0.1,
)

cost_model = SquareRootCostModel(
    base_fee=0.001,
    impact_coeff=0.1,
)

# 2. 최적화 실행
optimizer = LabelOptimizer(
    grid_config=grid_config,
    cost_model=cost_model,
)

result = optimizer.optimize(feature_df)

# 3. 결과 확인
print(optimizer.get_summary(result))
# Best Parameters:
#   SL Multiplier: 2.0σ
#   PT Multiplier: 2.5σ
#   Vertical Bars: 100

# 4. 라벨 사용
valid_indices = result.valid_indices  # 학습에 사용할 인덱스
labels = result.labels                # {-1, 0, +1} 라벨
```

---

## Blueprint 대비 구현 상태

| Blueprint 항목 | 구현 상태 | 비고 |
|---------------|-----------|------|
| Triple Barrier Method | ✅ 완료 | tbm.py |
| Fee Trap Guard (Per-bar) | ✅ 완료 | cost_model.is_viable() |
| Square Root Cost Model | ✅ 완료 | cost_model.py |
| Entropy 계산 | ✅ 완료 | scoring.py |
| Mutual Information | ✅ 완료 | sklearn 사용 |
| Rank IC (Spearman) | ✅ 완료 | scipy.stats 사용 |
| Composite Score | ✅ 완료 | 가중합 + 정규화 |
| Plateau Search | ✅ 완료 | scipy.ndimage 사용 |
| Grid Search | ✅ 완료 | 75개 조합 (3×5×5) |
| Sample-level Drop | ✅ 완료 | valid_indices 반환 |

---

## 누락 항목 체크

### ⚠️ 추가 검토 필요

1. **Ordinal Label 변환 (K=5)**
   - Blueprint: 5-class ordinal labels (quantile 기반)
   - 현재 구현: 3-class labels {-1, 0, +1}
   - **상태**: 현재는 3-class, K=5 변환은 별도 모듈 필요

2. **Grid 파라미터 범위**
   - Blueprint: SL [1.0, 2.0, 3.0], PT [1.5, 2.0, 2.5, 3.0, 3.5], Time [50, 100, 200, 500, 1000]
   - 현재 구현: GridConfig default와 일치 ✅

3. **실제 Feature 파일 연동**
   - 현재: DataFrame 직접 입력
   - 추가 필요: Parquet 로더 함수

---

## 리뷰 결정 사항

### 1. MI Downsampling ✅ 수용
- **문제**: `mutual_info_classif`가 느림 (KNN 기반)
- **해결**: `mi_sample_ratio=0.1` 옵션 추가
- **근거**: MI는 분포 통계량이므로 샘플링해도 경향성 유지

### 2. Plateau Interpolation ❌ 거부
- **제안**: scipy.ndimage.zoom으로 grid upsampling
- **거부 이유**: 평가하지 않은 파라미터에 점수 부여 = 가짜 데이터
- **대안**: Grid를 촘촘하게 (75 → 315) 설정 가능

### 3. Top-K Rank IC ❌ 거부
- **제안**: 상위 50% feature의 IC 평균만 사용
- **거부 이유**: 소수 feature 과적합 위험
- **유지**: 전체 평균 IC = 다양한 feature가 동의하는 robust label 선택
- **철학**: "Feature가 noise면 Feature Engineering 문제, Label Optimizer 문제 아님"

---

## 다음 단계

1. **K=5 Ordinal Label 변환기** (optional)
2. **Parquet Feature 로더** 연동
3. **CLI 인터페이스** 추가
4. **Stage 3 (Fine-tuning)** 구현

---

**Last Updated**: 2026-01-01
**Status**: ✅ Core Implementation Complete (48 tests passing)
