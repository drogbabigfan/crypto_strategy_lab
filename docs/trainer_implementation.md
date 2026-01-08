# Fine-tuning Trainer Implementation (v4.6)

## 개요

Blueprint v4.6 Section 8 (Step 3.2: Fine-tuning) 구현 완료.

**목적**: Pre-trained Encoder + Classification Head로 지도 학습

---

## 파일 구조

```
research/trainer/
├── __init__.py          # Package exports
├── dataset.py           # FineTuningConfig, FineTuningDataset
├── model.py             # PatchTSTClassifier, ResidualBottleneckHead
├── loss.py              # FocalLoss, CrossEntropyWithSmoothing
└── train.py             # FineTuningTrainer, EarlyStopping

tests/
└── test_trainer.py      # 37 test cases
```

---

## 모듈별 상세

### 1. dataset.py - Sample-level Drop

**클래스/함수**:

| 이름 | 설명 |
|------|------|
| `FineTuningConfig` | 데이터 설정 (context_len, feature_cols) |
| `FineTuningDataset` | Sample-level Drop 기반 Dataset |

**핵심 동작**:
```
1. valid_indices: Label Optimizer (Step 3.1)에서 생성
2. Context Window 내부의 바는 삭제하지 않음
3. Fee Trap인 샘플만 전체 제외
4. 연속 512개 바 유지 (시계열 연속성 보존)
```

**테스트 케이스** (9개):
- `test_dataset_creation`: Dataset 생성
- `test_getitem_shapes`: 출력 shape 검증
- `test_label_shift`: {-1,0,1} → {0,1,2} 변환
- `test_context_window_continuous`: 연속성 검증
- `test_insufficient_context_filtered`: 컨텍스트 부족 필터링
- `test_length_mismatch_error`: 길이 불일치 에러
- `test_no_usable_samples_error`: 사용 가능 샘플 없음 에러
- `test_class_distribution`: 클래스 분포 계산
- `test_class_weights`: 클래스 가중치 계산

---

### 2. model.py - PatchTSTClassifier

**아키텍처**:
```
[Pre-trained PatchTST Encoder] ← foundation_encoder.pt 로드
              ↓
      Flatten (n_features × n_patches × d_model)
              ↓
      Residual Bottleneck Head:
        Linear(flatten_dim → 256) → GELU → Dropout(0.3)
        Linear(256 → flatten_dim)
        Residual Connection + LayerNorm
              ↓
      Classification: Linear(flatten_dim → 3)
              ↓
      Output: Logits [Short, Neutral, Long]
```

**클래스**:

| 클래스 | 설명 |
|--------|------|
| `ClassifierConfig` | 분류기 설정 (n_classes, bottleneck_dim) |
| `ResidualBottleneckHead` | Residual 기반 Classification Head |
| `PatchTSTClassifier` | Encoder + Head 통합 모델 |

**핵심 메서드**:

| 메서드 | 설명 |
|--------|------|
| `freeze_encoder()` | Phase 1: Encoder 가중치 동결 |
| `unfreeze_encoder()` | Phase 2: Encoder 가중치 해제 |
| `from_pretrained()` | Pre-trained encoder에서 생성 |
| `save()` / `load()` | 모델 저장/로드 |

**테스트 케이스** (5개):
- `test_forward_shape`: 출력 shape 검증
- `test_freeze_encoder`: Encoder 동결 검증
- `test_unfreeze_encoder`: Encoder 해제 검증
- `test_save_load`: 저장/로드 일관성
- `test_from_pretrained`: Pre-trained 로드

---

### 3. loss.py - Focal Loss

**수식**:
```
FL(p_t) = -(1 - p_t)^γ × log(p_t)

- γ = 0: Standard Cross Entropy
- γ > 0: Easy example 가중치 감소
```

**클래스**:

| 클래스 | 설명 |
|--------|------|
| `FocalLoss` | Focal Loss + Label Smoothing |
| `CrossEntropyWithSmoothing` | 표준 CE + Label Smoothing |

**FocalLoss 파라미터**:
- `gamma`: Focusing parameter (default 2.0)
- `label_smoothing`: Smoothing factor (default 0.1)
- `weight`: Optional class weights
- `reduction`: 'mean', 'sum', 'none'

**테스트 케이스** (7개):
- `test_loss_computation`: 손실 계산
- `test_gamma_effect`: gamma 효과 검증
- `test_label_smoothing_effect`: smoothing 효과
- `test_class_weights`: 클래스 가중치 적용
- `test_reduction_modes`: 리덕션 모드
- CE: `test_loss_computation`, `test_matches_standard_ce`

---

### 4. train.py - 2-Phase Fine-tuning

**Phase 구성**:
```
Phase 1 (frozen_epochs): Encoder Frozen
  - Classification Head만 학습
  - Pre-trained 표현 보존
  - Learning Rate: head_lr (1e-4)

Phase 2 (나머지 epochs): Full Unfrozen
  - 전체 모델 미세조정
  - Encoder LR: encoder_lr (1e-5) - 더 낮음
  - Head LR: head_lr (1e-4)
```

**클래스**:

| 클래스 | 설명 |
|--------|------|
| `FineTuningTrainingConfig` | 학습 설정 |
| `EarlyStopping` | Early stopping 핸들러 (F1 기반) |
| `FineTuningTrainer` | 학습 오케스트레이터 |

**Fail-Fast 정책**:
```python
MIN_SAMPLES = 1000
DIVERGENCE_CHECK_EPOCHS = 5
MIN_LOSS_REDUCTION = 0.05  # 5%
```

**학습 파이프라인**:
```
1. Pre-trained Encoder 로드
2. Classification Head 부착
3. Phase 1: Encoder Frozen (5 epochs)
4. Phase 2: Full Unfrozen (remaining epochs)
5. AdamW 최적화 (differential LR)
6. Focal Loss
7. Early Stopping (patience=5, F1 기반)
8. Best Model 저장
```

**테스트 케이스** (8개):
- `test_no_stop_improving`: Early stopping 미동작
- `test_stop_not_improving`: Early stopping 동작
- `test_restore_best`: Best 모델 복원
- `test_insufficient_data_error`: 데이터 부족 에러
- `test_training_small_data`: 소규모 데이터 학습
- `test_two_phase_training`: 2-Phase 검증
- `test_model_save_during_training`: 학습 중 저장
- `test_training_summary`: 학습 요약

---

### 5. Integration Tests (3개)

- `test_end_to_end_pipeline`: 전체 파이프라인 테스트
- `test_gradient_flow`: Gradient 흐름 검증
- `test_frozen_phase_no_encoder_gradients`: Phase 1 Encoder gradient 없음

---

## 테스트 요약

| 카테고리 | 테스트 수 | 상태 |
|----------|-----------|------|
| FineTuningConfig | 2 | ✅ Pass |
| FineTuningDataset | 9 | ✅ Pass |
| ResidualBottleneckHead | 2 | ✅ Pass |
| PatchTSTClassifier | 5 | ✅ Pass |
| FocalLoss | 5 | ✅ Pass |
| CrossEntropyWithSmoothing | 2 | ✅ Pass |
| EarlyStopping | 3 | ✅ Pass |
| FineTuningTrainer | 5 | ✅ Pass |
| finetune function | 1 | ✅ Pass |
| Integration | 3 | ✅ Pass |
| **Total** | **37** | **✅ All Pass** |

---

## 사용 예시

```python
from research.ssl import SSLModel
from research.trainer import (
    FineTuningDataset,
    PatchTSTClassifier,
    FineTuningTrainer,
    FineTuningTrainingConfig,
    FocalLoss,
    finetune,
)

# 1. 데이터 준비 (Label Optimizer 출력 사용)
dataset = FineTuningDataset(
    features=train_features,      # (N, 29)
    valid_indices=valid_indices,  # Step 3.1 출력
    labels=labels,                # Step 3.1 출력
    context_len=512,
)

# 2. Pre-trained Encoder에서 Classifier 생성
classifier = PatchTSTClassifier.from_pretrained(
    encoder_path="artifacts/ssl/foundation_encoder.pt",
    n_classes=3,
    bottleneck_dim=256,
    dropout=0.3,
)

# 3. 학습 설정
config = FineTuningTrainingConfig(
    frozen_epochs=5,
    total_epochs=30,
    batch_size=64,
    encoder_lr=1e-5,
    head_lr=1e-4,
    focal_gamma=2.0,
    label_smoothing=0.1,
    patience=5,
)

# 4. 학습 실행
trainer = FineTuningTrainer(config)
model = trainer.train(
    classifier,
    train_dataset=dataset,
    val_dataset=val_dataset,
    class_weights=dataset.get_class_weights(),
    output_path="artifacts/classifier/fold_0_model.pt",
)

# 또는 convenience 함수 사용
model = finetune(
    encoder_path="artifacts/ssl/foundation_encoder.pt",
    train_dataset=dataset,
    val_dataset=val_dataset,
    config=config,
    output_path="artifacts/classifier/fold_0_model.pt",
)

# 5. 추론
model.eval()
with torch.no_grad():
    logits = model(x)  # (Batch, 3)
    preds = torch.argmax(logits, dim=-1)  # {0, 1, 2}
    signals = preds - 1  # {-1, 0, +1}
```

---

## 모델 파라미터

기본 설정 기준 (SSL Encoder 포함):
- **n_features**: 29
- **context_len**: 512
- **n_patches**: 32
- **d_model**: 128
- **flatten_dim**: 29 × 32 × 128 = 118,784
- **bottleneck_dim**: 256
- **n_classes**: 3

Encoder 파라미터: ~300K
Classifier Head 파라미터: ~30.5M (bottleneck + classifier)
**총 파라미터**: ~31M

---

## Blueprint 대비 구현 상태

| Blueprint 항목 | 구현 상태 | 비고 |
|---------------|-----------|------|
| FineTuningDataset | ✅ 완료 | Sample-level Drop |
| Residual Bottleneck Head | ✅ 완료 | LayerNorm + Residual |
| PatchTSTClassifier | ✅ 완료 | from_pretrained 지원 |
| Focal Loss | ✅ 완료 | γ=2.0, smoothing=0.1 |
| 2-Phase Training | ✅ 완료 | Frozen → Unfrozen |
| Differential LR | ✅ 완료 | encoder_lr < head_lr |
| Fail-Fast Policy | ✅ 완료 | 데이터/수렴 체크 |
| Early Stopping | ✅ 완료 | F1 기반, patience=5 |
| Class Weights | ✅ 완료 | Inverse frequency |

---

## 다음 단계

1. **Step 3.3: Execution & Backtest** 구현
   - RealisticBacktester (Next Bar Entry)
   - WFA Engine (Walk-Forward 메인 루프)
   - Metrics 계산

2. **Utils** 구현
   - FailFastPolicy (공통 모듈화)
   - Feature Loader (Parquet 로딩)

---

**Last Updated**: 2026-01-01
**Status**: ✅ Core Implementation Complete (37 tests passing)
