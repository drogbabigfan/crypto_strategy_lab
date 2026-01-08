# SSL Pre-training Implementation (v4.6)

## 개요

Blueprint v4.6 Section 4 (Stage 0: One-off SSL Pre-training) 구현 완료.

**목적**: Multi-Asset Feature로부터 Foundation Encoder 학습

---

## 파일 구조

```
research/ssl/
├── __init__.py          # Package exports
├── dataset.py           # SSLDataConfig, PatchMaskingDataset
├── model.py             # PatchTSTEncoder, SSLModel
└── train.py             # SSLTrainer, EarlyStopping

tests/
└── test_ssl.py          # 25 test cases
```

---

## 모듈별 상세

### 1. dataset.py - Data Loading & Masking

**클래스/함수**:

| 이름 | 설명 |
|------|------|
| `SSLDataConfig` | 데이터 로딩 설정 (assets, context_len, feature_cols) |
| `load_ssl_data()` | Multi-asset feature 로드 + sliding window 생성 |
| `PatchMaskingDataset` | Patch-wise Random Masking (40%) |

**핵심 동작**:
```
1. 각 자산의 Feature Parquet 로드 (Go ETL에서 정규화 완료)
2. Sliding Window 생성 (context_len=512)
3. 자산 간 Shuffle (시간순 무시)
4. 학습 시 40% 패치 Zero Masking
```

**테스트 케이스** (7개):
- `test_default_assets`: 기본 자산 목록 확인
- `test_default_feature_cols`: 29개 Feature 확인
- `test_dataset_creation`: Dataset 생성
- `test_getitem_shapes`: 출력 shape 검증
- `test_masking_applied`: 마스킹 동작 확인
- `test_different_masks_per_sample`: 샘플별 다른 마스크
- `test_context_len_not_divisible_by_patch_len`: 에러 처리

---

### 2. model.py - PatchTST Encoder

**아키텍처**:
```
Input: (Batch, n_features, context_len)
        ↓ [29 channels, 512 timesteps]
Patching: (Batch, n_features, n_patches, patch_len)
        ↓ [29, 32 patches, 16]
Patch Embedding: Linear(16 → 128)
        ↓
Positional Embedding: Learnable (32 positions)
        ↓
Transformer Encoder: 4 heads, 2 layers, d_model=128
        ↓
Reconstruction Head: Linear(128 → 16)
        ↓
Output: Reconstructed patches (masked only)
```

**클래스**:

| 클래스 | 설명 |
|--------|------|
| `PatchTSTConfig` | 모델 설정 (d_model, n_heads, n_layers...) |
| `PatchEmbedding` | 패치 → 임베딩 변환 |
| `PositionalEncoding` | Learnable 위치 인코딩 |
| `TransformerEncoderLayer` | Pre-norm Transformer 레이어 |
| `PatchTSTEncoder` | 전체 인코더 |
| `ReconstructionHead` | SSL용 복원 헤드 |
| `SSLModel` | Encoder + Head 통합 |

**핵심 메서드**:

| 메서드 | 설명 |
|--------|------|
| `SSLModel.forward()` | 순전파 (return_encoded 옵션) |
| `SSLModel.save_encoder()` | Encoder만 저장 (Fine-tuning용) |
| `SSLModel.load_encoder()` | Pre-trained Encoder 로드 |
| `masked_reconstruction_loss()` | 마스킹된 패치만 MSE 계산 |

**테스트 케이스** (10개):
- `test_forward_shape` (PatchEmbedding)
- `test_forward_shape` (PositionalEncoding)
- `test_forward_shape` (TransformerEncoderLayer)
- `test_forward_shape` (PatchTSTEncoder)
- `test_output_dim`
- `test_reconstruction_shape`
- `test_encoded_shape`
- `test_save_load_encoder`
- `test_loss_computation`
- `test_zero_loss_perfect_reconstruction`

---

### 3. train.py - Training Loop

**클래스**:

| 클래스 | 설명 |
|--------|------|
| `SSLTrainingConfig` | 학습 설정 |
| `EarlyStopping` | Early stopping 핸들러 |
| `SSLTrainer` | 학습 오케스트레이터 |

**Fail-Fast 정책**:
```python
MIN_SAMPLES = 30000
DIVERGENCE_CHECK_EPOCHS = 5
MIN_LOSS_REDUCTION = 0.05  # 5%
```

**학습 파이프라인**:
```
1. Multi-asset Feature 로드
2. PatchMaskingDataset 생성
3. Train/Val 분할 (90/10)
4. AdamW 최적화
5. Masked Reconstruction Loss
6. Early Stopping (patience=5)
7. Encoder 저장 (artifacts/ssl/)
```

**테스트 케이스** (6개):
- `test_no_stop_improving`
- `test_stop_not_improving`
- `test_restore_best`
- `test_insufficient_data_error`
- `test_training_small_data`
- `test_encoder_save_during_training`

---

### 4. Integration Tests (2개)

- `test_end_to_end_small`: 전체 파이프라인 테스트
- `test_encoder_forward_after_loading`: 저장/로드 후 동일 출력 검증

---

## 테스트 요약

| 카테고리 | 테스트 수 | 상태 |
|----------|-----------|------|
| SSLDataConfig | 2 | ✅ Pass |
| PatchMaskingDataset | 5 | ✅ Pass |
| PatchEmbedding | 1 | ✅ Pass |
| PositionalEncoding | 1 | ✅ Pass |
| TransformerEncoderLayer | 1 | ✅ Pass |
| PatchTSTEncoder | 2 | ✅ Pass |
| SSLModel | 3 | ✅ Pass |
| MaskedReconstructionLoss | 2 | ✅ Pass |
| EarlyStopping | 3 | ✅ Pass |
| SSLTrainer | 3 | ✅ Pass |
| Integration | 2 | ✅ Pass |
| **Total** | **25** | **✅ All Pass** |

---

## 사용 예시

```python
from research.ssl import (
    SSLDataConfig,
    SSLTrainingConfig,
    SSLTrainer,
    SSLModel,
)

# 1. 데이터 설정
data_config = SSLDataConfig(
    assets=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT",
            "DOGEUSDT", "BNBUSDT", "LTCUSDT"],
    context_len=512,
    min_samples_per_asset=10000,
)

# 2. 학습 설정
train_config = SSLTrainingConfig(
    n_features=29,
    context_len=512,
    patch_len=16,
    d_model=128,
    n_heads=4,
    n_layers=2,
    mask_ratio=0.4,
    epochs=30,
    batch_size=64,
    learning_rate=1e-4,
    patience=5,
)

# 3. 학습 실행
trainer = SSLTrainer(train_config)
model = trainer.train(
    features_dir="data/features/futures",
    data_config=data_config,
    output_path="artifacts/ssl/foundation_encoder.pt",
)

# 4. Encoder 로드 (Fine-tuning용)
encoder = SSLModel.load_encoder("artifacts/ssl/foundation_encoder.pt")
```

---

## 모델 파라미터

기본 설정 기준:
- **n_features**: 29
- **context_len**: 512
- **patch_len**: 16
- **n_patches**: 32
- **d_model**: 128
- **n_heads**: 4
- **n_layers**: 2
- **d_ff**: 256

총 파라미터 수: ~300K (경량 모델)

---

## Blueprint 대비 구현 상태

| Blueprint 항목 | 구현 상태 | 비고 |
|---------------|-----------|------|
| Multi-Asset 데이터 로딩 | ✅ 완료 | 7개 자산 지원 |
| Patch-wise Random Masking (40%) | ✅ 완료 | Zero masking |
| PatchTST Encoder | ✅ 완료 | Pre-norm Transformer |
| Masked Reconstruction Loss | ✅ 완료 | MSE on masked patches |
| Early Stopping | ✅ 완료 | patience=5 |
| Fail-Fast Policy | ✅ 완료 | 데이터/수렴 체크 |
| Encoder 저장/로드 | ✅ 완료 | Fine-tuning용 |

---

## 다음 단계

1. **Stage 3: Fine-tuning** 구현
   - Encoder 로드 + Classification Head
   - Focal Loss
   - 2-Phase Training (Frozen → Unfrozen)

2. **Stage 4: WFA Engine** 구현
   - Walk-Forward Analysis Loop
   - Realistic Backtester

---

**Last Updated**: 2026-01-01
**Status**: ✅ Core Implementation Complete (25 tests passing)
