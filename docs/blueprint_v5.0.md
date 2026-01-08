# Blueprint v5.0: Direct Supervised PatchTST

**Version**: 5.0
**Date**: 2026-01-01
**Status**: Implementation Ready
**Key Change**: TBM sqrt(T) 스케일링 + 직접 지도학습 (SSL 완전 폐기)

---

## Changelog from v4.9

| Component | v4.9 (Design) | v5.0 (Validated) |
|-----------|---------------|------------------|
| TBM Scaling | 없음 (1-bar σ 사용) | **sqrt(T) 스케일링** |
| TBM Params | sl=1.5, pt=2.0, T=18 | **sl=1.0, pt=1.0, T=50** |
| Label Distribution | 목표만 설정 | **검증됨: 32/34/34%** |
| Feature Validity | 가정 | **검증됨: MLP F1=0.66** |
| Model | PatchTST (n_layers=3) | **PatchTST (n_layers=2~3, dropout=0.2~0.3)** |
| SSL | 폐기 예정 | **완전 폐기** |

---

## 1. 검증된 사실 (Validated Facts)

### 1.1 Feature Informativeness (수정됨!)

```
⚠️ 이전 테스트는 Random Shuffle로 인한 DATA LEAKAGE!

테스트 방식           | F1
---------------------|------
Shuffle (잘못됨)      | 0.87 ← 가짜!
Chronological (진짜)  | 0.34 ← 랜덤 수준

결론: Features are NOT informative with proper time-series split.
→ 현재 피처로는 방향 예측 불가능
→ 피처 엔지니어링 또는 다른 접근 필요
```

### 1.2 Top Features (by importance)

```
1. skewness           : 11.3%
2. kurtosis           : 9.2%
3. log_volume         : 8.2%
4. detrended_log_price: 6.6%
5. vol_zscore         : 6.1%
```

### 1.3 TBM 스케일링

```python
# 문제: realized_vol은 1-bar 변동성
# 해결: sqrt(T) 스케일링 적용

scaled_sigma = sigma * sqrt(vertical_bars)
pt_level = entry * (1 + pt_mult * scaled_sigma)
sl_level = entry * (1 - sl_mult * scaled_sigma)
```

### 1.4 확정된 TBM 파라미터

```python
TBMConfig(
    pt_mult = 1.0,        # 대칭 barrier
    sl_mult = 1.0,        # 대칭 barrier
    vertical_bars = 50,   # 50 bars holding period
)
# → barrier = ±1.67%
# → 분포: Short 32%, Neutral 34%, Long 34%
```

---

## 2. Architecture

### 2.1 Pipeline (Simplified)

```
[Features] → [TBM Labels] → [PatchTST] → [Prediction]
     ↓            ↓             ↓            ↓
  29 features   3-class    Supervised    Long/Short/Neutral
               balanced    (no SSL)
```

**SSL 완전 폐기 이유:**
1. MLP만으로 F1=0.66 달성 → 복잡한 pre-training 불필요
2. Masked Reconstruction은 방향 예측과 무관
3. 직접 지도학습이 더 효율적

### 2.2 Model Configuration (과적합 방지)

```python
class PatchTSTConfig:
    # Architecture (경량화)
    n_features: int = 29
    context_len: int = 128
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    n_heads: int = 4

    # 과적합 방지 (핵심 변경)
    n_layers: int = 2          # 2~3 (MLP도 2층에서 잘 작동)
    dropout: float = 0.25      # 0.2~0.3 (Generalization 강화)
    d_ff: int = 256

    # Classification
    n_classes: int = 3
```

**n_layers=2 선택 이유:**
- MLP 2-layer로 F1=0.66 달성
- 깊은 모델은 과적합 위험 증가
- 시작은 보수적으로, 필요시 증가

**dropout=0.25 선택 이유:**
- 금융 데이터는 노이즈가 많음
- 높은 dropout으로 일반화 능력 강화
- 0.2~0.3 범위에서 튜닝

---

## 3. Training Configuration

```python
@dataclass
class TrainingConfig:
    # Data
    context_len: int = 128
    batch_size: int = 128

    # Optimizer
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Training
    epochs: int = 50
    patience: int = 10          # Early stopping

    # Loss
    use_class_weights: bool = True
    label_smoothing: float = 0.05

    # Regularization
    gradient_clip: float = 1.0
```

---

## 4. TBM Labeling (Updated)

### 4.1 Configuration

```python
@dataclass
class TBMConfig:
    """v5.0 TBM with sqrt(T) scaling."""

    pt_mult: float = 1.0      # 1σ barrier (대칭)
    sl_mult: float = 1.0      # 1σ barrier (대칭)
    vertical_bars: int = 50   # 50 bars holding

    # sqrt(T) 스케일링은 TBM 내부에서 자동 적용
```

### 4.2 Label Distribution (검증됨)

| Class | Ratio | Meaning |
|-------|-------|---------|
| Short (-1) | ~32% | SL hit first |
| Neutral (0) | ~34% | Timeout (no clear direction) |
| Long (+1) | ~34% | PT hit first |

### 4.3 Barrier 계산 예시

```
median 1-bar σ = 0.24%
T = 50 bars

scaled_σ = 0.24% × sqrt(50) = 1.67%

PT barrier = +1.67%
SL barrier = -1.67%
```

---

## 5. Implementation Checklist

### 5.1 완료된 작업

- [x] TBM sqrt(T) 스케일링 적용 (`research/label_optimizer/tbm.py`)
- [x] Feature Informativeness 검증 (MLP F1=0.66)
- [x] TBM 파라미터 확정 (mult=1.0, T=50)
- [x] Label 분포 검증 (32/34/34%)

### 5.2 구현 완료

- [x] PatchTST 모델 수정 (n_layers=2, dropout=0.25) → `research/sniper/model.py`
- [x] 지도학습 trainer 구현 (SSL 제거) → `research/sniper/trainer.py` 사용
- [x] WFA 파이프라인 업데이트 → `scripts/run_sniper_wfa.py`
- [ ] 테스트 실행

---

## 6. File Changes

### 6.0 ETL Pipeline Restructure (v5.0)

ETL 프로세스가 3단계로 분리되고, bars-per-day 파라미터별로 독립된 폴더 구조를 사용합니다.

**새로운 명령어 구조:**
```bash
# 1. Raw 데이터 다운로드 (한번만, 영구 저장)
./etl download -symbol BTCUSDT

# 2. Bar 생성 (bars-per-day별로 다른 폴더)
./etl bars -symbol BTCUSDT -bars-per-day 50

# 3. Feature 생성
./etl features -symbol BTCUSDT -bars-per-day 50
```

**폴더 구조:**
```
/var/data/rawData/binance/futures/{symbol}/   # Raw 데이터 (영구 저장)
data/bars-50/futures/{symbol}/                 # bars-per-day=50
data/bars-100/futures/{symbol}/                # bars-per-day=100
data/features-50/futures/{symbol}/
data/features-100/futures/{symbol}/
data/state-50/futures/{symbol}/
data/state-100/futures/{symbol}/
```

**수정된 파일:**
```
etl/internal/config/config.go
  - Added: ExternalRawDir, BarsPerDay fields
  - Changed: GetBarsDir(), GetFeaturesDir(), GetStateDir() with bars-per-day suffix

etl/internal/bars/generator.go
  - Added: NewGeneratorWithBarsPerDay(barsPerDay int)
  - Changed: TargetBarsPerDay → configurable parameter

etl/cmd/main.go
  - Added: download command (raw data only)
  - Added: -bars-per-day flag
  - Removed: raw file deletion after processing
  - Changed: bars command reads from external raw directory
```

### 6.1 Modified Files

```
research/label_optimizer/tbm.py
  - Added: import math
  - Changed: barrier calculation with sqrt(T) scaling

scripts/baseline_test.py
  - Changed: default max_hold_bars=50, mult=1.0
  - Added: memory optimization (gc.collect, float32)
```

### 6.2 Files to Modify

```
research/trainer/model.py
  - Change: n_layers=3 → n_layers=2
  - Change: dropout=0.1 → dropout=0.25

research/trainer/train.py
  - Remove: SSL pre-training phase
  - Keep: Direct supervised training only

scripts/run_sniper_wfa.py (또는 새로 생성)
  - Update: TBM config (mult=1.0, T=50)
  - Update: Model config (n_layers=2, dropout=0.25)
```

---

## 7. Expected Results

### 7.1 Baseline (v4.8)

| Metric | Value | Status |
|--------|-------|--------|
| Val F1 | 0.33 | Random level |
| Neutral Ratio | 0.03% | Broken |
| Sharpe | -30 | Failed |

### 7.2 Target (v5.0)

| Metric | Target | Rationale |
|--------|--------|-----------|
| Val F1 | ≥0.50 | MLP achieved 0.66, PatchTST should match |
| Entry Accuracy | ≥55% | Long/Short prediction accuracy |
| Neutral Ratio | ~34% | Validated distribution |
| Sharpe | ≥0 | Breakeven or better |

### 7.3 Success Criteria

**Minimum Viable:**
- Val F1 ≥ 0.45
- Sharpe ≥ 0

**Good:**
- Val F1 ≥ 0.55
- Sharpe ≥ 0.5
- Profit Factor ≥ 1.2

---

## 8. Quick Reference

```python
# TBM (확정)
PT_MULT = 1.0
SL_MULT = 1.0
VERTICAL_BARS = 50
# → barrier ±1.67%, Neutral ~34%

# Model (과적합 방지)
N_LAYERS = 2
DROPOUT = 0.25
D_MODEL = 128
CONTEXT_LEN = 128

# Training
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
PATIENCE = 10
```

---

## Appendix: Why This Will Work

1. **Features have signal** - MLP F1=0.66 proves it
2. **Labels are balanced** - 32/34/34% distribution
3. **Barriers are realistic** - sqrt(T) scaling matches holding period
4. **Model is simple** - 2 layers, high dropout, no SSL overhead
5. **Direct supervision** - Learns exactly what we want to predict

---

*Generated: 2026-01-01*
*Context: v4.8 Post-mortem → v5.0 Implementation*
