# WFA Pipeline Debugging Report

**Date**: 2026-01-01
**Status**: Model not learning - predictions no better than random

---

## Executive Summary

SSL pre-training + Fine-tuning + WFA 파이프라인을 BTC 데이터로 실행한 결과, 모델이 랜덤 수준의 예측만 수행하며 실제 수익을 내지 못함. 여러 수정을 시도했으나 근본적인 문제 해결 필요.

---

## 1. Pipeline Overview

```
[Raw Data] → [ETL/Features] → [SSL Pre-training] → [Fine-tuning per Fold] → [Signal Generation] → [Backtest]
```

- **Data**: BTC 5-min bars, 2019-09 ~ 2025-11 (104,788 samples)
- **Features**: 29 features (volume, volatility, momentum, etc.)
- **SSL**: PatchTST encoder with masked reconstruction
- **Labeling**: Triple Barrier Method (SL=2σ, PT=2.5σ, max_hold=100 bars)
- **Classifier**: 3-class (Short, Neutral, Long)

---

## 2. Identified Problems

### 2.1 Label Distribution Imbalance

**문제**: Neutral 레이블이 거의 없음 (전체의 0.03%)

```
Distribution:
  Short (-1): 55% (~5,300 samples)
  Neutral (0): 0.03% (~3 samples)  ← 문제!
  Long (+1): 45% (~4,300 samples)
```

**원인**: 현재 TBM 파라미터에서 100 bars 내 PT/SL이 거의 항상 hit됨

| Metric | Value |
|--------|-------|
| 100 bars 최대 상승 | 평균 2.49% |
| 100 bars 최대 하락 | 평균 3.24% |
| PT threshold (2.5σ) | ~1% |
| SL threshold (2.0σ) | ~0.8% |
| **Timeout 확률** | **~0%** |

**영향**:
- 3-class classifier에서 1개 class가 비어있음
- Class weights 계산 시 neutral에 극단값 (weight=1000+)
- Loss function 불안정

---

### 2.2 Class Weights Explosion

**문제**: Neutral class weight가 폭발하여 학습 방해

```python
# Before fix
Class weights: [0.60, 1004.3, 0.74]  # Neutral = 1004!

# After fix (capping)
Class weights: [0.60, 10.0, 0.74]
```

**해결**: `max_weight=10.0`으로 capping 적용 (dataset.py)

---

### 2.3 Encoder Contamination Across Folds

**문제**: 동일한 encoder 참조가 모든 fold에서 수정됨

```python
# Before fix - same encoder reference
classifier = PatchTSTClassifier(encoder=encoder.encoder, ...)
# Phase 2 unfreezes and modifies encoder!

# After fix - deepcopy per fold
encoder_copy = copy.deepcopy(base_encoder)
classifier = PatchTSTClassifier(encoder=encoder_copy, ...)
```

**해결**: `copy.deepcopy()` 적용

---

### 2.4 Validation Dataset Missing

**문제**: `val_dataset=None`이어서 Val F1 항상 0.0

```python
# Before
classifier = trainer.train(model=classifier, train_dataset=train_dataset)

# After
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
classifier = trainer.train(..., val_dataset=val_dataset)
```

**해결**: 90/10 train/val split 적용

---

### 2.5 Model Collapse - Single Class Prediction

**문제**: 모델이 한 클래스만 예측 (모든 Short 또는 모든 Long)

```
Fold 0: Avg probs [Short, Neutral, Long] = [0.92, 0.00, 0.08]  → All Short
Fold 1: Avg probs [Short, Neutral, Long] = [0.00, 0.00, 1.00]  → All Long
```

**원인 분석**:
1. argmax가 항상 다수 클래스 선택
2. 학습 불안정으로 fold마다 다른 클래스로 collapse
3. Val F1 = 0.33~0.35 (랜덤 수준)에서 개선 안 됨

**시도한 해결책**:
- Threshold-based signal (10% margin 필요) → 효과 없음
- Encoder 완전 freeze → 효과 없음

---

### 2.6 SSL Representations Not Predictive

**핵심 문제**: SSL encoder가 학습한 표현이 가격 방향 예측에 유용하지 않음

```
Val F1:  0.33 ~ 0.35 (랜덤과 동일)
Val Acc: 45.7% ~ 54.3% (클래스 분포와 동일)
```

**분석**:
- SSL objective (masked reconstruction)는 시계열 패턴 복원 학습
- 이 표현이 "미래 방향"을 예측하는데 필요한 정보 포함 여부 불확실
- Encoder frozen 상태에서도 classifier가 학습 안 됨 → 표현 자체가 문제

---

## 3. Current Results

```
WFA Summary (2 folds before early stop):
  Sharpe Ratio:  -29.59 +/- 4.30
  Win Rate:      43.9% +/- 2.8%
  Total PnL:     -32.86% +/- 7.89%
  Cumulative:    -65.73%  (stopped at -50% threshold)
```

---

## 4. Potential Solutions

### Option A: Skip SSL, Use Raw Features
```
[Raw Features] → [Simple Classifier] → [Signals]
```
- SSL encoder 제거
- Raw 29 features로 직접 분류기 학습
- 장점: 단순, 빠름
- 단점: 시계열 context 활용 못함

### Option B: Change SSL Objective
```
Current: Masked Reconstruction (예측 = 마스킹된 패치 복원)
Alternative: Contrastive Learning (예측 = 유사/비유사 구분)
```
- 상승 vs 하락 구간을 contrastive pairs로 학습
- 방향 예측에 더 적합한 표현 학습 가능

### Option C: 2-Class Classification
```
현재: 3-class (Short, Neutral, Long) with Neutral=0%
변경: 2-class (Short, Long) only
```
- Neutral 클래스 제거
- Label/Loss/Model 단순화

### Option D: Different Architecture
- PatchTST 대신 다른 모델 (LSTM, Transformer, etc.)
- 또는 ensemble 방식

### Option E: Feature Engineering Review
- 현재 29개 features가 방향 예측에 충분한 정보 포함하는지 검토
- Additional features: order book, funding rate, sentiment, etc.

### Option F: TBM Parameter Tuning
```
현재: SL=2σ, PT=2.5σ, max_hold=100 bars
변경: vertical_bars 줄이거나 σ multiplier 증가
```
- Timeout (Neutral) 비율 조정
- 또는 asymmetric PT/SL로 상승장 bias 반영

---

## 5. Code Changes Made

| File | Change |
|------|--------|
| `scripts/run_btc_wfa.py` | Deepcopy encoder per fold |
| `scripts/run_btc_wfa.py` | Add validation split |
| `scripts/run_btc_wfa.py` | Skip class weights when neutral < 100 |
| `scripts/run_btc_wfa.py` | Early stop at -50% cumulative PnL |
| `scripts/run_btc_wfa.py` | Threshold-based signals |
| `research/trainer/dataset.py` | Cap class weights at max=10.0 |

---

## 6. Questions for Further Research

1. **SSL 목표와 downstream task의 alignment**
   - Masked reconstruction이 방향 예측에 적합한가?
   - 다른 self-supervised objective가 더 나은가?

2. **Feature 품질**
   - 29개 features가 충분한 예측력을 가지는가?
   - 어떤 features가 가장 중요한가? (feature importance 분석)

3. **Label 품질**
   - Triple Barrier labels이 실제 tradeable signals인가?
   - 다른 labeling 방식 (trend following, mean reversion)?

4. **Market Efficiency**
   - 5-min bar 수준에서 예측 가능한 alpha가 존재하는가?
   - 더 긴/짧은 timeframe이 나은가?

5. **Model Capacity**
   - PatchTST encoder 크기가 충분한가?
   - Classifier head 구조가 적절한가?

---

## 7. Appendix: Key File Locations

```
scripts/run_btc_wfa.py          # Main WFA pipeline
research/ssl/model.py           # PatchTST encoder
research/ssl/train.py           # SSL training
research/trainer/train.py       # Fine-tuning trainer
research/trainer/dataset.py     # Dataset with class weights
research/trainer/loss.py        # Focal loss
research/label_optimizer/tbm.py # Triple Barrier labeler
research/wfa/signal_generator.py # Signal generation
artifacts/ssl/foundation_encoder.pt # Trained SSL encoder
results/wfa/wfa_results_*.json  # WFA results
```
