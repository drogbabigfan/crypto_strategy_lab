#!/usr/bin/env python
"""
Feature Informativeness Test (Chronological Split)

Tests if current features have predictive power with proper time-series validation.
NO SHUFFLE - uses chronological split with purging.

Tests:
1. Different T values (5, 10, 20, 50, 100, 200, 500)
2. Binary classification (Long vs Short, Neutral excluded)
3. Volatility prediction (future vol > n * current vol)

Usage:
    python scripts/feature_informativeness_test.py
"""

import sys
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.label_optimizer.tbm import TripleBarrierLabeler, TBMConfig

# Paths
FEATURES_DIR = PROJECT_ROOT / "etl" / "data" / "features-24" / "futures" / "BTCUSDT"
BARS_DIR = PROJECT_ROOT / "etl" / "data" / "bars-24" / "futures" / "BTCUSDT"


def load_data(months=["2025-06"]):
    """Load data for specified months."""
    print(f"Loading data for {months}...")

    features_list = []
    bars_list = []

    for month in months:
        feat_file = FEATURES_DIR / f"BTCUSDT-features-{month}.parquet"
        bar_file = BARS_DIR / f"BTCUSDT-bars-{month}.parquet"

        if feat_file.exists() and bar_file.exists():
            features_list.append(pq.read_table(feat_file).to_pandas())
            bars_list.append(pq.read_table(bar_file).to_pandas())

    features = pd.concat(features_list, ignore_index=True)
    bars = pd.concat(bars_list, ignore_index=True)

    # Merge
    bars_cols = ['start_time', 'open', 'high', 'high_time', 'low', 'low_time', 'close', 'volume']
    feature_exclude = {'start_time', 'end_time', 'open', 'high', 'low', 'close'}
    feature_cols_to_use = ['start_time'] + [c for c in features.columns if c not in feature_exclude]

    merged = pd.merge(bars[bars_cols], features[feature_cols_to_use], on='start_time', how='inner')
    merged = merged.sort_values('start_time').reset_index(drop=True)

    if 'is_primed' in merged.columns:
        merged = merged[merged['is_primed']].reset_index(drop=True)

    return merged


def get_feature_cols(df):
    """Get feature column names."""
    exclude = {'start_time', 'end_time', 'open', 'high', 'high_time',
               'low', 'low_time', 'close', 'volume', 'is_primed'}
    return [c for c in df.columns if c not in exclude]


def chronological_split(X, y, test_ratio=0.2, purge_gap=50):
    """Split data chronologically with purging gap."""
    split_idx = int(len(X) * (1 - test_ratio))
    gap = min(purge_gap, split_idx // 2)

    X_train = X[:split_idx - gap]
    y_train = y[:split_idx - gap]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    return X_train, X_test, y_train, y_test


def test_direction_prediction(merged, feature_cols, T, mult=1.0, binary=False):
    """Test direction prediction with TBM labels."""
    config = TBMConfig(sl_mult=mult, pt_mult=mult, vertical_bars=T)
    labeler = TripleBarrierLabeler(config)
    result = labeler.label(merged)

    X = np.array([merged[feature_cols].iloc[idx].values
                  for idx in result.valid_indices], dtype=np.float32)
    y = result.labels
    X = np.nan_to_num(X, nan=0.0)

    if binary:
        # Exclude Neutral
        mask = y != 0
        X, y = X[mask], y[mask]
        if len(X) < 100:
            return None

    if len(X) < 200:
        return None

    # Chronological split
    X_tr, X_te, y_tr, y_te = chronological_split(X, y, purge_gap=T)

    if len(X_tr) < 50 or len(X_te) < 20:
        return None

    # Train and evaluate
    rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=2)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)

    f1 = f1_score(y_te, y_pred, average='macro')
    acc = accuracy_score(y_te, y_pred)

    # Label distribution
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique, counts))

    del rf
    gc.collect()

    return {
        'f1': f1,
        'accuracy': acc,
        'n_samples': len(X),
        'n_train': len(X_tr),
        'n_test': len(X_te),
        'distribution': dist,
    }


def test_volatility_prediction(merged, feature_cols, n_mult, lookahead=20):
    """Test if features can predict future volatility spikes."""
    X_list, y_list = [], []

    for i in range(len(merged) - lookahead):
        if 'realized_vol' not in merged.columns:
            break

        current_vol = merged['realized_vol'].iloc[i]
        if current_vol <= 0 or pd.isna(current_vol):
            continue

        # Calculate future realized volatility
        future_returns = merged['close'].iloc[i+1:i+1+lookahead].pct_change().dropna()
        if len(future_returns) < 5:
            continue
        future_vol = future_returns.std()

        # Label: 1 if future vol >= n_mult * current vol, else 0
        label = 1 if future_vol >= n_mult * current_vol else 0

        X_list.append(merged[feature_cols].iloc[i].values)
        y_list.append(label)

    if len(X_list) < 200:
        return None

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    X = np.nan_to_num(X, nan=0.0)

    # Chronological split
    X_tr, X_te, y_tr, y_te = chronological_split(X, y, purge_gap=lookahead)

    if len(X_tr) < 50 or len(X_te) < 20:
        return None

    rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=2)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)

    f1 = f1_score(y_te, y_pred, average='macro')
    acc = accuracy_score(y_te, y_pred)
    pos_ratio = y.mean()

    del rf
    gc.collect()

    return {
        'f1': f1,
        'accuracy': acc,
        'n_samples': len(X),
        'positive_ratio': pos_ratio,
    }


def main():
    print("=" * 70)
    print("Feature Informativeness Test (Chronological Split)")
    print("=" * 70)
    print()
    print("⚠️  Using CHRONOLOGICAL split (no shuffle)")
    print("⚠️  Purging gap applied to prevent label overlap leakage")
    print()

    # Load data
    merged = load_data(months=["2025-06"])
    feature_cols = get_feature_cols(merged)
    print(f"Data: {len(merged):,} rows, {len(feature_cols)} features")
    print(f"Features: {feature_cols}")
    print()

    results = {}

    # ============================================
    # TEST 1: Different T values (3-class direction)
    # ============================================
    print("=" * 70)
    print("TEST 1: Direction Prediction (3-class) - Different Holding Periods")
    print("=" * 70)
    print(f"{'T':>5} | {'F1':>6} | {'Acc':>6} | {'Short':>6} | {'Neut':>6} | {'Long':>6} | Status")
    print("-" * 70)

    for T in [5, 10, 20, 50, 100, 200, 500]:
        result = test_direction_prediction(merged, feature_cols, T, mult=1.0, binary=False)
        if result is None:
            print(f"{T:>5} | {'N/A':>6} | {'N/A':>6} | Not enough data")
            continue

        dist = result['distribution']
        n = result['n_samples']
        short_pct = dist.get(-1, 0) / n * 100
        neut_pct = dist.get(0, 0) / n * 100
        long_pct = dist.get(1, 0) / n * 100

        status = "✓ Signal" if result['f1'] > 0.40 else "✗ Random"
        print(f"{T:>5} | {result['f1']:>6.3f} | {result['accuracy']:>6.3f} | "
              f"{short_pct:>5.1f}% | {neut_pct:>5.1f}% | {long_pct:>5.1f}% | {status}")

        results[f'direction_T{T}'] = result

    # ============================================
    # TEST 2: Binary Classification (Long vs Short)
    # ============================================
    print()
    print("=" * 70)
    print("TEST 2: Binary Classification (Long vs Short, Neutral excluded)")
    print("=" * 70)
    print(f"{'T':>5} | {'F1':>6} | {'Acc':>6} | {'Samples':>8} | Status")
    print("-" * 70)

    for T in [10, 20, 50, 100]:
        result = test_direction_prediction(merged, feature_cols, T, mult=1.0, binary=True)
        if result is None:
            print(f"{T:>5} | {'N/A':>6} | {'N/A':>6} | Not enough data")
            continue

        status = "✓ Signal" if result['f1'] > 0.55 else "✗ Random"
        print(f"{T:>5} | {result['f1']:>6.3f} | {result['accuracy']:>6.3f} | "
              f"{result['n_samples']:>8,} | {status}")

        results[f'binary_T{T}'] = result

    # ============================================
    # TEST 3: Volatility Prediction
    # ============================================
    print()
    print("=" * 70)
    print("TEST 3: Volatility Prediction (future_vol >= n * current_vol)")
    print("=" * 70)
    print(f"{'n':>5} | {'F1':>6} | {'Acc':>6} | {'High%':>6} | Status")
    print("-" * 70)

    for n_mult in [1.0, 1.5, 2.0, 3.0]:
        result = test_volatility_prediction(merged, feature_cols, n_mult, lookahead=20)
        if result is None:
            print(f"{n_mult:>5.1f} | {'N/A':>6} | {'N/A':>6} | Not enough data")
            continue

        status = "✓ Signal" if result['f1'] > 0.55 else "✗ Random"
        print(f"{n_mult:>5.1f}x | {result['f1']:>6.3f} | {result['accuracy']:>6.3f} | "
              f"{result['positive_ratio']*100:>5.1f}% | {status}")

        results[f'vol_{n_mult}x'] = result

    # ============================================
    # SUMMARY
    # ============================================
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Random Baselines:")
    print("  3-class: F1 = 0.33")
    print("  2-class: F1 = 0.50")
    print()

    # Find best results
    best_direction = max(
        [(k, v['f1']) for k, v in results.items() if k.startswith('direction_')],
        key=lambda x: x[1],
        default=None
    )
    best_binary = max(
        [(k, v['f1']) for k, v in results.items() if k.startswith('binary_')],
        key=lambda x: x[1],
        default=None
    )
    best_vol = max(
        [(k, v['f1']) for k, v in results.items() if k.startswith('vol_')],
        key=lambda x: x[1],
        default=None
    )

    print("Best Results:")
    if best_direction:
        status = "✓" if best_direction[1] > 0.40 else "✗"
        print(f"  Direction (3-class): {best_direction[0]} F1={best_direction[1]:.3f} {status}")
    if best_binary:
        status = "✓" if best_binary[1] > 0.55 else "✗"
        print(f"  Binary (Long/Short): {best_binary[0]} F1={best_binary[1]:.3f} {status}")
    if best_vol:
        status = "✓" if best_vol[1] > 0.55 else "✗"
        print(f"  Volatility:          {best_vol[0]} F1={best_vol[1]:.3f} {status}")

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    any_signal = any([
        best_direction and best_direction[1] > 0.40,
        best_binary and best_binary[1] > 0.55,
        best_vol and best_vol[1] > 0.55,
    ])

    if any_signal:
        print("✓ Some predictive signal found in features")
        print("→ Worth trying model training on the informative task")
    else:
        print("✗ No significant predictive signal in current features")
        print("→ Need feature engineering or different data sources")

    print()


if __name__ == "__main__":
    main()
