#!/usr/bin/env python
"""
Walk-Forward Monthly Test (Realistic Bot Simulation)

Train on month M -> Predict on month M+1
Repeat for all consecutive months, then average F1.

If avg F1 >= 0.53, a monthly-retrained bot is viable.

Usage:
    python scripts/walk_forward_test.py
"""

import sys
import gc
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.label_optimizer.tbm import TripleBarrierLabeler, TBMConfig

# Paths
FEATURES_DIR = PROJECT_ROOT / "etl" / "data" / "features-24" / "futures" / "BTCUSDT"
BARS_DIR = PROJECT_ROOT / "etl" / "data" / "bars-24" / "futures" / "BTCUSDT"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Configuration
T = 10
MULT = 1.0
TRAIN_MONTHS = 12  # 1 year training window


def get_available_months():
    """Get sorted list of available months."""
    files = sorted(FEATURES_DIR.glob("BTCUSDT-features-*.parquet"))
    months = []
    for f in files:
        parts = f.stem.split("-")
        if len(parts) >= 4:
            month = f"{parts[2]}-{parts[3]}"
            months.append(month)
    return sorted(set(months))


def load_month_data(month):
    """Load and process single month data."""
    feat_file = FEATURES_DIR / f"BTCUSDT-features-{month}.parquet"
    bar_file = BARS_DIR / f"BTCUSDT-bars-{month}.parquet"

    if not feat_file.exists():
        return None, None, None
    if not bar_file.exists():
        bar_file = BARS_DIR / "BTCUSDT" / f"BTCUSDT-bars-{month}.parquet"
        if not bar_file.exists():
            return None, None, None

    # Load
    features = pq.read_table(feat_file).to_pandas()
    bars = pq.read_table(bar_file).to_pandas()

    # Float32
    for col in features.select_dtypes(include=[np.float64]).columns:
        features[col] = features[col].astype(np.float32)
    for col in bars.select_dtypes(include=[np.float64]).columns:
        bars[col] = bars[col].astype(np.float32)

    # Merge
    bars_cols = ['start_time', 'open', 'high', 'high_time', 'low', 'low_time', 'close', 'volume']
    feature_exclude = {'start_time', 'end_time', 'open', 'high', 'low', 'close'}
    feat_cols = ['start_time'] + [c for c in features.columns if c not in feature_exclude]

    merged = features[feat_cols].merge(bars[bars_cols], on='start_time', how='inner')
    merged = merged.sort_values('start_time').reset_index(drop=True)

    del features, bars
    gc.collect()

    if 'is_primed' in merged.columns:
        merged = merged[merged['is_primed']].reset_index(drop=True)

    # Feature columns
    exclude = {'start_time', 'end_time', 'open', 'high', 'high_time',
               'low', 'low_time', 'close', 'volume', 'is_primed'}
    feature_cols = [c for c in merged.columns if c not in exclude]

    # Create labels
    config = TBMConfig(sl_mult=MULT, pt_mult=MULT, vertical_bars=T)
    labeler = TripleBarrierLabeler(config)
    result = labeler.label(merged)

    if len(result.valid_indices) == 0:
        del merged
        gc.collect()
        return None, None, feature_cols

    # Extract X, y
    X = np.array([merged[feature_cols].iloc[idx].values
                  for idx in result.valid_indices], dtype=np.float32)
    y = result.labels.astype(np.int8).copy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    del merged, result
    gc.collect()

    return X, y, feature_cols


def walk_forward_step(train_months, test_month, feature_cols=None):
    """
    Train on train_months (list), predict on test_month.
    Returns metrics dict or None if failed.
    """
    # Load and concatenate train data
    X_train_list = []
    y_train_list = []

    for month in train_months:
        X, y, fc = load_month_data(month)
        if X is None:
            continue

        if feature_cols is None:
            feature_cols = fc

        # Binary only
        mask = y != 0
        X_train_list.append(X[mask])
        y_train_list.append(y[mask])

        del X, y
        gc.collect()

    if not X_train_list:
        return None

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    del X_train_list, y_train_list
    gc.collect()

    if len(X_train) < 100:
        del X_train, y_train
        gc.collect()
        return None

    # Load test data
    X_test, y_test, _ = load_month_data(test_month)
    if X_test is None:
        del X_train, y_train
        gc.collect()
        return None

    # Binary only
    mask_test = y_test != 0
    X_test = X_test[mask_test]
    y_test = y_test[mask_test]

    if len(X_test) < 50:
        del X_train, y_train, X_test, y_test
        gc.collect()
        return None

    # Scale features for MLP
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train MLP
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False
    )
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Metrics
    f1 = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)

    # Class distribution
    n_short_train = np.sum(y_train == -1)
    n_long_train = np.sum(y_train == 1)
    n_short_test = np.sum(y_test == -1)
    n_long_test = np.sum(y_test == 1)

    # Prediction distribution
    n_pred_short = np.sum(y_pred == -1)
    n_pred_long = np.sum(y_pred == 1)

    del model, X_train, y_train, X_test, y_test
    gc.collect()

    return {
        'f1': float(f1),
        'accuracy': float(acc),
        'train_samples': int(n_short_train + n_long_train),
        'test_samples': int(n_short_test + n_long_test),
        'train_short': int(n_short_train),
        'train_long': int(n_long_train),
        'test_short': int(n_short_test),
        'test_long': int(n_long_test),
        'pred_short': int(n_pred_short),
        'pred_long': int(n_pred_long),
    }


def main():
    print("=" * 70)
    print("Walk-Forward Monthly Test")
    print(f"Train on {TRAIN_MONTHS} months -> Predict on next month")
    print("=" * 70)
    print()
    print(f"Config: T={T}, mult={MULT}, train_window={TRAIN_MONTHS}mo, Binary (Long/Short only)")
    print()

    # Get available months
    all_months = get_available_months()
    print(f"Available: {len(all_months)} months ({all_months[0]} ~ {all_months[-1]})")

    # Filter to test range (need TRAIN_MONTHS before first test)
    # Test: 2024-07 ~ 2025-06, need training data from 2023-07
    test_months = [m for m in all_months if "2024-07" <= m <= "2025-06"]
    print(f"Test range: {test_months[0]} ~ {test_months[-1]} ({len(test_months)} months)")
    print()

    # Walk-forward pairs
    results = []

    print("=" * 70)
    print(f"{'Train Period':>25} -> {'Test':>10} | {'F1':>6} | {'Acc':>6} | {'Train':>7} | {'Test':>6} | Status")
    print("-" * 85)

    for test_month in test_months:
        # Find training months (TRAIN_MONTHS before test_month)
        test_idx = all_months.index(test_month)
        if test_idx < TRAIN_MONTHS:
            continue

        train_months = all_months[test_idx - TRAIN_MONTHS:test_idx]
        train_period = f"{train_months[0]}~{train_months[-1]}"

        result = walk_forward_step(train_months, test_month)

        if result is None:
            print(f"{train_period:>25} -> {test_month:>10} | {'N/A':>6} | {'N/A':>6} | Failed")
            continue

        status = "SIGNAL" if result['f1'] > 0.53 else "random"
        print(f"{train_period:>25} -> {test_month:>10} | {result['f1']:>6.3f} | {result['accuracy']:>6.3f} | "
              f"{result['train_samples']:>7} | {result['test_samples']:>6} | {status}")

        results.append({
            'train_months': train_period,
            'test_month': test_month,
            **result
        })

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    if not results:
        print("No valid results!")
        return

    f1_scores = [r['f1'] for r in results]
    acc_scores = [r['accuracy'] for r in results]

    print(f"Total Steps: {len(results)}")
    print()
    print(f"F1 Score:")
    print(f"  Mean:   {np.mean(f1_scores):.4f}")
    print(f"  Std:    {np.std(f1_scores):.4f}")
    print(f"  Min:    {np.min(f1_scores):.4f}")
    print(f"  Max:    {np.max(f1_scores):.4f}")
    print(f"  Median: {np.median(f1_scores):.4f}")
    print()
    print(f"Accuracy:")
    print(f"  Mean:   {np.mean(acc_scores):.4f}")
    print(f"  Std:    {np.std(acc_scores):.4f}")
    print()

    # Count signal months
    signal_count = sum(1 for f1 in f1_scores if f1 > 0.53)
    print(f"Signal Months (F1 > 0.53): {signal_count}/{len(f1_scores)} ({signal_count/len(f1_scores):.1%})")
    print()

    # Verdict
    print("-" * 70)
    print("VERDICT:")
    print("-" * 70)

    avg_f1 = np.mean(f1_scores)
    if avg_f1 >= 0.53:
        print(f"  AVG F1 = {avg_f1:.4f} >= 0.53")
        print("  => MONTHLY RETRAINED BOT IS VIABLE!")
        print("  => Proceed to implement production system")
    elif avg_f1 >= 0.50:
        print(f"  AVG F1 = {avg_f1:.4f} (0.50 ~ 0.53)")
        print("  => MARGINAL signal, needs improvement")
        print("  => Consider: more features, longer training window")
    else:
        print(f"  AVG F1 = {avg_f1:.4f} < 0.50")
        print("  => NO SIGNAL, random level")
        print("  => Need: different features or approach")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"walk_forward_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({
            'config': {'T': T, 'mult': MULT},
            'summary': {
                'n_steps': len(results),
                'f1_mean': float(np.mean(f1_scores)),
                'f1_std': float(np.std(f1_scores)),
                'f1_min': float(np.min(f1_scores)),
                'f1_max': float(np.max(f1_scores)),
                'acc_mean': float(np.mean(acc_scores)),
                'signal_months': signal_count,
            },
            'steps': results,
            'timestamp': timestamp
        }, f, indent=2)

    print()
    print(f"Results saved: {results_file}")
    print()


if __name__ == "__main__":
    main()
