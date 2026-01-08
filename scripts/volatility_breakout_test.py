#!/usr/bin/env python
"""
Volatility Breakout Walk-Forward Test

Predicts if future volatility will "breakout" (Range >= 2 * ATR)
instead of predicting direction.

Label:
- 1 (Active): Future 20-bar range >= 2 * current ATR
- 0 (Quiet): Sideways, low volatility

Usage:
    python scripts/volatility_breakout_test.py
"""

import sys
import gc
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score, accuracy_score, classification_report
import xgboost as xgb

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
FEATURES_DIR = PROJECT_ROOT / "etl" / "data" / "features-24" / "futures" / "BTCUSDT"
BARS_DIR = PROJECT_ROOT / "etl" / "data" / "bars-24" / "futures" / "BTCUSDT"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Configuration
LOOKAHEAD = 20       # Future bars to measure range
BREAKOUT_K = 1.3     # Range >= sqrt(T) * K * ATR → Active (above normal)
ATR_PERIOD = 14      # ATR calculation period
TRAIN_MONTHS = 36    # Training window (3 years)

# sqrt(20) ≈ 4.47, so threshold = 4.47 * 1.5 * ATR ≈ 6.7 * ATR


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
    """Load single month data."""
    feat_file = FEATURES_DIR / f"BTCUSDT-features-{month}.parquet"
    bar_file = BARS_DIR / f"BTCUSDT-bars-{month}.parquet"

    if not feat_file.exists():
        return None, None
    if not bar_file.exists():
        bar_file = BARS_DIR / "BTCUSDT" / f"BTCUSDT-bars-{month}.parquet"
        if not bar_file.exists():
            return None, None

    # Load
    features = pq.read_table(feat_file).to_pandas()
    bars = pq.read_table(bar_file).to_pandas()

    # Float32
    for col in features.select_dtypes(include=[np.float64]).columns:
        features[col] = features[col].astype(np.float32)
    for col in bars.select_dtypes(include=[np.float64]).columns:
        bars[col] = bars[col].astype(np.float32)

    # Merge
    bars_cols = ['start_time', 'open', 'high', 'low', 'close', 'volume']
    feature_exclude = {'start_time', 'end_time', 'open', 'high', 'low', 'close'}
    feat_cols = ['start_time'] + [c for c in features.columns if c not in feature_exclude]

    merged = features[feat_cols].merge(bars[bars_cols], on='start_time', how='inner')
    merged = merged.sort_values('start_time').reset_index(drop=True)

    del features, bars
    gc.collect()

    if 'is_primed' in merged.columns:
        merged = merged[merged['is_primed']].reset_index(drop=True)

    # Feature columns
    exclude = {'start_time', 'end_time', 'open', 'high', 'low', 'close', 'volume', 'is_primed'}
    feature_cols = [c for c in merged.columns if c not in exclude]

    return merged, feature_cols


def calculate_atr(df, period=14):
    """Calculate ATR (Average True Range)."""
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    tr = np.zeros(len(df))
    tr[0] = high[0] - low[0]

    for i in range(1, len(df)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )

    # Simple moving average of TR
    atr = np.zeros(len(df))
    for i in range(len(df)):
        if i < period:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = np.mean(tr[i-period+1:i+1])

    return atr


def create_volatility_labels(df, feature_cols, lookahead=20, breakout_k=1.5):
    """
    Create volatility breakout labels with sqrt(T) scaling.

    Expected range after T bars = ATR * sqrt(T)
    Breakout = Range >= sqrt(T) * K * ATR (above normal)

    Label = 1 (Active/Breakout) if future range >= sqrt(T) * K * ATR
    Label = 0 (Quiet/Normal) otherwise
    """
    import math
    atr = calculate_atr(df, period=ATR_PERIOD)
    sqrt_T = math.sqrt(lookahead)

    X_list = []
    y_list = []

    for i in range(ATR_PERIOD, len(df) - lookahead):
        # Current ATR
        current_atr = atr[i]
        if current_atr <= 0 or np.isnan(current_atr):
            continue

        # Future range (max high - min low in next lookahead bars)
        future_high = df['high'].iloc[i+1:i+1+lookahead].max()
        future_low = df['low'].iloc[i+1:i+1+lookahead].min()
        future_range = future_high - future_low

        # Threshold with sqrt(T) scaling
        # Expected normal range = ATR * sqrt(T)
        # Breakout threshold = ATR * sqrt(T) * K
        threshold = current_atr * sqrt_T * breakout_k

        # Label: 1 if range >= threshold (breakout)
        label = 1 if future_range >= threshold else 0

        # Features
        X_list.append(df[feature_cols].iloc[i].values)
        y_list.append(label)

    if not X_list:
        return None, None

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int8)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y


def load_and_label_months(months):
    """Load multiple months and create labels."""
    X_list = []
    y_list = []
    feature_cols = None

    for month in months:
        df, fc = load_month_data(month)
        if df is None:
            continue

        if feature_cols is None:
            feature_cols = fc

        X, y = create_volatility_labels(df, feature_cols, LOOKAHEAD, BREAKOUT_K)

        del df
        gc.collect()

        if X is None:
            continue

        X_list.append(X)
        y_list.append(y)

        del X, y
        gc.collect()

    if not X_list:
        return None, None, None

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    del X_list, y_list
    gc.collect()

    return X_all, y_all, feature_cols


def walk_forward_step(train_months, test_month, feature_cols=None):
    """Train on train_months, predict on test_month."""

    # Load train data
    X_train, y_train, fc = load_and_label_months(train_months)
    if X_train is None or len(X_train) < 100:
        return None

    if feature_cols is None:
        feature_cols = fc

    # Load test data
    X_test, y_test, _ = load_and_label_months([test_month])
    if X_test is None or len(X_test) < 50:
        del X_train, y_train
        gc.collect()
        return None

    # Class distribution
    train_active = np.sum(y_train == 1)
    train_quiet = np.sum(y_train == 0)
    test_active = np.sum(y_test == 1)
    test_quiet = np.sum(y_test == 0)

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=2,
        eval_metric='logloss',
        verbosity=0
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    f1 = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)

    # Per-class F1
    f1_per_class = f1_score(y_test, y_pred, average=None)

    del model, X_train, y_train, X_test, y_test
    gc.collect()

    return {
        'f1': float(f1),
        'accuracy': float(acc),
        'f1_quiet': float(f1_per_class[0]) if len(f1_per_class) > 0 else 0,
        'f1_active': float(f1_per_class[1]) if len(f1_per_class) > 1 else 0,
        'train_samples': int(train_active + train_quiet),
        'test_samples': int(test_active + test_quiet),
        'train_active_ratio': float(train_active / (train_active + train_quiet)),
        'test_active_ratio': float(test_active / (test_active + test_quiet)),
    }


def main():
    print("=" * 70)
    print("Volatility Breakout Walk-Forward Test")
    print("=" * 70)
    print()
    import math
    sqrt_T = math.sqrt(LOOKAHEAD)
    effective_mult = sqrt_T * BREAKOUT_K

    print(f"Config:")
    print(f"  Lookahead: {LOOKAHEAD} bars")
    print(f"  ATR Period: {ATR_PERIOD}")
    print(f"  Breakout K: {BREAKOUT_K} (above sqrt(T) normal)")
    print(f"  Effective Threshold: sqrt({LOOKAHEAD}) * {BREAKOUT_K} = {effective_mult:.2f}x ATR")
    print(f"  Training Window: {TRAIN_MONTHS} months")
    print(f"  Model: XGBoost")
    print()

    # Get available months
    all_months = get_available_months()
    print(f"Available: {len(all_months)} months ({all_months[0]} ~ {all_months[-1]})")

    # Test range
    test_months = [m for m in all_months if "2024-07" <= m <= "2025-06"]
    print(f"Test range: {test_months[0]} ~ {test_months[-1]} ({len(test_months)} months)")
    print()

    results = []

    print("=" * 85)
    print(f"{'Train Period':>25} -> {'Test':>10} | {'F1':>6} | {'Acc':>6} | {'Active%':>7} | Status")
    print("-" * 85)

    for test_month in test_months:
        test_idx = all_months.index(test_month)
        if test_idx < TRAIN_MONTHS:
            continue

        train_months = all_months[test_idx - TRAIN_MONTHS:test_idx]
        train_period = f"{train_months[0]}~{train_months[-1]}"

        result = walk_forward_step(train_months, test_month)

        if result is None:
            print(f"{train_period:>25} -> {test_month:>10} | {'N/A':>6} | Failed")
            continue

        status = "SIGNAL" if result['f1'] > 0.55 else "random"
        print(f"{train_period:>25} -> {test_month:>10} | {result['f1']:>6.3f} | "
              f"{result['accuracy']:>6.3f} | {result['test_active_ratio']*100:>6.1f}% | {status}")

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
    active_ratios = [r['test_active_ratio'] for r in results]

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
    print()
    print(f"Active Ratio (test):")
    print(f"  Mean:   {np.mean(active_ratios)*100:.1f}%")
    print()

    signal_count = sum(1 for f1 in f1_scores if f1 > 0.55)
    print(f"Signal Months (F1 > 0.55): {signal_count}/{len(f1_scores)} ({signal_count/len(f1_scores):.1%})")
    print()

    # Verdict
    print("-" * 70)
    print("VERDICT:")
    print("-" * 70)

    avg_f1 = np.mean(f1_scores)
    if avg_f1 >= 0.60:
        print(f"  AVG F1 = {avg_f1:.4f} >= 0.60")
        print("  => STRONG SIGNAL! Volatility breakout is predictable")
        print("  => Proceed with straddle/breakout strategy")
    elif avg_f1 >= 0.55:
        print(f"  AVG F1 = {avg_f1:.4f} >= 0.55")
        print("  => GOOD SIGNAL! Worth implementing")
    elif avg_f1 >= 0.50:
        print(f"  AVG F1 = {avg_f1:.4f} (0.50 ~ 0.55)")
        print("  => MARGINAL signal")
    else:
        print(f"  AVG F1 = {avg_f1:.4f} < 0.50")
        print("  => NO SIGNAL")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"volatility_breakout_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'lookahead': LOOKAHEAD,
                'breakout_k': BREAKOUT_K,
                'effective_mult': float(effective_mult),
                'atr_period': ATR_PERIOD,
                'train_months': TRAIN_MONTHS,
            },
            'summary': {
                'n_steps': len(results),
                'f1_mean': float(np.mean(f1_scores)),
                'f1_std': float(np.std(f1_scores)),
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
