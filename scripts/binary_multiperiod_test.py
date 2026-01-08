#!/usr/bin/env python
"""
Binary Classification Multi-Period Validation (Chunk Streaming)

Tests Binary T=50 (Long vs Short) across multiple time periods.
Uses chunk-based streaming with sample capping to prevent OOM.

Memory Strategy:
- Process 1 month at a time
- Cap samples per period (max 300K)
- Use reservoir sampling for large periods
- Never hold more than 2 months in memory

Usage:
    python scripts/binary_multiperiod_test.py
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
T = 50
MULT = 1.0
TEST_RATIO = 0.2
PURGE_GAP = T
MAX_SAMPLES = 300000  # Cap to prevent OOM


def get_available_months():
    """Get list of available months."""
    files = sorted(FEATURES_DIR.glob("BTCUSDT-features-*.parquet"))
    months = []
    for f in files:
        parts = f.stem.split("-")
        if len(parts) >= 4:
            month = f"{parts[2]}-{parts[3]}"
            months.append(month)
    return sorted(set(months))


def get_period_months(period_name, available_months):
    """Get months for each period."""
    if period_name == "1mo":
        return ["2025-06"]
    elif period_name == "3mo":
        return [m for m in available_months if "2025-04" <= m <= "2025-06"]
    elif period_name == "6mo":
        return [m for m in available_months if "2025-01" <= m <= "2025-06"]
    elif period_name == "1yr":
        return [m for m in available_months if "2024-07" <= m <= "2025-06"]
    elif period_name == "2yr":
        return [m for m in available_months if "2023-07" <= m <= "2025-06"]
    return []


def process_month_chunk(month, feature_cols_ref=None):
    """
    Process single month in chunks and return X, y arrays.
    Memory efficient: loads, processes, returns numpy only.
    """
    feat_file = FEATURES_DIR / f"BTCUSDT-features-{month}.parquet"
    bar_file = BARS_DIR / f"BTCUSDT-bars-{month}.parquet"

    if not feat_file.exists():
        return None, None, None
    if not bar_file.exists():
        bar_file = BARS_DIR / "BTCUSDT" / f"BTCUSDT-bars-{month}.parquet"
        if not bar_file.exists():
            return None, None, None

    # Load with minimal columns
    features = pq.read_table(feat_file).to_pandas()
    bars = pq.read_table(bar_file).to_pandas()

    # Float32 conversion
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

    # Free immediately
    del features, bars
    gc.collect()

    if 'is_primed' in merged.columns:
        merged = merged[merged['is_primed']].reset_index(drop=True)

    # Feature columns
    exclude = {'start_time', 'end_time', 'open', 'high', 'high_time',
               'low', 'low_time', 'close', 'volume', 'is_primed'}
    feature_cols = [c for c in merged.columns if c not in exclude]

    if feature_cols_ref is not None:
        feature_cols = feature_cols_ref

    # Create labels
    config = TBMConfig(sl_mult=MULT, pt_mult=MULT, vertical_bars=T)
    labeler = TripleBarrierLabeler(config)
    result = labeler.label(merged)

    if len(result.valid_indices) == 0:
        del merged
        gc.collect()
        return None, None, feature_cols

    # Extract to numpy immediately
    X = np.array([merged[feature_cols].iloc[idx].values
                  for idx in result.valid_indices], dtype=np.float32)
    y = result.labels.astype(np.int8).copy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Free DataFrame
    del merged, result
    gc.collect()

    return X, y, feature_cols


def stream_and_sample(months, max_samples=MAX_SAMPLES):
    """
    Stream through months and sample if needed.
    Returns X, y with at most max_samples.
    """
    X_chunks = []
    y_chunks = []
    feature_cols = None
    total_samples = 0

    # First pass: count samples per month
    samples_per_month = []
    for month in months:
        X, y, fc = process_month_chunk(month, feature_cols)
        if X is None:
            samples_per_month.append(0)
            continue

        if feature_cols is None:
            feature_cols = fc

        # Binary only
        mask = y != 0
        n_binary = mask.sum()
        samples_per_month.append(n_binary)

        del X, y
        gc.collect()

    total_available = sum(samples_per_month)
    print(f"  Total available: {total_available:,} binary samples")

    # Calculate sampling rate
    if total_available > max_samples:
        sample_rate = max_samples / total_available
        print(f"  Sampling rate: {sample_rate:.2%} (capped at {max_samples:,})")
    else:
        sample_rate = 1.0

    # Second pass: collect with sampling
    np.random.seed(42)
    for i, month in enumerate(months):
        if samples_per_month[i] == 0:
            print(f"    {month}: skip")
            continue

        X, y, _ = process_month_chunk(month, feature_cols)
        if X is None:
            continue

        # Binary only
        mask = y != 0
        X = X[mask]
        y = y[mask]

        # Sample if needed
        if sample_rate < 1.0:
            n_keep = max(int(len(X) * sample_rate), 1)
            indices = np.random.choice(len(X), n_keep, replace=False)
            indices = np.sort(indices)  # Keep chronological order
            X = X[indices]
            y = y[indices]

        X_chunks.append(X)
        y_chunks.append(y)
        total_samples += len(X)

        print(f"    {month}: {len(X):,} samples (total: {total_samples:,})")

        del X, y
        gc.collect()

        # Early exit if enough
        if total_samples >= max_samples:
            print(f"  Reached sample limit")
            break

    if not X_chunks:
        return None, None, None

    # Concatenate
    X_all = np.concatenate(X_chunks, axis=0)
    y_all = np.concatenate(y_chunks, axis=0)

    del X_chunks, y_chunks
    gc.collect()

    return X_all, y_all, feature_cols


def test_period(months, period_name):
    """Test single period with streaming."""
    print(f"\n  Processing {len(months)} months...")

    X_all, y_all, feature_cols = stream_and_sample(months)

    if X_all is None or len(X_all) < 200:
        return None

    print(f"  Final: {len(X_all):,} samples, {X_all.nbytes / 1024**2:.1f} MB")

    # Stats
    n_short = np.sum(y_all == -1)
    n_long = np.sum(y_all == 1)

    # Chronological split
    split_idx = int(len(X_all) * (1 - TEST_RATIO))
    gap = min(PURGE_GAP, split_idx // 2)

    X_tr = X_all[:split_idx - gap]
    y_tr = y_all[:split_idx - gap]
    X_te = X_all[split_idx:]
    y_te = y_all[split_idx:]

    del X_all, y_all
    gc.collect()

    if len(X_tr) < 50 or len(X_te) < 20:
        return None

    print(f"  Train: {len(X_tr):,}, Test: {len(X_te):,}")

    # Train lightweight RF
    rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=2)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)

    f1 = f1_score(y_te, y_pred, average='macro')
    acc = accuracy_score(y_te, y_pred)

    # Feature importance
    top_feat = []
    if feature_cols:
        imp = rf.feature_importances_
        top_feat = sorted(zip(feature_cols, imp), key=lambda x: -x[1])[:5]

    del rf, X_tr, X_te, y_tr, y_te
    gc.collect()

    return {
        'f1_macro': float(f1),
        'accuracy': float(acc),
        'n_samples': int(n_short + n_long),
        'n_short': int(n_short),
        'n_long': int(n_long),
        'short_ratio': float(n_short / (n_short + n_long)),
        'long_ratio': float(n_long / (n_short + n_long)),
        'top_features': [(f, float(i)) for f, i in top_feat],
    }


def main():
    print("=" * 70)
    print("Binary Classification Multi-Period Validation")
    print("(Chunk Streaming with Sample Capping)")
    print("=" * 70)
    print()
    print(f"Config: T={T}, mult={MULT}, max_samples={MAX_SAMPLES:,}")
    print()

    available = get_available_months()
    print(f"Available: {len(available)} months ({available[0]} ~ {available[-1]})")

    results = {}

    for period in ["1mo", "3mo", "6mo", "1yr", "2yr"]:
        print()
        print("=" * 70)
        print(f"Period: {period}")
        print("=" * 70)

        months = get_period_months(period, available)
        if not months:
            print("  No data")
            results[period] = None
            continue

        print(f"  Range: {months[0]} ~ {months[-1]} ({len(months)} months)")

        result = test_period(months, period)
        results[period] = result

        if result:
            status = "SIGNAL" if result['f1_macro'] > 0.55 else "random"
            print(f"\n  => F1: {result['f1_macro']:.4f} | Acc: {result['accuracy']:.4f} | [{status}]")
        else:
            print("  => Test failed")

        gc.collect()

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Period':>8} | {'Samples':>10} | {'F1':>7} | {'Acc':>7} | Status")
    print("-" * 55)

    for period in ["1mo", "3mo", "6mo", "1yr", "2yr"]:
        r = results.get(period)
        if r is None:
            print(f"{period:>8} | {'N/A':>10} | {'N/A':>7} | {'N/A':>7} | No data")
        else:
            status = "SIGNAL" if r['f1_macro'] > 0.55 else "random"
            print(f"{period:>8} | {r['n_samples']:>10,} | {r['f1_macro']:>7.4f} | {r['accuracy']:>7.4f} | {status}")

    # Consistency check
    valid = [r for r in results.values() if r is not None]
    if len(valid) >= 2:
        f1s = [r['f1_macro'] for r in valid]
        print()
        print(f"F1 Mean: {np.mean(f1s):.4f}, Std: {np.std(f1s):.4f}")

        if np.mean(f1s) > 0.55 and np.std(f1s) < 0.05:
            print("=> CONSISTENT SIGNAL across periods")
        elif np.mean(f1s) < 0.52:
            print("=> NO SIGNAL (random level)")
        else:
            print("=> MARGINAL/INCONSISTENT signal")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"binary_multiperiod_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({
            'config': {'T': T, 'mult': MULT, 'max_samples': MAX_SAMPLES},
            'results': results,
            'timestamp': timestamp
        }, f, indent=2)

    print()
    print(f"Results saved: {results_file}")
    print()


if __name__ == "__main__":
    main()
