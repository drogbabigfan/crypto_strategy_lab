#!/usr/bin/env python
"""
Baseline Test: XGBoost/RandomForest on Raw Features

Purpose:
- If XGBoost F1 > 0.55 → PatchTST is the problem
- If XGBoost F1 ≈ 0.5 (random) → Features are non-informative

Usage:
    python scripts/baseline_test.py
"""

import sys
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# XGBoost (optional)
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed, using RandomForest only")

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.label_optimizer.tbm import TripleBarrierLabeler, TBMConfig

# Paths
DATA_DIR = PROJECT_ROOT / "etl" / "data"
FEATURES_DIR = DATA_DIR / "features-24" / "futures" / "BTCUSDT"
BARS_DIR = DATA_DIR / "bars-24" / "futures" / "BTCUSDT"

# Memory-safe: Only use recent 1 year of data
USE_RECENT_YEARS = 1  # Set to None to use all data


def load_data():
    """Load and merge data with memory optimization."""
    print("Loading data...")

    # Get file lists
    feature_files = sorted(FEATURES_DIR.glob("BTCUSDT-features-*.parquet"))

    # Filter to recent years if specified (memory safe)
    if USE_RECENT_YEARS is not None:
        cutoff_year = 2025 - USE_RECENT_YEARS + 1  # e.g., 2024 for 2 years
        feature_files = [f for f in feature_files if int(f.stem.split('-')[2]) >= cutoff_year]
        print(f"Using data from {cutoff_year} onwards ({len(feature_files)} files)")

    # Load features in chunks for memory safety
    features_list = []
    for f in feature_files:
        df = pq.read_table(f).to_pandas()
        # Convert to float32 to save memory
        for col in df.select_dtypes(include=[np.float64]).columns:
            df[col] = df[col].astype(np.float32)
        features_list.append(df)
    features_df = pd.concat(features_list, ignore_index=True)
    del features_list
    gc.collect()

    # Load bars (same filter)
    bar_files = sorted(BARS_DIR.glob("BTCUSDT-bars-*.parquet"))

    if USE_RECENT_YEARS is not None:
        bar_files = [f for f in bar_files if int(f.stem.split('-')[2]) >= cutoff_year]

    bars_list = []
    for f in bar_files:
        df = pq.read_table(f).to_pandas()
        for col in df.select_dtypes(include=[np.float64]).columns:
            df[col] = df[col].astype(np.float32)
        bars_list.append(df)
    bars_df = pd.concat(bars_list, ignore_index=True)
    del bars_list
    gc.collect()

    # Remove duplicates
    features_df = features_df.drop_duplicates(subset=["start_time"]).reset_index(drop=True)
    bars_df = bars_df.drop_duplicates(subset=["start_time"]).reset_index(drop=True)

    # Merge - drop OHLC from features to avoid duplicates
    bars_cols = ["start_time", "open", "high", "high_time", "low", "low_time", "close", "volume"]
    feature_exclude = {"start_time", "end_time", "open", "high", "low", "close"}
    feature_cols_to_use = ["start_time"] + [c for c in features_df.columns if c not in feature_exclude]

    merged = pd.merge(
        bars_df[bars_cols],
        features_df[feature_cols_to_use],
        on="start_time",
        how="inner"
    )

    # Free memory
    del features_df, bars_df
    gc.collect()

    # Filter primed
    if "is_primed" in merged.columns:
        merged = merged[merged["is_primed"]].reset_index(drop=True)

    print(f"Loaded {len(merged):,} rows")
    print(f"Memory usage: {merged.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    return merged


def get_feature_columns(df):
    """Get feature column names."""
    exclude = {"start_time", "end_time", "open", "high", "high_time",
               "low", "low_time", "close", "volume", "is_primed"}
    return [c for c in df.columns if c not in exclude]


def create_labels(df, max_hold_bars=50, mult=1.0):
    """Create labels using TBM with sqrt(T) scaling."""
    config = TBMConfig(
        sl_mult=mult,
        pt_mult=mult,
        vertical_bars=max_hold_bars,
    )
    labeler = TripleBarrierLabeler(config)
    result = labeler.label(df)
    return result


def print_label_distribution(labels, name=""):
    """Print label distribution."""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    print(f"\n{'='*40}")
    print(f"Label Distribution {name}")
    print(f"{'='*40}")
    for label, count in zip(unique, counts):
        label_name = {-1: "Short", 0: "Neutral", 1: "Long"}.get(label, str(label))
        print(f"  {label_name}: {count:,} ({count/total:.1%})")
    print(f"  Total: {total:,}")

    # Check neutral ratio
    neutral_count = counts[unique == 0].sum() if 0 in unique else 0
    neutral_ratio = neutral_count / total
    print(f"\n  Neutral ratio: {neutral_ratio:.1%} {'✓' if neutral_ratio >= 0.15 else '✗ (need ≥15%)'}")


def prepare_ml_data(df, feature_cols, label_result, context_len=1):
    """Prepare data for ML (no context window, just current features)."""
    X_list = []
    y_list = []

    for i, (idx, label) in enumerate(zip(label_result.valid_indices, label_result.labels)):
        if idx < context_len:
            continue

        # Use current bar features only (no context window for baseline)
        features = df[feature_cols].iloc[idx].values
        X_list.append(features)
        y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)

    return X, y


def run_baseline_test(X, y, model_name, model, purge_gap=50):
    """Run baseline test with CHRONOLOGICAL split (no shuffle!)."""
    print(f"\n{'='*50}")
    print(f"Testing: {model_name}")
    print(f"{'='*50}")

    # CHRONOLOGICAL split (시계열 분할 - 셔플 금지!)
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx - purge_gap]  # Purging: label overlap 방지
    y_train = y[:split_idx - purge_gap]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    print(f"Train: {len(X_train):,} (older), Test: {len(X_test):,} (newer)")

    # Scale features for MLP
    if "MLP" in model_name:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Train
    print("Training...")
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")

    # Classification report
    print(f"\nClassification Report:")
    target_names = ["Short (-1)", "Neutral (0)", "Long (+1)"]
    # Filter target names based on actual classes
    unique_classes = np.unique(np.concatenate([y_test, y_pred]))
    filtered_names = [target_names[c + 1] for c in unique_classes if -1 <= c <= 1]
    print(classification_report(y_test, y_pred, target_names=filtered_names))

    # Feature importance (for tree-based models)
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_.copy()

    # Clean up to save memory
    del X_train, X_test, y_train, y_test, y_pred
    gc.collect()

    return feature_importance, f1_macro


def print_feature_importance(feature_cols, importances, top_n=15):
    """Print top feature importances."""
    if importances is None:
        return

    print(f"\n{'='*50}")
    print(f"Top {top_n} Feature Importances")
    print(f"{'='*50}")

    # Sort by importance
    indices = np.argsort(importances)[::-1]

    for i in range(min(top_n, len(feature_cols))):
        idx = indices[i]
        print(f"  {i+1:2}. {feature_cols[idx]:25} : {importances[idx]:.4f}")

    # Check if any feature is dominant
    top_importance = importances[indices[0]]
    if top_importance > 0.2:
        print(f"\n  ⚠️ Top feature has {top_importance:.1%} importance (possibly leaky)")


def main():
    print("="*60)
    print("Baseline Test: Feature Informativeness Check")
    print("="*60)

    # Load data
    df = load_data()
    feature_cols = get_feature_columns(df)
    print(f"Features: {len(feature_cols)}")
    print(f"Feature names: {feature_cols}")

    # Test: TBM with sqrt(T) scaling (mult=1.0, T=50)
    print("\n" + "="*60)
    print("TEST: TBM (mult=1.0, T=50)")
    print("="*60)

    label_result = create_labels(df, max_hold_bars=50, mult=1.0)
    print_label_distribution(label_result.labels, "(mult=1.0, T=50)")

    # Prepare data (3-class)
    X, y = prepare_ml_data(df, feature_cols, label_result)
    print(f"\nML Dataset: {X.shape[0]:,} samples, {X.shape[1]} features")

    # Check class balance
    unique_labels = np.unique(y)
    n_classes = len(unique_labels)
    print(f"Classes in dataset: {unique_labels} ({n_classes} classes)")

    # Models to test (memory-safe: limited n_jobs)
    N_JOBS = 4  # Limit parallel jobs to prevent memory explosion
    models = []

    # RandomForest
    models.append((
        "RandomForest",
        RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=50,
            random_state=42,
            n_jobs=N_JOBS,
        )
    ))

    # XGBoost
    if HAS_XGB:
        models.append((
            "XGBoost",
            xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=N_JOBS,
                eval_metric='mlogloss',
            )
        ))

    # Simple MLP
    models.append((
        "MLP (2-layer)",
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=100,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
    ))

    # Run tests
    results = {}
    feature_importance = None

    for name, model in models:
        importance, f1 = run_baseline_test(X, y, name, model)
        results[name] = f1
        if importance is not None and feature_importance is None:
            feature_importance = importance

    # Feature importance
    print_feature_importance(feature_cols, feature_importance)


    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nF1 Scores (macro):")
    for name, f1 in results.items():
        status = "✓ Informative" if f1 > 0.55 else "✗ Random-level" if f1 < 0.52 else "? Marginal"
        print(f"  {name:25}: {f1:.4f}  {status}")

    # Diagnosis
    best_f1 = max(results.values())
    print("\n" + "-"*60)
    print("DIAGNOSIS:")
    print("-"*60)

    if best_f1 > 0.55:
        print("✓ Features ARE informative (F1 > 0.55)")
        print("→ Problem is likely with PatchTST architecture or training")
        print("→ Try: Direct MLP/Transformer, different pretraining task")
    elif best_f1 > 0.52:
        print("? Features have MARGINAL informativeness (0.52 < F1 < 0.55)")
        print("→ Some signal exists but weak")
        print("→ Try: Better feature engineering, longer context window")
    else:
        print("✗ Features are NON-INFORMATIVE (F1 ≈ 0.50)")
        print("→ Current 29 features don't predict direction")
        print("→ Need: New features (order flow, funding rate, etc.)")

    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)


if __name__ == "__main__":
    main()
