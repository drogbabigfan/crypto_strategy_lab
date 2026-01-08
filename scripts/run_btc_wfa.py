#!/usr/bin/env python
"""
Run SSL + WFA Pipeline on BTC Data Only.

This script:
1. Loads BTC feature data
2. Trains SSL encoder (one-off)
3. Runs WFA loop with trained encoder

Usage:
    python scripts/run_btc_wfa.py
"""

import copy
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import random_split

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.ssl.train import SSLTrainer, SSLTrainingConfig
from research.ssl.model import SSLModel
from research.label_optimizer.tbm import TripleBarrierLabeler, TBMConfig
from research.trainer.model import PatchTSTClassifier
from research.trainer.dataset import FineTuningDataset
from research.trainer.train import FineTuningTrainer, FineTuningTrainingConfig
from research.wfa.engine import WFAEngine, WFAConfig, FoldResult
from research.wfa.signal_generator import SignalGenerator, SignalGeneratorConfig
from research.wfa.go_bridge import GoBridge, BacktestConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / "etl" / "data"
FEATURES_DIR = DATA_DIR / "features-24" / "futures" / "BTCUSDT"
BARS_DIR = DATA_DIR / "bars-24" / "futures" / "BTCUSDT"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results" / "wfa"
BACKTESTER_PATH = PROJECT_ROOT / "etl" / "bin" / "backtester"


def load_all_btc_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all available BTC data."""
    logger.info("Loading BTC data...")

    # Load features
    feature_files = sorted(FEATURES_DIR.glob("BTCUSDT-features-*.parquet"))
    features_dfs = []
    for f in feature_files:
        df = pq.read_table(f).to_pandas()
        features_dfs.append(df)
    features_df = pd.concat(features_dfs, ignore_index=True)

    # Load bars (with high_time/low_time)
    bar_files = []
    for bars_path in [BARS_DIR, BARS_DIR / "BTCUSDT"]:
        bar_files.extend(sorted(bars_path.glob("BTCUSDT-bars-*.parquet")))

    bars_dfs = []
    for f in bar_files:
        df = pq.read_table(f).to_pandas()
        bars_dfs.append(df)
    bars_df = pd.concat(bars_dfs, ignore_index=True)

    # Remove duplicates by start_time
    features_df = features_df.drop_duplicates(subset=["start_time"]).reset_index(drop=True)
    bars_df = bars_df.drop_duplicates(subset=["start_time"]).reset_index(drop=True)

    logger.info(f"Loaded {len(features_df)} feature rows, {len(bars_df)} bar rows")
    logger.info(f"Date range: {pd.to_datetime(features_df['start_time'].min(), unit='ms')} ~ "
                f"{pd.to_datetime(features_df['start_time'].max(), unit='ms')}")

    return features_df, bars_df


def merge_data(features_df: pd.DataFrame, bars_df: pd.DataFrame) -> pd.DataFrame:
    """Merge features with bars (for high_time/low_time)."""
    # Select needed columns from bars
    bars_cols = ["start_time", "open", "high", "high_time", "low", "low_time", "close", "volume"]
    bars_subset = bars_df[bars_cols].copy()

    # Get feature columns
    exclude = {"start_time", "end_time", "open", "high", "low", "close"}
    feature_cols = ["start_time"] + [c for c in features_df.columns if c not in exclude]
    features_subset = features_df[feature_cols].copy()

    # Merge
    merged = pd.merge(bars_subset, features_subset, on="start_time", how="inner")
    logger.info(f"Merged: {len(merged)} rows")

    return merged


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get feature column names for model input."""
    exclude = {"start_time", "end_time", "open", "high", "high_time",
               "low", "low_time", "close", "volume", "is_primed"}
    return [c for c in df.columns if c not in exclude]


def create_windows(data: np.ndarray, context_len: int) -> np.ndarray:
    """Create sliding windows for SSL training."""
    n_samples = len(data) - context_len + 1
    if n_samples <= 0:
        raise ValueError(f"Not enough data for context_len={context_len}")

    windows = np.zeros((n_samples, context_len, data.shape[1]), dtype=np.float32)
    for i in range(n_samples):
        windows[i] = data[i:i + context_len]

    return windows


def train_ssl_encoder(
    features: np.ndarray,
    config: SSLTrainingConfig,
    output_path: Path,
) -> SSLModel:
    """Train SSL encoder."""
    logger.info("="*60)
    logger.info("Stage 2: SSL Pre-training")
    logger.info("="*60)

    # Create windows
    logger.info(f"Creating windows (context_len={config.context_len})...")
    windows = create_windows(features, config.context_len)
    logger.info(f"Created {len(windows)} windows")

    # Train
    trainer = SSLTrainer(config)
    model = trainer.train_from_array(windows, str(output_path))

    logger.info(f"SSL encoder saved to: {output_path}")
    return model


def run_wfa_single_fold(
    fold_id: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    base_encoder,  # PatchTSTEncoder - will be deepcopied
    go_bridge: GoBridge,
    config: dict,
) -> FoldResult:
    """Run single WFA fold."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Fold {fold_id}")
    logger.info(f"  Train: {len(train_df)} bars")
    logger.info(f"  Test: {len(test_df)} bars")
    logger.info(f"{'='*60}")

    # Step 3.1: Label Optimization
    logger.info("\n[Step 3.1] Label Optimization...")
    tbm_config = TBMConfig(
        sl_mult=config["sl_mult"],
        pt_mult=config["pt_mult"],
        vertical_bars=config["max_hold_bars"],
    )
    labeler = TripleBarrierLabeler(tbm_config)
    label_result = labeler.label(train_df)

    logger.info(f"  Valid labels: {len(label_result.valid_indices)}")
    unique, counts = np.unique(label_result.labels, return_counts=True)
    logger.info(f"  Distribution: {dict(zip(unique, counts))}")

    if len(label_result.valid_indices) < 1000:
        logger.warning("  Insufficient labels, skipping fold")
        return FoldResult(
            fold_id=fold_id,
            train_start=str(pd.to_datetime(train_df["start_time"].min(), unit="ms").date()),
            train_end=str(pd.to_datetime(train_df["start_time"].max(), unit="ms").date()),
            test_start=str(pd.to_datetime(test_df["start_time"].min(), unit="ms").date()),
            test_end=str(pd.to_datetime(test_df["start_time"].max(), unit="ms").date()),
            n_train_samples=len(train_df),
            n_test_samples=len(test_df),
            metrics={"error": "insufficient_labels"},
        )

    # Step 3.2: Fine-tuning
    logger.info("\n[Step 3.2] Fine-tuning...")

    # Prepare training data
    train_features = train_df[feature_cols].values.astype(np.float32)
    train_features = np.nan_to_num(train_features, nan=0.0)

    # CRITICAL: Deepcopy encoder to prevent contamination across folds
    encoder_copy = copy.deepcopy(base_encoder)

    # Create classifier from deepcopied encoder
    classifier = PatchTSTClassifier(
        encoder=encoder_copy,
        n_classes=3,
        bottleneck_dim=256,
        dropout=0.3,
    )

    # Create dataset - labels are {-1, 0, 1}, dataset converts to {0, 1, 2}
    full_dataset = FineTuningDataset(
        features=train_features,
        valid_indices=label_result.valid_indices,
        labels=label_result.labels,  # Pass original labels, dataset converts
        context_len=config["context_len"],
    )

    # Split into train/val (90/10)
    val_ratio = 0.1
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    logger.info(f"  Training samples: {len(train_dataset)}, Validation: {len(val_dataset)}")

    # Check class distribution and decide on weighting
    class_dist = full_dataset.get_class_distribution()
    logger.info(f"  Class distribution: {class_dist}")

    # Don't use class weights if neutral class is nearly empty (causes training instability)
    neutral_count = class_dist.get(0, {}).get('count', 0)
    if neutral_count < 100:
        logger.info(f"  Skipping class weights (neutral count={neutral_count} < 100)")
        class_weights = None
    else:
        class_weights = full_dataset.get_class_weights()
        logger.info(f"  Class weights: {class_weights.numpy()}")

    # Train classifier - FREEZE ENCODER to prevent collapse
    trainer_config = FineTuningTrainingConfig(
        frozen_epochs=50,  # Keep encoder frozen for ALL epochs
        total_epochs=50,   # More epochs for head-only training
        batch_size=config.get("batch_size", 64),
        head_lr=1e-3,  # LR for head
        encoder_lr=0,  # Not used since always frozen
        device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        min_samples=500,
        min_loss_reduction=-1e10,  # Effectively disable divergence check
        use_class_weights=False,  # Disabled since neutral class is empty
    )
    trainer = FineTuningTrainer(trainer_config)

    classifier = trainer.train(
        model=classifier,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        class_weights=class_weights,
    )

    # Step 3.3: Signal Generation + Backtest
    logger.info("\n[Step 3.3] Signal Generation & Backtest...")

    # Generate signals
    test_features = test_df[feature_cols].values.astype(np.float32)
    test_features = np.nan_to_num(test_features, nan=0.0)

    generator = SignalGenerator(
        classifier,
        SignalGeneratorConfig(
            context_len=config["context_len"],
            batch_size=64,
        ),
    )
    _, probs = generator.generate_with_probs(test_features)

    # Threshold-based signals (not argmax!)
    # Class 0 = Short, Class 2 = Long
    # If Long prob > Short prob, go Long. Otherwise Short.
    signals = np.zeros(len(test_features), dtype=np.int8)
    for i in range(config["context_len"], len(test_features)):
        short_prob = probs[i, 0]
        long_prob = probs[i, 2]

        # Use relative threshold - need clear edge to trade
        if long_prob > short_prob * 1.1:  # Long needs 10% higher prob
            signals[i] = 1  # Long
        elif short_prob > long_prob * 1.1:  # Short needs 10% higher prob
            signals[i] = -1  # Short
        # else: stay neutral (0)

    # Signal stats
    unique, counts = np.unique(signals, return_counts=True)
    logger.info(f"  Signals: {dict(zip(unique, counts))}")

    # Debug: Show average prediction probabilities
    valid_probs = probs[config["context_len"]:]  # Exclude warmup
    if len(valid_probs) > 0:
        avg_probs = valid_probs.mean(axis=0)
        logger.info(f"  Avg probs [Short, Neutral, Long]: {avg_probs}")

    # Prepare backtest features
    n = len(test_df)
    bt_features = np.zeros((n, 9), dtype=np.float64)
    bt_features[:, 0] = test_df["start_time"].values
    bt_features[:, 1] = test_df["open"].values
    bt_features[:, 2] = test_df["high"].values
    bt_features[:, 3] = test_df["high_time"].values
    bt_features[:, 4] = test_df["low"].values
    bt_features[:, 5] = test_df["low_time"].values
    bt_features[:, 6] = test_df["close"].values
    bt_features[:, 7] = test_df["volume"].values
    bt_features[:, 8] = test_df["realized_vol"].values

    # Run backtest
    backtest_config = BacktestConfig(
        sl_mult=config["sl_mult"],
        pt_mult=config["pt_mult"],
        max_hold_bars=config["max_hold_bars"],
    )

    result = go_bridge.run_backtest_with_data(
        signals=signals,
        features=bt_features,
        config=backtest_config,
    )

    logger.info(f"\n  Results:")
    logger.info(f"    Trades: {result.total_trades}")
    logger.info(f"    Win Rate: {result.win_rate:.1%}")
    logger.info(f"    Avg PnL: {result.avg_pnl:.4%}")
    logger.info(f"    Total PnL: {result.total_pnl:.2%}")
    logger.info(f"    Sharpe: {result.sharpe_ratio:.2f}")

    return FoldResult(
        fold_id=fold_id,
        train_start=str(pd.to_datetime(train_df["start_time"].min(), unit="ms").date()),
        train_end=str(pd.to_datetime(train_df["start_time"].max(), unit="ms").date()),
        test_start=str(pd.to_datetime(test_df["start_time"].min(), unit="ms").date()),
        test_end=str(pd.to_datetime(test_df["start_time"].max(), unit="ms").date()),
        best_params={
            "sl": config["sl_mult"],
            "pt": config["pt_mult"],
            "time": config["max_hold_bars"],
        },
        n_train_samples=len(train_df),
        n_test_samples=len(test_df),
        metrics={
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "avg_pnl": result.avg_pnl,
            "total_pnl": result.total_pnl,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "profit_factor": result.profit_factor,
            "tp_count": result.tp_count,
            "sl_count": result.sl_count,
            "timeout_count": result.timeout_count,
        },
    )


def main():
    """Main entry point."""
    logger.info("="*60)
    logger.info("BTC WFA Pipeline")
    logger.info("="*60)

    # Check backtester
    if not BACKTESTER_PATH.exists():
        logger.error(f"Go backtester not found: {BACKTESTER_PATH}")
        logger.error("Build it with: cd etl && go build -o bin/backtester ./cmd/backtester")
        sys.exit(1)

    # Configuration
    config = {
        # SSL
        "context_len": 512,
        "ssl_epochs": 20,
        "ssl_batch_size": 64,

        # WFA
        "train_months": 6,
        "test_months": 1,

        # Label/Backtest
        "sl_mult": 2.0,
        "pt_mult": 2.5,
        "max_hold_bars": 100,

        # Fine-tuning
        "finetune_epochs": 10,
        "batch_size": 64,
        "learning_rate": 1e-4,

        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    logger.info(f"Device: {config['device']}")

    # Load data
    features_df, bars_df = load_all_btc_data()
    merged_df = merge_data(features_df, bars_df)

    # Filter primed rows
    if "is_primed" in merged_df.columns:
        merged_df = merged_df[merged_df["is_primed"]].reset_index(drop=True)
        logger.info(f"After primed filter: {len(merged_df)} rows")

    # Get feature columns
    feature_cols = get_feature_columns(merged_df)
    logger.info(f"Feature columns: {len(feature_cols)}")

    # Prepare features for SSL
    features = merged_df[feature_cols].values.astype(np.float32)
    features = np.nan_to_num(features, nan=0.0)

    # SSL Training (or load existing)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    encoder_path = ARTIFACTS_DIR / "ssl" / "foundation_encoder.pt"
    encoder_path.parent.mkdir(parents=True, exist_ok=True)

    if encoder_path.exists():
        logger.info("="*60)
        logger.info("Loading existing SSL encoder...")
        logger.info("="*60)
        # Load just the encoder (not full SSLModel)
        base_encoder = SSLModel.load_encoder(str(encoder_path))
        base_encoder = base_encoder.to(config["device"])
        logger.info(f"Loaded encoder from: {encoder_path}")
    else:
        ssl_config = SSLTrainingConfig(
            n_features=len(feature_cols),
            context_len=config["context_len"],
            epochs=config["ssl_epochs"],
            batch_size=config["ssl_batch_size"],
            device=config["device"],
            min_samples=10000,  # Lower threshold for testing
        )
        ssl_model = train_ssl_encoder(features, ssl_config, encoder_path)
        base_encoder = ssl_model.encoder

    # WFA Loop
    logger.info("\n" + "="*60)
    logger.info("Stage 3: Walk-Forward Analysis")
    logger.info("="*60)

    # Generate folds
    merged_df["timestamp"] = merged_df["start_time"]
    timestamps = pd.to_datetime(merged_df["timestamp"], unit="ms")

    train_months = config["train_months"]
    test_months = config["test_months"]

    start_date = timestamps.min()
    end_date = timestamps.max()

    folds = []
    fold_id = 0
    current = start_date + pd.DateOffset(months=train_months)

    while current + pd.DateOffset(months=test_months) <= end_date:
        train_start = current - pd.DateOffset(months=train_months)
        train_end = current
        test_start = current
        test_end = current + pd.DateOffset(months=test_months)

        folds.append({
            "fold_id": fold_id,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })

        fold_id += 1
        current += pd.DateOffset(months=1)  # Step by 1 month

    logger.info(f"Generated {len(folds)} folds")

    # Run WFA with early stopping
    go_bridge = GoBridge(str(BACKTESTER_PATH))
    results = []
    cumulative_pnl = 0.0
    NAV_STOP_THRESHOLD = -0.5  # -50%

    for fold in folds:
        # Split data
        train_mask = (timestamps >= fold["train_start"]) & (timestamps < fold["train_end"])
        test_mask = (timestamps >= fold["test_start"]) & (timestamps < fold["test_end"])

        train_df = merged_df[train_mask].copy()
        test_df = merged_df[test_mask].copy()

        if len(train_df) < 1000 or len(test_df) < 100:
            logger.warning(f"Fold {fold['fold_id']}: Insufficient data, skipping")
            continue

        result = run_wfa_single_fold(
            fold_id=fold["fold_id"],
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            base_encoder=base_encoder,
            go_bridge=go_bridge,
            config=config,
        )
        results.append(result)

        # Early stopping: check cumulative PnL
        if "error" not in result.metrics:
            cumulative_pnl += result.metrics.get("total_pnl", 0.0)
            logger.info(f"  [Cumulative PnL: {cumulative_pnl:.2%}]")

            if cumulative_pnl < NAV_STOP_THRESHOLD:
                logger.warning(f"STOPPING: Cumulative PnL {cumulative_pnl:.2%} < {NAV_STOP_THRESHOLD:.0%}")
                break

    # Summary
    logger.info("\n" + "="*60)
    logger.info("WFA SUMMARY")
    logger.info("="*60)

    valid_results = [r for r in results if "error" not in r.metrics]

    if valid_results:
        sharpes = [r.metrics["sharpe_ratio"] for r in valid_results]
        pnls = [r.metrics["total_pnl"] for r in valid_results]
        win_rates = [r.metrics["win_rate"] for r in valid_results]

        logger.info(f"\nTotal Folds: {len(results)}")
        logger.info(f"Valid Folds: {len(valid_results)}")
        logger.info(f"\nAggregate Statistics:")
        logger.info(f"  Sharpe Ratio:  {np.mean(sharpes):.2f} +/- {np.std(sharpes):.2f}")
        logger.info(f"  Win Rate:      {np.mean(win_rates):.1%} +/- {np.std(win_rates):.1%}")
        logger.info(f"  Total PnL:     {np.mean(pnls):.2%} +/- {np.std(pnls):.2%}")
        logger.info(f"  Cumulative:    {np.sum(pnls):.2%}")

        # Save results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        import json
        results_file = RESULTS_DIR / f"wfa_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)

        logger.info(f"\nResults saved to: {results_file}")
    else:
        logger.warning("No valid results!")

    logger.info("\n" + "="*60)
    logger.info("Pipeline Complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
