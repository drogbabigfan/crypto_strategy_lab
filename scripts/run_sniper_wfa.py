#!/usr/bin/env python
"""
Sniper WFA Pipeline (v4.9)

End-to-End supervised learning for high-confidence entry prediction.

Usage:
    python scripts/run_sniper_wfa.py
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.sniper.labeler import SniperLabeler, SniperTBMConfig
from research.sniper.model import SniperClassifier, SniperModelConfig
from research.sniper.dataset import SniperDataset, train_val_split
from research.sniper.trainer import SniperTrainer, SniperTrainingConfig
from research.sniper.signal_generator import ConfidenceSignalGenerator
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
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "sniper"
RESULTS_DIR = PROJECT_ROOT / "results" / "sniper"
BACKTESTER_PATH = PROJECT_ROOT / "etl" / "bin" / "backtester"


@dataclass
class WFAConfig:
    """Walk-Forward Analysis configuration."""

    # Fold structure (in bars)
    train_bars: int = 10000       # ~1-2 months
    test_bars: int = 2000         # ~1 week
    step_bars: int = 2000         # Roll forward by test_bars

    # Early stopping
    max_cumulative_loss: float = -0.30  # Stop at -30%

    # Model
    context_len: int = 128
    confidence_threshold: float = 0.6

    # Sniper TBM (v5.0: symmetric barriers, longer holding)
    sl_mult: float = 1.0
    pt_mult: float = 1.0
    max_hold_bars: int = 50
    min_expected_return: float = 0.0045

    # Training
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load features and bars data."""
    logger.info("Loading data...")

    # Load features
    feature_files = sorted(FEATURES_DIR.glob("BTCUSDT-features-*.parquet"))
    features_dfs = [pq.read_table(f).to_pandas() for f in feature_files]
    features_df = pd.concat(features_dfs, ignore_index=True)

    # Load bars
    bar_files = []
    for path in [BARS_DIR, BARS_DIR / "BTCUSDT"]:
        bar_files.extend(sorted(path.glob("BTCUSDT-bars-*.parquet")))

    bars_dfs = [pq.read_table(f).to_pandas() for f in bar_files]
    bars_df = pd.concat(bars_dfs, ignore_index=True)

    # Remove duplicates
    features_df = features_df.drop_duplicates(subset=["start_time"]).reset_index(drop=True)
    bars_df = bars_df.drop_duplicates(subset=["start_time"]).reset_index(drop=True)

    logger.info(f"Loaded {len(features_df)} features, {len(bars_df)} bars")

    return features_df, bars_df


def merge_data(features_df: pd.DataFrame, bars_df: pd.DataFrame) -> pd.DataFrame:
    """Merge features with bars."""
    # Select columns from bars
    bars_cols = ["start_time", "open", "high", "high_time", "low", "low_time", "close", "volume"]
    bars_subset = bars_df[bars_cols].copy()

    # Get feature columns
    exclude = {"start_time", "end_time", "open", "high", "low", "close"}
    feature_cols = ["start_time"] + [c for c in features_df.columns if c not in exclude]
    features_subset = features_df[feature_cols].copy()

    # Merge
    merged = pd.merge(bars_subset, features_subset, on="start_time", how="inner")

    # Filter primed rows
    if "is_primed" in merged.columns:
        merged = merged[merged["is_primed"]].reset_index(drop=True)

    logger.info(f"Merged: {len(merged)} rows")

    return merged


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get feature column names."""
    exclude = {"start_time", "end_time", "open", "high", "high_time",
               "low", "low_time", "close", "volume", "is_primed"}
    return [c for c in df.columns if c not in exclude]


def run_fold(
    fold_id: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    go_bridge: GoBridge,
    config: WFAConfig,
) -> dict:
    """Run a single WFA fold."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Fold {fold_id}")
    logger.info(f"  Train: {len(train_df)} bars")
    logger.info(f"  Test: {len(test_df)} bars")
    logger.info(f"{'='*60}")

    # Step 1: Labeling
    logger.info("\n[Step 1] Sniper Labeling...")
    labeler = SniperLabeler(SniperTBMConfig(
        sl_mult=config.sl_mult,
        pt_mult=config.pt_mult,
        max_hold_bars=config.max_hold_bars,
        min_expected_return=config.min_expected_return,
    ))
    label_result = labeler.label(train_df)

    # Check label quality
    if label_result.stats["total"] < 1000:
        logger.warning("Insufficient labels, skipping fold")
        return {"error": "insufficient_labels"}

    # Step 2: Prepare data
    logger.info("\n[Step 2] Preparing data...")
    train_features = train_df[feature_cols].values.astype(np.float32)
    train_features = np.nan_to_num(train_features, nan=0.0)

    # Create dataset
    train_dataset, val_dataset = train_val_split(
        features=train_features,
        labels=label_result.labels,
        context_len=config.context_len,
        val_ratio=0.15,
    )

    # Step 3: Train model
    logger.info("\n[Step 3] Training model...")
    model = SniperClassifier(SniperModelConfig(
        n_features=len(feature_cols),
        context_len=config.context_len,
    ))

    trainer = SniperTrainer(SniperTrainingConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        device=config.device,
    ))

    model = trainer.train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    # Get training summary
    train_summary = trainer.get_training_summary()
    logger.info(f"  Best Val F1: {train_summary.get('best_val_f1', 0):.4f}")

    # Step 4: Generate signals
    logger.info("\n[Step 4] Generating signals...")
    test_features = test_df[feature_cols].values.astype(np.float32)
    test_features = np.nan_to_num(test_features, nan=0.0)

    generator = ConfidenceSignalGenerator(
        model=model,
        context_len=config.context_len,
        confidence_threshold=config.confidence_threshold,
        device=config.device,
    )
    signals, confidences, probs = generator.generate(test_features)

    # Step 5: Backtest
    logger.info("\n[Step 5] Backtesting...")

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

    backtest_config = BacktestConfig(
        sl_mult=config.sl_mult,
        pt_mult=config.pt_mult,
        max_hold_bars=config.max_hold_bars,
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

    return {
        "fold_id": fold_id,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_df),
        "label_stats": label_result.stats,
        "train_summary": train_summary,
        "backtest": {
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "avg_pnl": result.avg_pnl,
            "total_pnl": result.total_pnl,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "profit_factor": result.profit_factor,
        },
    }


def main():
    """Main entry point."""
    logger.info("="*60)
    logger.info("Sniper WFA Pipeline (v4.9)")
    logger.info("="*60)

    # Check backtester
    if not BACKTESTER_PATH.exists():
        logger.error(f"Go backtester not found: {BACKTESTER_PATH}")
        sys.exit(1)

    # Configuration
    config = WFAConfig()
    logger.info(f"Device: {config.device}")
    logger.info(f"Context len: {config.context_len}")
    logger.info(f"Confidence threshold: {config.confidence_threshold}")

    # Load data
    features_df, bars_df = load_data()
    merged_df = merge_data(features_df, bars_df)
    feature_cols = get_feature_columns(merged_df)
    logger.info(f"Features: {len(feature_cols)}")

    # Create directories
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate folds
    n_total = len(merged_df)
    folds = []
    fold_id = 0

    start_idx = 0
    while start_idx + config.train_bars + config.test_bars <= n_total:
        train_end = start_idx + config.train_bars
        test_end = train_end + config.test_bars

        folds.append({
            "fold_id": fold_id,
            "train_start": start_idx,
            "train_end": train_end,
            "test_start": train_end,
            "test_end": test_end,
        })

        fold_id += 1
        start_idx += config.step_bars

    logger.info(f"Generated {len(folds)} folds")

    # Run WFA
    go_bridge = GoBridge(str(BACKTESTER_PATH))
    results = []
    cumulative_pnl = 0.0

    for fold in folds:
        train_df = merged_df.iloc[fold["train_start"]:fold["train_end"]].copy()
        test_df = merged_df.iloc[fold["test_start"]:fold["test_end"]].copy()

        result = run_fold(
            fold_id=fold["fold_id"],
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            go_bridge=go_bridge,
            config=config,
        )
        results.append(result)

        # Track cumulative PnL
        if "error" not in result:
            pnl = result["backtest"]["total_pnl"]
            cumulative_pnl += pnl
            logger.info(f"  [Cumulative PnL: {cumulative_pnl:.2%}]")

            if cumulative_pnl < config.max_cumulative_loss:
                logger.warning(f"STOPPING: Cumulative PnL {cumulative_pnl:.2%} < {config.max_cumulative_loss:.0%}")
                break

    # Summary
    logger.info("\n" + "="*60)
    logger.info("WFA SUMMARY")
    logger.info("="*60)

    valid_results = [r for r in results if "error" not in r]

    if valid_results:
        sharpes = [r["backtest"]["sharpe_ratio"] for r in valid_results]
        pnls = [r["backtest"]["total_pnl"] for r in valid_results]
        win_rates = [r["backtest"]["win_rate"] for r in valid_results]
        val_f1s = [r["train_summary"].get("best_val_f1", 0) for r in valid_results]

        logger.info(f"\nTotal Folds: {len(results)}")
        logger.info(f"Valid Folds: {len(valid_results)}")
        logger.info(f"\nModel Performance:")
        logger.info(f"  Val F1:        {np.mean(val_f1s):.4f} +/- {np.std(val_f1s):.4f}")
        logger.info(f"\nBacktest Performance:")
        logger.info(f"  Sharpe Ratio:  {np.mean(sharpes):.2f} +/- {np.std(sharpes):.2f}")
        logger.info(f"  Win Rate:      {np.mean(win_rates):.1%} +/- {np.std(win_rates):.1%}")
        logger.info(f"  Total PnL:     {np.mean(pnls):.2%} +/- {np.std(pnls):.2%}")
        logger.info(f"  Cumulative:    {np.sum(pnls):.2%}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = RESULTS_DIR / f"sniper_wfa_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"\nResults saved to: {results_file}")
    else:
        logger.warning("No valid results!")

    logger.info("\n" + "="*60)
    logger.info("Pipeline Complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
