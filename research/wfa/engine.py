"""
WFA Engine - Walk-Forward Analysis orchestrator.

Controls the entire WFA loop:
1. Split data into Train/Test windows
2. For each fold:
   a. Optimize labels (Step 3.1)
   b. Fine-tune model (Step 3.2)
   c. Generate signals
   d. Run Go backtester (Step 3.3)
3. Aggregate and report results
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .go_bridge import GoBridge, BacktestConfig, BacktestResult
from .signal_generator import SignalGenerator, SignalGeneratorConfig


@dataclass
class WFAConfig:
    """Configuration for Walk-Forward Analysis."""

    # Window sizes (in months)
    train_months: int = 12
    test_months: int = 1
    step_months: int = 1

    # Minimum requirements
    min_train_bars: int = 30000
    min_test_bars: int = 1000

    # Model settings
    context_len: int = 512

    # Paths
    encoder_path: str = "./artifacts/ssl/foundation_encoder.pt"
    output_dir: str = "./results/wfa"

    # Go backtester
    backtester_path: str = "./etl/bin/backtester"


@dataclass
class FoldResult:
    """Result from a single WFA fold."""

    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    # Optimization results
    best_params: dict = field(default_factory=dict)
    n_train_samples: int = 0
    n_test_samples: int = 0

    # Backtest metrics
    metrics: dict = field(default_factory=dict)

    # Signal stats
    signal_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)


@dataclass
class Fold:
    """A single train/test fold."""

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class WFAEngine:
    """
    Walk-Forward Analysis orchestrator.

    Example:
        engine = WFAEngine(config)
        results = engine.run(
            features_df=features,
            label_optimizer=optimizer,
            trainer=trainer,
        )
    """

    def __init__(self, config: WFAConfig):
        """
        Initialize WFA engine.

        Args:
            config: WFA configuration
        """
        self.config = config
        self.go_bridge = GoBridge(config.backtester_path)
        self.results: list[FoldResult] = []

    def generate_folds(
        self,
        data: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> list[Fold]:
        """
        Generate train/test folds from data.

        Args:
            data: DataFrame with timestamp column
            timestamp_col: Name of timestamp column

        Returns:
            List of Fold objects
        """
        # Convert to datetime if needed
        if data[timestamp_col].dtype == np.int64:
            timestamps = pd.to_datetime(data[timestamp_col], unit="ms")
        else:
            timestamps = pd.to_datetime(data[timestamp_col])

        start_date = timestamps.min()
        end_date = timestamps.max()

        folds = []
        fold_id = 0

        # Start from beginning + train_months
        current = start_date + pd.DateOffset(months=self.config.train_months)

        while current + pd.DateOffset(months=self.config.test_months) <= end_date:
            train_start = current - pd.DateOffset(months=self.config.train_months)
            train_end = current
            test_start = current
            test_end = current + pd.DateOffset(months=self.config.test_months)

            folds.append(Fold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            ))

            fold_id += 1
            current += pd.DateOffset(months=self.config.step_months)

        return folds

    def run(
        self,
        features_df: pd.DataFrame,
        label_optimizer: Callable,
        trainer: Callable,
        feature_cols: list[str],
        timestamp_col: str = "timestamp",
        price_cols: Optional[dict] = None,
    ) -> list[FoldResult]:
        """
        Execute full WFA loop.

        Args:
            features_df: DataFrame with features and price data
            label_optimizer: Function to optimize labels (returns best_params, labels)
            trainer: Function to train classifier (returns trained model)
            feature_cols: List of feature column names
            timestamp_col: Timestamp column name
            price_cols: Dict mapping to price columns for backtester
                       e.g., {"open": "open", "high": "high", ...}

        Returns:
            List of FoldResult objects
        """
        if price_cols is None:
            price_cols = {
                "open": "open",
                "high": "high",
                "high_time": "high_time",
                "low": "low",
                "low_time": "low_time",
                "close": "close",
                "volume": "volume",
                "realized_vol": "realized_vol",
            }

        # Generate folds
        folds = self.generate_folds(features_df, timestamp_col)

        if len(folds) == 0:
            raise ValueError(
                f"No valid folds generated. Data range may be too short. "
                f"Need at least {self.config.train_months + self.config.test_months} months."
            )

        print(f"\n{'='*60}")
        print(f"WFA Engine: {len(folds)} folds")
        print(f"Train: {self.config.train_months}M, Test: {self.config.test_months}M")
        print(f"{'='*60}")

        self.results = []

        for fold in folds:
            result = self._process_fold(
                fold=fold,
                features_df=features_df,
                label_optimizer=label_optimizer,
                trainer=trainer,
                feature_cols=feature_cols,
                timestamp_col=timestamp_col,
                price_cols=price_cols,
            )
            self.results.append(result)

            # Print summary
            print(f"\nFold {fold.fold_id} Complete:")
            print(f"  Sharpe: {result.metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  Win Rate: {result.metrics.get('win_rate', 0):.1%}")
            print(f"  Total PnL: {result.metrics.get('total_pnl', 0):.2%}")

        # Save results
        self._save_results()

        return self.results

    def _process_fold(
        self,
        fold: Fold,
        features_df: pd.DataFrame,
        label_optimizer: Callable,
        trainer: Callable,
        feature_cols: list[str],
        timestamp_col: str,
        price_cols: dict,
    ) -> FoldResult:
        """Process a single WFA fold."""
        print(f"\n{'='*60}")
        print(f"Fold {fold.fold_id}")
        print(f"  Train: {fold.train_start.date()} ~ {fold.train_end.date()}")
        print(f"  Test:  {fold.test_start.date()} ~ {fold.test_end.date()}")
        print(f"{'='*60}")

        # Convert timestamps for filtering
        if features_df[timestamp_col].dtype == np.int64:
            ts = pd.to_datetime(features_df[timestamp_col], unit="ms")
        else:
            ts = pd.to_datetime(features_df[timestamp_col])

        # Split data
        train_mask = (ts >= fold.train_start) & (ts < fold.train_end)
        test_mask = (ts >= fold.test_start) & (ts < fold.test_end)

        train_df = features_df[train_mask].copy()
        test_df = features_df[test_mask].copy()

        n_train = len(train_df)
        n_test = len(test_df)

        print(f"  Train samples: {n_train:,}")
        print(f"  Test samples:  {n_test:,}")

        # Check minimum requirements
        if n_train < self.config.min_train_bars:
            print(f"  WARNING: Insufficient train data ({n_train} < {self.config.min_train_bars})")
            return FoldResult(
                fold_id=fold.fold_id,
                train_start=str(fold.train_start.date()),
                train_end=str(fold.train_end.date()),
                test_start=str(fold.test_start.date()),
                test_end=str(fold.test_end.date()),
                n_train_samples=n_train,
                n_test_samples=n_test,
                metrics={"error": "insufficient_train_data"},
            )

        if n_test < self.config.min_test_bars:
            print(f"  WARNING: Insufficient test data ({n_test} < {self.config.min_test_bars})")
            return FoldResult(
                fold_id=fold.fold_id,
                train_start=str(fold.train_start.date()),
                train_end=str(fold.train_end.date()),
                test_start=str(fold.test_start.date()),
                test_end=str(fold.test_end.date()),
                n_train_samples=n_train,
                n_test_samples=n_test,
                metrics={"error": "insufficient_test_data"},
            )

        # Step 3.1: Label Optimization
        print("\n  [Step 3.1] Label Optimization...")
        train_features = train_df[feature_cols].values
        opt_result = label_optimizer(train_df)
        best_params = opt_result.best_params
        print(f"  Best params: SL={best_params['sl']}, PT={best_params['pt']}, Time={best_params['time']}")

        # Step 3.2: Fine-tuning
        print("\n  [Step 3.2] Fine-tuning...")
        model = trainer(
            train_features=train_features,
            valid_indices=opt_result.valid_indices,
            labels=opt_result.labels,
            encoder_path=self.config.encoder_path,
        )

        # Step 3.3: Signal Generation + Backtest
        print("\n  [Step 3.3] Signal Generation & Backtest...")

        # Generate signals on test data
        test_features = test_df[feature_cols].values
        generator = SignalGenerator(
            model=model,
            config=SignalGeneratorConfig(context_len=self.config.context_len),
        )
        signals = generator.generate(test_features)

        # Prepare backtest config
        backtest_config = BacktestConfig(
            sl_mult=best_params["sl"],
            pt_mult=best_params["pt"],
            max_hold_bars=best_params["time"],
        )

        # Prepare features for Go backtester
        bt_features = self._prepare_backtest_features(test_df, timestamp_col, price_cols)

        # Run Go backtester
        result = self.go_bridge.run_backtest_with_data(
            signals=signals,
            features=bt_features,
            config=backtest_config,
        )

        # Compute signal stats
        from .signal_generator import compute_signal_stats
        signal_stats = compute_signal_stats(signals)

        return FoldResult(
            fold_id=fold.fold_id,
            train_start=str(fold.train_start.date()),
            train_end=str(fold.train_end.date()),
            test_start=str(fold.test_start.date()),
            test_end=str(fold.test_end.date()),
            best_params=best_params,
            n_train_samples=n_train,
            n_test_samples=n_test,
            metrics={
                "total_trades": result.total_trades,
                "win_rate": result.win_rate,
                "avg_pnl": result.avg_pnl,
                "total_pnl": result.total_pnl,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "profit_factor": result.profit_factor,
                "avg_hold_bars": result.avg_hold_bars,
                "tp_count": result.tp_count,
                "sl_count": result.sl_count,
                "timeout_count": result.timeout_count,
            },
            signal_stats=signal_stats,
        )

    def _prepare_backtest_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        price_cols: dict,
    ) -> np.ndarray:
        """
        Prepare features array for Go backtester.

        Returns (N, 9) array with columns:
        [timestamp, open, high, high_time, low, low_time, close, volume, realized_vol]
        """
        n = len(df)
        features = np.zeros((n, 9), dtype=np.float64)

        # Timestamp
        ts = df[timestamp_col].values
        if not np.issubdtype(ts.dtype, np.integer):
            ts = pd.to_datetime(ts).astype(np.int64) // 10**6  # to milliseconds
        features[:, 0] = ts

        # OHLCV
        features[:, 1] = df[price_cols["open"]].values
        features[:, 2] = df[price_cols["high"]].values
        features[:, 3] = df[price_cols.get("high_time", timestamp_col)].values
        features[:, 4] = df[price_cols["low"]].values
        features[:, 5] = df[price_cols.get("low_time", timestamp_col)].values
        features[:, 6] = df[price_cols["close"]].values
        features[:, 7] = df[price_cols["volume"]].values
        features[:, 8] = df[price_cols["realized_vol"]].values

        return features

    def _save_results(self) -> None:
        """Save WFA results to output directory."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual fold results
        for result in self.results:
            fold_file = output_dir / f"fold_{result.fold_id}.json"
            with open(fold_file, "w") as f:
                json.dump(result.to_dict(), f, indent=2)

        # Save summary
        summary = self._compute_summary()
        summary_file = output_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {output_dir}")

    def _compute_summary(self) -> dict:
        """Compute aggregate summary across all folds."""
        valid_results = [r for r in self.results if "error" not in r.metrics]

        if not valid_results:
            return {"error": "no_valid_folds", "total_folds": len(self.results)}

        # Aggregate metrics
        metrics_keys = [
            "total_trades", "win_rate", "avg_pnl", "total_pnl",
            "sharpe_ratio", "max_drawdown", "profit_factor"
        ]

        aggregated = {}
        for key in metrics_keys:
            values = [r.metrics.get(key, 0) for r in valid_results]
            aggregated[f"mean_{key}"] = float(np.mean(values))
            aggregated[f"std_{key}"] = float(np.std(values))
            aggregated[f"min_{key}"] = float(np.min(values))
            aggregated[f"max_{key}"] = float(np.max(values))

        return {
            "total_folds": len(self.results),
            "valid_folds": len(valid_results),
            "timestamp": datetime.now().isoformat(),
            "config": {
                "train_months": self.config.train_months,
                "test_months": self.config.test_months,
            },
            "metrics": aggregated,
        }

    def get_summary(self) -> dict:
        """Get summary of WFA results."""
        return self._compute_summary()


def print_wfa_summary(results: list[FoldResult]) -> None:
    """Print formatted WFA summary."""
    valid_results = [r for r in results if "error" not in r.metrics]

    print("\n" + "=" * 70)
    print("WFA SUMMARY")
    print("=" * 70)

    print(f"\nTotal Folds: {len(results)}")
    print(f"Valid Folds: {len(valid_results)}")

    if not valid_results:
        print("\nNo valid results to summarize.")
        return

    # Per-fold table
    print("\nPer-Fold Results:")
    print("-" * 70)
    print(f"{'Fold':>4} {'Period':<20} {'Trades':>7} {'WinRate':>8} {'Sharpe':>8} {'PnL':>10}")
    print("-" * 70)

    for r in valid_results:
        period = f"{r.test_start}~{r.test_end}"
        print(
            f"{r.fold_id:>4} {period:<20} "
            f"{r.metrics['total_trades']:>7} "
            f"{r.metrics['win_rate']:>7.1%} "
            f"{r.metrics['sharpe_ratio']:>8.2f} "
            f"{r.metrics['total_pnl']:>9.2%}"
        )

    # Aggregate stats
    sharpes = [r.metrics["sharpe_ratio"] for r in valid_results]
    pnls = [r.metrics["total_pnl"] for r in valid_results]
    win_rates = [r.metrics["win_rate"] for r in valid_results]

    print("-" * 70)
    print(f"\nAggregate Statistics:")
    print(f"  Sharpe Ratio:  {np.mean(sharpes):.2f} +/- {np.std(sharpes):.2f}")
    print(f"  Win Rate:      {np.mean(win_rates):.1%} +/- {np.std(win_rates):.1%}")
    print(f"  Total PnL:     {np.mean(pnls):.2%} +/- {np.std(pnls):.2%}")
    print(f"  Cumulative:    {np.sum(pnls):.2%}")
    print("=" * 70)
