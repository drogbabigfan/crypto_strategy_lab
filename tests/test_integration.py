"""
Integration tests with real BTC data.

Tests the full pipeline components with actual parquet data.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import torch
import torch.nn as nn

# Skip all tests if data not available
DATA_DIR = Path("/home/kimhoyeon/dev/dl_rl_btc/etl/data")
FEATURES_DIR = DATA_DIR / "bars" / "features"
BARS_DIR = DATA_DIR / "bars" / "futures"

pytestmark = pytest.mark.skipif(
    not FEATURES_DIR.exists() or not any(FEATURES_DIR.glob("*.parquet")),
    reason="Real data not available"
)


# =============================================================================
# Data Loading Utilities
# =============================================================================


def load_features(months: list[str]) -> pd.DataFrame:
    """Load feature parquet files for given months."""
    dfs = []
    for month in months:
        path = FEATURES_DIR / f"BTCUSDT-features-{month}.parquet"
        if path.exists():
            df = pq.read_table(path).to_pandas()
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No feature files found for months: {months}")

    return pd.concat(dfs, ignore_index=True)


def load_bars(months: list[str]) -> pd.DataFrame:
    """Load bar parquet files for given months."""
    dfs = []
    for month in months:
        # Try both locations
        for bars_path in [BARS_DIR / f"BTCUSDT-bars-{month}.parquet",
                          BARS_DIR / "BTCUSDT" / f"BTCUSDT-bars-{month}.parquet"]:
            if bars_path.exists():
                df = pq.read_table(bars_path).to_pandas()
                dfs.append(df)
                break

    if not dfs:
        raise FileNotFoundError(f"No bar files found for months: {months}")

    return pd.concat(dfs, ignore_index=True)


def merge_bars_features(bars: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """
    Merge bars (with high_time/low_time) and features.

    Join on start_time.
    """
    # Select needed columns from bars
    bars_cols = ["start_time", "open", "high", "high_time", "low", "low_time",
                 "close", "volume"]
    bars_subset = bars[bars_cols].copy()

    # Get feature columns (exclude price/time columns that will come from bars)
    feature_cols = [c for c in features.columns
                    if c not in ["start_time", "end_time", "open", "high", "low", "close"]]
    features_subset = features[["start_time"] + feature_cols].copy()

    # Merge on start_time
    merged = pd.merge(bars_subset, features_subset, on="start_time", how="inner")

    return merged


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get list of feature columns (excluding metadata/price columns)."""
    exclude = {"start_time", "end_time", "open", "high", "high_time",
               "low", "low_time", "close", "volume", "is_primed"}
    return [c for c in df.columns if c not in exclude]


# =============================================================================
# Mock Model for Testing
# =============================================================================


class SimpleClassifier(nn.Module):
    """Simple classifier for integration testing."""

    def __init__(self, n_features: int = 28, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, hidden_dim, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features, context_len)
        h = self.encoder(x)  # (B, hidden_dim, 1)
        h = h.squeeze(-1)    # (B, hidden_dim)
        return self.classifier(h)  # (B, 3)


# =============================================================================
# Data Loading Tests
# =============================================================================


class TestDataLoading:
    """Test data loading from parquet files."""

    def test_load_features_single_month(self):
        """Test loading a single month of features."""
        df = load_features(["2020-01"])

        assert len(df) > 0
        assert "start_time" in df.columns
        assert "realized_vol" in df.columns
        print(f"Loaded {len(df)} rows from 2020-01")

    def test_load_features_multiple_months(self):
        """Test loading multiple months of features."""
        df = load_features(["2020-01", "2020-02", "2020-03"])

        assert len(df) > 3000  # Should be several thousand rows
        print(f"Loaded {len(df)} rows from 3 months")

    def test_load_bars_single_month(self):
        """Test loading bars with high_time/low_time."""
        df = load_bars(["2020-01"])

        assert len(df) > 0
        assert "high_time" in df.columns
        assert "low_time" in df.columns
        print(f"Loaded {len(df)} bars from 2020-01")

    def test_merge_bars_features(self):
        """Test merging bars and features."""
        bars = load_bars(["2020-01"])
        features = load_features(["2020-01"])

        merged = merge_bars_features(bars, features)

        # Should have both price data and features
        assert "high_time" in merged.columns
        assert "low_time" in merged.columns
        assert "realized_vol" in merged.columns

        # Merged should have same length as features (assuming 1:1 join)
        assert abs(len(merged) - len(features)) < 10  # Allow small mismatch
        print(f"Merged: {len(merged)} rows with {len(merged.columns)} columns")


# =============================================================================
# Signal Generator Tests
# =============================================================================


class TestSignalGeneratorWithRealData:
    """Test SignalGenerator with real data."""

    def test_generate_signals_on_real_data(self):
        """Test signal generation on real feature data."""
        from research.wfa.signal_generator import SignalGenerator, SignalGeneratorConfig

        # Load real features
        features_df = load_features(["2020-01"])
        feature_cols = get_feature_columns(features_df)

        # Use a subset of features
        feature_cols = [c for c in feature_cols if not c.startswith("momentum_zscore")][:28]
        features = features_df[feature_cols].values.astype(np.float32)

        # Handle NaN
        features = np.nan_to_num(features, nan=0.0)

        # Create model and generator
        model = SimpleClassifier(n_features=len(feature_cols))
        config = SignalGeneratorConfig(context_len=128, batch_size=32)
        generator = SignalGenerator(model, config)

        # Generate signals
        signals = generator.generate(features)

        assert signals.shape == (len(features),)
        assert signals.dtype == np.int8
        assert np.all(np.isin(signals, [-1, 0, 1]))

        # First context_len signals should be 0 (warmup)
        assert np.all(signals[:config.context_len] == 0)

        # Print distribution
        unique, counts = np.unique(signals, return_counts=True)
        print("Signal distribution:", dict(zip(unique, counts)))


# =============================================================================
# GoBridge Tests
# =============================================================================


class TestGoBridgeWithRealData:
    """Test GoBridge with real data."""

    @pytest.fixture
    def go_bridge(self):
        """Create GoBridge if backtester exists."""
        from research.wfa.go_bridge import GoBridge

        backtester_path = Path("/home/kimhoyeon/dev/dl_rl_btc/etl/bin/backtester")
        if not backtester_path.exists():
            pytest.skip("Go backtester not built")

        return GoBridge(str(backtester_path))

    def test_backtest_with_real_data(self, go_bridge):
        """Test backtesting with real BTC data."""
        from research.wfa.go_bridge import BacktestConfig

        # Load and merge data
        bars = load_bars(["2020-01"])
        features_df = load_features(["2020-01"])
        merged = merge_bars_features(bars, features_df)

        # Create random signals for testing
        n = len(merged)
        np.random.seed(42)
        signals = np.zeros(n, dtype=np.int8)
        # Generate some random entry signals
        entry_indices = np.random.choice(range(100, n-100), size=20, replace=False)
        signals[entry_indices] = np.random.choice([-1, 1], size=20)

        # Prepare features array for Go backtester
        # [timestamp, open, high, high_time, low, low_time, close, volume, realized_vol]
        bt_features = np.zeros((n, 9), dtype=np.float64)
        bt_features[:, 0] = merged["start_time"].values
        bt_features[:, 1] = merged["open"].values
        bt_features[:, 2] = merged["high"].values
        bt_features[:, 3] = merged["high_time"].values
        bt_features[:, 4] = merged["low"].values
        bt_features[:, 5] = merged["low_time"].values
        bt_features[:, 6] = merged["close"].values
        bt_features[:, 7] = merged["volume"].values
        bt_features[:, 8] = merged["realized_vol"].values

        # Run backtest
        config = BacktestConfig(sl_mult=2.0, pt_mult=2.5, max_hold_bars=50)
        result = go_bridge.run_backtest_with_data(
            signals=signals,
            features=bt_features,
            config=config,
        )

        print(f"\nBacktest Results (random signals):")
        print(f"  Total trades: {result.total_trades}")
        print(f"  Win rate: {result.win_rate:.1%}")
        print(f"  Sharpe ratio: {result.sharpe_ratio:.2f}")
        print(f"  Total PnL: {result.total_pnl:.2%}")

        assert result.total_trades >= 0
        assert 0 <= result.win_rate <= 1


# =============================================================================
# Label Optimizer Tests
# =============================================================================


class TestLabelOptimizerWithRealData:
    """Test label optimizer with real data."""

    def test_triple_barrier_on_real_data(self):
        """Test triple barrier labeling on real BTC data."""
        from research.label_optimizer.tbm import TripleBarrierLabeler, TBMConfig

        # Load data
        bars = load_bars(["2020-01", "2020-02"])
        features_df = load_features(["2020-01", "2020-02"])
        merged = merge_bars_features(bars, features_df)

        # Filter primed rows
        if "is_primed" in features_df.columns:
            primed_mask = features_df["is_primed"].values[:len(merged)]
            merged = merged.iloc[:len(primed_mask)][primed_mask].reset_index(drop=True)

        n = len(merged)
        print(f"Testing triple barrier on {n} bars")

        # Create labeler
        config = TBMConfig(sl_mult=2.0, pt_mult=2.5, vertical_bars=100)
        labeler = TripleBarrierLabeler(config)

        # Apply triple barrier
        result = labeler.label(merged)

        n_valid = len(result.valid_indices)
        print(f"Valid labels: {n_valid} ({n_valid/n:.1%})")
        print(f"Filtered by Fee Trap: {result.n_filtered}")

        # Check label distribution
        unique, counts = np.unique(result.labels, return_counts=True)
        print("Label distribution:", dict(zip(unique, counts)))

        assert n_valid > 0
        assert np.all(np.isin(result.labels, [-1, 0, 1]))


# =============================================================================
# WFA Engine Tests
# =============================================================================


class TestWFAEngineWithRealData:
    """Test WFA Engine with real data."""

    def test_generate_folds_on_real_data(self):
        """Test fold generation with real data spanning multiple months."""
        from research.wfa.engine import WFAEngine, WFAConfig

        # Load 12 months of data
        months = [f"2020-{m:02d}" for m in range(1, 13)]
        features_df = load_features(months)

        # Rename start_time to timestamp for WFAEngine
        features_df = features_df.rename(columns={"start_time": "timestamp"})

        config = WFAConfig(
            train_months=6,
            test_months=1,
            step_months=1,
            min_train_bars=1000,
            min_test_bars=100,
        )
        engine = WFAEngine(config)

        folds = engine.generate_folds(features_df, timestamp_col="timestamp")

        print(f"\nGenerated {len(folds)} folds from 12 months of data")
        for fold in folds[:3]:
            print(f"  Fold {fold.fold_id}: Train {fold.train_start.date()} ~ {fold.train_end.date()}, "
                  f"Test {fold.test_start.date()} ~ {fold.test_end.date()}")

        assert len(folds) >= 5  # Should have at least 5 folds with 6m train + 1m test


# =============================================================================
# End-to-End Pipeline Test
# =============================================================================


class TestEndToEndPipeline:
    """End-to-end pipeline test with real data."""

    @pytest.fixture
    def go_bridge(self):
        """Create GoBridge if backtester exists."""
        from research.wfa.go_bridge import GoBridge

        backtester_path = Path("/home/kimhoyeon/dev/dl_rl_btc/etl/bin/backtester")
        if not backtester_path.exists():
            pytest.skip("Go backtester not built")

        return GoBridge(str(backtester_path))

    def test_single_fold_pipeline(self, go_bridge):
        """Test single fold of the WFA pipeline."""
        from research.wfa.signal_generator import SignalGenerator, SignalGeneratorConfig
        from research.wfa.go_bridge import BacktestConfig

        print("\n" + "="*60)
        print("End-to-End Single Fold Test")
        print("="*60)

        # 1. Load train/test data (3 months train, 1 month test)
        train_months = ["2020-01", "2020-02", "2020-03"]
        test_months = ["2020-04"]

        train_bars = load_bars(train_months)
        train_features = load_features(train_months)
        train_merged = merge_bars_features(train_bars, train_features)

        test_bars = load_bars(test_months)
        test_features = load_features(test_months)
        test_merged = merge_bars_features(test_bars, test_features)

        print(f"Train: {len(train_merged)} bars ({train_months[0]} ~ {train_months[-1]})")
        print(f"Test: {len(test_merged)} bars ({test_months[0]})")

        # 2. Prepare feature columns
        feature_cols = get_feature_columns(train_features)
        feature_cols = [c for c in feature_cols if c in test_features.columns][:28]

        # 3. Create and "train" a simple model (just initialize for testing)
        model = SimpleClassifier(n_features=len(feature_cols))
        model.eval()

        # 4. Generate signals on test data
        test_features_arr = test_merged[feature_cols].values.astype(np.float32)
        test_features_arr = np.nan_to_num(test_features_arr, nan=0.0)

        generator = SignalGenerator(
            model=model,
            config=SignalGeneratorConfig(context_len=128, batch_size=64),
        )
        signals = generator.generate(test_features_arr)

        # Print signal distribution
        unique, counts = np.unique(signals, return_counts=True)
        print(f"Signals: {dict(zip(unique, counts))}")

        # 5. Run Go backtester
        n = len(test_merged)
        bt_features = np.zeros((n, 9), dtype=np.float64)
        bt_features[:, 0] = test_merged["start_time"].values
        bt_features[:, 1] = test_merged["open"].values
        bt_features[:, 2] = test_merged["high"].values
        bt_features[:, 3] = test_merged["high_time"].values
        bt_features[:, 4] = test_merged["low"].values
        bt_features[:, 5] = test_merged["low_time"].values
        bt_features[:, 6] = test_merged["close"].values
        bt_features[:, 7] = test_merged["volume"].values
        bt_features[:, 8] = test_merged["realized_vol"].values

        config = BacktestConfig(
            sl_mult=2.0,
            pt_mult=2.5,
            max_hold_bars=100,
        )

        result = go_bridge.run_backtest_with_data(
            signals=signals,
            features=bt_features,
            config=config,
        )

        # 6. Print results
        print(f"\nBacktest Results:")
        print(f"  Total trades: {result.total_trades}")
        print(f"  Win rate: {result.win_rate:.1%}")
        print(f"  Avg PnL: {result.avg_pnl:.4%}")
        print(f"  Total PnL: {result.total_pnl:.2%}")
        print(f"  Sharpe ratio: {result.sharpe_ratio:.2f}")
        print(f"  Max drawdown: {result.max_drawdown:.2%}")
        print(f"  TP/SL/Timeout: {result.tp_count}/{result.sl_count}/{result.timeout_count}")

        # Assertions
        assert result.total_trades >= 0
        assert 0 <= result.win_rate <= 1
        print("\n✅ End-to-end test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
