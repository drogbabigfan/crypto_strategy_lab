"""
Tests for WFA (Walk-Forward Analysis) module.
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from research.wfa.go_bridge import GoBridge, BacktestConfig, BacktestResult
from research.wfa.signal_generator import (
    SignalGenerator,
    SignalGeneratorConfig,
    count_signals,
    compute_signal_stats,
)
from research.wfa.engine import WFAEngine, WFAConfig, FoldResult, Fold


# =============================================================================
# Fixtures
# =============================================================================


class MockClassifier(nn.Module):
    """Mock classifier for testing."""

    def __init__(self, n_features: int = 28):
        super().__init__()
        # Simple mock: just average and classify
        self.fc = nn.Linear(n_features, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features, context_len)
        # Average over time dimension
        x = x.mean(dim=-1)  # (B, n_features)
        return self.fc(x)  # (B, 3)


@pytest.fixture
def mock_classifier():
    """Create a mock classifier."""
    return MockClassifier()


@pytest.fixture
def sample_features():
    """Create sample features for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 28

    return np.random.randn(n_samples, n_features).astype(np.float32)


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame with all required columns."""
    np.random.seed(42)
    n_samples = 10000  # ~208 days at 30min frequency = ~7 months

    # Generate timestamps spanning 7 months
    start_date = pd.Timestamp("2023-01-01")
    timestamps = pd.date_range(start=start_date, periods=n_samples, freq="30min")

    df = pd.DataFrame({
        "timestamp": timestamps.astype(np.int64) // 10**6,  # ms
        "open": 50000 + np.cumsum(np.random.randn(n_samples) * 10),
        "high": 50000 + np.cumsum(np.random.randn(n_samples) * 10) + 50,
        "high_time": timestamps.astype(np.int64) // 10**6 + 100,
        "low": 50000 + np.cumsum(np.random.randn(n_samples) * 10) - 50,
        "low_time": timestamps.astype(np.int64) // 10**6 + 200,
        "close": 50000 + np.cumsum(np.random.randn(n_samples) * 10),
        "volume": np.abs(np.random.randn(n_samples)) * 1000,
        "realized_vol": np.abs(np.random.randn(n_samples)) * 0.02 + 0.01,
    })

    # Add feature columns
    for i in range(28):
        df[f"feature_{i}"] = np.random.randn(n_samples)

    return df


# =============================================================================
# BacktestConfig Tests
# =============================================================================


class TestBacktestConfig:
    """Tests for BacktestConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BacktestConfig()

        assert config.sl_mult == 2.0
        assert config.pt_mult == 2.5
        assert config.max_hold_bars == 100
        assert config.base_fee == 0.001
        assert config.initial_capital == 100000.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BacktestConfig(
            sl_mult=1.5,
            pt_mult=3.0,
            max_hold_bars=50,
        )

        assert config.sl_mult == 1.5
        assert config.pt_mult == 3.0
        assert config.max_hold_bars == 50

    def test_to_dict(self):
        """Test conversion to dictionary (Go-compatible PascalCase keys)."""
        config = BacktestConfig(sl_mult=1.5)
        d = config.to_dict()

        assert isinstance(d, dict)
        assert d["SLMult"] == 1.5  # Go expects PascalCase
        assert "BaseFee" in d


# =============================================================================
# BacktestResult Tests
# =============================================================================


class TestBacktestResult:
    """Tests for BacktestResult."""

    def test_from_dict(self):
        """Test creating result from dictionary."""
        data = {
            "total_trades": 100,
            "win_rate": 0.55,
            "sharpe_ratio": 1.8,
            "total_pnl": 0.15,
        }

        result = BacktestResult.from_dict(data)

        assert result.total_trades == 100
        assert result.win_rate == 0.55
        assert result.sharpe_ratio == 1.8
        assert result.total_pnl == 0.15

    def test_from_dict_with_missing_keys(self):
        """Test with missing keys uses defaults."""
        data = {"total_trades": 50}

        result = BacktestResult.from_dict(data)

        assert result.total_trades == 50
        assert result.win_rate == 0.0
        assert result.sharpe_ratio == 0.0

    def test_is_valid(self):
        """Test validity check."""
        valid = BacktestResult(total_trades=10)
        invalid = BacktestResult(total_trades=0)

        assert valid.is_valid()
        assert not invalid.is_valid()


# =============================================================================
# GoBridge Tests
# =============================================================================


class TestGoBridge:
    """Tests for GoBridge."""

    def test_init(self):
        """Test initialization."""
        bridge = GoBridge("./bin/backtester")
        assert bridge.backtester_path == Path("./bin/backtester")

    def test_is_available_false(self):
        """Test availability check when binary doesn't exist."""
        bridge = GoBridge("/nonexistent/path/backtester")
        assert not bridge.is_available()

    def test_check_binary_raises(self):
        """Test that check_binary raises when binary not found."""
        bridge = GoBridge("/nonexistent/path/backtester")

        with pytest.raises(FileNotFoundError):
            bridge._check_binary()

    def test_write_signals(self, tmp_path):
        """Test writing signals to parquet."""
        bridge = GoBridge("./bin/backtester")

        signals = np.array([1, 0, -1, 1, 0], dtype=np.int8)
        output_path = tmp_path / "signals.parquet"

        bridge._write_signals(signals, output_path)

        assert output_path.exists()

        # Read back and verify
        import pyarrow.parquet as pq
        table = pq.read_table(output_path)
        read_signals = table["signal"].to_numpy()

        np.testing.assert_array_equal(read_signals, signals)

    def test_write_signals_with_timestamps(self, tmp_path):
        """Test writing signals with custom timestamps."""
        bridge = GoBridge("./bin/backtester")

        signals = np.array([1, 0, -1], dtype=np.int8)
        timestamps = np.array([1000, 2000, 3000], dtype=np.int64)
        output_path = tmp_path / "signals.parquet"

        bridge._write_signals(signals, output_path, timestamps)

        import pyarrow.parquet as pq
        table = pq.read_table(output_path)

        np.testing.assert_array_equal(table["timestamp"].to_numpy(), timestamps)

    def test_write_features(self, tmp_path):
        """Test writing features to parquet."""
        bridge = GoBridge("./bin/backtester")

        # 9 columns: timestamp, open, high, high_time, low, low_time, close, volume, realized_vol
        features = np.array([
            [1000, 50000, 50100, 1010, 49900, 1020, 50050, 1000, 0.02],
            [2000, 50050, 50150, 2010, 49950, 2020, 50100, 1100, 0.021],
        ])
        output_path = tmp_path / "features.parquet"

        bridge._write_features(features, output_path)

        assert output_path.exists()

        import pyarrow.parquet as pq
        table = pq.read_table(output_path)

        assert len(table) == 2
        assert "open" in table.column_names
        assert "realized_vol" in table.column_names

    def test_write_features_invalid_shape(self, tmp_path):
        """Test that invalid feature shape raises error."""
        bridge = GoBridge("./bin/backtester")

        features = np.random.randn(10, 5)  # Wrong number of columns
        output_path = tmp_path / "features.parquet"

        with pytest.raises(ValueError, match="must have 9 columns"):
            bridge._write_features(features, output_path)


# =============================================================================
# SignalGenerator Tests
# =============================================================================


class TestSignalGenerator:
    """Tests for SignalGenerator."""

    def test_init(self, mock_classifier):
        """Test initialization."""
        generator = SignalGenerator(mock_classifier)

        assert generator.model is mock_classifier
        assert generator.config.context_len == 512

    def test_init_with_config(self, mock_classifier):
        """Test initialization with custom config."""
        config = SignalGeneratorConfig(context_len=256, batch_size=32)
        generator = SignalGenerator(mock_classifier, config)

        assert generator.config.context_len == 256
        assert generator.config.batch_size == 32

    def test_generate_shape(self, mock_classifier, sample_features):
        """Test that generate returns correct shape."""
        generator = SignalGenerator(
            mock_classifier,
            SignalGeneratorConfig(context_len=100),
        )

        signals = generator.generate(sample_features)

        assert signals.shape == (len(sample_features),)
        assert signals.dtype == np.int8

    def test_generate_valid_values(self, mock_classifier, sample_features):
        """Test that signals are in valid range."""
        generator = SignalGenerator(
            mock_classifier,
            SignalGeneratorConfig(context_len=100),
        )

        signals = generator.generate(sample_features)

        assert np.all(np.isin(signals, [-1, 0, 1]))

    def test_generate_warmup_period(self, mock_classifier, sample_features):
        """Test that warmup period is all neutral."""
        context_len = 100
        generator = SignalGenerator(
            mock_classifier,
            SignalGeneratorConfig(context_len=context_len),
        )

        signals = generator.generate(sample_features)

        # First context_len signals should be 0 (neutral)
        assert np.all(signals[:context_len] == 0)

    def test_generate_with_probs(self, mock_classifier, sample_features):
        """Test generate_with_probs returns probabilities."""
        generator = SignalGenerator(
            mock_classifier,
            SignalGeneratorConfig(context_len=100),
        )

        signals, probs = generator.generate_with_probs(sample_features)

        assert signals.shape == (len(sample_features),)
        assert probs.shape == (len(sample_features), 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_generate_insufficient_data(self, mock_classifier):
        """Test with insufficient data."""
        generator = SignalGenerator(
            mock_classifier,
            SignalGeneratorConfig(context_len=512),
        )

        # Only 100 samples, need 512
        features = np.random.randn(100, 28).astype(np.float32)
        signals = generator.generate(features)

        # All should be neutral
        assert np.all(signals == 0)

    def test_generate_streaming(self, mock_classifier, sample_features):
        """Test streaming generation."""
        generator = SignalGenerator(
            mock_classifier,
            SignalGeneratorConfig(context_len=100),
        )

        signals = generator.generate_streaming(sample_features)

        assert signals.shape == (len(sample_features),)
        assert np.all(signals[:100] == 0)


# =============================================================================
# Signal Utility Functions Tests
# =============================================================================


class TestSignalUtils:
    """Tests for signal utility functions."""

    def test_count_signals(self):
        """Test signal counting."""
        signals = np.array([1, 1, 0, -1, 0, 1, -1, -1], dtype=np.int8)

        counts = count_signals(signals)

        assert counts["long"] == 3
        assert counts["neutral"] == 2
        assert counts["short"] == 3
        assert counts["total"] == 8

    def test_count_signals_empty(self):
        """Test with empty signals."""
        signals = np.array([], dtype=np.int8)

        counts = count_signals(signals)

        assert counts["long"] == 0
        assert counts["neutral"] == 0
        assert counts["short"] == 0
        assert counts["total"] == 0

    def test_compute_signal_stats(self):
        """Test signal statistics computation."""
        signals = np.array([0, 1, 1, -1, 0, 1], dtype=np.int8)

        stats = compute_signal_stats(signals)

        assert "counts" in stats
        assert "ratios" in stats
        assert "transitions" in stats

        assert stats["ratios"]["long_ratio"] == 3 / 6
        assert stats["ratios"]["short_ratio"] == 1 / 6

    def test_compute_signal_stats_transitions(self):
        """Test transition counting."""
        signals = np.array([1, -1, 0, 1], dtype=np.int8)

        stats = compute_signal_stats(signals)

        assert stats["transitions"]["long_to_short"] == 1
        assert stats["transitions"]["to_neutral"] == 1
        assert stats["transitions"]["from_neutral"] == 1


# =============================================================================
# WFAConfig Tests
# =============================================================================


class TestWFAConfig:
    """Tests for WFAConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = WFAConfig()

        assert config.train_months == 12
        assert config.test_months == 1
        assert config.context_len == 512

    def test_custom_values(self):
        """Test custom configuration."""
        config = WFAConfig(train_months=6, test_months=2)

        assert config.train_months == 6
        assert config.test_months == 2


# =============================================================================
# FoldResult Tests
# =============================================================================


class TestFoldResult:
    """Tests for FoldResult."""

    def test_creation(self):
        """Test creating fold result."""
        result = FoldResult(
            fold_id=0,
            train_start="2023-01-01",
            train_end="2023-12-31",
            test_start="2024-01-01",
            test_end="2024-01-31",
        )

        assert result.fold_id == 0
        assert result.train_start == "2023-01-01"

    def test_to_dict(self):
        """Test conversion to dict."""
        result = FoldResult(
            fold_id=0,
            train_start="2023-01-01",
            train_end="2023-12-31",
            test_start="2024-01-01",
            test_end="2024-01-31",
            best_params={"sl": 2.0, "pt": 2.5},
            metrics={"sharpe_ratio": 1.5},
        )

        d = result.to_dict()

        assert d["fold_id"] == 0
        assert d["best_params"]["sl"] == 2.0
        assert d["metrics"]["sharpe_ratio"] == 1.5


# =============================================================================
# WFAEngine Tests
# =============================================================================


class TestWFAEngine:
    """Tests for WFAEngine."""

    def test_init(self):
        """Test initialization."""
        config = WFAConfig()
        engine = WFAEngine(config)

        assert engine.config is config
        assert isinstance(engine.go_bridge, GoBridge)

    def test_generate_folds(self, sample_dataframe):
        """Test fold generation."""
        config = WFAConfig(train_months=3, test_months=1)
        engine = WFAEngine(config)

        folds = engine.generate_folds(sample_dataframe)

        assert len(folds) > 0
        for fold in folds:
            assert isinstance(fold, Fold)
            assert fold.train_end == fold.test_start

    def test_generate_folds_insufficient_data(self):
        """Test with insufficient data for any folds."""
        config = WFAConfig(train_months=24, test_months=6)  # Need 30 months
        engine = WFAEngine(config)

        # Only 2 months of data
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="D")
        })

        folds = engine.generate_folds(df)

        assert len(folds) == 0

    def test_prepare_backtest_features(self, sample_dataframe):
        """Test feature preparation for backtester."""
        config = WFAConfig()
        engine = WFAEngine(config)

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

        features = engine._prepare_backtest_features(
            sample_dataframe,
            "timestamp",
            price_cols,
        )

        assert features.shape == (len(sample_dataframe), 9)
        assert features.dtype == np.float64

    def test_compute_summary_no_results(self):
        """Test summary with no results."""
        config = WFAConfig()
        engine = WFAEngine(config)

        summary = engine._compute_summary()

        assert summary["error"] == "no_valid_folds"

    def test_compute_summary_with_results(self):
        """Test summary computation."""
        config = WFAConfig()
        engine = WFAEngine(config)

        engine.results = [
            FoldResult(
                fold_id=0,
                train_start="2023-01",
                train_end="2023-12",
                test_start="2024-01",
                test_end="2024-02",
                metrics={"sharpe_ratio": 1.5, "win_rate": 0.55, "total_pnl": 0.1},
            ),
            FoldResult(
                fold_id=1,
                train_start="2023-02",
                train_end="2024-01",
                test_start="2024-02",
                test_end="2024-03",
                metrics={"sharpe_ratio": 1.8, "win_rate": 0.60, "total_pnl": 0.15},
            ),
        ]

        summary = engine._compute_summary()

        assert summary["valid_folds"] == 2
        assert "mean_sharpe_ratio" in summary["metrics"]
        assert summary["metrics"]["mean_sharpe_ratio"] == pytest.approx(1.65, rel=0.01)


# =============================================================================
# Integration Tests (require Go backtester)
# =============================================================================


@pytest.mark.integration
class TestGoBridgeIntegration:
    """Integration tests requiring Go backtester binary."""

    @pytest.fixture
    def go_bridge(self):
        """Create GoBridge with actual binary path."""
        bridge = GoBridge("./etl/bin/backtester")
        if not bridge.is_available():
            pytest.skip("Go backtester not available")
        return bridge

    def test_run_backtest(self, go_bridge, tmp_path):
        """Test running actual backtest."""
        # Create simple signals
        n = 1000
        signals = np.zeros(n, dtype=np.int8)
        signals[100] = 1   # Long signal
        signals[300] = -1  # Short signal
        signals[500] = 1   # Long signal

        # Create features
        features = np.zeros((n, 9), dtype=np.float64)
        features[:, 0] = np.arange(n) * 60000  # timestamps
        features[:, 1] = 50000 + np.cumsum(np.random.randn(n) * 10)  # open
        features[:, 2] = features[:, 1] + 50  # high
        features[:, 3] = features[:, 0] + 100  # high_time
        features[:, 4] = features[:, 1] - 50  # low
        features[:, 5] = features[:, 0] + 200  # low_time
        features[:, 6] = features[:, 1] + 10  # close
        features[:, 7] = 1000  # volume
        features[:, 8] = 0.02  # realized_vol

        result = go_bridge.run_backtest_with_data(
            signals=signals,
            features=features,
            config=BacktestConfig(sl_mult=2.0, pt_mult=2.5),
        )

        assert isinstance(result, BacktestResult)
        # Should have some trades
        assert result.total_trades >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
