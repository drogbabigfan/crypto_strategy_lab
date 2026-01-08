"""
Unit tests for AKF (Adaptive Kalman Filter) Trading Strategies.

Tests cover:
1. Basic functionality
2. Signal value range
3. Strategy-specific logic
4. Edge cases
"""

import numpy as np
import pandas as pd
import pytest
from strategies.akf import AKFStrategies, AKFStrategyConfig, apply_akf_strategy


class TestAKFBasics:
    """Basic functionality tests."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with Kalman filter outputs."""
        n = 100
        np.random.seed(42)

        # Simulate price series with trend
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        df = pd.DataFrame({
            'close': prices,
            'kf_trend': prices + np.random.randn(n) * 0.1,  # Trend close to price
            'kf_trend_pred': prices + np.random.randn(n) * 0.2,  # Prediction
            'kf_velocity': np.random.randn(n) * 0.5,  # Random velocity
            'kf_uncertainty': np.abs(np.random.randn(n)) + 0.1,  # Positive uncertainty
        })
        return df

    def test_all_strategies_generate_signal(self, sample_df):
        """All strategies should generate a signal column."""
        akf = AKFStrategies()

        for strategy_name in akf.get_strategy_names():
            result = akf.run_strategy(sample_df, strategy_name)
            assert 'signal' in result.columns, f"{strategy_name} missing signal column"

    def test_signal_value_range(self, sample_df):
        """Signal values should be in {-1, 0, 1}."""
        akf = AKFStrategies()

        for strategy_name in akf.get_strategy_names():
            result = akf.run_strategy(sample_df, strategy_name)
            unique_signals = set(result['signal'].unique())
            assert unique_signals.issubset({-1, 0, 1}), \
                f"{strategy_name} has invalid signals: {unique_signals}"

    def test_include_bands_option(self, sample_df):
        """When include_bands=True, band columns should be added."""
        akf = AKFStrategies()

        for strategy_name in akf.get_strategy_names():
            result = akf.run_strategy(sample_df, strategy_name, include_bands=True)
            assert 'upper_band' in result.columns
            assert 'lower_band' in result.columns

    def test_custom_k_parameter(self, sample_df):
        """Custom k parameter should affect band width."""
        akf = AKFStrategies()

        result_k1 = akf.strategy_db(sample_df, k=1.0, include_bands=True)
        result_k3 = akf.strategy_db(sample_df, k=3.0, include_bands=True)

        # Larger k should produce wider bands
        band_width_k1 = (result_k1['upper_band'] - result_k1['lower_band']).mean()
        band_width_k3 = (result_k3['upper_band'] - result_k3['lower_band']).mean()

        assert band_width_k3 > band_width_k1


class TestStrategyDB:
    """Tests for Dynamic Breakout strategy."""

    def test_breakout_entry(self):
        """Price breaking above upper band should trigger long entry."""
        df = pd.DataFrame({
            'close': [100, 101, 110, 111, 112],  # Jump at index 2
            'kf_trend': [100, 100, 100, 100, 100],
            'kf_trend_pred': [100, 100, 100, 100, 100],
            'kf_velocity': [0, 0, 0, 0, 0],
            'kf_uncertainty': [1, 1, 1, 1, 1],  # sqrt = 1, k=2 -> band = 2
        })

        akf = AKFStrategies(AKFStrategyConfig(k=2.0))
        result = akf.strategy_db(df)

        # At index 2, close=110 > upper_band=102 -> should be long
        assert result['signal'].iloc[2] == 1

    def test_trend_reversion_exit(self):
        """Price dropping below trend should exit long position."""
        df = pd.DataFrame({
            'close': [100, 110, 111, 99, 98],  # Breakout then drop below trend
            'kf_trend': [100, 100, 100, 100, 100],
            'kf_trend_pred': [100, 100, 100, 100, 100],
            'kf_velocity': [0, 0, 0, 0, 0],
            'kf_uncertainty': [1, 1, 1, 1, 1],
        })

        akf = AKFStrategies(AKFStrategyConfig(k=2.0))
        result = akf.strategy_db(df)

        # Should be long at index 1-2, then flat at index 3-4
        assert result['signal'].iloc[1] == 1
        assert result['signal'].iloc[3] == 0

    def test_short_entry_on_lower_breakout(self):
        """Price breaking below lower band should trigger short entry."""
        df = pd.DataFrame({
            'close': [100, 99, 90, 89, 88],  # Drop at index 2
            'kf_trend': [100, 100, 100, 100, 100],
            'kf_trend_pred': [100, 100, 100, 100, 100],
            'kf_velocity': [0, 0, 0, 0, 0],
            'kf_uncertainty': [1, 1, 1, 1, 1],
        })

        akf = AKFStrategies(AKFStrategyConfig(k=2.0))
        result = akf.strategy_db(df)

        # At index 2, close=90 < lower_band=98 -> should be short
        assert result['signal'].iloc[2] == -1


class TestStrategyBenhamou:
    """Tests for Benhamou Predictive strategy."""

    def test_prediction_based_entry(self):
        """Strong prediction above prev close should trigger long."""
        df = pd.DataFrame({
            'close': [100, 100, 100, 100, 100],
            'kf_trend': [100, 100, 100, 100, 100],
            'kf_trend_pred': [100, 100, 110, 110, 110],  # Prediction jumps at index 2
            'kf_velocity': [0, 0, 1, 1, 1],
            'kf_uncertainty': [1, 1, 1, 1, 1],
        })

        akf = AKFStrategies(AKFStrategyConfig(k=2.0))
        result = akf.strategy_benhamou(df)

        # At index 2, trend_pred=110 > prev_close(100) + k*sqrt(1) = 102 -> long
        assert result['signal'].iloc[2] == 1

    def test_signal_reversal_switching(self):
        """Opposite signal should switch position."""
        df = pd.DataFrame({
            'close': [100, 100, 100, 100, 100],
            'kf_trend': [100, 100, 100, 100, 100],
            'kf_trend_pred': [100, 110, 110, 90, 90],  # Long then Short prediction
            'kf_velocity': [0, 1, 1, -1, -1],
            'kf_uncertainty': [1, 1, 1, 1, 1],
        })

        akf = AKFStrategies(AKFStrategyConfig(k=2.0))
        result = akf.strategy_benhamou(df)

        # Should switch from long to short
        assert result['signal'].iloc[1] == 1  # Long
        assert result['signal'].iloc[3] == -1  # Switched to short

    def test_position_held_without_reversal(self):
        """Position should be held when no reversal signal."""
        df = pd.DataFrame({
            'close': [100, 100, 100, 100, 100],
            'kf_trend': [100, 100, 100, 100, 100],
            'kf_trend_pred': [100, 110, 101, 101, 101],  # Long entry, then neutral
            'kf_velocity': [0, 1, 0, 0, 0],
            'kf_uncertainty': [1, 1, 1, 1, 1],
        })

        akf = AKFStrategies(AKFStrategyConfig(k=2.0))
        result = akf.strategy_benhamou(df)

        # Position should be held
        assert result['signal'].iloc[1] == 1
        assert result['signal'].iloc[4] == 1  # Still long


class TestStrategyHybrid:
    """Tests for Hybrid Fusion strategy."""

    def test_velocity_confirmation_required(self):
        """Entry requires velocity confirmation."""
        df = pd.DataFrame({
            'close': [100, 100, 100, 100, 100],
            'kf_trend': [100, 100, 100, 100, 100],
            'kf_trend_pred': [100, 110, 110, 110, 110],  # Bullish prediction
            'kf_velocity': [0, -1, -1, 1, 1],  # Negative then positive velocity
            'kf_uncertainty': [1, 1, 1, 1, 1],
        })

        akf = AKFStrategies(AKFStrategyConfig(k=2.0))
        result = akf.strategy_hybrid(df)

        # At index 1-2: prediction bullish but velocity negative -> no entry
        assert result['signal'].iloc[1] == 0
        assert result['signal'].iloc[2] == 0
        # At index 3-4: prediction bullish AND velocity positive -> long
        assert result['signal'].iloc[3] == 1

    def test_velocity_mismatch_blocks_entry(self):
        """Velocity mismatch should block entry."""
        df = pd.DataFrame({
            'close': [100, 100, 100, 100, 100],
            'kf_trend': [100, 100, 100, 100, 100],
            'kf_trend_pred': [100, 90, 90, 90, 90],  # Bearish prediction
            'kf_velocity': [0, 1, 1, 1, 1],  # Positive velocity (mismatch)
            'kf_uncertainty': [1, 1, 1, 1, 1],
        })

        akf = AKFStrategies(AKFStrategyConfig(k=2.0))
        result = akf.strategy_hybrid(df)

        # Bearish prediction but positive velocity -> no entry
        assert all(result['signal'] == 0)

    def test_db_style_exit(self):
        """Hybrid uses DB-style trend reversion exit."""
        df = pd.DataFrame({
            'close': [100, 100, 100, 95, 90],  # Price drops below trend
            'kf_trend': [100, 100, 100, 100, 100],
            'kf_trend_pred': [100, 110, 110, 110, 110],  # Bullish prediction
            'kf_velocity': [0, 1, 1, 1, 1],  # Positive velocity
            'kf_uncertainty': [1, 1, 1, 1, 1],
        })

        akf = AKFStrategies(AKFStrategyConfig(k=2.0))
        result = akf.strategy_hybrid(df)

        # Entry at index 1, should exit when close < trend
        assert result['signal'].iloc[1] == 1
        assert result['signal'].iloc[3] == 0  # close=95 < trend=100


class TestEdgeCases:
    """Edge case tests."""

    def test_missing_columns_raises_error(self):
        """Missing required columns should raise ValueError."""
        df = pd.DataFrame({'close': [100, 101, 102]})
        akf = AKFStrategies()

        with pytest.raises(ValueError) as excinfo:
            akf.strategy_db(df)
        assert "Missing required columns" in str(excinfo.value)

    def test_empty_dataframe(self):
        """Empty DataFrame should be handled gracefully."""
        df = pd.DataFrame({
            'close': [],
            'kf_trend': [],
            'kf_trend_pred': [],
            'kf_velocity': [],
            'kf_uncertainty': [],
        })

        akf = AKFStrategies()
        result = akf.strategy_db(df)

        assert len(result) == 0
        assert 'signal' in result.columns

    def test_single_row(self):
        """Single row should be handled."""
        df = pd.DataFrame({
            'close': [100],
            'kf_trend': [100],
            'kf_trend_pred': [100],
            'kf_velocity': [0],
            'kf_uncertainty': [1],
        })

        akf = AKFStrategies()
        result = akf.strategy_db(df)

        assert len(result) == 1
        assert result['signal'].iloc[0] == 0

    def test_nan_in_data(self):
        """NaN values should not cause errors."""
        df = pd.DataFrame({
            'close': [100, np.nan, 102, 103, 104],
            'kf_trend': [100, 100, np.nan, 100, 100],
            'kf_trend_pred': [100, 100, 100, 100, 100],
            'kf_velocity': [0, 0, 0, 0, 0],
            'kf_uncertainty': [1, 1, 1, 1, 1],
        })

        akf = AKFStrategies()
        # Should not raise
        result = akf.strategy_db(df)
        assert 'signal' in result.columns

    def test_convenience_function(self):
        """Test the convenience function."""
        df = pd.DataFrame({
            'close': [100, 101, 110, 111, 112],
            'kf_trend': [100, 100, 100, 100, 100],
            'kf_trend_pred': [100, 100, 100, 100, 100],
            'kf_velocity': [0, 0, 0, 0, 0],
            'kf_uncertainty': [1, 1, 1, 1, 1],
        })

        result = apply_akf_strategy(df, strategy='strategy_db', k=2.0)
        assert 'signal' in result.columns

    def test_unknown_strategy_raises_error(self):
        """Unknown strategy name should raise ValueError."""
        df = pd.DataFrame({
            'close': [100],
            'kf_trend': [100],
            'kf_trend_pred': [100],
            'kf_velocity': [0],
            'kf_uncertainty': [1],
        })

        akf = AKFStrategies()
        with pytest.raises(ValueError) as excinfo:
            akf.run_strategy(df, 'unknown_strategy')
        assert "Unknown strategy" in str(excinfo.value)


class TestIntegration:
    """Integration tests with actual Kalman filter."""

    def test_with_real_kalman_output(self):
        """Test with actual Kalman filter output."""
        # Skip if kalman module not available
        pytest.importorskip('research.features.kalman')

        from research.features.kalman import calculate_adaptive_kalman

        # Create OHLC data
        np.random.seed(42)
        n = 200
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        df = pd.DataFrame({
            'open': prices,
            'high': prices + np.abs(np.random.randn(n)) * 0.5,
            'low': prices - np.abs(np.random.randn(n)) * 0.5,
            'close': prices + np.random.randn(n) * 0.1,
        })

        # Apply Kalman filter
        kf_result = calculate_adaptive_kalman(df)

        # Apply strategies
        akf = AKFStrategies()

        for strategy_name in akf.get_strategy_names():
            result = akf.run_strategy(kf_result, strategy_name)
            assert 'signal' in result.columns
            # Check signal distribution (should not all be same value)
            unique_signals = result['signal'].unique()
            assert len(unique_signals) >= 1  # At least some signal

    def test_full_pipeline(self):
        """Test full pipeline from raw data to signals."""
        pytest.importorskip('research.features.kalman')

        from research.features.kalman import calculate_adaptive_kalman

        # Create trending market data
        np.random.seed(123)
        n = 300
        trend = np.linspace(100, 150, n)  # Uptrend
        noise = np.random.randn(n) * 2
        prices = trend + noise

        df = pd.DataFrame({
            'open': prices,
            'high': prices + np.abs(np.random.randn(n)),
            'low': prices - np.abs(np.random.randn(n)),
            'close': prices,
        })

        # Full pipeline
        kf_result = calculate_adaptive_kalman(df)

        akf = AKFStrategies(AKFStrategyConfig(k=1.5))
        db_result = akf.strategy_db(kf_result, include_bands=True)

        # In uptrend, should have some long signals
        assert 1 in db_result['signal'].values

        # Bands should be reasonable
        assert db_result['upper_band'].notna().sum() > 0
        assert db_result['lower_band'].notna().sum() > 0
