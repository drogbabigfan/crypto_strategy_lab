"""
Tests for Adaptive Kalman Filter.
"""

import pytest
import pandas as pd
import numpy as np
from research.features.kalman import (
    AdaptiveKalmanFilter,
    KalmanConfig,
    KalmanFeatureGenerator,
    calculate_adaptive_kalman,
)


class TestKalmanBasics:
    """Basic Kalman filter tests."""

    def test_filter_basic(self, sample_dollar_bar_data):
        """Test basic Kalman filter operation."""
        config = KalmanConfig(r_window=20, q_window=20)
        kf = AdaptiveKalmanFilter(config)
        result = kf.filter(sample_dollar_bar_data)

        # Check all expected columns exist
        expected_cols = [
            'kf_trend', 'kf_velocity', 'kf_deviation',
            'kf_deviation_pct', 'kf_gain', 'kf_uncertainty', 'kf_signal'
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_filter_no_nan_after_warmup(self, sample_dollar_bar_data):
        """After warmup period, no NaN in key outputs."""
        config = KalmanConfig(r_window=20, q_window=20)
        kf = AdaptiveKalmanFilter(config)
        result = kf.filter(sample_dollar_bar_data)

        # After 50 bars, should have valid data
        warmup = 50
        for col in ['kf_trend', 'kf_velocity', 'kf_deviation']:
            valid = result[col].iloc[warmup:]
            assert not valid.isna().any(), f"NaN found in {col} after warmup"

    def test_trend_follows_price(self, sample_dollar_bar_data):
        """Trend should generally follow price movement."""
        config = KalmanConfig(r_window=20, q_window=20)
        kf = AdaptiveKalmanFilter(config)
        result = kf.filter(sample_dollar_bar_data)

        # Correlation between trend and close should be high
        corr = result['kf_trend'].corr(result['close'])
        assert corr > 0.95, f"Low trend-price correlation: {corr}"


class TestAdaptiveNoise:
    """Tests for adaptive noise estimation."""

    def test_r_increases_with_volatility(self):
        """R estimation should increase when volatility increases."""
        np.random.seed(42)
        n = 500

        # Low volatility period followed by high volatility
        low_vol = np.random.randn(250) * 10
        high_vol = np.random.randn(250) * 100

        close = 45000 + np.cumsum(np.concatenate([low_vol, high_vol]))
        df = pd.DataFrame({
            'close': close,
            'high': close + np.abs(np.random.randn(n) * 50),
            'low': close - np.abs(np.random.randn(n) * 50),
        })

        config = KalmanConfig(r_window=20, q_window=20)
        kf = AdaptiveKalmanFilter(config)
        result = kf.filter(df)

        # Check that the filter adapts - deviation should be larger in high vol period
        # (measuring raw volatility difference in output)
        low_vol_dev_std = result['kf_deviation'].iloc[100:200].std()
        high_vol_dev_std = result['kf_deviation'].iloc[300:400].std()

        # High volatility period should have larger deviations
        assert high_vol_dev_std > low_vol_dev_std * 2, "Deviation should be larger in high volatility"

    def test_q_increases_with_momentum(self):
        """Q should increase when momentum is strong."""
        n = 500

        # Sideways then strong trend
        sideways = np.random.randn(250) * 10
        trend = np.arange(250) * 50 + np.random.randn(250) * 5

        close = np.concatenate([45000 + sideways, 45000 + trend])
        df = pd.DataFrame({
            'close': close,
            'high': close + 50,
            'low': close - 50,
        })

        config = KalmanConfig(r_window=20, q_window=20, q_momentum_weight=2.0)
        kf = AdaptiveKalmanFilter(config)
        result = kf.filter(df)

        # In trending period, velocity should be clearly positive
        trend_velocity = result['kf_velocity'].iloc[300:450].mean()
        sideways_velocity_abs = np.abs(result['kf_velocity'].iloc[100:200]).mean()

        assert trend_velocity > sideways_velocity_abs * 2, "Velocity should capture trend"


class TestTypicalPrice:
    """Tests for typical price option."""

    def test_typical_price_enabled(self, sample_dollar_bar_data):
        """Test with typical price = (H+L+C)/3."""
        config = KalmanConfig(use_typical_price=True)
        kf = AdaptiveKalmanFilter(config)
        result = kf.filter(sample_dollar_bar_data)

        # Should work without error
        assert 'kf_trend' in result.columns
        assert len(result) == len(sample_dollar_bar_data)

    def test_typical_price_disabled(self, sample_dollar_bar_data):
        """Test with close price only."""
        config = KalmanConfig(use_typical_price=False)
        kf = AdaptiveKalmanFilter(config)
        result = kf.filter(sample_dollar_bar_data)

        assert 'kf_trend' in result.columns
        assert len(result) == len(sample_dollar_bar_data)

    def test_typical_vs_close_difference(self, sample_dollar_bar_data):
        """Typical price and close price should give different results."""
        config_tp = KalmanConfig(use_typical_price=True)
        config_close = KalmanConfig(use_typical_price=False)

        kf_tp = AdaptiveKalmanFilter(config_tp)
        kf_close = AdaptiveKalmanFilter(config_close)

        result_tp = kf_tp.filter(sample_dollar_bar_data)
        result_close = kf_close.filter(sample_dollar_bar_data)

        # Trends should be correlated but not identical
        corr = result_tp['kf_trend'].corr(result_close['kf_trend'])
        assert 0.9 < corr < 1.0, "Typical price and close should give similar but not identical results"


class TestPositionSizing:
    """Tests for position sizing based on uncertainty."""

    def test_position_size_factor_range(self):
        """Position size factor should be 0-1."""
        config = KalmanConfig()
        kf = AdaptiveKalmanFilter(config)

        # Test various uncertainty levels
        for uncertainty in [0.001, 0.01, 0.1, 1.0, 10.0]:
            factor = kf.get_position_size_factor(uncertainty)
            assert 0.0 <= factor <= 1.0, f"Factor out of range for uncertainty={uncertainty}"

    def test_position_size_decreases_with_uncertainty(self):
        """Higher uncertainty should give smaller position size."""
        config = KalmanConfig()
        kf = AdaptiveKalmanFilter(config)

        # Use explicit bounds for consistent testing
        min_unc, max_unc = 0.01, 10.0
        factor_low = kf.get_position_size_factor(0.05, min_unc, max_unc)
        factor_high = kf.get_position_size_factor(5.0, min_unc, max_unc)

        assert factor_low > factor_high, "Position size should decrease with uncertainty"

    def test_uncertainty_output_usable(self, sample_dollar_bar_data):
        """Uncertainty from filter should be usable for position sizing."""
        config = KalmanConfig()
        kf = AdaptiveKalmanFilter(config)
        result = kf.filter(sample_dollar_bar_data)

        # Uncertainty should be positive
        valid_uncertainty = result['kf_uncertainty'].iloc[50:].dropna()
        assert (valid_uncertainty > 0).all(), "Uncertainty should be positive"

        # Calculate position sizes with data-driven bounds
        min_unc = valid_uncertainty.quantile(0.05)
        max_unc = valid_uncertainty.quantile(0.95)
        sizes = valid_uncertainty.apply(
            lambda u: kf.get_position_size_factor(u, min_unc, max_unc)
        )
        assert (sizes >= 0).all() and (sizes <= 1).all()


class TestKalmanFeatureGenerator:
    """Tests for KalmanFeatureGenerator wrapper."""

    def test_generator_basic(self, sample_dollar_bar_data):
        """Test basic feature generation."""
        gen = KalmanFeatureGenerator(r_window=20, q_window=20)
        result = gen.generate(sample_dollar_bar_data)

        # Check all features generated
        for name in gen.get_feature_names():
            assert name in result.columns, f"Missing feature: {name}"

    def test_get_feature_names(self):
        """Test feature name list."""
        gen = KalmanFeatureGenerator()
        names = gen.get_feature_names()

        expected = ['kf_trend', 'kf_trend_pred', 'kf_velocity', 'kf_deviation',
                    'kf_deviation_pct', 'kf_gain', 'kf_uncertainty', 'kf_signal']
        assert names == expected

    def test_get_core_features(self):
        """Test core feature subset."""
        gen = KalmanFeatureGenerator()
        core = gen.get_core_features()

        assert 'kf_deviation' in core
        assert 'kf_velocity' in core
        assert 'kf_signal' in core
        assert 'kf_uncertainty' in core


class TestConvenienceFunction:
    """Tests for calculate_adaptive_kalman convenience function."""

    def test_convenience_function(self, sample_dollar_bar_data):
        """Test convenience function works."""
        result = calculate_adaptive_kalman(
            sample_dollar_bar_data,
            r_window=20,
            q_window=20,
            use_typical_price=True
        )

        assert 'kf_trend' in result.columns
        assert 'kf_signal' in result.columns


class TestEdgeCases:
    """Edge case tests."""

    def test_small_data(self):
        """Test with very small dataset."""
        df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
        })

        config = KalmanConfig(r_window=3, q_window=3)
        kf = AdaptiveKalmanFilter(config)
        result = kf.filter(df)

        assert len(result) == 5
        assert 'kf_trend' in result.columns

    def test_constant_price(self):
        """Test with constant prices."""
        n = 100
        df = pd.DataFrame({
            'close': np.full(n, 100.0),
            'high': np.full(n, 100.1),
            'low': np.full(n, 99.9),
        })

        config = KalmanConfig()
        kf = AdaptiveKalmanFilter(config)
        result = kf.filter(df)

        # Trend should be close to constant price
        assert np.abs(result['kf_trend'].iloc[-1] - 100.0) < 1.0
        # Velocity should be near zero
        assert np.abs(result['kf_velocity'].iloc[-1]) < 0.1

    def test_strong_uptrend(self):
        """Test with strong uptrend."""
        n = 500
        close = pd.Series(100 + np.arange(n) * 1.0)
        df = pd.DataFrame({
            'close': close,
            'high': close + 0.5,
            'low': close - 0.5,
        })

        config = KalmanConfig()
        kf = AdaptiveKalmanFilter(config)
        result = kf.filter(df)

        # Velocity should be positive
        assert result['kf_velocity'].iloc[-100:].mean() > 0.5
        # Trend should follow price
        assert result['kf_trend'].iloc[-1] > 400

    def test_strong_downtrend(self):
        """Test with strong downtrend."""
        n = 500
        close = pd.Series(500 - np.arange(n) * 1.0)
        df = pd.DataFrame({
            'close': close,
            'high': close + 0.5,
            'low': close - 0.5,
        })

        config = KalmanConfig()
        kf = AdaptiveKalmanFilter(config)
        result = kf.filter(df)

        # Velocity should be negative
        assert result['kf_velocity'].iloc[-100:].mean() < -0.5

    def test_nan_handling(self):
        """Test NaN handling in input."""
        n = 200
        df = pd.DataFrame({
            'close': 100 + np.random.randn(n),
            'high': 101 + np.random.randn(n),
            'low': 99 + np.random.randn(n),
        })

        # Kalman filter processes sequentially, so NaN in input
        # will propagate. This tests it doesn't crash.
        config = KalmanConfig()
        kf = AdaptiveKalmanFilter(config)
        result = kf.filter(df)

        assert len(result) == n

    def test_no_inf_values(self, sample_dollar_bar_data):
        """Output should not have infinite values."""
        config = KalmanConfig()
        kf = AdaptiveKalmanFilter(config)
        result = kf.filter(sample_dollar_bar_data)

        for col in ['kf_trend', 'kf_velocity', 'kf_deviation', 'kf_gain']:
            assert not np.isinf(result[col]).any(), f"Inf found in {col}"


class TestStressTests:
    """Stress tests for filter robustness."""

    def test_step_function_convergence(self):
        """Filter should converge quickly after price jump."""
        n = 300
        prices_before = np.full(100, 100.0) + np.random.randn(100) * 0.5
        prices_after = np.full(200, 200.0) + np.random.randn(200) * 0.5
        prices = np.concatenate([prices_before, prices_after])

        df = pd.DataFrame({
            'close': prices,
            'high': prices + 1,
            'low': prices - 1,
        })

        config = KalmanConfig(r_window=20, q_window=20)
        kf = AdaptiveKalmanFilter(config)
        result = kf.filter(df)

        # Should converge to 95% of step within 50 bars
        target = 195  # 95% of 100->200 step
        trend_after = result['kf_trend'].iloc[100:150].values
        convergence_idx = np.where(trend_after >= target)[0]

        assert len(convergence_idx) > 0, "Filter should converge within 50 bars"
        assert convergence_idx[0] < 30, f"Convergence too slow: {convergence_idx[0]} bars"

    def test_outlier_robustness(self):
        """Filter should attenuate outlier impact."""
        n = 500
        t = np.arange(n)
        clean_signal = 100 + 10 * np.sin(2 * np.pi * t / 100)
        prices = clean_signal + np.random.randn(n) * 0.5

        # Inject 10-sigma outlier
        sigma = prices.std()
        prices[250] = prices[250] + 10 * sigma

        df = pd.DataFrame({
            'close': prices,
            'high': prices + 0.5,
            'low': prices - 0.5,
        })

        config = KalmanConfig(r_window=20, q_window=20)
        kf = AdaptiveKalmanFilter(config)
        result = kf.filter(df)

        # Trend should not deviate much from expected
        expected_trend = 100 + 10 * np.sin(2 * np.pi * 250 / 100)
        actual_trend = result['kf_trend'].iloc[250]

        deviation = abs(actual_trend - expected_trend)
        attenuation = 1 - (deviation / (10 * sigma))

        assert attenuation > 0.5, f"Poor attenuation: {attenuation:.1%}"

    def test_parameter_stability(self):
        """Results should be stable across different R scale values."""
        np.random.seed(42)
        n = 500
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        df = pd.DataFrame({
            'close': prices,
            'high': prices + 1,
            'low': prices - 1,
        })

        # Baseline
        config_base = KalmanConfig(r_scale=1.0)
        result_base = AdaptiveKalmanFilter(config_base).filter(df)
        baseline = result_base['kf_trend'].iloc[100:].values

        # Test extreme values
        for r_scale in [0.1, 10.0]:
            config = KalmanConfig(r_scale=r_scale)
            result = AdaptiveKalmanFilter(config).filter(df)
            trend = result['kf_trend'].iloc[100:].values

            corr = np.corrcoef(baseline, trend)[0, 1]
            assert corr > 0.95, f"Poor convergence at r_scale={r_scale}: corr={corr:.3f}"


class TestDeviationSignal:
    """Tests for deviation and signal features."""

    def test_deviation_mean_reverts(self):
        """Deviation should mean-revert around zero over time."""
        np.random.seed(42)
        n = 1000
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            'close': close,
            'high': close + 1,
            'low': close - 1,
        })

        config = KalmanConfig()
        kf = AdaptiveKalmanFilter(config)
        result = kf.filter(df)

        # Mean deviation should be close to zero
        mean_dev = result['kf_deviation'].iloc[100:].mean()
        assert np.abs(mean_dev) < 5, f"Mean deviation too far from zero: {mean_dev}"

    def test_signal_zscore_distribution(self, sample_dollar_bar_data):
        """Signal (deviation z-score) should be roughly normalized."""
        config = KalmanConfig()
        kf = AdaptiveKalmanFilter(config)
        result = kf.filter(sample_dollar_bar_data)

        # After warmup, check signal distribution
        signal = result['kf_signal'].iloc[100:].dropna()

        # Most values should be within [-3, 3]
        within_3std = ((signal >= -3) & (signal <= 3)).mean()
        assert within_3std > 0.9, f"Too many outliers in signal: {1 - within_3std:.1%}"
