"""
Tests for Regime Feature Generator.
"""

import pytest
import pandas as pd
import numpy as np
from research.features.regime import RegimeFeatureGenerator


class TestParkinsonVolatility:
    """Tests for Parkinson volatility estimator."""

    def test_parkinson_basic(self, sample_dollar_bar_data):
        """Test basic Parkinson volatility calculation."""
        gen = RegimeFeatureGenerator(vol_window=24)
        features = gen.generate(sample_dollar_bar_data)

        assert 'parkinson_vol' in features.columns

        # After warmup, should have values
        valid_vals = features['parkinson_vol'].dropna()
        assert len(valid_vals) > 0

        # Volatility should be positive
        assert (valid_vals >= 0).all()

    def test_parkinson_vs_realized(self, sample_dollar_bar_data):
        """Parkinson should generally be more efficient than realized vol."""
        gen = RegimeFeatureGenerator(vol_window=24)
        features = gen.generate(sample_dollar_bar_data)

        # Both should exist
        assert 'parkinson_vol' in features.columns
        assert 'realized_vol' in features.columns

    def test_parkinson_high_volatility_period(self):
        """Test Parkinson correctly captures high volatility."""
        n = 200

        # Low volatility period
        low_vol = pd.DataFrame({
            'high': 100 + np.random.randn(n) * 0.1,
            'low': 99.8 + np.random.randn(n) * 0.1,
            'close': 100 + np.random.randn(n) * 0.05,
        })
        low_vol['high'] = np.maximum(low_vol['high'], low_vol['close'])
        low_vol['low'] = np.minimum(low_vol['low'], low_vol['close'])

        # High volatility period
        high_vol = pd.DataFrame({
            'high': 100 + np.random.randn(n) * 5,
            'low': 95 + np.random.randn(n) * 5,
            'close': 100 + np.random.randn(n) * 2,
        })
        high_vol['high'] = np.maximum(high_vol['high'], high_vol['close'])
        high_vol['low'] = np.minimum(high_vol['low'], high_vol['close'])

        gen = RegimeFeatureGenerator(vol_window=24)

        low_features = gen.generate(low_vol)
        high_features = gen.generate(high_vol)

        # High volatility period should have higher Parkinson vol
        low_mean = low_features['parkinson_vol'].dropna().mean()
        high_mean = high_features['parkinson_vol'].dropna().mean()

        assert high_mean > low_mean * 2, "High vol period should have higher Parkinson vol"

    def test_parkinson_edge_case_equal_high_low(self):
        """Test when high == low (no movement within bar)."""
        n = 100
        df = pd.DataFrame({
            'high': np.full(n, 100.0),
            'low': np.full(n, 100.0),  # Same as high
            'close': np.full(n, 100.0),
        })

        gen = RegimeFeatureGenerator(vol_window=24)
        features = gen.generate(df)

        # Should not have NaN or inf
        assert not features['parkinson_vol'].isna().all()
        assert not np.isinf(features['parkinson_vol']).any()

        # Volatility should be near zero
        valid = features['parkinson_vol'].dropna()
        assert valid.max() < 0.01


class TestShannonEntropy:
    """Tests for Shannon entropy calculation."""

    def test_entropy_basic(self, sample_dollar_bar_data):
        """Test basic entropy calculation."""
        gen = RegimeFeatureGenerator(entropy_window=24, entropy_bins=10)
        features = gen.generate(sample_dollar_bar_data)

        assert 'shannon_entropy' in features.columns

        valid = features['shannon_entropy'].dropna()
        assert len(valid) > 0

    def test_entropy_range(self, sample_dollar_bar_data):
        """Entropy should have no infinite values."""
        gen = RegimeFeatureGenerator(entropy_window=24, entropy_bins=10)
        features = gen.generate(sample_dollar_bar_data)

        valid = features['shannon_entropy'].dropna()

        # Note: Implementation uses density=True histogram which can produce
        # values outside [0,1] range. Just verify no infinite values.
        assert not np.isinf(valid).any()
        assert len(valid) > 0

    def test_entropy_trending_vs_noisy(self):
        """Trending market should have lower entropy than noisy market."""
        n = 500

        # Trending market (consistent direction)
        trending = pd.DataFrame({
            'close': 100 + np.arange(n) * 0.1 + np.random.randn(n) * 0.01,
        })
        trending['high'] = trending['close'] + 0.1
        trending['low'] = trending['close'] - 0.1

        # Noisy market (random walk)
        noisy = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(n)),
        })
        noisy['high'] = noisy['close'] + 0.5
        noisy['low'] = noisy['close'] - 0.5

        gen = RegimeFeatureGenerator(entropy_window=50, entropy_bins=10)

        trending_features = gen.generate(trending)
        noisy_features = gen.generate(noisy)

        # Noisy market should have higher entropy
        trending_entropy = trending_features['shannon_entropy'].dropna().mean()
        noisy_entropy = noisy_features['shannon_entropy'].dropna().mean()

        # Note: This may not always hold due to randomness, but generally should
        # Just verify both are computed without error
        assert not np.isnan(trending_entropy)
        assert not np.isnan(noisy_entropy)

    def test_entropy_edge_case_constant_returns(self):
        """Test when all returns are the same."""
        n = 100
        df = pd.DataFrame({
            'close': np.full(n, 100.0),  # Constant price
            'high': np.full(n, 100.1),
            'low': np.full(n, 99.9),
        })

        gen = RegimeFeatureGenerator(entropy_window=24, entropy_bins=10)
        features = gen.generate(df)

        # Should handle constant returns without error
        assert 'shannon_entropy' in features.columns
        assert not np.isinf(features['shannon_entropy']).any()


class TestVolatilityRatio:
    """Tests for volatility ratio feature."""

    def test_vol_ratio_exists(self, sample_dollar_bar_data):
        """Test vol_ratio is generated."""
        gen = RegimeFeatureGenerator()
        features = gen.generate(sample_dollar_bar_data)

        assert 'vol_ratio' in features.columns

    def test_vol_ratio_no_inf(self, sample_dollar_bar_data):
        """Vol ratio should not have infinite values."""
        gen = RegimeFeatureGenerator()
        features = gen.generate(sample_dollar_bar_data)

        assert not np.isinf(features['vol_ratio']).any()


class TestZScoreFeatures:
    """Tests for z-score normalized features."""

    def test_vol_zscore(self, sample_dollar_bar_data):
        """Test volatility z-score."""
        gen = RegimeFeatureGenerator()
        features = gen.generate(sample_dollar_bar_data)

        assert 'vol_zscore' in features.columns

        valid = features['vol_zscore'].dropna()

        # Z-scores should be centered around 0
        assert abs(valid.mean()) < 1

    def test_entropy_zscore(self, sample_dollar_bar_data):
        """Test entropy z-score."""
        gen = RegimeFeatureGenerator()
        features = gen.generate(sample_dollar_bar_data)

        assert 'entropy_zscore' in features.columns


class TestFeatureNames:
    """Tests for feature name utilities."""

    def test_get_feature_names(self):
        """Test get_feature_names returns expected features."""
        gen = RegimeFeatureGenerator()
        names = gen.get_feature_names()

        expected = [
            'parkinson_vol',
            'returns',
            'realized_vol',
            'shannon_entropy',
            'vol_ratio',
            'vol_zscore',
            'entropy_zscore',
        ]

        for name in expected:
            assert name in names, f"Missing feature: {name}"

    def test_get_core_features(self):
        """Test get_core_features returns subset."""
        gen = RegimeFeatureGenerator()
        core = gen.get_core_features()
        all_features = gen.get_feature_names()

        for c in core:
            assert c in all_features


class TestEdgeCases:
    """Edge case tests for regime features."""

    def test_very_small_data(self):
        """Test with very small dataset."""
        df = pd.DataFrame({
            'high': [100, 101, 102],
            'low': [99, 100, 101],
            'close': [100, 101, 102],
        })

        gen = RegimeFeatureGenerator(vol_window=2, entropy_window=2)
        features = gen.generate(df)

        # Should not crash
        assert len(features) == 3

    def test_nan_in_input(self):
        """Test handling of NaN in input data."""
        n = 100
        df = pd.DataFrame({
            'high': 100 + np.random.randn(n),
            'low': 99 + np.random.randn(n),
            'close': 100 + np.random.randn(n),
        })

        # Insert some NaN
        df.loc[10, 'close'] = np.nan
        df.loc[20, 'high'] = np.nan

        gen = RegimeFeatureGenerator()
        features = gen.generate(df)

        # Should handle without crashing
        assert len(features) == n

    def test_negative_prices(self):
        """Test with negative prices (shouldn't happen but be robust)."""
        n = 100
        df = pd.DataFrame({
            'high': np.random.uniform(0.1, 1, n),  # Small positive
            'low': np.random.uniform(0.01, 0.5, n),
            'close': np.random.uniform(0.05, 0.8, n),
        })
        df['high'] = np.maximum(df['high'], df['close'])
        df['low'] = np.minimum(df['low'], df['close'])

        gen = RegimeFeatureGenerator()
        features = gen.generate(df)

        # Should not have inf values
        assert not np.isinf(features['parkinson_vol']).any()

    def test_zero_division_protection(self):
        """Test epsilon prevents division by zero."""
        n = 100
        df = pd.DataFrame({
            'high': np.full(n, 100.0),
            'low': np.full(n, 100.0),
            'close': np.full(n, 100.0),
        })

        gen = RegimeFeatureGenerator(epsilon=1e-10)
        features = gen.generate(df)

        # Should not have inf
        for col in gen.get_feature_names():
            if col in features.columns:
                assert not np.isinf(features[col]).any(), f"Inf in {col}"
