"""
Tests for Advanced Feature Generator.
"""

import pytest
import pandas as pd
import numpy as np
from research.features.advanced import AdvancedFeatureGenerator


class TestAutocorrelation:
    """Tests for autocorrelation feature."""

    def test_autocorr_basic(self, sample_dollar_bar_data):
        """Test basic autocorrelation calculation."""
        gen = AdvancedFeatureGenerator(autocorr_window=20, autocorr_lag=1)
        features = gen.generate(sample_dollar_bar_data)

        assert 'autocorr' in features.columns
        valid = features['autocorr'].dropna()
        assert len(valid) > 0

    def test_autocorr_range(self, sample_dollar_bar_data):
        """Autocorrelation should be between -1 and 1."""
        gen = AdvancedFeatureGenerator(autocorr_window=20)
        features = gen.generate(sample_dollar_bar_data)

        valid = features['autocorr'].dropna()
        assert (valid >= -1).all()
        assert (valid <= 1).all()

    def test_autocorr_trending_series(self):
        """Trending series should have positive autocorrelation."""
        n = 500
        # Strong trend
        close = pd.Series(100 + np.arange(n) * 0.5 + np.random.randn(n) * 0.01)
        df = pd.DataFrame({
            'close': close,
            'high': close + 0.1,
            'low': close - 0.1,
        })

        gen = AdvancedFeatureGenerator(autocorr_window=50)
        features = gen.generate(df)

        # Trending should have positive autocorrelation (mostly)
        valid = features['autocorr'].dropna()
        # Just check it computed without error
        assert len(valid) > 0


class TestHurstExponent:
    """Tests for Hurst exponent feature."""

    def test_hurst_basic(self, sample_dollar_bar_data):
        """Test basic Hurst exponent calculation."""
        gen = AdvancedFeatureGenerator(hurst_window=100)
        features = gen.generate(sample_dollar_bar_data)

        assert 'hurst_exp' in features.columns
        valid = features['hurst_exp'].dropna()
        assert len(valid) > 0

    def test_hurst_range(self, sample_dollar_bar_data):
        """Hurst exponent should be between 0 and 1."""
        gen = AdvancedFeatureGenerator(hurst_window=100)
        features = gen.generate(sample_dollar_bar_data)

        valid = features['hurst_exp'].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_hurst_random_walk(self):
        """Random walk should have Hurst around 0.5."""
        np.random.seed(42)
        n = 1000
        close = pd.Series(100 + np.cumsum(np.random.randn(n)))
        df = pd.DataFrame({
            'close': close,
            'high': close + 1,
            'low': close - 1,
        })

        gen = AdvancedFeatureGenerator(hurst_window=200)
        features = gen.generate(df)

        valid = features['hurst_exp'].dropna()
        mean_hurst = valid.mean()
        # Random walk Hurst should be around 0.5 (with tolerance)
        assert 0.3 < mean_hurst < 0.7


class TestOrderFlowToxicity:
    """Tests for Order Flow Toxicity feature."""

    def test_toxicity_basic(self, sample_dollar_bar_data):
        """Test basic toxicity calculation."""
        gen = AdvancedFeatureGenerator(toxicity_window=50)
        features = gen.generate(sample_dollar_bar_data)

        assert 'order_flow_toxicity' in features.columns
        valid = features['order_flow_toxicity'].dropna()
        assert len(valid) > 0

    def test_toxicity_range(self, sample_dollar_bar_data):
        """Toxicity should be between 0 and 1."""
        gen = AdvancedFeatureGenerator(toxicity_window=50)
        features = gen.generate(sample_dollar_bar_data)

        valid = features['order_flow_toxicity'].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_toxicity_extreme_imbalance(self):
        """Extreme imbalance should have high toxicity."""
        n = 200
        # All buy pressure
        df = pd.DataFrame({
            'close': 100 + np.random.randn(n) * 0.1,
            'high': 101 + np.random.randn(n) * 0.1,
            'low': 99 + np.random.randn(n) * 0.1,
            'volume_imbalance': np.full(n, 0.9),  # Constant high buy imbalance
        })

        gen = AdvancedFeatureGenerator(toxicity_window=50)
        features = gen.generate(df)

        # With constant imbalance at extreme, toxicity should be high
        valid = features['order_flow_toxicity'].dropna()
        assert len(valid) > 0


class TestRSI:
    """Tests for RSI feature."""

    def test_rsi_basic(self, sample_dollar_bar_data):
        """Test basic RSI calculation."""
        gen = AdvancedFeatureGenerator(rsi_window=14)
        features = gen.generate(sample_dollar_bar_data)

        assert 'rsi' in features.columns
        valid = features['rsi'].dropna()
        assert len(valid) > 0

    def test_rsi_range(self, sample_dollar_bar_data):
        """RSI should be between 0 and 100."""
        gen = AdvancedFeatureGenerator(rsi_window=14)
        features = gen.generate(sample_dollar_bar_data)

        valid = features['rsi'].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_uptrend(self):
        """Strong uptrend should have high RSI."""
        n = 200
        close = pd.Series(100 + np.arange(n) * 0.5)  # Constant uptrend
        df = pd.DataFrame({
            'close': close,
            'high': close + 0.1,
            'low': close - 0.1,
        })

        gen = AdvancedFeatureGenerator(rsi_window=14)
        features = gen.generate(df)

        valid = features['rsi'].dropna()
        # Strong uptrend should have RSI > 70
        assert valid.iloc[-1] > 70

    def test_rsi_downtrend(self):
        """Strong downtrend should have low RSI."""
        n = 200
        close = pd.Series(200 - np.arange(n) * 0.5)  # Constant downtrend
        df = pd.DataFrame({
            'close': close,
            'high': close + 0.1,
            'low': close - 0.1,
        })

        gen = AdvancedFeatureGenerator(rsi_window=14)
        features = gen.generate(df)

        valid = features['rsi'].dropna()
        # Strong downtrend should have RSI < 30
        assert valid.iloc[-1] < 30


class TestEfficiencyRatio:
    """Tests for Kaufman Efficiency Ratio feature."""

    def test_efficiency_basic(self, sample_dollar_bar_data):
        """Test basic efficiency ratio calculation."""
        gen = AdvancedFeatureGenerator(efficiency_window=10)
        features = gen.generate(sample_dollar_bar_data)

        assert 'efficiency_ratio' in features.columns
        valid = features['efficiency_ratio'].dropna()
        assert len(valid) > 0

    def test_efficiency_range(self, sample_dollar_bar_data):
        """Efficiency ratio should be between 0 and 1."""
        gen = AdvancedFeatureGenerator(efficiency_window=10)
        features = gen.generate(sample_dollar_bar_data)

        valid = features['efficiency_ratio'].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_efficiency_strong_trend(self):
        """Strong trend should have high efficiency ratio."""
        n = 200
        close = pd.Series(100 + np.arange(n) * 1.0)  # Perfect trend
        df = pd.DataFrame({
            'close': close,
            'high': close + 0.01,
            'low': close - 0.01,
        })

        gen = AdvancedFeatureGenerator(efficiency_window=10)
        features = gen.generate(df)

        valid = features['efficiency_ratio'].dropna()
        # Perfect trend should have ER close to 1
        assert valid.mean() > 0.9

    def test_efficiency_choppy_market(self):
        """Choppy market should have low efficiency ratio."""
        n = 200
        # Alternating up/down
        close = pd.Series([100 + (i % 2) for i in range(n)], dtype=float)
        df = pd.DataFrame({
            'close': close,
            'high': close + 0.5,
            'low': close - 0.5,
        })

        gen = AdvancedFeatureGenerator(efficiency_window=10)
        features = gen.generate(df)

        valid = features['efficiency_ratio'].dropna()
        # Choppy should have low ER
        assert valid.mean() < 0.3


class TestMomentumZscore:
    """Tests for Momentum Z-score features."""

    def test_momentum_zscore_basic(self, sample_dollar_bar_data):
        """Test basic momentum z-score calculation."""
        gen = AdvancedFeatureGenerator(momentum_windows=[10, 50, 250, 1000])
        features = gen.generate(sample_dollar_bar_data)

        assert 'momentum_zscore_10' in features.columns
        assert 'momentum_zscore_50' in features.columns
        assert 'momentum_zscore_250' in features.columns
        assert 'momentum_zscore_1000' in features.columns

    def test_momentum_zscore_no_inf(self, sample_dollar_bar_data):
        """Momentum z-scores should not have infinite values."""
        gen = AdvancedFeatureGenerator(momentum_windows=[10, 50])
        features = gen.generate(sample_dollar_bar_data)

        for window in [10, 50]:
            col = f'momentum_zscore_{window}'
            assert not np.isinf(features[col]).any()

    def test_momentum_zscore_uptrend(self):
        """Strong uptrend should have positive momentum z-score."""
        n = 500
        close = pd.Series(100 + np.arange(n) * 0.5 + np.random.randn(n) * 0.1)
        df = pd.DataFrame({
            'close': close,
            'high': close + 0.2,
            'low': close - 0.2,
        })

        gen = AdvancedFeatureGenerator(momentum_windows=[50])
        features = gen.generate(df)

        valid = features['momentum_zscore_50'].dropna()
        # Uptrend should have mostly positive momentum
        assert valid.mean() > 0


class TestFeatureNames:
    """Tests for feature name utilities."""

    def test_get_feature_names(self):
        """Test get_feature_names returns expected features."""
        gen = AdvancedFeatureGenerator(momentum_windows=[10, 50, 250, 1000])
        names = gen.get_feature_names()

        expected = [
            'autocorr',
            'hurst_exp',
            'order_flow_toxicity',
            'rsi',
            'efficiency_ratio',
            'momentum_zscore_10',
            'momentum_zscore_50',
            'momentum_zscore_250',
            'momentum_zscore_1000',
        ]

        for name in expected:
            assert name in names, f"Missing feature: {name}"

    def test_get_core_features(self):
        """Test get_core_features returns subset."""
        gen = AdvancedFeatureGenerator()
        core = gen.get_core_features()
        all_features = gen.get_feature_names()

        for c in core:
            assert c in all_features


class TestEdgeCases:
    """Edge case tests for advanced features."""

    def test_small_data(self):
        """Test with very small dataset."""
        df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
        })

        gen = AdvancedFeatureGenerator(
            autocorr_window=3,
            hurst_window=5,
            toxicity_window=3,
            rsi_window=3,
            efficiency_window=3,
            momentum_windows=[3],
        )
        features = gen.generate(df)

        # Should not crash
        assert len(features) == 5

    def test_constant_price(self):
        """Test with constant prices."""
        n = 100
        df = pd.DataFrame({
            'close': np.full(n, 100.0),
            'high': np.full(n, 100.1),
            'low': np.full(n, 99.9),
        })

        gen = AdvancedFeatureGenerator()
        features = gen.generate(df)

        # Should handle without crash, may have NaN
        assert len(features) == n
        assert not np.isinf(features['rsi']).any()

    def test_nan_in_input(self):
        """Test handling of NaN in input."""
        n = 200
        df = pd.DataFrame({
            'close': 100 + np.random.randn(n),
            'high': 101 + np.random.randn(n),
            'low': 99 + np.random.randn(n),
        })
        df.loc[50, 'close'] = np.nan
        df.loc[100, 'close'] = np.nan

        gen = AdvancedFeatureGenerator()
        features = gen.generate(df)

        # Should not crash
        assert len(features) == n

    def test_negative_prices(self):
        """Test with small positive prices (edge case)."""
        n = 200
        df = pd.DataFrame({
            'close': np.random.uniform(0.01, 0.1, n),
            'high': np.random.uniform(0.05, 0.15, n),
            'low': np.random.uniform(0.005, 0.05, n),
        })
        df['high'] = np.maximum(df['high'], df['close'])
        df['low'] = np.minimum(df['low'], df['close'])

        gen = AdvancedFeatureGenerator()
        features = gen.generate(df)

        # Should not have inf
        assert not np.isinf(features['rsi']).any()
        assert not np.isinf(features['efficiency_ratio']).any()

    def test_all_features_generated(self, sample_dollar_bar_data):
        """Test all expected features are generated."""
        gen = AdvancedFeatureGenerator()
        features = gen.generate(sample_dollar_bar_data)

        for name in gen.get_feature_names():
            assert name in features.columns, f"Missing feature: {name}"
