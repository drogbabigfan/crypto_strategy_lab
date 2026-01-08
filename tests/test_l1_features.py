"""
Tests for L1 Feature Generator.
"""

import pytest
import pandas as pd
import numpy as np
from research.features.l1 import L1FeatureGenerator


def test_l1_log_transforms(sample_dollar_bar_data):
    """Test log transforms: log_volume, log_tick_count, log_duration."""
    gen = L1FeatureGenerator()
    features = gen.generate(sample_dollar_bar_data)

    # Check all log columns exist
    # Note: log_dollar_value removed - constant in Dollar Bars
    log_cols = ['log_volume', 'log_tick_count', 'log_duration']
    for col in log_cols:
        assert col in features.columns, f"Missing column: {col}"

    # Log values should be positive (log1p of positive numbers)
    assert (features['log_volume'] >= 0).all()
    assert (features['log_tick_count'] >= 0).all()

    # Verify calculation
    expected_log_volume = np.log1p(sample_dollar_bar_data['volume'].iloc[0])
    assert np.isclose(features['log_volume'].iloc[0], expected_log_volume)


def test_vwap_calculation(sample_dollar_bar_data):
    """Test VWAP calculation: dollar_value / volume."""
    gen = L1FeatureGenerator()
    features = gen.generate(sample_dollar_bar_data)

    assert 'vwap' in features.columns

    # VWAP = dollar_value / volume
    expected_vwap = (
        sample_dollar_bar_data['dollar_value'].iloc[0] /
        sample_dollar_bar_data['volume'].iloc[0]
    )
    assert np.isclose(features['vwap'].iloc[0], expected_vwap, rtol=1e-5)


def test_vwap_deviation(sample_dollar_bar_data):
    """Test VWAP deviation: (close - vwap) / close."""
    gen = L1FeatureGenerator()
    features = gen.generate(sample_dollar_bar_data)

    assert 'vwap_deviation' in features.columns

    # Calculate expected
    vwap = sample_dollar_bar_data['dollar_value'].iloc[0] / sample_dollar_bar_data['volume'].iloc[0]
    close = sample_dollar_bar_data['close'].iloc[0]
    expected = (close - vwap) / close

    assert np.isclose(features['vwap_deviation'].iloc[0], expected, rtol=1e-5)


def test_volume_imbalance(sample_dollar_bar_data):
    """Test volume imbalance: net_imbalance / dollar_value."""
    gen = L1FeatureGenerator()
    features = gen.generate(sample_dollar_bar_data)

    assert 'volume_imbalance' in features.columns

    # volume_imbalance = net_imbalance / dollar_value
    expected = (
        sample_dollar_bar_data['net_imbalance'].iloc[0] /
        sample_dollar_bar_data['dollar_value'].iloc[0]
    )
    assert np.isclose(features['volume_imbalance'].iloc[0], expected, rtol=1e-5)

    # Imbalance should be between -1 and 1
    assert (features['volume_imbalance'] >= -1).all()
    assert (features['volume_imbalance'] <= 1).all()


def test_buy_ratio(sample_dollar_bar_data):
    """Test buy ratio: buy_dollar_vol / dollar_value."""
    gen = L1FeatureGenerator()
    features = gen.generate(sample_dollar_bar_data)

    assert 'buy_ratio' in features.columns

    # buy_ratio should be between 0 and 1
    assert (features['buy_ratio'] >= 0).all()
    assert (features['buy_ratio'] <= 1).all()


def test_bar_range(sample_dollar_bar_data):
    """Test bar range: (high - low) / close."""
    gen = L1FeatureGenerator()
    features = gen.generate(sample_dollar_bar_data)

    assert 'bar_range' in features.columns

    expected = (
        (sample_dollar_bar_data['high'].iloc[0] - sample_dollar_bar_data['low'].iloc[0]) /
        sample_dollar_bar_data['close'].iloc[0]
    )
    assert np.isclose(features['bar_range'].iloc[0], expected, rtol=1e-5)

    # Range should be non-negative
    assert (features['bar_range'] >= 0).all()


def test_trade_intensity(sample_dollar_bar_data):
    """Test trade intensity: dollar_value / duration."""
    gen = L1FeatureGenerator()
    features = gen.generate(sample_dollar_bar_data)

    assert 'trade_intensity' in features.columns
    assert 'log_trade_intensity' in features.columns

    # Trade intensity should be positive
    assert (features['trade_intensity'] > 0).all()


def test_avg_trade_size(sample_dollar_bar_data):
    """Test average trade size: dollar_value / tick_count."""
    gen = L1FeatureGenerator()
    features = gen.generate(sample_dollar_bar_data)

    assert 'avg_trade_size' in features.columns
    assert 'log_avg_trade_size' in features.columns

    # Average trade size should be positive
    assert (features['avg_trade_size'] > 0).all()


def test_feature_names(sample_dollar_bar_data):
    """Test get_feature_names returns all generated features."""
    gen = L1FeatureGenerator()
    features = gen.generate(sample_dollar_bar_data)

    feature_names = gen.get_feature_names()

    for name in feature_names:
        assert name in features.columns, f"Feature {name} not in generated columns"


def test_core_features(sample_dollar_bar_data):
    """Test get_core_features returns subset of features."""
    gen = L1FeatureGenerator()
    features = gen.generate(sample_dollar_bar_data)

    core_features = gen.get_core_features()

    # Core features should be subset of all features
    all_features = gen.get_feature_names()
    for core in core_features:
        assert core in all_features, f"Core feature {core} not in feature list"
        assert core in features.columns, f"Core feature {core} not generated"


def test_no_nan_in_output(sample_dollar_bar_data):
    """Test that L1 features don't produce NaN values."""
    gen = L1FeatureGenerator()
    features = gen.generate(sample_dollar_bar_data)

    feature_names = gen.get_feature_names()

    for name in feature_names:
        nan_count = features[name].isna().sum()
        assert nan_count == 0, f"Feature {name} has {nan_count} NaN values"


def test_epsilon_prevents_division_by_zero():
    """Test that epsilon parameter prevents division by zero."""
    # Create data with zero values
    df = pd.DataFrame({
        'open': [100, 100],
        'high': [101, 101],
        'low': [99, 99],
        'close': [100, 100],
        'volume': [0, 1],  # Zero volume in first row
        'dollar_value': [0, 100],  # Zero dollar value
        'tick_count': [0, 10],  # Zero tick count
        'duration': [0, 60],  # Zero duration
        'buy_dollar_vol': [0, 60],
        'sell_dollar_vol': [0, 40],
        'net_imbalance': [0, 20],
    })

    gen = L1FeatureGenerator(epsilon=1e-10)
    features = gen.generate(df)

    # Should not have inf values
    for col in gen.get_feature_names():
        assert not np.isinf(features[col]).any(), f"Inf in {col}"
