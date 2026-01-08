"""
Tests for Adaptive Stitching module.
"""

import pytest
import pandas as pd
import numpy as np
from research.features.stitching import AdaptiveStitcher, MultiFeatureStitcher


def test_adaptive_stitching_cross_file():
    """
    Test that stitching maintains continuity across file boundaries.

    Scenario:
    - Chunk 1: First file of data
    - Chunk 2: Second file (simulates next month's data)

    Z-score should be computable from the start of Chunk 2
    because buffer from Chunk 1 provides history.
    """
    np.random.seed(42)

    # Generate Chunk 1 data
    chunk1 = pd.DataFrame({
        "close": np.random.normal(100, 5, 100).cumsum(),
        "volume": np.random.uniform(10, 20, 100)
    })

    # Generate Chunk 2 data (continuation)
    chunk2 = pd.DataFrame({
        "close": np.random.normal(chunk1["close"].iloc[-1], 5, 100).cumsum(),
        "volume": np.random.uniform(10, 20, 100)
    })

    # Init Stitcher with close column for Z-score
    stitcher = AdaptiveStitcher(buffer_size=50, base_halflife=20, zscore_columns=['close'])

    # Process Chunk 1
    processed_1 = stitcher.process(chunk1)

    # Process Chunk 2 (with stitching)
    processed_2 = stitcher.process(chunk2)

    # Assertions
    # 1. Z-score column should exist
    assert 'close_zscore' in processed_2.columns

    # 2. First element of chunk 2 should not be NaN (buffer provides history)
    assert not np.isnan(processed_2["close_zscore"].iloc[0]), \
        "Stitching failed: Z-score is NaN at start of Chunk 2"

    # 3. Z-scores should be reasonable (mostly within -4 to 4)
    zscores = processed_2["close_zscore"]
    assert zscores.abs().max() < 10, "Z-scores are unreasonably large"


def test_buffer_management():
    """Verify that stitcher maintains a buffer of correct size."""
    df = pd.DataFrame({"close": range(100), "volume": range(100)})
    stitcher = AdaptiveStitcher(buffer_size=10)
    stitcher.process(df)

    buffer = stitcher.get_buffer()
    assert len(buffer) == 10
    assert buffer.iloc[-1]["close"] == 99


def test_buffer_size_larger_than_data():
    """Test when buffer size is larger than input data."""
    df = pd.DataFrame({"close": range(5), "volume": range(5)})
    stitcher = AdaptiveStitcher(buffer_size=100)
    stitcher.process(df)

    buffer = stitcher.get_buffer()
    assert len(buffer) == 5  # Should keep all data


def test_reset():
    """Test reset clears the buffer."""
    df = pd.DataFrame({"close": range(100), "volume": range(100)})
    stitcher = AdaptiveStitcher(buffer_size=50)
    stitcher.process(df)

    assert stitcher.get_buffer_size() == 50

    stitcher.reset()

    assert stitcher.get_buffer_size() == 0


def test_multi_feature_stitcher():
    """Test MultiFeatureStitcher with multiple columns."""
    np.random.seed(42)

    # Create data with multiple features
    df = pd.DataFrame({
        "close": np.random.normal(100, 5, 200).cumsum(),
        "volume": np.random.uniform(10, 100, 200),
        "trade_intensity": np.random.uniform(1, 10, 200),
    })

    stitcher = MultiFeatureStitcher(
        buffer_size=50,
        halflife_map={'close': 20, 'volume': 30},
        default_halflife=25
    )

    result = stitcher.process(df, columns=['close', 'volume', 'trade_intensity'])

    # Check Z-score columns exist
    assert 'close_zscore' in result.columns
    assert 'volume_zscore' in result.columns
    assert 'trade_intensity_zscore' in result.columns

    # Check no NaN after warmup period
    for col in ['close_zscore', 'volume_zscore', 'trade_intensity_zscore']:
        # First few values might be NaN due to EWM warmup
        assert result[col].iloc[50:].isna().sum() == 0, f"NaN values in {col} after warmup"


def test_multi_feature_stitcher_cross_file():
    """Test MultiFeatureStitcher maintains continuity across chunks."""
    np.random.seed(42)

    chunk1 = pd.DataFrame({
        "close": np.random.normal(100, 5, 100).cumsum(),
        "volume": np.random.uniform(10, 100, 100),
    })

    chunk2 = pd.DataFrame({
        "close": np.random.normal(chunk1["close"].iloc[-1], 5, 100).cumsum(),
        "volume": np.random.uniform(10, 100, 100),
    })

    stitcher = MultiFeatureStitcher(buffer_size=50)

    _ = stitcher.process(chunk1, columns=['close', 'volume'])
    result2 = stitcher.process(chunk2, columns=['close', 'volume'])

    # First row of chunk 2 should have valid Z-scores
    assert not np.isnan(result2["close_zscore"].iloc[0])
    assert not np.isnan(result2["volume_zscore"].iloc[0])


def test_ewm_zscore_properties():
    """Test that EWM-based Z-scores have expected statistical properties."""
    np.random.seed(42)

    # Generate stationary data
    n = 1000
    data = np.random.normal(0, 1, n)
    df = pd.DataFrame({"value": data})

    stitcher = AdaptiveStitcher(buffer_size=100, base_halflife=50, zscore_columns=['value'])
    result = stitcher.process(df)

    # After warmup, Z-scores should have mean close to 0 and std close to 1
    zscores = result["value_zscore"].iloc[200:]  # Skip warmup

    assert abs(zscores.mean()) < 0.2, f"Z-score mean too far from 0: {zscores.mean()}"
    assert 0.5 < zscores.std() < 2.0, f"Z-score std unexpected: {zscores.std()}"
