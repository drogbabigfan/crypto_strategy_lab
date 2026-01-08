"""
Pytest fixtures for feature engineering tests.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_dollar_bar_data():
    """
    Generates sample Dollar Bar DataFrame matching ETL output schema.

    Columns match DynamicDollarBar from etl/internal/bars/models.go:
        - start_time, end_time: int64 (ms)
        - open, high, low, close: float64
        - volume: float64 (BTC)
        - dollar_value: float64 (USD)
        - tick_count: int64
        - threshold_used, duration: float64
        - buy_dollar_vol, sell_dollar_vol, net_imbalance: float64
    """
    n = 1000
    base_time = 1704067200000  # 2024-01-01 00:00:00 UTC in ms

    # Generate realistic BTC prices around $45,000
    base_price = 45000
    prices = base_price + np.cumsum(np.random.randn(n) * 50)

    # Generate OHLC
    opens = prices + np.random.randn(n) * 10
    highs = np.maximum(opens, prices) + np.abs(np.random.randn(n) * 20)
    lows = np.minimum(opens, prices) - np.abs(np.random.randn(n) * 20)
    closes = prices

    # Volume and dollar value
    volumes = np.random.uniform(0.5, 5.0, n)  # BTC
    dollar_values = volumes * closes  # Approximately

    # Imbalance (60% buy on average)
    buy_ratios = np.random.uniform(0.4, 0.8, n)
    buy_dollar_vols = dollar_values * buy_ratios
    sell_dollar_vols = dollar_values * (1 - buy_ratios)
    net_imbalances = buy_dollar_vols - sell_dollar_vols

    # Timing
    durations = np.random.uniform(60, 3600, n)  # 1 min to 1 hour
    tick_counts = np.random.randint(50, 500, n)

    # Thresholds (around $20M based on 50 bars/day target)
    thresholds = np.full(n, 20_000_000.0) + np.random.randn(n) * 1_000_000

    # High/Low timestamps (for backtest precision)
    start_times = base_time + np.arange(n) * 1800000
    high_times = start_times + np.random.randint(0, 1800000, n)
    low_times = start_times + np.random.randint(0, 1800000, n)

    df = pd.DataFrame({
        'start_time': start_times,  # 30 min intervals
        'end_time': base_time + np.arange(n) * 1800000 + (durations * 1000).astype(int),
        'open': opens,
        'high': highs,
        'high_time': high_times,  # When high occurred
        'low': lows,
        'low_time': low_times,  # When low occurred
        'close': closes,
        'volume': volumes,
        'dollar_value': dollar_values,
        'tick_count': tick_counts,
        'threshold_used': thresholds,
        'duration': durations,
        'buy_dollar_vol': buy_dollar_vols,
        'sell_dollar_vol': sell_dollar_vols,
        'net_imbalance': net_imbalances,
    })

    return df


@pytest.fixture
def sample_tib_data():
    """
    Legacy fixture for backward compatibility.
    Maps to Dollar Bar format.
    """
    return sample_dollar_bar_data()


@pytest.fixture
def sample_trades_data():
    """Generates sample raw trades for verification."""
    return pd.DataFrame({
        'price': [100, 101, 100, 99, 99],
        'quantity': [1, 1, 1, 1, 1],
        'timestamp': range(5)
    })


@pytest.fixture
def small_dollar_bar_data():
    """Small dataset for quick tests."""
    n = 100
    base_price = 45000
    start_times = np.arange(n) * 1000

    df = pd.DataFrame({
        'start_time': start_times,
        'end_time': np.arange(n) * 1000 + 500,
        'open': base_price + np.random.randn(n) * 10,
        'high': base_price + 50 + np.abs(np.random.randn(n) * 20),
        'high_time': start_times + np.random.randint(0, 500, n),
        'low': base_price - 50 - np.abs(np.random.randn(n) * 20),
        'low_time': start_times + np.random.randint(0, 500, n),
        'close': base_price + np.random.randn(n) * 10,
        'volume': np.random.uniform(1, 5, n),
        'dollar_value': np.random.uniform(40000, 200000, n),
        'tick_count': np.random.randint(10, 100, n),
        'threshold_used': np.full(n, 20_000_000.0),
        'duration': np.random.uniform(60, 600, n),
        'buy_dollar_vol': np.random.uniform(20000, 120000, n),
        'sell_dollar_vol': np.random.uniform(20000, 80000, n),
        'net_imbalance': np.random.uniform(-20000, 40000, n),
    })

    return df
