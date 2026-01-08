"""
L1 Feature Generator for Dollar Bar data.

Generates basic features from Adaptive Dynamic Dollar Bars:
- Log transforms for normalization
- Volume imbalance (buy/sell pressure)
- VWAP deviation
- Duration features (time-aware for irregular bars)
"""

import numpy as np
import pandas as pd
from typing import List, Optional


class L1FeatureGenerator:
    """
    Generate L1 (Level 1) features from Dollar Bar data.

    Expected input columns (from ETL Parquet):
        - start_time, end_time: int64 (ms)
        - open, high, low, close: float64
        - volume: float64 (BTC)
        - dollar_value: float64 (USD)
        - tick_count: int64
        - duration: float64 (seconds)
        - buy_dollar_vol, sell_dollar_vol: float64
        - net_imbalance: float64
    """

    def __init__(self, epsilon: float = 1e-10):
        """
        Args:
            epsilon: Small value to prevent division by zero
        """
        self.epsilon = epsilon

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate L1 features from Dollar Bar data.

        Args:
            df: DataFrame with Dollar Bar columns

        Returns:
            DataFrame with original columns + L1 features
        """
        features = df.copy()

        # 1. Log Transforms (normalize skewed distributions)
        # Note: log_dollar_value removed - constant in Dollar Bars (threshold-based)
        features['log_volume'] = np.log1p(features['volume'])
        features['log_tick_count'] = np.log1p(features['tick_count'])
        features['log_duration'] = np.log1p(features['duration'])

        # 2. VWAP Calculation and Deviation
        # VWAP = Total Dollar Value / Total Volume
        features['vwap'] = features['dollar_value'] / (features['volume'] + self.epsilon)
        features['vwap_deviation'] = (features['close'] - features['vwap']) / (features['close'] + self.epsilon)

        # 3. Volume Imbalance (normalized)
        # Already have net_imbalance = buy_dollar_vol - sell_dollar_vol
        # Normalize by total dollar value
        features['volume_imbalance'] = features['net_imbalance'] / (features['dollar_value'] + self.epsilon)

        # 4. Buy Ratio (proportion of buy volume)
        features['buy_ratio'] = features['buy_dollar_vol'] / (features['dollar_value'] + self.epsilon)

        # 5. Price Range Features
        features['bar_range'] = (features['high'] - features['low']) / (features['close'] + self.epsilon)
        features['bar_body'] = (features['close'] - features['open']) / (features['close'] + self.epsilon)

        # 6. Trade Intensity (dollars per second)
        features['trade_intensity'] = features['dollar_value'] / (features['duration'] + self.epsilon)
        features['log_trade_intensity'] = np.log1p(features['trade_intensity'])

        # 7. Tick Size (average trade size)
        features['avg_trade_size'] = features['dollar_value'] / (features['tick_count'] + self.epsilon)
        features['log_avg_trade_size'] = np.log1p(features['avg_trade_size'])

        return features

    def get_feature_names(self) -> List[str]:
        """Return list of generated feature names."""
        return [
            'log_volume',
            'log_tick_count',
            'log_duration',
            'vwap',
            'vwap_deviation',
            'volume_imbalance',
            'buy_ratio',
            'bar_range',
            'bar_body',
            'trade_intensity',
            'log_trade_intensity',
            'avg_trade_size',
            'log_avg_trade_size',
        ]

    def get_core_features(self) -> List[str]:
        """Return list of core features for model input."""
        return [
            'log_volume',
            'log_duration',
            'volume_imbalance',
            'vwap_deviation',
            'log_tick_count',
        ]
