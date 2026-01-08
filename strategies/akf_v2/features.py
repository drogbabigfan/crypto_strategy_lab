"""
Log-Price Kalman Filter Feature Engineering.

Key insight: With log-price transformation,
- velocity = expected log-return per bar
- uncertainty = confidence in the estimate

This module creates trading-specific features from Kalman outputs.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from research.features.kalman import (
    AdaptiveKalmanFilter,
    KalmanConfig,
    calculate_adaptive_kalman,
)


@dataclass
class FeatureConfig:
    """Configuration for feature generation."""

    # Kalman parameters
    r_window: int = 20
    q_window: int = 20
    use_log_price: bool = True  # Must be True for this module

    # Velocity features
    velocity_ma_fast: int = 5
    velocity_ma_slow: int = 20

    # Regime detection
    efficiency_window: int = 20
    trend_threshold: float = 0.0001  # Min velocity for trend

    # Z-score parameters
    zscore_window: int = 20

    # Volatility scaling
    vol_window: int = 20


class LogKalmanFeatures:
    """
    Generate trading features from Log-Price Kalman Filter.

    Features:
    - velocity_signal: Normalized velocity direction and magnitude
    - velocity_acceleration: Rate of change of velocity
    - trend_strength: How strong is the current trend
    - mean_reversion_signal: Deviation from trend, uncertainty-adjusted
    - regime: Trending (1) / Ranging (0) / Volatile (-1)
    - position_confidence: Inverse of uncertainty, normalized
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from OHLC data.

        Args:
            df: DataFrame with OHLC columns

        Returns:
            DataFrame with all features added
        """
        # Step 1: Apply Kalman filter
        result = calculate_adaptive_kalman(
            df,
            r_window=self.config.r_window,
            q_window=self.config.q_window,
            use_log_price=self.config.use_log_price,
        )

        # Step 2: Rename columns for clarity
        result = self._rename_columns(result)

        # Step 3: Generate derived features
        result = self._velocity_features(result)
        result = self._trend_features(result)
        result = self._mean_reversion_features(result)
        result = self._regime_features(result)
        result = self._confidence_features(result)

        return result

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename Kalman columns for clarity."""
        result = df.copy()

        rename_map = {
            'kf_trend': 'trend',
            'kf_trend_pred': 'trend_pred',
            'kf_velocity': 'velocity',  # This is log-return per bar
            'kf_deviation': 'deviation',
            'kf_deviation_pct': 'deviation_pct',
            'kf_gain': 'kalman_gain',
            'kf_uncertainty': 'uncertainty',
            'kf_signal': 'kf_zscore',
        }

        for old, new in rename_map.items():
            if old in result.columns:
                result[new] = result[old]

        return result

    def _velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Velocity-based features.

        velocity is log-return per bar, so:
        - velocity > 0: uptrend
        - velocity < 0: downtrend
        - |velocity| = trend strength (in % per bar)
        """
        result = df.copy()

        # Moving averages for smoothing
        result['velocity_ma_fast'] = result['velocity'].rolling(
            window=self.config.velocity_ma_fast, min_periods=1
        ).mean()

        result['velocity_ma_slow'] = result['velocity'].rolling(
            window=self.config.velocity_ma_slow, min_periods=1
        ).mean()

        # Velocity acceleration (second derivative)
        result['velocity_diff'] = result['velocity'].diff()
        result['velocity_accel'] = result['velocity_diff'].rolling(
            window=5, min_periods=1
        ).mean()

        # Normalized velocity signal (-1 to 1 scale)
        vel_std = result['velocity'].rolling(
            window=self.config.zscore_window, min_periods=5
        ).std()
        result['velocity_zscore'] = result['velocity'] / (vel_std + 1e-10)
        result['velocity_signal'] = np.tanh(result['velocity_zscore'])

        # MA crossover signal
        result['velocity_cross'] = (
            result['velocity_ma_fast'] - result['velocity_ma_slow']
        )

        return result

    def _trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trend strength and direction features.
        """
        result = df.copy()

        # Efficiency ratio: directional move / total move
        price_change = result['close'].diff(self.config.efficiency_window)
        abs_changes = result['close'].diff().abs().rolling(
            window=self.config.efficiency_window, min_periods=1
        ).sum()

        result['efficiency_ratio'] = price_change.abs() / (abs_changes + 1e-10)
        result['efficiency_ratio'] = result['efficiency_ratio'].clip(0, 1)

        # Trend strength: velocity magnitude * efficiency
        result['trend_strength'] = (
            result['velocity'].abs() * result['efficiency_ratio']
        )

        # Trend direction with confidence
        result['trend_direction'] = np.sign(result['velocity_ma_slow'])

        return result

    def _mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mean reversion features: price deviation from trend.
        """
        result = df.copy()

        # Deviation z-score (already computed as kf_zscore)
        # Re-compute with uncertainty scaling

        # Uncertainty-scaled deviation
        # When uncertainty is low, deviation is more meaningful
        uncertainty_inv = 1.0 / (np.sqrt(result['uncertainty']) + 1e-10)
        uncertainty_norm = uncertainty_inv / uncertainty_inv.rolling(
            window=self.config.zscore_window, min_periods=5
        ).mean()

        result['deviation_adjusted'] = result['deviation'] * uncertainty_norm

        # Z-score of adjusted deviation
        dev_std = result['deviation_adjusted'].rolling(
            window=self.config.zscore_window, min_periods=5
        ).std()
        result['mr_signal'] = result['deviation_adjusted'] / (dev_std + 1e-10)

        # Band-based signal (deviation as % of trend)
        result['band_position'] = result['deviation_pct']

        return result

    def _regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Market regime classification.

        Regimes:
        - TRENDING (1): Strong directional move, high efficiency
        - RANGING (0): Low velocity, oscillating
        - VOLATILE (-1): High uncertainty, erratic moves
        """
        result = df.copy()

        # Conditions
        velocity_strong = result['velocity'].abs() > self.config.trend_threshold
        efficiency_high = result['efficiency_ratio'] > 0.4

        # Uncertainty percentile
        uncertainty_pct = result['uncertainty'].rolling(
            window=100, min_periods=20
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        uncertainty_pct = uncertainty_pct.fillna(0.5)

        uncertainty_high = uncertainty_pct > 0.8

        # Regime classification
        result['regime'] = 0  # Default: ranging
        result.loc[velocity_strong & efficiency_high, 'regime'] = 1  # Trending
        result.loc[uncertainty_high, 'regime'] = -1  # Volatile

        # Regime duration
        result['regime_change'] = result['regime'].diff().abs() > 0
        result['regime_bars'] = result.groupby(
            result['regime_change'].cumsum()
        ).cumcount() + 1

        return result

    def _confidence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Position sizing confidence based on uncertainty.
        """
        result = df.copy()

        # Inverse uncertainty, normalized
        uncertainty_inv = 1.0 / (np.sqrt(result['uncertainty']) + 1e-10)

        # Rolling percentile for normalization
        result['confidence'] = uncertainty_inv.rolling(
            window=100, min_periods=20
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        result['confidence'] = result['confidence'].fillna(0.5)

        # Kalman gain as confidence (high gain = filter trusts observation)
        result['observation_trust'] = result['kalman_gain'].clip(0, 1)

        return result

    def get_signal_features(self) -> list:
        """Return list of features useful for signal generation."""
        return [
            'velocity_signal',      # Trend direction/strength
            'velocity_accel',       # Momentum acceleration
            'velocity_cross',       # MA crossover
            'trend_strength',       # Trend magnitude
            'mr_signal',            # Mean reversion signal
            'regime',               # Market regime
            'confidence',           # Position sizing factor
        ]

    def get_all_features(self) -> list:
        """Return all generated feature names."""
        return [
            # Velocity
            'velocity', 'velocity_ma_fast', 'velocity_ma_slow',
            'velocity_diff', 'velocity_accel', 'velocity_zscore',
            'velocity_signal', 'velocity_cross',
            # Trend
            'efficiency_ratio', 'trend_strength', 'trend_direction',
            # Mean reversion
            'deviation', 'deviation_pct', 'deviation_adjusted',
            'mr_signal', 'band_position',
            # Regime
            'regime', 'regime_bars',
            # Confidence
            'uncertainty', 'confidence', 'observation_trust',
            # Original
            'trend', 'trend_pred', 'kalman_gain',
        ]
