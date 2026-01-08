"""
Regime Feature Generator.

Detects market regime through volatility and entropy measures:
- Parkinson Volatility: High-efficiency volatility estimator using High/Low
- Shannon Entropy: Measures randomness in price movements (noise vs trend)
"""

import numpy as np
import pandas as pd
from typing import Optional


class RegimeFeatureGenerator:
    """
    Generate regime detection features from Dollar Bar data.

    Features:
        - Parkinson Volatility: More efficient than close-to-close
        - Shannon Entropy: Low = trending, High = noisy/ranging
        - Realized Volatility: Standard close-to-close volatility
    """

    def __init__(
        self,
        vol_window: int = 24,
        entropy_window: int = 24,
        entropy_bins: int = 10,
        epsilon: float = 1e-10,
    ):
        """
        Args:
            vol_window: Rolling window for volatility calculation
            entropy_window: Rolling window for entropy calculation
            entropy_bins: Number of bins for entropy histogram
            epsilon: Small value to prevent log(0)
        """
        self.vol_window = vol_window
        self.entropy_window = entropy_window
        self.entropy_bins = entropy_bins
        self.epsilon = epsilon

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate regime features.

        Args:
            df: DataFrame with at least 'high', 'low', 'close' columns

        Returns:
            DataFrame with original columns + regime features
        """
        features = df.copy()

        # 1. Parkinson Volatility
        features['parkinson_vol'] = self._parkinson_volatility(
            features['high'], features['low']
        )

        # 2. Realized Volatility (close-to-close)
        features['returns'] = features['close'].pct_change()
        features['realized_vol'] = (
            features['returns']
            .rolling(window=self.vol_window)
            .std()
        )

        # 3. Shannon Entropy
        features['shannon_entropy'] = self._rolling_entropy(features['returns'])

        # 4. Volatility Ratio (Parkinson / Realized)
        # High ratio may indicate jumps or gaps
        features['vol_ratio'] = (
            features['parkinson_vol'] /
            (features['realized_vol'] + self.epsilon)
        )

        # 5. Normalized Volatility (z-score of volatility)
        vol_mean = features['parkinson_vol'].rolling(window=self.vol_window * 4).mean()
        vol_std = features['parkinson_vol'].rolling(window=self.vol_window * 4).std()
        features['vol_zscore'] = (
            (features['parkinson_vol'] - vol_mean) /
            (vol_std + self.epsilon)
        )

        # 6. Entropy Z-score
        ent_mean = features['shannon_entropy'].rolling(window=self.entropy_window * 4).mean()
        ent_std = features['shannon_entropy'].rolling(window=self.entropy_window * 4).std()
        features['entropy_zscore'] = (
            (features['shannon_entropy'] - ent_mean) /
            (ent_std + self.epsilon)
        )

        return features

    def _parkinson_volatility(
        self,
        high: pd.Series,
        low: pd.Series,
    ) -> pd.Series:
        """
        Calculate Parkinson volatility estimator.

        More efficient than close-to-close volatility because it uses
        intrabar price range information.

        Formula: sqrt(sum(log(H/L)^2) / (4 * ln(2) * N))

        Args:
            high: High prices
            low: Low prices

        Returns:
            Rolling Parkinson volatility
        """
        # Prevent log(0) or log(negative)
        log_hl = np.log((high + self.epsilon) / (low + self.epsilon))
        log_hl_sq = log_hl ** 2

        # Rolling sum
        rolling_sum = log_hl_sq.rolling(window=self.vol_window).sum()

        # Parkinson constant: 1 / (4 * ln(2))
        parkinson_const = 1 / (4 * np.log(2))

        return np.sqrt(parkinson_const * rolling_sum / self.vol_window)

    def _rolling_entropy(self, returns: pd.Series) -> pd.Series:
        """
        Calculate rolling Shannon entropy of returns.

        Low entropy = more predictable (trending)
        High entropy = more random (noisy/ranging)

        Args:
            returns: Price returns

        Returns:
            Rolling Shannon entropy
        """
        def calc_entropy(x):
            # Remove NaN values
            x = x.dropna()
            if len(x) < 2:
                return np.nan

            # Create histogram
            hist, _ = np.histogram(x, bins=self.entropy_bins, density=True)

            # Remove zero bins and normalize
            hist = hist[hist > 0]
            if len(hist) == 0:
                return 0.0

            # Calculate entropy: -sum(p * log(p))
            # Normalize by log(bins) to get value between 0 and 1
            entropy = -np.sum(hist * np.log(hist + self.epsilon))
            max_entropy = np.log(self.entropy_bins)

            return entropy / max_entropy

        return returns.rolling(window=self.entropy_window).apply(
            calc_entropy, raw=False
        )

    def get_feature_names(self) -> list:
        """Return list of generated feature names."""
        return [
            'parkinson_vol',
            'returns',
            'realized_vol',
            'shannon_entropy',
            'vol_ratio',
            'vol_zscore',
            'entropy_zscore',
        ]

    def get_core_features(self) -> list:
        """Return core regime features for model input."""
        return [
            'parkinson_vol',
            'shannon_entropy',
            'vol_zscore',
        ]
