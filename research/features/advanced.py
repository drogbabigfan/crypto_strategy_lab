"""
Advanced Feature Generator.

Includes:
- Autocorrelation: Return serial correlation
- Hurst Exponent: Trend/mean-reversion detection
- Order Flow Toxicity: CDF-based informed trading indicator
- RSI: Relative Strength Index
- Efficiency Ratio: Kaufman's trend efficiency
- Momentum Z-score: Risk-adjusted momentum at multiple scales
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from scipy import stats


class AdvancedFeatureGenerator:
    """
    Generate advanced technical and microstructure features.
    """

    def __init__(
        self,
        epsilon: float = 1e-10,
        autocorr_window: int = 20,
        autocorr_lag: int = 1,
        hurst_window: int = 100,
        toxicity_window: int = 50,
        rsi_window: int = 14,
        efficiency_window: int = 10,
        momentum_windows: List[int] = None,
    ):
        """
        Args:
            epsilon: Small value to prevent division by zero
            autocorr_window: Rolling window for autocorrelation
            autocorr_lag: Lag for autocorrelation calculation
            hurst_window: Window for Hurst exponent
            toxicity_window: Window for order flow toxicity CDF
            rsi_window: Window for RSI calculation
            efficiency_window: Window for Kaufman Efficiency Ratio
            momentum_windows: Windows for momentum z-score [10, 50, 250, 1000]
        """
        self.epsilon = epsilon
        self.autocorr_window = autocorr_window
        self.autocorr_lag = autocorr_lag
        self.hurst_window = hurst_window
        self.toxicity_window = toxicity_window
        self.rsi_window = rsi_window
        self.efficiency_window = efficiency_window
        self.momentum_windows = momentum_windows or [10, 50, 250, 1000]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all advanced features.

        Args:
            df: DataFrame with 'close', 'volume_imbalance' (or will compute from net_imbalance/dollar_value)

        Returns:
            DataFrame with original columns + advanced features
        """
        features = df.copy()

        # Ensure we have returns
        if 'returns' not in features.columns:
            features['returns'] = features['close'].pct_change()

        # Ensure we have volume_imbalance
        if 'volume_imbalance' not in features.columns and 'net_imbalance' in features.columns:
            features['volume_imbalance'] = (
                features['net_imbalance'] / (features['dollar_value'] + self.epsilon)
            )

        # 1. Autocorrelation
        features['autocorr'] = self._rolling_autocorrelation(
            features['returns'],
            window=self.autocorr_window,
            lag=self.autocorr_lag
        )

        # 2. Hurst Exponent
        features['hurst_exp'] = self._rolling_hurst(
            features['close'],
            window=self.hurst_window
        )

        # 3. Order Flow Toxicity (CDF of volume_imbalance)
        if 'volume_imbalance' in features.columns:
            features['order_flow_toxicity'] = self._order_flow_toxicity(
                features['volume_imbalance'],
                window=self.toxicity_window
            )

        # 4. RSI
        features['rsi'] = self._rsi(
            features['close'],
            window=self.rsi_window
        )

        # 5. Efficiency Ratio (Kaufman)
        features['efficiency_ratio'] = self._efficiency_ratio(
            features['close'],
            window=self.efficiency_window
        )

        # 6. Momentum Z-scores at multiple scales
        for window in self.momentum_windows:
            features[f'momentum_zscore_{window}'] = self._momentum_zscore(
                features['close'],
                window=window
            )

        return features

    def _rolling_autocorrelation(
        self,
        series: pd.Series,
        window: int = 20,
        lag: int = 1
    ) -> pd.Series:
        """
        Calculate rolling autocorrelation.

        High positive: trending/momentum
        Negative: mean-reverting
        Near zero: random walk

        Args:
            series: Price returns
            window: Rolling window size
            lag: Autocorrelation lag

        Returns:
            Rolling autocorrelation series
        """
        def autocorr_func(x):
            if len(x) < lag + 2:
                return np.nan
            return pd.Series(x).autocorr(lag=lag)

        return series.rolling(window=window, min_periods=lag + 2).apply(
            autocorr_func, raw=True
        )

    def _rolling_hurst(
        self,
        series: pd.Series,
        window: int = 100
    ) -> pd.Series:
        """
        Calculate rolling Hurst exponent using R/S analysis.

        H < 0.5: Mean-reverting
        H = 0.5: Random walk
        H > 0.5: Trending

        Args:
            series: Price series
            window: Rolling window size

        Returns:
            Rolling Hurst exponent
        """
        def hurst_rs(prices):
            """Simplified R/S Hurst estimation."""
            if len(prices) < 20:
                return np.nan

            prices = np.array(prices)
            n = len(prices)

            # Log returns
            returns = np.diff(np.log(prices + self.epsilon))
            if len(returns) < 10:
                return np.nan

            # Mean-adjusted cumulative sum
            mean_ret = np.mean(returns)
            adjusted = returns - mean_ret
            cumsum = np.cumsum(adjusted)

            # Range
            R = np.max(cumsum) - np.min(cumsum)

            # Standard deviation
            S = np.std(returns, ddof=1)
            if S < self.epsilon:
                return np.nan

            # R/S ratio
            RS = R / S

            # Hurst = log(R/S) / log(n)
            if RS <= 0:
                return np.nan

            H = np.log(RS) / np.log(n)

            # Clamp to reasonable range
            return np.clip(H, 0, 1)

        min_periods = min(20, window)
        return series.rolling(window=window, min_periods=min_periods).apply(
            hurst_rs, raw=True
        )

    def _order_flow_toxicity(
        self,
        volume_imbalance: pd.Series,
        window: int = 50
    ) -> pd.Series:
        """
        Calculate Order Flow Toxicity using CDF of volume imbalance.

        Based on VPIN (Volume-Synchronized Probability of Informed Trading).
        Uses rolling percentile rank as CDF approximation.

        Values close to 0 or 1: Extreme imbalance (informed trading)
        Values close to 0.5: Normal trading

        Args:
            volume_imbalance: Normalized buy/sell imbalance
            window: Rolling window for CDF calculation

        Returns:
            Order flow toxicity (0-1 scale)
        """
        def percentile_rank(x):
            """Calculate percentile rank of last value in window."""
            if len(x) < 2:
                return np.nan
            current = x[-1]
            # Percentile of current value within the window
            rank = stats.percentileofscore(x, current, kind='mean') / 100.0
            return rank

        toxicity = volume_imbalance.rolling(window=window, min_periods=10).apply(
            percentile_rank, raw=True
        )

        # Convert to toxicity: distance from 0.5 (normal)
        # |rank - 0.5| * 2 gives 0 for normal, 1 for extreme
        return (toxicity - 0.5).abs() * 2

    def _rsi(
        self,
        close: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """
        Calculate Relative Strength Index.

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss

        > 70: Overbought
        < 30: Oversold

        Args:
            close: Close prices
            window: RSI period

        Returns:
            RSI values (0-100)
        """
        delta = close.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))

        # Wilder's smoothing (EMA with alpha = 1/window)
        avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

        rs = avg_gain / (avg_loss + self.epsilon)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _efficiency_ratio(
        self,
        close: pd.Series,
        window: int = 10
    ) -> pd.Series:
        """
        Calculate Kaufman's Efficiency Ratio.

        ER = Direction / Volatility
        Direction = |Close - Close_n|
        Volatility = Sum of |Close_i - Close_{i-1}| over n periods

        ER close to 1: Strong trend (efficient movement)
        ER close to 0: Choppy/noisy (inefficient movement)

        Args:
            close: Close prices
            window: Lookback period

        Returns:
            Efficiency ratio (0-1)
        """
        # Direction: net price change
        direction = (close - close.shift(window)).abs()

        # Volatility: sum of absolute single-bar changes
        volatility = close.diff().abs().rolling(window=window).sum()

        er = direction / (volatility + self.epsilon)

        # Clamp to [0, 1]
        return er.clip(0, 1)

    def _momentum_zscore(
        self,
        close: pd.Series,
        window: int
    ) -> pd.Series:
        """
        Calculate momentum z-score: return_n / volatility_n.

        Risk-adjusted momentum - how many standard deviations
        the return is from zero.

        Args:
            close: Close prices
            window: Lookback period for both return and volatility

        Returns:
            Momentum z-score
        """
        # N-period return
        return_n = (close - close.shift(window)) / (close.shift(window) + self.epsilon)

        # N-period volatility (std of 1-bar returns)
        vol_n = close.pct_change().rolling(window=window).std()

        # Z-score: return / volatility
        momentum_z = return_n / (vol_n * np.sqrt(window) + self.epsilon)

        return momentum_z

    def get_feature_names(self) -> List[str]:
        """Return list of generated feature names."""
        names = [
            'autocorr',
            'hurst_exp',
            'order_flow_toxicity',
            'rsi',
            'efficiency_ratio',
        ]
        for window in self.momentum_windows:
            names.append(f'momentum_zscore_{window}')
        return names

    def get_core_features(self) -> List[str]:
        """Return core advanced features for model input."""
        return [
            'autocorr',
            'hurst_exp',
            'order_flow_toxicity',
            'efficiency_ratio',
            'momentum_zscore_10',
            'momentum_zscore_50',
        ]
