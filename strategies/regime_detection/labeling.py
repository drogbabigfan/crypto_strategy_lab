"""
Posterior Regime Labeling Methods.

Provides ground truth labels for training/validating regime detection models:
- ZigZag: Geometric approach using pivot points
- Trend Scanning: Statistical approach using t-statistics
- Drawdown-based: Risk-focused regime identification

These labels use future information and are only for training/backtesting.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
from scipy import stats


class TrendLabel(Enum):
    """Trend direction labels."""
    UP = 1
    DOWN = -1
    SIDEWAYS = 0


@dataclass
class ZigZagConfig:
    """Configuration for ZigZag labeling."""
    threshold: float = 0.05         # Min % move to confirm reversal
    use_atr: bool = True            # Use ATR for dynamic threshold
    atr_multiplier: float = 2.0     # ATR multiplier for threshold
    atr_period: int = 14            # ATR calculation period
    min_bars: int = 5               # Min bars between pivots


@dataclass
class TrendScanConfig:
    """Configuration for Trend Scanning labeling."""
    min_window: int = 20            # Minimum look-forward window
    max_window: int = 100           # Maximum look-forward window
    t_threshold: float = 2.0        # t-statistic threshold for significance


@dataclass
class PivotPoint:
    """Represents a price pivot (peak or trough)."""
    index: int
    price: float
    is_peak: bool
    timestamp: Optional[pd.Timestamp] = None


class ZigZagLabeler:
    """
    ZigZag-based regime labeling.

    Identifies significant price swings by filtering out noise.
    A reversal is confirmed when price moves threshold% from the last pivot.
    """

    def __init__(self, config: Optional[ZigZagConfig] = None):
        self.config = config or ZigZagConfig()
        self.pivots: List[PivotPoint] = []

    def label(
        self,
        data: pd.DataFrame,
        price_col: str = 'close'
    ) -> pd.DataFrame:
        """
        Generate regime labels using ZigZag algorithm.

        Args:
            data: DataFrame with price data
            price_col: Column to use for price

        Returns:
            DataFrame with columns: label, pivot_price, pct_to_pivot
        """
        prices = data[price_col].values
        n = len(prices)

        # Calculate dynamic threshold if using ATR
        if self.config.use_atr and 'high' in data.columns and 'low' in data.columns:
            thresholds = self._calculate_atr_threshold(data)
        else:
            thresholds = np.full(n, self.config.threshold)

        # Find pivots
        self.pivots = self._find_pivots(prices, thresholds)

        # Generate labels based on pivots
        labels = self._pivots_to_labels(prices)

        # Build result DataFrame
        result = pd.DataFrame(index=data.index)
        result['label'] = labels
        result['label_name'] = result['label'].map({
            1: 'up', -1: 'down', 0: 'sideways'
        })

        # Add pivot information
        result['is_pivot'] = False
        result['pivot_type'] = None

        for pivot in self.pivots:
            if pivot.index < n:
                result.iloc[pivot.index, result.columns.get_loc('is_pivot')] = True
                result.iloc[pivot.index, result.columns.get_loc('pivot_type')] = (
                    'peak' if pivot.is_peak else 'trough'
                )

        return result

    def get_pivots(self) -> pd.DataFrame:
        """Get detected pivot points as DataFrame."""
        return pd.DataFrame([
            {
                'index': p.index,
                'price': p.price,
                'type': 'peak' if p.is_peak else 'trough',
                'timestamp': p.timestamp
            }
            for p in self.pivots
        ])

    def _calculate_atr_threshold(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate ATR-based dynamic threshold."""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # True Range
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]

        # ATR
        atr = pd.Series(tr).rolling(self.config.atr_period).mean().values

        # Threshold as percentage of price
        thresholds = (atr * self.config.atr_multiplier) / close
        thresholds = np.nan_to_num(thresholds, nan=self.config.threshold)

        return thresholds

    def _find_pivots(
        self,
        prices: np.ndarray,
        thresholds: np.ndarray
    ) -> List[PivotPoint]:
        """
        Find significant pivot points using ZigZag logic.

        A pivot is confirmed when:
        1. Price moves threshold% in opposite direction
        2. Minimum number of bars have passed
        """
        n = len(prices)
        if n < 2:
            return []

        pivots = []

        # Initialize with first point
        current_pivot = PivotPoint(index=0, price=prices[0], is_peak=True)
        searching_for_peak = True

        high_since_pivot = prices[0]
        low_since_pivot = prices[0]
        high_idx = 0
        low_idx = 0

        for i in range(1, n):
            price = prices[i]
            threshold = thresholds[i]

            # Track extremes
            if price > high_since_pivot:
                high_since_pivot = price
                high_idx = i
            if price < low_since_pivot:
                low_since_pivot = price
                low_idx = i

            if searching_for_peak:
                # Looking for peak after trough
                if price < high_since_pivot * (1 - threshold):
                    # Confirm peak
                    if high_idx - current_pivot.index >= self.config.min_bars:
                        pivots.append(PivotPoint(
                            index=high_idx,
                            price=high_since_pivot,
                            is_peak=True
                        ))
                        current_pivot = pivots[-1]

                        # Reset for trough search
                        low_since_pivot = price
                        low_idx = i
                        searching_for_peak = False
            else:
                # Looking for trough after peak
                if price > low_since_pivot * (1 + threshold):
                    # Confirm trough
                    if low_idx - current_pivot.index >= self.config.min_bars:
                        pivots.append(PivotPoint(
                            index=low_idx,
                            price=low_since_pivot,
                            is_peak=False
                        ))
                        current_pivot = pivots[-1]

                        # Reset for peak search
                        high_since_pivot = price
                        high_idx = i
                        searching_for_peak = True

        return pivots

    def _pivots_to_labels(self, prices: np.ndarray) -> np.ndarray:
        """Convert pivot points to per-bar labels."""
        n = len(prices)
        labels = np.zeros(n, dtype=int)

        if len(self.pivots) < 2:
            return labels

        for i in range(len(self.pivots) - 1):
            start = self.pivots[i].index
            end = self.pivots[i + 1].index

            if self.pivots[i].is_peak:
                # Peak to trough = downtrend
                labels[start:end] = TrendLabel.DOWN.value
            else:
                # Trough to peak = uptrend
                labels[start:end] = TrendLabel.UP.value

        # Handle last segment
        if self.pivots:
            last = self.pivots[-1]
            if last.is_peak:
                labels[last.index:] = TrendLabel.DOWN.value
            else:
                labels[last.index:] = TrendLabel.UP.value

        return labels


class TrendScanningLabeler:
    """
    Trend Scanning labeling using statistical significance.

    For each point, fits linear regression over multiple forward windows
    and selects the window with the most significant t-statistic.

    Based on: Lopez de Prado, "Advances in Financial Machine Learning"
    """

    def __init__(self, config: Optional[TrendScanConfig] = None):
        self.config = config or TrendScanConfig()

    def label(
        self,
        data: pd.DataFrame,
        price_col: str = 'close'
    ) -> pd.DataFrame:
        """
        Generate trend labels using statistical regression.

        Args:
            data: DataFrame with price data
            price_col: Column to use for price

        Returns:
            DataFrame with trend labels and statistics
        """
        prices = data[price_col].values
        n = len(prices)

        labels = np.zeros(n, dtype=int)
        t_values = np.zeros(n)
        best_windows = np.zeros(n, dtype=int)
        slopes = np.zeros(n)

        windows = range(self.config.min_window, self.config.max_window + 1)

        for i in range(n):
            best_t = 0
            best_window = 0
            best_slope = 0

            for window in windows:
                if i + window > n:
                    break

                # Get forward price segment
                y = prices[i:i + window]
                x = np.arange(window)

                # Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                if std_err > 0:
                    t_stat = slope / std_err
                else:
                    t_stat = 0

                # Track best (highest absolute t-stat)
                if abs(t_stat) > abs(best_t):
                    best_t = t_stat
                    best_window = window
                    best_slope = slope

            t_values[i] = best_t
            best_windows[i] = best_window
            slopes[i] = best_slope

            # Assign label based on t-statistic
            if best_t > self.config.t_threshold:
                labels[i] = TrendLabel.UP.value
            elif best_t < -self.config.t_threshold:
                labels[i] = TrendLabel.DOWN.value
            else:
                labels[i] = TrendLabel.SIDEWAYS.value

        result = pd.DataFrame(index=data.index)
        result['label'] = labels
        result['label_name'] = result['label'].map({
            1: 'up', -1: 'down', 0: 'sideways'
        })
        result['t_value'] = t_values
        result['best_window'] = best_windows
        result['slope'] = slopes
        result['abs_t'] = np.abs(t_values)

        return result


class DrawdownLabeler:
    """
    Drawdown-based regime labeling.

    Identifies crisis periods based on peak-to-trough drawdowns.
    Useful for risk management and tail-risk strategies.
    """

    def __init__(
        self,
        crisis_threshold: float = -0.15,     # -15% drawdown = crisis
        recovery_threshold: float = -0.05,   # Within -5% of peak = recovered
        vol_percentile: float = 80           # Top 20% volatility = high vol
    ):
        self.crisis_threshold = crisis_threshold
        self.recovery_threshold = recovery_threshold
        self.vol_percentile = vol_percentile

    def label(
        self,
        data: pd.DataFrame,
        price_col: str = 'close',
        vol_window: int = 20
    ) -> pd.DataFrame:
        """
        Generate regime labels based on drawdown and volatility.

        Regimes:
        - crisis: Drawdown exceeds crisis_threshold
        - recovery: Coming back from crisis
        - normal_low_vol: Normal market, low volatility
        - normal_high_vol: Normal market, high volatility

        Args:
            data: DataFrame with price data
            price_col: Column to use for price
            vol_window: Window for volatility calculation

        Returns:
            DataFrame with regime labels
        """
        prices = data[price_col].values
        n = len(prices)

        # Calculate drawdown
        running_max = np.maximum.accumulate(prices)
        drawdown = (prices - running_max) / running_max

        # Calculate volatility
        returns = np.diff(prices) / prices[:-1]
        returns = np.insert(returns, 0, 0)
        volatility = pd.Series(returns).rolling(vol_window).std().values

        vol_threshold = np.nanpercentile(volatility, self.vol_percentile)

        # Label regimes
        labels = []
        in_crisis = False

        for i in range(n):
            dd = drawdown[i]
            vol = volatility[i] if not np.isnan(volatility[i]) else 0

            if dd < self.crisis_threshold:
                labels.append('crisis')
                in_crisis = True
            elif in_crisis and dd < self.recovery_threshold:
                labels.append('recovery')
            elif in_crisis and dd >= self.recovery_threshold:
                labels.append('normal_low_vol' if vol < vol_threshold else 'normal_high_vol')
                in_crisis = False
            elif vol >= vol_threshold:
                labels.append('normal_high_vol')
            else:
                labels.append('normal_low_vol')

        result = pd.DataFrame(index=data.index)
        result['label'] = labels
        result['drawdown'] = drawdown
        result['volatility'] = volatility
        result['peak'] = running_max
        result['in_crisis'] = [l in ['crisis', 'recovery'] for l in labels]

        # Numeric encoding
        label_map = {
            'crisis': -2,
            'recovery': -1,
            'normal_low_vol': 0,
            'normal_high_vol': 1
        }
        result['label_id'] = [label_map[l] for l in labels]

        return result


class VolatilityRegimeLabeler:
    """
    Volatility-based regime labeling.

    Divides market into regimes based on volatility percentiles:
    - Low volatility: < 33rd percentile
    - Medium volatility: 33rd - 66th percentile
    - High volatility: > 66th percentile
    """

    def __init__(
        self,
        low_percentile: float = 33,
        high_percentile: float = 66,
        vol_window: int = 20,
        vol_type: str = 'parkinson'  # 'realized' or 'parkinson'
    ):
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.vol_window = vol_window
        self.vol_type = vol_type

    def label(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate volatility regime labels.

        Args:
            data: DataFrame with OHLC data

        Returns:
            DataFrame with volatility regime labels
        """
        # Calculate volatility
        if self.vol_type == 'parkinson' and 'high' in data.columns and 'low' in data.columns:
            vol = self._parkinson_volatility(data['high'], data['low'])
        else:
            returns = data['close'].pct_change()
            vol = returns.rolling(self.vol_window).std()

        # Compute thresholds
        low_thresh = np.nanpercentile(vol, self.low_percentile)
        high_thresh = np.nanpercentile(vol, self.high_percentile)

        # Label
        labels = []
        for v in vol:
            if np.isnan(v):
                labels.append('unknown')
            elif v < low_thresh:
                labels.append('low_vol')
            elif v > high_thresh:
                labels.append('high_vol')
            else:
                labels.append('medium_vol')

        result = pd.DataFrame(index=data.index)
        result['label'] = labels
        result['volatility'] = vol
        result['low_threshold'] = low_thresh
        result['high_threshold'] = high_thresh

        # Numeric encoding
        label_map = {'low_vol': 0, 'medium_vol': 1, 'high_vol': 2, 'unknown': -1}
        result['label_id'] = [label_map[l] for l in labels]

        return result

    def _parkinson_volatility(
        self,
        high: pd.Series,
        low: pd.Series
    ) -> pd.Series:
        """Calculate Parkinson volatility estimator."""
        log_hl = np.log(high / low)
        log_hl_sq = log_hl ** 2
        rolling_sum = log_hl_sq.rolling(self.vol_window).sum()
        parkinson_const = 1 / (4 * np.log(2))
        return np.sqrt(parkinson_const * rolling_sum / self.vol_window)


def combine_labels(
    price_labels: pd.Series,
    vol_labels: pd.Series
) -> pd.Series:
    """
    Combine price direction and volatility regime labels.

    Creates composite labels like:
    - 'bull_low_vol': Uptrend with low volatility (ideal for momentum)
    - 'bear_high_vol': Downtrend with high volatility (crisis)

    Args:
        price_labels: Series with 'up', 'down', 'sideways'
        vol_labels: Series with 'low_vol', 'medium_vol', 'high_vol'

    Returns:
        Combined label Series
    """
    return price_labels.astype(str) + '_' + vol_labels.astype(str)
