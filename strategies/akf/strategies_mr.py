"""
AKF Mean Reversion Strategies.

Opposite of breakout - enter against the move, expect reversion to trend.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class AKFMeanReversionConfig:
    """Configuration for Mean Reversion strategies."""
    k: float = 2.0  # Band multiplier


class AKFMeanReversion:
    """
    Mean Reversion strategies using Kalman Filter bands.

    Logic (opposite of breakout):
    - Long: close < lower_band (oversold → expect bounce)
    - Short: close > upper_band (overbought → expect drop)
    - Exit: price reverts to trend
    """

    REQUIRED_COLUMNS = [
        'close',
        'kf_trend',
        'kf_uncertainty',
    ]

    def __init__(self, config: Optional[AKFMeanReversionConfig] = None):
        self.config = config or AKFMeanReversionConfig()

    def _validate_input(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _calculate_bands(self, df: pd.DataFrame, k: float) -> tuple:
        """Calculate uncertainty-based bands around trend."""
        uncertainty_sqrt = np.sqrt(df['kf_uncertainty'].values)
        k_band = k * uncertainty_sqrt
        center = df['kf_trend'].values
        upper_band = center + k_band
        lower_band = center - k_band
        return upper_band, lower_band

    def strategy_mr_basic(
        self,
        df: pd.DataFrame,
        k: Optional[float] = None,
        include_bands: bool = False
    ) -> pd.DataFrame:
        """
        Basic Mean Reversion Strategy.

        Entry (fade the move):
            - Long: close < lower_band (oversold)
            - Short: close > upper_band (overbought)

        Exit (reversion to mean):
            - Long exit: close > kf_trend
            - Short exit: close < kf_trend
        """
        self._validate_input(df)
        k = k if k is not None else self.config.k

        result = df.copy()
        close = df['close'].values
        trend = df['kf_trend'].values
        upper_band, lower_band = self._calculate_bands(df, k)

        # Entry signals (OPPOSITE of breakout)
        entry_long = close < lower_band   # Oversold → buy
        entry_short = close > upper_band  # Overbought → sell

        n = len(df)
        signal = np.zeros(n, dtype=np.int32)
        position = 0

        for i in range(n):
            if position == 0:
                if entry_long[i]:
                    position = 1
                elif entry_short[i]:
                    position = -1
            elif position == 1:
                # Long - exit when price reverts above trend
                if close[i] > trend[i]:
                    position = 0
                elif entry_short[i]:  # Reversal
                    position = -1
            elif position == -1:
                # Short - exit when price reverts below trend
                if close[i] < trend[i]:
                    position = 0
                elif entry_long[i]:  # Reversal
                    position = 1

            signal[i] = position

        result['signal'] = signal

        if include_bands:
            result['upper_band'] = upper_band
            result['lower_band'] = lower_band

        return result

    def strategy_mr_tight(
        self,
        df: pd.DataFrame,
        k: Optional[float] = None,
        exit_k: float = 0.5,
        include_bands: bool = False
    ) -> pd.DataFrame:
        """
        Tight Mean Reversion - smaller exit target.

        Entry: same as basic (k bands)
        Exit: when price crosses k*0.5 band (tighter target)
        """
        self._validate_input(df)
        k = k if k is not None else self.config.k

        result = df.copy()
        close = df['close'].values
        trend = df['kf_trend'].values

        # Entry bands (wider)
        upper_band, lower_band = self._calculate_bands(df, k)

        # Exit bands (tighter)
        exit_upper, exit_lower = self._calculate_bands(df, exit_k)

        entry_long = close < lower_band
        entry_short = close > upper_band

        n = len(df)
        signal = np.zeros(n, dtype=np.int32)
        position = 0

        for i in range(n):
            if position == 0:
                if entry_long[i]:
                    position = 1
                elif entry_short[i]:
                    position = -1
            elif position == 1:
                # Exit when price reaches exit band (tighter)
                if close[i] > exit_lower[i]:  # Above lower exit band
                    position = 0
                elif entry_short[i]:
                    position = -1
            elif position == -1:
                if close[i] < exit_upper[i]:  # Below upper exit band
                    position = 0
                elif entry_long[i]:
                    position = 1

            signal[i] = position

        result['signal'] = signal

        if include_bands:
            result['upper_band'] = upper_band
            result['lower_band'] = lower_band

        return result

    def strategy_mr_reversal(
        self,
        df: pd.DataFrame,
        k: Optional[float] = None,
        include_bands: bool = False
    ) -> pd.DataFrame:
        """
        Always-in-market Mean Reversion.

        Entry: fade the extreme
        Exit: reverse position (no flat periods)
        """
        self._validate_input(df)
        k = k if k is not None else self.config.k

        result = df.copy()
        close = df['close'].values
        trend = df['kf_trend'].values
        upper_band, lower_band = self._calculate_bands(df, k)

        entry_long = close < lower_band
        entry_short = close > upper_band

        n = len(df)
        signal = np.zeros(n, dtype=np.int32)
        position = 0

        for i in range(n):
            if position == 0:
                if entry_long[i]:
                    position = 1
                elif entry_short[i]:
                    position = -1
            elif position == 1:
                # Reversal: price goes overbought → switch to short
                if close[i] > trend[i]:
                    position = -1  # Reverse instead of flat
                elif entry_short[i]:
                    position = -1
            elif position == -1:
                if close[i] < trend[i]:
                    position = 1  # Reverse instead of flat
                elif entry_long[i]:
                    position = 1

            signal[i] = position

        result['signal'] = signal

        if include_bands:
            result['upper_band'] = upper_band
            result['lower_band'] = lower_band

        return result

    def get_strategy_names(self) -> list:
        return ['strategy_mr_basic', 'strategy_mr_tight', 'strategy_mr_reversal']

    def run_strategy(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        k: Optional[float] = None,
        include_bands: bool = False
    ) -> pd.DataFrame:
        strategies = {
            'strategy_mr_basic': self.strategy_mr_basic,
            'strategy_mr_tight': self.strategy_mr_tight,
            'strategy_mr_reversal': self.strategy_mr_reversal,
        }

        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        return strategies[strategy_name](df, k=k, include_bands=include_bands)
