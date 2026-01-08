"""
AKF V3 Strategies: Always-in-Market (Reversal) Version.

Exit = Reversal to opposite position (no flat periods).
This ensures Go backtester properly executes exits.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class AKFStrategyConfigV3:
    """Configuration for AKF V3 strategies."""
    k: float = 2.0


class AKFStrategiesV3:
    """
    AKF V3: Always-in-market strategies.

    Key difference from V1:
    - V1: Exit to flat (signal=0) on trend reversion
    - V3: Reverse to opposite position on trend reversion

    This means no flat periods - always Long or Short.
    """

    REQUIRED_COLUMNS = [
        'close',
        'kf_trend',
        'kf_trend_pred',
        'kf_velocity',
        'kf_uncertainty',
    ]

    def __init__(self, config: Optional[AKFStrategyConfigV3] = None):
        self.config = config or AKFStrategyConfigV3()

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

    def strategy_db_reversal(
        self,
        df: pd.DataFrame,
        k: Optional[float] = None,
        include_bands: bool = False
    ) -> pd.DataFrame:
        """
        Dynamic Breakout with Reversal Exit.

        Entry:
            - Long: close > upper_band
            - Short: close < lower_band

        Exit (Reversal):
            - Long exit: close < kf_trend → Switch to Short
            - Short exit: close > kf_trend → Switch to Long

        Always in market after first entry.
        """
        self._validate_input(df)
        k = k if k is not None else self.config.k

        result = df.copy()
        close = df['close'].values
        trend = df['kf_trend'].values
        upper_band, lower_band = self._calculate_bands(df, k)

        # Entry signals (for initial position only)
        entry_long = close > upper_band
        entry_short = close < lower_band

        n = len(df)
        signal = np.zeros(n, dtype=np.int32)
        position = 0

        for i in range(n):
            if position == 0:
                # No position - check for entry
                if entry_long[i]:
                    position = 1
                elif entry_short[i]:
                    position = -1
            elif position == 1:
                # Long position
                if close[i] < trend[i]:
                    # Trend reversion → Reverse to Short
                    position = -1
                elif entry_short[i]:
                    # Breakout short
                    position = -1
            elif position == -1:
                # Short position
                if close[i] > trend[i]:
                    # Trend reversion → Reverse to Long
                    position = 1
                elif entry_long[i]:
                    # Breakout long
                    position = 1

            signal[i] = position

        result['signal'] = signal

        if include_bands:
            result['upper_band'] = upper_band
            result['lower_band'] = lower_band

        return result

    def strategy_benhamou_reversal(
        self,
        df: pd.DataFrame,
        k: Optional[float] = None,
        include_bands: bool = False
    ) -> pd.DataFrame:
        """
        Benhamou Predictive with Reversal.

        Same as V1 Benhamou - already always-in-market (signal reversal).
        """
        self._validate_input(df)
        k = k if k is not None else self.config.k

        result = df.copy()
        trend_pred = df['kf_trend_pred'].values
        prev_close = df['close'].shift(1).values
        prev_uncertainty_sqrt = np.sqrt(df['kf_uncertainty'].shift(1).fillna(1.0).values)

        k_band = k * prev_uncertainty_sqrt
        upper = prev_close + k_band
        lower = prev_close - k_band

        entry_long = trend_pred > upper
        entry_short = trend_pred < lower

        n = len(df)
        signal = np.zeros(n, dtype=np.int32)
        position = 0

        for i in range(n):
            if entry_long[i] and not entry_short[i]:
                position = 1
            elif entry_short[i] and not entry_long[i]:
                position = -1

            signal[i] = position

        result['signal'] = signal

        if include_bands:
            result['upper_band'] = upper
            result['lower_band'] = lower

        return result

    def strategy_hybrid_reversal(
        self,
        df: pd.DataFrame,
        k: Optional[float] = None,
        include_bands: bool = False
    ) -> pd.DataFrame:
        """
        Hybrid Fusion with Reversal Exit.

        Entry: Benhamou + velocity confirmation
        Exit: Trend reversion → Reverse position
        """
        self._validate_input(df)
        k = k if k is not None else self.config.k

        result = df.copy()
        close = df['close'].values
        trend = df['kf_trend'].values
        trend_pred = df['kf_trend_pred'].values
        velocity = df['kf_velocity'].values
        prev_close = df['close'].shift(1).values
        prev_uncertainty_sqrt = np.sqrt(df['kf_uncertainty'].shift(1).fillna(1.0).values)

        k_band = k * prev_uncertainty_sqrt
        upper = prev_close + k_band
        lower = prev_close - k_band

        # Benhamou signals
        benhamou_long = trend_pred > upper
        benhamou_short = trend_pred < lower

        # Velocity confirmation
        velocity_up = velocity > 0
        velocity_down = velocity < 0

        # Entry requires both
        entry_long = benhamou_long & velocity_up
        entry_short = benhamou_short & velocity_down

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
                if close[i] < trend[i]:
                    # Trend reversion → Reverse to Short
                    position = -1
                elif entry_short[i]:
                    position = -1
            elif position == -1:
                if close[i] > trend[i]:
                    # Trend reversion → Reverse to Long
                    position = 1
                elif entry_long[i]:
                    position = 1

            signal[i] = position

        result['signal'] = signal

        if include_bands:
            result['upper_band'] = upper
            result['lower_band'] = lower

        return result

    def get_strategy_names(self) -> list:
        return ['strategy_db_reversal', 'strategy_benhamou_reversal', 'strategy_hybrid_reversal']

    def run_strategy(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        k: Optional[float] = None,
        include_bands: bool = False
    ) -> pd.DataFrame:
        strategies = {
            'strategy_db_reversal': self.strategy_db_reversal,
            'strategy_benhamou_reversal': self.strategy_benhamou_reversal,
            'strategy_hybrid_reversal': self.strategy_hybrid_reversal,
        }

        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        return strategies[strategy_name](df, k=k, include_bands=include_bands)
