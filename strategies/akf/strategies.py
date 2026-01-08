"""
AKF (Adaptive Kalman Filter) Trading Strategies.

Implements three Kalman filter-based trading strategies:
1. Strategy_DB (Dynamic Breakout): Band breakout with trend reversion exit
2. Strategy_Benhamou (Predictive): Prediction-based entry with signal reversal exit
3. Strategy_Hybrid (Fusion): Benhamou entry + velocity confirmation + DB exit

All strategies use Kalman filter outputs from research/features/kalman.py.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class AKFStrategyConfig:
    """Configuration for AKF strategies."""

    k: float = 2.0  # Band multiplier (uncertainty-based)


class AKFStrategies:
    """
    AKF-based trading strategy generator.

    Uses Kalman filter outputs to generate trading signals:
    - kf_trend: Filtered trend (a posteriori estimate)
    - kf_trend_pred: Predicted trend (a priori estimate)
    - kf_velocity: Trend velocity/momentum
    - kf_uncertainty: Error covariance P (for band width)
    """

    REQUIRED_COLUMNS = [
        'close',
        'kf_trend',
        'kf_trend_pred',
        'kf_velocity',
        'kf_uncertainty',
    ]

    def __init__(self, config: Optional[AKFStrategyConfig] = None):
        """
        Initialize AKF strategies.

        Args:
            config: Strategy configuration with band multiplier k
        """
        self.config = config or AKFStrategyConfig()

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate that required columns exist."""
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _calculate_bands(
        self,
        df: pd.DataFrame,
        k: float,
        use_trend: bool = True
    ) -> tuple:
        """
        Calculate uncertainty-based bands.

        Args:
            df: DataFrame with Kalman filter outputs
            k: Band multiplier
            use_trend: If True, bands around kf_trend; else around prev close

        Returns:
            Tuple of (upper_band, lower_band) arrays
        """
        uncertainty_sqrt = np.sqrt(df['kf_uncertainty'].values)
        k_band = k * uncertainty_sqrt

        if use_trend:
            center = df['kf_trend'].values
        else:
            center = df['close'].shift(1).values

        upper_band = center + k_band
        lower_band = center - k_band

        return upper_band, lower_band

    def strategy_db(
        self,
        df: pd.DataFrame,
        k: Optional[float] = None,
        include_bands: bool = False
    ) -> pd.DataFrame:
        """
        Dynamic Breakout Strategy.

        Entry:
            - Long: close > upper_band (kf_trend + k * sqrt(uncertainty))
            - Short: close < lower_band (kf_trend - k * sqrt(uncertainty))

        Exit:
            - Long exit: close < kf_trend (reversion to trend)
            - Short exit: close > kf_trend (reversion to trend)

        Args:
            df: DataFrame with Kalman filter outputs
            k: Band multiplier (default: config.k)
            include_bands: Include band columns in output

        Returns:
            DataFrame with signal column (1: long, -1: short, 0: flat)
        """
        self._validate_input(df)
        k = k if k is not None else self.config.k

        result = df.copy()
        close = df['close'].values
        trend = df['kf_trend'].values

        upper_band, lower_band = self._calculate_bands(df, k, use_trend=True)

        # Entry signals
        entry_long = close > upper_band
        entry_short = close < lower_band

        n = len(df)
        signal = np.zeros(n, dtype=np.int32)
        position = 0  # Current position

        for i in range(n):
            if position == 0:
                # No position - check for entry
                if entry_long[i]:
                    position = 1
                elif entry_short[i]:
                    position = -1
            elif position == 1:
                # Long position - check for exit
                if close[i] < trend[i]:
                    position = 0
                # Check for reversal to short
                elif entry_short[i]:
                    position = -1
            elif position == -1:
                # Short position - check for exit
                if close[i] > trend[i]:
                    position = 0
                # Check for reversal to long
                elif entry_long[i]:
                    position = 1

            signal[i] = position

        result['signal'] = signal

        if include_bands:
            result['upper_band'] = upper_band
            result['lower_band'] = lower_band

        return result

    def strategy_benhamou(
        self,
        df: pd.DataFrame,
        k: Optional[float] = None,
        include_bands: bool = False
    ) -> pd.DataFrame:
        """
        Benhamou Predictive Strategy.

        Entry (based on a priori prediction):
            - Long: kf_trend_pred > prev_close + k * sqrt(uncertainty)
            - Short: kf_trend_pred < prev_close - k * sqrt(uncertainty)

        Exit:
            - Signal reversal (opposite signal switches position)
            - No separate exit logic; position held until reversal

        Args:
            df: DataFrame with Kalman filter outputs
            k: Band multiplier (default: config.k)
            include_bands: Include band columns in output

        Returns:
            DataFrame with signal column (1: long, -1: short, 0: flat)
        """
        self._validate_input(df)
        k = k if k is not None else self.config.k

        result = df.copy()
        trend_pred = df['kf_trend_pred'].values
        prev_close = df['close'].shift(1).values
        # Use previous bar's uncertainty to avoid look-ahead bias
        prev_uncertainty_sqrt = np.sqrt(df['kf_uncertainty'].shift(1).fillna(1.0).values)

        k_band = k * prev_uncertainty_sqrt
        upper = prev_close + k_band
        lower = prev_close - k_band

        # Raw entry signals
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
            # If both or neither, keep current position

            signal[i] = position

        result['signal'] = signal

        if include_bands:
            result['upper_band'] = upper
            result['lower_band'] = lower

        return result

    def strategy_hybrid(
        self,
        df: pd.DataFrame,
        k: Optional[float] = None,
        include_bands: bool = False
    ) -> pd.DataFrame:
        """
        Hybrid Fusion Strategy.

        Entry:
            - Benhamou signal + velocity direction confirmation
            - Long: benhamou_long AND velocity > 0
            - Short: benhamou_short AND velocity < 0

        Exit:
            - DB-style exit: reversion to trend
            - Long exit: close < kf_trend
            - Short exit: close > kf_trend

        Args:
            df: DataFrame with Kalman filter outputs
            k: Band multiplier (default: config.k)
            include_bands: Include band columns in output

        Returns:
            DataFrame with signal column (1: long, -1: short, 0: flat)
        """
        self._validate_input(df)
        k = k if k is not None else self.config.k

        result = df.copy()
        close = df['close'].values
        trend = df['kf_trend'].values
        trend_pred = df['kf_trend_pred'].values
        velocity = df['kf_velocity'].values
        prev_close = df['close'].shift(1).values
        # Use previous bar's uncertainty to avoid look-ahead bias
        prev_uncertainty_sqrt = np.sqrt(df['kf_uncertainty'].shift(1).fillna(1.0).values)

        k_band = k * prev_uncertainty_sqrt
        upper = prev_close + k_band
        lower = prev_close - k_band

        # Benhamou raw signals
        benhamou_long = trend_pred > upper
        benhamou_short = trend_pred < lower

        # Velocity confirmation
        velocity_up = velocity > 0
        velocity_down = velocity < 0

        # Entry: Benhamou + velocity confirmation
        entry_long = benhamou_long & velocity_up
        entry_short = benhamou_short & velocity_down

        n = len(df)
        signal = np.zeros(n, dtype=np.int32)
        position = 0

        for i in range(n):
            if position == 0:
                # No position - check for confirmed entry
                if entry_long[i]:
                    position = 1
                elif entry_short[i]:
                    position = -1
            elif position == 1:
                # Long position - DB-style exit
                if close[i] < trend[i]:
                    position = 0
                # Allow reversal with confirmation
                elif entry_short[i]:
                    position = -1
            elif position == -1:
                # Short position - DB-style exit
                if close[i] > trend[i]:
                    position = 0
                # Allow reversal with confirmation
                elif entry_long[i]:
                    position = 1

            signal[i] = position

        result['signal'] = signal

        if include_bands:
            result['upper_band'] = upper
            result['lower_band'] = lower

        return result

    def get_strategy_names(self) -> list:
        """Return list of available strategy names."""
        return ['strategy_db', 'strategy_benhamou', 'strategy_hybrid']

    def run_strategy(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        k: Optional[float] = None,
        include_bands: bool = False
    ) -> pd.DataFrame:
        """
        Run a strategy by name.

        Args:
            df: DataFrame with Kalman filter outputs
            strategy_name: One of 'strategy_db', 'strategy_benhamou', 'strategy_hybrid'
            k: Band multiplier
            include_bands: Include band columns in output

        Returns:
            DataFrame with signal column
        """
        strategies = {
            'strategy_db': self.strategy_db,
            'strategy_benhamou': self.strategy_benhamou,
            'strategy_hybrid': self.strategy_hybrid,
        }

        if strategy_name not in strategies:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Available: {list(strategies.keys())}"
            )

        return strategies[strategy_name](df, k=k, include_bands=include_bands)


def apply_akf_strategy(
    df: pd.DataFrame,
    strategy: str = 'strategy_db',
    k: float = 2.0,
    include_bands: bool = False
) -> pd.DataFrame:
    """
    Convenience function to apply AKF strategy.

    Args:
        df: DataFrame with Kalman filter outputs
        strategy: Strategy name ('strategy_db', 'strategy_benhamou', 'strategy_hybrid')
        k: Band multiplier
        include_bands: Include band columns in output

    Returns:
        DataFrame with signal column
    """
    config = AKFStrategyConfig(k=k)
    akf = AKFStrategies(config)
    return akf.run_strategy(df, strategy, k=k, include_bands=include_bands)
