"""
Benhamou's Model 4 Strategy Implementation.

Based on "Trend Without Hiccups: A Kalman Filter Approach" (2018)

Model 4 combines:
1. Kalman Filter prediction for trend direction
2. Stochastic Oscillator for overbought/oversold filtering

Entry Logic:
- Long: KF_predict > Close[t-1] + delta AND NOT overbought
- Short: KF_predict < Close[t-1] - delta AND NOT oversold

Exit: Reverse on opposite signal (switching strategy)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class BenhamouModel4Config:
    """Configuration for Benhamou Model 4 strategy."""

    # Threshold multiplier (delta = k * volatility)
    k: float = 2.0

    # Oscillator parameters
    osc_period: int = 14
    osc_overbought: float = 80.0
    osc_oversold: float = 20.0

    # Oscillator filter mode
    use_osc_filter: bool = True  # If False, ignore oscillator (pure Kalman)


class BenhamouModel4:
    """
    Benhamou's Model 4: Kalman + Oscillator Strategy.

    Key Insight:
    - Kalman prediction gives trend direction
    - Oscillator prevents entry at extreme levels (mean reversion zones)
    - Combining both should reduce false signals

    Entry Conditions:
    - Long: kf_trend_pred > close[t-1] + k*sqrt(uncertainty) AND stoch < overbought
    - Short: kf_trend_pred < close[t-1] - k*sqrt(uncertainty) AND stoch > oversold

    Exit: Switch on opposite signal (always in market when triggered)
    """

    REQUIRED_COLUMNS = [
        'close',
        'kf_trend',
        'kf_trend_pred',
        'kf_uncertainty',
        'stoch_k_smooth',
    ]

    def __init__(self, config: Optional[BenhamouModel4Config] = None):
        self.config = config or BenhamouModel4Config()

    def _validate_input(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def strategy_model4_basic(
        self,
        df: pd.DataFrame,
        k: Optional[float] = None,
        include_debug: bool = False
    ) -> pd.DataFrame:
        """
        Basic Model 4 Strategy with oscillator filter.

        Entry:
            - Long: kf_trend_pred > prev_close + delta AND stoch < overbought
            - Short: kf_trend_pred < prev_close - delta AND stoch > oversold

        Exit: Switch on opposite signal
        """
        self._validate_input(df)
        k = k if k is not None else self.config.k

        result = df.copy()

        close = df['close'].values
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        trend_pred = df['kf_trend_pred'].values
        uncertainty_sqrt = np.sqrt(df['kf_uncertainty'].values)
        stoch = df['stoch_k_smooth'].values

        # Threshold (delta)
        delta = k * uncertainty_sqrt

        # Kalman prediction signals
        kalman_long = trend_pred > (prev_close + delta)
        kalman_short = trend_pred < (prev_close - delta)

        # Oscillator filter
        if self.config.use_osc_filter:
            not_overbought = stoch < self.config.osc_overbought
            not_oversold = stoch > self.config.osc_oversold

            entry_long = kalman_long & not_overbought
            entry_short = kalman_short & not_oversold
        else:
            entry_long = kalman_long
            entry_short = kalman_short

        # Generate signals with state machine (switching on opposite)
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
                # In long: switch to short on short signal
                if entry_short[i]:
                    position = -1
            elif position == -1:
                # In short: switch to long on long signal
                if entry_long[i]:
                    position = 1

            signal[i] = position

        result['signal'] = signal

        if include_debug:
            result['kalman_long'] = kalman_long.astype(int)
            result['kalman_short'] = kalman_short.astype(int)
            result['entry_long'] = entry_long.astype(int)
            result['entry_short'] = entry_short.astype(int)
            result['delta'] = delta

        return result

    def strategy_model4_with_exit(
        self,
        df: pd.DataFrame,
        k: Optional[float] = None,
        exit_k: float = 0.5,
        include_debug: bool = False
    ) -> pd.DataFrame:
        """
        Model 4 with trend reversion exit (not pure switching).

        Entry: Same as basic
        Exit: When price reverts to trend (within exit_k * uncertainty)
        """
        self._validate_input(df)
        k = k if k is not None else self.config.k

        result = df.copy()

        close = df['close'].values
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        trend = df['kf_trend'].values
        trend_pred = df['kf_trend_pred'].values
        uncertainty_sqrt = np.sqrt(df['kf_uncertainty'].values)
        stoch = df['stoch_k_smooth'].values

        # Entry threshold
        delta = k * uncertainty_sqrt

        # Exit threshold (tighter)
        exit_delta = exit_k * uncertainty_sqrt

        # Entry conditions
        kalman_long = trend_pred > (prev_close + delta)
        kalman_short = trend_pred < (prev_close - delta)

        if self.config.use_osc_filter:
            not_overbought = stoch < self.config.osc_overbought
            not_oversold = stoch > self.config.osc_oversold

            entry_long = kalman_long & not_overbought
            entry_short = kalman_short & not_oversold
        else:
            entry_long = kalman_long
            entry_short = kalman_short

        # Exit conditions: price near trend
        exit_long = close > (trend - exit_delta)  # Price above trend - exit_delta
        exit_short = close < (trend + exit_delta)  # Price below trend + exit_delta

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
                # Exit on reversion to trend OR opposite signal
                if exit_long[i] and close[i] > trend[i]:
                    position = 0
                elif entry_short[i]:
                    position = -1
            elif position == -1:
                if exit_short[i] and close[i] < trend[i]:
                    position = 0
                elif entry_long[i]:
                    position = 1

            signal[i] = position

        result['signal'] = signal

        if include_debug:
            result['entry_long'] = entry_long.astype(int)
            result['entry_short'] = entry_short.astype(int)
            result['exit_long'] = exit_long.astype(int)
            result['exit_short'] = exit_short.astype(int)

        return result

    def strategy_model4_oscillator_reversal(
        self,
        df: pd.DataFrame,
        k: Optional[float] = None,
        include_debug: bool = False
    ) -> pd.DataFrame:
        """
        Model 4 with Oscillator-based mean reversion.

        Uses oscillator extremes for ENTRY (contrary to basic):
        - Long: Oversold (stoch < 20) + Kalman suggests upside potential
        - Short: Overbought (stoch > 80) + Kalman suggests downside potential

        This is the opposite logic - fade extremes with Kalman confirmation.
        """
        self._validate_input(df)
        k = k if k is not None else self.config.k

        result = df.copy()

        close = df['close'].values
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        trend = df['kf_trend'].values
        trend_pred = df['kf_trend_pred'].values
        uncertainty_sqrt = np.sqrt(df['kf_uncertainty'].values)
        stoch = df['stoch_k_smooth'].values

        # Oscillator extremes
        is_oversold = stoch < self.config.osc_oversold
        is_overbought = stoch > self.config.osc_overbought

        # Kalman suggests upside/downside potential
        # Not a strong trend yet, but prediction leans one way
        kalman_upside = trend_pred > prev_close
        kalman_downside = trend_pred < prev_close

        # Entry: fade the extreme with Kalman confirmation
        entry_long = is_oversold & kalman_upside
        entry_short = is_overbought & kalman_downside

        # Exit: when oscillator returns to neutral
        exit_long = stoch > 50  # Moved back to neutral
        exit_short = stoch < 50

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
                if exit_long[i]:
                    position = 0
                elif entry_short[i]:
                    position = -1
            elif position == -1:
                if exit_short[i]:
                    position = 0
                elif entry_long[i]:
                    position = 1

            signal[i] = position

        result['signal'] = signal

        if include_debug:
            result['is_oversold'] = is_oversold.astype(int)
            result['is_overbought'] = is_overbought.astype(int)
            result['entry_long'] = entry_long.astype(int)
            result['entry_short'] = entry_short.astype(int)

        return result

    def strategy_model4_combo(
        self,
        df: pd.DataFrame,
        k: Optional[float] = None,
        include_debug: bool = False
    ) -> pd.DataFrame:
        """
        Combined Model 4: Trend following + Mean Reversion.

        - In trending market (neutral oscillator): Use Kalman breakout
        - In extreme oscillator: Use mean reversion

        This adapts to market regime.
        """
        self._validate_input(df)
        k = k if k is not None else self.config.k

        result = df.copy()

        close = df['close'].values
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        trend = df['kf_trend'].values
        trend_pred = df['kf_trend_pred'].values
        velocity = df['kf_velocity'].values
        uncertainty_sqrt = np.sqrt(df['kf_uncertainty'].values)
        stoch = df['stoch_k_smooth'].values

        # Threshold
        delta = k * uncertainty_sqrt

        # Zones
        is_oversold = stoch < self.config.osc_oversold
        is_overbought = stoch > self.config.osc_overbought
        is_neutral = ~is_oversold & ~is_overbought

        # Trend following signals (in neutral zone)
        trend_long = is_neutral & (trend_pred > prev_close + delta) & (velocity > 0)
        trend_short = is_neutral & (trend_pred < prev_close - delta) & (velocity < 0)

        # Mean reversion signals (in extreme zones)
        mr_long = is_oversold & (trend_pred > prev_close)  # Oversold + upside prediction
        mr_short = is_overbought & (trend_pred < prev_close)  # Overbought + downside

        # Combined entry
        entry_long = trend_long | mr_long
        entry_short = trend_short | mr_short

        n = len(df)
        signal = np.zeros(n, dtype=np.int32)
        position = 0
        entry_type = None  # 'trend' or 'mr'

        for i in range(n):
            if position == 0:
                if entry_long[i]:
                    position = 1
                    entry_type = 'mr' if mr_long[i] else 'trend'
                elif entry_short[i]:
                    position = -1
                    entry_type = 'mr' if mr_short[i] else 'trend'
            elif position == 1:
                # Exit logic depends on entry type
                if entry_type == 'mr':
                    # MR exit: oscillator returns to neutral
                    if stoch[i] > 50:
                        position = 0
                        entry_type = None
                else:
                    # Trend exit: price below trend
                    if close[i] < trend[i]:
                        position = 0
                        entry_type = None

                # Allow reversal
                if entry_short[i]:
                    position = -1
                    entry_type = 'mr' if mr_short[i] else 'trend'

            elif position == -1:
                if entry_type == 'mr':
                    if stoch[i] < 50:
                        position = 0
                        entry_type = None
                else:
                    if close[i] > trend[i]:
                        position = 0
                        entry_type = None

                if entry_long[i]:
                    position = 1
                    entry_type = 'mr' if mr_long[i] else 'trend'

            signal[i] = position

        result['signal'] = signal

        if include_debug:
            result['trend_long'] = trend_long.astype(int)
            result['trend_short'] = trend_short.astype(int)
            result['mr_long'] = mr_long.astype(int)
            result['mr_short'] = mr_short.astype(int)

        return result

    def get_strategy_names(self) -> list:
        return [
            'strategy_model4_basic',
            'strategy_model4_with_exit',
            'strategy_model4_oscillator_reversal',
            'strategy_model4_combo',
        ]

    def run_strategy(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        k: Optional[float] = None,
        include_debug: bool = False
    ) -> pd.DataFrame:
        strategies = {
            'strategy_model4_basic': self.strategy_model4_basic,
            'strategy_model4_with_exit': self.strategy_model4_with_exit,
            'strategy_model4_oscillator_reversal': self.strategy_model4_oscillator_reversal,
            'strategy_model4_combo': self.strategy_model4_combo,
        }

        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        return strategies[strategy_name](df, k=k, include_debug=include_debug)
