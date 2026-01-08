"""
AKF Turtle Strategy - Donchian-Kalman Hybrid.

Combines classic Donchian Channel breakout with Kalman Filter regime detection
to filter out false breakouts.

Entry Logic:
- Donchian breakout (20-bar High/Low)
- Filtered by Kalman velocity direction
- Filtered by uncertainty regime (avoid panic)

Exit Logic:
- Dynamic trailing stop using Kalman trend +/- uncertainty buffer
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class TurtleConfig:
    """Configuration for Donchian-Kalman Turtle Strategy."""

    # Donchian Channel
    donchian_period: int = 20

    # Regime filter
    uncertainty_percentile_cap: float = 95  # Avoid top 5% extreme volatility

    # Velocity filter mode: 'simple' or 'normalized'
    velocity_mode: str = 'simple'
    z_velocity_threshold: float = 1.5  # For 'normalized' mode (t-statistic)

    # Innovation filter: require price > trend_pred for long (momentum confirm)
    use_innovation_filter: bool = False

    # Acceleration filter: require velocity to be increasing (trend strengthening)
    use_acceleration_filter: bool = False

    # Regime-adaptive: adjust entry strictness based on uncertainty level
    use_regime_adaptive: bool = False
    regime_low_percentile: float = 30   # Below this = low uncertainty (aggressive)
    regime_high_percentile: float = 70  # Above this = high uncertainty (conservative)
    regime_high_buffer_mult: float = 0.5  # Extra buffer required in high uncertainty

    # Exit mode: 'kalman', 'donchian', 'atr', 'trend_cross'
    exit_mode: str = 'donchian'

    # Exit parameters by mode
    exit_donchian_period: int = 10  # For 'donchian' mode (classic: half of entry)
    exit_buffer_mult: float = 1.5   # For 'kalman' mode
    exit_atr_mult: float = 2.0      # For 'atr' mode
    atr_period: int = 14            # For 'atr' mode

    # Risk management
    k_sl: float = 2.0
    target_risk_pct: float = 0.02
    max_leverage: float = 3.0


class AKFTurtleStrategy:
    """
    Donchian-Kalman Hybrid Strategy.

    Classic Turtle-style breakout filtered by Kalman regime detection.
    Uses Kalman uncertainty for dynamic trailing stops instead of slow
    Donchian opposite channel.

    Required columns:
    - close, high, low: OHLC data
    - trend, velocity, uncertainty: Kalman filter outputs
    """

    REQUIRED_COLUMNS = [
        'close', 'high', 'low',
        'trend', 'velocity', 'uncertainty', 'trend_pred'
    ]

    def __init__(self, config: Optional[TurtleConfig] = None):
        self.config = config or TurtleConfig()

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate that required columns exist."""
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def calculate_donchian(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Donchian Channels.

        Uses shift(1) to avoid look-ahead bias:
        - Upper: Max High of bars [i-period, i-1]
        - Lower: Min Low of bars [i-period, i-1]
        """
        result = df.copy()
        period = self.config.donchian_period
        exit_period = self.config.exit_donchian_period

        # Entry channels (shift by 1 to use only past data)
        result['donchian_upper'] = (
            result['high']
            .shift(1)
            .rolling(window=period, min_periods=period)
            .max()
        )

        result['donchian_lower'] = (
            result['low']
            .shift(1)
            .rolling(window=period, min_periods=period)
            .min()
        )

        # Exit channels (shorter period for faster exit)
        result['donchian_exit_lower'] = (
            result['low']
            .shift(1)
            .rolling(window=exit_period, min_periods=exit_period)
            .min()
        )

        result['donchian_exit_upper'] = (
            result['high']
            .shift(1)
            .rolling(window=exit_period, min_periods=exit_period)
            .max()
        )

        return result

    def calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range for ATR-based exit."""
        result = df.copy()
        period = self.config.atr_period

        high = result['high']
        low = result['low']
        close = result['close'].shift(1)

        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        result['atr'] = true_range.rolling(window=period, min_periods=period).mean()

        return result

    def detect_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify market regime based on uncertainty.

        regime = 1: Normal (tradeable)
        regime = 0: Extreme uncertainty (avoid)
        """
        result = df.copy()

        # Rolling percentile cap for uncertainty
        uncertainty_cap = result['uncertainty'].rolling(
            window=100, min_periods=20
        ).quantile(self.config.uncertainty_percentile_cap / 100)

        uncertainty_cap = uncertainty_cap.fillna(result['uncertainty'].max())

        # Tradeable if uncertainty not extreme
        result['regime'] = (result['uncertainty'] < uncertainty_cap).astype(int)

        return result

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.

        Entry Logic:
        - Long: close > donchian_upper AND velocity > 0 AND regime == 1
        - Short: close < donchian_lower AND velocity < 0 AND regime == 1

        Exit Logic (configurable via exit_mode):
        - 'donchian': Classic Turtle exit (shorter period Donchian)
        - 'kalman': Kalman trend +/- uncertainty buffer
        - 'atr': ATR-based trailing stop
        - 'trend_cross': Exit when price crosses Kalman trend
        """
        result = df.copy()
        cfg = self.config

        # Regime filter
        is_tradeable = result['regime'] == 1

        # =====================================================================
        # Velocity Filter (simple or normalized)
        # =====================================================================
        if cfg.velocity_mode == 'normalized':
            # Normalized velocity (t-statistic): velocity / sqrt(uncertainty)
            # This checks if velocity is statistically significant
            z_velocity = result['velocity'] / (np.sqrt(result['uncertainty']) + 1e-10)
            result['z_velocity'] = z_velocity
            velocity_up = z_velocity > cfg.z_velocity_threshold
            velocity_down = z_velocity < -cfg.z_velocity_threshold
        else:
            # Simple mode: just check direction
            velocity_up = result['velocity'] > 0
            velocity_down = result['velocity'] < 0

        # =====================================================================
        # Innovation Filter (momentum confirmation)
        # =====================================================================
        if cfg.use_innovation_filter:
            # Innovation = close - trend_pred (prediction error)
            # Positive innovation means price is above prediction (bullish)
            result['innovation'] = result['close'] - result['trend_pred']
            innovation_bullish = result['innovation'] > 0
            innovation_bearish = result['innovation'] < 0
        else:
            innovation_bullish = True
            innovation_bearish = True

        # =====================================================================
        # Acceleration Filter (trend strengthening)
        # =====================================================================
        if cfg.use_acceleration_filter:
            # Acceleration = velocity change (velocity is increasing/decreasing)
            result['acceleration'] = result['velocity'].diff()
            accel_positive = result['acceleration'] > 0  # Velocity increasing
            accel_negative = result['acceleration'] < 0  # Velocity decreasing
        else:
            accel_positive = True
            accel_negative = True

        # =====================================================================
        # Regime-Adaptive Entry (dynamic threshold based on uncertainty)
        # =====================================================================
        if cfg.use_regime_adaptive:
            # Calculate rolling percentiles for uncertainty
            unc_low = result['uncertainty'].rolling(window=100, min_periods=20).quantile(
                cfg.regime_low_percentile / 100
            ).fillna(result['uncertainty'].quantile(cfg.regime_low_percentile / 100))

            unc_high = result['uncertainty'].rolling(window=100, min_periods=20).quantile(
                cfg.regime_high_percentile / 100
            ).fillna(result['uncertainty'].quantile(cfg.regime_high_percentile / 100))

            # In high uncertainty regime, require stronger breakout
            is_high_uncertainty = result['uncertainty'] > unc_high
            extra_buffer = cfg.regime_high_buffer_mult * result['uncertainty']

            # Adjust breakout thresholds
            upper_threshold = np.where(
                is_high_uncertainty,
                result['donchian_upper'] + extra_buffer,
                result['donchian_upper']
            )
            lower_threshold = np.where(
                is_high_uncertainty,
                result['donchian_lower'] - extra_buffer,
                result['donchian_lower']
            )

            result['regime_level'] = np.where(
                result['uncertainty'] < unc_low, 0,  # Low uncertainty
                np.where(result['uncertainty'] > unc_high, 2, 1)  # High / Medium
            )
        else:
            upper_threshold = result['donchian_upper']
            lower_threshold = result['donchian_lower']

        # =====================================================================
        # Entry Conditions
        # =====================================================================
        entry_long = (
            is_tradeable &
            velocity_up &
            innovation_bullish &
            accel_positive &
            (result['close'] > upper_threshold)
        )

        entry_short = (
            is_tradeable &
            velocity_down &
            innovation_bearish &
            accel_negative &
            (result['close'] < lower_threshold)
        )

        # =====================================================================
        # Exit Conditions (based on exit_mode)
        # =====================================================================
        if cfg.exit_mode == 'donchian':
            # Classic Turtle: exit on shorter period Donchian break
            exit_long = result['close'] < result['donchian_exit_lower']
            exit_short = result['close'] > result['donchian_exit_upper']
            result['exit_long_level'] = result['donchian_exit_lower']
            result['exit_short_level'] = result['donchian_exit_upper']

        elif cfg.exit_mode == 'atr':
            # ATR trailing stop
            atr_buffer = cfg.exit_atr_mult * result['atr']
            exit_long = result['close'] < (result['trend'] - atr_buffer)
            exit_short = result['close'] > (result['trend'] + atr_buffer)
            result['exit_long_level'] = result['trend'] - atr_buffer
            result['exit_short_level'] = result['trend'] + atr_buffer

        elif cfg.exit_mode == 'trend_cross':
            # Simple trend cross exit
            exit_long = result['close'] < result['trend']
            exit_short = result['close'] > result['trend']
            result['exit_long_level'] = result['trend']
            result['exit_short_level'] = result['trend']

        else:  # 'kalman' (default)
            # Kalman uncertainty buffer
            exit_buffer = cfg.exit_buffer_mult * result['uncertainty']
            exit_long = result['close'] < (result['trend'] - exit_buffer)
            exit_short = result['close'] > (result['trend'] + exit_buffer)
            result['exit_long_level'] = result['trend'] - exit_buffer
            result['exit_short_level'] = result['trend'] + exit_buffer

        # =====================================================================
        # State Machine
        # =====================================================================
        result['signal'] = self._apply_state_machine(
            entry_long.values,
            entry_short.values,
            exit_long.values,
            exit_short.values
        )

        return result

    def _apply_state_machine(
        self,
        entry_long: np.ndarray,
        entry_short: np.ndarray,
        exit_long: np.ndarray,
        exit_short: np.ndarray
    ) -> np.ndarray:
        """
        State machine for signal generation.

        Handles position transitions:
        - 0 (flat) -> 1 (long) or -1 (short)
        - 1 (long) -> 0 (exit) or -1 (reversal)
        - -1 (short) -> 0 (exit) or 1 (reversal)
        """
        n = len(entry_long)
        signal = np.zeros(n, dtype=np.int32)
        position = 0

        for i in range(n):
            if position == 0:
                # Flat - check for entry
                if entry_long[i]:
                    position = 1
                elif entry_short[i]:
                    position = -1

            elif position == 1:
                # Long - check for exit
                if exit_long[i]:
                    position = 0
                    # Immediate reversal check
                    if entry_short[i]:
                        position = -1
                elif entry_short[i]:
                    # Direct reversal without exit trigger
                    position = -1

            elif position == -1:
                # Short - check for exit
                if exit_short[i]:
                    position = 0
                    # Immediate reversal check
                    if entry_long[i]:
                        position = 1
                elif entry_long[i]:
                    # Direct reversal without exit trigger
                    position = 1

            signal[i] = position

        return signal

    def apply_risk_management(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply position sizing and stop loss levels."""
        result = df.copy()
        cfg = self.config

        # Stop loss distance based on uncertainty
        sl_distance = cfg.k_sl * result['uncertainty']

        # Stop loss price
        result['stop_loss_price'] = np.where(
            result['signal'] == 1,
            result['close'] - sl_distance,
            np.where(
                result['signal'] == -1,
                result['close'] + sl_distance,
                np.nan
            )
        )

        # Volatility-targeted position sizing
        risk_per_unit = sl_distance * result['close']
        target_risk = 10000 * cfg.target_risk_pct  # Assume 10k capital

        raw_size = target_risk / (risk_per_unit + 1e-10)
        leverage = (raw_size * result['close']) / 10000
        leverage = np.clip(leverage, 0, cfg.max_leverage)

        result['position_size'] = np.where(
            result['signal'] != 0, leverage, 0.0
        )

        return result

    def run_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute complete strategy pipeline."""
        self._validate_input(df)

        result = self.calculate_donchian(df)

        # Calculate ATR if needed
        if self.config.exit_mode == 'atr':
            result = self.calculate_atr(result)

        result = self.detect_regime(result)
        result = self.generate_signals(result)
        result = self.apply_risk_management(result)

        return result

    def get_output_columns(self) -> list:
        """Return list of output columns added by this strategy."""
        return [
            'signal', 'regime',
            'donchian_upper', 'donchian_lower',
            'exit_long_level', 'exit_short_level',
            'position_size', 'stop_loss_price'
        ]


def prepare_kalman_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame with required Kalman columns.

    Maps common column names:
    - kf_trend -> trend
    - kf_velocity -> velocity
    - kf_uncertainty -> uncertainty (sqrt applied)
    """
    result = df.copy()

    column_map = {
        'kf_trend': 'trend',
        'kf_velocity': 'velocity',
        'kf_trend_pred': 'trend_pred',
    }

    for old_name, new_name in column_map.items():
        if old_name in result.columns and new_name not in result.columns:
            result[new_name] = result[old_name]

    # Uncertainty: apply sqrt if using covariance (kf_uncertainty)
    if 'kf_uncertainty' in result.columns and 'uncertainty' not in result.columns:
        result['uncertainty'] = np.sqrt(result['kf_uncertainty'])

    return result
