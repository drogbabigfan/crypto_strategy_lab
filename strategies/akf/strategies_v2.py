"""
AKF Trading Strategies V2.

4 distinct strategies based on Adaptive Kalman Filter outputs:
1. Strategy_DB (Dynamic Breakout) - Trend-Following with velocity filter
2. Strategy_Benhamou (Predictive) - Priori prediction based
3. Strategy_Triple_Fusion (Hybrid) - Predictive entry + Momentum filter + Trend exit
4. Strategy_Velocity_Scalp (Pure Momentum) - Normalized velocity based

Changes from V1:
- Strategy_DB: Added velocity direction filter for entry
- Strategy_Velocity_Scalp: New pure momentum strategy
- Unified interface with run_all_strategies()
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AKFStrategyConfigV2:
    """Configuration for AKF strategies V2."""

    # Strategy_DB parameters
    db_k: float = 1.5  # Band multiplier for DB

    # Strategy_Benhamou parameters
    benhamou_k: float = 1.0  # Threshold multiplier for Benhamou

    # Strategy_Triple_Fusion parameters
    fusion_k: float = 1.0  # Threshold multiplier for Fusion

    # Strategy_Velocity_Scalp parameters
    velocity_threshold: float = 0.0005  # Normalized velocity threshold


class AKFStrategiesV2:
    """
    Adaptive Kalman Filter based trading strategies V2.

    Input DataFrame columns:
        - close: Actual close price
        - trend: Posterior state estimate (kf_trend)
        - velocity: Posterior slope estimate (kf_velocity)
        - trend_pred: Priori state estimate (kf_trend_pred)
        - uncertainty: Sqrt of posterior covariance (sqrt(kf_uncertainty))

    All strategies output:
        - signal: Entry signal (1=Long, -1=Short, 0=Neutral)
        - position: Current position after applying entry/exit logic
    """

    # Column mapping from our Kalman filter output
    COLUMN_MAP = {
        "close": "close",
        "trend": "kf_trend",
        "velocity": "kf_velocity",
        "trend_pred": "kf_trend_pred",
        "uncertainty": "kf_uncertainty",
    }

    def __init__(self, config: Optional[AKFStrategyConfigV2] = None):
        """Initialize with configuration."""
        self.config = config or AKFStrategyConfigV2()

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data by mapping column names and computing derived values.

        Returns DataFrame with standardized column names.
        """
        result = df.copy()

        # Map columns if using Kalman filter output names
        for standard_name, kf_name in self.COLUMN_MAP.items():
            if kf_name in df.columns and standard_name not in df.columns:
                if standard_name == "uncertainty":
                    # Convert covariance to sqrt
                    result[standard_name] = np.sqrt(df[kf_name].values)
                else:
                    result[standard_name] = df[kf_name].values

        return result

    def _validate_columns(self, df: pd.DataFrame, required: list[str]) -> None:
        """Validate that required columns exist."""
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def run_strategy_db(
        self, df: pd.DataFrame, k: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Strategy_DB: Dynamic Breakout with velocity filter.

        Entry Logic:
            - Long: close > upper_band AND velocity > 0
            - Short: close < lower_band AND velocity < 0

        Exit Logic:
            - Long Exit: close < trend
            - Short Exit: close > trend

        Args:
            df: DataFrame with required columns
            k: Band multiplier (default from config)

        Returns:
            DataFrame with 'signal' and 'position' columns
        """
        k = k if k is not None else self.config.db_k
        result = self._prepare_data(df)
        self._validate_columns(
            result, ["close", "trend", "velocity", "uncertainty"]
        )

        close = result["close"].values
        trend = result["trend"].values
        velocity = result["velocity"].values
        uncertainty = result["uncertainty"].values

        # Calculate bands
        upper_band = trend + k * uncertainty
        lower_band = trend - k * uncertainty

        # Entry signals with velocity filter
        long_entry = (close > upper_band) & (velocity > 0)
        short_entry = (close < lower_band) & (velocity < 0)

        # Raw entry signal
        entry_signal = np.where(long_entry, 1, np.where(short_entry, -1, 0))

        # Exit conditions
        long_exit = close < trend
        short_exit = close > trend

        # Apply position logic with exits
        position = self._apply_position_with_exit(
            entry_signal, long_exit, short_exit
        )

        result["signal"] = entry_signal
        result["position"] = position
        result["upper_band"] = upper_band
        result["lower_band"] = lower_band

        return result

    def run_strategy_benhamou(
        self, df: pd.DataFrame, k: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Strategy_Benhamou: Predictive strategy using priori estimate.

        Uses t-1 values for uncertainty and close to prevent look-ahead bias.

        Entry Logic:
            - Threshold = k * uncertainty[t-1]
            - Long: trend_pred > close[t-1] + Threshold
            - Short: trend_pred < close[t-1] - Threshold

        Exit Logic:
            - Signal reversal (Always-in-Market style)
            - Neutral if within threshold

        Args:
            df: DataFrame with required columns
            k: Threshold multiplier (default from config)

        Returns:
            DataFrame with 'signal' and 'position' columns
        """
        k = k if k is not None else self.config.benhamou_k
        result = self._prepare_data(df)
        self._validate_columns(
            result, ["close", "trend_pred", "uncertainty"]
        )

        close = result["close"].values
        trend_pred = result["trend_pred"].values
        uncertainty = result["uncertainty"].values

        # Use t-1 values to prevent look-ahead bias
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        prev_uncertainty = np.roll(uncertainty, 1)
        prev_uncertainty[0] = uncertainty[0]

        # Calculate threshold
        threshold = k * prev_uncertainty

        # Entry signals
        long_signal = trend_pred > (prev_close + threshold)
        short_signal = trend_pred < (prev_close - threshold)

        # Signal: Always-in-Market style with neutral zone
        signal = np.where(long_signal, 1, np.where(short_signal, -1, 0))

        # Position follows signal directly (signal reversal exits)
        position = self._apply_signal_based_position(signal)

        result["signal"] = signal
        result["position"] = position
        result["threshold"] = threshold

        return result

    def run_strategy_triple_fusion(
        self, df: pd.DataFrame, k: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Strategy_Triple_Fusion: Hybrid combining predictive + momentum + trend.

        Entry Logic (Strict):
            - Long: (Benhamou Long) AND (velocity > 0)
            - Short: (Benhamou Short) AND (velocity < 0)

        Exit Logic (Loose):
            - Long Exit: close < trend
            - Short Exit: close > trend

        Args:
            df: DataFrame with required columns
            k: Threshold multiplier (default from config)

        Returns:
            DataFrame with 'signal' and 'position' columns
        """
        k = k if k is not None else self.config.fusion_k
        result = self._prepare_data(df)
        self._validate_columns(
            result, ["close", "trend", "velocity", "trend_pred", "uncertainty"]
        )

        close = result["close"].values
        trend = result["trend"].values
        velocity = result["velocity"].values
        trend_pred = result["trend_pred"].values
        uncertainty = result["uncertainty"].values

        # Use t-1 values for Benhamou logic
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        prev_uncertainty = np.roll(uncertainty, 1)
        prev_uncertainty[0] = uncertainty[0]

        threshold = k * prev_uncertainty

        # Benhamou conditions
        benhamou_long = trend_pred > (prev_close + threshold)
        benhamou_short = trend_pred < (prev_close - threshold)

        # Combined entry with velocity filter
        long_entry = benhamou_long & (velocity > 0)
        short_entry = benhamou_short & (velocity < 0)

        entry_signal = np.where(long_entry, 1, np.where(short_entry, -1, 0))

        # Exit conditions (loose - trend based)
        long_exit = close < trend
        short_exit = close > trend

        # Apply position logic
        position = self._apply_position_with_exit(
            entry_signal, long_exit, short_exit
        )

        result["signal"] = entry_signal
        result["position"] = position
        result["threshold"] = threshold

        return result

    def run_strategy_velocity_scalp(
        self, df: pd.DataFrame, threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Strategy_Velocity_Scalp: Pure momentum based on normalized velocity.

        Entry Logic:
            - norm_vel = velocity / close (price-normalized)
            - Long: norm_vel > threshold
            - Short: norm_vel < -threshold

        Exit Logic:
            - Long Exit: velocity < 0
            - Short Exit: velocity > 0

        Args:
            df: DataFrame with required columns
            threshold: Normalized velocity threshold (default from config)

        Returns:
            DataFrame with 'signal' and 'position' columns
        """
        threshold = threshold if threshold is not None else self.config.velocity_threshold
        result = self._prepare_data(df)
        self._validate_columns(result, ["close", "velocity"])

        close = result["close"].values
        velocity = result["velocity"].values

        # Normalize velocity by price
        norm_vel = velocity / close

        # Entry signals
        long_entry = norm_vel > threshold
        short_entry = norm_vel < -threshold

        entry_signal = np.where(long_entry, 1, np.where(short_entry, -1, 0))

        # Exit conditions (momentum lost)
        long_exit = velocity < 0
        short_exit = velocity > 0

        # Apply position logic
        position = self._apply_position_with_exit(
            entry_signal, long_exit, short_exit
        )

        result["signal"] = entry_signal
        result["position"] = position
        result["norm_velocity"] = norm_vel

        return result

    def _apply_position_with_exit(
        self,
        entry_signal: np.ndarray,
        long_exit: np.ndarray,
        short_exit: np.ndarray,
    ) -> np.ndarray:
        """
        Apply position logic with explicit exit conditions.

        Position is maintained until:
        - Opposite entry signal
        - Exit condition triggered

        Args:
            entry_signal: Entry signals (1, -1, 0)
            long_exit: Boolean array for long exit conditions
            short_exit: Boolean array for short exit conditions

        Returns:
            Position array (1, -1, 0)
        """
        n = len(entry_signal)
        position = np.zeros(n, dtype=np.int8)

        current_pos = 0
        for i in range(n):
            # Check exit conditions first
            if current_pos == 1 and long_exit[i]:
                current_pos = 0
            elif current_pos == -1 and short_exit[i]:
                current_pos = 0

            # Check entry signals
            if entry_signal[i] == 1 and current_pos != 1:
                current_pos = 1
            elif entry_signal[i] == -1 and current_pos != -1:
                current_pos = -1

            position[i] = current_pos

        return position

    def _apply_signal_based_position(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply signal-based position (always-in-market with neutral).

        Position changes only on non-zero signals.

        Args:
            signal: Entry signals (1, -1, 0)

        Returns:
            Position array (1, -1, 0)
        """
        n = len(signal)
        position = np.zeros(n, dtype=np.int8)

        current_pos = 0
        for i in range(n):
            if signal[i] != 0:
                current_pos = signal[i]
            position[i] = current_pos

        return position

    def run_all_strategies(
        self, df: pd.DataFrame, params: Optional[dict] = None
    ) -> dict[str, pd.DataFrame]:
        """
        Run all 4 strategies and return results for comparison.

        Args:
            df: DataFrame with required columns
            params: Optional dict to override default parameters
                    e.g., {'db_k': 2.0, 'velocity_threshold': 0.001}

        Returns:
            Dictionary mapping strategy name to result DataFrame
        """
        params = params or {}

        results = {}

        # Strategy_DB
        db_k = params.get("db_k", self.config.db_k)
        results["strategy_db"] = self.run_strategy_db(df, k=db_k)

        # Strategy_Benhamou
        benhamou_k = params.get("benhamou_k", self.config.benhamou_k)
        results["strategy_benhamou"] = self.run_strategy_benhamou(df, k=benhamou_k)

        # Strategy_Triple_Fusion
        fusion_k = params.get("fusion_k", self.config.fusion_k)
        results["strategy_triple_fusion"] = self.run_strategy_triple_fusion(df, k=fusion_k)

        # Strategy_Velocity_Scalp
        vel_threshold = params.get("velocity_threshold", self.config.velocity_threshold)
        results["strategy_velocity_scalp"] = self.run_strategy_velocity_scalp(
            df, threshold=vel_threshold
        )

        return results

    def get_strategy_summary(self, results: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate summary statistics for all strategies.

        Args:
            results: Dictionary from run_all_strategies()

        Returns:
            Summary DataFrame with signal statistics
        """
        summary_data = []

        for name, df in results.items():
            if "position" not in df.columns:
                continue

            position = df["position"].values
            total_bars = len(position)

            # Count positions
            long_bars = np.sum(position == 1)
            short_bars = np.sum(position == -1)
            flat_bars = np.sum(position == 0)

            # Count position changes (trades)
            pos_changes = np.diff(position)
            entries = np.sum(pos_changes != 0)

            summary_data.append({
                "strategy": name,
                "total_bars": total_bars,
                "long_bars": long_bars,
                "short_bars": short_bars,
                "flat_bars": flat_bars,
                "long_pct": long_bars / total_bars * 100,
                "short_pct": short_bars / total_bars * 100,
                "flat_pct": flat_bars / total_bars * 100,
                "position_changes": entries,
            })

        return pd.DataFrame(summary_data)


# Convenience function for quick testing
def run_akf_strategies_v2(
    df: pd.DataFrame,
    params: Optional[dict] = None,
) -> dict[str, pd.DataFrame]:
    """
    Convenience function to run all V2 strategies.

    Args:
        df: DataFrame with Kalman filter outputs
        params: Optional parameter overrides

    Returns:
        Dictionary of strategy results
    """
    strategies = AKFStrategiesV2()
    return strategies.run_all_strategies(df, params)
