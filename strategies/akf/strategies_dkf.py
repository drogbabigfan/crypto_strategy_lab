"""
Dual Kalman Filter (DKF) Crossover Strategy.

Runs TWO Kalman Filters with different sensitivities:
- Fast KF: High reactivity (low R) - catches quick moves
- Slow KF: High stability (high R) - captures trend

Signal Logic:
- Long: Fast crosses above Slow + Slow velocity not falling sharply
- Short: Fast crosses below Slow + Slow velocity not rising sharply

Similar to MACD but with adaptive noise estimation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DKFConfig:
    """Configuration for Dual Kalman Filter Strategy."""

    # Kalman Filter parameters
    fast_r_scale: float = 0.1    # Fast KF: low R = high reactivity
    slow_r_scale: float = 1.0    # Slow KF: high R = smooth trend
    q_scale: float = 0.1         # Process noise scale (same for both)

    # Windows for adaptive noise
    r_window: int = 20
    q_window: int = 20

    # Velocity filter thresholds
    velocity_filter_long: float = -0.05   # Slow velocity must be > this for long
    velocity_filter_short: float = 0.05   # Slow velocity must be < this for short

    # Use typical price
    use_typical_price: bool = True


class DualKalmanFilter:
    """
    Kalman Filter implementation for DKF strategy.

    State: [trend, velocity]
    Observation: price
    """

    def __init__(
        self,
        r_scale: float = 1.0,
        q_scale: float = 0.1,
        r_window: int = 20,
        q_window: int = 20,
        initial_p: float = 1.0
    ):
        self.r_scale = r_scale
        self.q_scale = q_scale
        self.r_window = r_window
        self.q_window = q_window
        self.initial_p = initial_p

        # State transition: constant velocity model
        self.F = np.array([[1.0, 1.0],
                          [0.0, 1.0]])
        # Observation matrix
        self.H = np.array([[1.0, 0.0]])

    def _estimate_r(self, prices: np.ndarray, idx: int) -> float:
        """Estimate measurement noise from recent variance."""
        start = max(0, idx - self.r_window + 1)
        if start >= idx:
            return self.initial_p * self.r_scale

        recent = prices[start:idx + 1]
        if len(recent) < 2:
            return self.initial_p * self.r_scale

        returns = np.diff(recent)
        r = np.var(returns) if len(returns) > 0 else self.initial_p
        return max(r * self.r_scale, 1e-8)

    def _estimate_q(self, velocities: list, idx: int) -> float:
        """Estimate process noise from velocity changes."""
        start = max(0, idx - self.q_window + 1)
        if start >= idx or len(velocities) < 2:
            return 1e-10

        recent_vel = velocities[start:min(idx + 1, len(velocities))]
        if len(recent_vel) < 2:
            return 1e-10

        vel_var = np.var(recent_vel)
        return max(vel_var * self.q_scale, 1e-10)

    def filter(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply Kalman Filter.

        Returns:
            trend: Filtered price level
            velocity: Estimated slope
            uncertainty: Error covariance P[0,0]
        """
        n = len(prices)

        trend = np.full(n, np.nan)
        velocity = np.full(n, np.nan)
        uncertainty = np.full(n, np.nan)

        # Initialize state
        x = np.array([prices[0], 0.0])
        P = np.array([[self.initial_p, 0.0],
                      [0.0, self.initial_p]])

        vel_history = []

        for i in range(n):
            z = prices[i]

            # Prediction
            x_pred = self.F @ x
            Q_t = self._estimate_q(vel_history, i)
            Q_matrix = np.array([[Q_t, 0.0], [0.0, Q_t]])
            P_pred = self.F @ P @ self.F.T + Q_matrix

            # Update
            R_t = self._estimate_r(prices, i)
            y = z - (self.H @ x_pred)[0]
            S = (self.H @ P_pred @ self.H.T)[0, 0] + R_t
            K = (P_pred @ self.H.T) / S
            x = x_pred + K.flatten() * y
            P = (np.eye(2) - K @ self.H) @ P_pred

            # Store
            trend[i] = x[0]
            velocity[i] = x[1]
            uncertainty[i] = P[0, 0]
            vel_history.append(x[1])

        return trend, velocity, uncertainty


class DKFStrategy:
    """
    Dual Kalman Filter Crossover Strategy.

    Uses two KFs with different R values:
    - Fast (low R): Reacts quickly to price changes
    - Slow (high R): Smooth trend line

    Crossover signals filtered by slow velocity.
    """

    def __init__(self, config: Optional[DKFConfig] = None):
        self.config = config or DKFConfig()

    def _get_price(self, df: pd.DataFrame) -> np.ndarray:
        """Get price series (typical or close)."""
        if self.config.use_typical_price and 'high' in df.columns and 'low' in df.columns:
            return ((df['high'] + df['low'] + df['close']) / 3).values
        return df['close'].values

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate dual KF indicators.

        Returns DataFrame with:
        - trend_fast, velocity_fast
        - trend_slow, velocity_slow
        - spread (K-MACD)
        """
        result = df.copy()
        cfg = self.config

        prices = self._get_price(df)

        # Fast KF (reactive)
        kf_fast = DualKalmanFilter(
            r_scale=cfg.fast_r_scale,
            q_scale=cfg.q_scale,
            r_window=cfg.r_window,
            q_window=cfg.q_window
        )
        trend_fast, velocity_fast, _ = kf_fast.filter(prices)

        # Slow KF (smooth)
        kf_slow = DualKalmanFilter(
            r_scale=cfg.slow_r_scale,
            q_scale=cfg.q_scale,
            r_window=cfg.r_window,
            q_window=cfg.q_window
        )
        trend_slow, velocity_slow, _ = kf_slow.filter(prices)

        result['trend_fast'] = trend_fast
        result['velocity_fast'] = velocity_fast
        result['trend_slow'] = trend_slow
        result['velocity_slow'] = velocity_slow
        result['spread'] = trend_fast - trend_slow  # K-MACD

        return result

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate crossover signals (vectorized).

        Long: Fast crosses above Slow + velocity_slow > threshold
        Short: Fast crosses below Slow + velocity_slow < threshold
        """
        result = df.copy()
        cfg = self.config

        # Crossover detection using sign change
        spread = result['spread']
        spread_sign = np.sign(spread)
        spread_sign_prev = spread_sign.shift(1).fillna(0)

        # Crossover events
        golden_cross = (spread_sign == 1) & (spread_sign_prev <= 0)  # Fast > Slow
        dead_cross = (spread_sign == -1) & (spread_sign_prev >= 0)   # Fast < Slow

        # Velocity filters
        velocity_ok_long = result['velocity_slow'] > cfg.velocity_filter_long
        velocity_ok_short = result['velocity_slow'] < cfg.velocity_filter_short

        # Entry signals
        entry_long = golden_cross & velocity_ok_long
        entry_short = dead_cross & velocity_ok_short

        # Exit signals (opposite crossover)
        exit_long = dead_cross
        exit_short = golden_cross

        # State machine (vectorized with loop for correctness)
        n = len(df)
        signal = np.zeros(n, dtype=np.int32)
        position = 0

        for i in range(n):
            if position == 0:
                if entry_long.iloc[i]:
                    position = 1
                elif entry_short.iloc[i]:
                    position = -1
            elif position == 1:
                if exit_long.iloc[i]:
                    position = 0
                    # Check for immediate reversal
                    if entry_short.iloc[i]:
                        position = -1
            elif position == -1:
                if exit_short.iloc[i]:
                    position = 0
                    if entry_long.iloc[i]:
                        position = 1

            signal[i] = position

        result['signal'] = signal

        # Debug columns
        result['golden_cross'] = golden_cross.astype(int)
        result['dead_cross'] = dead_cross.astype(int)

        return result

    def run_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute complete strategy pipeline."""
        result = self.calculate_indicators(df)
        result = self.generate_signals(result)
        return result

    def get_output_columns(self) -> list:
        return [
            'trend_fast', 'trend_slow', 'spread',
            'velocity_slow', 'signal'
        ]
