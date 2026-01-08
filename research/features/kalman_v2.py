"""
Kalman Filter V2 - 3-State Model with Acceleration.

State: [position, velocity, acceleration]
- position: 현재 가격 수준 (log scale)
- velocity: 1차 변화율 (log-return per bar)
- acceleration: 2차 변화율 (velocity의 변화율)

Acceleration을 추가하면:
1. 추세 전환을 더 빨리 감지
2. 모멘텀 가속/감속 구분
3. 더 정확한 예측
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass


@dataclass
class KalmanV2Config:
    """Configuration for 3-State Kalman Filter."""

    # Noise estimation windows
    r_window: int = 20
    q_window: int = 20

    # Initial covariance
    initial_p: float = 1.0

    # Noise scaling
    r_scale: float = 1.0
    q_trend: float = 0.01      # Process noise for trend
    q_velocity: float = 0.1    # Process noise for velocity
    q_accel: float = 1.0       # Process noise for acceleration

    # Price type
    use_typical_price: bool = True
    use_log_price: bool = True

    # Minimum noise floors
    min_r: float = 1e-8
    min_q: float = 1e-10


class AdaptiveKalmanFilterV2:
    """
    3-State Adaptive Kalman Filter with Acceleration.

    State model:
    x = [position, velocity, acceleration]^T

    State transition (constant acceleration model):
    F = [[1, 1, 0.5],
         [0, 1, 1  ],
         [0, 0, 1  ]]

    Measurement:
    H = [1, 0, 0]
    """

    def __init__(self, config: Optional[KalmanV2Config] = None):
        self.config = config or KalmanV2Config()

        # State transition matrix (constant acceleration)
        # position' = position + velocity + 0.5 * accel
        # velocity' = velocity + accel
        # accel' = accel (random walk)
        self.F = np.array([
            [1.0, 1.0, 0.5],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0]
        ])

        # Measurement matrix (observe position only)
        self.H = np.array([[1.0, 0.0, 0.0]])

    def _estimate_r(self, prices: np.ndarray, idx: int) -> float:
        """Estimate measurement noise from recent variance."""
        window = self.config.r_window
        start = max(0, idx - window + 1)

        if start >= idx:
            return self.config.initial_p * self.config.r_scale

        recent = prices[start:idx + 1]
        recent_valid = recent[~np.isnan(recent)]

        if len(recent_valid) < 2:
            return self.config.initial_p * self.config.r_scale

        returns = np.diff(recent_valid)
        r = np.var(returns) if len(returns) > 0 else self.config.initial_p

        return max(r * self.config.r_scale, self.config.min_r)

    def _get_q_matrix(self, accel_var: float) -> np.ndarray:
        """
        Get process noise covariance matrix.

        Different scales for each state:
        - Trend: lowest noise (most stable)
        - Velocity: medium noise
        - Acceleration: highest noise (most volatile)
        """
        cfg = self.config

        # Scale by recent acceleration variance
        accel_scale = max(1.0, accel_var * 10)

        return np.diag([
            cfg.q_trend * cfg.min_q,
            cfg.q_velocity * cfg.min_q * accel_scale,
            cfg.q_accel * cfg.min_q * accel_scale
        ])

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply 3-state Kalman Filter."""

        # Prepare price series
        if self.config.use_typical_price and 'high' in df.columns and 'low' in df.columns:
            raw_prices = ((df['high'] + df['low'] + df['close']) / 3).values
        else:
            raw_prices = df['close'].values

        # Log transform
        if self.config.use_log_price:
            prices = np.where(raw_prices > 0, np.log(raw_prices), np.nan)
        else:
            prices = raw_prices.copy()

        n = len(prices)

        # Output arrays
        trend = np.full(n, np.nan)
        velocity = np.full(n, np.nan)
        acceleration = np.full(n, np.nan)
        kalman_gain = np.full(n, np.nan)
        uncertainty = np.full(n, np.nan)
        innovation = np.full(n, np.nan)
        innovation_cov = np.full(n, np.nan)

        # Find first valid price
        first_valid_idx = 0
        for i in range(n):
            if not np.isnan(prices[i]):
                first_valid_idx = i
                break

        # Initialize state
        x = np.array([prices[first_valid_idx], 0.0, 0.0])

        # Initialize covariance
        P = np.diag([self.config.initial_p] * 3)

        # History for Q estimation
        accel_history = []
        initialized = False

        for i in range(n):
            z = prices[i]

            # Handle NaN
            if np.isnan(z):
                if not initialized:
                    continue

                # Prediction only
                x = self.F @ x
                accel_var = np.var(accel_history[-20:]) if len(accel_history) >= 2 else 0.0
                Q = self._get_q_matrix(accel_var)
                P = self.F @ P @ self.F.T + Q

                trend[i] = x[0]
                velocity[i] = x[1]
                acceleration[i] = x[2]
                uncertainty[i] = P[0, 0]
                accel_history.append(x[2])
                continue

            # First valid observation
            if not initialized:
                x = np.array([z, 0.0, 0.0])
                initialized = True

            # === PREDICTION ===
            x_pred = self.F @ x

            accel_var = np.var(accel_history[-20:]) if len(accel_history) >= 2 else 0.0
            Q = self._get_q_matrix(accel_var)
            P_pred = self.F @ P @ self.F.T + Q

            # === UPDATE ===
            R_t = self._estimate_r(prices, i)

            # Innovation
            y = z - (self.H @ x_pred)[0]

            # Innovation covariance
            S = (self.H @ P_pred @ self.H.T)[0, 0] + R_t

            # Kalman gain
            K = (P_pred @ self.H.T) / S

            # Update state
            x = x_pred + K.flatten() * y

            # Update covariance
            KH = K @ self.H
            P = (np.eye(3) - KH) @ P_pred

            # Store results
            trend[i] = x[0]
            velocity[i] = x[1]
            acceleration[i] = x[2]
            kalman_gain[i] = K[0, 0]
            uncertainty[i] = P[0, 0]
            innovation[i] = y
            innovation_cov[i] = S

            accel_history.append(x[2])

        # Convert back from log space
        if self.config.use_log_price:
            trend_original = np.exp(trend)
            deviation = raw_prices - trend_original
            deviation_pct = (raw_prices - trend_original) / (trend_original + 1e-10) * 100
        else:
            trend_original = trend
            deviation = raw_prices - trend
            deviation_pct = (raw_prices - trend) / (trend + 1e-10) * 100

        # Build result
        result = df.copy()
        result['kf_trend'] = trend_original
        result['kf_velocity'] = velocity
        result['kf_acceleration'] = acceleration  # NEW
        result['kf_deviation'] = deviation
        result['kf_deviation_pct'] = deviation_pct
        result['kf_gain'] = kalman_gain
        result['kf_uncertainty'] = uncertainty
        result['kf_innovation'] = innovation
        result['kf_std_innovation'] = innovation / (np.sqrt(innovation_cov) + 1e-10)

        return result


def calculate_adaptive_kalman_v2(
    df: pd.DataFrame,
    r_window: int = 20,
    q_window: int = 20,
    use_typical_price: bool = True,
    use_log_price: bool = True,
) -> pd.DataFrame:
    """
    Apply 3-state Kalman Filter.

    Returns DataFrame with:
    - kf_trend: Filtered price level
    - kf_velocity: Rate of change
    - kf_acceleration: Rate of velocity change (NEW)
    - kf_deviation, kf_deviation_pct
    - kf_gain, kf_uncertainty
    - kf_innovation, kf_std_innovation
    """
    config = KalmanV2Config(
        r_window=r_window,
        q_window=q_window,
        use_typical_price=use_typical_price,
        use_log_price=use_log_price,
    )
    kf = AdaptiveKalmanFilterV2(config)
    return kf.filter(df)
