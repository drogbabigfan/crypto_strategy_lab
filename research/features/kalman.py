"""
Adaptive Kalman Filter for Financial Time Series.

Implements a state-space model with adaptive noise estimation:
- State equation: x_t = F * x_{t-1} + w_t (process noise)
- Measurement equation: z_t = H * x_t + v_t (measurement noise)

Key Features:
- Adaptive R_t: Measurement noise estimated from rolling volatility
- Adaptive Q_t: Process noise estimated from momentum changes
- Error covariance P for position sizing (uncertainty measure)
- Typical Price option: (High + Low + Close) / 3

References:
- Alpha Architect: Noise-Adaptive Kalman Filter
- Benhamou: Kalman Filter in Finance
- Marcos Lopez de Prado: Advances in Financial Machine Learning
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass


@dataclass
class KalmanConfig:
    """Configuration for Adaptive Kalman Filter."""

    # Noise estimation windows
    r_window: int = 20          # Window for measurement noise (R) estimation
    q_window: int = 20          # Window for process noise (Q) estimation

    # Initial state covariance
    initial_p: float = 1.0      # Initial error covariance

    # Noise scaling factors
    r_scale: float = 1.0        # Scale factor for R estimation
    q_scale: float = 0.4        # Scale factor for Q (optimized from 0.1)

    # Adaptive Q momentum sensitivity
    q_momentum_weight: float = 2.0  # Weight for momentum-based Q adjustment

    # === NIS-based Q Adjustment ===
    # Boosts Q when predictions are wrong (NIS > 1)
    # Q_new = Q_base * min(max_cap, NIS * alpha)
    use_nis_adaptive: bool = True   # Enable NIS-based adaptive Q (recommended with PWNA)
    nis_window: int = 10            # Window for NIS smoothing
    nis_alpha: float = 3.0          # Sensitivity multiplier (crypto: 2.0~5.0)
    nis_boost_factor: float = 20.0  # Max boost cap (crypto: 10~50)

    # === PWNA Model (Piecewise White Noise Acceleration) ===
    # Q = σ_a² * [[dt⁴/4, dt³/2], [dt³/2, dt²]]
    # 단일 파라미터 sigma_a²로 물리적으로 정확한 Q 행렬 생성
    # 위치-속도 오차를 물리적으로 커플링 (off-diagonal ≠ 0)
    use_pwna_q: bool = True         # PWNA 모델 사용 (권장)
    sigma_a_scale: float = 0.4      # 가속도 노이즈 스케일 (최적화됨)

    # === Directional Innovation Mapping ===
    # σ_a² = σ_base² * exp(-η * std_innovation)
    # - 하락 (innov < 0): exp(-negative) > 1 → Q 증폭 (빠른 반응)
    # - 상승 (innov > 0): exp(-positive) < 1 → Q 축소 (부드러운 추종)
    # "Fear is faster than greed" 비대칭성을 모델에 내장
    use_directional_q: bool = False     # Directional Q 사용 (실험적)
    directional_eta: float = 0.5        # 민감도 파라미터 (0.3~1.0 권장)
    directional_max_boost: float = 5.0  # 최대 부스트 캡
    directional_min_boost: float = 0.2  # 최소 부스트 (상승 시)

    # Price type
    use_typical_price: bool = True  # Use (H+L+C)/3 instead of Close
    use_log_price: bool = True      # Apply log transform for scale invariance

    # Minimum noise floors (prevent division by zero)
    min_r: float = 1e-8
    min_q: float = 1e-10


class AdaptiveKalmanFilter:
    """
    Adaptive Kalman Filter with dynamic noise estimation.

    Uses a 2-state model:
    - State 1: Trend (filtered price level)
    - State 2: Velocity (rate of change)

    The filter adapts to market conditions by:
    - Increasing R (smoothing) when volatility is high
    - Increasing Q (tracking) when momentum is strong
    """

    def __init__(self, config: Optional[KalmanConfig] = None):
        """
        Initialize Adaptive Kalman Filter.

        Args:
            config: KalmanConfig with filter parameters
        """
        self.config = config or KalmanConfig()

        # State transition matrix F (constant velocity model)
        # [trend]     [1  1] [trend]
        # [velocity] = [0  1] [velocity]
        self.F = np.array([[1.0, 1.0],
                          [0.0, 1.0]])

        # Measurement matrix H (we observe trend only)
        # z = [1  0] * [trend, velocity]^T
        self.H = np.array([[1.0, 0.0]])

    def _estimate_r_close(self, prices: np.ndarray, idx: int) -> float:
        """
        Estimate measurement noise R_trend from close-to-close variance.
        Used for state update (smooth trend tracking).

        Args:
            prices: Close price array (log space)
            idx: Current index

        Returns:
            Estimated R_trend
        """
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

    def _estimate_r_parkinson(
        self,
        log_high: np.ndarray,
        log_low: np.ndarray,
        idx: int
    ) -> float:
        """
        Estimate measurement noise R_risk from Parkinson volatility.
        Used for stop distance calculation (accounts for intrabar wicks).

        Parkinson Volatility: σ² = (1/4ln2) × mean[(ln(H) - ln(L))²]

        Args:
            log_high: Log of high prices
            log_low: Log of low prices
            idx: Current index

        Returns:
            Estimated R_risk (Parkinson-based)
        """
        window = self.config.r_window
        start = max(0, idx - window + 1)

        if start >= idx:
            return self.config.initial_p * self.config.r_scale

        # High-Low range in log space
        log_range = log_high[start:idx + 1] - log_low[start:idx + 1]
        valid_range = log_range[~np.isnan(log_range)]

        if len(valid_range) < 2:
            return self.config.initial_p * self.config.r_scale

        # Parkinson variance: (1/4ln2) × mean(range²)
        parkinson_factor = 1.0 / (4.0 * np.log(2.0))
        r = parkinson_factor * np.mean(valid_range ** 2)

        return max(r * self.config.r_scale, self.config.min_r)

    def _estimate_q(
        self,
        prices: np.ndarray,
        velocities: np.ndarray,
        idx: int
    ) -> float:
        """
        Estimate process noise Q_t from momentum changes.

        Higher Q when:
        - Velocity (momentum) is changing rapidly
        - Strong directional moves detected

        Args:
            prices: Price array
            velocities: Estimated velocity history
            idx: Current index

        Returns:
            Estimated Q_t
        """
        window = self.config.q_window
        start = max(0, idx - window + 1)

        if start >= idx or len(velocities) < 2:
            return self.config.min_q

        # Base Q from velocity variance
        recent_vel = velocities[start:idx + 1] if idx < len(velocities) else velocities[start:]

        # Filter out NaN values
        recent_vel_valid = recent_vel[~np.isnan(recent_vel)]

        if len(recent_vel_valid) < 2:
            return self.config.min_q

        vel_var = np.var(recent_vel_valid)

        # Momentum boost: increase Q when velocity is accelerating
        if len(recent_vel_valid) >= 3:
            vel_diff = np.diff(recent_vel_valid[-3:])
            accel = np.abs(vel_diff).mean() if len(vel_diff) > 0 else 0
            momentum_factor = 1.0 + self.config.q_momentum_weight * accel / (np.abs(recent_vel_valid).mean() + 1e-10)
        else:
            momentum_factor = 1.0

        q = vel_var * self.config.q_scale * momentum_factor

        return max(q, self.config.min_q)

    def _compute_nis(self, innovation: float, innovation_cov: float) -> float:
        """
        Compute Normalized Innovation Squared (NIS).

        NIS = innovation² / S

        When NIS > 1: prediction is significantly wrong, need to boost Q
        When NIS ≈ 1: filter is well-tuned
        When NIS < 1: filter is over-reactive

        Args:
            innovation: y = z - H*x_pred
            innovation_cov: S = H*P_pred*H^T + R

        Returns:
            NIS value (α)
        """
        if innovation_cov <= 0:
            return 1.0
        return (innovation ** 2) / innovation_cov

    def _get_nis_boost(self, nis_history: list) -> float:
        """
        Get Q boost factor from CURRENT (instant) NIS.

        Uses instant NIS for immediate reaction to prediction errors.
        Q_new = Q_base * min(max_cap, current_nis * alpha)

        Args:
            nis_history: Recent NIS values (uses last value only)

        Returns:
            Boost factor (>= 1.0)
        """
        if len(nis_history) < 1:
            return 1.0

        # Use CURRENT NIS, not mean - for instant reaction
        current_nis = nis_history[-1]

        # Instant boost: current_nis * alpha, capped at max_cap
        # NIS < 1: normal state, boost stays low
        # NIS > 1: prediction error, boost = NIS * alpha immediately
        raw_boost = current_nis * self.config.nis_alpha
        boost = max(1.0, min(raw_boost, self.config.nis_boost_factor))

        return boost

    def _get_directional_boost(self, std_innovation: float) -> float:
        """
        Get directional Q boost from standardized innovation.

        Uses exponential mapping to encode direction:
        σ_a² = σ_base² * exp(-η * std_innovation)

        - 가격 급락 (std_innov < 0): exp(-negative) > 1 → Q 증폭
        - 가격 상승 (std_innov > 0): exp(-positive) < 1 → Q 축소

        This encodes "Fear is faster than greed" asymmetry.

        Args:
            std_innovation: Standardized innovation (ε / √S)

        Returns:
            Directional boost factor
        """
        if np.isnan(std_innovation):
            return 1.0

        cfg = self.config
        eta = cfg.directional_eta

        # exp(-η * std_innovation)
        # std_innov < 0 (가격 하락) → exp(positive) > 1
        # std_innov > 0 (가격 상승) → exp(negative) < 1
        raw_boost = np.exp(-eta * std_innovation)

        # Clamp to [min, max]
        boost = np.clip(raw_boost, cfg.directional_min_boost, cfg.directional_max_boost)

        return boost

    def _get_q_matrix(
        self,
        base_q: float,
        nis_boost: float = 1.0
    ) -> np.ndarray:
        """
        Get Q matrix using PWNA (Piecewise White Noise Acceleration) model.

        PWNA Model (Δt = 1):
        Q = σ_a² * [[Δt⁴/4, Δt³/2],
                    [Δt³/2, Δt²  ]]
          = σ_a² * [[0.25, 0.5],
                    [0.5,  1.0]]

        This properly couples position and velocity errors:
        - 속도 오차가 커지면 위치 오차도 자동으로 커짐
        - 물리적으로 정확한 커플링
        - 튜닝 파라미터가 σ_a² 하나로 단순화

        Args:
            base_q: Base Q value (σ_a² estimate from _estimate_q)
            nis_boost: Boost factor from NIS (>= 1.0)

        Returns:
            2x2 PWNA Q matrix
        """
        cfg = self.config

        # σ_a² = base_q * scale * NIS boost
        sigma_a_sq = base_q * cfg.sigma_a_scale * nis_boost

        if cfg.use_pwna_q:
            # PWNA Q matrix (물리적으로 정확한 커플링)
            # Q = σ_a² * [[0.25, 0.5], [0.5, 1.0]]
            dt = 1.0  # 1 bar
            q_matrix = sigma_a_sq * np.array([
                [dt**4 / 4, dt**3 / 2],
                [dt**3 / 2, dt**2]
            ])
        else:
            # Fallback: 기존 diagonal Q (비권장)
            q_matrix = np.array([
                [sigma_a_sq, 0.0],
                [0.0, sigma_a_sq]
            ])

        # Ensure minimum values
        q_matrix = np.maximum(q_matrix, cfg.min_q)

        return q_matrix

    def _get_f_matrix(self, dt: float) -> np.ndarray:
        """
        Get state transition matrix F for given time delta.

        F = [[1, dt],
             [0, 1 ]]

        Args:
            dt: Time delta (normalized, e.g., hours)

        Returns:
            2x2 F matrix
        """
        return np.array([[1.0, dt],
                        [0.0, 1.0]])

    def _get_q_matrix_with_dt(
        self,
        base_q: float,
        dt: float,
        nis_boost: float = 1.0
    ) -> np.ndarray:
        """
        Get Q matrix using PWNA model with actual time delta.

        PWNA Model:
        Q = σ_a² * [[Δt⁴/4, Δt³/2],
                    [Δt³/2, Δt²  ]]

        Args:
            base_q: Base Q value (σ_a² estimate)
            dt: Time delta (normalized)
            nis_boost: Boost factor from NIS (>= 1.0)

        Returns:
            2x2 PWNA Q matrix
        """
        cfg = self.config
        sigma_a_sq = base_q * cfg.sigma_a_scale * nis_boost

        if cfg.use_pwna_q:
            q_matrix = sigma_a_sq * np.array([
                [dt**4 / 4, dt**3 / 2],
                [dt**3 / 2, dt**2]
            ])
        else:
            q_matrix = np.array([
                [sigma_a_sq * dt**2, 0.0],
                [0.0, sigma_a_sq]
            ])

        return np.maximum(q_matrix, cfg.min_q)

    def filter(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply Adaptive Kalman Filter to price data.

        Args:
            df: DataFrame with 'close' (and optionally 'high', 'low', 'log_duration')

        Returns:
            DataFrame with Kalman filter outputs:
            - kf_trend: Filtered trend (a posteriori state estimate)
            - kf_trend_pred: Predicted trend (a priori state estimate)
            - kf_velocity: Estimated velocity/momentum (per hour)
            - kf_deviation: Price deviation from trend (z - trend)
            - kf_deviation_pct: Deviation as percentage
            - kf_gain: Kalman gain (filter responsiveness)
            - kf_uncertainty: Error covariance P (for position sizing)
            - kf_signal: Normalized deviation z-score
        """
        # Prepare price series
        if self.config.use_typical_price and 'high' in df.columns and 'low' in df.columns:
            raw_prices = ((df['high'] + df['low'] + df['close']) / 3).values
        else:
            raw_prices = df['close'].values

        # Prepare high/low for Parkinson R (Risk Eye)
        has_hl = 'high' in df.columns and 'low' in df.columns
        if has_hl:
            raw_high = df['high'].values
            raw_low = df['low'].values
        else:
            raw_high = raw_prices.copy()
            raw_low = raw_prices.copy()

        # Apply log transform for scale invariance
        if self.config.use_log_price:
            prices = np.where(raw_prices > 0, np.log(raw_prices), np.nan)
            log_high = np.where(raw_high > 0, np.log(raw_high), np.nan)
            log_low = np.where(raw_low > 0, np.log(raw_low), np.nan)
        else:
            prices = raw_prices.copy()
            log_high = raw_high.copy()
            log_low = raw_low.copy()

        n = len(prices)

        # Get time delta (in hours) for each bar
        # Default: 6 hours (typical TIB bar)
        if 'log_duration' in df.columns:
            duration_seconds = np.exp(df['log_duration'].values)
            dt_hours = duration_seconds / 3600.0  # Convert to hours
        else:
            dt_hours = np.full(n, 6.0)  # Default 6 hours

        # Output arrays (in log space if use_log_price)
        trend = np.full(n, np.nan)
        trend_pred = np.full(n, np.nan)  # a priori state estimate
        velocity = np.full(n, np.nan)
        velocity_var = np.full(n, np.nan)  # P[1,1] - velocity variance for Z-score
        kalman_gain = np.full(n, np.nan)
        uncertainty = np.full(n, np.nan)  # P (a posteriori)
        uncertainty_pred = np.full(n, np.nan)  # P_pred (a priori) - for stop calculation
        innovation = np.full(n, np.nan)  # y = z - H*x_pred
        innovation_cov = np.full(n, np.nan)  # S_trend = H*P_pred*H^T + R_close (for update)
        innovation_cov_risk = np.full(n, np.nan)  # S_risk = H*P_pred*H^T + R_parkinson (for stop)
        nis_arr = np.full(n, np.nan)  # Normalized Innovation Squared
        nis_boost_arr = np.full(n, np.nan)  # Applied NIS boost factor
        r_trend_arr = np.full(n, np.nan)  # R_trend (close-based measurement noise)
        r_risk_arr = np.full(n, np.nan)  # R_risk (Parkinson-based measurement noise)

        # Find first valid (non-NaN) price for initialization
        first_valid_idx = 0
        for i in range(n):
            if not np.isnan(prices[i]):
                first_valid_idx = i
                break

        # Initialize state: [trend, velocity]
        x = np.array([prices[first_valid_idx], 0.0])

        # Initialize error covariance P
        P = np.array([[self.config.initial_p, 0.0],
                      [0.0, self.config.initial_p]])

        # Store velocity history for Q estimation
        vel_history = []
        # NIS history for adaptive Q
        nis_history = []
        # Std innovation history for directional Q
        std_innov_history = []
        initialized = False

        for i in range(n):
            z = prices[i]  # Observation

            # === HANDLE NaN: Skip update, only predict ===
            # Get time delta for this bar
            dt = dt_hours[i]

            if np.isnan(z):
                if not initialized:
                    # Not yet initialized, skip entirely
                    continue

                # Prediction step only (no observation)
                F_t = self._get_f_matrix(dt)
                x_pred = F_t @ x
                Q_t = self._estimate_q(prices, np.array(vel_history), i)
                # Use NIS boost even for prediction-only steps
                nis_boost = self._get_nis_boost(nis_history) if self.config.use_nis_adaptive else 1.0
                Q_matrix = self._get_q_matrix_with_dt(Q_t, dt, nis_boost)
                P_pred = F_t @ P @ F_t.T + Q_matrix

                # Use prediction as state (no update)
                x = x_pred
                P = P_pred

                # Store results (prediction only)
                trend_pred[i] = x_pred[0]
                trend[i] = x[0]
                velocity[i] = x[1]
                velocity_var[i] = P[1, 1]  # Velocity variance for Z-score
                kalman_gain[i] = 0.0  # No update happened
                uncertainty[i] = P[0, 0]
                uncertainty_pred[i] = P_pred[0, 0]
                vel_history.append(x[1])
                continue

            # First valid observation - initialize
            if not initialized:
                x = np.array([z, 0.0])
                initialized = True

            # === PREDICTION STEP ===
            # Get F matrix for current time delta
            F_t = self._get_f_matrix(dt)

            # x_pred = F * x
            x_pred = F_t @ x

            # Estimate adaptive Q
            Q_t = self._estimate_q(prices, np.array(vel_history), i)

            # Get Q boost factor (NIS-based or Directional)
            if self.config.use_directional_q and len(std_innov_history) > 0:
                # Directional boost: exp(-η * std_innovation)
                # Uses previous bar's std_innovation
                prev_std_innov = std_innov_history[-1]
                directional_boost = self._get_directional_boost(prev_std_innov)
                # Combine with NIS boost if enabled
                if self.config.use_nis_adaptive:
                    nis_boost = self._get_nis_boost(nis_history)
                    # Use max of both (aggressive) or multiply (very aggressive)
                    q_boost = max(nis_boost, directional_boost)
                else:
                    q_boost = directional_boost
                nis_boost = q_boost  # Store for output
            else:
                # Original NIS-only boost
                nis_boost = self._get_nis_boost(nis_history) if self.config.use_nis_adaptive else 1.0

            # Use PWNA Q matrix with actual time delta (with optional boost)
            Q_matrix = self._get_q_matrix_with_dt(Q_t, dt, nis_boost)

            # P_pred = F * P * F^T + Q
            P_pred = F_t @ P @ F_t.T + Q_matrix

            # === UPDATE STEP (Dual-Eye System) ===
            # Eye 1: Trend Eye (Close-based R) - for smooth trend tracking
            R_trend = self._estimate_r_close(prices, i)

            # Eye 2: Risk Eye (Parkinson-based R) - for stop distance
            R_risk = self._estimate_r_parkinson(log_high, log_low, i)

            # Innovation (measurement residual)
            y = z - (self.H @ x_pred)[0]

            # S_trend: for Kalman update (smooth tracking)
            S_trend = (self.H @ P_pred @ self.H.T)[0, 0] + R_trend

            # S_risk: for stop distance (accounts for intrabar wicks)
            S_risk = (self.H @ P_pred @ self.H.T)[0, 0] + R_risk

            # Compute and store NIS for adaptive Q adjustment (using trend S)
            nis = self._compute_nis(y, S_trend)
            nis_history.append(nis)
            nis_arr[i] = nis
            nis_boost_arr[i] = nis_boost
            r_trend_arr[i] = R_trend
            r_risk_arr[i] = R_risk

            # Store std_innovation for directional Q (next bar)
            std_innov_current = y / (np.sqrt(S_trend) + 1e-10)
            std_innov_history.append(std_innov_current)

            # Kalman Gain: K = P_pred * H^T * S_trend^(-1)
            K = (P_pred @ self.H.T) / S_trend

            # Update state: x = x_pred + K * y
            x = x_pred + K.flatten() * y

            # Update error covariance: P = (I - K * H) * P_pred
            KH = K @ self.H
            P = (np.eye(2) - KH) @ P_pred

            # Store results (in log space)
            trend_pred[i] = x_pred[0]  # a priori estimate (prediction)
            trend[i] = x[0]
            velocity[i] = x[1]
            velocity_var[i] = P[1, 1]  # Velocity variance for Z-score
            kalman_gain[i] = K[0, 0]
            uncertainty[i] = P[0, 0]  # P (a posteriori)
            uncertainty_pred[i] = P_pred[0, 0]  # P_pred (a priori)
            innovation[i] = y  # Measurement residual
            innovation_cov[i] = S_trend  # S_trend = P_pred + R_close (for update)
            innovation_cov_risk[i] = S_risk  # S_risk = P_pred + R_parkinson (for stop)

            vel_history.append(x[1])

        # Convert back from log space if needed
        if self.config.use_log_price:
            trend_original = np.exp(trend)
            trend_pred_original = np.exp(trend_pred)
            # Deviation in original price space
            deviation = raw_prices - trend_original
            deviation_pct = (raw_prices - trend_original) / (trend_original + 1e-10) * 100
        else:
            trend_original = trend
            trend_pred_original = trend_pred
            deviation = raw_prices - trend
            deviation_pct = (raw_prices - trend) / (trend + 1e-10) * 100

        # Build result DataFrame
        result = df.copy()
        result['kf_trend'] = trend_original
        result['kf_trend_pred'] = trend_pred_original
        result['kf_velocity'] = velocity  # In log space (log-return per hour, time-normalized)
        result['kf_velocity_var'] = velocity_var  # P[1,1] - velocity variance
        result['kf_deviation'] = deviation
        result['kf_deviation_pct'] = deviation_pct
        result['kf_gain'] = kalman_gain
        result['kf_uncertainty'] = uncertainty  # P (a posteriori)
        result['kf_uncertainty_pred'] = uncertainty_pred  # P_pred (a priori)
        result['kf_innovation_cov'] = innovation_cov  # S_trend = P_pred + R_close (for update)
        result['kf_innovation_cov_risk'] = innovation_cov_risk  # S_risk = P_pred + R_parkinson (for stop!)
        result['kf_innovation'] = innovation  # Measurement residual (log space)
        # NIS and NIS boost for short signal triggers
        result['kf_nis'] = nis_arr  # Normalized Innovation Squared
        result['kf_nis_boost'] = nis_boost_arr  # Applied NIS boost factor
        # Dual-Eye R values for asymmetry detection
        result['kf_r_trend'] = r_trend_arr  # R_trend (close-based)
        result['kf_r_risk'] = r_risk_arr  # R_risk (Parkinson-based)
        # Standardized Innovation: innovation / sqrt(S_trend)
        result['kf_std_innovation'] = innovation / (np.sqrt(innovation_cov) + 1e-10)
        # Stop Distance (k=3): 3 * sqrt(S_risk) - ready to use!
        result['kf_stop_distance'] = 3.0 * np.sqrt(innovation_cov_risk)
        # Velocity Z-Score: velocity / sqrt(P[1,1]) - for statistically significant trend
        result['kf_velocity_zscore'] = velocity / (np.sqrt(velocity_var) + 1e-10)

        # Normalized signal: deviation z-score
        signal_window = self.config.r_window
        min_periods = min(5, signal_window)
        dev_std = pd.Series(deviation).rolling(
            window=signal_window,
            min_periods=min_periods
        ).std()
        result['kf_signal'] = deviation / (dev_std.values + 1e-10)

        return result

    def get_position_size_factor(
        self,
        uncertainty: float,
        min_uncertainty: float = None,
        max_uncertainty: float = None
    ) -> float:
        """
        Calculate position size factor based on uncertainty.

        Position size is inversely proportional to sqrt(P).
        When uncertainty is high, reduce position size.

        Args:
            uncertainty: Current error covariance P[0,0]
            min_uncertainty: Minimum uncertainty for scaling (default: initial_p * 0.1)
            max_uncertainty: Maximum uncertainty for scaling (default: initial_p * 100)

        Returns:
            Position size factor (0-1)
        """
        if min_uncertainty is None:
            min_uncertainty = self.config.initial_p * 0.1
        if max_uncertainty is None:
            max_uncertainty = self.config.initial_p * 100

        if uncertainty <= 0:
            return 1.0

        # Clamp uncertainty to valid range
        uncertainty = np.clip(uncertainty, min_uncertainty, max_uncertainty)

        # Log-scale normalization for better distribution
        log_min = np.log(min_uncertainty)
        log_max = np.log(max_uncertainty)
        log_unc = np.log(uncertainty)

        # Invert: low uncertainty -> high size, high uncertainty -> low size
        normalized = 1.0 - (log_unc - log_min) / (log_max - log_min + 1e-10)

        return np.clip(normalized, 0.0, 1.0)


class KalmanFeatureGenerator:
    """
    Generate Kalman Filter features for integration with feature pipeline.
    """

    def __init__(
        self,
        r_window: int = 20,
        q_window: int = 20,
        use_typical_price: bool = True,
        use_log_price: bool = True,
        q_scale: float = 0.4,           # 최적화됨 (기존 0.1)
        r_scale: float = 1.0,
        use_nis_adaptive: bool = True,  # NIS 기반 적응형 Q (권장)
        use_pwna_q: bool = True,        # PWNA Q 행렬 (권장)
        sigma_a_scale: float = 0.4,     # 가속도 노이즈 스케일
        # Directional Q parameters
        use_directional_q: bool = False,  # Directional Q (실험적)
        directional_eta: float = 0.5,     # 민감도 (0.3~1.0)
    ):
        """
        Args:
            r_window: Window for R estimation
            q_window: Window for Q estimation
            use_typical_price: Use (H+L+C)/3 instead of Close
            use_log_price: Apply log transform for scale invariance
            q_scale: Process noise scale factor (optimized: 0.4)
            r_scale: Measurement noise scale factor
            use_nis_adaptive: Use NIS-based adaptive Q (recommended)
            use_pwna_q: Use PWNA Q matrix (proper pos-vel coupling)
            sigma_a_scale: Acceleration noise scale for PWNA
            use_directional_q: Use directional innovation mapping (experimental)
            directional_eta: Sensitivity for directional Q (0.3~1.0)
        """
        self.config = KalmanConfig(
            r_window=r_window,
            q_window=q_window,
            use_typical_price=use_typical_price,
            use_log_price=use_log_price,
            q_scale=q_scale,
            r_scale=r_scale,
            use_nis_adaptive=use_nis_adaptive,
            use_pwna_q=use_pwna_q,
            sigma_a_scale=sigma_a_scale,
            use_directional_q=use_directional_q,
            directional_eta=directional_eta,
        )
        self.kf = AdaptiveKalmanFilter(self.config)

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Kalman filter features.

        Args:
            df: DataFrame with 'close' (and optionally 'high', 'low')

        Returns:
            DataFrame with Kalman features added
        """
        return self.kf.filter(df)

    def get_feature_names(self) -> list:
        """Return list of generated feature names."""
        return [
            'kf_trend',
            'kf_trend_pred',
            'kf_velocity',
            'kf_velocity_var',          # P[1,1] - velocity variance
            'kf_velocity_zscore',       # velocity / sqrt(P[1,1]) - for significant trend detection
            'kf_deviation',
            'kf_deviation_pct',
            'kf_gain',
            'kf_uncertainty',           # P (a posteriori)
            'kf_uncertainty_pred',      # P_pred (a priori)
            'kf_innovation_cov',        # S_trend = P_pred + R_close
            'kf_innovation_cov_risk',   # S_risk = P_pred + R_parkinson (for stop!)
            'kf_innovation',            # Measurement residual (log space)
            'kf_nis',                   # Normalized Innovation Squared
            'kf_nis_boost',             # Applied NIS boost factor
            'kf_r_trend',               # R_trend (close-based measurement noise)
            'kf_r_risk',                # R_risk (Parkinson-based measurement noise)
            'kf_stop_distance',         # 3 * sqrt(S_risk) - ready to use!
            'kf_signal',
            'kf_std_innovation',
        ]

    def get_core_features(self) -> list:
        """Return core features for model input."""
        return [
            'kf_deviation',      # Price deviation from trend
            'kf_velocity',       # Trend momentum
            'kf_signal',         # Normalized deviation z-score
            'kf_uncertainty',    # For position sizing
        ]


def calculate_adaptive_kalman(
    df: pd.DataFrame,
    r_window: int = 20,
    q_window: int = 20,
    use_typical_price: bool = True,
    use_log_price: bool = True,
) -> pd.DataFrame:
    """
    Convenience function for Adaptive Kalman Filter.

    Args:
        df: DataFrame with OHLC data
        r_window: Window for measurement noise estimation
        q_window: Window for process noise estimation
        use_typical_price: Use typical price (H+L+C)/3
        use_log_price: Apply log transform for scale invariance

    Returns:
        DataFrame with Kalman features
    """
    generator = KalmanFeatureGenerator(
        r_window=r_window,
        q_window=q_window,
        use_typical_price=use_typical_price,
        use_log_price=use_log_price,
    )
    return generator.generate(df)


def calculate_stochastic_oscillator(
    df: pd.DataFrame,
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator (%K, %D).

    Used in Benhamou's Model 4 to capture overbought/oversold conditions.

    Args:
        df: DataFrame with 'high', 'low', 'close'
        period: Lookback period for high/low (default 14)
        smooth_k: Smoothing for %K (default 3)
        smooth_d: Smoothing for %D (default 3)

    Returns:
        DataFrame with oscillator columns added:
        - stoch_k: Fast %K (raw oscillator)
        - stoch_k_smooth: Smoothed %K
        - stoch_d: %D (signal line)
    """
    result = df.copy()

    # Rolling high/low
    high_n = df['high'].rolling(window=period, min_periods=1).max()
    low_n = df['low'].rolling(window=period, min_periods=1).min()

    # Raw %K: (Close - Low_n) / (High_n - Low_n)
    range_hl = high_n - low_n
    range_hl = range_hl.replace(0, np.nan)  # Avoid division by zero

    stoch_k = (df['close'] - low_n) / range_hl * 100
    stoch_k = stoch_k.fillna(50)  # Neutral when no range

    # Smoothed %K
    stoch_k_smooth = stoch_k.rolling(window=smooth_k, min_periods=1).mean()

    # %D (signal line)
    stoch_d = stoch_k_smooth.rolling(window=smooth_d, min_periods=1).mean()

    result['stoch_k'] = stoch_k.values
    result['stoch_k_smooth'] = stoch_k_smooth.values
    result['stoch_d'] = stoch_d.values

    return result


def calculate_kalman_model4(
    df: pd.DataFrame,
    r_window: int = 20,
    q_window: int = 20,
    osc_period: int = 14,
    use_typical_price: bool = True,
) -> pd.DataFrame:
    """
    Benhamou's Model 4: Kalman Filter + Stochastic Oscillator.

    Combines:
    1. Kalman Filter: Trend tracking with prediction
    2. Stochastic Oscillator: Overbought/Oversold detection

    The oscillator acts as a filter to prevent entries at extreme levels.

    Args:
        df: DataFrame with OHLC data
        r_window: Window for Kalman R estimation
        q_window: Window for Kalman Q estimation
        osc_period: Period for Stochastic oscillator (default 14)
        use_typical_price: Use typical price for Kalman

    Returns:
        DataFrame with Kalman + Oscillator features:
        - All standard Kalman features (kf_*)
        - stoch_k, stoch_k_smooth, stoch_d
        - osc_overbought: True if stoch_k > 80
        - osc_oversold: True if stoch_k < 20
        - osc_neutral: True if 20 <= stoch_k <= 80
    """
    # Step 1: Apply Kalman Filter
    result = calculate_adaptive_kalman(
        df,
        r_window=r_window,
        q_window=q_window,
        use_typical_price=use_typical_price,
    )

    # Step 2: Add Stochastic Oscillator
    result = calculate_stochastic_oscillator(
        result,
        period=osc_period,
    )

    # Step 3: Zone classification
    result['osc_overbought'] = result['stoch_k_smooth'] > 80
    result['osc_oversold'] = result['stoch_k_smooth'] < 20
    result['osc_neutral'] = (result['stoch_k_smooth'] >= 20) & (result['stoch_k_smooth'] <= 80)

    return result
