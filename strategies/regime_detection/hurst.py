"""
Hurst Exponent Calculator.

Measures long-term memory and persistence in time series:
- H > 0.5: Persistent (trending) - use momentum strategies
- H = 0.5: Random walk - prediction difficult
- H < 0.5: Anti-persistent (mean-reverting) - use mean reversion strategies

Methods implemented:
- R/S Analysis (Rescaled Range)
- DFA (Detrended Fluctuation Analysis)
- Rolling Hurst for real-time regime detection
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class HurstRegime(Enum):
    """Market regime based on Hurst exponent."""
    TRENDING = "trending"           # H > 0.55
    RANDOM_WALK = "random_walk"     # 0.45 <= H <= 0.55
    MEAN_REVERTING = "mean_reverting"  # H < 0.45


@dataclass
class HurstResult:
    """Result of Hurst exponent calculation."""
    hurst: float
    regime: HurstRegime
    confidence: float  # R-squared of the fit
    method: str

    @property
    def is_trending(self) -> bool:
        return self.regime == HurstRegime.TRENDING

    @property
    def is_mean_reverting(self) -> bool:
        return self.regime == HurstRegime.MEAN_REVERTING


class HurstCalculator:
    """
    Calculate Hurst exponent using multiple methods.

    The Hurst exponent H measures the long-term memory of a time series:
    - H > 0.5: Persistent behavior (trend-following)
    - H = 0.5: Random walk (Brownian motion)
    - H < 0.5: Anti-persistent behavior (mean-reverting)
    """

    def __init__(
        self,
        min_window: int = 10,
        max_window: Optional[int] = None,
        num_scales: int = 20,
        trending_threshold: float = 0.55,
        mean_revert_threshold: float = 0.45,
    ):
        """
        Args:
            min_window: Minimum window size for scaling analysis
            max_window: Maximum window size (defaults to len(data)//4)
            num_scales: Number of scales for DFA/R-S analysis
            trending_threshold: H above this is considered trending
            mean_revert_threshold: H below this is considered mean-reverting
        """
        self.min_window = min_window
        self.max_window = max_window
        self.num_scales = num_scales
        self.trending_threshold = trending_threshold
        self.mean_revert_threshold = mean_revert_threshold

    def calculate_rs(self, data: np.ndarray) -> HurstResult:
        """
        Calculate Hurst exponent using R/S (Rescaled Range) analysis.

        Classical method by H.E. Hurst. Computes the ratio of range to
        standard deviation across different time scales.

        Args:
            data: Time series data (prices or returns)

        Returns:
            HurstResult with Hurst exponent and regime classification
        """
        data = np.asarray(data, dtype=np.float64)
        n = len(data)

        if n < self.min_window * 2:
            return HurstResult(
                hurst=0.5,
                regime=HurstRegime.RANDOM_WALK,
                confidence=0.0,
                method='rs'
            )

        max_k = self.max_window or n // 4

        # Generate scale sizes
        scales = np.unique(np.geomspace(
            self.min_window,
            max_k,
            num=self.num_scales
        ).astype(int))

        rs_values = []
        valid_scales = []

        for scale in scales:
            if scale < 2:
                continue

            # Number of segments
            num_segments = n // scale
            if num_segments < 1:
                continue

            rs_scale = []

            for i in range(num_segments):
                segment = data[i * scale:(i + 1) * scale]

                # Mean-adjusted cumulative deviation
                mean_val = np.mean(segment)
                cumsum = np.cumsum(segment - mean_val)

                # Range
                r = np.max(cumsum) - np.min(cumsum)

                # Standard deviation
                s = np.std(segment, ddof=1)

                if s > 1e-10:
                    rs_scale.append(r / s)

            if rs_scale:
                rs_values.append(np.mean(rs_scale))
                valid_scales.append(scale)

        if len(valid_scales) < 3:
            return HurstResult(
                hurst=0.5,
                regime=HurstRegime.RANDOM_WALK,
                confidence=0.0,
                method='rs'
            )

        # Log-log regression: log(R/S) = H * log(n) + c
        log_scales = np.log(valid_scales)
        log_rs = np.log(rs_values)

        hurst, confidence = self._fit_hurst(log_scales, log_rs)
        regime = self._classify_regime(hurst)

        return HurstResult(
            hurst=hurst,
            regime=regime,
            confidence=confidence,
            method='rs'
        )

    def calculate_dfa(
        self,
        data: np.ndarray,
        order: int = 1
    ) -> HurstResult:
        """
        Calculate Hurst exponent using DFA (Detrended Fluctuation Analysis).

        More robust than R/S for non-stationary time series. Removes local
        polynomial trends before computing fluctuations.

        Args:
            data: Time series data (prices or returns)
            order: Order of polynomial detrending (1=linear, 2=quadratic)

        Returns:
            HurstResult with Hurst exponent and regime classification
        """
        data = np.asarray(data, dtype=np.float64)
        n = len(data)

        if n < self.min_window * 2:
            return HurstResult(
                hurst=0.5,
                regime=HurstRegime.RANDOM_WALK,
                confidence=0.0,
                method='dfa'
            )

        # Step 1: Integrate the series (cumulative sum of deviations from mean)
        mean_val = np.mean(data)
        profile = np.cumsum(data - mean_val)

        max_k = self.max_window or n // 4

        # Generate scale sizes
        scales = np.unique(np.geomspace(
            self.min_window,
            max_k,
            num=self.num_scales
        ).astype(int))

        fluctuations = []
        valid_scales = []

        for scale in scales:
            if scale < order + 2:
                continue

            # Number of segments
            num_segments = n // scale
            if num_segments < 1:
                continue

            f2 = []

            for i in range(num_segments):
                segment = profile[i * scale:(i + 1) * scale]

                # Fit polynomial and detrend
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, order)
                trend = np.polyval(coeffs, x)

                # Compute fluctuation (RMS)
                detrended = segment - trend
                f2.append(np.mean(detrended ** 2))

            # Also process from the end for better coverage
            for i in range(num_segments):
                start = n - (i + 1) * scale
                if start < 0:
                    break
                segment = profile[start:start + scale]

                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, order)
                trend = np.polyval(coeffs, x)

                detrended = segment - trend
                f2.append(np.mean(detrended ** 2))

            if f2:
                fluctuations.append(np.sqrt(np.mean(f2)))
                valid_scales.append(scale)

        if len(valid_scales) < 3:
            return HurstResult(
                hurst=0.5,
                regime=HurstRegime.RANDOM_WALK,
                confidence=0.0,
                method='dfa'
            )

        # Log-log regression: log(F(n)) = H * log(n) + c
        log_scales = np.log(valid_scales)
        log_fluct = np.log(fluctuations)

        hurst, confidence = self._fit_hurst(log_scales, log_fluct)
        regime = self._classify_regime(hurst)

        return HurstResult(
            hurst=hurst,
            regime=regime,
            confidence=confidence,
            method='dfa'
        )

    def calculate_rolling(
        self,
        data: pd.Series,
        window: int = 100,
        method: str = 'dfa',
        min_periods: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling Hurst exponent for real-time regime detection.

        Args:
            data: Time series (prices or returns)
            window: Rolling window size
            method: 'rs' or 'dfa'
            min_periods: Minimum periods required (defaults to window//2)

        Returns:
            DataFrame with columns: hurst, regime, confidence
        """
        if min_periods is None:
            min_periods = window // 2

        n = len(data)
        results = {
            'hurst': np.full(n, np.nan),
            'regime': np.full(n, '', dtype=object),
            'confidence': np.full(n, np.nan),
        }

        calc_func = self.calculate_dfa if method == 'dfa' else self.calculate_rs

        for i in range(min_periods - 1, n):
            start = max(0, i - window + 1)
            segment = data.iloc[start:i + 1].values

            if len(segment) >= min_periods:
                result = calc_func(segment)
                results['hurst'][i] = result.hurst
                results['regime'][i] = result.regime.value
                results['confidence'][i] = result.confidence

        return pd.DataFrame(results, index=data.index)

    def _fit_hurst(
        self,
        log_scales: np.ndarray,
        log_values: np.ndarray
    ) -> Tuple[float, float]:
        """
        Fit Hurst exponent using linear regression in log-log space.

        Returns:
            Tuple of (hurst_exponent, r_squared)
        """
        # Remove any invalid values
        mask = np.isfinite(log_scales) & np.isfinite(log_values)
        log_scales = log_scales[mask]
        log_values = log_values[mask]

        if len(log_scales) < 2:
            return 0.5, 0.0

        # Linear regression
        coeffs = np.polyfit(log_scales, log_values, 1)
        hurst = coeffs[0]

        # R-squared
        predicted = np.polyval(coeffs, log_scales)
        ss_res = np.sum((log_values - predicted) ** 2)
        ss_tot = np.sum((log_values - np.mean(log_values)) ** 2)

        r_squared = 1 - (ss_res / (ss_tot + 1e-10)) if ss_tot > 0 else 0.0

        # Clamp hurst to valid range
        hurst = np.clip(hurst, 0.0, 1.0)

        return hurst, max(0.0, r_squared)

    def _classify_regime(self, hurst: float) -> HurstRegime:
        """Classify market regime based on Hurst exponent."""
        if hurst > self.trending_threshold:
            return HurstRegime.TRENDING
        elif hurst < self.mean_revert_threshold:
            return HurstRegime.MEAN_REVERTING
        else:
            return HurstRegime.RANDOM_WALK


def compute_hurst_simple(data: np.ndarray, method: str = 'dfa') -> float:
    """
    Simple function to compute Hurst exponent.

    Args:
        data: Time series data
        method: 'rs' or 'dfa'

    Returns:
        Hurst exponent value
    """
    calculator = HurstCalculator()
    if method == 'dfa':
        result = calculator.calculate_dfa(data)
    else:
        result = calculator.calculate_rs(data)
    return result.hurst
