"""
LPPLS Confidence Indicator and Multi-Window Analysis.

Implements the DS LPPLS Confidence Indicator methodology:
    Confidence = Number of Qualified Fits / Total Number of Fits

Multi-window analysis performs LPPLS fitting across multiple time windows
with varying start dates (t1) while keeping end date (t2) fixed.

Window Grid:
    - Minimum window: 120 days (~6 months)
    - Maximum window: 750 days (~3 years)
    - Step: 5 days
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from numpy.typing import NDArray

from .lppls import LPPLSModel, LPPLSResult, fit_lppls
from .filter import FilterConfig, FilterResult, apply_filters, get_qualified_results


@dataclass
class WindowConfig:
    """Configuration for multi-window analysis."""
    min_window: int = 120   # Minimum window size (days)
    max_window: int = 750   # Maximum window size (days)
    step: int = 5           # Step size for window grid


@dataclass
class ConfidenceResult:
    """Result of confidence indicator calculation."""
    confidence: float                    # Confidence indicator (0-1)
    n_qualified: int                     # Number of qualified fits
    n_total: int                         # Total number of fits attempted
    qualified_results: List[LPPLSResult] # Results that passed filters
    all_results: List[LPPLSResult]       # All fitting results
    tc_mean: Optional[float]             # Mean tc from qualified fits
    tc_std: Optional[float]              # Std of tc from qualified fits
    tc_median: Optional[float]           # Median tc from qualified fits


@dataclass
class CLIPPoint:
    """Single point for Crash Lock-In Plot (CLIP)."""
    date_index: int          # Current date index
    tc: float                # Predicted tc
    confidence: float        # Confidence at this date
    tc_std: float            # Standard deviation of tc


class LPPLSIndicator:
    """
    LPPLS Confidence Indicator with multi-window analysis.

    The indicator performs LPPLS fitting across multiple windows
    and calculates the proportion of fits that pass quality filters.
    """

    def __init__(
        self,
        window_config: Optional[WindowConfig] = None,
        filter_config: Optional[FilterConfig] = None,
        model_kwargs: Optional[dict] = None,
    ):
        """
        Initialize LPPLS indicator.

        Args:
            window_config: Configuration for multi-window analysis
            filter_config: Configuration for parameter filtering
            model_kwargs: Additional arguments for LPPLSModel
        """
        self.window_config = window_config or WindowConfig()
        self.filter_config = filter_config or FilterConfig()
        self.model_kwargs = model_kwargs or {}

    def calculate(
        self,
        log_prices: NDArray[np.float64],
        n_jobs: int = 1,
    ) -> ConfidenceResult:
        """
        Calculate confidence indicator for given price data.

        Args:
            log_prices: Array of log prices
            n_jobs: Number of parallel jobs (1 for sequential)

        Returns:
            ConfidenceResult with confidence indicator and details
        """
        n = len(log_prices)
        windows = self._generate_windows(n)

        if not windows:
            return ConfidenceResult(
                confidence=0.0,
                n_qualified=0,
                n_total=0,
                qualified_results=[],
                all_results=[],
                tc_mean=None,
                tc_std=None,
                tc_median=None,
            )

        # Fit LPPLS for each window
        if n_jobs > 1:
            all_results = self._fit_parallel(log_prices, windows, n_jobs)
        else:
            all_results = self._fit_sequential(log_prices, windows)

        # Filter results
        qualified_results = get_qualified_results(all_results, self.filter_config)

        # Calculate confidence
        n_total = len(all_results)
        n_qualified = len(qualified_results)
        confidence = n_qualified / n_total if n_total > 0 else 0.0

        # Calculate tc statistics from qualified results
        if qualified_results:
            tc_values = [r.params.tc for r in qualified_results]
            tc_mean = np.mean(tc_values)
            tc_std = np.std(tc_values)
            tc_median = np.median(tc_values)
        else:
            tc_mean = None
            tc_std = None
            tc_median = None

        return ConfidenceResult(
            confidence=confidence,
            n_qualified=n_qualified,
            n_total=n_total,
            qualified_results=qualified_results,
            all_results=all_results,
            tc_mean=tc_mean,
            tc_std=tc_std,
            tc_median=tc_median,
        )

    def _generate_windows(self, n: int) -> List[Tuple[int, int]]:
        """Generate window (t1, t2) pairs for multi-window analysis."""
        windows = []
        t2 = n - 1  # End at last data point

        # Generate t1 values from (t2 - max_window) to (t2 - min_window)
        t1_start = max(0, t2 - self.window_config.max_window)
        t1_end = max(0, t2 - self.window_config.min_window)

        for t1 in range(t1_start, t1_end + 1, self.window_config.step):
            if t2 - t1 >= self.window_config.min_window:
                windows.append((t1, t2))

        return windows

    def _fit_sequential(
        self,
        log_prices: NDArray[np.float64],
        windows: List[Tuple[int, int]],
    ) -> List[LPPLSResult]:
        """Fit LPPLS sequentially for each window."""
        results = []
        model = LPPLSModel(**self.model_kwargs)

        for t1, t2 in windows:
            window_data = log_prices[t1:t2+1]
            result = model.fit(window_data)
            if result is not None:
                # Adjust indices to global coordinates
                result.t1 = t1
                result.t2 = t2
                result.params.tc = result.params.tc + t1
                results.append(result)

        return results

    def _fit_parallel(
        self,
        log_prices: NDArray[np.float64],
        windows: List[Tuple[int, int]],
        n_jobs: int,
    ) -> List[LPPLSResult]:
        """Fit LPPLS in parallel for each window."""
        results = []

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {}
            for t1, t2 in windows:
                window_data = log_prices[t1:t2+1]
                future = executor.submit(
                    _fit_window,
                    window_data,
                    t1,
                    t2,
                    self.model_kwargs,
                )
                futures[future] = (t1, t2)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception:
                    continue

        return results


def _fit_window(
    window_data: NDArray[np.float64],
    t1: int,
    t2: int,
    model_kwargs: dict,
) -> Optional[LPPLSResult]:
    """Helper function for parallel fitting (must be top-level for pickling)."""
    model = LPPLSModel(**model_kwargs)
    result = model.fit(window_data)
    if result is not None:
        result.t1 = t1
        result.t2 = t2
        result.params.tc = result.params.tc + t1
    return result


def calculate_rolling_confidence(
    log_prices: NDArray[np.float64],
    lookback: int = 60,
    window_config: Optional[WindowConfig] = None,
    filter_config: Optional[FilterConfig] = None,
) -> NDArray[np.float64]:
    """
    Calculate rolling confidence indicator over time.

    Args:
        log_prices: Array of log prices
        lookback: Lookback period for rolling calculation
        window_config: Configuration for multi-window analysis
        filter_config: Configuration for parameter filtering

    Returns:
        Array of confidence values for each date
    """
    n = len(log_prices)
    confidence = np.zeros(n)
    indicator = LPPLSIndicator(window_config, filter_config)

    min_data = window_config.min_window if window_config else 120

    for i in range(min_data, n):
        end_idx = i + 1
        result = indicator.calculate(log_prices[:end_idx])
        confidence[i] = result.confidence

    return confidence


def build_clip(
    log_prices: NDArray[np.float64],
    dates: Optional[List[int]] = None,
    window_config: Optional[WindowConfig] = None,
    filter_config: Optional[FilterConfig] = None,
) -> List[CLIPPoint]:
    """
    Build Crash Lock-In Plot (CLIP) data.

    CLIP shows how tc predictions converge as crash approaches.
    When tc values start clustering (low std), crash is imminent.

    Args:
        log_prices: Array of log prices
        dates: Specific date indices to calculate (None for all)
        window_config: Configuration for multi-window analysis
        filter_config: Configuration for parameter filtering

    Returns:
        List of CLIPPoint data for plotting
    """
    n = len(log_prices)
    indicator = LPPLSIndicator(window_config, filter_config)

    min_data = window_config.min_window if window_config else 120

    if dates is None:
        dates = list(range(min_data, n))

    clip_data = []

    for date_idx in dates:
        if date_idx < min_data:
            continue

        result = indicator.calculate(log_prices[:date_idx+1])

        if result.tc_mean is not None:
            clip_data.append(CLIPPoint(
                date_index=date_idx,
                tc=result.tc_mean,
                confidence=result.confidence,
                tc_std=result.tc_std or 0.0,
            ))

    return clip_data


def get_bubble_signal(
    confidence: float,
    tc_std: Optional[float] = None,
    confidence_threshold: float = 0.8,
    tc_std_threshold: float = 5.0,
) -> str:
    """
    Get trading signal based on confidence indicator.

    Args:
        confidence: Confidence indicator value
        tc_std: Standard deviation of tc predictions
        confidence_threshold: Threshold for high confidence
        tc_std_threshold: Threshold for tc convergence

    Returns:
        Signal string: 'STRONG', 'MODERATE', 'WEAK', 'NONE'
    """
    if confidence >= confidence_threshold:
        if tc_std is not None and tc_std <= tc_std_threshold:
            return 'STRONG'  # High confidence + converged tc
        return 'MODERATE'    # High confidence but uncertain tc
    elif confidence >= 0.5:
        return 'WEAK'        # Some evidence of bubble
    else:
        return 'NONE'        # No clear bubble signal
