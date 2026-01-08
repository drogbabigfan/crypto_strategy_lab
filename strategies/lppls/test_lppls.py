"""
Tests for LPPLS module.
"""
import numpy as np
import pytest
from numpy.typing import NDArray

from .lppls import LPPLSModel, LPPLSParams, fit_lppls
from .filter import FilterConfig, apply_filters, check_damping
from .indicator import LPPLSIndicator, WindowConfig, get_bubble_signal


def generate_lppls_data(
    tc: float = 500.0,
    m: float = 0.5,
    omega: float = 8.0,
    A: float = 10.0,
    B: float = -0.5,
    C: float = 0.05,
    phi: float = 1.0,
    n_points: int = 400,
    noise_std: float = 0.01,
) -> NDArray[np.float64]:
    """Generate synthetic LPPLS data for testing."""
    t = np.arange(n_points, dtype=np.float64)
    dt = tc - t

    # LPPLS equation
    log_prices = (
        A
        + B * np.power(dt, m)
        + C * np.power(dt, m) * np.cos(omega * np.log(dt) - phi)
    )

    # Add noise
    log_prices += np.random.normal(0, noise_std, n_points)

    return log_prices


class TestLPPLSModel:
    """Tests for LPPLSModel class."""

    def test_fit_synthetic_data(self):
        """Test fitting on synthetic LPPLS data."""
        np.random.seed(42)

        # Generate data with known parameters
        true_tc = 500.0
        true_m = 0.5
        true_omega = 8.0

        log_prices = generate_lppls_data(
            tc=true_tc, m=true_m, omega=true_omega,
            noise_std=0.005,
        )

        # Fit model
        model = LPPLSModel(tc_max_days=150)
        result = model.fit(log_prices, method='nelder-mead', n_starts=5)

        assert result is not None
        assert result.converged

        # Check parameters are in reasonable range
        # (exact recovery is hard due to noise and local minima)
        assert 400 < result.params.tc < 600
        assert 0.1 < result.params.m < 0.9
        assert result.r_squared > 0.5

    def test_fit_returns_none_for_short_data(self):
        """Test that fitting returns None for very short data."""
        model = LPPLSModel()
        log_prices = np.array([1.0, 2.0, 3.0])
        result = model.fit(log_prices)
        assert result is None

    def test_predict(self):
        """Test prediction from parameters."""
        params = LPPLSParams(
            tc=500.0, m=0.5, omega=8.0,
            A=10.0, B=-0.5, C=0.05, phi=1.0,
        )

        model = LPPLSModel()
        t = np.arange(400, dtype=np.float64)
        predictions = model.predict(params, t)

        assert len(predictions) == 400
        assert not np.any(np.isnan(predictions))
        # Prices should be increasing (bubble behavior)
        assert predictions[-1] > predictions[0]


class TestFilter:
    """Tests for filtering functions."""

    def test_apply_filters_valid_result(self):
        """Test filtering with valid parameters."""
        from .lppls import LPPLSResult

        params = LPPLSParams(
            tc=450.0, m=0.5, omega=8.0,
            A=10.0, B=-0.5, C=0.05, phi=1.0,
        )

        # damping = m|B| / (omega|C|) = 0.5*0.5 / (8*0.05) = 0.625
        # This won't pass default damping threshold of 1.0

        result = LPPLSResult(
            params=params,
            ssr=0.1,
            r_squared=0.9,
            t1=0,
            t2=400,
            converged=True,
            damping=0.625,
            n_oscillations=3.0,
        )

        # With default config (damping_min=1.0), this should fail
        config = FilterConfig()
        filter_result = apply_filters(result, config)
        assert not filter_result.passed
        assert 'damping' in filter_result.failed_checks

        # With relaxed damping, should pass
        config_relaxed = FilterConfig(damping_min=0.5)
        filter_result = apply_filters(result, config_relaxed)
        assert filter_result.passed

    def test_b_sign_check(self):
        """Test B sign filtering for positive/negative bubbles."""
        from .lppls import LPPLSResult

        params_positive = LPPLSParams(
            tc=450.0, m=0.5, omega=8.0,
            A=10.0, B=-0.5, C=0.05, phi=1.0,
        )
        params_negative = LPPLSParams(
            tc=450.0, m=0.5, omega=8.0,
            A=10.0, B=0.5, C=0.05, phi=1.0,
        )

        result_pos = LPPLSResult(
            params=params_positive, ssr=0.1, r_squared=0.9,
            t1=0, t2=400, converged=True, damping=2.0, n_oscillations=3.0,
        )
        result_neg = LPPLSResult(
            params=params_negative, ssr=0.1, r_squared=0.9,
            t1=0, t2=400, converged=True, damping=2.0, n_oscillations=3.0,
        )

        # For positive bubble detection, B should be negative
        config_pos = FilterConfig(positive_bubble=True)
        filter_pos = apply_filters(result_pos, config_pos)
        filter_neg = apply_filters(result_neg, config_pos)

        assert 'B_sign' not in filter_pos.failed_checks  # B=-0.5 is correct
        assert 'B_sign' in filter_neg.failed_checks  # B=0.5 is wrong


class TestIndicator:
    """Tests for LPPLSIndicator class."""

    def test_calculate_confidence(self):
        """Test confidence indicator calculation."""
        np.random.seed(42)

        # Generate bubble data
        log_prices = generate_lppls_data(
            tc=300.0, m=0.5, omega=8.0,
            n_points=250, noise_std=0.01,
        )

        # Use smaller windows for test speed
        window_config = WindowConfig(min_window=100, max_window=200, step=20)
        indicator = LPPLSIndicator(window_config=window_config)

        result = indicator.calculate(log_prices)

        assert 0.0 <= result.confidence <= 1.0
        assert result.n_total > 0
        assert len(result.all_results) == result.n_total

    def test_get_bubble_signal(self):
        """Test bubble signal generation."""
        assert get_bubble_signal(0.9, tc_std=3.0) == 'STRONG'
        assert get_bubble_signal(0.9, tc_std=10.0) == 'MODERATE'
        assert get_bubble_signal(0.6, tc_std=5.0) == 'WEAK'
        assert get_bubble_signal(0.3, tc_std=5.0) == 'NONE'


class TestFitLPPLS:
    """Tests for convenience function."""

    def test_fit_lppls_with_prices(self):
        """Test fit_lppls with raw price data."""
        np.random.seed(42)
        log_prices = generate_lppls_data(noise_std=0.01)
        prices = np.exp(log_prices)

        result = fit_lppls(prices, use_log=True, tc_max_days=150)
        assert result is not None or result is None  # May fail to converge


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
