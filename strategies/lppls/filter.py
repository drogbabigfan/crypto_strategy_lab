"""
LPPLS Parameter Filtering Module.

Implements filtering conditions from Filimonov-Sornette research
to identify valid bubble signals and reduce false positives.

Filtering Criteria:
    1. tc: t2 < tc < t2 + 60 days (critical time within reasonable range)
    2. m: 0.1 < m < 0.9 (strict: 0.3 < m < 0.7) for super-exponential growth
    3. omega: 6 < omega < 13 (log-periodic frequency in empirical range)
    4. B < 0 for positive bubble (price going up before crash)
    5. Damping condition: D = m|B| / (omega|C|) >= 1
    6. Oscillation count: (omega/2pi) * ln((tc-t1)/(tc-t2)) >= 2.5
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from .lppls import LPPLSResult, LPPLSParams


@dataclass
class FilterConfig:
    """Configuration for LPPLS parameter filtering."""
    # tc constraints
    tc_min_days: int = 1      # Min days ahead for tc
    tc_max_days: int = 60     # Max days ahead for tc

    # m constraints (power law exponent)
    m_min: float = 0.1
    m_max: float = 0.9
    m_strict_min: float = 0.3  # Stricter bound for higher confidence
    m_strict_max: float = 0.7

    # omega constraints (angular frequency)
    omega_min: float = 6.0
    omega_max: float = 13.0

    # Damping condition threshold
    damping_min: float = 1.0   # D = m|B| / (omega|C|) >= 1

    # Minimum oscillations in window
    min_oscillations: float = 2.5

    # R-squared threshold
    r_squared_min: float = 0.5

    # Use strict bounds
    use_strict_bounds: bool = False

    # Bubble direction
    positive_bubble: bool = True  # B < 0 for positive bubble


@dataclass
class FilterResult:
    """Result of applying filters to an LPPLS fit."""
    passed: bool                     # Whether all filters passed
    result: LPPLSResult              # Original LPPLS result
    failed_checks: List[str]         # List of failed check names
    quality_score: float             # Quality score (0-1)


def check_tc(result: LPPLSResult, config: FilterConfig, t2: int) -> bool:
    """Check if tc is within valid range."""
    tc = result.params.tc
    tc_min = t2 + config.tc_min_days
    tc_max = t2 + config.tc_max_days
    return tc_min <= tc <= tc_max


def check_m(result: LPPLSResult, config: FilterConfig) -> bool:
    """Check if m (power law exponent) is within valid range."""
    m = result.params.m
    if config.use_strict_bounds:
        return config.m_strict_min <= m <= config.m_strict_max
    return config.m_min <= m <= config.m_max


def check_omega(result: LPPLSResult, config: FilterConfig) -> bool:
    """Check if omega (angular frequency) is within valid range."""
    omega = result.params.omega
    return config.omega_min <= omega <= config.omega_max


def check_B_sign(result: LPPLSResult, config: FilterConfig) -> bool:
    """Check if B has correct sign for bubble direction."""
    B = result.params.B
    if config.positive_bubble:
        return B < 0  # B < 0 for positive bubble (price going up)
    return B > 0  # B > 0 for negative bubble (price going down)


def check_damping(result: LPPLSResult, config: FilterConfig) -> bool:
    """
    Check damping condition: D = m|B| / (omega|C|) >= threshold.

    This ensures oscillations don't dominate the trend.
    """
    return result.damping >= config.damping_min


def check_oscillations(result: LPPLSResult, config: FilterConfig) -> bool:
    """Check if there are enough oscillations in the window."""
    return result.n_oscillations >= config.min_oscillations


def check_r_squared(result: LPPLSResult, config: FilterConfig) -> bool:
    """Check if R-squared is above threshold."""
    return result.r_squared >= config.r_squared_min


def apply_filters(
    result: LPPLSResult,
    config: Optional[FilterConfig] = None,
) -> FilterResult:
    """
    Apply all LPPLS filters to a fitting result.

    Args:
        result: LPPLS fitting result
        config: Filter configuration (uses defaults if None)

    Returns:
        FilterResult with pass/fail status and details
    """
    if config is None:
        config = FilterConfig()

    t2 = result.t2
    failed_checks = []

    # Apply each filter
    checks = [
        ('tc_range', check_tc(result, config, t2)),
        ('m_range', check_m(result, config)),
        ('omega_range', check_omega(result, config)),
        ('B_sign', check_B_sign(result, config)),
        ('damping', check_damping(result, config)),
        ('oscillations', check_oscillations(result, config)),
        ('r_squared', check_r_squared(result, config)),
    ]

    for check_name, passed in checks:
        if not passed:
            failed_checks.append(check_name)

    # Calculate quality score (0-1)
    quality_score = calculate_quality_score(result, config)

    return FilterResult(
        passed=len(failed_checks) == 0,
        result=result,
        failed_checks=failed_checks,
        quality_score=quality_score,
    )


def calculate_quality_score(
    result: LPPLSResult,
    config: FilterConfig,
) -> float:
    """
    Calculate quality score for an LPPLS fit (0-1).

    Higher scores indicate more reliable fits.
    Considers:
        - How well parameters fit within preferred ranges
        - R-squared value
        - Damping condition satisfaction
        - Number of oscillations
    """
    scores = []

    # m score: prefer middle of range (0.4-0.6)
    m = result.params.m
    if 0.4 <= m <= 0.6:
        m_score = 1.0
    elif 0.3 <= m <= 0.7:
        m_score = 0.8
    elif 0.1 <= m <= 0.9:
        m_score = 0.5
    else:
        m_score = 0.0
    scores.append(m_score)

    # omega score: prefer 6-13 range
    omega = result.params.omega
    if 6 <= omega <= 13:
        omega_score = 1.0
    elif 4 <= omega <= 25:
        omega_score = 0.5
    else:
        omega_score = 0.0
    scores.append(omega_score)

    # Damping score
    if result.damping >= 1.5:
        damping_score = 1.0
    elif result.damping >= 1.0:
        damping_score = 0.8
    elif result.damping >= 0.5:
        damping_score = 0.5
    else:
        damping_score = 0.0
    scores.append(damping_score)

    # Oscillation score
    if result.n_oscillations >= 3.0:
        osc_score = 1.0
    elif result.n_oscillations >= 2.5:
        osc_score = 0.8
    elif result.n_oscillations >= 2.0:
        osc_score = 0.5
    else:
        osc_score = 0.0
    scores.append(osc_score)

    # R-squared score
    r2 = result.r_squared
    if r2 >= 0.8:
        r2_score = 1.0
    elif r2 >= 0.6:
        r2_score = 0.7
    elif r2 >= 0.4:
        r2_score = 0.4
    else:
        r2_score = 0.0
    scores.append(r2_score)

    # B sign score (binary)
    B = result.params.B
    b_score = 1.0 if (config.positive_bubble and B < 0) or (not config.positive_bubble and B > 0) else 0.0
    scores.append(b_score)

    return np.mean(scores)


def filter_results(
    results: List[LPPLSResult],
    config: Optional[FilterConfig] = None,
) -> List[FilterResult]:
    """
    Apply filters to multiple LPPLS results.

    Args:
        results: List of LPPLS fitting results
        config: Filter configuration

    Returns:
        List of FilterResults
    """
    return [apply_filters(r, config) for r in results]


def get_qualified_results(
    results: List[LPPLSResult],
    config: Optional[FilterConfig] = None,
) -> List[LPPLSResult]:
    """
    Get only results that pass all filters.

    Args:
        results: List of LPPLS fitting results
        config: Filter configuration

    Returns:
        List of LPPLSResult that passed all filters
    """
    filtered = filter_results(results, config)
    return [fr.result for fr in filtered if fr.passed]
