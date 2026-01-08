"""
LPPLS (Log-Periodic Power Law Singularity) Module.

Implements the JLS (Johansen-Ledoit-Sornette) model for financial bubble detection.

Main Components:
    - LPPLSModel: Core LPPLS model with Filimonov-Sornette optimization
    - LPPLSIndicator: Multi-window confidence indicator
    - FilterConfig: Parameter filtering configuration

Quick Start:
    >>> from strategies.lppls import LPPLSIndicator, fit_lppls
    >>> import numpy as np
    >>>
    >>> # Single fit
    >>> prices = np.array([...])  # price data
    >>> result = fit_lppls(prices)
    >>> if result:
    ...     print(f"tc={result.params.tc}, m={result.params.m}")
    >>>
    >>> # Multi-window confidence indicator
    >>> indicator = LPPLSIndicator()
    >>> conf_result = indicator.calculate(np.log(prices))
    >>> print(f"Confidence: {conf_result.confidence:.2%}")

Trading Signals:
    - confidence >= 0.8: Strong bubble signal (consider short position)
    - confidence >= 0.5: Moderate signal (watch closely)
    - confidence < 0.5: No clear bubble pattern
"""

# Core model
from .lppls import (
    LPPLSModel,
    LPPLSParams,
    LPPLSResult,
    fit_lppls,
)

# Filtering
from .filter import (
    FilterConfig,
    FilterResult,
    apply_filters,
    filter_results,
    get_qualified_results,
)

# Indicators
from .indicator import (
    LPPLSIndicator,
    WindowConfig,
    ConfidenceResult,
    CLIPPoint,
    calculate_rolling_confidence,
    build_clip,
    get_bubble_signal,
)

# Strategy
from .strategy import (
    LPPLSStrategy,
    LPPLSStrategyConfig,
    BacktestResult,
    Trade,
    run_lppls_backtest,
    load_btc_data,
)

# Walk-Forward Optimization
from .walk_forward import (
    WalkForwardOptimizer,
    WalkForwardConfig,
    WalkForwardResult,
    run_walk_forward_backtest,
)

# Deep LPPLS (Neural Network)
from .deep_lppls import (
    DeepLPPLSConfig,
    DeepLPPLSPredictor,
    DeepLPPLSTrainer,
    PLNN,
    PLNNWithConv,
    LPPLSDataGenerator,
    compare_speed,
)

# Fast Strategy (Deep LPPLS based)
from .fast_strategy import (
    FastLPPLSStrategy,
    FastLPPLSConfig,
    FastSignal,
    run_fast_backtest,
)

# Optimized configuration
from .optimized_config import OPTIMIZED_PARAMS

# Fast optimization
from .fast_optimize import (
    calculate_all_signals,
    filter_signals,
    run_trading_logic,
    fast_grid_search,
    run_fast_optimization,
)

__all__ = [
    # Core
    'LPPLSModel',
    'LPPLSParams',
    'LPPLSResult',
    'fit_lppls',

    # Filter
    'FilterConfig',
    'FilterResult',
    'apply_filters',
    'filter_results',
    'get_qualified_results',

    # Indicator
    'LPPLSIndicator',
    'WindowConfig',
    'ConfidenceResult',
    'CLIPPoint',
    'calculate_rolling_confidence',
    'build_clip',
    'get_bubble_signal',

    # Strategy
    'LPPLSStrategy',
    'LPPLSStrategyConfig',
    'BacktestResult',
    'Trade',
    'run_lppls_backtest',
    'load_btc_data',

    # Walk-Forward
    'WalkForwardOptimizer',
    'WalkForwardConfig',
    'WalkForwardResult',
    'run_walk_forward_backtest',

    # Deep LPPLS
    'DeepLPPLSConfig',
    'DeepLPPLSPredictor',
    'DeepLPPLSTrainer',
    'PLNN',
    'PLNNWithConv',
    'LPPLSDataGenerator',
    'compare_speed',

    # Fast Strategy
    'FastLPPLSStrategy',
    'FastLPPLSConfig',
    'FastSignal',
    'run_fast_backtest',

    # Optimized config
    'OPTIMIZED_PARAMS',

    # Fast optimization
    'calculate_all_signals',
    'filter_signals',
    'run_trading_logic',
    'fast_grid_search',
    'run_fast_optimization',
]
