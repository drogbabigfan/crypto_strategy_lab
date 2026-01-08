"""
Regime Detection Module.

Provides tools for identifying market regimes and selecting appropriate
trading strategies based on market conditions.

Components:
- HMM: Hidden Markov Model for state identification (Bull/Bear/Sideways)
- Hurst: Hurst exponent for persistence analysis (Trending/Mean-Reverting)
- Labeling: Ground truth labeling for training (ZigZag, Trend Scanning)
- Hybrid: Combined HMM-Hurst model with strategy recommendations

Usage:
    # Basic HMM regime detection
    from strategies.regime_detection import HMMRegimeDetector, HMMConfig

    detector = HMMRegimeDetector(HMMConfig(n_states=3))
    detector.fit(train_data)
    regimes = detector.predict(test_data)

    # Hurst exponent analysis
    from strategies.regime_detection import HurstCalculator

    calculator = HurstCalculator()
    result = calculator.calculate_dfa(returns)
    print(f"Hurst: {result.hurst}, Regime: {result.regime}")

    # Hybrid regime detection with strategy recommendations
    from strategies.regime_detection import HybridRegimeDetector, HybridConfig

    hybrid = HybridRegimeDetector(HybridConfig())
    hybrid.fit(train_data)
    state = hybrid.get_current_state(current_data)
    print(f"Strategy: {state.strategy_mode}, Scale: {state.position_scale}")

    # Labeling for training data
    from strategies.regime_detection import ZigZagLabeler, TrendScanningLabeler

    labeler = ZigZagLabeler()
    labels = labeler.label(data)
"""

# Hurst Exponent
from .hurst import (
    HurstCalculator,
    HurstResult,
    HurstRegime,
    compute_hurst_simple,
)

# HMM Regime Detection
from .hmm import (
    HMMRegimeDetector,
    HMMConfig,
    MarketRegime,
    RegimeState,
    GaussianHMMFromScratch,
)

# Labeling Methods
from .labeling import (
    ZigZagLabeler,
    ZigZagConfig,
    TrendScanningLabeler,
    TrendScanConfig,
    DrawdownLabeler,
    VolatilityRegimeLabeler,
    TrendLabel,
    PivotPoint,
    combine_labels,
)

# Hybrid Model
from .hybrid import (
    HybridRegimeDetector,
    HybridConfig,
    HybridRegimeState,
    StrategyMode,
    RegimeStrategySwitch,
    create_default_hybrid_detector,
)

# Strategy
from .strategy import (
    RegimeStrategy,
    RegimeStrategyConfig,
    RegimeBacktester,
    PositionType,
)

__all__ = [
    # Hurst
    'HurstCalculator',
    'HurstResult',
    'HurstRegime',
    'compute_hurst_simple',

    # HMM
    'HMMRegimeDetector',
    'HMMConfig',
    'MarketRegime',
    'RegimeState',
    'GaussianHMMFromScratch',

    # Labeling
    'ZigZagLabeler',
    'ZigZagConfig',
    'TrendScanningLabeler',
    'TrendScanConfig',
    'DrawdownLabeler',
    'VolatilityRegimeLabeler',
    'TrendLabel',
    'PivotPoint',
    'combine_labels',

    # Hybrid
    'HybridRegimeDetector',
    'HybridConfig',
    'HybridRegimeState',
    'StrategyMode',
    'RegimeStrategySwitch',
    'create_default_hybrid_detector',

    # Strategy
    'RegimeStrategy',
    'RegimeStrategyConfig',
    'RegimeBacktester',
    'PositionType',
]
