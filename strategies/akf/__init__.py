"""
AKF (Adaptive Kalman Filter) Trading Strategies.

Recommended Strategy:
- AKFDualTrackStrategy: Best performer (+101% PnL, Sharpe 0.95)

Other Strategies:
- AKFAdvancedStrategy: Good (+31% PnL, Sharpe 0.45)
- DKFStrategy: Fair (+13% PnL, Sharpe 0.18)

Deprecated (failed after look-ahead bias fix):
- AKFStrategies (V1): -50%
- AKFStrategiesV2: -50%
- AKFStrategiesV3: -50%
- BenhamouModel4: -50%
- AKFMeanReversion: -50%
"""

# Recommended - Best Performer
from .strategies_dual_track import (
    AKFDualTrackStrategy,
    DualTrackConfig,
    prepare_kalman_data,
)

# Good Alternative
from .strategies_advanced import (
    AKFAdvancedStrategy,
    AKFAdvancedConfig,
)

# Fair - Dual Kalman Crossover
from .strategies_dkf import (
    DKFStrategy,
    DKFConfig,
)

# Turtle - Donchian-Kalman Hybrid
from .strategies_turtle import (
    AKFTurtleStrategy,
    TurtleConfig,
)

# Legacy (deprecated but kept for reference)
from .strategies import (
    AKFStrategyConfig,
    AKFStrategies,
)

__all__ = [
    # Recommended
    'AKFDualTrackStrategy',
    'DualTrackConfig',
    'prepare_kalman_data',

    # Good
    'AKFAdvancedStrategy',
    'AKFAdvancedConfig',

    # Fair
    'DKFStrategy',
    'DKFConfig',

    # Turtle - Donchian-Kalman Hybrid
    'AKFTurtleStrategy',
    'TurtleConfig',

    # Legacy
    'AKFStrategyConfig',
    'AKFStrategies',
]
