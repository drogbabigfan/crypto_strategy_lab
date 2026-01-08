"""
AKF V2 - Log-Price Kalman Filter Based Strategies.

New approach focusing on:
1. Log-return velocity as direct profit signal
2. Uncertainty-scaled position sizing
3. Adaptive regime detection
4. Aggressive entry with risk-managed exits

Target: High Total PnL (50%+ annually)
"""

from .features import LogKalmanFeatures
from .base import BaseStrategy, StrategyConfig

__all__ = [
    'LogKalmanFeatures',
    'BaseStrategy',
    'StrategyConfig',
]
