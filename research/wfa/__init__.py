"""
WFA (Walk-Forward Analysis) Engine.

Orchestrates the entire WFA loop:
1. Window splitting (Train/Test)
2. Label optimization (Step 3.1)
3. Fine-tuning (Step 3.2)
4. Signal generation
5. Backtest via Go (Step 3.3)
6. Results aggregation
"""

from .go_bridge import GoBridge, BacktestConfig, BacktestResult
from .signal_generator import SignalGenerator
from .engine import WFAEngine, WFAConfig, FoldResult

__all__ = [
    "GoBridge",
    "BacktestConfig",
    "BacktestResult",
    "SignalGenerator",
    "WFAEngine",
    "WFAConfig",
    "FoldResult",
]
