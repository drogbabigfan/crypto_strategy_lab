"""
Label Optimizer Module (v4.6)

Triple Barrier labeling with Plateau Search optimization.
"""

from .cost_model import SquareRootCostModel
from .tbm import TBMConfig, TripleBarrierLabeler
from .scoring import compute_entropy, compute_mi, compute_rank_ic
from .plateau import find_plateau_center
from .optimizer import LabelOptimizer, OptimizationResult

__all__ = [
    "SquareRootCostModel",
    "TBMConfig",
    "TripleBarrierLabeler",
    "compute_entropy",
    "compute_mi",
    "compute_rank_ic",
    "find_plateau_center",
    "LabelOptimizer",
    "OptimizationResult",
]
