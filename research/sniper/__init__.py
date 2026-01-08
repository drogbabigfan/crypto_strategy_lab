"""
Sniper Classification System (v4.9)

End-to-End supervised learning for high-confidence entry prediction.
"""

from .labeler import SniperLabeler, SniperTBMConfig, LabelResult
from .model import SniperClassifier
from .dataset import SniperDataset, create_balanced_dataloader
from .trainer import SniperTrainer, SniperTrainingConfig
from .signal_generator import ConfidenceSignalGenerator

__all__ = [
    "SniperLabeler",
    "SniperTBMConfig",
    "LabelResult",
    "SniperClassifier",
    "SniperDataset",
    "create_balanced_dataloader",
    "SniperTrainer",
    "SniperTrainingConfig",
    "ConfidenceSignalGenerator",
]
