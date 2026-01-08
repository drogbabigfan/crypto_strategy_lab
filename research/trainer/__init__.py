"""
Fine-tuning Trainer Module (v4.6)

Step 3.2: Fine-tuning - Pre-trained Encoder + Classification Head
"""

from .dataset import FineTuningDataset, FineTuningConfig
from .model import PatchTSTClassifier, ResidualBottleneckHead
from .loss import FocalLoss
from .train import (
    FineTuningTrainer,
    FineTuningTrainingConfig,
)

__all__ = [
    # Dataset
    "FineTuningDataset",
    "FineTuningConfig",
    # Model
    "PatchTSTClassifier",
    "ResidualBottleneckHead",
    # Loss
    "FocalLoss",
    # Training
    "FineTuningTrainer",
    "FineTuningTrainingConfig",
]
