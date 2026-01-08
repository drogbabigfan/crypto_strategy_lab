"""
SSL Pre-training Module (v4.6)

Stage 0: One-off SSL Pre-training for Foundation Encoder.
"""

from .dataset import SSLDataConfig, PatchMaskingDataset, load_ssl_data
from .model import PatchTSTEncoder, SSLModel
from .train import SSLTrainingConfig, SSLTrainer

__all__ = [
    "SSLDataConfig",
    "PatchMaskingDataset",
    "load_ssl_data",
    "PatchTSTEncoder",
    "SSLModel",
    "SSLTrainingConfig",
    "SSLTrainer",
]
