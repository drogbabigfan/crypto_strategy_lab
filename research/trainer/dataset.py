"""
Fine-tuning Dataset (v4.6)

Sample-level Drop: Context Window 내부의 바를 삭제하지 않음.
Fee Trap인 샘플 전체를 학습에서 제외.
"""

from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class FineTuningConfig:
    """Fine-tuning dataset configuration."""

    context_len: int = 512
    n_features: int = 29  # Excluding is_primed
    feature_cols: List[str] = field(default_factory=lambda: [
        "log_volume", "log_tick_count", "log_duration",
        "log_trade_intensity", "vwap_deviation", "volume_imbalance",
        "bar_range", "bar_body", "garman_klass_vol", "realized_vol",
        "shannon_entropy", "vol_ratio", "vol_zscore", "entropy_zscore",
        "skewness", "kurtosis", "frac_diff_close", "detrended_log_price",
        "returns", "vw_momentum", "momentum_zscore_10", "momentum_zscore_50",
        "momentum_zscore_250", "momentum_zscore_1000", "connors_rsi",
        "sin_time", "cos_time", "sin_week", "cos_week",
    ])


class FineTuningDataset(Dataset):
    """
    Fine-tuning Dataset with Sample-level Drop.

    v4.6: Context Window 내부의 바를 삭제하지 않음.
    Fee Trap인 샘플 전체를 학습에서 제외.

    예시:
        Sample = (bars[100:612], label[612])
        - label[612]가 Fee Trap이면 → 이 샘플 제외
        - bars[100:612]는 항상 연속 512개 유지

    Args:
        features: (N_total, n_features) array of all features
        valid_indices: Indices from Label Optimizer (Step 3.1)
        labels: Labels from Label Optimizer {-1, 0, +1}
        context_len: Context window length (default 512)
    """

    def __init__(
        self,
        features: np.ndarray,      # (N_total, n_features)
        valid_indices: np.ndarray,  # Indices from Step 3.1
        labels: np.ndarray,         # {-1, 0, +1}
        context_len: int = 512,
    ):
        if len(valid_indices) != len(labels):
            raise ValueError(
                f"valid_indices length ({len(valid_indices)}) != labels length ({len(labels)})"
            )

        self.features = features
        self.context_len = context_len

        # Filter: valid_indices must have enough context
        # valid_indices 중 context window를 만들 수 있는 것만 필터
        mask = valid_indices >= context_len
        self.usable_indices = valid_indices[mask]
        self.usable_labels = labels[mask]

        if len(self.usable_indices) == 0:
            raise ValueError(
                f"No usable samples after filtering (context_len={context_len})"
            )

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            x: (n_features, context_len) tensor - Channel-first format
            y: Long tensor - class index {0, 1, 2}
        """
        target_idx = self.usable_indices[idx]

        # 연속 context_len개 바 (항상 연속, Drop 없음)
        window = self.features[target_idx - self.context_len:target_idx]
        label = self.usable_labels[idx]

        # (context_len, n_features) → (n_features, context_len) Channel-first
        x = torch.tensor(window.T, dtype=torch.float32)

        # {-1, 0, +1} → {0, 1, 2}
        y = torch.tensor(label + 1, dtype=torch.long)

        return x, y

    def __len__(self) -> int:
        return len(self.usable_indices)

    @property
    def n_features(self) -> int:
        """Number of input features."""
        return self.features.shape[1]

    @property
    def n_samples(self) -> int:
        """Number of usable samples."""
        return len(self.usable_indices)

    def get_class_distribution(self) -> dict:
        """Get class distribution for the dataset."""
        unique, counts = np.unique(self.usable_labels, return_counts=True)
        total = len(self.usable_labels)
        return {
            int(label): {
                "count": int(count),
                "ratio": count / total
            }
            for label, count in zip(unique, counts)
        }

    def get_class_weights(self, max_weight: float = 10.0) -> torch.Tensor:
        """
        Compute inverse frequency class weights for loss balancing.

        Args:
            max_weight: Maximum allowed weight to prevent extreme values

        Returns:
            Tensor of shape (3,) with class weights
        """
        labels_shifted = self.usable_labels + 1  # {-1,0,1} → {0,1,2}
        counts = np.bincount(labels_shifted, minlength=3)
        total = len(labels_shifted)

        # Inverse frequency weighting
        weights = total / (3 * counts + 1e-10)

        # Cap weights to prevent extreme values for rare classes
        weights = np.clip(weights, 0.1, max_weight)

        return torch.tensor(weights, dtype=torch.float32)
