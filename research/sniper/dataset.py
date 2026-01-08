"""
Sniper Dataset (v4.9)

Dataset with WeightedRandomSampler for balanced training.

Key insight:
- Raw data may be 10:10:80 (Long:Short:Neutral)
- Training batches should be 33:33:33
- WeightedRandomSampler achieves this without oversampling data
"""

from typing import Optional, Tuple
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

logger = logging.getLogger(__name__)


class SniperDataset(Dataset):
    """
    Dataset for Sniper classification.

    Features:
    - Context windows of configurable length
    - Label conversion: {-1, 0, +1} -> {0, 1, 2}
    - Balanced sampling via WeightedRandomSampler
    """

    def __init__(
        self,
        features: np.ndarray,    # (N, n_features)
        labels: np.ndarray,      # (N,) with values {-1, 0, +1} or NaN
        context_len: int = 128,
    ):
        """
        Args:
            features: Feature array (N, n_features)
            labels: Label array (N,) with {-1, 0, +1, NaN}
            context_len: Context window length
        """
        self.features = features.astype(np.float32)
        self.labels = labels
        self.context_len = context_len

        # Find valid indices (have label AND enough context)
        has_label = ~np.isnan(labels)
        has_context = np.arange(len(labels)) >= context_len

        valid_mask = has_label & has_context
        self.valid_indices = np.where(valid_mask)[0]

        if len(self.valid_indices) == 0:
            raise ValueError("No valid samples after filtering")

        # Get labels for valid indices
        self.valid_labels = labels[self.valid_indices].astype(np.int8)

        # Log distribution
        self._log_distribution()

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.

        Returns:
            x: (n_features, context_len) - channel first
            y: scalar long tensor - class index {0, 1, 2}
        """
        target_idx = self.valid_indices[idx]

        # Get context window
        start_idx = target_idx - self.context_len
        window = self.features[start_idx:target_idx]

        # Get label
        label = self.valid_labels[idx]

        # Convert to tensors
        # (context_len, n_features) -> (n_features, context_len)
        x = torch.from_numpy(window.T.copy())

        # {-1, 0, +1} -> {0, 1, 2}
        y = torch.tensor(label + 1, dtype=torch.long)

        return x, y

    def _log_distribution(self):
        """Log class distribution."""
        unique, counts = np.unique(self.valid_labels, return_counts=True)
        total = len(self.valid_labels)

        logger.info(f"Dataset: {total} samples")
        for label, count in zip(unique, counts):
            name = {-1: "Short", 0: "Neutral", 1: "Long"}[label]
            logger.info(f"  {name}: {count} ({count/total:.1%})")

    def get_class_counts(self) -> np.ndarray:
        """Get counts for each class {0, 1, 2}."""
        labels_shifted = self.valid_labels + 1  # {-1,0,1} -> {0,1,2}
        return np.bincount(labels_shifted, minlength=3)

    def get_class_weights(self) -> torch.Tensor:
        """
        Get inverse frequency class weights.

        Note: Use this for loss weighting, NOT with sampler.
        """
        counts = self.get_class_counts()
        total = len(self.valid_labels)
        weights = total / (3 * counts + 1e-10)
        return torch.from_numpy(weights).float()

    def get_balanced_sampler(self) -> WeightedRandomSampler:
        """
        Create WeightedRandomSampler for 1:1:1 class balance.

        This ensures each batch has roughly equal representation
        of all classes, even if raw data is imbalanced.

        Returns:
            WeightedRandomSampler instance
        """
        # Shift labels to {0, 1, 2}
        labels_shifted = self.valid_labels + 1

        # Count per class
        class_counts = np.bincount(labels_shifted, minlength=3)
        n_samples = len(labels_shifted)

        # Weight = 1 / class_count
        # Rare classes (Long/Short) get higher weight
        class_weights = n_samples / (class_counts + 1e-10)

        # Assign weight to each sample based on its class
        sample_weights = class_weights[labels_shifted]

        logger.info(f"Sampler class weights: {class_weights}")
        logger.info(f"  Short weight: {class_weights[0]:.2f}")
        logger.info(f"  Neutral weight: {class_weights[1]:.2f}")
        logger.info(f"  Long weight: {class_weights[2]:.2f}")

        return WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(sample_weights),
            replacement=True,  # Allow resampling minority classes
        )

    @property
    def n_features(self) -> int:
        return self.features.shape[1]


def create_balanced_dataloader(
    dataset: SniperDataset,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create DataLoader with balanced sampling for TRAINING.

    Uses WeightedRandomSampler to ensure 1:1:1 class balance
    in each batch.

    Args:
        dataset: SniperDataset instance
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for GPU

    Returns:
        DataLoader with balanced sampling
    """
    sampler = dataset.get_balanced_sampler()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use sampler instead of shuffle
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Consistent batch sizes
    )


def create_validation_dataloader(
    dataset: SniperDataset,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create DataLoader for VALIDATION.

    NO balancing - use original distribution to measure true performance.

    Args:
        dataset: SniperDataset instance
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for GPU

    Returns:
        DataLoader without sampling
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for reproducibility
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def train_val_split(
    features: np.ndarray,
    labels: np.ndarray,
    context_len: int = 128,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[SniperDataset, SniperDataset]:
    """
    Split data into train and validation datasets.

    Uses temporal split (not random) to avoid look-ahead bias.

    Args:
        features: Feature array
        labels: Label array
        context_len: Context window length
        val_ratio: Validation ratio
        seed: Random seed (not used - temporal split)

    Returns:
        (train_dataset, val_dataset)
    """
    n_samples = len(features)
    split_idx = int(n_samples * (1 - val_ratio))

    # Temporal split
    train_features = features[:split_idx]
    train_labels = labels[:split_idx]
    val_features = features[split_idx:]
    val_labels = labels[split_idx:]

    train_dataset = SniperDataset(train_features, train_labels, context_len)
    val_dataset = SniperDataset(val_features, val_labels, context_len)

    logger.info(f"Train/Val split: {len(train_dataset)}/{len(val_dataset)}")

    return train_dataset, val_dataset
