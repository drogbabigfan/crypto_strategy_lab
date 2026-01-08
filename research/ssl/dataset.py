"""
SSL Dataset (v4.6)

Patch-wise Random Masking for Self-Supervised Learning.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class SSLDataConfig:
    """SSL Pre-training data configuration."""

    assets: list[str] = field(
        default_factory=lambda: [
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "XRPUSDT",
            "DOGEUSDT",
            "BNBUSDT",
            "LTCUSDT",
        ]
    )
    context_len: int = 512
    min_samples_per_asset: int = 10000

    # Shuffle across assets (time order ignored for SSL)
    shuffle: bool = True
    random_seed: int = 42

    # Feature columns (excluding is_primed)
    feature_cols: list[str] = field(
        default_factory=lambda: [
            "log_volume",
            "log_tick_count",
            "log_duration",
            "log_trade_intensity",
            "vwap_deviation",
            "volume_imbalance",
            "bar_range",
            "bar_body",
            "garman_klass_vol",
            "realized_vol",
            "shannon_entropy",
            "vol_ratio",
            "vol_zscore",
            "entropy_zscore",
            "skewness",
            "kurtosis",
            "frac_diff_close",
            "detrended_log_price",
            "returns",
            "vw_momentum",
            "momentum_zscore_10",
            "momentum_zscore_50",
            "momentum_zscore_250",
            "momentum_zscore_1000",
            "connors_rsi",
            "sin_time",
            "cos_time",
            "sin_week",
            "cos_week",
        ]
    )


def load_features_for_asset(
    features_dir: str,
    asset: str,
    feature_cols: list[str],
) -> Optional[np.ndarray]:
    """
    Load feature parquet files for a single asset.

    Args:
        features_dir: Base directory for features
        asset: Asset symbol (e.g., "BTCUSDT")
        feature_cols: Feature columns to extract

    Returns:
        numpy array (N, n_features) or None if not enough data
    """
    asset_dir = Path(features_dir) / asset

    if not asset_dir.exists():
        logger.warning(f"Asset directory not found: {asset_dir}")
        return None

    # Find all parquet files
    parquet_files = sorted(asset_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning(f"No parquet files found for {asset}")
        return None

    # Load and concatenate
    dfs = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            # Filter primed rows only
            if "is_primed" in df.columns:
                df = df[df["is_primed"] == True]
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {pf}: {e}")

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)

    # Select feature columns (use available ones)
    available_cols = [c for c in feature_cols if c in combined.columns]
    if len(available_cols) < len(feature_cols):
        missing = set(feature_cols) - set(available_cols)
        logger.warning(f"Missing columns for {asset}: {missing}")

    if not available_cols:
        return None

    return combined[available_cols].values.astype(np.float32)


def load_ssl_data(
    features_dir: str,
    config: SSLDataConfig,
) -> np.ndarray:
    """
    Load features from all assets and create sliding windows.

    Each asset is independently Z-Score normalized by Go ETL,
    so no additional normalization is needed.

    Args:
        features_dir: Base directory for features (e.g., "data/features/futures")
        config: SSL data configuration

    Returns:
        numpy array (N_windows, context_len, n_features)
    """
    all_windows = []

    for asset in config.assets:
        logger.info(f"Loading {asset}...")

        features = load_features_for_asset(
            features_dir,
            asset,
            config.feature_cols,
        )

        if features is None:
            logger.warning(f"Skipping {asset}: no data")
            continue

        if len(features) < config.min_samples_per_asset:
            logger.warning(
                f"Skipping {asset}: only {len(features)} samples "
                f"(need {config.min_samples_per_asset})"
            )
            continue

        # Handle NaN values
        nan_mask = np.isnan(features).any(axis=1)
        if nan_mask.sum() > 0:
            logger.warning(f"{asset}: dropping {nan_mask.sum()} NaN rows")
            features = features[~nan_mask]

        # Create sliding windows
        n_windows = len(features) - config.context_len + 1
        for i in range(n_windows):
            window = features[i : i + config.context_len]  # (512, n_features)
            all_windows.append(window)

        logger.info(f"{asset}: {n_windows} windows created")

    if not all_windows:
        raise ValueError("No valid windows created from any asset")

    all_windows = np.array(all_windows, dtype=np.float32)
    logger.info(f"Total windows: {len(all_windows)}")

    # Shuffle across all assets
    if config.shuffle:
        rng = np.random.RandomState(config.random_seed)
        rng.shuffle(all_windows)
        logger.info("Windows shuffled")

    return all_windows


class PatchMaskingDataset(Dataset):
    """
    Patch-wise Random Masking Dataset for SSL Pre-training.

    v4.6: Simple patch-based masking (40%).
    - Each sample gets independent random mask
    - Zero masking (learnable token is an alternative)
    """

    def __init__(
        self,
        data: np.ndarray,  # (N, context_len, n_features)
        patch_len: int = 16,
        mask_ratio: float = 0.4,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            data: Feature windows (N, context_len, n_features)
            patch_len: Length of each patch
            mask_ratio: Fraction of patches to mask (0.4 = 40%)
            random_seed: Random seed for reproducibility (None for random)
        """
        self.data = data
        self.patch_len = patch_len
        self.mask_ratio = mask_ratio
        self.rng = np.random.RandomState(random_seed)

        # Compute number of patches
        self.context_len = data.shape[1]
        self.n_features = data.shape[2]
        self.n_patches = self.context_len // patch_len

        if self.context_len % patch_len != 0:
            raise ValueError(
                f"context_len ({self.context_len}) must be divisible "
                f"by patch_len ({patch_len})"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        # Original: (context_len, n_features)
        # Convert to channel-first: (n_features, context_len)
        x = self.data[idx].T.copy()  # (n_features, context_len)

        # Number of patches to mask
        n_mask = int(self.n_patches * self.mask_ratio)

        # Randomly select patches to mask (different per sample)
        mask_indices = self.rng.choice(self.n_patches, n_mask, replace=False)
        mask_indices = np.sort(mask_indices)  # Sort for consistency

        # Create masked version
        x_masked = x.copy()
        for patch_idx in mask_indices:
            start = patch_idx * self.patch_len
            end = start + self.patch_len
            x_masked[:, start:end] = 0  # Zero masking

        return {
            "input": torch.tensor(x_masked, dtype=torch.float32),
            "target": torch.tensor(x, dtype=torch.float32),
            "mask_indices": torch.tensor(mask_indices, dtype=torch.long),
        }

    @property
    def n_masked_patches(self) -> int:
        """Number of masked patches per sample."""
        return int(self.n_patches * self.mask_ratio)
