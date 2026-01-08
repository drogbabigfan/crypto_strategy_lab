"""
Sniper TBM Labeler (v4.9)

Tight labeling for high-confidence entries only.
Key changes from v4.8:
- Short timeout (12-24 bars instead of 100)
- Minimum expected return filter
- Target: Neutral 70-80% (strict labeling)
"""

from dataclasses import dataclass
from typing import Optional
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SniperTBMConfig:
    """Sniper Triple Barrier configuration."""

    # Barrier multipliers (volatility-based)
    sl_mult: float = 1.5          # Stop Loss = 1.5σ
    pt_mult: float = 2.0          # Profit Target = 2.0σ (asymmetric)

    # SHORT timeout - key change from v4.8
    max_hold_bars: int = 18       # ~half day for dollar bars

    # Minimum expected return filter
    min_expected_return: float = 0.0045  # 0.45% (fee 0.15% × 3)

    # Volatility column name
    vol_column: str = "realized_vol"


@dataclass
class LabelResult:
    """Result of labeling operation."""

    labels: np.ndarray           # (N,) with values {-1, 0, +1, NaN}
    valid_mask: np.ndarray       # (N,) boolean - True if label is valid
    stats: dict                  # Label statistics


class SniperLabeler:
    """
    Sniper TBM Labeler - strict labeling for high-confidence entries.

    Labels:
        +1 (Long): PT hit first within max_hold_bars
        -1 (Short): SL hit first within max_hold_bars
         0 (Neutral): Timeout OR low expected return

    Philosophy:
        "확실한 기회만 Long/Short, 나머지는 전부 Neutral"
    """

    REQUIRED_COLS = ["close", "high", "low", "high_time", "low_time"]

    def __init__(self, config: Optional[SniperTBMConfig] = None):
        self.config = config or SniperTBMConfig()

    def label(self, df: pd.DataFrame) -> LabelResult:
        """
        Apply Sniper TBM labeling.

        Args:
            df: DataFrame with OHLCV and volatility

        Returns:
            LabelResult with labels and statistics
        """
        self._validate_data(df)

        n_samples = len(df)
        labels = np.full(n_samples, np.nan)

        # Stats tracking
        stats = {
            "total": 0,
            "long": 0,
            "short": 0,
            "neutral_timeout": 0,
            "neutral_low_vol": 0,
            "skipped_no_future": 0,
            "skipped_invalid_vol": 0,
        }

        # Get volatility
        vol = df[self.config.vol_column].values
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # Check if high_time/low_time exist
        has_time_cols = "high_time" in df.columns and "low_time" in df.columns
        if has_time_cols:
            high_time = df["high_time"].values
            low_time = df["low_time"].values

        # Label each bar
        max_idx = n_samples - self.config.max_hold_bars - 1

        for i in range(max_idx):
            sigma = vol[i]

            # Skip invalid volatility
            if pd.isna(sigma) or sigma <= 0:
                stats["skipped_invalid_vol"] += 1
                continue

            stats["total"] += 1
            entry_price = close[i]

            # Filter 1: Minimum expected return check
            expected_return = self.config.pt_mult * sigma
            if expected_return < self.config.min_expected_return:
                labels[i] = 0  # Neutral - insufficient expected return
                stats["neutral_low_vol"] += 1
                continue

            # Calculate barriers
            pt_level = entry_price * (1 + self.config.pt_mult * sigma)
            sl_level = entry_price * (1 - self.config.sl_mult * sigma)

            # Scan forward
            label = self._find_first_hit(
                i, high, low, high_time if has_time_cols else None,
                low_time if has_time_cols else None,
                pt_level, sl_level
            )

            labels[i] = label

            # Update stats
            if label == 1:
                stats["long"] += 1
            elif label == -1:
                stats["short"] += 1
            else:
                stats["neutral_timeout"] += 1

        # Compute final stats
        valid_mask = ~np.isnan(labels)
        valid_labels = labels[valid_mask]

        if len(valid_labels) > 0:
            stats["long_ratio"] = stats["long"] / len(valid_labels)
            stats["short_ratio"] = stats["short"] / len(valid_labels)
            stats["neutral_ratio"] = (stats["neutral_timeout"] + stats["neutral_low_vol"]) / len(valid_labels)
        else:
            stats["long_ratio"] = 0
            stats["short_ratio"] = 0
            stats["neutral_ratio"] = 0

        # Log summary
        logger.info(f"Labeling complete:")
        logger.info(f"  Total valid: {stats['total']}")
        logger.info(f"  Long: {stats['long']} ({stats['long_ratio']:.1%})")
        logger.info(f"  Short: {stats['short']} ({stats['short_ratio']:.1%})")
        logger.info(f"  Neutral (timeout): {stats['neutral_timeout']}")
        logger.info(f"  Neutral (low vol): {stats['neutral_low_vol']}")
        logger.info(f"  Neutral ratio: {stats['neutral_ratio']:.1%}")

        return LabelResult(
            labels=labels,
            valid_mask=valid_mask,
            stats=stats,
        )

    def _find_first_hit(
        self,
        start_idx: int,
        high: np.ndarray,
        low: np.ndarray,
        high_time: Optional[np.ndarray],
        low_time: Optional[np.ndarray],
        pt_level: float,
        sl_level: float,
    ) -> int:
        """Find which barrier is hit first."""

        pt_bar = None
        sl_bar = None

        for j in range(1, self.config.max_hold_bars + 1):
            idx = start_idx + j

            # Check PT hit
            if pt_bar is None and high[idx] >= pt_level:
                pt_bar = j

            # Check SL hit
            if sl_bar is None and low[idx] <= sl_level:
                sl_bar = j

            # Early exit if both found
            if pt_bar is not None and sl_bar is not None:
                break

        # Determine label
        if pt_bar is not None and sl_bar is not None:
            if pt_bar < sl_bar:
                return 1  # Long
            elif sl_bar < pt_bar:
                return -1  # Short
            else:
                # Same bar - use high_time/low_time if available
                if high_time is not None and low_time is not None:
                    idx = start_idx + pt_bar
                    return 1 if high_time[idx] < low_time[idx] else -1
                else:
                    # Fallback: assume PT wins (bullish bias for crypto)
                    return 1
        elif pt_bar is not None:
            return 1  # Long
        elif sl_bar is not None:
            return -1  # Short
        else:
            return 0  # Neutral (timeout)

    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate input data."""
        missing = []
        for col in self.REQUIRED_COLS:
            if col not in df.columns:
                # high_time/low_time are optional
                if col not in ["high_time", "low_time"]:
                    missing.append(col)

        if self.config.vol_column not in df.columns:
            missing.append(self.config.vol_column)

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def get_class_counts(self, labels: np.ndarray) -> dict:
        """Get class counts from labels."""
        valid = labels[~np.isnan(labels)]
        unique, counts = np.unique(valid.astype(int), return_counts=True)
        return dict(zip(unique, counts))
