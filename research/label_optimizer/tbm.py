"""
Triple Barrier Method (v4.6)

Labeling with Fee Trap Guard for filtering unprofitable samples.
"""

from dataclasses import dataclass
from typing import Optional
import math
import numpy as np
import pandas as pd

from .cost_model import SquareRootCostModel


@dataclass
class TBMConfig:
    """Triple Barrier configuration."""

    sl_mult: float  # Stop Loss σ multiplier
    pt_mult: float  # Profit Target σ multiplier
    vertical_bars: int  # Maximum holding period (bars)


@dataclass
class LabelResult:
    """Result of labeling operation."""

    valid_indices: np.ndarray  # Indices of valid samples
    labels: np.ndarray  # Labels for valid samples (-1, 0, +1)
    returns: np.ndarray  # Actual returns for valid samples
    n_filtered: int  # Number of samples filtered by Fee Trap


class TripleBarrierLabeler:
    """
    Triple Barrier Labeler with Fee Trap Guard.

    Labels:
        +1: Profit Target hit first (Long opportunity)
        -1: Stop Loss hit first (Short opportunity)
         0: Vertical Barrier hit (Timeout)

    Fee Trap Guard:
        Filters out samples where expected profit < cost threshold.
    """

    # Required columns in data
    REQUIRED_COLS = ["close", "high", "low", "realized_vol"]

    def __init__(
        self,
        config: TBMConfig,
        cost_model: Optional[SquareRootCostModel] = None,
        vol_column: str = "realized_vol",
        volume_lookback: int = 100,
    ):
        """
        Initialize labeler.

        Args:
            config: TBM configuration
            cost_model: Cost model for Fee Trap Guard (None to disable)
            vol_column: Column name for volatility
            volume_lookback: Lookback for average volume calculation
        """
        self.config = config
        self.cost_model = cost_model
        self.vol_column = vol_column
        self.volume_lookback = volume_lookback

    def _check_data(self, data: pd.DataFrame) -> None:
        """Validate input data has required columns."""
        missing = set(self.REQUIRED_COLS) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _get_volume_ratio(
        self, data: pd.DataFrame, idx: int, volume_col: str = "log_volume"
    ) -> float:
        """Calculate volume ratio for cost model."""
        if volume_col not in data.columns:
            return 1.0

        start_idx = max(0, idx - self.volume_lookback)
        if start_idx == idx:
            return 1.0

        avg_volume = data[volume_col].iloc[start_idx:idx].mean()
        current_volume = data[volume_col].iloc[idx]

        if avg_volume <= 0:
            return 1.0

        return current_volume / avg_volume

    def label(self, data: pd.DataFrame) -> LabelResult:
        """
        Apply Triple Barrier labeling with Fee Trap Guard.

        Args:
            data: DataFrame with OHLC and volatility

        Returns:
            LabelResult with valid indices, labels, and statistics
        """
        self._check_data(data)

        valid_indices = []
        labels = []
        returns = []
        n_filtered = 0

        n_samples = len(data) - self.config.vertical_bars

        for i in range(n_samples):
            row = data.iloc[i]
            sigma = row[self.vol_column]

            # Skip if volatility is invalid
            if pd.isna(sigma) or sigma <= 0:
                continue

            entry_price = row["close"]

            # Fee Trap Check
            if self.cost_model is not None:
                volume_ratio = self._get_volume_ratio(data, i)
                if not self.cost_model.is_viable(
                    pt_mult=self.config.pt_mult,
                    sigma=sigma,
                    volume_ratio=volume_ratio,
                ):
                    n_filtered += 1
                    continue

            # Calculate barriers with time-scaling
            # σ_T = σ_1 * sqrt(T) for holding period T bars
            scaled_sigma = sigma * math.sqrt(self.config.vertical_bars)
            pt_level = entry_price * (1 + self.config.pt_mult * scaled_sigma)
            sl_level = entry_price * (1 - self.config.sl_mult * scaled_sigma)

            # Forward path (next bars)
            path_start = i + 1
            path_end = i + 1 + self.config.vertical_bars
            path = data.iloc[path_start:path_end]

            # Find first barrier hit
            label, exit_price = self._find_first_hit(
                path, pt_level, sl_level, entry_price
            )

            # Calculate return
            ret = (exit_price - entry_price) / entry_price

            valid_indices.append(i)
            labels.append(label)
            returns.append(ret)

        return LabelResult(
            valid_indices=np.array(valid_indices),
            labels=np.array(labels),
            returns=np.array(returns),
            n_filtered=n_filtered,
        )

    def _find_first_hit(
        self,
        path: pd.DataFrame,
        pt_level: float,
        sl_level: float,
        entry_price: float,
    ) -> tuple[int, float]:
        """
        Find which barrier is hit first.

        Returns:
            (label, exit_price)
        """
        pt_bar = None
        sl_bar = None

        for bar_idx, (_, bar) in enumerate(path.iterrows()):
            # Check PT hit (high crosses above)
            if pt_bar is None and bar["high"] >= pt_level:
                pt_bar = bar_idx

            # Check SL hit (low crosses below)
            if sl_bar is None and bar["low"] <= sl_level:
                sl_bar = bar_idx

            # Early exit if both found
            if pt_bar is not None and sl_bar is not None:
                break

        # Determine which hit first
        if pt_bar is not None and sl_bar is not None:
            if pt_bar < sl_bar:
                return 1, pt_level
            elif sl_bar < pt_bar:
                return -1, sl_level
            else:
                # Same bar - use close direction
                last_close = path.iloc[pt_bar]["close"]
                if last_close >= entry_price:
                    return 1, pt_level
                else:
                    return -1, sl_level
        elif pt_bar is not None:
            return 1, pt_level
        elif sl_bar is not None:
            return -1, sl_level
        else:
            # Vertical barrier (timeout)
            exit_price = path.iloc[-1]["close"] if len(path) > 0 else entry_price
            return 0, exit_price

    def get_label_distribution(self, labels: np.ndarray) -> dict:
        """Get distribution of labels."""
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)

        dist = {}
        for label, count in zip(unique, counts):
            label_name = {-1: "short", 0: "neutral", 1: "long"}.get(label, str(label))
            dist[label_name] = {"count": int(count), "pct": count / total * 100}

        return dist
