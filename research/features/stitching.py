"""
Adaptive Stitching for Cross-File Feature Continuity.

Handles monthly Parquet file boundaries by:
- Maintaining a buffer of recent bars for rolling calculations
- Using EWM (Exponential Weighted Moving) for adaptive decay
- Providing Z-score normalization with regime adaptation
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict


class AdaptiveStitcher:
    """
    Maintains continuity across file boundaries for rolling statistics.

    Key features:
        - Buffer-based stitching: Keeps last N rows for seamless rolling calcs
        - Adaptive halflife: Uses EWM for natural decay
        - Z-score normalization: Standardizes features for model input
    """

    def __init__(
        self,
        buffer_size: int = 2000,
        base_halflife: int = 50,
        zscore_columns: Optional[List[str]] = None,
    ):
        """
        Args:
            buffer_size: Number of rows to keep in buffer
            base_halflife: Base halflife for EWM calculations
            zscore_columns: Columns to apply Z-score normalization.
                           If None, defaults to ['close']
        """
        self.buffer_size = buffer_size
        self.base_halflife = base_halflife
        self.zscore_columns = zscore_columns or ['close']

        # State
        self.buffer = pd.DataFrame()
        self._ewm_state: Dict[str, Dict] = {}  # Store EWM state per column

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a new dataframe chunk with adaptive stitching.

        Args:
            df: New chunk of data

        Returns:
            DataFrame with Z-score columns added (only the new rows, not buffer)
        """
        # Stitch: Prepend buffer if available
        if not self.buffer.empty:
            full_df = pd.concat([self.buffer, df], axis=0, ignore_index=True)
        else:
            full_df = df.copy()

        result = df.copy()

        # Apply Z-score normalization to each specified column
        for col in self.zscore_columns:
            if col not in full_df.columns:
                continue

            zscore_col = f"{col}_zscore"
            zscore_values = self._calc_zscore(full_df[col])

            # Extract only the new rows (not buffer)
            start_idx = len(full_df) - len(df)
            result[zscore_col] = zscore_values.iloc[start_idx:].values

        # Update buffer (keep last N rows of original data)
        self.buffer = df.iloc[-self.buffer_size:].copy() if len(df) >= self.buffer_size else df.copy()

        return result

    def _calc_zscore(self, series: pd.Series) -> pd.Series:
        """
        Calculate EWM-based Z-score.

        Args:
            series: Input series

        Returns:
            Z-score series
        """
        ewm_mean = series.ewm(halflife=self.base_halflife).mean()
        ewm_std = series.ewm(halflife=self.base_halflife).std()

        # Prevent division by zero
        zscore = (series - ewm_mean) / (ewm_std + 1e-10)

        return zscore

    def process_multi(
        self,
        df: pd.DataFrame,
        columns: List[str],
    ) -> pd.DataFrame:
        """
        Process multiple columns at once.

        Args:
            df: Input DataFrame
            columns: List of column names to Z-score normalize

        Returns:
            DataFrame with Z-score columns added
        """
        # Temporarily update zscore_columns
        original_cols = self.zscore_columns
        self.zscore_columns = columns

        result = self.process(df)

        # Restore
        self.zscore_columns = original_cols

        return result

    def reset(self):
        """Reset the stitcher state."""
        self.buffer = pd.DataFrame()
        self._ewm_state = {}

    def get_buffer(self) -> pd.DataFrame:
        """Get current buffer contents."""
        return self.buffer.copy()

    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)


class MultiFeatureStitcher:
    """
    Stitcher optimized for multiple feature columns.

    Creates individual Z-score columns for each feature while
    maintaining a single shared buffer.
    """

    def __init__(
        self,
        buffer_size: int = 2000,
        halflife_map: Optional[Dict[str, int]] = None,
        default_halflife: int = 50,
    ):
        """
        Args:
            buffer_size: Number of rows to keep in buffer
            halflife_map: Dict mapping column names to their halflife values
            default_halflife: Default halflife for columns not in map
        """
        self.buffer_size = buffer_size
        self.halflife_map = halflife_map or {}
        self.default_halflife = default_halflife
        self.buffer = pd.DataFrame()

    def process(
        self,
        df: pd.DataFrame,
        columns: List[str],
    ) -> pd.DataFrame:
        """
        Process multiple columns with stitching.

        Args:
            df: Input DataFrame
            columns: List of columns to process

        Returns:
            DataFrame with Z-score columns added
        """
        # Validate columns exist
        valid_columns = [c for c in columns if c in df.columns]

        if not valid_columns:
            return df

        # Stitch with buffer
        if not self.buffer.empty:
            # Only keep columns that exist in both
            common_cols = list(set(self.buffer.columns) & set(df.columns))
            full_df = pd.concat(
                [self.buffer[common_cols], df[common_cols]],
                axis=0,
                ignore_index=True
            )
        else:
            full_df = df.copy()

        result = df.copy()

        # Process each column
        for col in valid_columns:
            halflife = self.halflife_map.get(col, self.default_halflife)

            # Calculate EWM stats on full (stitched) data
            ewm_mean = full_df[col].ewm(halflife=halflife).mean()
            ewm_std = full_df[col].ewm(halflife=halflife).std()

            zscore = (full_df[col] - ewm_mean) / (ewm_std + 1e-10)

            # Extract only new rows
            start_idx = len(full_df) - len(df)
            result[f"{col}_zscore"] = zscore.iloc[start_idx:].values

        # Update buffer
        self.buffer = df.iloc[-self.buffer_size:].copy()

        return result

    def reset(self):
        """Reset state."""
        self.buffer = pd.DataFrame()
