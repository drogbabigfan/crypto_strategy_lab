"""
Feature Engineering Pipeline.

Main entry point for generating features from Dollar Bar data.
Orchestrates L1, Regime, Stationarity, and Stitching modules.
"""

import os
import glob
import hashlib
import logging
from typing import Optional, List
from pathlib import Path

import yaml
import pandas as pd
import numpy as np

from .l1 import L1FeatureGenerator
from .regime import RegimeFeatureGenerator
from .stationarity import StationarityEngine
from .stitching import AdaptiveStitcher
from .advanced import AdvancedFeatureGenerator
from .kalman import KalmanFeatureGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Feature engineering pipeline for Dollar Bar data.

    Pipeline stages:
        1. Load Dollar Bar Parquet files
        2. Generate L1 features (log transforms, imbalance, VWAP)
        3. Generate Regime features (Parkinson vol, entropy)
        4. Generate Advanced features (autocorr, hurst, toxicity, RSI, momentum)
        5. Apply Stationarity transforms (FracDiff, Detrend)
        6. Apply Stitching for cross-file continuity
        7. Save to output Parquet
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary (from config.yaml)
        """
        self.config = config

        # Initialize generators
        self.l1_gen = L1FeatureGenerator()

        regime_cfg = config.get('features', {}).get('regime', {})
        self.regime_gen = RegimeFeatureGenerator(
            vol_window=regime_cfg.get('parkinson_window', 24),
            entropy_window=regime_cfg.get('entropy_window', 24),
            entropy_bins=regime_cfg.get('entropy_bins', 10),
        )

        self.stat_engine = StationarityEngine()

        # Advanced features
        advanced_cfg = config.get('features', {}).get('advanced', {})
        self.advanced_gen = AdvancedFeatureGenerator(
            autocorr_window=advanced_cfg.get('autocorr_window', 20),
            autocorr_lag=advanced_cfg.get('autocorr_lag', 1),
            hurst_window=advanced_cfg.get('hurst_window', 100),
            toxicity_window=advanced_cfg.get('toxicity_window', 50),
            rsi_window=advanced_cfg.get('rsi_window', 14),
            efficiency_window=advanced_cfg.get('efficiency_window', 10),
            momentum_windows=advanced_cfg.get('momentum_windows', [10, 50, 250, 1000]),
        )

        stitch_cfg = config.get('features', {}).get('stitching', {})
        self.stitcher = AdaptiveStitcher(
            buffer_size=stitch_cfg.get('buffer_size', 2000),
            base_halflife=stitch_cfg.get('base_halflife', 50),
        )

        # Kalman Filter
        kalman_cfg = config.get('features', {}).get('kalman', {})
        self.kalman_gen = KalmanFeatureGenerator(
            r_window=kalman_cfg.get('r_window', 20),
            q_window=kalman_cfg.get('q_window', 20),
            use_typical_price=kalman_cfg.get('use_typical_price', True),
            q_scale=kalman_cfg.get('q_scale', 0.1),
            r_scale=kalman_cfg.get('r_scale', 1.0),
        )

        # Stationarity config
        stat_cfg = config.get('features', {}).get('stationarity', {})
        self.frac_diff_d = stat_cfg.get('frac_diff_d', 0.4)
        self.detrend_span = stat_cfg.get('detrend_ema_span', 100)

    def run(self, bar_files: Optional[List[str]] = None) -> str:
        """
        Run the feature pipeline.

        Args:
            bar_files: List of Parquet file paths. If None, glob from config.

        Returns:
            Path to output Parquet file
        """
        # 1. Get bar files
        if bar_files is None:
            bar_path = self.config.get('data', {}).get('bar_path', './etl/data/bars')
            pattern = os.path.join(bar_path, '**', '*.parquet')
            bar_files = sorted(glob.glob(pattern, recursive=True))

        if not bar_files:
            raise ValueError(f"No Parquet files found")

        logger.info(f"Processing {len(bar_files)} files")

        all_features = []

        # 2. Process each file
        for i, bar_file in enumerate(bar_files):
            logger.info(f"[{i+1}/{len(bar_files)}] Processing {os.path.basename(bar_file)}")

            try:
                df = pd.read_parquet(bar_file)
                df = self._process_chunk(df)
                all_features.append(df)
            except Exception as e:
                logger.error(f"Error processing {bar_file}: {e}")
                continue

        if not all_features:
            raise ValueError("No data processed successfully")

        # 3. Concatenate all features
        logger.info("Concatenating features...")
        result = pd.concat(all_features, ignore_index=True)

        # 4. Drop rows with NaN in critical columns
        critical_cols = ['frac_diff_close', 'detrended_log_price', 'parkinson_vol']
        existing_critical = [c for c in critical_cols if c in result.columns]
        initial_len = len(result)
        result = result.dropna(subset=existing_critical)
        logger.info(f"Dropped {initial_len - len(result)} rows with NaN in critical columns")

        # 5. Save output
        output_path = self._get_output_path()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(result)} rows to {output_path}")

        return output_path

    def _process_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk of data."""
        # L1 Features
        df = self.l1_gen.generate(df)

        # Regime Features
        df = self.regime_gen.generate(df)

        # Advanced Features (autocorr, hurst, toxicity, RSI, momentum)
        df = self.advanced_gen.generate(df)

        # Kalman Filter features
        df = self.kalman_gen.generate(df)

        # Stationarity: FracDiff on log(price)
        df['frac_diff_close'] = self.stat_engine.frac_diff_ffd(
            np.log(df['close']), d=self.frac_diff_d
        )

        # Stationarity: Detrended Log Price
        df['detrended_log_price'] = self.stat_engine.detrend_log_price(
            df['close'], window=self.detrend_span
        )

        # Stitching (updates internal buffer)
        df = self.stitcher.process(df)

        return df

    def _get_output_path(self) -> str:
        """Generate output path with config hash."""
        feature_output = self.config.get('data', {}).get('feature_output', './data/features')

        # Create hash from feature config
        feature_cfg = self.config.get('features', {})
        config_str = yaml.dump(feature_cfg, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        return os.path.join(feature_output, f"features_{config_hash}.parquet")

    def get_dual_input_columns(self) -> dict:
        """
        Get column names for dual-input model architecture.

        Returns:
            Dict with 'dynamics' and 'context' column lists
        """
        return {
            'dynamics': ['frac_diff_close'],  # Stationary, short-term patterns
            'context': ['detrended_log_price'],  # Trend context
        }

    def get_all_feature_columns(self) -> List[str]:
        """Get all generated feature column names."""
        return (
            self.l1_gen.get_feature_names() +
            self.regime_gen.get_feature_names() +
            self.advanced_gen.get_feature_names() +
            self.kalman_gen.get_feature_names() +
            ['frac_diff_close', 'detrended_log_price', 'close_zscore']
        )


def run_feature_pipeline(config_path: str) -> str:
    """
    Main entry point for feature pipeline.

    Args:
        config_path: Path to config.yaml

    Returns:
        Path to output Parquet file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    pipeline = FeaturePipeline(config)
    return pipeline.run()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run feature engineering pipeline')
    parser.add_argument(
        '--config',
        type=str,
        default='../../config/config.yaml',
        help='Path to config file'
    )
    args = parser.parse_args()

    output_path = run_feature_pipeline(args.config)
    print(f"Features saved to: {output_path}")
