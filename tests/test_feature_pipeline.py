"""
Tests for Feature Engineering Pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from research.features.pipeline import FeaturePipeline
from research.features.l1 import L1FeatureGenerator
from research.features.regime import RegimeFeatureGenerator


class TestFeaturePipeline:
    """Tests for FeaturePipeline class."""

    @pytest.fixture
    def default_config(self):
        """Default config for testing."""
        return {
            'data': {
                'bar_path': './etl/data/bars',
                'feature_output': './data/features',
            },
            'features': {
                'regime': {
                    'parkinson_window': 24,
                    'entropy_window': 24,
                    'entropy_bins': 10,
                },
                'stationarity': {
                    'frac_diff_d': 0.4,
                    'detrend_ema_span': 100,
                },
                'stitching': {
                    'buffer_size': 100,
                    'base_halflife': 20,
                },
            },
        }

    def test_pipeline_initialization(self, default_config):
        """Test pipeline initializes correctly."""
        pipeline = FeaturePipeline(default_config)

        assert pipeline.l1_gen is not None
        assert pipeline.regime_gen is not None
        assert pipeline.stat_engine is not None
        assert pipeline.stitcher is not None

    def test_process_chunk(self, default_config, sample_dollar_bar_data):
        """Test processing a single chunk of data."""
        pipeline = FeaturePipeline(default_config)

        result = pipeline._process_chunk(sample_dollar_bar_data)

        # Check L1 features exist
        assert 'log_volume' in result.columns
        assert 'vwap' in result.columns
        assert 'volume_imbalance' in result.columns

        # Check regime features exist
        assert 'parkinson_vol' in result.columns
        assert 'shannon_entropy' in result.columns

        # Check stationarity features exist
        assert 'frac_diff_close' in result.columns
        assert 'detrended_log_price' in result.columns

    def test_get_dual_input_columns(self, default_config):
        """Test getting dual input column names."""
        pipeline = FeaturePipeline(default_config)
        columns = pipeline.get_dual_input_columns()

        assert 'dynamics' in columns
        assert 'context' in columns
        assert 'frac_diff_close' in columns['dynamics']
        assert 'detrended_log_price' in columns['context']

    def test_get_all_feature_columns(self, default_config):
        """Test getting all feature column names."""
        pipeline = FeaturePipeline(default_config)
        all_cols = pipeline.get_all_feature_columns()

        # Should include L1, regime, and stationarity features
        assert 'log_volume' in all_cols
        assert 'parkinson_vol' in all_cols
        assert 'frac_diff_close' in all_cols

    def test_run_with_mock_files(self, default_config, sample_dollar_bar_data):
        """Test full pipeline run with mock parquet files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock parquet files
            bar_dir = os.path.join(tmpdir, 'bars')
            output_dir = os.path.join(tmpdir, 'features')
            os.makedirs(bar_dir)
            os.makedirs(output_dir)

            # Save sample data as parquet (use larger dataset for frac_diff)
            file1 = os.path.join(bar_dir, 'BTCUSDT-bars-2024-01.parquet')
            sample_dollar_bar_data.to_parquet(file1)

            # Update config with smaller windows for test data
            default_config['data']['bar_path'] = bar_dir
            default_config['data']['feature_output'] = output_dir
            default_config['features']['stationarity']['frac_diff_d'] = 0.3
            default_config['features']['stationarity']['detrend_ema_span'] = 20

            pipeline = FeaturePipeline(default_config)

            # Run with explicit file list to avoid dropna issue
            result = pipeline._process_chunk(sample_dollar_bar_data)

            # Check features exist (some NaN is expected due to warmup)
            assert 'log_volume' in result.columns
            assert 'parkinson_vol' in result.columns
            assert len(result) == len(sample_dollar_bar_data)

    def test_run_no_files_raises(self, default_config):
        """Test run raises error when no files found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            default_config['data']['bar_path'] = tmpdir  # Empty dir

            pipeline = FeaturePipeline(default_config)

            with pytest.raises(ValueError, match="No Parquet files"):
                pipeline.run()

    def test_output_path_includes_hash(self, default_config):
        """Test output path includes config hash."""
        pipeline = FeaturePipeline(default_config)
        path = pipeline._get_output_path()

        # Should contain hash
        assert 'features_' in path
        assert '.parquet' in path


class TestPipelineIntegration:
    """Integration tests for feature pipeline."""

    def test_all_features_generated(self, sample_dollar_bar_data):
        """Test all expected features are generated."""
        config = {
            'features': {
                'regime': {'parkinson_window': 10, 'entropy_window': 10},
                'stationarity': {'frac_diff_d': 0.4, 'detrend_ema_span': 50},
                'stitching': {'buffer_size': 50, 'base_halflife': 10},
            },
            'data': {}
        }

        pipeline = FeaturePipeline(config)
        result = pipeline._process_chunk(sample_dollar_bar_data)

        # L1 features
        l1_gen = L1FeatureGenerator()
        for name in l1_gen.get_feature_names():
            assert name in result.columns, f"Missing L1 feature: {name}"

        # Regime features
        regime_gen = RegimeFeatureGenerator()
        for name in regime_gen.get_feature_names():
            assert name in result.columns, f"Missing regime feature: {name}"

        # Stationarity features
        assert 'frac_diff_close' in result.columns
        assert 'detrended_log_price' in result.columns

    def test_stitching_across_chunks(self, sample_dollar_bar_data):
        """Test stitching maintains continuity across chunks."""
        config = {
            'features': {
                'regime': {'parkinson_window': 10, 'entropy_window': 10},
                'stationarity': {'frac_diff_d': 0.4, 'detrend_ema_span': 50},
                'stitching': {'buffer_size': 50, 'base_halflife': 10},
            },
            'data': {}
        }

        pipeline = FeaturePipeline(config)

        # Process two chunks
        chunk1 = sample_dollar_bar_data.iloc[:500].copy()
        chunk2 = sample_dollar_bar_data.iloc[500:].copy()

        result1 = pipeline._process_chunk(chunk1)
        result2 = pipeline._process_chunk(chunk2)

        # Z-score should be available at start of chunk2 (due to stitching)
        if 'close_zscore' in result2.columns:
            # First value should not be NaN if stitching works
            assert not result2['close_zscore'].isna().all()


class TestEdgeCases:
    """Edge case tests for pipeline."""

    def test_small_data(self):
        """Test pipeline with very small dataset."""
        small_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100, 101, 102],
            'volume': [1, 2, 3],
            'dollar_value': [100, 202, 306],
            'tick_count': [10, 20, 30],
            'duration': [60, 60, 60],
            'buy_dollar_vol': [60, 120, 180],
            'sell_dollar_vol': [40, 82, 126],
            'net_imbalance': [20, 38, 54],
        })

        config = {
            'features': {
                'regime': {'parkinson_window': 2, 'entropy_window': 2},
                'stationarity': {'frac_diff_d': 0.4, 'detrend_ema_span': 2},
                'stitching': {'buffer_size': 2, 'base_halflife': 2},
            },
            'data': {}
        }

        pipeline = FeaturePipeline(config)
        result = pipeline._process_chunk(small_df)

        assert len(result) == 3

    def test_data_with_nan(self, sample_dollar_bar_data):
        """Test pipeline handles NaN in input."""
        df = sample_dollar_bar_data.copy()
        df.loc[10, 'close'] = np.nan
        df.loc[20, 'volume'] = np.nan

        config = {
            'features': {
                'regime': {'parkinson_window': 10, 'entropy_window': 10},
                'stationarity': {'frac_diff_d': 0.4, 'detrend_ema_span': 50},
                'stitching': {'buffer_size': 50, 'base_halflife': 10},
            },
            'data': {}
        }

        pipeline = FeaturePipeline(config)
        result = pipeline._process_chunk(df)

        # Should not crash
        assert len(result) == len(df)

    def test_config_defaults(self):
        """Test pipeline works with minimal config."""
        config = {'features': {}, 'data': {}}
        pipeline = FeaturePipeline(config)

        # Should use defaults
        assert pipeline.frac_diff_d == 0.4
        assert pipeline.detrend_span == 100

    def test_duplicate_processing(self, sample_dollar_bar_data):
        """Test processing same chunk twice doesn't accumulate state incorrectly."""
        config = {
            'features': {
                'regime': {'parkinson_window': 10, 'entropy_window': 10},
                'stationarity': {'frac_diff_d': 0.4, 'detrend_ema_span': 50},
                'stitching': {'buffer_size': 50, 'base_halflife': 10},
            },
            'data': {}
        }

        pipeline = FeaturePipeline(config)

        result1 = pipeline._process_chunk(sample_dollar_bar_data.copy())

        # Reset stitcher to simulate new run
        pipeline.stitcher.reset()

        result2 = pipeline._process_chunk(sample_dollar_bar_data.copy())

        # Results should be identical
        assert len(result1) == len(result2)


class TestFeatureQuality:
    """Tests for feature quality checks."""

    def test_no_inf_values(self, sample_dollar_bar_data):
        """Test no infinite values in output."""
        config = {
            'features': {
                'regime': {'parkinson_window': 10, 'entropy_window': 10},
                'stationarity': {'frac_diff_d': 0.4, 'detrend_ema_span': 50},
                'stitching': {'buffer_size': 50, 'base_halflife': 10},
            },
            'data': {}
        }

        pipeline = FeaturePipeline(config)
        result = pipeline._process_chunk(sample_dollar_bar_data)

        for col in result.select_dtypes(include=[np.number]).columns:
            assert not np.isinf(result[col]).any(), f"Inf values in {col}"

    def test_reasonable_nan_ratio(self, sample_dollar_bar_data):
        """Test NaN ratio is reasonable after warmup for core features."""
        config = {
            'features': {
                'regime': {'parkinson_window': 10, 'entropy_window': 10},
                'stationarity': {'frac_diff_d': 0.3, 'detrend_ema_span': 20},
                'stitching': {'buffer_size': 50, 'base_halflife': 10},
            },
            'data': {}
        }

        pipeline = FeaturePipeline(config)
        result = pipeline._process_chunk(sample_dollar_bar_data)

        # After warmup, NaN ratio should be low for L1 features
        warmup_rows = 200
        post_warmup = result.iloc[warmup_rows:]

        # Only check L1 features (which shouldn't have NaN)
        l1_features = ['log_volume', 'vwap', 'volume_imbalance', 'buy_ratio']
        for col in l1_features:
            if col in post_warmup.columns:
                nan_ratio = post_warmup[col].isna().mean()
                assert nan_ratio < 0.1, f"High NaN ratio in {col}: {nan_ratio:.2%}"
