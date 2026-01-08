"""
Tests for Feature Validation Module.
"""

import pytest
import pandas as pd
import numpy as np
from research.features.validation import (
    adf_test,
    generate_adf_report,
    validate_features,
    print_validation_report,
)


class TestADFTest:
    """Tests for ADF stationarity test."""

    def test_adf_stationary_series(self):
        """Test ADF on stationary series (white noise)."""
        np.random.seed(42)
        stationary = pd.Series(np.random.randn(1000))

        result = adf_test(stationary)

        assert 'adf_statistic' in result
        assert 'p_value' in result
        assert 'is_stationary' in result

        # White noise should be stationary
        assert result['is_stationary'], "White noise should be stationary"
        assert result['p_value'] < 0.05

    def test_adf_non_stationary_series(self):
        """Test ADF on non-stationary series (random walk)."""
        np.random.seed(42)
        random_walk = pd.Series(np.cumsum(np.random.randn(1000)))

        result = adf_test(random_walk)

        # Random walk should NOT be stationary
        assert not result['is_stationary'], "Random walk should not be stationary"
        assert result['p_value'] > 0.05

    def test_adf_with_nan(self):
        """Test ADF handles NaN values."""
        series = pd.Series([1, 2, np.nan, 3, 4, np.nan, 5, 6, 7, 8] * 100)

        result = adf_test(series)

        # Should handle NaN without error
        assert 'error' in result or result['n_obs'] > 0

    def test_adf_insufficient_data(self):
        """Test ADF with too few data points."""
        series = pd.Series([1, 2, 3, 4, 5])  # Only 5 points

        result = adf_test(series)

        assert result['is_stationary'] == False
        assert 'Insufficient' in str(result.get('error', ''))

    def test_adf_constant_series(self):
        """Test ADF on constant series."""
        series = pd.Series(np.full(100, 42.0))

        result = adf_test(series)

        # Should handle without crash
        assert 'adf_statistic' in result

    def test_adf_empty_series(self):
        """Test ADF on empty series."""
        series = pd.Series([], dtype=float)

        result = adf_test(series)

        assert result['is_stationary'] == False
        assert 'error' in result or result['n_obs'] == 0


class TestGenerateADFReport:
    """Tests for ADF report generation."""

    def test_report_multiple_columns(self):
        """Test report generation for multiple columns."""
        np.random.seed(42)
        df = pd.DataFrame({
            'stationary': np.random.randn(500),
            'non_stationary': np.cumsum(np.random.randn(500)),
            'also_stationary': np.random.randn(500),
        })

        report = generate_adf_report(df, ['stationary', 'non_stationary', 'also_stationary'])

        assert len(report) == 3
        assert 'column' in report.columns
        assert 'is_stationary' in report.columns

        # Check correct detection
        stat_row = report[report['column'] == 'stationary'].iloc[0]
        non_stat_row = report[report['column'] == 'non_stationary'].iloc[0]

        assert stat_row['is_stationary']
        assert not non_stat_row['is_stationary']

    def test_report_missing_column(self):
        """Test report handles missing columns."""
        df = pd.DataFrame({'col1': np.random.randn(100)})

        report = generate_adf_report(df, ['col1', 'col2'])  # col2 doesn't exist

        assert len(report) == 2

        col2_row = report[report['column'] == 'col2'].iloc[0]
        assert col2_row['is_stationary'] == False
        assert 'not found' in str(col2_row.get('error', '')).lower()

    def test_report_custom_threshold(self):
        """Test report with custom p-value threshold."""
        np.random.seed(42)
        df = pd.DataFrame({'series': np.random.randn(500)})

        # Strict threshold
        report_strict = generate_adf_report(df, ['series'], threshold=0.01)

        # Lenient threshold
        report_lenient = generate_adf_report(df, ['series'], threshold=0.1)

        # Both should have same p-value but potentially different is_stationary
        assert report_strict['p_value'].iloc[0] == report_lenient['p_value'].iloc[0]

    def test_report_empty_columns(self):
        """Test report with empty column list."""
        df = pd.DataFrame({'col1': np.random.randn(100)})

        report = generate_adf_report(df, [])

        # Empty columns list should return empty DataFrame
        assert isinstance(report, pd.DataFrame)
        assert len(report) == 0 or report.empty


class TestValidateFeatures:
    """Tests for feature validation function."""

    def test_valid_features(self, sample_dollar_bar_data):
        """Test validation passes for valid data."""
        is_valid, details = validate_features(
            sample_dollar_bar_data,
            check_stationarity=False  # Skip stationarity for simple test
        )

        assert details['n_rows'] == len(sample_dollar_bar_data)
        assert 'issues' in details

    def test_missing_required_columns(self):
        """Test detection of missing required columns."""
        df = pd.DataFrame({'col1': [1, 2, 3]})

        is_valid, details = validate_features(
            df,
            required_columns=['col1', 'col2', 'col3'],
            check_stationarity=False
        )

        assert not is_valid
        assert 'missing_columns' in details
        assert 'col2' in details['missing_columns']
        assert 'col3' in details['missing_columns']

    def test_infinite_values_detection(self):
        """Test detection of infinite values."""
        df = pd.DataFrame({
            'normal': [1, 2, 3, 4],
            'with_inf': [1, np.inf, 3, -np.inf],
        })

        is_valid, details = validate_features(df, check_stationarity=False)

        assert not is_valid
        assert 'infinite_columns' in details
        assert 'with_inf' in details['infinite_columns']

    def test_high_nan_ratio(self):
        """Test detection of high NaN ratio."""
        n = 100
        df = pd.DataFrame({
            'good': np.random.randn(n),
            'bad': [np.nan] * 50 + list(np.random.randn(50)),  # 50% NaN
        })

        is_valid, details = validate_features(df, check_stationarity=False)

        assert not is_valid
        assert 'high_nan_columns' in details
        assert 'bad' in details['high_nan_columns']

    def test_stationarity_check(self):
        """Test stationarity validation."""
        np.random.seed(42)
        df = pd.DataFrame({
            'frac_diff_close': np.random.randn(500),  # Stationary
            'returns': np.random.randn(500),  # Stationary
        })

        is_valid, details = validate_features(
            df,
            check_stationarity=True
        )

        assert 'adf_report' in details

    def test_all_validations_pass(self):
        """Test when all validations pass."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
        })

        is_valid, details = validate_features(
            df,
            required_columns=['feature1', 'feature2'],
            check_stationarity=False
        )

        assert is_valid
        assert len(details['issues']) == 0


class TestPrintValidationReport:
    """Tests for validation report printing."""

    def test_print_valid_report(self, capsys):
        """Test printing valid report."""
        details = {
            'n_rows': 1000,
            'n_cols': 10,
            'is_valid': True,
            'issues': []
        }

        print_validation_report(details)
        captured = capsys.readouterr()

        assert 'VALIDATION REPORT' in captured.out
        assert '1,000 rows' in captured.out
        assert 'PASS' in captured.out

    def test_print_invalid_report(self, capsys):
        """Test printing invalid report with issues."""
        details = {
            'n_rows': 500,
            'n_cols': 5,
            'is_valid': False,
            'issues': ['Missing columns: [col1]', 'Infinite values in: [col2]']
        }

        print_validation_report(details)
        captured = capsys.readouterr()

        assert 'FAIL' in captured.out
        assert 'Missing columns' in captured.out

    def test_print_with_adf_report(self, capsys):
        """Test printing report with ADF results."""
        details = {
            'n_rows': 1000,
            'n_cols': 10,
            'is_valid': True,
            'issues': [],
            'adf_report': [
                {'column': 'returns', 'p_value': 0.001, 'is_stationary': True},
                {'column': 'price', 'p_value': 0.5, 'is_stationary': False},
            ]
        }

        print_validation_report(details)
        captured = capsys.readouterr()

        assert 'Stationarity Report' in captured.out
        assert 'returns' in captured.out
        assert 'price' in captured.out


class TestEdgeCases:
    """Edge case tests for validation module."""

    def test_empty_dataframe(self):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame()

        is_valid, details = validate_features(df, check_stationarity=False)

        assert details['n_rows'] == 0

    def test_single_row_dataframe(self):
        """Test validation of single-row DataFrame."""
        df = pd.DataFrame({'col': [1]})

        is_valid, details = validate_features(df, check_stationarity=False)

        assert details['n_rows'] == 1

    def test_all_nan_column(self):
        """Test column with all NaN values."""
        df = pd.DataFrame({
            'all_nan': [np.nan] * 100,
            'good': np.random.randn(100),
        })

        is_valid, details = validate_features(df, check_stationarity=False)

        assert 'high_nan_columns' in details
        assert 'all_nan' in details['high_nan_columns']

    def test_mixed_types(self):
        """Test DataFrame with mixed column types."""
        df = pd.DataFrame({
            'numeric': [1.0, 2.0, 3.0],
            'string': ['a', 'b', 'c'],
            'bool': [True, False, True],
        })

        # Should only check numeric columns for inf
        is_valid, details = validate_features(df, check_stationarity=False)

        # Should not crash
        assert 'n_rows' in details
