"""
Feature Validation Module.

Provides stationarity verification using ADF tests and generates reports.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def adf_test(series: pd.Series, maxlag: Optional[int] = None) -> Dict:
    """
    Perform Augmented Dickey-Fuller test for stationarity.

    Args:
        series: Time series data
        maxlag: Maximum lag to include in test

    Returns:
        Dict with test results
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        logger.warning("statsmodels not installed, using simple variance ratio test")
        return _simple_stationarity_test(series)

    # Drop NaN values
    clean_series = series.dropna()

    if len(clean_series) < 20:
        return {
            'adf_statistic': np.nan,
            'p_value': np.nan,
            'used_lag': np.nan,
            'n_obs': len(clean_series),
            'critical_values': {},
            'is_stationary': False,
            'error': 'Insufficient data'
        }

    try:
        result = adfuller(clean_series, maxlag=maxlag, autolag='AIC')

        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'used_lag': result[2],
            'n_obs': result[3],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05,
            'error': None
        }
    except Exception as e:
        return {
            'adf_statistic': np.nan,
            'p_value': np.nan,
            'used_lag': np.nan,
            'n_obs': len(clean_series),
            'critical_values': {},
            'is_stationary': False,
            'error': str(e)
        }


def _simple_stationarity_test(series: pd.Series) -> Dict:
    """
    Simple variance ratio test as fallback.

    Compares variance of first half vs second half.
    """
    clean_series = series.dropna()

    if len(clean_series) < 20:
        return {
            'adf_statistic': np.nan,
            'p_value': np.nan,
            'used_lag': np.nan,
            'n_obs': len(clean_series),
            'critical_values': {},
            'is_stationary': False,
            'error': 'Insufficient data'
        }

    mid = len(clean_series) // 2
    var1 = clean_series.iloc[:mid].var()
    var2 = clean_series.iloc[mid:].var()

    # Simple heuristic: variance ratio close to 1 suggests stationarity
    var_ratio = var1 / (var2 + 1e-10)
    is_stationary = 0.5 < var_ratio < 2.0

    return {
        'adf_statistic': np.nan,
        'p_value': np.nan,
        'used_lag': np.nan,
        'n_obs': len(clean_series),
        'critical_values': {},
        'is_stationary': is_stationary,
        'variance_ratio': var_ratio,
        'error': 'Using variance ratio test (statsmodels not available)'
    }


def generate_adf_report(
    df: pd.DataFrame,
    columns: List[str],
    threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Generate ADF test report for multiple columns.

    Args:
        df: DataFrame with feature columns
        columns: List of column names to test
        threshold: P-value threshold for stationarity (default 0.05)

    Returns:
        DataFrame with test results for each column
    """
    results = []

    for col in columns:
        if col not in df.columns:
            results.append({
                'column': col,
                'adf_statistic': np.nan,
                'p_value': np.nan,
                'is_stationary': False,
                'n_obs': 0,
                'error': 'Column not found'
            })
            continue

        test_result = adf_test(df[col])

        results.append({
            'column': col,
            'adf_statistic': test_result['adf_statistic'],
            'p_value': test_result['p_value'],
            'is_stationary': test_result['p_value'] < threshold if not np.isnan(test_result['p_value']) else False,
            'n_obs': test_result['n_obs'],
            'error': test_result.get('error')
        })

    report_df = pd.DataFrame(results)

    # Summary logging
    if len(report_df) > 0:
        n_stationary = report_df['is_stationary'].sum()
        n_total = len(report_df)
        logger.info(f"ADF Report: {n_stationary}/{n_total} columns are stationary (p < {threshold})")

    return report_df


def validate_features(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    check_stationarity: bool = True,
) -> Tuple[bool, Dict]:
    """
    Validate feature DataFrame.

    Checks:
        1. Required columns exist
        2. No infinite values
        3. NaN ratio is acceptable
        4. Stationarity of key columns (optional)

    Args:
        df: Feature DataFrame
        required_columns: List of required column names
        check_stationarity: Whether to run ADF tests

    Returns:
        Tuple of (is_valid, details_dict)
    """
    details = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'issues': []
    }

    # 1. Check required columns
    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            details['issues'].append(f"Missing columns: {missing}")
            details['missing_columns'] = missing

    # 2. Check for infinite values
    inf_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if np.isinf(df[col]).any():
            inf_cols.append(col)

    if inf_cols:
        details['issues'].append(f"Infinite values in: {inf_cols}")
        details['infinite_columns'] = inf_cols

    # 3. Check NaN ratio
    nan_ratios = df.isna().mean()
    high_nan_cols = nan_ratios[nan_ratios > 0.1].to_dict()

    if high_nan_cols:
        details['issues'].append(f"High NaN ratio (>10%) in: {list(high_nan_cols.keys())}")
        details['high_nan_columns'] = high_nan_cols

    # 4. Stationarity check
    if check_stationarity:
        stationary_cols = ['frac_diff_close', 'returns', 'volume_imbalance']
        existing_cols = [c for c in stationary_cols if c in df.columns]

        if existing_cols:
            adf_report = generate_adf_report(df, existing_cols)
            non_stationary = adf_report[~adf_report['is_stationary']]['column'].tolist()

            if non_stationary:
                details['issues'].append(f"Non-stationary columns: {non_stationary}")
                details['non_stationary_columns'] = non_stationary

            details['adf_report'] = adf_report.to_dict('records')

    is_valid = len(details['issues']) == 0
    details['is_valid'] = is_valid

    return is_valid, details


def print_validation_report(details: Dict):
    """Print validation report to console."""
    print("\n" + "=" * 60)
    print("FEATURE VALIDATION REPORT")
    print("=" * 60)

    print(f"\nDataset: {details['n_rows']:,} rows, {details['n_cols']} columns")

    if details['is_valid']:
        print("\n[PASS] All validation checks passed")
    else:
        print("\n[FAIL] Issues found:")
        for issue in details['issues']:
            print(f"  - {issue}")

    if 'adf_report' in details:
        print("\nStationarity Report:")
        print("-" * 40)
        for row in details['adf_report']:
            status = "[PASS]" if row['is_stationary'] else "[FAIL]"
            p_val = f"{row['p_value']:.4f}" if not np.isnan(row['p_value']) else "N/A"
            print(f"  {status} {row['column']}: p-value = {p_val}")

    print("=" * 60 + "\n")
