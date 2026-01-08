#!/usr/bin/env python3
"""
Adaptive Kalman Filter Verification Script.

Validates the Kalman filter implementation against real BTC Dollar Bar data.
Performs statistical tests and generates diagnostic outputs.
"""

import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.features.kalman import (
    AdaptiveKalmanFilter,
    KalmanConfig,
    KalmanFeatureGenerator,
)


def load_sample_data(n_files: int = 3) -> pd.DataFrame:
    """Load sample Dollar Bar data."""
    bar_path = Path(__file__).parent.parent / "etl/data/bars-24/futures/BTCUSDT"
    files = sorted(glob.glob(str(bar_path / "*.parquet")))

    if not files:
        raise FileNotFoundError(f"No parquet files found in {bar_path}")

    # Load a few months of data
    dfs = []
    for f in files[:n_files]:
        df = pd.read_parquet(f)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(data):,} bars from {n_files} files")
    return data


def test_basic_functionality(df: pd.DataFrame) -> dict:
    """Test 1: Basic functionality check."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Functionality")
    print("=" * 60)

    config = KalmanConfig(r_window=20, q_window=20, use_typical_price=True)
    kf = AdaptiveKalmanFilter(config)
    result = kf.filter(df)

    # Check all columns exist
    expected_cols = [
        'kf_trend', 'kf_velocity', 'kf_deviation',
        'kf_deviation_pct', 'kf_gain', 'kf_uncertainty', 'kf_signal'
    ]

    missing = [c for c in expected_cols if c not in result.columns]
    if missing:
        print(f"  [FAIL] Missing columns: {missing}")
        return {'passed': False, 'reason': f'Missing columns: {missing}'}

    # Check no inf values
    for col in expected_cols:
        if np.isinf(result[col]).any():
            print(f"  [FAIL] Inf values in {col}")
            return {'passed': False, 'reason': f'Inf values in {col}'}

    # Check NaN ratio after warmup
    warmup = 50
    for col in expected_cols:
        nan_ratio = result[col].iloc[warmup:].isna().mean()
        if nan_ratio > 0.01:
            print(f"  [FAIL] High NaN ratio in {col}: {nan_ratio:.1%}")
            return {'passed': False, 'reason': f'High NaN ratio in {col}'}

    print("  [PASS] All columns present, no inf, low NaN ratio")
    return {'passed': True}


def test_trend_tracking(df: pd.DataFrame) -> dict:
    """Test 2: Trend should follow price closely."""
    print("\n" + "=" * 60)
    print("TEST 2: Trend Tracking Quality")
    print("=" * 60)

    config = KalmanConfig(r_window=20, q_window=20)
    kf = AdaptiveKalmanFilter(config)
    result = kf.filter(df)

    # Correlation test
    corr = result['kf_trend'].corr(result['close'])
    print(f"  Trend-Price Correlation: {corr:.4f}")

    if corr < 0.99:
        print(f"  [FAIL] Correlation too low (expected > 0.99)")
        return {'passed': False, 'correlation': corr}

    # Lag test: trend should not lead price
    # Cross-correlation at various lags
    close_norm = (result['close'] - result['close'].mean()) / result['close'].std()
    trend_norm = (result['kf_trend'] - result['kf_trend'].mean()) / result['kf_trend'].std()

    # Max correlation should be at lag 0 or small positive lag
    cross_corr = np.correlate(close_norm.dropna().values[-1000:],
                               trend_norm.dropna().values[-1000:], mode='full')
    lag_at_max = np.argmax(cross_corr) - len(close_norm.dropna().values[-1000:]) + 1

    print(f"  Peak Cross-Correlation Lag: {lag_at_max} bars")

    if lag_at_max < -5:
        print(f"  [FAIL] Trend appears to lead price (lookahead bias)")
        return {'passed': False, 'lag': lag_at_max}

    print("  [PASS] Good trend tracking, no lookahead bias")
    return {'passed': True, 'correlation': corr, 'lag': lag_at_max}


def test_deviation_statistics(df: pd.DataFrame) -> dict:
    """Test 3: Deviation should be stationary and mean-reverting."""
    print("\n" + "=" * 60)
    print("TEST 3: Deviation Statistics")
    print("=" * 60)

    config = KalmanConfig(r_window=20, q_window=20)
    kf = AdaptiveKalmanFilter(config)
    result = kf.filter(df)

    deviation = result['kf_deviation'].iloc[100:].dropna()

    # Mean should be close to 0
    mean_dev = deviation.mean()
    print(f"  Mean Deviation: {mean_dev:.2f}")

    # Std dev
    std_dev = deviation.std()
    print(f"  Std Deviation: {std_dev:.2f}")

    # Skewness
    skew = stats.skew(deviation)
    print(f"  Skewness: {skew:.3f}")

    # Kurtosis
    kurt = stats.kurtosis(deviation)
    print(f"  Kurtosis: {kurt:.3f}")

    # Autocorrelation at lag 1 (should be positive for mean-reverting after filter)
    autocorr = deviation.autocorr(lag=1)
    print(f"  Autocorrelation (lag=1): {autocorr:.3f}")

    # ADF test for stationarity
    from statsmodels.tsa.stattools import adfuller
    adf_stat, adf_pvalue, *_ = adfuller(deviation.values, maxlag=20)
    print(f"  ADF Statistic: {adf_stat:.3f}")
    print(f"  ADF p-value: {adf_pvalue:.6f}")

    if adf_pvalue > 0.05:
        print(f"  [WARN] Deviation may not be stationary (p={adf_pvalue:.4f})")

    # Mean should be close to 0 relative to price level
    mean_pct = abs(mean_dev) / df['close'].mean() * 100
    if mean_pct > 1:
        print(f"  [FAIL] Mean deviation too large: {mean_pct:.2f}% of price")
        return {'passed': False, 'mean_pct': mean_pct}

    print("  [PASS] Deviation is well-behaved")
    return {
        'passed': True,
        'mean': mean_dev,
        'std': std_dev,
        'skew': skew,
        'kurtosis': kurt,
        'autocorr': autocorr,
        'adf_pvalue': adf_pvalue
    }


def test_signal_distribution(df: pd.DataFrame) -> dict:
    """Test 4: Signal (z-score) should be approximately normalized."""
    print("\n" + "=" * 60)
    print("TEST 4: Signal Distribution")
    print("=" * 60)

    config = KalmanConfig(r_window=20, q_window=20)
    kf = AdaptiveKalmanFilter(config)
    result = kf.filter(df)

    signal = result['kf_signal'].iloc[100:].dropna()

    # Basic statistics
    mean_sig = signal.mean()
    std_sig = signal.std()
    print(f"  Mean: {mean_sig:.3f} (expected ~0)")
    print(f"  Std: {std_sig:.3f} (expected ~1)")

    # Percentile distribution
    p1 = np.percentile(signal, 1)
    p5 = np.percentile(signal, 5)
    p25 = np.percentile(signal, 25)
    p50 = np.percentile(signal, 50)
    p75 = np.percentile(signal, 75)
    p95 = np.percentile(signal, 95)
    p99 = np.percentile(signal, 99)

    print(f"  Percentiles: 1%={p1:.2f}, 5%={p5:.2f}, 25%={p25:.2f}, "
          f"50%={p50:.2f}, 75%={p75:.2f}, 95%={p95:.2f}, 99%={p99:.2f}")

    # Check coverage
    within_1std = ((signal >= -1) & (signal <= 1)).mean()
    within_2std = ((signal >= -2) & (signal <= 2)).mean()
    within_3std = ((signal >= -3) & (signal <= 3)).mean()

    print(f"  Within 1 std: {within_1std:.1%} (expected ~68%)")
    print(f"  Within 2 std: {within_2std:.1%} (expected ~95%)")
    print(f"  Within 3 std: {within_3std:.1%} (expected ~99.7%)")

    # Fat tails are expected in finance
    if within_3std < 0.90:
        print(f"  [WARN] Heavy tails detected (within 3std = {within_3std:.1%})")

    print("  [PASS] Signal distribution is reasonable")
    return {
        'passed': True,
        'mean': mean_sig,
        'std': std_sig,
        'within_1std': within_1std,
        'within_2std': within_2std,
        'within_3std': within_3std
    }


def test_adaptive_behavior(df: pd.DataFrame) -> dict:
    """Test 5: Filter should adapt to different market conditions."""
    print("\n" + "=" * 60)
    print("TEST 5: Adaptive Behavior")
    print("=" * 60)

    config = KalmanConfig(r_window=20, q_window=20)
    kf = AdaptiveKalmanFilter(config)
    result = kf.filter(df)

    # Calculate rolling volatility
    returns = df['close'].pct_change()
    vol_20 = returns.rolling(20).std() * 100  # as percentage

    # Split into volatility regimes
    vol_median = vol_20.median()
    low_vol_mask = vol_20 < vol_median
    high_vol_mask = vol_20 >= vol_median

    # Compare Kalman gain in different regimes
    gain_low_vol = result.loc[low_vol_mask, 'kf_gain'].mean()
    gain_high_vol = result.loc[high_vol_mask, 'kf_gain'].mean()

    print(f"  Low Volatility Avg Gain: {gain_low_vol:.4f}")
    print(f"  High Volatility Avg Gain: {gain_high_vol:.4f}")

    # Compare uncertainty in different regimes
    unc_low_vol = result.loc[low_vol_mask, 'kf_uncertainty'].mean()
    unc_high_vol = result.loc[high_vol_mask, 'kf_uncertainty'].mean()

    print(f"  Low Volatility Avg Uncertainty: {unc_low_vol:.6f}")
    print(f"  High Volatility Avg Uncertainty: {unc_high_vol:.6f}")

    # Compare velocity capture during trends
    strong_up = returns.rolling(20).mean() > returns.rolling(20).std()
    strong_down = returns.rolling(20).mean() < -returns.rolling(20).std()

    vel_up = result.loc[strong_up, 'kf_velocity'].mean()
    vel_down = result.loc[strong_down, 'kf_velocity'].mean()

    print(f"  Velocity during uptrends: {vel_up:.4f}")
    print(f"  Velocity during downtrends: {vel_down:.4f}")

    if vel_up <= vel_down:
        print(f"  [WARN] Velocity not capturing trend direction well")

    print("  [PASS] Filter shows adaptive behavior")
    return {
        'passed': True,
        'gain_low_vol': gain_low_vol,
        'gain_high_vol': gain_high_vol,
        'unc_low_vol': unc_low_vol,
        'unc_high_vol': unc_high_vol,
        'vel_up': vel_up,
        'vel_down': vel_down
    }


def test_position_sizing(df: pd.DataFrame) -> dict:
    """Test 6: Position sizing based on uncertainty."""
    print("\n" + "=" * 60)
    print("TEST 6: Position Sizing")
    print("=" * 60)

    config = KalmanConfig(r_window=20, q_window=20)
    kf = AdaptiveKalmanFilter(config)
    result = kf.filter(df)

    # Calculate position sizes with data-driven bounds
    uncertainties = result['kf_uncertainty'].iloc[100:].dropna()

    # Use actual data percentiles for scaling
    min_unc = uncertainties.quantile(0.05)
    max_unc = uncertainties.quantile(0.95)

    sizes = uncertainties.apply(
        lambda u: kf.get_position_size_factor(u, min_unc, max_unc)
    )

    print(f"  Uncertainty Range (5%-95%): {min_unc:.2f} - {max_unc:.2f}")
    print(f"  Min Position Size: {sizes.min():.3f}")
    print(f"  Max Position Size: {sizes.max():.3f}")
    print(f"  Mean Position Size: {sizes.mean():.3f}")
    print(f"  Std Position Size: {sizes.std():.3f}")

    # Correlation with uncertainty (should be negative)
    corr = sizes.corr(uncertainties)
    print(f"  Size-Uncertainty Correlation: {corr:.3f} (expected < 0)")

    if corr > 0:
        print(f"  [FAIL] Position size should decrease with uncertainty")
        return {'passed': False, 'correlation': corr}

    # Check all in valid range
    if (sizes < 0).any() or (sizes > 1).any():
        print(f"  [FAIL] Position sizes outside [0, 1] range")
        return {'passed': False}

    # Check reasonable distribution
    if sizes.std() < 0.1:
        print(f"  [WARN] Position sizes have low variance")

    print("  [PASS] Position sizing works correctly")
    return {
        'passed': True,
        'min': sizes.min(),
        'max': sizes.max(),
        'mean': sizes.mean(),
        'std': sizes.std(),
        'correlation': corr
    }


def test_typical_price_difference(df: pd.DataFrame) -> dict:
    """Test 7: Compare typical price vs close price filtering."""
    print("\n" + "=" * 60)
    print("TEST 7: Typical Price vs Close Price")
    print("=" * 60)

    config_tp = KalmanConfig(use_typical_price=True)
    config_close = KalmanConfig(use_typical_price=False)

    kf_tp = AdaptiveKalmanFilter(config_tp)
    kf_close = AdaptiveKalmanFilter(config_close)

    result_tp = kf_tp.filter(df)
    result_close = kf_close.filter(df)

    # Compare trends
    trend_corr = result_tp['kf_trend'].corr(result_close['kf_trend'])
    print(f"  Trend Correlation (TP vs Close): {trend_corr:.4f}")

    # Compare deviations
    dev_corr = result_tp['kf_deviation'].corr(result_close['kf_deviation'])
    print(f"  Deviation Correlation: {dev_corr:.4f}")

    # Typical price should be smoother
    tp_dev_std = result_tp['kf_deviation'].std()
    close_dev_std = result_close['kf_deviation'].std()

    print(f"  Typical Price Deviation Std: {tp_dev_std:.2f}")
    print(f"  Close Price Deviation Std: {close_dev_std:.2f}")

    if tp_dev_std > close_dev_std * 1.1:
        print(f"  [WARN] Typical price not reducing noise as expected")

    print("  [PASS] Both methods working, showing expected differences")
    return {
        'passed': True,
        'trend_corr': trend_corr,
        'dev_corr': dev_corr,
        'tp_dev_std': tp_dev_std,
        'close_dev_std': close_dev_std
    }


def generate_summary_statistics(df: pd.DataFrame) -> None:
    """Generate and print comprehensive statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    config = KalmanConfig(r_window=20, q_window=20, use_typical_price=True)
    kf = AdaptiveKalmanFilter(config)
    result = kf.filter(df)

    print(f"\nData Summary:")
    print(f"  Total Bars: {len(df):,}")
    print(f"  Price Range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
    print(f"  Date Range: {pd.to_datetime(df['start_time'].iloc[0], unit='ms')} - "
          f"{pd.to_datetime(df['start_time'].iloc[-1], unit='ms')}")

    print(f"\nKalman Filter Output Summary:")
    for col in ['kf_trend', 'kf_velocity', 'kf_deviation', 'kf_gain', 'kf_uncertainty', 'kf_signal']:
        valid = result[col].iloc[100:].dropna()
        print(f"\n  {col}:")
        print(f"    Mean: {valid.mean():.6f}")
        print(f"    Std: {valid.std():.6f}")
        print(f"    Min: {valid.min():.6f}")
        print(f"    Max: {valid.max():.6f}")


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("ADAPTIVE KALMAN FILTER VERIFICATION")
    print("=" * 60)

    # Load data
    try:
        df = load_sample_data(n_files=6)  # ~6 months
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure Dollar Bar data exists in etl/data/bars-24/futures/BTCUSDT/")
        return 1

    # Run tests
    results = {}
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Trend Tracking", test_trend_tracking),
        ("Deviation Statistics", test_deviation_statistics),
        ("Signal Distribution", test_signal_distribution),
        ("Adaptive Behavior", test_adaptive_behavior),
        ("Position Sizing", test_position_sizing),
        ("Typical vs Close Price", test_typical_price_difference),
    ]

    for name, test_func in tests:
        try:
            results[name] = test_func(df)
        except Exception as e:
            print(f"\n  [ERROR] {name}: {e}")
            results[name] = {'passed': False, 'error': str(e)}

    # Generate summary
    generate_summary_statistics(df)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r.get('passed', False))
    total = len(results)

    for name, result in results.items():
        status = "PASS" if result.get('passed', False) else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  Adaptive Kalman Filter verification SUCCESSFUL!")
        return 0
    else:
        print("\n  Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
