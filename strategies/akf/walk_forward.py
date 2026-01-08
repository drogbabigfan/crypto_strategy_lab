#!/usr/bin/env python3
"""
Walk-Forward Validation for AKF Strategies.

Tests for overfitting by:
1. Splitting data into rolling train/test windows
2. Optionally optimizing parameters on in-sample (train) data
3. Testing on out-of-sample (test) data
4. Comparing IS vs OOS performance

Usage:
    python strategies/akf/walk_forward.py --train-months 12 --test-months 3
    python strategies/akf/walk_forward.py --fixed-k 2.0  # No optimization
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import tempfile
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from research.features.kalman import calculate_adaptive_kalman
from strategies.akf import AKFStrategies, AKFStrategyConfig


# Paths
DATA_DIR = PROJECT_ROOT / "etl/data/features-24/futures/BTCUSDT"
OUTPUT_DIR = PROJECT_ROOT / "strategies/akf/results"
BACKTESTER_PATH = PROJECT_ROOT / "etl/bin/backtester"


@dataclass
class WalkForwardWindow:
    """A single walk-forward window."""
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_k: float
    train_pnl: float
    train_sharpe: float
    train_mdd: float
    train_trades: int
    test_pnl: float
    test_sharpe: float
    test_mdd: float
    test_trades: int


def generate_months(start: str, end: str) -> list[str]:
    """Generate list of months between start and end (inclusive)."""
    start_dt = datetime.strptime(start, '%Y-%m')
    end_dt = datetime.strptime(end, '%Y-%m')
    months = []
    current = start_dt
    while current <= end_dt:
        months.append(current.strftime('%Y-%m'))
        current += relativedelta(months=1)
    return months


def load_data(months: list[str], data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and concatenate feature data for specified months."""
    dfs = []
    for month in months:
        path = data_dir / f"BTCUSDT-features-{month}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def run_backtest(
    df: pd.DataFrame,
    strategy: str,
    k: float,
    warmup: int = 130,
) -> Optional[dict]:
    """Run backtest and return metrics."""
    import subprocess

    if len(df) < warmup + 10:
        return None

    try:
        # Apply Kalman filter
        df_kf = calculate_adaptive_kalman(df.copy())

        # Generate signals
        akf = AKFStrategies(AKFStrategyConfig(k=k))
        result = akf.run_strategy(df_kf, strategy, include_bands=True)

        # Zero out warmup
        result.loc[:warmup - 1, 'signal'] = 0

        # Save to temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            signals_path = Path(tmpdir) / "signals.parquet"
            features_path = Path(tmpdir) / "features.parquet"

            # Save signals
            signals_df = pd.DataFrame({
                'timestamp': result['start_time'].astype(np.int64),
                'signal': result['signal'].astype(np.int8),
            })
            schema = pa.schema([
                ('timestamp', pa.int64()),
                ('signal', pa.int8()),
            ])
            table = pa.Table.from_pandas(signals_df, schema=schema)
            pq.write_table(table, signals_path)

            # Save features
            if 'log_volume' in result.columns and 'volume' not in result.columns:
                volume = np.exp(result['log_volume'])
            elif 'volume' in result.columns:
                volume = result['volume']
            else:
                volume = np.ones(len(result)) * 1000

            features_df = pd.DataFrame({
                'timestamp': result['start_time'].astype(np.int64),
                'open': result['open'].astype(np.float64),
                'high': result['high'].astype(np.float64),
                'low': result['low'].astype(np.float64),
                'close': result['close'].astype(np.float64),
                'volume': volume.astype(np.float64),
                'realized_vol': result['realized_vol'].astype(np.float64),
            })
            schema = pa.schema([
                ('timestamp', pa.int64()),
                ('open', pa.float64()),
                ('high', pa.float64()),
                ('low', pa.float64()),
                ('close', pa.float64()),
                ('volume', pa.float64()),
                ('realized_vol', pa.float64()),
            ])
            table = pa.Table.from_pandas(features_df, schema=schema)
            pq.write_table(table, features_path)

            # Run Go backtester
            cmd = [
                str(BACKTESTER_PATH),
                '-signals', str(signals_path),
                '-features', str(features_path),
                '-exit-mode', 'signal',
                '-quiet',
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True)

            if proc.returncode != 0:
                return None

            results = json.loads(proc.stdout)

            return {
                'trades': results.get('total_trades', 0),
                'win_rate': results.get('win_rate', 0) * 100,
                'avg_pnl': results.get('avg_pnl', 0) * 100,
                'total_pnl': results.get('total_pnl', 0) * 100,
                'sharpe': results.get('sharpe_ratio', 0),
                'mdd': results.get('max_drawdown', 0) * 100,
                'profit_factor': results.get('profit_factor', 0),
            }

    except Exception as e:
        print(f"Error in backtest: {e}")
        return None


def optimize_k(
    df: pd.DataFrame,
    strategy: str,
    k_values: list[float],
    warmup: int = 130,
) -> tuple[float, dict]:
    """
    Find optimal k on in-sample data.

    Returns (best_k, best_metrics)
    """
    best_k = k_values[0]
    best_sharpe = float('-inf')
    best_metrics = None

    for k in k_values:
        metrics = run_backtest(df, strategy, k, warmup)
        if metrics and metrics['sharpe'] > best_sharpe:
            best_sharpe = metrics['sharpe']
            best_k = k
            best_metrics = metrics

    return best_k, best_metrics


def run_walk_forward(
    start: str,
    end: str,
    train_months: int,
    test_months: int,
    strategy: str = 'strategy_db',
    fixed_k: Optional[float] = None,
    k_values: list[float] = None,
    warmup: int = 130,
) -> list[WalkForwardWindow]:
    """
    Run walk-forward validation.

    Args:
        start: Start month (YYYY-MM)
        end: End month (YYYY-MM)
        train_months: Number of months for training window
        test_months: Number of months for testing window
        strategy: Strategy name
        fixed_k: If set, use fixed k (no optimization)
        k_values: K values to test during optimization
        warmup: Warmup period in bars

    Returns:
        List of WalkForwardWindow results
    """
    k_values = k_values or [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    all_months = generate_months(start, end)
    results = []

    # Calculate number of windows
    window_size = train_months + test_months
    num_windows = (len(all_months) - train_months) // test_months

    print(f"\nWalk-Forward Configuration:")
    print(f"  Period: {start} to {end} ({len(all_months)} months)")
    print(f"  Train window: {train_months} months")
    print(f"  Test window: {test_months} months")
    print(f"  Number of windows: {num_windows}")
    print(f"  Fixed k: {fixed_k if fixed_k else 'Optimized'}")
    print(f"  Strategy: {strategy}")

    for i in range(num_windows):
        train_start_idx = i * test_months
        train_end_idx = train_start_idx + train_months - 1
        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + test_months - 1

        if test_end_idx >= len(all_months):
            break

        train_start = all_months[train_start_idx]
        train_end = all_months[train_end_idx]
        test_start = all_months[test_start_idx]
        test_end = all_months[test_end_idx]

        print(f"\n[Window {i+1}/{num_windows}]")
        print(f"  Train: {train_start} to {train_end}")
        print(f"  Test:  {test_start} to {test_end}")

        # Load train data
        train_months_list = generate_months(train_start, train_end)
        train_df = load_data(train_months_list)

        if train_df.empty:
            print(f"  Warning: No train data, skipping")
            continue

        # Optimize or use fixed k
        if fixed_k is not None:
            best_k = fixed_k
            train_metrics = run_backtest(train_df, strategy, best_k, warmup)
        else:
            best_k, train_metrics = optimize_k(train_df, strategy, k_values, warmup)
            print(f"  Optimized k: {best_k}")

        if train_metrics is None:
            print(f"  Warning: Train backtest failed, skipping")
            continue

        print(f"  Train: PnL={train_metrics['total_pnl']:.1f}%, Sharpe={train_metrics['sharpe']:.2f}")

        # Load test data
        test_months_list = generate_months(test_start, test_end)
        test_df = load_data(test_months_list)

        if test_df.empty:
            print(f"  Warning: No test data, skipping")
            continue

        # Test with best k
        test_metrics = run_backtest(test_df, strategy, best_k, warmup)

        if test_metrics is None:
            print(f"  Warning: Test backtest failed, skipping")
            continue

        print(f"  Test:  PnL={test_metrics['total_pnl']:.1f}%, Sharpe={test_metrics['sharpe']:.2f}")

        # Store results
        window = WalkForwardWindow(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            best_k=best_k,
            train_pnl=train_metrics['total_pnl'],
            train_sharpe=train_metrics['sharpe'],
            train_mdd=train_metrics['mdd'],
            train_trades=train_metrics['trades'],
            test_pnl=test_metrics['total_pnl'],
            test_sharpe=test_metrics['sharpe'],
            test_mdd=test_metrics['mdd'],
            test_trades=test_metrics['trades'],
        )
        results.append(window)

    return results


def analyze_results(windows: list[WalkForwardWindow]) -> dict:
    """Analyze walk-forward results for overfitting."""
    if not windows:
        return {}

    # Extract metrics
    train_pnls = [w.train_pnl for w in windows]
    test_pnls = [w.test_pnl for w in windows]
    train_sharpes = [w.train_sharpe for w in windows]
    test_sharpes = [w.test_sharpe for w in windows]
    ks = [w.best_k for w in windows]

    # Calculate statistics
    analysis = {
        'num_windows': len(windows),

        # PnL statistics
        'train_pnl_mean': np.mean(train_pnls),
        'train_pnl_std': np.std(train_pnls),
        'test_pnl_mean': np.mean(test_pnls),
        'test_pnl_std': np.std(test_pnls),
        'pnl_degradation': np.mean(train_pnls) - np.mean(test_pnls),
        'pnl_degradation_pct': (np.mean(train_pnls) - np.mean(test_pnls)) / abs(np.mean(train_pnls)) * 100 if np.mean(train_pnls) != 0 else 0,

        # Sharpe statistics
        'train_sharpe_mean': np.mean(train_sharpes),
        'test_sharpe_mean': np.mean(test_sharpes),
        'sharpe_degradation': np.mean(train_sharpes) - np.mean(test_sharpes),

        # Win rate (positive PnL windows)
        'train_win_rate': sum(1 for p in train_pnls if p > 0) / len(train_pnls) * 100,
        'test_win_rate': sum(1 for p in test_pnls if p > 0) / len(test_pnls) * 100,

        # Cumulative OOS PnL
        'cumulative_oos_pnl': sum(test_pnls),

        # K stability
        'k_mean': np.mean(ks),
        'k_std': np.std(ks),
        'k_values': ks,

        # Correlation between train and test
        'train_test_correlation': np.corrcoef(train_pnls, test_pnls)[0, 1] if len(windows) > 1 else 0,
    }

    return analysis


def print_analysis(windows: list[WalkForwardWindow], analysis: dict) -> None:
    """Print walk-forward analysis."""
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 80)

    # Window details
    print("\n[Window Details]")
    print("-" * 80)
    print(f"{'Window':<8} {'Train Period':<20} {'Test Period':<20} {'k':<5} {'Train PnL':<12} {'Test PnL':<12}")
    print("-" * 80)

    for i, w in enumerate(windows):
        train_period = f"{w.train_start} - {w.train_end}"
        test_period = f"{w.test_start} - {w.test_end}"
        print(f"{i+1:<8} {train_period:<20} {test_period:<20} {w.best_k:<5.1f} {w.train_pnl:>10.1f}% {w.test_pnl:>10.1f}%")

    print("-" * 80)

    # Summary statistics
    print("\n[Summary Statistics]")
    print("-" * 80)
    print(f"{'Metric':<30} {'In-Sample (Train)':<20} {'Out-of-Sample (Test)':<20}")
    print("-" * 80)
    print(f"{'Mean PnL':<30} {analysis['train_pnl_mean']:>18.1f}% {analysis['test_pnl_mean']:>18.1f}%")
    print(f"{'Std PnL':<30} {analysis['train_pnl_std']:>18.1f}% {analysis['test_pnl_std']:>18.1f}%")
    print(f"{'Mean Sharpe':<30} {analysis['train_sharpe_mean']:>18.2f} {analysis['test_sharpe_mean']:>18.2f}")
    print(f"{'Win Rate (>0%)':<30} {analysis['train_win_rate']:>17.1f}% {analysis['test_win_rate']:>17.1f}%")
    print("-" * 80)

    # Overfitting indicators
    print("\n[Overfitting Analysis]")
    print("-" * 80)
    print(f"PnL Degradation (IS - OOS):     {analysis['pnl_degradation']:>10.1f}%")
    print(f"PnL Degradation Ratio:          {analysis['pnl_degradation_pct']:>10.1f}%")
    print(f"Sharpe Degradation:             {analysis['sharpe_degradation']:>10.2f}")
    print(f"Train-Test Correlation:         {analysis['train_test_correlation']:>10.2f}")
    print(f"Cumulative OOS PnL:             {analysis['cumulative_oos_pnl']:>10.1f}%")
    print(f"K Mean ± Std:                   {analysis['k_mean']:.2f} ± {analysis['k_std']:.2f}")
    print("-" * 80)

    # Verdict
    print("\n[Verdict]")

    overfitting_score = 0
    issues = []

    # Check degradation
    if analysis['pnl_degradation_pct'] > 50:
        overfitting_score += 2
        issues.append(f"High PnL degradation ({analysis['pnl_degradation_pct']:.0f}%)")
    elif analysis['pnl_degradation_pct'] > 25:
        overfitting_score += 1
        issues.append(f"Moderate PnL degradation ({analysis['pnl_degradation_pct']:.0f}%)")

    # Check OOS performance
    if analysis['test_pnl_mean'] < 0:
        overfitting_score += 2
        issues.append(f"Negative mean OOS PnL ({analysis['test_pnl_mean']:.1f}%)")
    elif analysis['test_sharpe_mean'] < 0.5:
        overfitting_score += 1
        issues.append(f"Low OOS Sharpe ({analysis['test_sharpe_mean']:.2f})")

    # Check win rate
    if analysis['test_win_rate'] < 50:
        overfitting_score += 1
        issues.append(f"Low OOS win rate ({analysis['test_win_rate']:.0f}%)")

    # Check correlation
    if analysis['train_test_correlation'] < 0.3:
        overfitting_score += 1
        issues.append(f"Low train-test correlation ({analysis['train_test_correlation']:.2f})")

    # Check k stability
    if analysis['k_std'] > 0.5:
        overfitting_score += 1
        issues.append(f"Unstable optimal k (std={analysis['k_std']:.2f})")

    if overfitting_score == 0:
        print("✓ No significant overfitting detected")
        print("  - Consistent performance across windows")
        print("  - Positive OOS returns")
    elif overfitting_score <= 2:
        print("⚠ Mild overfitting concerns:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✗ Significant overfitting detected:")
        for issue in issues:
            print(f"  - {issue}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Validation")
    parser.add_argument('--start', default='2020-01', help='Start month')
    parser.add_argument('--end', default='2025-11', help='End month')
    parser.add_argument('--train-months', type=int, default=12, help='Train window months')
    parser.add_argument('--test-months', type=int, default=3, help='Test window months')
    parser.add_argument('--strategy', default='strategy_db', help='Strategy name')
    parser.add_argument('--fixed-k', type=float, default=None, help='Fixed k (no optimization)')
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR / 'walk_forward_results.csv')

    args = parser.parse_args()

    # Run walk-forward
    windows = run_walk_forward(
        start=args.start,
        end=args.end,
        train_months=args.train_months,
        test_months=args.test_months,
        strategy=args.strategy,
        fixed_k=args.fixed_k,
    )

    if not windows:
        print("No valid windows completed")
        return

    # Analyze results
    analysis = analyze_results(windows)

    # Print analysis
    print_analysis(windows, analysis)

    # Save results
    df = pd.DataFrame([vars(w) for w in windows])
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
