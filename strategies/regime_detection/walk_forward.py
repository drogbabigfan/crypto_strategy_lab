#!/usr/bin/env python3
"""
Walk-Forward Validation for Regime Detection Strategy.

Implements anchored walk-forward analysis:
1. Expanding training window (anchored at start)
2. Fixed-size out-of-sample test window
3. Parameter optimization on in-sample data
4. Performance evaluation on out-of-sample data

Usage:
    python strategies/regime_detection/walk_forward.py
    python strategies/regime_detection/walk_forward.py --train-months 12 --test-months 3
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import itertools
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import json

import numpy as np
import pandas as pd

from strategies.regime_detection.hurst import HurstCalculator
from strategies.regime_detection.strategy import (
    RegimeStrategy,
    RegimeStrategyConfig,
    RegimeBacktester,
)


# Paths
DATA_DIR = PROJECT_ROOT / "etl/data/bars-24/futures/BTCUSDT"
OUTPUT_DIR = PROJECT_ROOT / "strategies/regime_detection/results"


@dataclass
class WFAWindow:
    """Walk-forward analysis window result."""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    # Best parameters
    best_params: Dict[str, Any]

    # In-sample performance
    is_total_return: float
    is_sharpe: float
    is_max_drawdown: float
    is_trades: int
    is_win_rate: float

    # Out-of-sample performance
    oos_total_return: float
    oos_sharpe: float
    oos_max_drawdown: float
    oos_trades: int
    oos_win_rate: float


@dataclass
class WFAConfig:
    """Walk-forward analysis configuration."""
    # Window configuration
    train_months: int = 12
    test_months: int = 3
    anchored: bool = True  # Expanding window if True

    # Parameter grid
    hurst_windows: List[int] = field(default_factory=lambda: [50, 100, 150])
    momentum_entry_zscores: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0])
    mean_revert_entry_zscores: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5])
    stop_loss_atr_mults: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5])

    # Optimization target
    optimization_metric: str = 'sharpe'  # 'sharpe', 'return', 'profit_factor'

    # Data
    symbol: str = 'BTCUSDT'
    warmup_bars: int = 200


def generate_months(start: str, end: str) -> List[str]:
    """Generate list of months between start and end (inclusive)."""
    start_dt = datetime.strptime(start, '%Y-%m')
    end_dt = datetime.strptime(end, '%Y-%m')
    months = []
    current = start_dt
    while current <= end_dt:
        months.append(current.strftime('%Y-%m'))
        current += relativedelta(months=1)
    return months


def load_data(months: List[str], data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and concatenate bar data for specified months."""
    dfs = []

    for month in months:
        path = data_dir / f"{data_dir.name}-bars-{month}.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")

    if not dfs:
        return pd.DataFrame()

    data = pd.concat(dfs, ignore_index=True)

    # Rename columns if needed
    column_mapping = {
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
    }
    data = data.rename(columns=column_mapping)

    # Sort by timestamp
    if 'start_time' in data.columns:
        data = data.sort_values('start_time').reset_index(drop=True)

    return data


def run_backtest(
    df: pd.DataFrame,
    config: RegimeStrategyConfig,
    warmup: int = 200,
) -> Optional[Dict]:
    """
    Run single backtest with given configuration.

    Returns metrics dict or None if failed.
    """
    if len(df) < warmup + 50:
        return None

    try:
        strategy = RegimeStrategy(config)
        signals = strategy.generate_signals(df)

        # Zero out warmup period
        signals.loc[:warmup, 'signal'] = 0

        backtester = RegimeBacktester(
            initial_capital=100000,
            commission_rate=0.001,
            slippage_rate=0.0005,
        )

        results = backtester.run(signals)
        metrics = backtester.metrics

        return {
            'total_return': metrics['total_return_pct'],
            'sharpe': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown_pct'],
            'trades': metrics['total_trades'],
            'win_rate': metrics['win_rate_pct'],
            'profit_factor': metrics['profit_factor'],
        }

    except Exception as e:
        print(f"Backtest error: {e}")
        return None


def generate_param_grid(wfa_config: WFAConfig) -> List[Dict]:
    """Generate parameter combinations for optimization."""
    param_names = [
        'hurst_window',
        'momentum_entry_zscore',
        'mean_revert_entry_zscore',
        'stop_loss_atr_mult',
    ]

    param_values = [
        wfa_config.hurst_windows,
        wfa_config.momentum_entry_zscores,
        wfa_config.mean_revert_entry_zscores,
        wfa_config.stop_loss_atr_mults,
    ]

    combinations = list(itertools.product(*param_values))

    return [
        dict(zip(param_names, combo))
        for combo in combinations
    ]


def params_to_config(params: Dict) -> RegimeStrategyConfig:
    """Convert parameter dict to RegimeStrategyConfig."""
    return RegimeStrategyConfig(
        hurst_window=params.get('hurst_window', 100),
        momentum_entry_zscore=params.get('momentum_entry_zscore', 1.5),
        mean_revert_entry_zscore=params.get('mean_revert_entry_zscore', 2.0),
        stop_loss_atr_mult=params.get('stop_loss_atr_mult', 2.0),
        take_profit_atr_mult=params.get('stop_loss_atr_mult', 2.0) * 1.5,
    )


def optimize_params(
    train_df: pd.DataFrame,
    param_grid: List[Dict],
    wfa_config: WFAConfig,
) -> Tuple[Dict, Dict]:
    """
    Optimize parameters on training data.

    Returns (best_params, best_metrics)
    """
    best_params = param_grid[0]
    best_score = float('-inf')
    best_metrics = None

    for params in param_grid:
        config = params_to_config(params)
        metrics = run_backtest(train_df, config, wfa_config.warmup_bars)

        if metrics is None:
            continue

        # Get optimization score
        if wfa_config.optimization_metric == 'sharpe':
            score = metrics['sharpe']
        elif wfa_config.optimization_metric == 'return':
            score = metrics['total_return']
        elif wfa_config.optimization_metric == 'profit_factor':
            score = metrics['profit_factor']
        else:
            score = metrics['sharpe']

        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics

    return best_params, best_metrics


def run_walk_forward(
    start: str,
    end: str,
    wfa_config: WFAConfig,
) -> List[WFAWindow]:
    """
    Run walk-forward analysis.

    Args:
        start: Start month (YYYY-MM)
        end: End month (YYYY-MM)
        wfa_config: WFA configuration

    Returns:
        List of WFAWindow results
    """
    all_months = generate_months(start, end)
    param_grid = generate_param_grid(wfa_config)
    results = []

    # Calculate windows
    total_months = len(all_months)
    min_train = wfa_config.train_months
    test_size = wfa_config.test_months

    print("\n" + "=" * 70)
    print("WALK-FORWARD ANALYSIS - REGIME DETECTION STRATEGY")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Period: {start} to {end} ({total_months} months)")
    print(f"  Train window: {wfa_config.train_months} months {'(anchored)' if wfa_config.anchored else '(rolling)'}")
    print(f"  Test window: {wfa_config.test_months} months")
    print(f"  Parameter combinations: {len(param_grid)}")
    print(f"  Optimization metric: {wfa_config.optimization_metric}")

    window_id = 0
    current_test_start = min_train

    while current_test_start + test_size <= total_months:
        window_id += 1

        # Define window boundaries
        if wfa_config.anchored:
            train_start_idx = 0
        else:
            train_start_idx = current_test_start - min_train

        train_end_idx = current_test_start - 1
        test_start_idx = current_test_start
        test_end_idx = min(current_test_start + test_size - 1, total_months - 1)

        train_start = all_months[train_start_idx]
        train_end = all_months[train_end_idx]
        test_start = all_months[test_start_idx]
        test_end = all_months[test_end_idx]

        print(f"\n[Window {window_id}]")
        print(f"  Train: {train_start} to {train_end} ({train_end_idx - train_start_idx + 1} months)")
        print(f"  Test:  {test_start} to {test_end} ({test_end_idx - test_start_idx + 1} months)")

        # Load training data
        train_months = generate_months(train_start, train_end)
        train_df = load_data(train_months)

        if train_df.empty:
            print("  Warning: No training data, skipping window")
            current_test_start += test_size
            continue

        print(f"  Train data: {len(train_df)} bars")

        # Optimize parameters
        print("  Optimizing parameters...")
        best_params, train_metrics = optimize_params(train_df, param_grid, wfa_config)

        if train_metrics is None:
            print("  Warning: Optimization failed, skipping window")
            current_test_start += test_size
            continue

        print(f"  Best params: hurst={best_params['hurst_window']}, "
              f"mom_z={best_params['momentum_entry_zscore']}, "
              f"mr_z={best_params['mean_revert_entry_zscore']}, "
              f"sl={best_params['stop_loss_atr_mult']}")
        print(f"  Train: Return={train_metrics['total_return']:.1f}%, "
              f"Sharpe={train_metrics['sharpe']:.2f}, "
              f"Trades={train_metrics['trades']}")

        # Load test data
        test_months = generate_months(test_start, test_end)
        test_df = load_data(test_months)

        if test_df.empty:
            print("  Warning: No test data, skipping window")
            current_test_start += test_size
            continue

        print(f"  Test data: {len(test_df)} bars")

        # Test with best parameters
        test_config = params_to_config(best_params)
        test_metrics = run_backtest(test_df, test_config, wfa_config.warmup_bars)

        if test_metrics is None:
            print("  Warning: Test backtest failed, skipping window")
            current_test_start += test_size
            continue

        print(f"  Test:  Return={test_metrics['total_return']:.1f}%, "
              f"Sharpe={test_metrics['sharpe']:.2f}, "
              f"Trades={test_metrics['trades']}")

        # Store result
        window = WFAWindow(
            window_id=window_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            best_params=best_params,
            is_total_return=train_metrics['total_return'],
            is_sharpe=train_metrics['sharpe'],
            is_max_drawdown=train_metrics['max_drawdown'],
            is_trades=train_metrics['trades'],
            is_win_rate=train_metrics['win_rate'],
            oos_total_return=test_metrics['total_return'],
            oos_sharpe=test_metrics['sharpe'],
            oos_max_drawdown=test_metrics['max_drawdown'],
            oos_trades=test_metrics['trades'],
            oos_win_rate=test_metrics['win_rate'],
        )
        results.append(window)

        current_test_start += test_size

    return results


def analyze_results(windows: List[WFAWindow]) -> Dict:
    """Analyze walk-forward results."""
    if not windows:
        return {}

    is_returns = [w.is_total_return for w in windows]
    oos_returns = [w.oos_total_return for w in windows]
    is_sharpes = [w.is_sharpe for w in windows]
    oos_sharpes = [w.oos_sharpe for w in windows]

    # Parameter stability
    hurst_windows = [w.best_params['hurst_window'] for w in windows]
    mom_zscores = [w.best_params['momentum_entry_zscore'] for w in windows]

    analysis = {
        'num_windows': len(windows),

        # In-sample
        'is_return_mean': np.mean(is_returns),
        'is_return_std': np.std(is_returns),
        'is_sharpe_mean': np.mean(is_sharpes),
        'is_win_rate': sum(1 for r in is_returns if r > 0) / len(is_returns) * 100,

        # Out-of-sample
        'oos_return_mean': np.mean(oos_returns),
        'oos_return_std': np.std(oos_returns),
        'oos_sharpe_mean': np.mean(oos_sharpes),
        'oos_win_rate': sum(1 for r in oos_returns if r > 0) / len(oos_returns) * 100,

        # Cumulative OOS
        'oos_cumulative_return': sum(oos_returns),

        # Degradation
        'return_degradation': np.mean(is_returns) - np.mean(oos_returns),
        'sharpe_degradation': np.mean(is_sharpes) - np.mean(oos_sharpes),

        # Correlation
        'is_oos_correlation': np.corrcoef(is_returns, oos_returns)[0, 1] if len(windows) > 1 else 0,

        # Parameter stability
        'hurst_window_std': np.std(hurst_windows),
        'mom_zscore_std': np.std(mom_zscores),
    }

    return analysis


def print_results(windows: List[WFAWindow], analysis: Dict):
    """Print formatted results."""
    print("\n" + "=" * 80)
    print("WALK-FORWARD ANALYSIS RESULTS")
    print("=" * 80)

    # Window details
    print("\n[Window Performance]")
    print("-" * 80)
    print(f"{'#':<3} {'Train Period':<18} {'Test Period':<18} "
          f"{'IS Return':>10} {'OOS Return':>10} {'OOS Sharpe':>10}")
    print("-" * 80)

    for w in windows:
        train_period = f"{w.train_start}-{w.train_end}"
        test_period = f"{w.test_start}-{w.test_end}"
        print(f"{w.window_id:<3} {train_period:<18} {test_period:<18} "
              f"{w.is_total_return:>9.1f}% {w.oos_total_return:>9.1f}% "
              f"{w.oos_sharpe:>10.2f}")

    print("-" * 80)

    # Summary
    print("\n[Summary Statistics]")
    print("-" * 50)
    print(f"{'Metric':<25} {'In-Sample':>12} {'Out-of-Sample':>12}")
    print("-" * 50)
    print(f"{'Mean Return':<25} {analysis['is_return_mean']:>11.1f}% {analysis['oos_return_mean']:>11.1f}%")
    print(f"{'Std Return':<25} {analysis['is_return_std']:>11.1f}% {analysis['oos_return_std']:>11.1f}%")
    print(f"{'Mean Sharpe':<25} {analysis['is_sharpe_mean']:>12.2f} {analysis['oos_sharpe_mean']:>12.2f}")
    print(f"{'Win Rate (>0%)':<25} {analysis['is_win_rate']:>11.1f}% {analysis['oos_win_rate']:>11.1f}%")
    print("-" * 50)

    # OOS Summary
    print(f"\n{'Cumulative OOS Return':<25} {analysis['oos_cumulative_return']:>11.1f}%")
    print(f"{'Return Degradation':<25} {analysis['return_degradation']:>11.1f}%")
    print(f"{'Sharpe Degradation':<25} {analysis['sharpe_degradation']:>12.2f}")
    print(f"{'IS-OOS Correlation':<25} {analysis['is_oos_correlation']:>12.2f}")

    # Verdict
    print("\n[Overfitting Assessment]")
    print("-" * 50)

    issues = []
    if analysis['oos_return_mean'] < 0:
        issues.append(f"Negative mean OOS return ({analysis['oos_return_mean']:.1f}%)")
    if analysis['return_degradation'] > 20:
        issues.append(f"High return degradation ({analysis['return_degradation']:.1f}%)")
    if analysis['oos_sharpe_mean'] < 0.3:
        issues.append(f"Low OOS Sharpe ({analysis['oos_sharpe_mean']:.2f})")
    if analysis['is_oos_correlation'] < 0.2:
        issues.append(f"Low IS-OOS correlation ({analysis['is_oos_correlation']:.2f})")

    if not issues:
        print("✓ Strategy shows robust out-of-sample performance")
    else:
        print("⚠ Potential overfitting indicators:")
        for issue in issues:
            print(f"  - {issue}")

    print("=" * 80)


def save_results(
    windows: List[WFAWindow],
    analysis: Dict,
    output_dir: Path,
):
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save windows to CSV
    windows_data = []
    for w in windows:
        row = {
            'window_id': w.window_id,
            'train_start': w.train_start,
            'train_end': w.train_end,
            'test_start': w.test_start,
            'test_end': w.test_end,
            'is_return': w.is_total_return,
            'is_sharpe': w.is_sharpe,
            'is_mdd': w.is_max_drawdown,
            'is_trades': w.is_trades,
            'oos_return': w.oos_total_return,
            'oos_sharpe': w.oos_sharpe,
            'oos_mdd': w.oos_max_drawdown,
            'oos_trades': w.oos_trades,
            **{f'param_{k}': v for k, v in w.best_params.items()}
        }
        windows_data.append(row)

    pd.DataFrame(windows_data).to_csv(output_dir / 'wfa_windows.csv', index=False)

    # Save analysis to JSON
    with open(output_dir / 'wfa_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Analysis for Regime Strategy")
    parser.add_argument('--start', default='2020-01', help='Start month (YYYY-MM)')
    parser.add_argument('--end', default='2024-12', help='End month (YYYY-MM)')
    parser.add_argument('--train-months', type=int, default=12, help='Training window months')
    parser.add_argument('--test-months', type=int, default=3, help='Test window months')
    parser.add_argument('--anchored', action='store_true', default=True, help='Use anchored (expanding) window')
    parser.add_argument('--rolling', action='store_true', help='Use rolling window')
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR, help='Output directory')

    args = parser.parse_args()

    # Configure WFA
    wfa_config = WFAConfig(
        train_months=args.train_months,
        test_months=args.test_months,
        anchored=not args.rolling,
        hurst_windows=[50, 100, 150],
        momentum_entry_zscores=[1.0, 1.5, 2.0],
        mean_revert_entry_zscores=[1.5, 2.0, 2.5],
        stop_loss_atr_mults=[1.5, 2.0, 2.5],
    )

    # Run WFA
    windows = run_walk_forward(args.start, args.end, wfa_config)

    if not windows:
        print("\nNo valid windows completed. Check data availability.")
        return

    # Analyze
    analysis = analyze_results(windows)

    # Print results
    print_results(windows, analysis)

    # Save results
    save_results(windows, analysis, args.output)


if __name__ == '__main__':
    main()
