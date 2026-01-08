#!/usr/bin/env python3
"""
AKF Strategy Parameter Sweep.

Tests all strategies across various k values and outputs comparison table.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import numpy as np

from research.features.kalman import calculate_adaptive_kalman
from strategies.akf import AKFStrategies, AKFStrategyConfig
from strategies.akf.strategies_v2 import AKFStrategiesV2, AKFStrategyConfigV2


# Paths
DATA_DIR = PROJECT_ROOT / "etl/data/features-24/futures/BTCUSDT"
OUTPUT_DIR = PROJECT_ROOT / "strategies/akf/results"
BACKTESTER_PATH = PROJECT_ROOT / "etl/bin/backtester"


def generate_months(start_year: int, start_month: int, end_year: int, end_month: int) -> list[str]:
    """Generate list of months in YYYY-MM format."""
    start = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)
    months = []
    current = start
    while current <= end:
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
        raise ValueError(f"No data found for months: {months}")

    return pd.concat(dfs, ignore_index=True)


def run_backtest_direct(
    df_kf: pd.DataFrame,
    strategy: str,
    version: int,
    k: float,
    velocity_threshold: float = 0.0005,
    warmup: int = 130,
) -> Optional[dict]:
    """
    Run backtest directly in Python (without subprocess).

    Returns dict with metrics or None on failure.
    """
    import subprocess
    import tempfile
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        # Generate signals
        if version == 1:
            akf = AKFStrategies(AKFStrategyConfig(k=k))
            result = akf.run_strategy(df_kf.copy(), strategy, include_bands=True)
        else:
            config = AKFStrategyConfigV2(
                db_k=k, benhamou_k=k, fusion_k=k,
                velocity_threshold=velocity_threshold,
            )
            akf = AKFStrategiesV2(config)

            if strategy == "strategy_db":
                result = akf.run_strategy_db(df_kf.copy(), k=k)
            elif strategy == "strategy_benhamou":
                result = akf.run_strategy_benhamou(df_kf.copy(), k=k)
            elif strategy == "strategy_triple_fusion":
                result = akf.run_strategy_triple_fusion(df_kf.copy(), k=k)
            elif strategy == "strategy_velocity_scalp":
                result = akf.run_strategy_velocity_scalp(df_kf.copy(), threshold=velocity_threshold)
            else:
                return None

            if 'position' in result.columns:
                result['signal'] = result['position']

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

            # Parse JSON output
            results = json.loads(proc.stdout)

            return {
                'strategy': strategy,
                'version': version,
                'k': k,
                'vel_thresh': velocity_threshold if strategy == 'strategy_velocity_scalp' else None,
                'trades': results.get('total_trades', 0),
                'win_rate': results.get('win_rate', 0) * 100,
                'avg_pnl': results.get('avg_pnl', 0) * 100,
                'total_pnl': results.get('total_pnl', 0) * 100,
                'sharpe': results.get('sharpe_ratio', 0),
                'mdd': results.get('max_drawdown', 0) * 100,
                'profit_factor': results.get('profit_factor', 0),
                'avg_hold': results.get('avg_hold_bars', 0),
            }

    except Exception as e:
        print(f"Error: {strategy} v{version} k={k}: {e}")
        return None


def run_param_sweep(
    df_kf: pd.DataFrame,
    strategies: list[tuple],
    k_values: list[float],
    velocity_thresholds: list[float] = None,
) -> pd.DataFrame:
    """
    Run parameter sweep across all strategies and k values.

    Args:
        df_kf: DataFrame with Kalman filter outputs
        strategies: List of (strategy_name, version) tuples
        k_values: List of k values to test
        velocity_thresholds: List of velocity thresholds (for velocity_scalp)

    Returns:
        DataFrame with all results
    """
    velocity_thresholds = velocity_thresholds or [0.0005]
    results = []

    total_tests = 0
    for strategy, version in strategies:
        if strategy == 'strategy_velocity_scalp':
            total_tests += len(velocity_thresholds)
        else:
            total_tests += len(k_values)

    current = 0

    for strategy, version in strategies:
        if strategy == 'strategy_velocity_scalp':
            # Test velocity thresholds instead of k
            for vel_thresh in velocity_thresholds:
                current += 1
                print(f"[{current}/{total_tests}] {strategy} v{version} vel_thresh={vel_thresh}")

                result = run_backtest_direct(
                    df_kf, strategy, version, k=2.0,
                    velocity_threshold=vel_thresh
                )
                if result:
                    results.append(result)
        else:
            for k in k_values:
                current += 1
                print(f"[{current}/{total_tests}] {strategy} v{version} k={k}")

                result = run_backtest_direct(df_kf, strategy, version, k)
                if result:
                    results.append(result)

    return pd.DataFrame(results)


def print_results(df: pd.DataFrame) -> None:
    """Print formatted results table."""
    if df.empty:
        print("No results to display")
        return

    # Sort by total_pnl descending
    df_sorted = df.sort_values('total_pnl', ascending=False)

    print("\n" + "=" * 120)
    print("PARAMETER SWEEP RESULTS (sorted by Total PnL)")
    print("=" * 120)

    # Format for display
    display_df = df_sorted[[
        'strategy', 'version', 'k', 'trades', 'win_rate',
        'avg_pnl', 'total_pnl', 'sharpe', 'mdd', 'profit_factor'
    ]].copy()

    display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.1f}%")
    display_df['avg_pnl'] = display_df['avg_pnl'].apply(lambda x: f"{x:.3f}%")
    display_df['total_pnl'] = display_df['total_pnl'].apply(lambda x: f"{x:.1f}%")
    display_df['sharpe'] = display_df['sharpe'].apply(lambda x: f"{x:.2f}")
    display_df['mdd'] = display_df['mdd'].apply(lambda x: f"{x:.1f}%")
    display_df['profit_factor'] = display_df['profit_factor'].apply(lambda x: f"{x:.2f}")

    print(display_df.to_string(index=False))
    print("=" * 120)


def print_best_per_strategy(df: pd.DataFrame) -> None:
    """Print best k value for each strategy."""
    if df.empty:
        return

    print("\n" + "=" * 80)
    print("BEST PARAMETERS PER STRATEGY")
    print("=" * 80)

    for (strategy, version), group in df.groupby(['strategy', 'version']):
        best = group.loc[group['total_pnl'].idxmax()]
        print(f"\n{strategy} (v{version}):")
        print(f"  Best k: {best['k']}")
        print(f"  Total PnL: {best['total_pnl']:.1f}%")
        print(f"  Sharpe: {best['sharpe']:.2f}")
        print(f"  MDD: {best['mdd']:.1f}%")
        print(f"  Trades: {best['trades']}")


def main():
    parser = argparse.ArgumentParser(description="AKF Strategy Parameter Sweep")
    parser.add_argument(
        '--start', default='2020-01',
        help='Start month (YYYY-MM)'
    )
    parser.add_argument(
        '--end', default='2025-11',
        help='End month (YYYY-MM)'
    )
    parser.add_argument(
        '--k-min', type=float, default=0.5,
        help='Minimum k value'
    )
    parser.add_argument(
        '--k-max', type=float, default=3.0,
        help='Maximum k value'
    )
    parser.add_argument(
        '--k-step', type=float, default=0.5,
        help='K value step'
    )
    parser.add_argument(
        '--output', type=Path, default=OUTPUT_DIR / 'param_sweep_results.csv',
        help='Output CSV path'
    )
    parser.add_argument(
        '--v1-only', action='store_true',
        help='Only test V1 strategies'
    )
    parser.add_argument(
        '--v2-only', action='store_true',
        help='Only test V2 strategies'
    )

    args = parser.parse_args()

    # Parse date range
    start_year, start_month = map(int, args.start.split('-'))
    end_year, end_month = map(int, args.end.split('-'))

    # Generate months
    months = generate_months(start_year, start_month, end_year, end_month)
    print(f"Period: {months[0]} to {months[-1]} ({len(months)} months)")

    # Generate k values
    k_values = list(np.arange(args.k_min, args.k_max + args.k_step/2, args.k_step))
    print(f"K values: {k_values}")

    # Define strategies to test
    strategies = []

    if not args.v2_only:
        # V1 strategies
        strategies.extend([
            ('strategy_db', 1),
            ('strategy_benhamou', 1),
            ('strategy_hybrid', 1),
        ])

    if not args.v1_only:
        # V2 strategies
        strategies.extend([
            ('strategy_db', 2),
            ('strategy_benhamou', 2),
            ('strategy_triple_fusion', 2),
            ('strategy_velocity_scalp', 2),
        ])

    print(f"Strategies: {[f'{s} v{v}' for s, v in strategies]}")

    # Load data
    print(f"\nLoading data...")
    df = load_data(months)
    print(f"Loaded {len(df)} bars")

    # Apply Kalman filter
    print(f"Applying Kalman filter...")
    df_kf = calculate_adaptive_kalman(df)

    # Velocity thresholds for velocity_scalp
    velocity_thresholds = [0.0001, 0.0003, 0.0005, 0.0007, 0.001]

    # Run parameter sweep
    print(f"\nRunning parameter sweep...")
    results_df = run_param_sweep(df_kf, strategies, k_values, velocity_thresholds)

    # Print results
    print_results(results_df)
    print_best_per_strategy(results_df)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")

    return results_df


if __name__ == '__main__':
    main()
