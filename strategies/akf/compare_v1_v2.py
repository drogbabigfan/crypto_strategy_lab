#!/usr/bin/env python3
"""
Compare V1 and V2 AKF strategies.

Runs all strategies on the full period and outputs comparison table.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import subprocess
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd


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


def run_backtest(
    months: list[str],
    strategy: str,
    k: float,
    version: int = 1,
    velocity_threshold: float = 0.0005,
) -> dict:
    """Run a single backtest and return results."""
    cmd = [
        sys.executable, 'strategies/akf/run.py',
        '--months', *months,
        '--strategy', strategy,
        '--k', str(k),
        '--version', str(version),
        '--backtest',
    ]

    if strategy == 'strategy_velocity_scalp':
        cmd.extend(['--velocity-threshold', str(velocity_threshold)])

    print(f"\n{'='*60}")
    print(f"Running: {strategy} (v{version}, k={k})")
    print(f"{'='*60}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None

    # Parse results from output
    lines = result.stdout.split('\n')
    metrics = {}

    for line in lines:
        if 'Total Trades:' in line:
            metrics['trades'] = int(line.split(':')[1].strip())
        elif 'Win Rate:' in line:
            metrics['win_rate'] = float(line.split(':')[1].strip().replace('%', ''))
        elif 'Avg PnL:' in line:
            metrics['avg_pnl'] = float(line.split(':')[1].strip().replace('%', ''))
        elif 'Total PnL:' in line:
            metrics['total_pnl'] = float(line.split(':')[1].strip().replace('%', ''))
        elif 'Sharpe Ratio:' in line:
            metrics['sharpe'] = float(line.split(':')[1].strip())
        elif 'Max Drawdown:' in line:
            metrics['mdd'] = float(line.split(':')[1].strip().replace('%', ''))
        elif 'Profit Factor:' in line:
            metrics['profit_factor'] = float(line.split(':')[1].strip())
        elif 'Avg Hold Bars:' in line:
            metrics['avg_hold'] = float(line.split(':')[1].strip())

    return metrics


def main():
    # Full period
    months = generate_months(2020, 1, 2025, 11)
    print(f"Testing period: {months[0]} to {months[-1]} ({len(months)} months)")

    # Define test configurations
    # (strategy, version, k, velocity_threshold)
    configs = [
        # V1 strategies
        ('strategy_db', 1, 2.0, None),
        ('strategy_benhamou', 1, 2.0, None),
        ('strategy_hybrid', 1, 2.0, None),

        # V2 strategies (same k=2.0 for comparison)
        ('strategy_db', 2, 2.0, None),
        ('strategy_benhamou', 2, 2.0, None),
        ('strategy_triple_fusion', 2, 2.0, None),
        ('strategy_velocity_scalp', 2, 2.0, 0.0005),

        # V2 with default k=1.5/1.0
        ('strategy_db', 2, 1.5, None),
        ('strategy_benhamou', 2, 1.0, None),
        ('strategy_triple_fusion', 2, 1.0, None),
    ]

    results = []

    for strategy, version, k, vel_thresh in configs:
        try:
            metrics = run_backtest(
                months, strategy, k, version,
                velocity_threshold=vel_thresh or 0.0005
            )
            if metrics:
                metrics['strategy'] = strategy
                metrics['version'] = version
                metrics['k'] = k
                results.append(metrics)
        except Exception as e:
            print(f"Failed: {strategy} v{version} k={k}: {e}")

    # Create comparison table
    if results:
        df = pd.DataFrame(results)
        df = df[[
            'strategy', 'version', 'k', 'trades', 'win_rate',
            'avg_pnl', 'total_pnl', 'sharpe', 'mdd', 'profit_factor', 'avg_hold'
        ]]

        print("\n" + "=" * 100)
        print("COMPARISON RESULTS")
        print("=" * 100)
        print(df.to_string(index=False))

        # Save to CSV
        output_path = PROJECT_ROOT / 'strategies/akf/results/v1_v2_comparison.csv'
        df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    main()
