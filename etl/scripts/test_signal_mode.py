#!/usr/bin/env python3
"""
Integration test for signal-based exit mode.

Tests the backtester with Long/Short signals where positions
are closed on opposite signals (not TP/SL).
"""

import json
import subprocess
import tempfile
import os
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def create_test_data():
    """
    Create test data with clear signal-based trading scenario:

    Price trend:
    - Bar 0-4: Uptrend (100 -> 120)
    - Bar 5-9: Downtrend (120 -> 100)
    - Bar 10-14: Uptrend again (100 -> 115)

    Signals:
    - Bar 0: Long signal -> Enter at bar 1
    - Bar 5: Short signal -> Exit long, enter short at bar 6
    - Bar 10: Long signal -> Exit short, enter long at bar 11
    - Bar 14: End (force close)

    Expected trades:
    1. Long: Entry ~100, Exit ~120 (+20%)
    2. Short: Entry ~120, Exit ~100 (+16.7%)
    3. Long: Entry ~100, Force close at ~115 (+15%)
    """
    n_bars = 15

    # Price series: uptrend -> downtrend -> uptrend
    prices = [
        100, 105, 110, 115, 120,   # 0-4: Uptrend
        120, 115, 110, 105, 100,   # 5-9: Downtrend
        100, 105, 110, 113, 115,   # 10-14: Uptrend
    ]

    base_time = 1704067200000  # 2024-01-01 00:00:00 UTC

    features = []
    for i in range(n_bars):
        price = prices[i]
        features.append({
            'timestamp': base_time + i * 3600000,
            'open': float(price),
            'high': float(price * 1.01),
            'low': float(price * 0.99),
            'close': float(price),
            'volume': 1000.0,
            'realized_vol': 0.02,  # Not used in signal mode but included
        })

    # Signals: Long at 0, Short at 5, Long at 10
    signals = []
    for i in range(n_bars):
        if i == 0:
            sig = 1   # Long
        elif i == 5:
            sig = -1  # Short
        elif i == 10:
            sig = 1   # Long
        else:
            sig = 0   # Neutral
        signals.append({
            'timestamp': base_time + i * 3600000,
            'signal': sig,
        })

    return features, signals, prices


def save_parquet(data, path, schema_type):
    """Save data to parquet file."""
    df = pd.DataFrame(data)

    if schema_type == 'features':
        schema = pa.schema([
            ('timestamp', pa.int64()),
            ('open', pa.float64()),
            ('high', pa.float64()),
            ('low', pa.float64()),
            ('close', pa.float64()),
            ('volume', pa.float64()),
            ('realized_vol', pa.float64()),
        ])
    else:  # signals
        schema = pa.schema([
            ('timestamp', pa.int64()),
            ('signal', pa.int8()),
        ])

    table = pa.Table.from_pandas(df, schema=schema)
    pq.write_table(table, path)


def run_backtest(features_path, signals_path, exit_mode='signal'):
    """Run the Go backtester and return results."""
    cmd = [
        './bin/backtester',
        '-signals', signals_path,
        '-features', features_path,
        '-exit-mode', exit_mode,
        '-equity',
        '-quiet',
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd='/home/kimhoyeon/dev/dl_rl_btc/etl'
    )

    if result.returncode != 0:
        print(f"Backtest failed: {result.stderr}")
        return None

    return json.loads(result.stdout)


def verify_results(result, prices):
    """Verify the backtest results."""
    print(f"\n{'='*60}")
    print("SIGNAL MODE INTEGRATION TEST")
    print(f"{'='*60}")

    print(f"\nBacktest Summary:")
    print(f"  Total Trades: {result.get('total_trades', 0)}")
    print(f"  Win Rate: {result.get('win_rate', 0)*100:.1f}%")
    print(f"  Avg PnL: {result.get('avg_pnl', 0)*100:.2f}%")
    print(f"  Total PnL: {result.get('total_pnl', 0)*100:.2f}%")
    print(f"  Max Drawdown: {result.get('max_drawdown', 0)*100:.2f}%")
    print(f"  TP Count: {result.get('tp_count', 0)}")
    print(f"  SL Count: {result.get('sl_count', 0)}")
    print(f"  Timeout Count: {result.get('timeout_count', 0)}")

    # Check equity curve
    equity = result.get('equity_curve', [])
    if len(equity) > 0:
        print(f"\nEquity Curve (sampled):")
        print(f"  {'Bar':>4} | {'Price':>8} | {'Equity':>12}")
        print(f"  {'-'*35}")
        for i in [0, 1, 4, 5, 6, 9, 10, 11, 14]:
            if i < len(equity):
                print(f"  {i:>4} | {prices[i]:>8.2f} | {equity[i]:>12.2f}")

    # Verify expected behavior
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")

    all_passed = True

    # Check 1: Should have 3 trades
    expected_trades = 3
    actual_trades = result.get('total_trades', 0)
    check1 = actual_trades == expected_trades
    print(f"  Trade count: {actual_trades} (expected {expected_trades}) - {'OK' if check1 else 'FAIL'}")
    if not check1:
        all_passed = False

    # Check 2: No TP/SL exits (all should be SIGNAL or TIMEOUT for last)
    tp_count = result.get('tp_count', 0)
    sl_count = result.get('sl_count', 0)
    check2 = tp_count == 0 and sl_count == 0
    print(f"  No TP/SL exits: TP={tp_count}, SL={sl_count} - {'OK' if check2 else 'FAIL'}")
    if not check2:
        all_passed = False

    # Check 3: Total PnL should be positive (all trades profitable in this scenario)
    total_pnl = result.get('total_pnl', 0)
    check3 = total_pnl > 0
    print(f"  Positive total PnL: {total_pnl*100:.2f}% - {'OK' if check3 else 'FAIL'}")
    if not check3:
        all_passed = False

    # Check 4: Equity should change during positions (mark-to-market)
    if len(equity) >= 5:
        equity_changes = equity[1] != equity[2] or equity[2] != equity[3]
        print(f"  Equity changes during position: {'YES' if equity_changes else 'NO'} - {'OK' if equity_changes else 'FAIL'}")
        if not equity_changes:
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("INTEGRATION TEST PASSED")
    else:
        print("INTEGRATION TEST FAILED")
    print(f"{'='*60}")

    return all_passed


def main():
    print("Creating test data for signal mode...")
    features, signals, prices = create_test_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        features_path = os.path.join(tmpdir, 'features.parquet')
        signals_path = os.path.join(tmpdir, 'signals.parquet')

        save_parquet(features, features_path, 'features')
        save_parquet(signals, signals_path, 'signals')

        print("Running backtest in signal mode...")
        result = run_backtest(features_path, signals_path, 'signal')

        if result is None:
            print("FAILED: Backtest did not complete")
            return 1

        success = verify_results(result, prices)
        return 0 if success else 1


if __name__ == '__main__':
    exit(main())
