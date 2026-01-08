#!/usr/bin/env python3
"""
Integration test for mark-to-market equity curve.

This script creates test data with known price movements to verify
that unrealized PnL is correctly tracked in the equity curve.
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
    Create test data where we know exactly what should happen:

    Scenario (with vol=0.02, SL=3*vol=6%, PT=3*vol=6%):
    - Bar 0: Price = 100, Signal = Long (entry at bar 1)
    - Bar 1: Entry at open=100, close=100
    - Bar 2: Close = 96 (-4% unrealized, within SL range)
    - Bar 3: Close = 98 (-2% unrealized)
    - Bar 4: Close = 102 (+2% unrealized)
    - Bar 5: Close = 107 (+7%, exit via TP at 106)
    - Bar 6-9: Flat, no position

    SL = 100 * (1 - 3 * 0.02) = 94
    TP = 100 * (1 + 3 * 0.02) = 106

    Expected equity curve (with 2% risk per trade):
    - Bar 0: 100000 (initial, no position yet)
    - Bar 1: 100000 (just entered, unrealized = 0)
    - Bar 2: 100000 * (1 + (-0.04) * 0.02) = 99920 (-4% unrealized)
    - Bar 3: 100000 * (1 + (-0.02) * 0.02) = 99960 (-2% unrealized)
    - Bar 4: 100000 * (1 + 0.02 * 0.02) = 100040 (+2% unrealized)
    - Bar 5: Exit at TP=106 → realized +6%
    """
    n_bars = 10

    # Create features data
    features = []
    base_time = 1704067200000  # 2024-01-01 00:00:00 UTC

    # Price series designed to stay within barriers until bar 5
    # SL=94, TP=106, so prices must stay between 94 and 106 until exit
    prices = [100, 100, 96, 98, 102, 107, 107, 107, 107, 107]
    # High/Low that don't touch barriers until bar 5
    highs = [101, 101, 97, 100, 104, 108, 108, 108, 108, 108]  # Bar 5 high > 106 triggers TP
    lows = [99, 99, 95, 97, 101, 106, 106, 106, 106, 106]  # Stay above SL=94

    for i in range(n_bars):
        price = prices[i]
        high = highs[i]
        low = lows[i]
        features.append({
            'timestamp': base_time + i * 3600000,  # 1 hour bars
            'open': float(price),
            'high': float(high),
            'high_time': base_time + i * 3600000 + 1800000,  # Mid-bar
            'low': float(low),
            'low_time': base_time + i * 3600000 + 900000,
            'close': float(price),
            'volume': 1000.0,
            'realized_vol': 0.02,  # 2% volatility for barrier calculation
        })

    # Create signals: Long at bar 0 (entry at bar 1)
    signals = []
    for i in range(n_bars):
        signals.append({
            'timestamp': base_time + i * 3600000,
            'signal': 1 if i == 0 else 0,  # Long signal only at bar 0
        })

    return features, signals


def save_parquet(data, path, schema_type):
    """Save data to parquet file."""
    df = pd.DataFrame(data)

    if schema_type == 'features':
        schema = pa.schema([
            ('timestamp', pa.int64()),
            ('open', pa.float64()),
            ('high', pa.float64()),
            ('high_time', pa.int64()),
            ('low', pa.float64()),
            ('low_time', pa.int64()),
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


def run_backtest(features_path, signals_path):
    """Run the Go backtester and return results."""
    cmd = [
        './bin/backtester',
        '-signals', signals_path,
        '-features', features_path,
        '-equity',
        '-sl', '3.0',  # SL at 3 * vol = 6% (with vol=0.02)
        '-pt', '3.0',  # PT at 3 * vol = 6% (with vol=0.02)
        '-max-hold', '20',
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


def verify_equity_curve(result, features):
    """Verify that equity curve reflects unrealized PnL."""
    equity = result.get('equity_curve', [])

    if len(equity) == 0:
        print("ERROR: No equity curve returned")
        return False

    print(f"\n{'='*60}")
    print("EQUITY CURVE VERIFICATION")
    print(f"{'='*60}")
    print(f"{'Bar':>4} | {'Close':>8} | {'Equity':>12} | {'Expected':>12} | {'Status'}")
    print(f"{'-'*60}")

    initial_capital = 100000.0
    risk_per_trade = 0.02
    entry_price = 100.0  # Entry at bar 1 open
    tp_price = 106.0  # TP = 100 * (1 + 3 * 0.02) = 106

    all_passed = True

    # Track realized capital after trade
    realized_capital = initial_capital

    for i, (eq, feat) in enumerate(zip(equity, features)):
        close = feat['close']

        # Calculate expected equity based on scenario:
        # - Bar 0: No position yet
        # - Bar 1-4: In position, track unrealized
        # - Bar 5: Exit at TP
        # - Bar 6+: Flat
        if i == 0:
            expected = initial_capital
        elif i >= 1 and i <= 4:
            # During trade - unrealized PnL based on close
            unrealized_pnl = (close - entry_price) / entry_price
            expected = initial_capital * (1 + unrealized_pnl * risk_per_trade)
        elif i == 5:
            # Exit at TP = 106, realized PnL = 6%
            realized_pnl = (tp_price - entry_price) / entry_price
            realized_capital = initial_capital * (1 + realized_pnl * risk_per_trade)
            expected = realized_capital
        else:
            # After exit - flat
            expected = realized_capital

        # Check tolerance (allow for slippage in entry/exit)
        tolerance = 50.0  # $50 tolerance for slippage effects
        passed = abs(eq - expected) < tolerance
        status = "OK" if passed else "FAIL"

        if not passed:
            all_passed = False

        print(f"{i:>4} | {close:>8.2f} | {eq:>12.2f} | {expected:>12.2f} | {status}")

    print(f"{'-'*60}")

    # Check critical points for unrealized PnL tracking
    print("\nCRITICAL CHECKS:")

    # Bar 2 should show -4% unrealized (close=96, entry=100)
    bar2_unrealized = (96 - entry_price) / entry_price  # -4%
    bar2_expected = initial_capital * (1 + bar2_unrealized * risk_per_trade)  # 99920
    bar2_actual = equity[2]
    bar2_ok = abs(bar2_actual - bar2_expected) < 50.0
    print(f"  Bar 2 (-4% unrealized): {bar2_actual:.2f} vs ~{bar2_expected:.2f} - {'OK' if bar2_ok else 'FAIL'}")

    # Bar 3 should show -2% unrealized (close=98, entry=100)
    bar3_unrealized = (98 - entry_price) / entry_price  # -2%
    bar3_expected = initial_capital * (1 + bar3_unrealized * risk_per_trade)  # 99960
    bar3_actual = equity[3]
    bar3_ok = abs(bar3_actual - bar3_expected) < 50.0
    print(f"  Bar 3 (-2% unrealized): {bar3_actual:.2f} vs ~{bar3_expected:.2f} - {'OK' if bar3_ok else 'FAIL'}")

    # Bar 4 should show +2% unrealized (close=102, entry=100)
    bar4_unrealized = (102 - entry_price) / entry_price  # +2%
    bar4_expected = initial_capital * (1 + bar4_unrealized * risk_per_trade)  # 100040
    bar4_actual = equity[4]
    bar4_ok = abs(bar4_actual - bar4_expected) < 50.0
    print(f"  Bar 4 (+2% unrealized): {bar4_actual:.2f} vs ~{bar4_expected:.2f} - {'OK' if bar4_ok else 'FAIL'}")

    # MDD should capture the unrealized drawdown at bar 2
    mdd = result.get('max_drawdown', 0)
    # At bar 2, equity drops to ~99920 from 100000 = 0.08% MDD (approximately)
    # But with slippage, it might be slightly different
    mdd_ok = mdd > 0  # Just check that MDD is captured
    print(f"  MDD captured: {mdd*100:.4f}% - {'OK' if mdd_ok else 'FAIL'}")

    # Most critical: equity should CHANGE during position holding
    # Bar 2 should be different from Bar 3
    equity_changes = equity[2] != equity[3] or equity[3] != equity[4]
    print(f"  Equity changes during position: {'YES' if equity_changes else 'NO'} - {'OK' if equity_changes else 'FAIL'}")

    return bar2_ok and bar3_ok and bar4_ok and equity_changes


def main():
    print("Creating test data...")
    features, signals = create_test_data()

    with tempfile.TemporaryDirectory() as tmpdir:
        features_path = os.path.join(tmpdir, 'features.parquet')
        signals_path = os.path.join(tmpdir, 'signals.parquet')

        save_parquet(features, features_path, 'features')
        save_parquet(signals, signals_path, 'signals')

        print("Running backtest...")
        result = run_backtest(features_path, signals_path)

        if result is None:
            print("FAILED: Backtest did not complete")
            return 1

        print(f"\nBacktest Summary:")
        print(f"  Total Trades: {result.get('total_trades', 0)}")
        print(f"  Win Rate: {result.get('win_rate', 0)*100:.1f}%")
        print(f"  Total PnL: {result.get('total_pnl', 0)*100:.2f}%")
        print(f"  Max Drawdown: {result.get('max_drawdown', 0)*100:.2f}%")

        success = verify_equity_curve(result, features)

        print(f"\n{'='*60}")
        if success:
            print("INTEGRATION TEST PASSED: Equity curve correctly tracks unrealized PnL")
            return 0
        else:
            print("INTEGRATION TEST FAILED: Equity curve does not match expected values")
            return 1


if __name__ == '__main__':
    exit(main())
