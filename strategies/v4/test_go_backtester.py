#!/usr/bin/env python3
"""
Go Backtester Integration Tests

Tests for verifying correct exit price behavior:
1. Long Stop Exit - should exit at stop price
2. Long Signal Exit - should exit at next bar open
3. Short Stop Exit - should exit at stop price
4. Short Signal Exit - should exit at next bar open
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "research/wfa"))

import numpy as np
from go_bridge import GoBridge, BacktestConfig

GO_BACKTESTER_PATH = PROJECT_ROOT / "etl/bin/backtester"


def create_features(prices, highs, lows):
    """Create features array from price data."""
    n = len(prices)
    features = np.zeros((n, 9))
    for i in range(n):
        ts = i * 60000
        features[i] = [ts, prices[i], highs[i], ts+30000, lows[i], ts+45000, prices[i], 1e6, 0.02]
    return features


def test_long_stop_exit():
    """
    Long position, stop hit - should exit at stop price.

    Scenario:
    - Bar 5: Entry signal
    - Bar 6: Entry executed at open=102
    - Bar 10: Low=94 hits stop=95, exit at 95
    - Expected PnL: (95 - 102) / 102 = -6.86%
    """
    print("=" * 60)
    print("Test 1: Long Stop Exit")
    print("=" * 60)

    bridge = GoBridge(str(GO_BACKTESTER_PATH))

    prices = [100, 100, 100, 100, 100, 100, 102, 104, 106, 108, 105, 105, 105, 105, 105]
    lows =   [99,  99,  99,  99,  99,  99,  101, 103, 105, 107, 94,  104, 104, 104, 104]
    highs =  [101, 101, 101, 101, 101, 101, 103, 105, 107, 109, 106, 106, 106, 106, 106]

    features = create_features(prices, highs, lows)
    n_bars = len(prices)

    signals = np.zeros(n_bars, dtype=np.int8)
    signals[5] = 1      # Entry signal
    signals[6:10] = 1   # Maintain position
    signals[10] = 0     # Exit (stop hit)

    sizes = np.ones(n_bars)
    stop_prices = np.zeros(n_bars)
    stop_prices[6:11] = 95.0  # Stop at 95

    config = BacktestConfig(
        exit_mode="custom_stop",
        initial_capital=100000,
        risk_per_trade=1.0,
        base_fee=0,
        base_slippage=0,
    )

    result = bridge.run_backtest_with_data(
        signals=signals, features=features, config=config,
        sizes=sizes, stop_prices=stop_prices, include_equity=True
    )

    # Entry at 102, exit at 95
    expected_pnl = (95 - 102) / 102  # -6.86%

    print(f"  Entry: bar 6 open = 102")
    print(f"  Exit: stop price = 95 (bar 10, low=94 < stop=95)")
    print(f"  Expected PnL: {expected_pnl*100:.2f}%")
    print(f"  Actual PnL: {result.total_pnl*100:.2f}%")
    print(f"  SL count: {result.sl_count} (expected: 1)")

    passed = abs(result.total_pnl - expected_pnl) < 0.001 and result.sl_count == 1
    print(f"  PASS: {passed}")
    return passed


def test_long_signal_exit():
    """
    Long position, signal exit (no stop hit) - should exit at next bar open.

    Scenario:
    - Bar 5: Entry signal
    - Bar 6: Entry executed at open=102
    - Bar 10: Exit signal (Innovation Breaker style), stop_prices=0
    - Bar 11: Exit at open=108
    - Expected PnL: (108 - 102) / 102 = 5.88%
    """
    print("\n" + "=" * 60)
    print("Test 2: Long Signal Exit (next bar open)")
    print("=" * 60)

    bridge = GoBridge(str(GO_BACKTESTER_PATH))

    prices = [100, 100, 100, 100, 100, 100, 102, 104, 106, 108, 110, 108, 108, 108, 108]
    lows =   [99,  99,  99,  99,  99,  99,  101, 103, 105, 107, 109, 107, 107, 107, 107]
    highs =  [101, 101, 101, 101, 101, 101, 103, 105, 107, 109, 111, 109, 109, 109, 109]

    features = create_features(prices, highs, lows)
    n_bars = len(prices)

    signals = np.zeros(n_bars, dtype=np.int8)
    signals[5] = 1      # Entry signal
    signals[6:10] = 1   # Maintain position
    signals[10] = 0     # Exit signal (no stop hit)

    sizes = np.ones(n_bars)
    stop_prices = np.zeros(n_bars)
    stop_prices[6:11] = 0  # No stop (0 for long won't trigger)

    config = BacktestConfig(
        exit_mode="custom_stop",
        initial_capital=100000,
        risk_per_trade=1.0,
        base_fee=0,
        base_slippage=0,
    )

    result = bridge.run_backtest_with_data(
        signals=signals, features=features, config=config,
        sizes=sizes, stop_prices=stop_prices, include_equity=True
    )

    # Entry at 102, exit at next bar open 108
    expected_pnl = (108 - 102) / 102  # 5.88%

    print(f"  Entry: bar 6 open = 102")
    print(f"  Exit: bar 11 open = 108 (signal exit, next bar)")
    print(f"  Expected PnL: {expected_pnl*100:.2f}%")
    print(f"  Actual PnL: {result.total_pnl*100:.2f}%")
    print(f"  SL count: {result.sl_count} (expected: 0)")

    passed = abs(result.total_pnl - expected_pnl) < 0.001 and result.sl_count == 0
    print(f"  PASS: {passed}")
    return passed


def test_short_stop_exit():
    """
    Short position, stop hit - should exit at stop price.

    Scenario:
    - Bar 5: Entry signal (short)
    - Bar 6: Entry executed at open=100
    - Bar 10: High=106 hits stop=105, exit at 105
    - Expected PnL: (100 - 105) / 100 = -5%
    """
    print("\n" + "=" * 60)
    print("Test 3: Short Stop Exit")
    print("=" * 60)

    bridge = GoBridge(str(GO_BACKTESTER_PATH))

    prices = [100, 100, 100, 100, 100, 100, 100, 98, 96, 94, 95, 95, 95, 95, 95]
    lows =   [99,  99,  99,  99,  99,  99,  99,  97, 95, 93, 94, 94, 94, 94, 94]
    highs =  [101, 101, 101, 101, 101, 101, 101, 99, 97, 95, 106, 96, 96, 96, 96]

    features = create_features(prices, highs, lows)
    n_bars = len(prices)

    signals = np.zeros(n_bars, dtype=np.int8)
    signals[5] = -1     # Short entry signal
    signals[6:10] = -1  # Maintain position
    signals[10] = 0     # Exit (stop hit)

    sizes = np.ones(n_bars)
    stop_prices = np.zeros(n_bars)
    stop_prices[6:11] = 105.0  # Stop at 105 for short

    config = BacktestConfig(
        exit_mode="custom_stop",
        initial_capital=100000,
        risk_per_trade=1.0,
        base_fee=0,
        base_slippage=0,
    )

    result = bridge.run_backtest_with_data(
        signals=signals, features=features, config=config,
        sizes=sizes, stop_prices=stop_prices, include_equity=True
    )

    # Entry at 100, exit at 105
    expected_pnl = (100 - 105) / 100  # -5%

    print(f"  Entry: bar 6 open = 100")
    print(f"  Exit: stop price = 105 (bar 10, high=106 > stop=105)")
    print(f"  Expected PnL: {expected_pnl*100:.2f}%")
    print(f"  Actual PnL: {result.total_pnl*100:.2f}%")
    print(f"  SL count: {result.sl_count} (expected: 1)")

    passed = abs(result.total_pnl - expected_pnl) < 0.001 and result.sl_count == 1
    print(f"  PASS: {passed}")
    return passed


def test_short_signal_exit():
    """
    Short position, signal exit (no stop hit) - should exit at next bar open.

    Scenario:
    - Bar 5: Entry signal (short)
    - Bar 6: Entry executed at open=100
    - Bar 10: Exit signal, stop_prices=1e18 (won't trigger)
    - Bar 11: Exit at open=96
    - Expected PnL: (100 - 96) / 100 = 4%
    """
    print("\n" + "=" * 60)
    print("Test 4: Short Signal Exit (next bar open)")
    print("=" * 60)

    bridge = GoBridge(str(GO_BACKTESTER_PATH))

    prices = [100, 100, 100, 100, 100, 100, 100, 98, 96, 94, 95, 96, 96, 96, 96]
    lows =   [99,  99,  99,  99,  99,  99,  99,  97, 95, 93, 94, 95, 95, 95, 95]
    highs =  [101, 101, 101, 101, 101, 101, 101, 99, 97, 95, 96, 97, 97, 97, 97]

    features = create_features(prices, highs, lows)
    n_bars = len(prices)

    signals = np.zeros(n_bars, dtype=np.int8)
    signals[5] = -1     # Short entry signal
    signals[6:10] = -1  # Maintain position
    signals[10] = 0     # Exit signal

    sizes = np.ones(n_bars)
    stop_prices = np.zeros(n_bars)
    stop_prices[6:11] = 1e18  # Won't trigger for short (high < 1e18)

    config = BacktestConfig(
        exit_mode="custom_stop",
        initial_capital=100000,
        risk_per_trade=1.0,
        base_fee=0,
        base_slippage=0,
    )

    result = bridge.run_backtest_with_data(
        signals=signals, features=features, config=config,
        sizes=sizes, stop_prices=stop_prices, include_equity=True
    )

    # Entry at 100, exit at next bar open 96
    expected_pnl = (100 - 96) / 100  # 4%

    print(f"  Entry: bar 6 open = 100")
    print(f"  Exit: bar 11 open = 96 (signal exit, next bar)")
    print(f"  Expected PnL: {expected_pnl*100:.2f}%")
    print(f"  Actual PnL: {result.total_pnl*100:.2f}%")
    print(f"  SL count: {result.sl_count} (expected: 0)")

    passed = abs(result.total_pnl - expected_pnl) < 0.001 and result.sl_count == 0
    print(f"  PASS: {passed}")
    return passed


def test_stop_price_zero_no_trigger():
    """
    Verify that stop_prices=0 for long doesn't trigger false stops.
    Even if bar.Low is very low, stop should NOT trigger.
    """
    print("\n" + "=" * 60)
    print("Test 5: stop_prices=0 should NOT trigger for Long")
    print("=" * 60)

    bridge = GoBridge(str(GO_BACKTESTER_PATH))

    # Low drops to 50 but stop_prices=0, should not trigger
    prices = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 110, 110, 110, 110]
    lows =   [99,  99,  99,  99,  99,  99,  50,  50,  50,  50,  99,  109, 109, 109, 109]
    highs =  [101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 111, 111, 111, 111]

    features = create_features(prices, highs, lows)
    n_bars = len(prices)

    signals = np.zeros(n_bars, dtype=np.int8)
    signals[5] = 1      # Entry signal
    signals[6:10] = 1   # Maintain (with low=50 but stop=0)
    signals[10] = 0     # Exit via signal

    sizes = np.ones(n_bars)
    stop_prices = np.zeros(n_bars)  # All zeros - should never trigger

    config = BacktestConfig(
        exit_mode="custom_stop",
        initial_capital=100000,
        risk_per_trade=1.0,
        base_fee=0,
        base_slippage=0,
    )

    result = bridge.run_backtest_with_data(
        signals=signals, features=features, config=config,
        sizes=sizes, stop_prices=stop_prices, include_equity=True
    )

    # Entry at 100, exit at bar 11 open=110 (not at stop)
    expected_pnl = (110 - 100) / 100  # 10%

    print(f"  Entry: bar 6 open = 100")
    print(f"  Bar 6-9: Low=50 but stop_prices=0 (should NOT trigger)")
    print(f"  Exit: bar 11 open = 110")
    print(f"  Expected PnL: {expected_pnl*100:.2f}%")
    print(f"  Actual PnL: {result.total_pnl*100:.2f}%")
    print(f"  SL count: {result.sl_count} (expected: 0)")

    passed = abs(result.total_pnl - expected_pnl) < 0.001 and result.sl_count == 0
    print(f"  PASS: {passed}")
    return passed


def test_stop_price_1e18_no_trigger():
    """
    Verify that stop_prices=1e18 for short doesn't trigger false stops.
    """
    print("\n" + "=" * 60)
    print("Test 6: stop_prices=1e18 should NOT trigger for Short")
    print("=" * 60)

    bridge = GoBridge(str(GO_BACKTESTER_PATH))

    # High spikes to 150 but stop_prices=1e18, should not trigger
    prices = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 90, 90, 90, 90]
    lows =   [99,  99,  99,  99,  99,  99,  99,  99,  99,  99,  99,  89, 89, 89, 89]
    highs =  [101, 101, 101, 101, 101, 101, 150, 150, 150, 150, 101, 91, 91, 91, 91]

    features = create_features(prices, highs, lows)
    n_bars = len(prices)

    signals = np.zeros(n_bars, dtype=np.int8)
    signals[5] = -1     # Short entry signal
    signals[6:10] = -1  # Maintain (with high=150 but stop=1e18)
    signals[10] = 0     # Exit via signal

    sizes = np.ones(n_bars)
    stop_prices = np.full(n_bars, 1e18)  # All 1e18 - should never trigger for short

    config = BacktestConfig(
        exit_mode="custom_stop",
        initial_capital=100000,
        risk_per_trade=1.0,
        base_fee=0,
        base_slippage=0,
    )

    result = bridge.run_backtest_with_data(
        signals=signals, features=features, config=config,
        sizes=sizes, stop_prices=stop_prices, include_equity=True
    )

    # Entry at 100, exit at bar 11 open=90
    expected_pnl = (100 - 90) / 100  # 10%

    print(f"  Entry: bar 6 open = 100")
    print(f"  Bar 6-9: High=150 but stop_prices=1e18 (should NOT trigger)")
    print(f"  Exit: bar 11 open = 90")
    print(f"  Expected PnL: {expected_pnl*100:.2f}%")
    print(f"  Actual PnL: {result.total_pnl*100:.2f}%")
    print(f"  SL count: {result.sl_count} (expected: 0)")

    passed = abs(result.total_pnl - expected_pnl) < 0.001 and result.sl_count == 0
    print(f"  PASS: {passed}")
    return passed


def main():
    print("Go Backtester Integration Tests")
    print("=" * 60)

    results = []
    results.append(("Long Stop Exit", test_long_stop_exit()))
    results.append(("Long Signal Exit", test_long_signal_exit()))
    results.append(("Short Stop Exit", test_short_stop_exit()))
    results.append(("Short Signal Exit", test_short_signal_exit()))
    results.append(("stop_prices=0 No Trigger (Long)", test_stop_price_zero_no_trigger()))
    results.append(("stop_prices=1e18 No Trigger (Short)", test_stop_price_1e18_no_trigger()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("ALL TESTS PASSED!")
        return 0
    else:
        print("SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
