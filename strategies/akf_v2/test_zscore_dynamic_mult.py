#!/usr/bin/env python3
"""
Z-score 기반 Dynamic Multiplier 테스트

공식: Mult = base - slope * (|Z| - threshold)
     = 30 - 5 * |Z|, clipped [10, 20]

Z = 2.0 (평범) → Mult = 20 (여유있게)
Z = 4.0 (과열) → Mult = 10 (타이트하게)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from numba import njit

from research.features.kalman import calculate_adaptive_kalman
from strategies.akf_v2.common import SizingConfig
from strategies.akf_v2.common.sizing import calculate_position_sizes
from strategies.akf_v2.backtest_pyramid import run_pyramid_backtest, PyramidBacktestConfig
from strategies.akf_v2.strategy_v33 import load_data, calculate_v33_features, V33_PARAMS

DATA_ROOT = PROJECT_ROOT / "etl/data"
ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "LTCUSDT"]


@njit
def generate_signals_zscore_mult(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    kf_trend: np.ndarray,
    vel_zscore: np.ndarray,
    unc_pct: np.ndarray,
    sigma_hybrid: np.ndarray,
    v_ratio: np.ndarray,
    resid_std: np.ndarray,
    # Entry params
    entry_z: float = 2.0,
    unc_pct_max: float = 0.5,
    # Z-score based trailing mult: Mult = base - slope * |Z|
    trail_base: float = 30.0,
    trail_slope: float = 5.0,
    trail_mult_min: float = 10.0,
    trail_mult_max: float = 20.0,
    # Innovation params
    v_ratio_threshold: float = 1.0,
    base_innov_mult: float = 2.5,
    innov_mult_min: float = 1.5,
    innov_mult_max: float = 4.0,
    # Hard stop (also z-score based)
    hard_base: float = 15.0,
    hard_slope: float = 2.5,
    hard_mult_min: float = 5.0,
    hard_mult_max: float = 10.0,
    warmup: int = 210,
):
    """
    Z-score 기반 Dynamic Multiplier

    Trailing Mult = trail_base - trail_slope * |Z|, clipped [min, max]
    Hard Mult = hard_base - hard_slope * |Z|, clipped [min, max]
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    stop_prices = np.zeros(n, dtype=np.float64)

    position = 0
    entry_price = 0.0
    entry_z_value = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = np.inf
    prev_trail_stop = 0.0
    hard_stop = 0.0
    entry_trail_mult = 0.0

    trail_exits = 0
    signal_exits = 0
    innovation_exits = 0
    hard_exits = 0

    for i in range(warmup, n):
        low_uncertainty = unc_pct[i] < unc_pct_max
        log_residual = np.log(close[i]) - np.log(kf_trend[i])
        abs_z = abs(vel_zscore[i])

        # Dynamic multipliers based on current Z
        current_trail_mult = trail_base - trail_slope * abs_z
        current_trail_mult = max(trail_mult_min, min(trail_mult_max, current_trail_mult))

        # Dynamic innovation mult (v_ratio based)
        dynamic_innov_mult = base_innov_mult / (v_ratio[i] + 1e-10)
        dynamic_innov_mult = max(innov_mult_min, min(innov_mult_max, dynamic_innov_mult))

        if position == 0:
            if vel_zscore[i] > entry_z and low_uncertainty:
                position = 1
                entry_price = close[i]
                entry_z_value = abs_z
                highest_since_entry = high[i]

                # 진입 시점 Z로 multiplier 결정
                entry_trail_mult = trail_base - trail_slope * entry_z_value
                entry_trail_mult = max(trail_mult_min, min(trail_mult_max, entry_trail_mult))

                entry_hard_mult = hard_base - hard_slope * entry_z_value
                entry_hard_mult = max(hard_mult_min, min(hard_mult_max, entry_hard_mult))

                dist = entry_trail_mult * sigma_hybrid[i]
                prev_trail_stop = highest_since_entry * np.exp(-dist)
                hard_stop = entry_price * np.exp(-entry_hard_mult * sigma_hybrid[i])

                signals[i] = 1
                stop_prices[i] = max(prev_trail_stop, hard_stop)

            elif vel_zscore[i] < -entry_z and low_uncertainty:
                position = -1
                entry_price = close[i]
                entry_z_value = abs_z
                lowest_since_entry = low[i]

                entry_trail_mult = trail_base - trail_slope * entry_z_value
                entry_trail_mult = max(trail_mult_min, min(trail_mult_max, entry_trail_mult))

                entry_hard_mult = hard_base - hard_slope * entry_z_value
                entry_hard_mult = max(hard_mult_min, min(hard_mult_max, entry_hard_mult))

                dist = entry_trail_mult * sigma_hybrid[i]
                prev_trail_stop = lowest_since_entry * np.exp(dist)
                hard_stop = entry_price * np.exp(entry_hard_mult * sigma_hybrid[i])

                signals[i] = -1
                stop_prices[i] = min(prev_trail_stop, hard_stop)

        elif position == 1:
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]

            # Trailing stop: 현재 Z 기반으로 동적 조절
            dist = current_trail_mult * sigma_hybrid[i]
            new_trail_stop = highest_since_entry * np.exp(-dist)
            trail_stop = max(new_trail_stop, prev_trail_stop)
            prev_trail_stop = trail_stop
            stop_prices[i] = max(trail_stop, hard_stop)

            if close[i] < hard_stop:
                position = 0
                hard_exits += 1
                continue
            if close[i] < trail_stop:
                position = 0
                trail_exits += 1
                continue
            if v_ratio[i] >= v_ratio_threshold:
                if log_residual < -dynamic_innov_mult * resid_std[i]:
                    position = 0
                    innovation_exits += 1
                    continue
            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] < -entry_z:
                    position = 0
                    signal_exits += 1
                    continue
            signals[i] = 1

        elif position == -1:
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]

            dist = current_trail_mult * sigma_hybrid[i]
            new_trail_stop = lowest_since_entry * np.exp(dist)
            trail_stop = min(new_trail_stop, prev_trail_stop)
            prev_trail_stop = trail_stop
            stop_prices[i] = min(trail_stop, hard_stop)

            if close[i] > hard_stop:
                position = 0
                hard_exits += 1
                continue
            if close[i] > trail_stop:
                position = 0
                trail_exits += 1
                continue
            if v_ratio[i] >= v_ratio_threshold:
                if log_residual > dynamic_innov_mult * resid_std[i]:
                    position = 0
                    innovation_exits += 1
                    continue
            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] > entry_z:
                    position = 0
                    signal_exits += 1
                    continue
            signals[i] = -1

    return signals, stop_prices, trail_exits, signal_exits, innovation_exits, hard_exits


def run_backtest(df: pd.DataFrame, trail_base=30.0, trail_slope=5.0, hard_base=15.0, hard_slope=2.5):
    """Run backtest with z-score based dynamic mult"""
    df_kf = calculate_adaptive_kalman(df)
    features = calculate_v33_features(df_kf)

    signals, stop_prices, trail_exits, signal_exits, innovation_exits, hard_exits = generate_signals_zscore_mult(
        close=df_kf["close"].values,
        high=df_kf["high"].values,
        low=df_kf["low"].values,
        kf_trend=features["kf_trend"],
        vel_zscore=features["vel_zscore"],
        unc_pct=features["unc_pct"],
        sigma_hybrid=features["sigma_hybrid"],
        v_ratio=features["v_ratio"],
        resid_std=features["resid_std"],
        trail_base=trail_base,
        trail_slope=trail_slope,
        hard_base=hard_base,
        hard_slope=hard_slope,
    )

    df_kf["signal"] = signals
    df_kf["stop_price"] = stop_prices

    sizing_config = SizingConfig(method="fixed", min_size=1.0, max_size=1.0)
    df_kf["position_size"] = calculate_position_sizes(df_kf, sizing_config, entry_signals=signals)
    df_kf["sl_price"] = 0.0

    bt_config = PyramidBacktestConfig(
        initial_capital=100000.0,
        compounding=True,
        fee_rate=0.001,
        slippage_rate=0.0001,
    )

    metrics = run_pyramid_backtest(df_kf, bt_config)
    metrics["trail_exits"] = trail_exits
    metrics["signal_exits"] = signal_exits
    metrics["innovation_exits"] = innovation_exits
    metrics["hard_exits"] = hard_exits

    return metrics


def main():
    print("Loading data...")
    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars")

    # Test configs: (trail_base, trail_slope, hard_base, hard_slope)
    # Trail: Mult = base - slope * |Z|, [10, 20]
    # Hard: Mult = base - slope * |Z|, [5, 10]
    configs = [
        # Original proposal: Z=2→20, Z=4→10
        (30.0, 5.0, 15.0, 2.5, "Z=2→20, Z=4→10 / Hard: Z=2→10, Z=4→5"),
        # Less aggressive
        (25.0, 2.5, 12.5, 1.25, "Z=2→20, Z=4→15 / Hard: Z=2→10, Z=4→7.5"),
        # More aggressive
        (35.0, 7.5, 17.5, 3.75, "Z=2→20, Z=4→5 / Hard: Z=2→10, Z=4→2.5"),
        # Fixed comparison (no Z dependency)
        (15.0, 0.0, 7.5, 0.0, "Fixed: Trail=15, Hard=7.5"),
    ]

    print("\n" + "=" * 160)
    print("Z-score 기반 Dynamic Multiplier 테스트")
    print("Trail Mult = base - slope * |Z|, clipped [10, 20]")
    print("Hard Mult = base - slope * |Z|, clipped [5, 10]")
    print("=" * 160)

    all_results = {}

    for trail_base, trail_slope, hard_base, hard_slope, desc in configs:
        print(f"\n[{desc}]")
        print(f"{'Symbol':<10} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'Trades':<8} {'Trail':<8} {'Signal':<8} {'Innov':<8} {'Hard':<8}")
        print("-" * 100)

        results = []
        for symbol, df in data.items():
            metrics = run_backtest(df, trail_base, trail_slope, hard_base, hard_slope)
            results.append({
                "symbol": symbol,
                "sharpe": metrics["sharpe_ratio"],
                "pnl": metrics["total_pnl"],
                "mdd": metrics["max_drawdown"],
                "trades": metrics["total_trades"],
                "trail_exits": metrics["trail_exits"],
                "signal_exits": metrics["signal_exits"],
                "innovation_exits": metrics["innovation_exits"],
                "hard_exits": metrics["hard_exits"],
            })
            print(f"{symbol:<10} {metrics['sharpe_ratio']:<10.2f} {metrics['total_pnl']*100:<11.1f}% "
                  f"{metrics['max_drawdown']*100:<9.1f}% {metrics['total_trades']:<8} "
                  f"{metrics['trail_exits']:<8} {metrics['signal_exits']:<8} "
                  f"{metrics['innovation_exits']:<8} {metrics['hard_exits']:<8}")

        all_results[desc] = results

        print("-" * 100)
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)
        btc = next(r for r in results if r["symbol"] == "BTCUSDT")
        sol = next(r for r in results if r["symbol"] == "SOLUSDT")

        print(f"Average: Sharpe={avg_sharpe:.2f}, PnL={avg_pnl*100:.1f}%, MDD={avg_mdd*100:.1f}%, Positive={positive}/7")
        print(f"BTC: Sharpe={btc['sharpe']:.2f}, PnL={btc['pnl']*100:.1f}%, MDD={btc['mdd']*100:.1f}%")
        print(f"SOL: Sharpe={sol['sharpe']:.2f}, PnL={sol['pnl']*100:.1f}%, MDD={sol['mdd']*100:.1f}%")

    # Summary
    print("\n" + "=" * 160)
    print("Summary")
    print("=" * 160)

    v33_baseline = {
        "avg_sharpe": 0.55, "avg_pnl": 229.7, "avg_mdd": 54.2,
        "btc_pnl": 505.0, "sol_pnl": 625.4, "btc_mdd": 20.7
    }

    print(f"\nV3.3 Baseline: Sharpe={v33_baseline['avg_sharpe']:.2f}, PnL={v33_baseline['avg_pnl']:.1f}%, "
          f"BTC={v33_baseline['btc_pnl']:.1f}%, SOL={v33_baseline['sol_pnl']:.1f}%")

    print(f"\n{'Config':<50} {'AvgSharpe':<12} {'AvgPnL':<12} {'AvgMDD':<12} {'BTC_PnL':<12} {'SOL_PnL':<12}")
    print("-" * 110)

    for desc, results in all_results.items():
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        btc = next(r for r in results if r["symbol"] == "BTCUSDT")
        sol = next(r for r in results if r["symbol"] == "SOLUSDT")

        print(f"{desc[:48]:<50} {avg_sharpe:<12.2f} {avg_pnl*100:<11.1f}% {avg_mdd*100:<11.1f}% "
              f"{btc['pnl']*100:<11.1f}% {sol['pnl']*100:<11.1f}%")


if __name__ == "__main__":
    main()
