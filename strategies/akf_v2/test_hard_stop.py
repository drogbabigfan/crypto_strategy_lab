#!/usr/bin/env python3
"""
V3.2 + Hard Stop 테스트

Hard Stop: 진입가 기준 고정 손절
- Long:  hard_stop = entry * exp(-safety_mult * sigma_entry)
- Short: hard_stop = entry * exp(+safety_mult * sigma_entry)
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
from strategies.akf_v2.strategy_v32 import load_data, calculate_v32_features, V32_PARAMS

DATA_ROOT = PROJECT_ROOT / "etl/data"
ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "LTCUSDT"]


@njit
def generate_signals_with_hard_stop(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    vel_zscore: np.ndarray,
    unc_pct: np.ndarray,
    sigma_hybrid: np.ndarray,
    dynamic_mult: np.ndarray,
    v_ratio: np.ndarray,
    entry_z: float = 2.0,
    unc_pct_max: float = 0.5,
    v_ratio_threshold: float = 1.0,
    safety_mult: float = 3.0,
    warmup: int = 210,
):
    """
    V3.2 + Hard Stop

    Exit conditions:
    1. Trailing Stop (고점 기준)
    2. Signal Exit (v_ratio < threshold)
    3. Hard Stop (진입가 기준) - NEW
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    stop_prices = np.zeros(n, dtype=np.float64)

    position = 0
    highest_since_entry = 0.0
    lowest_since_entry = np.inf
    prev_trail_stop = 0.0
    hard_stop = 0.0
    entry_price = 0.0

    trail_exits = 0
    signal_exits = 0
    hard_exits = 0

    for i in range(warmup, n):
        low_uncertainty = unc_pct[i] < unc_pct_max

        if position == 0:
            # Long Entry
            if vel_zscore[i] > entry_z and low_uncertainty:
                position = 1
                entry_price = close[i]
                highest_since_entry = high[i]

                # Trailing stop 초기화
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_trail_stop = highest_since_entry * np.exp(-dist)

                # Hard stop 설정 (진입가 기준)
                hard_stop = entry_price * np.exp(-safety_mult * sigma_hybrid[i])

                signals[i] = 1
                stop_prices[i] = max(prev_trail_stop, hard_stop)

            # Short Entry
            elif vel_zscore[i] < -entry_z and low_uncertainty:
                position = -1
                entry_price = close[i]
                lowest_since_entry = low[i]

                # Trailing stop 초기화
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_trail_stop = lowest_since_entry * np.exp(dist)

                # Hard stop 설정 (진입가 기준)
                hard_stop = entry_price * np.exp(safety_mult * sigma_hybrid[i])

                signals[i] = -1
                stop_prices[i] = min(prev_trail_stop, hard_stop)

        elif position == 1:  # Long
            # Update highest
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]

            # Trailing stop 계산
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_trail_stop = highest_since_entry * np.exp(-dist)
            trail_stop = max(new_trail_stop, prev_trail_stop)  # Ratchet
            prev_trail_stop = trail_stop

            # 실제 stop은 trailing과 hard 중 높은 것
            effective_stop = max(trail_stop, hard_stop)
            stop_prices[i] = effective_stop

            # Hard Stop 체크 (우선)
            if close[i] < hard_stop:
                position = 0
                hard_exits += 1
                continue

            # Trailing Stop 체크
            if close[i] < trail_stop:
                position = 0
                trail_exits += 1
                continue

            # Signal Exit (v_ratio < threshold)
            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] < -entry_z:
                    position = 0
                    signal_exits += 1
                    continue

            signals[i] = 1

        elif position == -1:  # Short
            # Update lowest
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]

            # Trailing stop 계산
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_trail_stop = lowest_since_entry * np.exp(dist)
            trail_stop = min(new_trail_stop, prev_trail_stop)  # Ratchet
            prev_trail_stop = trail_stop

            # 실제 stop은 trailing과 hard 중 낮은 것
            effective_stop = min(trail_stop, hard_stop)
            stop_prices[i] = effective_stop

            # Hard Stop 체크 (우선)
            if close[i] > hard_stop:
                position = 0
                hard_exits += 1
                continue

            # Trailing Stop 체크
            if close[i] > trail_stop:
                position = 0
                trail_exits += 1
                continue

            # Signal Exit (v_ratio < threshold)
            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] > entry_z:
                    position = 0
                    signal_exits += 1
                    continue

            signals[i] = -1

    return signals, stop_prices, trail_exits, signal_exits, hard_exits


def run_backtest(df: pd.DataFrame, safety_mult: float = 3.0):
    """Run V3.2 + Hard Stop backtest"""
    df_kf = calculate_adaptive_kalman(df)
    features = calculate_v32_features(df_kf)

    signals, stop_prices, trail_exits, signal_exits, hard_exits = generate_signals_with_hard_stop(
        close=df_kf["close"].values,
        high=df_kf["high"].values,
        low=df_kf["low"].values,
        vel_zscore=features["vel_zscore"],
        unc_pct=features["unc_pct"],
        sigma_hybrid=features["sigma_hybrid"],
        dynamic_mult=features["dynamic_mult"],
        v_ratio=features["v_ratio"],
        entry_z=V32_PARAMS["entry_z"],
        unc_pct_max=V32_PARAMS["unc_pct_max"],
        v_ratio_threshold=V32_PARAMS["v_ratio_threshold"],
        safety_mult=safety_mult,
        warmup=V32_PARAMS["warmup"],
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

    # Test different safety multipliers
    safety_mults = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]

    print("\n" + "=" * 140)
    print("V3.2 + Hard Stop 테스트")
    print("Hard Stop = Entry * exp(-safety_mult * sigma_entry)")
    print("=" * 140)

    summary = []

    for safety_mult in safety_mults:
        print(f"\n[safety_mult = {safety_mult}]")
        print(f"{'Symbol':<10} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'Trades':<8} {'WinRate':<10} {'Trail':<8} {'Signal':<8} {'Hard':<8}")
        print("-" * 100)

        results = []
        for symbol, df in data.items():
            metrics = run_backtest(df, safety_mult=safety_mult)
            results.append({
                "symbol": symbol,
                "sharpe": metrics["sharpe_ratio"],
                "pnl": metrics["total_pnl"],
                "mdd": metrics["max_drawdown"],
                "trades": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
                "trail_exits": metrics["trail_exits"],
                "signal_exits": metrics["signal_exits"],
                "hard_exits": metrics["hard_exits"],
            })
            print(f"{symbol:<10} {metrics['sharpe_ratio']:<10.2f} {metrics['total_pnl']*100:<11.1f}% "
                  f"{metrics['max_drawdown']*100:<9.1f}% {metrics['total_trades']:<8} "
                  f"{metrics['win_rate']*100:<9.1f}% {metrics['trail_exits']:<8} "
                  f"{metrics['signal_exits']:<8} {metrics['hard_exits']:<8}")

        print("-" * 100)
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)
        btc = next(r for r in results if r["symbol"] == "BTCUSDT")

        print(f"Average: Sharpe={avg_sharpe:.2f}, PnL={avg_pnl*100:.1f}%, MDD={avg_mdd*100:.1f}%, Positive={positive}/7")
        print(f"BTC: Sharpe={btc['sharpe']:.2f}, PnL={btc['pnl']*100:.1f}%, MDD={btc['mdd']*100:.1f}%")

        summary.append({
            "safety_mult": safety_mult,
            "avg_sharpe": avg_sharpe,
            "avg_pnl": avg_pnl,
            "avg_mdd": avg_mdd,
            "positive": positive,
            "btc_sharpe": btc["sharpe"],
            "btc_pnl": btc["pnl"],
            "btc_mdd": btc["mdd"],
        })

    # Summary
    print("\n" + "=" * 140)
    print("Summary 비교")
    print("=" * 140)
    print(f"{'SafetyMult':<12} {'AvgSharpe':<12} {'AvgPnL':<12} {'AvgMDD':<12} {'Positive':<10} {'BTC_Sharpe':<12} {'BTC_PnL':<12} {'BTC_MDD':<12}")
    print("-" * 100)
    for s in summary:
        print(f"{s['safety_mult']:<12} {s['avg_sharpe']:<12.2f} {s['avg_pnl']*100:<11.1f}% {s['avg_mdd']*100:<11.1f}% "
              f"{s['positive']}/7       {s['btc_sharpe']:<12.2f} {s['btc_pnl']*100:<11.1f}% {s['btc_mdd']*100:<11.1f}%")

    print("\n비교 기준 (V3.2 without Hard Stop):")
    print("  Avg Sharpe=0.53, AvgMDD=48.1%, Positive=5/7, BTC: Sharpe=1.02, PnL=464.8%, MDD=25.1%")


if __name__ == "__main__":
    main()
