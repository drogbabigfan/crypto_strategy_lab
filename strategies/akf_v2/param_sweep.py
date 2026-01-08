#!/usr/bin/env python3
"""
Entry/Exit 파라미터 스윕 테스트.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from itertools import product

from research.features.kalman import calculate_adaptive_kalman
from strategies.akf_v2.common import SizingConfig
from strategies.akf_v2.common.sizing import calculate_position_sizes
from strategies.akf_v2.backtest_pyramid import run_pyramid_backtest, PyramidBacktestConfig

DATA_ROOT = PROJECT_ROOT / "etl/data"


def load_data(bar_size: int = 6, start_year: int = 2020, end_year: int = 2025):
    data_dir = DATA_ROOT / f"features-{bar_size}/futures/BTCUSDT"
    dfs = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            path = data_dir / f"BTCUSDT-features-{year}-{month:02d}.parquet"
            if path.exists():
                dfs.append(pd.read_parquet(path))
    return pd.concat(dfs, ignore_index=True) if dfs else None


def generate_signals(df_kf, entry_z, exit_z):
    velocity = df_kf["kf_velocity"].values
    uncertainty = df_kf["kf_uncertainty"].values
    n = len(df_kf)

    zscore_window = 42
    uncertainty_window = 210
    warmup = 210

    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=zscore_window, min_periods=5).mean()
    vel_std = vel_series.rolling(window=zscore_window, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

    p_pct = (
        pd.Series(uncertainty)
        .rolling(window=uncertainty_window, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )
    low_uncertainty = p_pct < 0.5

    entry_long = (vel_zscore > entry_z) & low_uncertainty
    entry_short = (vel_zscore < -2.0) & low_uncertainty

    signals = np.zeros(n, dtype=np.int8)
    position = 0

    for i in range(warmup, n):
        if position == 0:
            if entry_long[i]:
                position = 1
                signals[i] = 1
            elif entry_short[i]:
                position = -1
                signals[i] = -1
        elif position == 1:
            if vel_zscore[i] < exit_z:
                position = 0
            else:
                signals[i] = 1
        elif position == -1:
            if vel_zscore[i] > 2.0:  # V2.1: short exit=+2.0
                position = 0
            else:
                signals[i] = -1

    return signals


def main():
    print("데이터 로딩...")
    df = load_data()
    print(f"  {len(df):,} bars")

    print("Kalman Filter 적용...")
    df_kf = calculate_adaptive_kalman(df)

    sizing_config = SizingConfig(method="kf_prob", min_size=0.0, max_size=2.0)
    bt_config = PyramidBacktestConfig(
        initial_capital=100000.0,
        compounding=True,
        fee_rate=0.001,
        slippage_rate=0.0001,
    )

    entry_range = [1.0, 1.5, 2.0]
    exit_range = [-1.0, -1.5, -2.0]

    print("\n" + "=" * 90)
    print(f"{'Entry':<8} {'Exit':<8} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'Trades':<10} {'WinRate':<10} {'Trades/Yr':<10}")
    print("-" * 90)

    results = []
    for entry_z, exit_z in product(entry_range, exit_range):
        signals = generate_signals(df_kf, entry_z, exit_z)

        df_test = df_kf.copy()
        df_test["signal"] = signals
        df_test["position_size"] = calculate_position_sizes(df_test, sizing_config, entry_signals=signals)
        df_test["sl_price"] = 0.0

        metrics = run_pyramid_backtest(df_test, bt_config)

        trades = metrics["total_trades"]
        trades_per_year = trades / 5.0  # 5년

        results.append({
            "entry": entry_z,
            "exit": exit_z,
            "sharpe": metrics["sharpe_ratio"],
            "pnl": metrics["total_pnl"],
            "mdd": metrics["max_drawdown"],
            "trades": trades,
            "win_rate": metrics["win_rate"],
            "trades_per_year": trades_per_year,
        })

        print(f"{entry_z:<8.1f} {exit_z:<8.1f} {metrics['sharpe_ratio']:<10.2f} "
              f"{metrics['total_pnl']*100:<11.1f}% {metrics['max_drawdown']*100:<9.1f}% "
              f"{trades:<10} {metrics['win_rate']*100:<9.1f}% {trades_per_year:<10.1f}")

    print("=" * 90)

    # 최적 조합 찾기
    best = max(results, key=lambda x: x["sharpe"])
    print(f"\n최고 Sharpe: entry={best['entry']}, exit={best['exit']}, "
          f"Sharpe={best['sharpe']:.2f}, Trades/Yr={best['trades_per_year']:.1f}")


if __name__ == "__main__":
    main()
