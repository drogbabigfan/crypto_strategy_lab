#!/usr/bin/env python3
"""
Vol-Scaled Fine Tuning - 좁은 클리핑 범위 집중 탐색
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from research.features.kalman import calculate_adaptive_kalman
from strategies.akf_v2.common import SizingConfig
from strategies.akf_v2.common.sizing import calculate_position_sizes
from strategies.akf_v2.backtest_pyramid import run_pyramid_backtest, PyramidBacktestConfig

DATA_ROOT = PROJECT_ROOT / "etl/data"

ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "LTCUSDT"]


def load_data(symbol: str, bar_size: int = 6):
    data_dir = DATA_ROOT / f"features-{bar_size}/futures/{symbol}"
    dfs = []
    for year in range(2020, 2026):
        for month in range(1, 13):
            path = data_dir / f"{symbol}-features-{year}-{month:02d}.parquet"
            if path.exists():
                dfs.append(pd.read_parquet(path))
    return pd.concat(dfs, ignore_index=True) if dfs else None


def generate_signals_vol_scaled(
    df_kf,
    base_entry_z=2.0,
    vol_clip_min=0.5,
    vol_clip_max=2.0,
    vol_window=180,
    stop_mult=3.0,
):
    velocity = df_kf["kf_velocity"].values
    uncertainty = df_kf["kf_uncertainty"].values
    close = df_kf["close"].values
    kf_trend = df_kf["kf_trend"].values
    n = len(df_kf)

    log_price = np.log(close)
    log_trend = np.log(kf_trend)
    log_residuals = log_price - log_trend

    resid_std = pd.Series(log_residuals).rolling(window=vol_window, min_periods=30).std().values
    vol_mean = pd.Series(resid_std).rolling(window=vol_window, min_periods=30).mean().values

    vel_series = pd.Series(velocity)
    vel_mean_roll = vel_series.rolling(window=42, min_periods=5).mean()
    vel_std_roll = vel_series.rolling(window=42, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean_roll) / (vel_std_roll + 1e-10)).values

    unc_pct = (
        pd.Series(uncertainty)
        .rolling(window=210, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )

    signals = np.zeros(n, dtype=np.int8)
    position = 0
    warmup = 210

    for i in range(warmup, n):
        vol_ratio = resid_std[i] / (vol_mean[i] + 1e-10) if not np.isnan(vol_mean[i]) else 1.0
        vol_ratio = np.clip(vol_ratio, vol_clip_min, vol_clip_max)
        entry_z = base_entry_z * vol_ratio
        exit_z = -base_entry_z * vol_ratio

        low_uncertainty = unc_pct[i] < 0.5

        if position == 0:
            if vel_zscore[i] > entry_z and low_uncertainty:
                position = 1
                signals[i] = 1
            elif vel_zscore[i] < -entry_z and low_uncertainty:
                position = -1
                signals[i] = -1

        elif position == 1:
            if stop_mult is not None and not np.isnan(resid_std[i]):
                if log_residuals[i] < -stop_mult * resid_std[i]:
                    position = 0
                    continue

            if vel_zscore[i] < exit_z:
                position = 0
            else:
                signals[i] = 1

        elif position == -1:
            if stop_mult is not None and not np.isnan(resid_std[i]):
                if log_residuals[i] > stop_mult * resid_std[i]:
                    position = 0
                    continue

            if vel_zscore[i] > -exit_z:
                position = 0
            else:
                signals[i] = -1

    return signals


def run_backtest(df, **params):
    df_kf = calculate_adaptive_kalman(df)
    signals = generate_signals_vol_scaled(df_kf, **params)

    df_kf["signal"] = signals
    sizing_config = SizingConfig(method="fixed", min_size=1.0, max_size=1.0)
    df_kf["position_size"] = calculate_position_sizes(df_kf, sizing_config, entry_signals=signals)
    df_kf["sl_price"] = 0.0

    bt_config = PyramidBacktestConfig(
        initial_capital=100000.0,
        compounding=True,
        fee_rate=0.001,
        slippage_rate=0.0001,
    )

    return run_pyramid_backtest(df_kf, bt_config)


def main():
    print("Loading data...")
    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df

    # Fine-tuning grid
    clip_ranges = [
        (0.9, 1.1),
        (0.85, 1.15),
        (0.8, 1.2),
        (0.75, 1.25),
        (0.7, 1.3),
    ]

    base_z_values = [1.8, 2.0, 2.2]

    results = []

    print("\n" + "=" * 110)
    print("Vol-Scaled Fine Tuning")
    print("=" * 110)

    for base_z in base_z_values:
        for vol_min, vol_max in clip_ranges:
            symbol_results = []
            for symbol, df in data.items():
                metrics = run_backtest(
                    df,
                    base_entry_z=base_z,
                    vol_clip_min=vol_min,
                    vol_clip_max=vol_max,
                )
                symbol_results.append({
                    "symbol": symbol,
                    "sharpe": metrics["sharpe_ratio"],
                    "pnl": metrics["total_pnl"],
                    "mdd": metrics["max_drawdown"],
                })

            avg_sharpe = np.mean([r["sharpe"] for r in symbol_results])
            avg_pnl = np.mean([r["pnl"] for r in symbol_results])
            positive_count = sum(1 for r in symbol_results if r["pnl"] > 0)
            btc = next(r for r in symbol_results if r["symbol"] == "BTCUSDT")

            results.append({
                "base_z": base_z,
                "clip": f"[{vol_min},{vol_max}]",
                "avg_sharpe": avg_sharpe,
                "avg_pnl": avg_pnl,
                "positive": positive_count,
                "btc_sharpe": btc["sharpe"],
                "btc_pnl": btc["pnl"],
                "details": symbol_results,
            })

    # 정렬
    results.sort(key=lambda x: x["avg_sharpe"], reverse=True)

    print(f"\n{'base_z':<8} {'clip':<14} {'Avg Sharpe':<12} {'Avg PnL':<12} {'Positive':<10} {'BTC Sharpe':<12} {'BTC PnL':<12}")
    print("-" * 80)

    for r in results:
        print(f"{r['base_z']:<8} {r['clip']:<14} {r['avg_sharpe']:<12.2f} {r['avg_pnl']*100:<11.1f}% "
              f"{r['positive']}/7       {r['btc_sharpe']:<12.2f} {r['btc_pnl']*100:<11.1f}%")

    # Top 3 상세
    print("\n" + "=" * 80)
    print("Top 3 Configs - Symbol Details")
    print("=" * 80)

    for i, r in enumerate(results[:3]):
        print(f"\n[#{i+1}] base_z={r['base_z']}, clip={r['clip']}")
        print(f"{'Symbol':<12} {'Sharpe':<10} {'PnL':<12} {'MDD':<10}")
        print("-" * 45)
        for sr in r["details"]:
            pnl_str = f"{sr['pnl']*100:.1f}%"
            print(f"{sr['symbol']:<12} {sr['sharpe']:<10.2f} {pnl_str:<12} {sr['mdd']*100:<9.1f}%")


if __name__ == "__main__":
    main()
