#!/usr/bin/env python3
"""
AKF V2.2 Strategy Runner

V2.1 + KF 3σ Dynamic Stop-Loss

Usage:
    python run_v22.py                    # BTC only
    python run_v22.py --all              # All symbols
    python run_v22.py --symbol ETHUSDT   # Specific symbol
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import pandas as pd

from research.features.kalman import calculate_adaptive_kalman
from strategies.akf_v2.common import SizingConfig
from strategies.akf_v2.common.sizing import calculate_position_sizes
from strategies.akf_v2.backtest_pyramid import run_pyramid_backtest, PyramidBacktestConfig

DATA_ROOT = PROJECT_ROOT / "etl/data"

# V2.2 Parameters
V22_PARAMS = {
    "zscore_window": 42,           # 7일
    "uncertainty_window": 210,     # 35일
    "warmup": 210,
    "uncertainty_pct_threshold": 0.5,
    "long_entry": 2.0,
    "long_exit": -2.0,
    "short_entry": -2.0,
    "short_exit": 2.0,
    "stop_mult": 3.0,              # 3σ stop
    "resid_window": 180,           # 30일
    "sizing_method": "fixed",
    "position_size": 1.0,
}

ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "LTCUSDT"]


def load_data(symbol: str, bar_size: int = 6, start_year: int = 2020, end_year: int = 2025):
    """Load data for a symbol."""
    data_dir = DATA_ROOT / f"features-{bar_size}/futures/{symbol}"
    dfs = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            path = data_dir / f"{symbol}-features-{year}-{month:02d}.parquet"
            if path.exists():
                dfs.append(pd.read_parquet(path))
    return pd.concat(dfs, ignore_index=True) if dfs else None


def generate_signals_v22(df_kf, params=None):
    """Generate V2.2 signals with KF 3σ stop."""
    if params is None:
        params = V22_PARAMS

    velocity = df_kf["kf_velocity"].values
    uncertainty = df_kf["kf_uncertainty"].values
    close = df_kf["close"].values
    kf_trend = df_kf["kf_trend"].values
    n = len(df_kf)

    # Log space calculations
    log_price = np.log(close)
    log_trend = np.log(kf_trend)
    log_residuals = log_price - log_trend

    # Rolling std of residuals
    resid_std = pd.Series(log_residuals).rolling(
        window=params["resid_window"], min_periods=30
    ).std().values

    # Velocity z-score
    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=params["zscore_window"], min_periods=5).mean()
    vel_std = vel_series.rolling(window=params["zscore_window"], min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

    # Uncertainty percentile
    unc_pct = (
        pd.Series(uncertainty)
        .rolling(window=params["uncertainty_window"], min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )
    low_uncertainty = unc_pct < params["uncertainty_pct_threshold"]

    # Entry conditions
    entry_long = (vel_zscore > params["long_entry"]) & low_uncertainty
    entry_short = (vel_zscore < params["short_entry"]) & low_uncertainty

    signals = np.zeros(n, dtype=np.int8)
    position = 0
    stop_hits = 0

    for i in range(params["warmup"], n):
        if position == 0:
            if entry_long[i]:
                position = 1
                signals[i] = 1
            elif entry_short[i]:
                position = -1
                signals[i] = -1

        elif position == 1:  # Long
            # KF 3σ Stop
            if params["stop_mult"] is not None and not np.isnan(resid_std[i]):
                stop_threshold = -params["stop_mult"] * resid_std[i]
                if log_residuals[i] < stop_threshold:
                    position = 0
                    stop_hits += 1
                    continue

            if vel_zscore[i] < params["long_exit"]:
                position = 0
            else:
                signals[i] = 1

        elif position == -1:  # Short
            # KF 3σ Stop
            if params["stop_mult"] is not None and not np.isnan(resid_std[i]):
                stop_threshold = params["stop_mult"] * resid_std[i]
                if log_residuals[i] > stop_threshold:
                    position = 0
                    stop_hits += 1
                    continue

            if vel_zscore[i] > params["short_exit"]:
                position = 0
            else:
                signals[i] = -1

    return signals, stop_hits


def run_backtest(df, params=None):
    """Run V2.2 backtest."""
    if params is None:
        params = V22_PARAMS

    df_kf = calculate_adaptive_kalman(df)
    signals, stop_hits = generate_signals_v22(df_kf, params)

    df_kf["signal"] = signals
    sizing_config = SizingConfig(
        method=params["sizing_method"],
        min_size=params["position_size"],
        max_size=params["position_size"],
    )
    df_kf["position_size"] = calculate_position_sizes(df_kf, sizing_config, entry_signals=signals)
    df_kf["sl_price"] = 0.0

    bt_config = PyramidBacktestConfig(
        initial_capital=100000.0,
        compounding=True,
        fee_rate=0.001,
        slippage_rate=0.0001,
    )

    metrics = run_pyramid_backtest(df_kf, bt_config)
    metrics["stop_hits"] = stop_hits
    return metrics, df_kf


def run_buy_hold(df):
    """Calculate Buy & Hold metrics."""
    start_price = df["close"].iloc[0]
    end_price = df["close"].iloc[-1]
    pnl = (end_price - start_price) / start_price

    prices = df["close"].values
    peak = prices[0]
    max_dd = 0
    for p in prices:
        if p > peak:
            peak = p
        dd = (peak - p) / peak
        if dd > max_dd:
            max_dd = dd

    returns = df["close"].pct_change().dropna()
    sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(365 * 6)

    return {"total_pnl": pnl, "max_drawdown": max_dd, "sharpe_ratio": sharpe}


def run_single_symbol(symbol: str, verbose: bool = True):
    """Run backtest for a single symbol."""
    if verbose:
        print(f"\n{'='*80}")
        print(f"AKF V2.2 - {symbol}")
        print(f"{'='*80}")

    df = load_data(symbol)
    if df is None:
        print(f"  No data for {symbol}")
        return None

    if verbose:
        print(f"Data: {len(df):,} bars ({df['open_time'].min()} ~ {df['open_time'].max()})")

    metrics, df_kf = run_backtest(df)
    bh_metrics = run_buy_hold(df)

    if verbose:
        print(f"\n[V2.2 vs Buy&Hold]")
        print(f"{'Metric':<15} {'V2.2':<15} {'Buy&Hold':<15}")
        print(f"{'-'*45}")
        print(f"{'Sharpe':<15} {metrics['sharpe_ratio']:<15.2f} {bh_metrics['sharpe_ratio']:<15.2f}")
        print(f"{'PnL':<15} {metrics['total_pnl']*100:<14.1f}% {bh_metrics['total_pnl']*100:<14.1f}%")
        print(f"{'MDD':<15} {metrics['max_drawdown']*100:<14.1f}% {bh_metrics['max_drawdown']*100:<14.1f}%")
        print(f"{'Trades':<15} {metrics['total_trades']:<15}")
        print(f"{'Win Rate':<15} {metrics['win_rate']*100:<14.1f}%")
        print(f"{'Stop Hits':<15} {metrics['stop_hits']:<15}")

    return {
        "symbol": symbol,
        "bars": len(df),
        "v22": metrics,
        "bh": bh_metrics,
    }


def run_all_symbols():
    """Run backtest for all symbols and show comparison."""
    results = []

    for symbol in ALL_SYMBOLS:
        print(f"Processing {symbol}...", end=" ", flush=True)
        df = load_data(symbol)
        if df is None:
            print("No data")
            continue

        metrics, _ = run_backtest(df)
        bh_metrics = run_buy_hold(df)

        results.append({
            "symbol": symbol,
            "bars": len(df),
            "sharpe": metrics["sharpe_ratio"],
            "pnl": metrics["total_pnl"],
            "mdd": metrics["max_drawdown"],
            "trades": metrics["total_trades"],
            "win_rate": metrics["win_rate"],
            "stop_hits": metrics["stop_hits"],
            "bh_sharpe": bh_metrics["sharpe_ratio"],
            "bh_pnl": bh_metrics["total_pnl"],
            "bh_mdd": bh_metrics["max_drawdown"],
        })
        print(f"Sharpe={metrics['sharpe_ratio']:.2f}, PnL={metrics['total_pnl']*100:.1f}%")

    # Summary table
    print("\n" + "=" * 120)
    print("AKF V2.2 - All Symbols Comparison")
    print("=" * 120)
    print(f"{'Symbol':<10} {'Bars':<10} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'Trades':<8} {'WinRate':<10} {'Stops':<8} {'B&H PnL':<12}")
    print("-" * 120)

    total_sharpe = 0
    for r in results:
        print(f"{r['symbol']:<10} {r['bars']:<10,} {r['sharpe']:<10.2f} {r['pnl']*100:<11.1f}% "
              f"{r['mdd']*100:<9.1f}% {r['trades']:<8} {r['win_rate']*100:<9.1f}% "
              f"{r['stop_hits']:<8} {r['bh_pnl']*100:<11.1f}%")
        total_sharpe += r['sharpe']

    print("-" * 120)
    avg_sharpe = total_sharpe / len(results) if results else 0
    avg_pnl = np.mean([r['pnl'] for r in results]) if results else 0
    avg_mdd = np.mean([r['mdd'] for r in results]) if results else 0
    avg_bh_pnl = np.mean([r['bh_pnl'] for r in results]) if results else 0

    print(f"{'Average':<10} {'':<10} {avg_sharpe:<10.2f} {avg_pnl*100:<11.1f}% "
          f"{avg_mdd*100:<9.1f}% {'':<8} {'':<10} {'':<8} {avg_bh_pnl*100:<11.1f}%")
    print("=" * 120)

    # Beat B&H count
    beat_bh = sum(1 for r in results if r['pnl'] > r['bh_pnl'])
    print(f"\nV2.2 > Buy&Hold: {beat_bh}/{len(results)} symbols")

    return results


def main():
    parser = argparse.ArgumentParser(description="AKF V2.2 Strategy Runner")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol to test")
    parser.add_argument("--all", action="store_true", help="Test all symbols")
    args = parser.parse_args()

    if args.all:
        run_all_symbols()
    else:
        run_single_symbol(args.symbol)


if __name__ == "__main__":
    main()
