#!/usr/bin/env python3
"""
Residual Std 기반 Trailing Stop 테스트

- residual = log(close) - log(kf_trend)  (종가 기준)
- resid_std = rolling std of residual
- Long stop = highest * exp(-k * resid_std)
- Short stop = lowest * exp(k * resid_std)
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


@njit
def calc_signals_resid_std_trail(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    kf_trend: np.ndarray,
    vel_zscore: np.ndarray,
    unc_pct: np.ndarray,
    resid_std: np.ndarray,
    stop_mult: float = 3.0,
    entry_z: float = 2.0,
    warmup: int = 210,
):
    """
    Residual Std 기반 Trailing Stop

    - Long: stop = highest * exp(-k * resid_std)
    - Short: stop = lowest * exp(k * resid_std)
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    stop_prices = np.zeros(n, dtype=np.float64)

    position = 0
    highest_since_entry = 0.0
    lowest_since_entry = np.inf
    prev_stop = 0.0

    for i in range(warmup, n):
        low_uncertainty = unc_pct[i] < 0.5

        if position == 0:
            if vel_zscore[i] > entry_z and low_uncertainty:
                position = 1
                highest_since_entry = high[i]
                dist = stop_mult * resid_std[i]
                prev_stop = highest_since_entry * np.exp(-dist)
                signals[i] = 1
                stop_prices[i] = prev_stop

            elif vel_zscore[i] < -entry_z and low_uncertainty:
                position = -1
                lowest_since_entry = low[i]
                dist = stop_mult * resid_std[i]
                prev_stop = lowest_since_entry * np.exp(dist)
                signals[i] = -1
                stop_prices[i] = prev_stop

        elif position == 1:  # Long
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]

            dist = stop_mult * resid_std[i]
            new_stop = highest_since_entry * np.exp(-dist)

            # Ratchet: Long stop은 절대 내려가면 안됨
            trail_stop = max(new_stop, prev_stop)
            prev_stop = trail_stop
            stop_prices[i] = trail_stop

            if close[i] < trail_stop:
                position = 0
                continue

            signals[i] = 1

        elif position == -1:  # Short
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]

            dist = stop_mult * resid_std[i]
            new_stop = lowest_since_entry * np.exp(dist)

            # Ratchet: Short stop은 절대 올라가면 안됨
            trail_stop = min(new_stop, prev_stop)
            prev_stop = trail_stop
            stop_prices[i] = trail_stop

            if close[i] > trail_stop:
                position = 0
                continue

            signals[i] = -1

    return signals, stop_prices


def run_backtest(df: pd.DataFrame, stop_mult: float = 3.0):
    """Run backtest with resid_std trailing stop"""
    df_kf = calculate_adaptive_kalman(df)

    close = df_kf["close"].values
    kf_trend = df_kf["kf_trend"].values

    # Residual = log(close) - log(trend) (종가 기준)
    log_residual = np.log(close) - np.log(kf_trend)
    resid_std = pd.Series(log_residual).rolling(window=180, min_periods=30).std().fillna(0.01).values

    # Velocity z-score
    velocity = df_kf["kf_velocity"].values
    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=42, min_periods=5).mean()
    vel_std = vel_series.rolling(window=42, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

    # Uncertainty percentile
    uncertainty = df_kf["kf_uncertainty"].values
    unc_pct = (
        pd.Series(uncertainty)
        .rolling(window=210, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )

    signals, stop_prices = calc_signals_resid_std_trail(
        close=close,
        high=df_kf["high"].values,
        low=df_kf["low"].values,
        kf_trend=kf_trend,
        vel_zscore=vel_zscore,
        unc_pct=unc_pct,
        resid_std=resid_std,
        stop_mult=stop_mult,
        entry_z=2.0,
        warmup=210,
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

    # 추가 통계
    metrics["avg_resid_std"] = np.nanmean(resid_std[210:])

    return metrics, df_kf


def main():
    print("Loading data...")
    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars")

    # Test different stop multipliers
    mults = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

    print("\n" + "=" * 120)
    print("Residual Std 기반 Trailing Stop 테스트")
    print("stop = highest * exp(-k * resid_std), resid_std = rolling_std(log(close) - log(trend))")
    print("=" * 120)

    for mult in mults:
        print(f"\n[stop_mult = {mult}]")
        print(f"{'Symbol':<10} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'Trades':<8} {'WinRate':<10} {'AvgResidStd':<12}")
        print("-" * 80)

        results = []
        for symbol, df in data.items():
            metrics, _ = run_backtest(df, stop_mult=mult)
            results.append({
                "symbol": symbol,
                "sharpe": metrics["sharpe_ratio"],
                "pnl": metrics["total_pnl"],
                "mdd": metrics["max_drawdown"],
                "trades": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
                "avg_resid_std": metrics["avg_resid_std"],
            })
            print(f"{symbol:<10} {metrics['sharpe_ratio']:<10.2f} {metrics['total_pnl']*100:<11.1f}% "
                  f"{metrics['max_drawdown']*100:<9.1f}% {metrics['total_trades']:<8} "
                  f"{metrics['win_rate']*100:<9.1f}% {metrics['avg_resid_std']*100:<11.2f}%")

        print("-" * 80)
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)
        btc = next(r for r in results if r["symbol"] == "BTCUSDT")

        print(f"Average: Sharpe={avg_sharpe:.2f}, PnL={avg_pnl*100:.1f}%, Positive={positive}/7")
        print(f"BTC: Sharpe={btc['sharpe']:.2f}, PnL={btc['pnl']*100:.1f}%")

    # Summary comparison
    print("\n" + "=" * 120)
    print("Summary: stop_mult별 비교")
    print("=" * 120)
    print(f"{'Mult':<8} {'AvgSharpe':<12} {'AvgPnL':<12} {'Positive':<10} {'BTC_Sharpe':<12} {'BTC_PnL':<12}")
    print("-" * 70)

    for mult in mults:
        results = []
        for symbol, df in data.items():
            metrics, _ = run_backtest(df, stop_mult=mult)
            results.append({
                "symbol": symbol,
                "sharpe": metrics["sharpe_ratio"],
                "pnl": metrics["total_pnl"],
            })
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)
        btc = next(r for r in results if r["symbol"] == "BTCUSDT")
        print(f"{mult:<8} {avg_sharpe:<12.2f} {avg_pnl*100:<11.1f}% {positive}/7       {btc['sharpe']:<12.2f} {btc['pnl']*100:<11.1f}%")


if __name__ == "__main__":
    main()
