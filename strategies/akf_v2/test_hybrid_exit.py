#!/usr/bin/env python3
"""
Hybrid Exit 테스트

- v_ratio < threshold: Signal Exit 허용
- v_ratio >= threshold: Trailing Stop만
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
def calc_signals_hybrid_exit(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    vel_zscore: np.ndarray,
    unc_pct: np.ndarray,
    sigma_hybrid: np.ndarray,
    dynamic_mult: np.ndarray,
    v_ratio: np.ndarray,
    v_ratio_threshold: float = 1.5,
    entry_z: float = 2.0,
    warmup: int = 210,
):
    """
    Hybrid Exit:
    - v_ratio < threshold: Signal Exit OR Trailing Stop
    - v_ratio >= threshold: Trailing Stop만
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    exit_types = np.zeros(n, dtype=np.int8)  # 0=none, 1=trail, 2=signal

    position = 0
    highest_since_entry = 0.0
    lowest_since_entry = np.inf
    prev_stop = 0.0

    trail_exits = 0
    signal_exits = 0

    for i in range(warmup, n):
        low_uncertainty = unc_pct[i] < 0.5

        if position == 0:
            if vel_zscore[i] > entry_z and low_uncertainty:
                position = 1
                highest_since_entry = high[i]
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = highest_since_entry * np.exp(-dist)
                signals[i] = 1

            elif vel_zscore[i] < -entry_z and low_uncertainty:
                position = -1
                lowest_since_entry = low[i]
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = lowest_since_entry * np.exp(dist)
                signals[i] = -1

        elif position == 1:  # Long
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]

            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_stop = highest_since_entry * np.exp(-dist)
            trail_stop = max(new_stop, prev_stop)
            prev_stop = trail_stop

            # Trailing Stop 체크
            if close[i] < trail_stop:
                position = 0
                trail_exits += 1
                exit_types[i] = 1
                continue

            # Signal Exit: v_ratio < threshold 일 때만
            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] < -entry_z:
                    position = 0
                    signal_exits += 1
                    exit_types[i] = 2
                    continue

            signals[i] = 1

        elif position == -1:  # Short
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]

            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_stop = lowest_since_entry * np.exp(dist)
            trail_stop = min(new_stop, prev_stop)
            prev_stop = trail_stop

            # Trailing Stop 체크
            if close[i] > trail_stop:
                position = 0
                trail_exits += 1
                exit_types[i] = 1
                continue

            # Signal Exit: v_ratio < threshold 일 때만
            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] > entry_z:
                    position = 0
                    signal_exits += 1
                    exit_types[i] = 2
                    continue

            signals[i] = -1

    return signals, trail_exits, signal_exits


def calculate_hybrid_features(df_kf: pd.DataFrame, rv_window: int = 42, baseline_window: int = 120, base_mult: float = 5.0):
    """V3.1 Hybrid Volatility features"""
    close = df_kf["close"].values
    kf_uncertainty = df_kf["kf_uncertainty"].values

    # RV (sqrt(6) 제거)
    log_returns = np.log(close[1:] / close[:-1])
    log_returns = np.concatenate([[0], log_returns])
    rv = pd.Series(log_returns).rolling(window=rv_window, min_periods=10).std().values

    # Hybrid Volatility
    sqrt_uncertainty = np.sqrt(kf_uncertainty)
    sigma_hybrid = np.maximum(sqrt_uncertainty, rv)

    # Baseline & V-Ratio
    baseline = pd.Series(sigma_hybrid).rolling(window=baseline_window, min_periods=30).mean().values
    v_ratio = sigma_hybrid / (baseline + 1e-10)

    # Dynamic Multiplier
    dynamic_mult = np.clip(base_mult / (v_ratio + 1e-10), base_mult * 0.25, base_mult * 1.5)

    return {
        "sigma_hybrid": sigma_hybrid,
        "v_ratio": v_ratio,
        "dynamic_mult": dynamic_mult,
    }


def run_backtest(df: pd.DataFrame, v_ratio_threshold: float = 1.5, base_mult: float = 5.0):
    """Run Hybrid Exit backtest"""
    df_kf = calculate_adaptive_kalman(df)
    features = calculate_hybrid_features(df_kf, base_mult=base_mult)

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

    signals, trail_exits, signal_exits = calc_signals_hybrid_exit(
        close=df_kf["close"].values,
        high=df_kf["high"].values,
        low=df_kf["low"].values,
        vel_zscore=vel_zscore,
        unc_pct=unc_pct,
        sigma_hybrid=features["sigma_hybrid"],
        dynamic_mult=features["dynamic_mult"],
        v_ratio=features["v_ratio"],
        v_ratio_threshold=v_ratio_threshold,
        entry_z=2.0,
        warmup=210,
    )

    df_kf["signal"] = signals
    df_kf["v_ratio"] = features["v_ratio"]

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
    metrics["avg_v_ratio"] = np.nanmean(features["v_ratio"][210:])

    return metrics


def main():
    print("Loading data...")
    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars")

    # Test different v_ratio thresholds
    thresholds = [1.0, 1.2, 1.5, 2.0, 999.0]  # 999 = always allow signal exit

    print("\n" + "=" * 130)
    print("Hybrid Exit 테스트: v_ratio < threshold → Signal Exit 허용")
    print("=" * 130)

    summary = []

    for threshold in thresholds:
        label = "Always Signal" if threshold >= 999 else f"v_ratio < {threshold}"
        print(f"\n[{label}]")
        print(f"{'Symbol':<10} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'Trades':<8} {'WinRate':<10} {'TrailExit':<10} {'SignalExit':<10}")
        print("-" * 95)

        results = []
        for symbol, df in data.items():
            metrics = run_backtest(df, v_ratio_threshold=threshold)
            results.append({
                "symbol": symbol,
                "sharpe": metrics["sharpe_ratio"],
                "pnl": metrics["total_pnl"],
                "mdd": metrics["max_drawdown"],
                "trades": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
                "trail_exits": metrics["trail_exits"],
                "signal_exits": metrics["signal_exits"],
            })
            print(f"{symbol:<10} {metrics['sharpe_ratio']:<10.2f} {metrics['total_pnl']*100:<11.1f}% "
                  f"{metrics['max_drawdown']*100:<9.1f}% {metrics['total_trades']:<8} "
                  f"{metrics['win_rate']*100:<9.1f}% {metrics['trail_exits']:<10} {metrics['signal_exits']:<10}")

        print("-" * 95)
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)
        btc = next(r for r in results if r["symbol"] == "BTCUSDT")

        print(f"Average: Sharpe={avg_sharpe:.2f}, PnL={avg_pnl*100:.1f}%, Positive={positive}/7")
        print(f"BTC: Sharpe={btc['sharpe']:.2f}, PnL={btc['pnl']*100:.1f}%")

        summary.append({
            "threshold": label,
            "avg_sharpe": avg_sharpe,
            "avg_pnl": avg_pnl,
            "positive": positive,
            "btc_sharpe": btc["sharpe"],
            "btc_pnl": btc["pnl"],
        })

    # Summary
    print("\n" + "=" * 130)
    print("Summary 비교")
    print("=" * 130)
    print(f"{'Threshold':<20} {'AvgSharpe':<12} {'AvgPnL':<12} {'Positive':<10} {'BTC_Sharpe':<12} {'BTC_PnL':<12}")
    print("-" * 80)
    for s in summary:
        print(f"{s['threshold']:<20} {s['avg_sharpe']:<12.2f} {s['avg_pnl']*100:<11.1f}% {s['positive']}/7       {s['btc_sharpe']:<12.2f} {s['btc_pnl']*100:<11.1f}%")

    # Baseline comparison
    print("\n비교 기준:")
    print("  - Trail Only (v_ratio < 0): Avg Sharpe 0.43, 6/7 positive, BTC 102%")
    print("  - Signal Only (V2.2): Avg Sharpe 0.34, 4/7 positive, BTC 1657%")


if __name__ == "__main__":
    main()
