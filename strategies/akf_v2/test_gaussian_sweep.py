#!/usr/bin/env python3
"""
Gaussian Decay Position Sizing - Parameter Sweep

S = M × exp(-u²/(2σ²))

Parameters:
- M (max_mult): 1.0, 2.0, 3.0
- σ (sigma): 0.1, 0.2, 0.3, ..., 1.0
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from numba import njit

from research.features.kalman import calculate_adaptive_kalman
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


def calculate_features(df_kf: pd.DataFrame):
    close = df_kf["close"].values
    kf_trend = df_kf["kf_trend"].values
    kf_uncertainty = df_kf["kf_uncertainty"].values
    velocity = df_kf["kf_velocity"].values

    log_returns = np.log(close[1:] / close[:-1])
    log_returns = np.concatenate([[0], log_returns])
    rv = pd.Series(log_returns).rolling(window=42, min_periods=10).std().values
    sqrt_unc = np.sqrt(kf_uncertainty)
    sigma_hybrid = np.maximum(sqrt_unc, rv)

    baseline = pd.Series(sigma_hybrid).rolling(window=120, min_periods=30).mean().values
    v_ratio = sigma_hybrid / (baseline + 1e-10)
    dynamic_mult = np.clip(5.0 / (v_ratio + 1e-10), 1.25, 7.5)

    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=42, min_periods=5).mean()
    vel_std = vel_series.rolling(window=42, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

    unc_pct = (
        pd.Series(kf_uncertainty)
        .rolling(window=210, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )

    log_resid = np.log(close) - np.log(kf_trend)
    resid_std = pd.Series(log_resid).rolling(window=180, min_periods=30).std().fillna(0.01).values

    return {
        "sigma_hybrid": sigma_hybrid,
        "v_ratio": v_ratio,
        "dynamic_mult": dynamic_mult,
        "vel_zscore": vel_zscore,
        "unc_pct": unc_pct,
        "resid_std": resid_std,
        "kf_trend": kf_trend,
    }


@njit
def sizing_gaussian(unc_pct: float, max_mult: float, sigma: float) -> float:
    """Gaussian Decay: S = M × exp(-u²/(2σ²))"""
    return max_mult * np.exp(-unc_pct**2 / (2 * sigma**2))


@njit
def calculate_position_size(
    sigma_hybrid: float, unc_pct: float,
    risk_target: float, hard_stop_mult: float,
    max_mult: float, gauss_sigma: float,
    size_min: float, size_max: float
) -> float:
    hard_stop_dist = hard_stop_mult * sigma_hybrid
    base_size = risk_target / (hard_stop_dist + 1e-10)
    conf_mult = sizing_gaussian(unc_pct, max_mult, gauss_sigma)
    final_size = base_size * conf_mult
    final_size = max(size_min, min(size_max, final_size))
    return final_size


@njit
def generate_signals_with_sizing(
    close, high, low, kf_trend, vel_zscore, unc_pct, sigma_hybrid, dynamic_mult, v_ratio, resid_std,
    entry_z, v_ratio_threshold, base_innov, innov_min, innov_max,
    hard_mult, risk_target, max_mult, gauss_sigma, size_min, size_max, warmup
):
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    position_sizes = np.zeros(n, dtype=np.float64)

    position = 0
    entry_price = 0.0
    current_size = 0.0
    highest = 0.0
    lowest = np.inf
    prev_stop = 0.0
    hard_stop = 0.0

    for i in range(warmup, n):
        log_resid = np.log(close[i]) - np.log(kf_trend[i])
        innov_mult = base_innov / (v_ratio[i] + 1e-10)
        innov_mult = max(innov_min, min(innov_max, innov_mult))

        if position == 0:
            if vel_zscore[i] > entry_z and unc_pct[i] < 0.5:
                position = 1
                entry_price = close[i]
                highest = high[i]
                current_size = calculate_position_size(
                    sigma_hybrid[i], unc_pct[i], risk_target, hard_mult,
                    max_mult, gauss_sigma, size_min, size_max
                )
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = highest * np.exp(-dist)
                hard_stop = entry_price * np.exp(-hard_mult * sigma_hybrid[i])
                signals[i] = 1
                position_sizes[i] = current_size

            elif vel_zscore[i] < -entry_z and unc_pct[i] < 0.5:
                position = -1
                entry_price = close[i]
                lowest = low[i]
                current_size = calculate_position_size(
                    sigma_hybrid[i], unc_pct[i], risk_target, hard_mult,
                    max_mult, gauss_sigma, size_min, size_max
                )
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = lowest * np.exp(dist)
                hard_stop = entry_price * np.exp(hard_mult * sigma_hybrid[i])
                signals[i] = -1
                position_sizes[i] = current_size

        elif position == 1:
            if high[i] > highest:
                highest = high[i]
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_stop = highest * np.exp(-dist)
            trail_stop = max(new_stop, prev_stop)
            prev_stop = trail_stop

            if close[i] < hard_stop:
                position = 0
                continue
            if close[i] < trail_stop:
                position = 0
                continue
            if v_ratio[i] >= v_ratio_threshold:
                if log_resid < -innov_mult * resid_std[i]:
                    position = 0
                    continue
            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] < -entry_z:
                    position = 0
                    continue
            signals[i] = 1
            position_sizes[i] = current_size

        elif position == -1:
            if low[i] < lowest:
                lowest = low[i]
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_stop = lowest * np.exp(dist)
            trail_stop = min(new_stop, prev_stop)
            prev_stop = trail_stop

            if close[i] > hard_stop:
                position = 0
                continue
            if close[i] > trail_stop:
                position = 0
                continue
            if v_ratio[i] >= v_ratio_threshold:
                if log_resid > innov_mult * resid_std[i]:
                    position = 0
                    continue
            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] > entry_z:
                    position = 0
                    continue
            signals[i] = -1
            position_sizes[i] = current_size

    return signals, position_sizes


def calculate_cagr(total_pnl: float, n_bars: int, bars_per_year: float = 365 * 4):
    years = n_bars / bars_per_year
    if years <= 0 or total_pnl <= -1:
        return 0.0
    final_value = 1 + total_pnl
    cagr = (final_value ** (1 / years)) - 1
    return cagr


def run_backtest(df: pd.DataFrame, max_mult: float = 2.0, gauss_sigma: float = 0.3,
                 risk_target: float = 0.02, hard_mult: float = 5.0,
                 size_min: float = 0.1, size_max: float = 3.0):
    df_kf = calculate_adaptive_kalman(df)
    features = calculate_features(df_kf)

    signals, position_sizes = generate_signals_with_sizing(
        close=df_kf["close"].values,
        high=df_kf["high"].values,
        low=df_kf["low"].values,
        kf_trend=features["kf_trend"],
        vel_zscore=features["vel_zscore"],
        unc_pct=features["unc_pct"],
        sigma_hybrid=features["sigma_hybrid"],
        dynamic_mult=features["dynamic_mult"],
        v_ratio=features["v_ratio"],
        resid_std=features["resid_std"],
        entry_z=2.0,
        v_ratio_threshold=1.0,
        base_innov=2.5,
        innov_min=1.5,
        innov_max=4.0,
        hard_mult=hard_mult,
        risk_target=risk_target,
        max_mult=max_mult,
        gauss_sigma=gauss_sigma,
        size_min=size_min,
        size_max=size_max,
        warmup=210,
    )

    df_kf["signal"] = signals
    df_kf["position_size"] = position_sizes
    df_kf["sl_price"] = 0.0

    bt_config = PyramidBacktestConfig(
        initial_capital=100000.0,
        compounding=True,
        fee_rate=0.001,
        slippage_rate=0.0001,
    )

    metrics = run_pyramid_backtest(df_kf, bt_config)
    n_bars = len(df_kf) - 210
    metrics["cagr"] = calculate_cagr(metrics["total_pnl"], n_bars)

    active_sizes = position_sizes[position_sizes > 0]
    metrics["avg_size"] = np.mean(active_sizes) if len(active_sizes) > 0 else 0

    return metrics


def main():
    print("Loading data...")
    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars")

    # Parameter grid
    max_mults = [1.0, 2.0, 3.0]
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    all_results = {}

    for max_mult in max_mults:
        print(f"\n{'#'*120}")
        print(f"# MAX MULTIPLIER = {max_mult}")
        print(f"{'#'*120}")

        for sigma in sigmas:
            desc = f"M={max_mult}, σ={sigma}"
            print(f"\n{'-'*80}")
            print(f"{desc}")
            print("-"*80)

            results = []
            for symbol, df in data.items():
                metrics = run_backtest(df, max_mult, sigma)
                cagr_mdd = metrics["cagr"] / metrics["max_drawdown"] if metrics["max_drawdown"] > 0 else 0
                results.append({
                    "symbol": symbol,
                    "sharpe": metrics["sharpe_ratio"],
                    "cagr": metrics["cagr"],
                    "pnl": metrics["total_pnl"],
                    "mdd": metrics["max_drawdown"],
                    "cagr_mdd": cagr_mdd,
                    "avg_size": metrics["avg_size"],
                })

            all_results[desc] = results

            avg_sharpe = np.mean([r["sharpe"] for r in results])
            avg_cagr = np.mean([r["cagr"] for r in results])
            avg_mdd = np.mean([r["mdd"] for r in results])
            avg_cagr_mdd = np.mean([r["cagr_mdd"] for r in results])
            btc = next(r for r in results if r["symbol"] == "BTCUSDT")
            positive = sum(1 for r in results if r["pnl"] > 0)

            print(f"Avg: Sharpe={avg_sharpe:.2f}, CAGR={avg_cagr*100:.1f}%, MDD={avg_mdd*100:.1f}%, CAGR/MDD={avg_cagr_mdd:.2f}, Pos={positive}/7")
            print(f"BTC: CAGR={btc['cagr']*100:.1f}%, MDD={btc['mdd']*100:.1f}%, CAGR/MDD={btc['cagr_mdd']:.2f}")

    # Summary Tables
    print("\n" + "="*180)
    print("SUMMARY BY MAX MULTIPLIER")
    print("="*180)

    for max_mult in max_mults:
        print(f"\n### M = {max_mult} ###")
        print(f"{'σ':<6} {'AvgSharpe':<10} {'AvgCAGR':<10} {'AvgMDD':<10} {'CAGR/MDD':<10} {'BTC_CAGR':<10} {'BTC_MDD':<10} {'BTC_C/M':<10} {'Pos':<6}")
        print("-"*100)

        for sigma in sigmas:
            desc = f"M={max_mult}, σ={sigma}"
            results = all_results[desc]

            avg_sharpe = np.mean([r["sharpe"] for r in results])
            avg_cagr = np.mean([r["cagr"] for r in results])
            avg_mdd = np.mean([r["mdd"] for r in results])
            avg_cagr_mdd = np.mean([r["cagr_mdd"] for r in results])
            btc = next(r for r in results if r["symbol"] == "BTCUSDT")
            positive = sum(1 for r in results if r["pnl"] > 0)

            print(f"{sigma:<6} {avg_sharpe:<10.2f} {avg_cagr*100:<9.1f}% {avg_mdd*100:<9.1f}% "
                  f"{avg_cagr_mdd:<10.2f} {btc['cagr']*100:<9.1f}% {btc['mdd']*100:<9.1f}% "
                  f"{btc['cagr_mdd']:<10.2f} {positive}/7")

    # Global Best
    print("\n" + "="*180)
    print("TOP 10 CONFIGURATIONS BY CAGR/MDD (MDD < 35%)")
    print("="*180)

    valid_configs = [(desc, results) for desc, results in all_results.items()
                     if np.mean([r["mdd"] for r in results]) < 0.35]

    if valid_configs:
        sorted_configs = sorted(valid_configs,
                               key=lambda x: np.mean([r["cagr_mdd"] for r in x[1]]),
                               reverse=True)[:10]

        print(f"{'Rank':<6} {'Config':<20} {'AvgCAGR':<10} {'AvgMDD':<10} {'CAGR/MDD':<10} {'BTC_CAGR':<10} {'BTC_MDD':<10} {'BTC_C/M':<10}")
        print("-"*100)

        for i, (desc, results) in enumerate(sorted_configs, 1):
            avg_cagr = np.mean([r["cagr"] for r in results])
            avg_mdd = np.mean([r["mdd"] for r in results])
            avg_cagr_mdd = np.mean([r["cagr_mdd"] for r in results])
            btc = next(r for r in results if r["symbol"] == "BTCUSDT")

            print(f"{i:<6} {desc:<20} {avg_cagr*100:<9.1f}% {avg_mdd*100:<9.1f}% "
                  f"{avg_cagr_mdd:<10.2f} {btc['cagr']*100:<9.1f}% {btc['mdd']*100:<9.1f}% {btc['cagr_mdd']:<10.2f}")

    # Per-symbol best
    print("\n" + "="*180)
    print("BEST CONFIG PER SYMBOL (BY CAGR/MDD)")
    print("="*180)

    for symbol in ALL_SYMBOLS:
        best_config = None
        best_cagr_mdd = -999

        for desc, results in all_results.items():
            r = next(x for x in results if x["symbol"] == symbol)
            if r["cagr_mdd"] > best_cagr_mdd and r["mdd"] < 0.40:
                best_cagr_mdd = r["cagr_mdd"]
                best_config = (desc, r)

        if best_config:
            desc, r = best_config
            print(f"{symbol:<10}: {desc:<20} CAGR={r['cagr']*100:.1f}%, MDD={r['mdd']*100:.1f}%, CAGR/MDD={r['cagr_mdd']:.2f}")


if __name__ == "__main__":
    main()
