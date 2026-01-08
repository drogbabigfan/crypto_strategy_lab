#!/usr/bin/env python3
"""
Stop Multiplier 최적화 (low/high 기준 스탑)

장중 터치 방식에서 최적의 trail_base_mult, hard_stop_mult 찾기
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
from strategies.akf_v2.strategy_v34_final import load_data, calculate_features, calculate_intensity, V34_PARAMS

PORTFOLIO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"]


@njit
def calculate_position_size(
    sigma_hybrid: float, unc_pct: float,
    risk_target: float, hard_stop_mult: float,
    gauss_max_mult: float, gauss_sigma: float,
    size_min: float, size_max: float
):
    hard_stop_dist = hard_stop_mult * sigma_hybrid
    base_size = risk_target / (hard_stop_dist + 1e-10)
    gauss_scale = gauss_max_mult * np.exp(-unc_pct**2 / (2 * gauss_sigma**2))
    final_size = base_size * gauss_scale
    return max(size_min, min(size_max, final_size))


@njit
def generate_signals(
    close, high, low, kf_trend, vel_zscore, unc_pct, sigma_hybrid, dynamic_mult, v_ratio, resid_std, intensity,
    entry_z, unc_pct_max, v_ratio_threshold, intensity_threshold,
    innov_base_mult, innov_mult_min, innov_mult_max,
    trail_base_mult, trail_mult_min, trail_mult_max,
    hard_stop_mult, risk_target, gauss_max_mult, gauss_sigma, size_min, size_max, warmup
):
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    position_sizes = np.zeros(n, dtype=np.float64)

    position = 0
    current_size = 0.0
    highest = 0.0
    lowest = np.inf
    prev_stop = 0.0
    hard_stop = 0.0
    entry_price = 0.0

    for i in range(warmup, n):
        innov_mult = innov_base_mult / (v_ratio[i] + 1e-10)
        innov_mult = max(innov_mult_min, min(innov_mult_max, innov_mult))
        log_resid = np.log(close[i]) - np.log(kf_trend[i])

        # Dynamic trail mult
        dyn_mult = trail_base_mult / (v_ratio[i] + 1e-10)
        dyn_mult = max(trail_mult_min, min(trail_mult_max, dyn_mult))

        if position == 0:
            if vel_zscore[i] > entry_z and unc_pct[i] < unc_pct_max:
                position = 1
                entry_price = close[i]
                highest = high[i]
                current_size = calculate_position_size(
                    sigma_hybrid[i], unc_pct[i], risk_target, hard_stop_mult,
                    gauss_max_mult, gauss_sigma, size_min, size_max
                )
                dist = dyn_mult * sigma_hybrid[i]
                prev_stop = highest * np.exp(-dist)
                hard_stop = entry_price * np.exp(-hard_stop_mult * sigma_hybrid[i])
                signals[i] = 1
                position_sizes[i] = current_size

            elif vel_zscore[i] < -entry_z and unc_pct[i] < unc_pct_max:
                position = -1
                entry_price = close[i]
                lowest = low[i]
                current_size = calculate_position_size(
                    sigma_hybrid[i], unc_pct[i], risk_target, hard_stop_mult,
                    gauss_max_mult, gauss_sigma, size_min, size_max
                )
                dist = dyn_mult * sigma_hybrid[i]
                prev_stop = lowest * np.exp(dist)
                hard_stop = entry_price * np.exp(hard_stop_mult * sigma_hybrid[i])
                signals[i] = -1
                position_sizes[i] = current_size

        elif position == 1:
            if high[i] > highest:
                highest = high[i]
            dist = dyn_mult * sigma_hybrid[i]
            new_stop = highest * np.exp(-dist)
            trail_stop = max(new_stop, prev_stop)
            prev_stop = trail_stop

            # Low 기준 스탑 체크
            if low[i] < hard_stop:
                position = 0
                continue
            if low[i] < trail_stop:
                position = 0
                continue
            if v_ratio[i] >= v_ratio_threshold:
                if log_resid < -innov_mult * resid_std[i]:
                    position = 0
                    continue
            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] < -entry_z:
                    if intensity[i] < intensity_threshold:
                        position = 0
                        continue

            signals[i] = 1
            position_sizes[i] = current_size

        elif position == -1:
            if low[i] < lowest:
                lowest = low[i]
            dist = dyn_mult * sigma_hybrid[i]
            new_stop = lowest * np.exp(dist)
            trail_stop = min(new_stop, prev_stop)
            prev_stop = trail_stop

            # High 기준 스탑 체크
            if high[i] > hard_stop:
                position = 0
                continue
            if high[i] > trail_stop:
                position = 0
                continue
            if v_ratio[i] >= v_ratio_threshold:
                if log_resid > innov_mult * resid_std[i]:
                    position = 0
                    continue
            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] > entry_z:
                    if intensity[i] < intensity_threshold:
                        position = 0
                        continue

            signals[i] = -1
            position_sizes[i] = current_size

    return signals, position_sizes


def run_backtest(df: pd.DataFrame, trail_base_mult: float, hard_stop_mult: float):
    params = V34_PARAMS.copy()
    params["trail_base_mult"] = trail_base_mult
    params["hard_stop_mult"] = hard_stop_mult

    df_kf = calculate_adaptive_kalman(df)
    features = calculate_features(df_kf, params)
    intensity = calculate_intensity(df, params["intensity_baseline_window"])

    signals, position_sizes = generate_signals(
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
        intensity=intensity,
        entry_z=params["entry_z"],
        unc_pct_max=params["unc_pct_max"],
        v_ratio_threshold=params["v_ratio_threshold"],
        intensity_threshold=params["intensity_threshold"],
        innov_base_mult=params["innov_base_mult"],
        innov_mult_min=params["innov_mult_min"],
        innov_mult_max=params["innov_mult_max"],
        trail_base_mult=trail_base_mult,
        trail_mult_min=params["trail_mult_min"],
        trail_mult_max=params["trail_mult_max"],
        hard_stop_mult=hard_stop_mult,
        risk_target=params["risk_target"],
        gauss_max_mult=params["gauss_max_mult"],
        gauss_sigma=params["gauss_sigma"],
        size_min=params["size_min"],
        size_max=params["size_max"],
        warmup=params["warmup"],
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

    n_bars = len(df_kf) - params["warmup"]
    years = n_bars / (365 * 4)
    if years > 0 and metrics["total_pnl"] > -1:
        metrics["cagr"] = ((1 + metrics["total_pnl"]) ** (1 / years)) - 1
    else:
        metrics["cagr"] = 0

    active_sizes = position_sizes[position_sizes > 0]
    metrics["avg_size"] = np.mean(active_sizes) if len(active_sizes) > 0 else 0

    return metrics


def main():
    print("Loading data...")
    data = {}
    for symbol in PORTFOLIO_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars")

    # 테스트 파라미터
    # 현재: trail_base=5, hard=5
    # 더 넓게: 6, 7, 8, 10
    trail_mults = [5, 6, 7, 8, 10]
    hard_mults = [5, 6, 7, 8, 10]

    all_results = {}

    print("\n" + "="*120)
    print("Stop Multiplier Sweep (low/high 기준)")
    print("="*120)

    for trail_mult in trail_mults:
        for hard_mult in hard_mults:
            desc = f"T={trail_mult}, H={hard_mult}"

            results = []
            for symbol, df in data.items():
                metrics = run_backtest(df, trail_mult, hard_mult)
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

    # Summary
    print(f"\n{'Config':<15} {'AvgSharpe':<10} {'AvgCAGR':<10} {'AvgMDD':<10} {'CAGR/MDD':<10} {'BTC_CAGR':<10} {'BTC_MDD':<10}")
    print("-"*90)

    for desc, results in all_results.items():
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_cagr = np.mean([r["cagr"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        avg_cagr_mdd = np.mean([r["cagr_mdd"] for r in results])
        btc = next(r for r in results if r["symbol"] == "BTCUSDT")

        print(f"{desc:<15} {avg_sharpe:<10.2f} {avg_cagr*100:<9.1f}% {avg_mdd*100:<9.1f}% "
              f"{avg_cagr_mdd:<10.2f} {btc['cagr']*100:<9.1f}% {btc['mdd']*100:<9.1f}%")

    # Best by CAGR/MDD
    print("\n" + "="*120)
    print("TOP 5 BY CAGR/MDD")
    print("="*120)

    sorted_configs = sorted(all_results.items(),
                           key=lambda x: np.mean([r["cagr_mdd"] for r in x[1]]),
                           reverse=True)[:5]

    for i, (desc, results) in enumerate(sorted_configs, 1):
        avg_cagr = np.mean([r["cagr"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        avg_cagr_mdd = np.mean([r["cagr_mdd"] for r in results])
        btc = next(r for r in results if r["symbol"] == "BTCUSDT")

        print(f"{i}. {desc}")
        print(f"   Avg: CAGR={avg_cagr*100:.1f}%, MDD={avg_mdd*100:.1f}%, CAGR/MDD={avg_cagr_mdd:.2f}")
        print(f"   BTC: CAGR={btc['cagr']*100:.1f}%, MDD={btc['mdd']*100:.1f}%, CAGR/MDD={btc['cagr_mdd']:.2f}")
        print()

    # Compare with close-based (baseline)
    print("="*120)
    print("참고: close 기준 (T=5, H=5) 성능")
    print("  BTC: CAGR=29.1%, MDD=17.5%, CAGR/MDD=1.66")
    print("  Avg: CAGR=20.0%, MDD=30.7%")
    print("="*120)


if __name__ == "__main__":
    main()
