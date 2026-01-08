#!/usr/bin/env python3
"""
V3.4 with T=7, H=7 (low/high 기준) - 4자산 상세 비교
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


def run_backtest(df: pd.DataFrame, trail_base_mult: float = 7.0, hard_stop_mult: float = 7.0):
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

    # Return equity curve for portfolio calculation
    metrics["bar_returns"] = np.zeros(len(metrics["equity_curve"]))
    metrics["bar_returns"][1:] = np.diff(metrics["equity_curve"]) / metrics["equity_curve"][:-1]

    return metrics


def main():
    print("="*120)
    print("V3.4 with T=7, H=7 (low/high 기준) - 4자산 상세")
    print("="*120)

    print("\nLoading data...")
    data = {}
    for symbol in PORTFOLIO_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars")

    # Individual results
    print("\n" + "="*120)
    print("개별 자산 성과")
    print("="*120)
    print(f"{'Symbol':<12} {'Sharpe':<10} {'CAGR':<12} {'PnL':<15} {'MDD':<12} {'CAGR/MDD':<12} {'Trades':<10} {'AvgSize':<10}")
    print("-"*100)

    individual_results = {}
    for symbol, df in data.items():
        metrics = run_backtest(df, trail_base_mult=7.0, hard_stop_mult=7.0)
        cagr_mdd = metrics["cagr"] / metrics["max_drawdown"] if metrics["max_drawdown"] > 0 else 0

        individual_results[symbol] = metrics

        print(f"{symbol:<12} {metrics['sharpe_ratio']:<10.2f} {metrics['cagr']*100:<11.1f}% "
              f"{metrics['total_pnl']*100:<14.1f}% {metrics['max_drawdown']*100:<11.1f}% "
              f"{cagr_mdd:<12.2f} {metrics['total_trades']:<10} {metrics['avg_size']:<10.2f}")

    # Portfolio (leverage stacking)
    print("\n" + "="*120)
    print("4자산 포트폴리오 (레버리지 누적)")
    print("="*120)

    min_len = min(len(m["bar_returns"]) for m in individual_results.values())

    combined_returns = np.zeros(min_len)
    for symbol, m in individual_results.items():
        combined_returns += m["bar_returns"][:min_len]

    initial_capital = 100000.0
    combined_equity = np.zeros(min_len)
    combined_equity[0] = initial_capital

    for i in range(1, min_len):
        combined_equity[i] = combined_equity[i-1] * (1 + combined_returns[i])

    total_pnl = (combined_equity[-1] - initial_capital) / initial_capital

    running_max = np.maximum.accumulate(combined_equity)
    drawdown = (combined_equity - running_max) / running_max
    max_dd = np.abs(np.min(drawdown))

    daily_returns = []
    for i in range(0, len(combined_returns), 6):
        chunk = combined_returns[i:i+6]
        if len(chunk) > 0:
            daily_returns.append(np.sum(chunk))

    if len(daily_returns) > 1:
        sharpe = (np.mean(daily_returns) / (np.std(daily_returns) + 1e-10)) * np.sqrt(252)
    else:
        sharpe = 0.0

    n_years = min_len / (6 * 252)
    if n_years > 0 and combined_equity[-1] > 0:
        cagr = (combined_equity[-1] / initial_capital) ** (1 / n_years) - 1
    else:
        cagr = 0.0

    total_avg_size = sum(m["avg_size"] for m in individual_results.values())

    print(f"  Total PnL:     {total_pnl*100:.1f}%")
    print(f"  CAGR:          {cagr*100:.1f}%")
    print(f"  Max Drawdown:  {max_dd*100:.1f}%")
    print(f"  CAGR/MDD:      {cagr/max_dd:.2f}")
    print(f"  Sharpe:        {sharpe:.2f}")
    print(f"  Avg Leverage:  {total_avg_size:.2f}x")

    # Yearly breakdown
    print("\n" + "="*120)
    print("연도별 수익률")
    print("="*120)
    bars_per_year = 6 * 252
    print(f"{'Year':<10} {'Return':<15} {'MDD':<15}")
    print("-"*40)

    for y in range(int(n_years) + 1):
        start_idx = y * bars_per_year
        end_idx = min((y + 1) * bars_per_year, len(combined_equity))
        if start_idx >= len(combined_equity):
            break

        year_equity = combined_equity[start_idx:end_idx]
        if len(year_equity) > 1:
            year_return = (year_equity[-1] / year_equity[0] - 1) * 100
            peak = np.maximum.accumulate(year_equity)
            dd = (year_equity - peak) / peak
            year_mdd = abs(np.min(dd)) * 100
            print(f"Year {y+1:<5} {year_return:<14.1f}% {year_mdd:<14.1f}%")

    # Compare with close-based T=5, H=5
    print("\n" + "="*120)
    print("비교: close 기준 T=5, H=5 (기존)")
    print("="*120)
    print("  4자산 포트폴리오:")
    print("    Total PnL: 13,060%")
    print("    CAGR: 91.2%")
    print("    MDD: 34.2%")
    print("    CAGR/MDD: 2.66")


if __name__ == "__main__":
    main()
