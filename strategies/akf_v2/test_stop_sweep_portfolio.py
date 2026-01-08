#!/usr/bin/env python3
"""
Stop Multiplier 최적화 - 4자산 포트폴리오 기준
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

            # low/high 기준 스탑
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

            # low/high 기준 스탑
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

    metrics["bar_returns"] = np.zeros(len(metrics["equity_curve"]))
    metrics["bar_returns"][1:] = np.diff(metrics["equity_curve"]) / metrics["equity_curve"][:-1]

    return metrics


def calculate_portfolio_metrics(individual_results):
    """4자산 레버리지 누적 포트폴리오 계산"""
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

    return {
        "total_pnl": total_pnl,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "avg_leverage": total_avg_size,
        "cagr_mdd": cagr / max_dd if max_dd > 0 else 0,
    }


def main():
    print("Loading data...")
    data = {}
    for symbol in PORTFOLIO_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars")

    # 더 세밀한 sweep
    trail_mults = [5, 5.5, 6, 6.5, 7, 7.5, 8]
    hard_mults = [5, 5.5, 6, 6.5, 7, 7.5, 8]

    all_results = {}

    print("\n" + "="*140)
    print("Stop Multiplier Sweep - 포트폴리오 기준")
    print("="*140)

    for trail_mult in trail_mults:
        for hard_mult in hard_mults:
            desc = f"T={trail_mult}, H={hard_mult}"

            individual_results = {}
            for symbol, df in data.items():
                metrics = run_backtest(df, trail_mult, hard_mult)
                individual_results[symbol] = metrics

            portfolio = calculate_portfolio_metrics(individual_results)

            all_results[desc] = {
                "individual": individual_results,
                "portfolio": portfolio,
            }

    # Summary - Portfolio
    print(f"\n{'Config':<15} {'Port CAGR':<12} {'Port MDD':<12} {'Port C/M':<10} {'Sharpe':<10} {'Leverage':<10}")
    print("-"*80)

    for desc in sorted(all_results.keys()):
        p = all_results[desc]["portfolio"]
        print(f"{desc:<15} {p['cagr']*100:<11.1f}% {p['max_drawdown']*100:<11.1f}% "
              f"{p['cagr_mdd']:<10.2f} {p['sharpe']:<10.2f} {p['avg_leverage']:<10.2f}x")

    # Per-asset breakdown for top configs
    print("\n" + "="*140)
    print("TOP 5 BY Portfolio CAGR/MDD")
    print("="*140)

    sorted_configs = sorted(all_results.items(),
                           key=lambda x: x[1]["portfolio"]["cagr_mdd"],
                           reverse=True)[:5]

    for i, (desc, result) in enumerate(sorted_configs, 1):
        p = result["portfolio"]
        ind = result["individual"]

        print(f"\n{i}. {desc}")
        print(f"   Portfolio: CAGR={p['cagr']*100:.1f}%, MDD={p['max_drawdown']*100:.1f}%, "
              f"CAGR/MDD={p['cagr_mdd']:.2f}, Sharpe={p['sharpe']:.2f}")
        print(f"   {'Symbol':<10} {'CAGR':<10} {'MDD':<10} {'CAGR/MDD':<10}")
        for symbol in PORTFOLIO_SYMBOLS:
            m = ind[symbol]
            cm = m["cagr"] / m["max_drawdown"] if m["max_drawdown"] > 0 else 0
            print(f"   {symbol:<10} {m['cagr']*100:<9.1f}% {m['max_drawdown']*100:<9.1f}% {cm:<10.2f}")

    # Best balanced (minimize worst asset)
    print("\n" + "="*140)
    print("BEST BALANCED (worst asset CAGR/MDD 최대화)")
    print("="*140)

    def get_worst_asset_cagr_mdd(result):
        ind = result["individual"]
        worst = min(
            m["cagr"] / m["max_drawdown"] if m["max_drawdown"] > 0 else 0
            for m in ind.values()
        )
        return worst

    balanced_configs = sorted(all_results.items(),
                             key=lambda x: get_worst_asset_cagr_mdd(x[1]),
                             reverse=True)[:5]

    for i, (desc, result) in enumerate(balanced_configs, 1):
        p = result["portfolio"]
        ind = result["individual"]
        worst = get_worst_asset_cagr_mdd(result)

        print(f"\n{i}. {desc} (worst CAGR/MDD = {worst:.2f})")
        print(f"   Portfolio: CAGR={p['cagr']*100:.1f}%, MDD={p['max_drawdown']*100:.1f}%, CAGR/MDD={p['cagr_mdd']:.2f}")
        print(f"   {'Symbol':<10} {'CAGR':<10} {'MDD':<10} {'CAGR/MDD':<10}")
        for symbol in PORTFOLIO_SYMBOLS:
            m = ind[symbol]
            cm = m["cagr"] / m["max_drawdown"] if m["max_drawdown"] > 0 else 0
            print(f"   {symbol:<10} {m['cagr']*100:<9.1f}% {m['max_drawdown']*100:<9.1f}% {cm:<10.2f}")


if __name__ == "__main__":
    main()
