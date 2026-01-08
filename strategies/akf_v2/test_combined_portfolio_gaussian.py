#!/usr/bin/env python3
"""
4-Asset Combined Portfolio - Gaussian Sizing vs Baseline 비교

레버리지 누적 방식: 각 자산의 수익률을 합산
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
from strategies.akf_v2.strategy_v33_final import load_data, V33_PARAMS, calculate_features
from strategies.akf_v2.test_duration_filter import calculate_intensity

PORTFOLIO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"]


# ============================================================
# Sizing Models
# ============================================================

@njit
def sizing_baseline(unc_pct: float, conf_lambda: float = 2.5) -> float:
    """Current tanh-based model"""
    return 1.0 + 0.5 * np.tanh(conf_lambda * (0.5 - unc_pct))


@njit
def sizing_gaussian(unc_pct: float, max_mult: float = 2.0, sigma: float = 0.3) -> float:
    """Gaussian Decay: S = M × exp(-u²/(2σ²))"""
    return max_mult * np.exp(-unc_pct**2 / (2 * sigma**2))


@njit
def calculate_position_size_baseline(
    sigma_hybrid: float, unc_pct: float,
    risk_target: float, hard_stop_mult: float,
    conf_lambda: float, size_min: float, size_max: float
) -> float:
    hard_stop_dist = hard_stop_mult * sigma_hybrid
    base_size = risk_target / (hard_stop_dist + 1e-10)
    conf_mult = sizing_baseline(unc_pct, conf_lambda)
    final_size = base_size * conf_mult
    return max(size_min, min(size_max, final_size))


@njit
def calculate_position_size_gaussian(
    sigma_hybrid: float, unc_pct: float,
    risk_target: float, hard_stop_mult: float,
    max_mult: float, gauss_sigma: float,
    size_min: float, size_max: float
) -> float:
    hard_stop_dist = hard_stop_mult * sigma_hybrid
    base_size = risk_target / (hard_stop_dist + 1e-10)
    conf_mult = sizing_gaussian(unc_pct, max_mult, gauss_sigma)
    final_size = base_size * conf_mult
    return max(size_min, min(size_max, final_size))


@njit
def generate_signals_with_intensity(
    close, high, low, kf_trend, vel_zscore, unc_pct, sigma_hybrid, dynamic_mult, v_ratio, resid_std,
    intensity, entry_z, unc_pct_max, v_ratio_threshold, intensity_threshold,
    innov_base_mult, innov_mult_min, innov_mult_max,
    hard_stop_mult, risk_target, size_min, size_max, warmup,
    use_gaussian: bool, gauss_max_mult: float, gauss_sigma: float, conf_lambda: float
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
        innov_mult = innov_base_mult / (v_ratio[i] + 1e-10)
        innov_mult = max(innov_mult_min, min(innov_mult_max, innov_mult))

        current_intensity = intensity[i] if not np.isnan(intensity[i]) else 1.0

        if position == 0:
            if vel_zscore[i] > entry_z and unc_pct[i] < unc_pct_max:
                position = 1
                entry_price = close[i]
                highest = high[i]

                if use_gaussian:
                    current_size = calculate_position_size_gaussian(
                        sigma_hybrid[i], unc_pct[i], risk_target, hard_stop_mult,
                        gauss_max_mult, gauss_sigma, size_min, size_max
                    )
                else:
                    current_size = calculate_position_size_baseline(
                        sigma_hybrid[i], unc_pct[i], risk_target, hard_stop_mult,
                        conf_lambda, size_min, size_max
                    )

                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = highest * np.exp(-dist)
                hard_stop = entry_price * np.exp(-hard_stop_mult * sigma_hybrid[i])
                signals[i] = 1
                position_sizes[i] = current_size

            elif vel_zscore[i] < -entry_z and unc_pct[i] < unc_pct_max:
                position = -1
                entry_price = close[i]
                lowest = low[i]

                if use_gaussian:
                    current_size = calculate_position_size_gaussian(
                        sigma_hybrid[i], unc_pct[i], risk_target, hard_stop_mult,
                        gauss_max_mult, gauss_sigma, size_min, size_max
                    )
                else:
                    current_size = calculate_position_size_baseline(
                        sigma_hybrid[i], unc_pct[i], risk_target, hard_stop_mult,
                        conf_lambda, size_min, size_max
                    )

                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = lowest * np.exp(dist)
                hard_stop = entry_price * np.exp(hard_stop_mult * sigma_hybrid[i])
                signals[i] = -1
                position_sizes[i] = current_size

        elif position == 1:
            if high[i] > highest:
                highest = high[i]
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_stop = highest * np.exp(-dist)
            trail_stop = max(new_stop, prev_stop)
            prev_stop = trail_stop

            # Hard stop & Trail stop (무조건 실행)
            if close[i] < hard_stop or close[i] < trail_stop:
                position = 0
                continue

            # Signal exits (intensity filter 적용)
            if current_intensity < intensity_threshold:
                if v_ratio[i] >= v_ratio_threshold and log_resid < -innov_mult * resid_std[i]:
                    position = 0
                    continue
                if v_ratio[i] < v_ratio_threshold and vel_zscore[i] < -entry_z:
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

            # Hard stop & Trail stop (무조건 실행)
            if close[i] > hard_stop or close[i] > trail_stop:
                position = 0
                continue

            # Signal exits (intensity filter 적용)
            if current_intensity < intensity_threshold:
                if v_ratio[i] >= v_ratio_threshold and log_resid > innov_mult * resid_std[i]:
                    position = 0
                    continue
                if v_ratio[i] < v_ratio_threshold and vel_zscore[i] > entry_z:
                    position = 0
                    continue

            signals[i] = -1
            position_sizes[i] = current_size

    return signals, position_sizes


def run_backtest(df: pd.DataFrame, use_gaussian: bool = False,
                 gauss_max_mult: float = 2.0, gauss_sigma: float = 0.3,
                 intensity_threshold: float = 4.0):
    """Run backtest with optional Gaussian sizing"""
    params = V33_PARAMS
    df_kf = calculate_adaptive_kalman(df)
    features = calculate_features(df_kf, params)
    intensity, _, _ = calculate_intensity(df, baseline_window=20)

    signals, position_sizes = generate_signals_with_intensity(
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
        intensity_threshold=intensity_threshold,
        innov_base_mult=params["innov_base_mult"],
        innov_mult_min=params["innov_mult_min"],
        innov_mult_max=params["innov_mult_max"],
        hard_stop_mult=params["hard_stop_mult"],
        risk_target=params["risk_target"],
        size_min=params["size_min"],
        size_max=params["size_max"],
        warmup=params["warmup"],
        use_gaussian=use_gaussian,
        gauss_max_mult=gauss_max_mult,
        gauss_sigma=gauss_sigma,
        conf_lambda=params["conf_lambda"],
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

    # Add avg_size
    active_sizes = position_sizes[position_sizes > 0]
    metrics["avg_size"] = np.mean(active_sizes) if len(active_sizes) > 0 else 0

    return metrics


def run_portfolio_backtest(use_gaussian: bool = False, gauss_max_mult: float = 2.0, gauss_sigma: float = 0.3):
    """Run combined portfolio backtest"""
    initial_capital = 100000.0

    individual_results = {}

    for symbol in PORTFOLIO_SYMBOLS:
        df = load_data(symbol)
        if df is None:
            continue

        m = run_backtest(df, use_gaussian, gauss_max_mult, gauss_sigma)
        equity_curve = m["equity_curve"]

        bar_returns = np.zeros(len(equity_curve))
        bar_returns[1:] = np.diff(equity_curve) / equity_curve[:-1]

        individual_results[symbol] = {
            "metrics": m,
            "equity_curve": equity_curve,
            "bar_returns": bar_returns,
        }

    # 레버리지 누적 합산
    min_len = min(len(r["bar_returns"]) for r in individual_results.values())

    combined_returns = np.zeros(min_len)
    for symbol, r in individual_results.items():
        combined_returns += r["bar_returns"][:min_len]

    combined_equity = np.zeros(min_len)
    combined_equity[0] = initial_capital

    for i in range(1, min_len):
        combined_equity[i] = combined_equity[i-1] * (1 + combined_returns[i])

    # Metrics 계산
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
        mean_daily = np.mean(daily_returns)
        std_daily = np.std(daily_returns)
        sharpe = (mean_daily / (std_daily + 1e-10)) * np.sqrt(252)
    else:
        sharpe = 0.0

    n_years = min_len / (6 * 252)
    if n_years > 0 and combined_equity[-1] > 0:
        cagr = (combined_equity[-1] / initial_capital) ** (1 / n_years) - 1
    else:
        cagr = 0.0

    # Yearly returns
    bars_per_year = 6 * 252
    yearly_returns = []
    yearly_mdds = []

    for y in range(int(n_years) + 1):
        start_idx = y * bars_per_year
        end_idx = min((y + 1) * bars_per_year, len(combined_equity))
        if start_idx >= len(combined_equity):
            break

        year_equity = combined_equity[start_idx:end_idx]
        if len(year_equity) > 1:
            year_return = (year_equity[-1] / year_equity[0] - 1)
            yearly_returns.append(year_return)

            peak = np.maximum.accumulate(year_equity)
            dd = (year_equity - peak) / peak
            year_mdd = abs(np.min(dd))
            yearly_mdds.append(year_mdd)

    total_avg_size = sum(r["metrics"]["avg_size"] for r in individual_results.values())

    return {
        "individual": individual_results,
        "combined_equity": combined_equity,
        "total_pnl": total_pnl,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "avg_leverage": total_avg_size,
        "yearly_returns": yearly_returns,
        "yearly_mdds": yearly_mdds,
    }


def main():
    print("=" * 120)
    print("4-Asset Combined Portfolio: Baseline vs Gaussian (M=2.0, σ=0.3)")
    print("=" * 120)

    # Baseline
    print("\n[1] Running Baseline (tanh sizing)...")
    baseline = run_portfolio_backtest(use_gaussian=False)

    # Gaussian
    print("\n[2] Running Gaussian (M=2.0, σ=0.3)...")
    gaussian = run_portfolio_backtest(use_gaussian=True, gauss_max_mult=2.0, gauss_sigma=0.3)

    # Individual comparison
    print("\n" + "=" * 120)
    print("개별 자산 비교")
    print("=" * 120)
    print(f"{'Symbol':<12} {'Baseline PnL':<15} {'Gaussian PnL':<15} {'Δ PnL':<12} {'Base MDD':<12} {'Gauss MDD':<12}")
    print("-" * 90)

    for symbol in PORTFOLIO_SYMBOLS:
        b = baseline["individual"][symbol]["metrics"]
        g = gaussian["individual"][symbol]["metrics"]
        delta = (g["total_pnl"] - b["total_pnl"]) * 100
        print(f"{symbol:<12} {b['total_pnl']*100:<14.1f}% {g['total_pnl']*100:<14.1f}% "
              f"{'+' if delta > 0 else ''}{delta:<11.1f}% {b['max_drawdown']*100:<11.1f}% {g['max_drawdown']*100:<11.1f}%")

    # Portfolio comparison
    print("\n" + "=" * 120)
    print("포트폴리오 합산 비교 (레버리지 누적)")
    print("=" * 120)
    print(f"{'Metric':<20} {'Baseline':<20} {'Gaussian':<20} {'Δ':<15} {'Δ %':<15}")
    print("-" * 90)

    comparisons = [
        ("Total PnL", baseline["total_pnl"]*100, gaussian["total_pnl"]*100, "%"),
        ("CAGR", baseline["cagr"]*100, gaussian["cagr"]*100, "%"),
        ("Max Drawdown", baseline["max_drawdown"]*100, gaussian["max_drawdown"]*100, "%"),
        ("CAGR/MDD", baseline["cagr"]/baseline["max_drawdown"], gaussian["cagr"]/gaussian["max_drawdown"], ""),
        ("Sharpe", baseline["sharpe"], gaussian["sharpe"], ""),
        ("Avg Leverage", baseline["avg_leverage"], gaussian["avg_leverage"], "x"),
    ]

    for name, b_val, g_val, suffix in comparisons:
        diff = g_val - b_val
        diff_pct = (g_val / b_val - 1) * 100 if b_val != 0 else 0
        sign = "+" if diff > 0 else ""
        print(f"{name:<20} {b_val:<19.2f}{suffix} {g_val:<19.2f}{suffix} {sign}{diff:<14.2f} {sign}{diff_pct:<14.1f}%")

    # Yearly comparison
    print("\n" + "=" * 120)
    print("연도별 비교")
    print("=" * 120)
    print(f"{'Year':<8} {'Base Ret':<12} {'Gauss Ret':<12} {'Δ Ret':<10} {'Base MDD':<12} {'Gauss MDD':<12}")
    print("-" * 70)

    for i in range(min(len(baseline["yearly_returns"]), len(gaussian["yearly_returns"]))):
        b_ret = baseline["yearly_returns"][i] * 100
        g_ret = gaussian["yearly_returns"][i] * 100
        b_mdd = baseline["yearly_mdds"][i] * 100
        g_mdd = gaussian["yearly_mdds"][i] * 100
        delta = g_ret - b_ret
        sign = "+" if delta > 0 else ""
        print(f"Year {i+1:<3} {b_ret:<11.1f}% {g_ret:<11.1f}% {sign}{delta:<9.1f}% {b_mdd:<11.1f}% {g_mdd:<11.1f}%")

    # Final summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)

    b_cagr_mdd = baseline["cagr"] / baseline["max_drawdown"]
    g_cagr_mdd = gaussian["cagr"] / gaussian["max_drawdown"]

    print(f"\nBaseline:")
    print(f"  Total PnL: {baseline['total_pnl']*100:.1f}%, CAGR: {baseline['cagr']*100:.1f}%, MDD: {baseline['max_drawdown']*100:.1f}%")
    print(f"  CAGR/MDD: {b_cagr_mdd:.2f}, Sharpe: {baseline['sharpe']:.2f}, Avg Leverage: {baseline['avg_leverage']:.2f}x")

    print(f"\nGaussian (M=2.0, σ=0.3):")
    print(f"  Total PnL: {gaussian['total_pnl']*100:.1f}%, CAGR: {gaussian['cagr']*100:.1f}%, MDD: {gaussian['max_drawdown']*100:.1f}%")
    print(f"  CAGR/MDD: {g_cagr_mdd:.2f}, Sharpe: {gaussian['sharpe']:.2f}, Avg Leverage: {gaussian['avg_leverage']:.2f}x")

    print(f"\n개선:")
    print(f"  PnL: +{(gaussian['total_pnl'] - baseline['total_pnl'])*100:.1f}%p ({(gaussian['total_pnl']/baseline['total_pnl']-1)*100:.1f}% 증가)")
    print(f"  CAGR: +{(gaussian['cagr'] - baseline['cagr'])*100:.1f}%p")
    print(f"  CAGR/MDD: +{g_cagr_mdd - b_cagr_mdd:.2f}")


if __name__ == "__main__":
    main()
