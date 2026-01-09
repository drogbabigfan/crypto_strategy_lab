#!/usr/bin/env python3
"""
Dual-Eye Kalman Stop + Rolling MAE 백테스트

Trail Stop: k × √S_risk (Dual-Eye Kalman) - 동적 추세 추종
Hard Stop: Rolling MAE 99th percentile - 통계적 최대 역행폭
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


def calculate_rolling_mae(
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    horizon: int = 6,      # 진입 후 H봉 동안의 MAE
    window: int = 180,     # 과거 W봉의 MAE 참고
    quantile: float = 0.99  # 99th percentile
) -> np.ndarray:
    """
    Rolling MAE (Maximum Adverse Excursion) 계산

    "지금 진입하면 향후 H봉 동안 재수 없으면 어디까지 빠질까?"

    Long 기준:
    MAE_t = (Open_t - min(Low_{t+1} ... Low_{t+H})) / Open_t

    Returns:
        rolling_mae_pct: 각 시점의 99th percentile MAE (%)
    """
    n = len(open_prices)
    mae = np.full(n, np.nan)

    # Step 1: 각 봉의 MAE 계산 (Long 기준)
    for t in range(n - horizon):
        entry_price = open_prices[t]
        if entry_price <= 0 or np.isnan(entry_price):
            continue

        # 진입 후 H봉 동안의 최저가
        future_lows = low_prices[t+1:t+1+horizon]
        if len(future_lows) == 0 or np.any(np.isnan(future_lows)):
            continue

        min_low = np.min(future_lows)
        mae[t] = (entry_price - min_low) / entry_price

    # Step 2: Rolling 99th percentile
    rolling_mae_pct = np.full(n, np.nan)

    for t in range(window, n):
        past_mae = mae[t-window:t]
        valid_mae = past_mae[~np.isnan(past_mae)]
        if len(valid_mae) >= 10:
            rolling_mae_pct[t] = np.percentile(valid_mae, quantile * 100)

    # 초기 구간은 전체 평균으로 채움
    valid_all = mae[~np.isnan(mae)]
    if len(valid_all) > 0:
        default_mae = np.percentile(valid_all, quantile * 100)
        rolling_mae_pct[:window] = default_mae

    return rolling_mae_pct


@njit
def calculate_position_size(
    sigma_hybrid: float, unc_pct: float,
    risk_target: float, stop_distance: float,
    gauss_max_mult: float, gauss_sigma: float,
    size_min: float, size_max: float
):
    """Gaussian Position Sizing with dynamic stop distance"""
    base_size = risk_target / (stop_distance + 1e-10)
    gauss_scale = gauss_max_mult * np.exp(-unc_pct**2 / (2 * gauss_sigma**2))
    final_size = base_size * gauss_scale
    return max(size_min, min(size_max, final_size))


@njit
def generate_signals_dual_eye_mae(
    close, high, low,
    kf_trend, kf_trend_pred, sqrt_s_risk, rolling_mae,
    vel_zscore, unc_pct, sigma_hybrid, v_ratio, resid_std, intensity,
    entry_z, unc_pct_max, v_ratio_threshold, intensity_threshold,
    innov_base_mult, innov_mult_min, innov_mult_max,
    trail_k,  # k multiplier for trail stop
    risk_target, gauss_max_mult, gauss_sigma, size_min, size_max, warmup
):
    """
    Signal generation with:
    - Trail Stop: k × √S_risk (Dual-Eye Kalman)
    - Hard Stop: Rolling MAE 99th percentile

    Hard Stop = entry_price × (1 - rolling_mae)
    Trail Stop = highest × exp(-trail_k × √S_risk)
    """
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
    entry_mae = 0.0

    for i in range(warmup, n):
        innov_mult = innov_base_mult / (v_ratio[i] + 1e-10)
        innov_mult = max(innov_mult_min, min(innov_mult_max, innov_mult))
        log_resid = np.log(close[i]) - np.log(kf_trend[i])

        # Trail stop distance based on S_risk
        stop_dist = sqrt_s_risk[i]

        if position == 0:
            # === LONG ENTRY ===
            if vel_zscore[i] > entry_z and unc_pct[i] < unc_pct_max:
                position = 1
                entry_price = close[i]
                highest = high[i]
                entry_mae = rolling_mae[i]

                # Position size based on MAE stop distance
                current_size = calculate_position_size(
                    sigma_hybrid[i], unc_pct[i], risk_target,
                    entry_mae,  # Hard stop distance from MAE
                    gauss_max_mult, gauss_sigma, size_min, size_max
                )

                # Trail Stop: Kalman S_risk
                prev_stop = highest * np.exp(-trail_k * stop_dist)
                # Hard Stop: Rolling MAE
                hard_stop = entry_price * (1.0 - entry_mae)

                signals[i] = 1
                position_sizes[i] = current_size

            # === SHORT ENTRY ===
            elif vel_zscore[i] < -entry_z and unc_pct[i] < unc_pct_max:
                position = -1
                entry_price = close[i]
                lowest = low[i]
                entry_mae = rolling_mae[i]

                current_size = calculate_position_size(
                    sigma_hybrid[i], unc_pct[i], risk_target,
                    entry_mae,
                    gauss_max_mult, gauss_sigma, size_min, size_max
                )

                prev_stop = lowest * np.exp(trail_k * stop_dist)
                hard_stop = entry_price * (1.0 + entry_mae)

                signals[i] = -1
                position_sizes[i] = current_size

        elif position == 1:  # LONG
            if high[i] > highest:
                highest = high[i]

            # Update trailing stop (Kalman based)
            new_stop = highest * np.exp(-trail_k * stop_dist)
            trail_stop = max(new_stop, prev_stop)
            prev_stop = trail_stop

            # Exit checks
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

        elif position == -1:  # SHORT
            if low[i] < lowest:
                lowest = low[i]

            new_stop = lowest * np.exp(trail_k * stop_dist)
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


def run_backtest(df: pd.DataFrame, trail_k: float, mae_horizon: int = 6, mae_window: int = 180, mae_quantile: float = 0.99):
    """Run backtest with Dual-Eye Trail + MAE Hard Stop"""
    params = V34_PARAMS.copy()

    df_kf = calculate_adaptive_kalman(df)
    features = calculate_features(df_kf, params)
    intensity = calculate_intensity(df, params["intensity_baseline_window"])

    # Get S_risk from Kalman filter
    sqrt_s_risk = np.sqrt(df_kf["kf_innovation_cov_risk"].values)

    # Calculate Rolling MAE for Hard Stop
    rolling_mae = calculate_rolling_mae(
        open_prices=df_kf["open"].values,
        high_prices=df_kf["high"].values,
        low_prices=df_kf["low"].values,
        horizon=mae_horizon,
        window=mae_window,
        quantile=mae_quantile
    )

    signals, position_sizes = generate_signals_dual_eye_mae(
        close=df_kf["close"].values,
        high=df_kf["high"].values,
        low=df_kf["low"].values,
        kf_trend=features["kf_trend"],
        kf_trend_pred=df_kf["kf_trend_pred"].values,
        sqrt_s_risk=sqrt_s_risk,
        rolling_mae=rolling_mae,
        vel_zscore=features["vel_zscore"],
        unc_pct=features["unc_pct"],
        sigma_hybrid=features["sigma_hybrid"],
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
        trail_k=trail_k,
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
    """4-asset portfolio metrics"""
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

    # Test configurations - Different trail_k with Q95
    configs = [
        {"trail_k": 3, "mae_q": 0.95, "desc": "k=3 MAE Q95"},
        {"trail_k": 4, "mae_q": 0.95, "desc": "k=4 MAE Q95"},
        {"trail_k": 5, "mae_q": 0.95, "desc": "k=5 MAE Q95"},
        {"trail_k": 6, "mae_q": 0.95, "desc": "k=6 MAE Q95"},
    ]

    all_results = {}

    print("\n" + "="*100)
    print("Dual-Eye Trail Stop + Rolling MAE Hard Stop 백테스트")
    print("Trail Stop: k × √S_risk (Kalman Dual-Eye)")
    print("Hard Stop: Rolling MAE (H=6, W=180, Q varies)")
    print("="*100)

    for config in configs:
        trail_k = config["trail_k"]
        mae_q = config["mae_q"]
        desc = config["desc"]

        print(f"\nTesting {desc}...")

        individual_results = {}
        for symbol, df in data.items():
            metrics = run_backtest(df, trail_k, mae_horizon=6, mae_window=180, mae_quantile=mae_q)
            individual_results[symbol] = metrics

        portfolio = calculate_portfolio_metrics(individual_results)

        all_results[desc] = {
            "individual": individual_results,
            "portfolio": portfolio,
            "config": config,
        }

    # Summary
    print("\n" + "="*100)
    print("포트폴리오 성과 비교")
    print("="*100)
    print(f"{'Config':<25} {'CAGR':<12} {'MDD':<12} {'CAGR/MDD':<10} {'Sharpe':<10} {'Leverage':<10}")
    print("-"*80)

    for desc, result in all_results.items():
        p = result["portfolio"]
        print(f"{desc:<25} {p['cagr']*100:<11.1f}% {p['max_drawdown']*100:<11.1f}% "
              f"{p['cagr_mdd']:<10.2f} {p['sharpe']:<10.2f} {p['avg_leverage']:<10.2f}x")

    # 비교
    print("\n" + "-"*80)
    print("비교:")
    print("  현재 V3.4 (7×σ_hybrid):        CAGR=60.2%, MDD=36.2%, CAGR/MDD=1.66, Leverage=2.41x")
    print("  이전 k=5 Dual-Eye (Hard+Trail): CAGR=73.3%, MDD=31.8%, CAGR/MDD=2.30")

    # 개별 자산 상세
    print("\n" + "="*100)
    print("개별 자산 상세 (k=5 MAE Q95)")
    print("="*100)

    best_config = "k=5 MAE Q95"
    if best_config in all_results:
        ind = all_results[best_config]["individual"]
        print(f"{'Symbol':<10} {'CAGR':<10} {'MDD':<10} {'C/M':<8} {'Sharpe':<8} {'Trades':<8} {'WinRate':<10} {'Hold':<8} {'Size':<8}")
        print("-"*85)
        for symbol in PORTFOLIO_SYMBOLS:
            m = ind[symbol]
            cm = m["cagr"] / m["max_drawdown"] if m["max_drawdown"] > 0 else 0

            # Calculate Sharpe for individual asset
            bar_ret = m["bar_returns"]
            daily_ret = []
            for i in range(0, len(bar_ret), 6):
                chunk = bar_ret[i:i+6]
                if len(chunk) > 0:
                    daily_ret.append(np.sum(chunk))
            sharpe = (np.mean(daily_ret) / (np.std(daily_ret) + 1e-10)) * np.sqrt(252) if len(daily_ret) > 1 else 0

            avg_hold_days = m['avg_hold_bars'] * 4 / 24  # 4h bars to days
            print(f"{symbol:<10} {m['cagr']*100:<9.1f}% {m['max_drawdown']*100:<9.1f}% "
                  f"{cm:<8.2f} {sharpe:<8.2f} {m['total_trades']:<8} {m['win_rate']*100:<9.1f}% {avg_hold_days:<8.1f}d {m['avg_size']:<8.2f}")

        # Portfolio summary
        p = all_results[best_config]["portfolio"]
        print("-"*85)
        print(f"{'Portfolio':<10} {p['cagr']*100:<9.1f}% {p['max_drawdown']*100:<9.1f}% "
              f"{p['cagr_mdd']:<8.2f} {p['sharpe']:<8.2f} {'-':<8} {'-':<10} {p['avg_leverage']:<10.2f}")


if __name__ == "__main__":
    main()
