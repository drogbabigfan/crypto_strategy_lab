#!/usr/bin/env python3
"""
4-Asset Portfolio Comparison
Baseline (tanh) vs Gaussian M=2.0, σ=0.3
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
PORTFOLIO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]


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


# ============================================================
# Sizing Models
# ============================================================

@njit
def sizing_baseline(unc_pct: float) -> float:
    """Current tanh-based model: 1.0 + 0.5 * tanh(2.5 * (0.5 - unc_pct))"""
    return 1.0 + 0.5 * np.tanh(2.5 * (0.5 - unc_pct))


@njit
def sizing_gaussian(unc_pct: float, max_mult: float, sigma: float) -> float:
    """Gaussian Decay: S = M × exp(-u²/(2σ²))"""
    return max_mult * np.exp(-unc_pct**2 / (2 * sigma**2))


@njit
def calculate_position_size_baseline(
    sigma_hybrid: float, unc_pct: float,
    risk_target: float, hard_stop_mult: float,
    size_min: float, size_max: float
) -> float:
    hard_stop_dist = hard_stop_mult * sigma_hybrid
    base_size = risk_target / (hard_stop_dist + 1e-10)
    conf_mult = sizing_baseline(unc_pct)
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
def generate_signals_baseline(
    close, high, low, kf_trend, vel_zscore, unc_pct, sigma_hybrid, dynamic_mult, v_ratio, resid_std,
    hard_mult, risk_target, size_min, size_max, warmup
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

    entry_z = 2.0
    v_ratio_threshold = 1.0
    base_innov = 2.5
    innov_min = 1.5
    innov_max = 4.0

    for i in range(warmup, n):
        log_resid = np.log(close[i]) - np.log(kf_trend[i])
        innov_mult = base_innov / (v_ratio[i] + 1e-10)
        innov_mult = max(innov_min, min(innov_max, innov_mult))

        if position == 0:
            if vel_zscore[i] > entry_z and unc_pct[i] < 0.5:
                position = 1
                entry_price = close[i]
                highest = high[i]
                current_size = calculate_position_size_baseline(
                    sigma_hybrid[i], unc_pct[i], risk_target, hard_mult, size_min, size_max
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
                current_size = calculate_position_size_baseline(
                    sigma_hybrid[i], unc_pct[i], risk_target, hard_mult, size_min, size_max
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

            if close[i] < hard_stop or close[i] < trail_stop:
                position = 0
                continue
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

            if close[i] > hard_stop or close[i] > trail_stop:
                position = 0
                continue
            if v_ratio[i] >= v_ratio_threshold and log_resid > innov_mult * resid_std[i]:
                position = 0
                continue
            if v_ratio[i] < v_ratio_threshold and vel_zscore[i] > entry_z:
                position = 0
                continue
            signals[i] = -1
            position_sizes[i] = current_size

    return signals, position_sizes


@njit
def generate_signals_gaussian(
    close, high, low, kf_trend, vel_zscore, unc_pct, sigma_hybrid, dynamic_mult, v_ratio, resid_std,
    hard_mult, risk_target, max_mult, gauss_sigma, size_min, size_max, warmup
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

    entry_z = 2.0
    v_ratio_threshold = 1.0
    base_innov = 2.5
    innov_min = 1.5
    innov_max = 4.0

    for i in range(warmup, n):
        log_resid = np.log(close[i]) - np.log(kf_trend[i])
        innov_mult = base_innov / (v_ratio[i] + 1e-10)
        innov_mult = max(innov_min, min(innov_max, innov_mult))

        if position == 0:
            if vel_zscore[i] > entry_z and unc_pct[i] < 0.5:
                position = 1
                entry_price = close[i]
                highest = high[i]
                current_size = calculate_position_size_gaussian(
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
                current_size = calculate_position_size_gaussian(
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

            if close[i] < hard_stop or close[i] < trail_stop:
                position = 0
                continue
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

            if close[i] > hard_stop or close[i] > trail_stop:
                position = 0
                continue
            if v_ratio[i] >= v_ratio_threshold and log_resid > innov_mult * resid_std[i]:
                position = 0
                continue
            if v_ratio[i] < v_ratio_threshold and vel_zscore[i] > entry_z:
                position = 0
                continue
            signals[i] = -1
            position_sizes[i] = current_size

    return signals, position_sizes


def run_backtest_with_equity(df: pd.DataFrame, model: str = "baseline",
                              max_mult: float = 2.0, gauss_sigma: float = 0.3):
    """Run backtest and return equity curve"""
    df_kf = calculate_adaptive_kalman(df)
    features = calculate_features(df_kf)

    risk_target = 0.02
    hard_mult = 5.0
    size_min = 0.1
    size_max = 3.0
    warmup = 210

    if model == "baseline":
        signals, position_sizes = generate_signals_baseline(
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
            hard_mult=hard_mult,
            risk_target=risk_target,
            size_min=size_min,
            size_max=size_max,
            warmup=warmup,
        )
    else:  # gaussian
        signals, position_sizes = generate_signals_gaussian(
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
            hard_mult=hard_mult,
            risk_target=risk_target,
            max_mult=max_mult,
            gauss_sigma=gauss_sigma,
            size_min=size_min,
            size_max=size_max,
            warmup=warmup,
        )

    df_kf["signal"] = signals
    df_kf["position_size"] = position_sizes
    df_kf["sl_price"] = 0.0

    # Calculate returns manually for equity curve
    close = df_kf["close"].values
    returns = np.zeros(len(close))
    returns[1:] = (close[1:] - close[:-1]) / close[:-1]

    # Strategy returns
    strategy_returns = np.zeros(len(close))
    for i in range(1, len(close)):
        if signals[i-1] != 0:
            strategy_returns[i] = signals[i-1] * position_sizes[i-1] * returns[i]

    # Apply fees on position changes
    fee_rate = 0.001
    slippage_rate = 0.0001
    for i in range(1, len(signals)):
        if signals[i] != signals[i-1] or (signals[i] != 0 and position_sizes[i] != position_sizes[i-1]):
            strategy_returns[i] -= (fee_rate + slippage_rate) * abs(position_sizes[i] if signals[i] != 0 else position_sizes[i-1])

    # Equity curve
    equity = np.cumprod(1 + strategy_returns)

    # Create result dataframe
    result_df = pd.DataFrame({
        "timestamp": df_kf["timestamp"].values if "timestamp" in df_kf.columns else range(len(df_kf)),
        "equity": equity,
        "returns": strategy_returns,
    })

    return result_df


def calculate_portfolio_metrics(equity_curve: np.ndarray, bars_per_year: float = 365 * 4):
    """Calculate portfolio metrics from equity curve"""
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[~np.isnan(returns)]

    # Total PnL
    total_pnl = equity_curve[-1] / equity_curve[0] - 1

    # CAGR
    n_bars = len(equity_curve)
    years = n_bars / bars_per_year
    if years > 0 and total_pnl > -1:
        cagr = ((1 + total_pnl) ** (1 / years)) - 1
    else:
        cagr = 0

    # MDD
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    mdd = abs(np.min(drawdown))

    # Sharpe
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(bars_per_year)
    else:
        sharpe = 0

    return {
        "total_pnl": total_pnl,
        "cagr": cagr,
        "max_drawdown": mdd,
        "sharpe_ratio": sharpe,
        "cagr_mdd": cagr / mdd if mdd > 0 else 0,
    }


def main():
    print("Loading data...")
    data = {}
    for symbol in PORTFOLIO_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars")

    print("\n" + "="*120)
    print("INDIVIDUAL ASSET COMPARISON")
    print("="*120)

    for model_name, model_type in [("Baseline (tanh)", "baseline"), ("Gaussian M=2.0 σ=0.3", "gaussian")]:
        print(f"\n### {model_name} ###")
        print(f"{'Symbol':<12} {'CAGR':<12} {'MDD':<12} {'CAGR/MDD':<12} {'Sharpe':<12} {'PnL':<12}")
        print("-"*80)

        for symbol, df in data.items():
            result_df = run_backtest_with_equity(df, model_type)
            metrics = calculate_portfolio_metrics(result_df["equity"].values)
            print(f"{symbol:<12} {metrics['cagr']*100:<11.1f}% {metrics['max_drawdown']*100:<11.1f}% "
                  f"{metrics['cagr_mdd']:<12.2f} {metrics['sharpe_ratio']:<12.2f} {metrics['total_pnl']*100:<11.1f}%")

    print("\n" + "="*120)
    print("4-ASSET PORTFOLIO COMPARISON (Equal Weight)")
    print("="*120)

    # Find common date range
    min_len = min(len(df) for df in data.values())
    print(f"\nUsing {min_len:,} bars (aligned to shortest series)")

    portfolio_results = {}

    for model_name, model_type in [("Baseline (tanh)", "baseline"), ("Gaussian M=2.0 σ=0.3", "gaussian")]:
        print(f"\n### {model_name} ###")

        # Get equity curves for each asset
        equity_curves = []
        for symbol, df in data.items():
            result_df = run_backtest_with_equity(df, model_type)
            # Align to min length (from the end)
            equity = result_df["equity"].values[-min_len:]
            # Normalize to start at 1
            equity = equity / equity[0]
            equity_curves.append(equity)

        # Equal weight portfolio (average of normalized equity curves)
        portfolio_equity = np.mean(equity_curves, axis=0)

        # Calculate portfolio metrics
        metrics = calculate_portfolio_metrics(portfolio_equity)
        portfolio_results[model_name] = {
            "metrics": metrics,
            "equity": portfolio_equity,
        }

        print(f"Portfolio Total PnL: {metrics['total_pnl']*100:.1f}%")
        print(f"Portfolio CAGR: {metrics['cagr']*100:.1f}%")
        print(f"Portfolio MDD: {metrics['max_drawdown']*100:.1f}%")
        print(f"Portfolio CAGR/MDD: {metrics['cagr_mdd']:.2f}")
        print(f"Portfolio Sharpe: {metrics['sharpe_ratio']:.2f}")

    # Summary comparison
    print("\n" + "="*120)
    print("SUMMARY COMPARISON")
    print("="*120)
    print(f"{'Metric':<20} {'Baseline':<20} {'Gaussian M=2 σ=0.3':<20} {'Diff':<15} {'Diff %':<15}")
    print("-"*90)

    baseline = portfolio_results["Baseline (tanh)"]["metrics"]
    gaussian = portfolio_results["Gaussian M=2.0 σ=0.3"]["metrics"]

    metrics_to_compare = [
        ("CAGR", "cagr", 100),
        ("MDD", "max_drawdown", 100),
        ("CAGR/MDD", "cagr_mdd", 1),
        ("Sharpe", "sharpe_ratio", 1),
        ("Total PnL", "total_pnl", 100),
    ]

    for name, key, mult in metrics_to_compare:
        b_val = baseline[key] * mult
        g_val = gaussian[key] * mult
        diff = g_val - b_val
        diff_pct = (g_val / b_val - 1) * 100 if b_val != 0 else 0

        suffix = "%" if mult == 100 else ""
        sign = "+" if diff > 0 else ""

        print(f"{name:<20} {b_val:<19.2f}{suffix} {g_val:<19.2f}{suffix} {sign}{diff:<14.2f} {sign}{diff_pct:<14.1f}%")

    # Yearly breakdown
    print("\n" + "="*120)
    print("YEARLY COMPARISON")
    print("="*120)

    bars_per_year = 365 * 4  # 6 bars/day * 365 days

    for model_name in ["Baseline (tanh)", "Gaussian M=2.0 σ=0.3"]:
        print(f"\n### {model_name} ###")
        equity = portfolio_results[model_name]["equity"]

        years = len(equity) // bars_per_year
        print(f"{'Year':<10} {'Return':<15} {'MDD':<15}")
        print("-"*40)

        for y in range(years):
            start_idx = y * bars_per_year
            end_idx = (y + 1) * bars_per_year
            year_equity = equity[start_idx:end_idx]

            year_return = (year_equity[-1] / year_equity[0] - 1) * 100
            peak = np.maximum.accumulate(year_equity)
            drawdown = (year_equity - peak) / peak
            year_mdd = abs(np.min(drawdown)) * 100

            print(f"Year {y+1:<5} {year_return:<14.1f}% {year_mdd:<14.1f}%")


if __name__ == "__main__":
    main()
