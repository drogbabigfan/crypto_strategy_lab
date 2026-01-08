#!/usr/bin/env python3
"""
Dynamic Position Sizing Models based on Uncertainty

1. Gaussian Decay (The Sniper): S = M × exp(-u²/(2σ²))
2. Scaled Cosine Decay (The Smooth Operator): S = M/2 × [1 + cos(π×u)]
3. Polynomial Decay (The Power Curve): S = M × (1-u)^k

M = 2.0 (max multiplier)
u = unc_pct (uncertainty percentage, 0~1)
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


# ============================================================
# Sizing Models
# ============================================================

@njit
def sizing_baseline(unc_pct: float, max_mult: float, conf_lambda: float = 2.5, scale: float = 0.5) -> float:
    """Current tanh-based model"""
    return 1.0 + scale * np.tanh(conf_lambda * (0.5 - unc_pct))


@njit
def sizing_gaussian(unc_pct: float, max_mult: float, sigma: float = 0.4) -> float:
    """Gaussian Decay (The Sniper): S = M × exp(-u²/(2σ²))"""
    return max_mult * np.exp(-unc_pct**2 / (2 * sigma**2))


@njit
def sizing_cosine(unc_pct: float, max_mult: float) -> float:
    """Scaled Cosine Decay (The Smooth Operator): S = M/2 × [1 + cos(π×u)]"""
    u = max(0.0, min(1.0, unc_pct))  # clamp to [0, 1]
    return (max_mult / 2.0) * (1.0 + np.cos(np.pi * u))


@njit
def sizing_polynomial(unc_pct: float, max_mult: float, k: float = 1.0) -> float:
    """Polynomial Decay (The Power Curve): S = M × (1-u)^k"""
    u = max(0.0, min(1.0, unc_pct))  # clamp to [0, 1]
    return max_mult * ((1.0 - u) ** k)


@njit
def calculate_position_size(
    sigma_hybrid: float, unc_pct: float,
    risk_target: float, hard_stop_mult: float,
    model_type: int, max_mult: float, param: float,
    size_min: float, size_max: float
) -> float:
    """
    model_type:
        0 = baseline (tanh)
        1 = gaussian
        2 = cosine
        3 = polynomial
    param: model-specific parameter (sigma for gaussian, k for polynomial)
    """
    hard_stop_dist = hard_stop_mult * sigma_hybrid
    base_size = risk_target / (hard_stop_dist + 1e-10)

    if model_type == 0:  # baseline tanh
        conf_mult = sizing_baseline(unc_pct, max_mult, conf_lambda=2.5, scale=0.5)
    elif model_type == 1:  # gaussian
        conf_mult = sizing_gaussian(unc_pct, max_mult, sigma=param)
    elif model_type == 2:  # cosine
        conf_mult = sizing_cosine(unc_pct, max_mult)
    else:  # polynomial
        conf_mult = sizing_polynomial(unc_pct, max_mult, k=param)

    final_size = base_size * conf_mult
    final_size = max(size_min, min(size_max, final_size))
    return final_size


@njit
def generate_signals_with_sizing(
    close, high, low, kf_trend, vel_zscore, unc_pct, sigma_hybrid, dynamic_mult, v_ratio, resid_std,
    entry_z, v_ratio_threshold, base_innov, innov_min, innov_max,
    hard_mult, risk_target, model_type, max_mult, param, size_min, size_max, warmup
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
                    model_type, max_mult, param, size_min, size_max
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
                    model_type, max_mult, param, size_min, size_max
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


def run_backtest(df: pd.DataFrame, model_type: int = 0, max_mult: float = 2.0,
                 param: float = 1.0, risk_target: float = 0.02,
                 hard_mult: float = 5.0, size_min: float = 0.1, size_max: float = 3.0):
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
        model_type=model_type,
        max_mult=max_mult,
        param=param,
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


def print_model_curves():
    """Print sizing multiplier at different uncertainty levels"""
    print("\n" + "="*100)
    print("SIZING MODELS - Multiplier at different uncertainty levels (M=2.0)")
    print("="*100)

    u_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    max_mult = 2.0

    print(f"{'Model':<25} " + " ".join([f"u={u:.1f}" for u in u_values]))
    print("-"*100)

    # Baseline tanh
    vals = [sizing_baseline(u, max_mult, 2.5, 0.5) for u in u_values]
    print(f"{'Baseline (tanh)':<25} " + " ".join([f"{v:5.2f}" for v in vals]))

    # Gaussian variations
    for sigma in [0.3, 0.4, 0.5, 0.6]:
        vals = [sizing_gaussian(u, max_mult, sigma) for u in u_values]
        print(f"{f'Gaussian σ={sigma}':<25} " + " ".join([f"{v:5.2f}" for v in vals]))

    # Cosine
    vals = [sizing_cosine(u, max_mult) for u in u_values]
    print(f"{'Cosine':<25} " + " ".join([f"{v:5.2f}" for v in vals]))

    # Polynomial variations
    for k in [0.5, 1.0, 1.5, 2.0]:
        vals = [sizing_polynomial(u, max_mult, k) for u in u_values]
        print(f"{f'Polynomial k={k}':<25} " + " ".join([f"{v:5.2f}" for v in vals]))


def main():
    print_model_curves()

    print("\n\nLoading data...")
    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars")

    # Model configurations: (model_type, max_mult, param, description)
    # model_type: 0=baseline, 1=gaussian, 2=cosine, 3=polynomial
    configs = [
        # Baseline (current)
        (0, 2.0, 0.0, "Baseline (tanh λ=2.5, s=0.5)"),

        # Gaussian variations
        (1, 2.0, 0.3, "Gaussian σ=0.3 (Strict)"),
        (1, 2.0, 0.4, "Gaussian σ=0.4"),
        (1, 2.0, 0.5, "Gaussian σ=0.5"),
        (1, 2.0, 0.6, "Gaussian σ=0.6 (Tolerant)"),

        # Cosine
        (2, 2.0, 0.0, "Cosine (Smooth S-curve)"),

        # Polynomial variations
        (3, 2.0, 0.5, "Polynomial k=0.5 (Concave)"),
        (3, 2.0, 1.0, "Polynomial k=1.0 (Linear)"),
        (3, 2.0, 1.5, "Polynomial k=1.5"),
        (3, 2.0, 2.0, "Polynomial k=2.0 (Convex)"),
    ]

    all_results = {}

    for model_type, max_mult, param, desc in configs:
        print(f"\n{'='*120}")
        print(f"{desc}")
        print("="*120)
        print(f"{'Symbol':<10} {'Sharpe':<10} {'CAGR':<12} {'PnL':<12} {'MDD':<10} {'CAGR/MDD':<10} {'AvgSize':<10}")
        print("-"*90)

        results = []
        for symbol, df in data.items():
            metrics = run_backtest(df, model_type, max_mult, param)
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
            print(f"{symbol:<10} {metrics['sharpe_ratio']:<10.2f} {metrics['cagr']*100:<11.1f}% "
                  f"{metrics['total_pnl']*100:<11.1f}% {metrics['max_drawdown']*100:<9.1f}% "
                  f"{cagr_mdd:<10.2f} {metrics['avg_size']:<10.2f}")

        all_results[desc] = results

        print("-"*90)
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_cagr = np.mean([r["cagr"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        avg_cagr_mdd = np.mean([r["cagr_mdd"] for r in results])
        avg_size = np.mean([r["avg_size"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)

        print(f"Average: Sharpe={avg_sharpe:.2f}, CAGR={avg_cagr*100:.1f}%, MDD={avg_mdd*100:.1f}%, "
              f"CAGR/MDD={avg_cagr_mdd:.2f}, AvgSize={avg_size:.2f}, Positive={positive}/7")

    # Summary
    print("\n" + "="*160)
    print("SUMMARY: Position Sizing Models Comparison")
    print("="*160)
    print(f"{'Model':<35} {'AvgSharpe':<10} {'AvgCAGR':<10} {'AvgMDD':<10} {'CAGR/MDD':<10} {'AvgSize':<10} {'Pos':<6} {'BTC_CAGR':<10} {'BTC_MDD':<10}")
    print("-"*150)

    for desc, results in all_results.items():
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_cagr = np.mean([r["cagr"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        avg_cagr_mdd = np.mean([r["cagr_mdd"] for r in results])
        avg_size = np.mean([r["avg_size"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)
        btc = next(r for r in results if r["symbol"] == "BTCUSDT")

        print(f"{desc:<35} {avg_sharpe:<10.2f} {avg_cagr*100:<9.1f}% {avg_mdd*100:<9.1f}% "
              f"{avg_cagr_mdd:<10.2f} {avg_size:<10.2f} {positive}/7    {btc['cagr']*100:<9.1f}% {btc['mdd']*100:<9.1f}%")

    # Best by CAGR/MDD
    print("\n" + "="*160)
    print("TOP 5 BY CAGR/MDD (MDD < 35%)")
    print("="*160)

    valid_configs = [(desc, results) for desc, results in all_results.items()
                     if np.mean([r["mdd"] for r in results]) < 0.35]

    if valid_configs:
        sorted_configs = sorted(valid_configs,
                               key=lambda x: np.mean([r["cagr_mdd"] for r in x[1]]),
                               reverse=True)[:5]

        for i, (desc, results) in enumerate(sorted_configs, 1):
            avg_sharpe = np.mean([r["sharpe"] for r in results])
            avg_cagr = np.mean([r["cagr"] for r in results])
            avg_mdd = np.mean([r["mdd"] for r in results])
            avg_cagr_mdd = np.mean([r["cagr_mdd"] for r in results])
            btc = next(r for r in results if r["symbol"] == "BTCUSDT")

            print(f"{i}. {desc}")
            print(f"   Avg: Sharpe={avg_sharpe:.2f}, CAGR={avg_cagr*100:.1f}%, MDD={avg_mdd*100:.1f}%, CAGR/MDD={avg_cagr_mdd:.2f}")
            print(f"   BTC: CAGR={btc['cagr']*100:.1f}%, MDD={btc['mdd']*100:.1f}%, CAGR/MDD={btc['cagr_mdd']:.2f}")
            print()


if __name__ == "__main__":
    main()
