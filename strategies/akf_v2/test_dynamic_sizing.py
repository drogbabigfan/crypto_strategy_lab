#!/usr/bin/env python3
"""
Dynamic Position Sizing 테스트

1단계: Base Size (리스크 기반)
   - Risk Target R = 2%
   - Hard Stop Distance D = 5 × σ_hybrid
   - Base Size = R / D

2단계: Sigmoid Confidence Scaling
   - M_confidence = 1.0 + 0.5 × tanh(λ × (0.5 - unc_pct))
   - unc_pct 낮을수록 → 최대 1.5배
   - unc_pct 높을수록 → 최소 0.5배

Final Size = Base Size × M_confidence
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
    """V3.3 features"""
    close = df_kf["close"].values
    kf_trend = df_kf["kf_trend"].values
    kf_uncertainty = df_kf["kf_uncertainty"].values
    velocity = df_kf["kf_velocity"].values

    # Hybrid Vol
    log_returns = np.log(close[1:] / close[:-1])
    log_returns = np.concatenate([[0], log_returns])
    rv = pd.Series(log_returns).rolling(window=42, min_periods=10).std().values
    sqrt_unc = np.sqrt(kf_uncertainty)
    sigma_hybrid = np.maximum(sqrt_unc, rv)

    # V-Ratio & Dynamic Mult
    baseline = pd.Series(sigma_hybrid).rolling(window=120, min_periods=30).mean().values
    v_ratio = sigma_hybrid / (baseline + 1e-10)
    dynamic_mult = np.clip(5.0 / (v_ratio + 1e-10), 1.25, 7.5)

    # Velocity z-score
    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=42, min_periods=5).mean()
    vel_std = vel_series.rolling(window=42, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

    # Uncertainty percentile
    unc_pct = (
        pd.Series(kf_uncertainty)
        .rolling(window=210, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )

    # Residual std
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
def calculate_position_size(sigma_hybrid: float, unc_pct: float,
                            risk_target: float = 0.02,
                            hard_stop_mult: float = 5.0,
                            conf_lambda: float = 2.5,
                            size_min: float = 0.1,
                            size_max: float = 2.0):
    """
    Dynamic Position Sizing

    1. Base Size = Risk Target / Hard Stop Distance
    2. Confidence Scaling = 1.0 + 0.5 * tanh(λ * (0.5 - unc_pct))
    """
    # 1. Hard Stop Distance
    hard_stop_dist = hard_stop_mult * sigma_hybrid

    # 2. Base Size
    base_size = risk_target / (hard_stop_dist + 1e-10)

    # 3. Confidence Scaling (Sigmoid)
    # unc_pct 낮을수록 → 1.5배, 높을수록 → 0.5배
    conf_scale = 1.0 + 0.5 * np.tanh(conf_lambda * (0.5 - unc_pct))

    # 4. Final Size
    final_size = base_size * conf_scale

    # 5. Clip
    final_size = max(size_min, min(size_max, final_size))

    return final_size, base_size, conf_scale


@njit
def generate_signals_with_sizing(
    close, high, low, kf_trend, vel_zscore, unc_pct, sigma_hybrid, dynamic_mult, v_ratio, resid_std,
    entry_z=2.0, v_ratio_threshold=1.0, base_innov=2.5, innov_min=1.5, innov_max=4.0,
    hard_mult=5.0, risk_target=0.02, conf_lambda=2.5, size_min=0.1, size_max=2.0, warmup=210
):
    """V3.3 + Dynamic Sizing"""
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

                # Dynamic sizing
                size, _, _ = calculate_position_size(
                    sigma_hybrid[i], unc_pct[i], risk_target, hard_mult, conf_lambda, size_min, size_max
                )
                current_size = size

                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = highest * np.exp(-dist)
                hard_stop = entry_price * np.exp(-hard_mult * sigma_hybrid[i])
                signals[i] = 1
                position_sizes[i] = current_size

            elif vel_zscore[i] < -entry_z and unc_pct[i] < 0.5:
                position = -1
                entry_price = close[i]
                lowest = low[i]

                # Dynamic sizing
                size, _, _ = calculate_position_size(
                    sigma_hybrid[i], unc_pct[i], risk_target, hard_mult, conf_lambda, size_min, size_max
                )
                current_size = size

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


def run_backtest_dynamic(df: pd.DataFrame, risk_target: float = 0.02, conf_lambda: float = 2.5,
                         size_min: float = 0.1, size_max: float = 2.0):
    """Run V3.3 with dynamic sizing"""
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
        risk_target=risk_target,
        conf_lambda=conf_lambda,
        size_min=size_min,
        size_max=size_max,
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

    # 사이즈 통계
    active_sizes = position_sizes[position_sizes > 0]
    metrics["avg_size"] = np.mean(active_sizes) if len(active_sizes) > 0 else 0
    metrics["min_size"] = np.min(active_sizes) if len(active_sizes) > 0 else 0
    metrics["max_size"] = np.max(active_sizes) if len(active_sizes) > 0 else 0

    return metrics


def run_backtest_fixed(df: pd.DataFrame):
    """Run V3.3 with fixed sizing (baseline)"""
    df_kf = calculate_adaptive_kalman(df)
    features = calculate_features(df_kf)

    signals, _ = generate_signals_with_sizing(
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
        size_min=1.0,
        size_max=1.0,
    )

    df_kf["signal"] = signals
    df_kf["position_size"] = np.where(signals != 0, 1.0, 0.0)
    df_kf["sl_price"] = 0.0

    bt_config = PyramidBacktestConfig(
        initial_capital=100000.0,
        compounding=True,
        fee_rate=0.001,
        slippage_rate=0.0001,
    )

    return run_pyramid_backtest(df_kf, bt_config)


def main():
    print("Loading data...")
    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars")

    # 1. Fixed Sizing (Baseline)
    print("\n" + "=" * 120)
    print("V3.3 Fixed Sizing (Baseline)")
    print("=" * 120)
    print(f"{'Symbol':<10} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'Trades':<8} {'WinRate':<10}")
    print("-" * 70)

    fixed_results = []
    for symbol, df in data.items():
        metrics = run_backtest_fixed(df)
        fixed_results.append({
            "symbol": symbol,
            "sharpe": metrics["sharpe_ratio"],
            "pnl": metrics["total_pnl"],
            "mdd": metrics["max_drawdown"],
            "trades": metrics["total_trades"],
            "win_rate": metrics["win_rate"],
        })
        print(f"{symbol:<10} {metrics['sharpe_ratio']:<10.2f} {metrics['total_pnl']*100:<11.1f}% "
              f"{metrics['max_drawdown']*100:<9.1f}% {metrics['total_trades']:<8} {metrics['win_rate']*100:<9.1f}%")

    avg_sharpe = np.mean([r["sharpe"] for r in fixed_results])
    avg_mdd = np.mean([r["mdd"] for r in fixed_results])
    print("-" * 70)
    print(f"Average: Sharpe={avg_sharpe:.2f}, MDD={avg_mdd*100:.1f}%")

    # 2. Dynamic Sizing
    configs = [
        (0.02, 2.5, 0.1, 2.0, "R=2%, λ=2.5, [0.1, 2.0]"),
        (0.02, 2.5, 0.2, 1.5, "R=2%, λ=2.5, [0.2, 1.5]"),
        (0.01, 2.5, 0.1, 1.5, "R=1%, λ=2.5, [0.1, 1.5]"),
        (0.03, 2.5, 0.2, 2.0, "R=3%, λ=2.5, [0.2, 2.0]"),
    ]

    all_dynamic_results = {}

    for risk_target, conf_lambda, size_min, size_max, desc in configs:
        print("\n" + "=" * 120)
        print(f"Dynamic Sizing: {desc}")
        print("=" * 120)
        print(f"{'Symbol':<10} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'Trades':<8} {'AvgSize':<10} {'MinSize':<10} {'MaxSize':<10}")
        print("-" * 100)

        results = []
        for symbol, df in data.items():
            metrics = run_backtest_dynamic(df, risk_target, conf_lambda, size_min, size_max)
            results.append({
                "symbol": symbol,
                "sharpe": metrics["sharpe_ratio"],
                "pnl": metrics["total_pnl"],
                "mdd": metrics["max_drawdown"],
                "trades": metrics["total_trades"],
                "avg_size": metrics["avg_size"],
                "min_size": metrics["min_size"],
                "max_size": metrics["max_size"],
            })
            print(f"{symbol:<10} {metrics['sharpe_ratio']:<10.2f} {metrics['total_pnl']*100:<11.1f}% "
                  f"{metrics['max_drawdown']*100:<9.1f}% {metrics['total_trades']:<8} "
                  f"{metrics['avg_size']:<10.2f} {metrics['min_size']:<10.2f} {metrics['max_size']:<10.2f}")

        all_dynamic_results[desc] = results

        print("-" * 100)
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)

        print(f"Average: Sharpe={avg_sharpe:.2f}, PnL={avg_pnl*100:.1f}%, MDD={avg_mdd*100:.1f}%, Positive={positive}/7")

    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY COMPARISON")
    print("=" * 120)
    print(f"{'Config':<30} {'AvgSharpe':<12} {'AvgPnL':<12} {'AvgMDD':<12} {'Positive':<10}")
    print("-" * 80)

    # Fixed
    avg_sharpe = np.mean([r["sharpe"] for r in fixed_results])
    avg_pnl = np.mean([r["pnl"] for r in fixed_results])
    avg_mdd = np.mean([r["mdd"] for r in fixed_results])
    positive = sum(1 for r in fixed_results if r["pnl"] > 0)
    print(f"{'Fixed (1.0)':<30} {avg_sharpe:<12.2f} {avg_pnl*100:<11.1f}% {avg_mdd*100:<11.1f}% {positive}/7")

    # Dynamic
    for desc, results in all_dynamic_results.items():
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)
        print(f"{desc:<30} {avg_sharpe:<12.2f} {avg_pnl*100:<11.1f}% {avg_mdd*100:<11.1f}% {positive}/7")

    # MDD 개선 비교
    print("\n" + "=" * 120)
    print("MDD COMPARISON (Fixed vs Dynamic R=2%)")
    print("=" * 120)

    best_dynamic = all_dynamic_results["R=2%, λ=2.5, [0.1, 2.0]"]

    print(f"{'Symbol':<10} {'Fixed MDD':<12} {'Dynamic MDD':<12} {'Δ MDD':<12} {'Fixed Size':<12} {'Dyn AvgSize':<12}")
    print("-" * 80)

    for i, symbol in enumerate(ALL_SYMBOLS):
        fixed = fixed_results[i]
        dynamic = best_dynamic[i]
        delta_mdd = (dynamic["mdd"] - fixed["mdd"]) * 100
        print(f"{symbol:<10} {fixed['mdd']*100:<11.1f}% {dynamic['mdd']*100:<11.1f}% {delta_mdd:+11.1f}% "
              f"{'1.00':<12} {dynamic['avg_size']:<12.2f}")


if __name__ == "__main__":
    main()
