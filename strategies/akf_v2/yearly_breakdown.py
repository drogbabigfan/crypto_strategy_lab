#!/usr/bin/env python3
"""
연도별 성과 분석.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from research.features.kalman import calculate_adaptive_kalman
from strategies.akf_v2.common import SizingConfig
from strategies.akf_v2.common.sizing import calculate_position_sizes
from strategies.akf_v2.backtest_pyramid import run_pyramid_backtest, PyramidBacktestConfig

DATA_ROOT = PROJECT_ROOT / "etl/data"


def load_year_data(bar_size: int, year: int):
    data_dir = DATA_ROOT / f"features-{bar_size}/futures/BTCUSDT"
    dfs = []
    for month in range(1, 13):
        path = data_dir / f"BTCUSDT-features-{year}-{month:02d}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
    return pd.concat(dfs, ignore_index=True) if dfs else None


def generate_signals(df_kf, entry_z=2.0, exit_z=-2.0):
    velocity = df_kf["kf_velocity"].values
    uncertainty = df_kf["kf_uncertainty"].values
    n = len(df_kf)

    zscore_window = 42
    uncertainty_window = 210
    warmup = 210

    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=zscore_window, min_periods=5).mean()
    vel_std = vel_series.rolling(window=zscore_window, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

    p_pct = (
        pd.Series(uncertainty)
        .rolling(window=uncertainty_window, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )
    low_uncertainty = p_pct < 0.5

    entry_long = (vel_zscore > entry_z) & low_uncertainty
    entry_short = (vel_zscore < -2.0) & low_uncertainty

    signals = np.zeros(n, dtype=np.int8)
    position = 0

    for i in range(warmup, n):
        if position == 0:
            if entry_long[i]:
                position = 1
                signals[i] = 1
            elif entry_short[i]:
                position = -1
                signals[i] = -1
        elif position == 1:
            if vel_zscore[i] < exit_z:
                position = 0
            else:
                signals[i] = 1
        elif position == -1:
            if vel_zscore[i] > 2.0:  # V2.1: short exit=+2.0
                position = 0
            else:
                signals[i] = -1

    return signals


def run_strategy(years, sizing_config, label):
    """전략 백테스트."""
    bt_config = PyramidBacktestConfig(
        initial_capital=100000.0,
        compounding=True,
        fee_rate=0.001,
        slippage_rate=0.0001,
    )

    results = []
    for year in years:
        df = load_year_data(6, year)
        if df is None:
            continue

        df_kf = calculate_adaptive_kalman(df)
        signals = generate_signals(df_kf)

        df_kf["signal"] = signals
        df_kf["position_size"] = calculate_position_sizes(df_kf, sizing_config, entry_signals=signals)
        df_kf["sl_price"] = 0.0

        metrics = run_pyramid_backtest(df_kf, bt_config)

        results.append({
            "year": year,
            "pnl": metrics["total_pnl"],
            "mdd": metrics["max_drawdown"],
            "sharpe": metrics["sharpe_ratio"],
        })

    return results


def run_buy_hold(years):
    """Buy & Hold 백테스트."""
    results = []
    for year in years:
        df = load_year_data(6, year)
        if df is None:
            continue

        start_price = df["close"].iloc[0]
        end_price = df["close"].iloc[-1]
        pnl = (end_price - start_price) / start_price

        # MDD 계산
        prices = df["close"].values
        peak = prices[0]
        max_dd = 0
        for p in prices:
            if p > peak:
                peak = p
            dd = (peak - p) / peak
            if dd > max_dd:
                max_dd = dd

        # Sharpe 계산 (일간 수익률 기준)
        returns = df["close"].pct_change().dropna()
        sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(365 * 6)  # 연환산

        results.append({
            "year": year,
            "pnl": pnl,
            "mdd": max_dd,
            "sharpe": sharpe,
        })

    return results


def main():
    years = [2020, 2021, 2022, 2023, 2024, 2025]

    # 3가지 방식 비교
    kf_prob_config = SizingConfig(method="kf_prob", min_size=0.0, max_size=2.0)
    fixed_config = SizingConfig(method="fixed", min_size=1.0, max_size=1.0)

    kf_prob_results = run_strategy(years, kf_prob_config, "kf_prob")
    fixed_results = run_strategy(years, fixed_config, "fixed")
    bh_results = run_buy_hold(years)

    # 비교 테이블
    print("=" * 100)
    print("연도별 성과 비교: kf_prob vs Fixed(1.0) vs Buy&Hold")
    print("=" * 100)
    print(f"{'Year':<8} {'kf_prob PnL':<14} {'Fixed PnL':<14} {'B&H PnL':<14} "
          f"{'kf_prob MDD':<14} {'Fixed MDD':<14} {'B&H MDD':<14}")
    print("-" * 100)

    for i, year in enumerate(years):
        kf = kf_prob_results[i]
        fx = fixed_results[i]
        bh = bh_results[i]
        print(f"{year:<8} {kf['pnl']*100:>10.1f}%    {fx['pnl']*100:>10.1f}%    {bh['pnl']*100:>10.1f}%    "
              f"{kf['mdd']*100:>10.1f}%    {fx['mdd']*100:>10.1f}%    {bh['mdd']*100:>10.1f}%")

    print("-" * 100)

    # 누적 통계
    kf_cum = np.prod([1 + r["pnl"] for r in kf_prob_results]) - 1
    fx_cum = np.prod([1 + r["pnl"] for r in fixed_results]) - 1
    bh_cum = np.prod([1 + r["pnl"] for r in bh_results]) - 1

    kf_avg_mdd = np.mean([r["mdd"] for r in kf_prob_results])
    fx_avg_mdd = np.mean([r["mdd"] for r in fixed_results])
    bh_avg_mdd = np.mean([r["mdd"] for r in bh_results])

    kf_avg_sharpe = np.mean([r["sharpe"] for r in kf_prob_results])
    fx_avg_sharpe = np.mean([r["sharpe"] for r in fixed_results])
    bh_avg_sharpe = np.mean([r["sharpe"] for r in bh_results])

    print(f"{'Cumul':<8} {kf_cum*100:>10.1f}%    {fx_cum*100:>10.1f}%    {bh_cum*100:>10.1f}%    "
          f"{kf_avg_mdd*100:>10.1f}%    {fx_avg_mdd*100:>10.1f}%    {bh_avg_mdd*100:>10.1f}%")
    print("=" * 100)

    # Sharpe 비교
    print(f"\nSharpe 비교:")
    print(f"  kf_prob: {kf_avg_sharpe:.2f}")
    print(f"  Fixed:   {fx_avg_sharpe:.2f}")
    print(f"  B&H:     {bh_avg_sharpe:.2f}")


if __name__ == "__main__":
    main()
