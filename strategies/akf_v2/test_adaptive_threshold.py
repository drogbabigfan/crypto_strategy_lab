#!/usr/bin/env python3
"""
동적 Threshold 테스트

1. 변동성 기반: entry_z = base_z * (vol / avg_vol)
2. Uncertainty 기반: entry_z = base_z * (1 + unc_pct)
3. 조합: 둘 다 적용
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


def generate_signals_adaptive(
    df_kf,
    mode="fixed",        # fixed, vol_scaled, unc_scaled, combined
    base_entry_z=2.0,
    base_exit_z=-2.0,
    vol_window=180,
    stop_mult=3.0,
):
    """
    Adaptive threshold signal generation.

    Modes:
    - fixed: 고정 threshold (V2.2 기본)
    - vol_scaled: 변동성 기반 조정
    - unc_scaled: uncertainty 기반 조정
    - combined: 둘 다 적용
    """
    velocity = df_kf["kf_velocity"].values
    uncertainty = df_kf["kf_uncertainty"].values
    close = df_kf["close"].values
    kf_trend = df_kf["kf_trend"].values
    n = len(df_kf)

    # Log space
    log_price = np.log(close)
    log_trend = np.log(kf_trend)
    log_residuals = log_price - log_trend

    # Residual std (변동성)
    resid_std = pd.Series(log_residuals).rolling(window=vol_window, min_periods=30).std().values

    # 변동성의 rolling 평균 (정규화용)
    vol_mean = pd.Series(resid_std).rolling(window=vol_window, min_periods=30).mean().values

    # Velocity z-score
    vel_series = pd.Series(velocity)
    vel_mean_roll = vel_series.rolling(window=42, min_periods=5).mean()
    vel_std_roll = vel_series.rolling(window=42, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean_roll) / (vel_std_roll + 1e-10)).values

    # Uncertainty percentile
    unc_pct = (
        pd.Series(uncertainty)
        .rolling(window=210, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )

    signals = np.zeros(n, dtype=np.int8)
    position = 0
    stop_hits = 0
    warmup = 210

    for i in range(warmup, n):
        # 동적 threshold 계산
        if mode == "fixed":
            entry_z = base_entry_z
            exit_z = base_exit_z
        elif mode == "vol_scaled":
            # 변동성 높으면 threshold 높임
            vol_ratio = resid_std[i] / (vol_mean[i] + 1e-10) if not np.isnan(vol_mean[i]) else 1.0
            vol_ratio = np.clip(vol_ratio, 0.5, 2.0)  # 0.5x ~ 2x 범위
            entry_z = base_entry_z * vol_ratio
            exit_z = base_exit_z * vol_ratio
        elif mode == "unc_scaled":
            # 불확실성 높으면 threshold 높임
            unc_factor = 0.5 + unc_pct[i]  # 0.5 ~ 1.5 범위
            entry_z = base_entry_z * unc_factor
            exit_z = base_exit_z * unc_factor
        elif mode == "combined":
            vol_ratio = resid_std[i] / (vol_mean[i] + 1e-10) if not np.isnan(vol_mean[i]) else 1.0
            vol_ratio = np.clip(vol_ratio, 0.5, 2.0)
            unc_factor = 0.5 + unc_pct[i]
            combined_factor = (vol_ratio + unc_factor) / 2  # 평균
            entry_z = base_entry_z * combined_factor
            exit_z = base_exit_z * combined_factor

        # Uncertainty filter (고정)
        low_uncertainty = unc_pct[i] < 0.5

        if position == 0:
            if vel_zscore[i] > entry_z and low_uncertainty:
                position = 1
                signals[i] = 1
            elif vel_zscore[i] < -entry_z and low_uncertainty:
                position = -1
                signals[i] = -1

        elif position == 1:
            # KF Stop
            if stop_mult is not None and not np.isnan(resid_std[i]):
                if log_residuals[i] < -stop_mult * resid_std[i]:
                    position = 0
                    stop_hits += 1
                    continue

            if vel_zscore[i] < exit_z:
                position = 0
            else:
                signals[i] = 1

        elif position == -1:
            # KF Stop
            if stop_mult is not None and not np.isnan(resid_std[i]):
                if log_residuals[i] > stop_mult * resid_std[i]:
                    position = 0
                    stop_hits += 1
                    continue

            if vel_zscore[i] > -exit_z:
                position = 0
            else:
                signals[i] = -1

    return signals, stop_hits


def run_backtest(df, mode, base_entry_z=2.0):
    df_kf = calculate_adaptive_kalman(df)
    signals, stop_hits = generate_signals_adaptive(df_kf, mode=mode, base_entry_z=base_entry_z)

    df_kf["signal"] = signals
    sizing_config = SizingConfig(method="fixed", min_size=1.0, max_size=1.0)
    df_kf["position_size"] = calculate_position_sizes(df_kf, sizing_config, entry_signals=signals)
    df_kf["sl_price"] = 0.0

    bt_config = PyramidBacktestConfig(
        initial_capital=100000.0,
        compounding=True,
        fee_rate=0.001,
        slippage_rate=0.0001,
    )

    metrics = run_pyramid_backtest(df_kf, bt_config)
    metrics["stop_hits"] = stop_hits
    return metrics


def main():
    modes = ["fixed", "vol_scaled", "unc_scaled", "combined"]

    print("=" * 140)
    print("Adaptive Threshold Test - All Symbols")
    print("=" * 140)

    # 각 모드별 결과 저장
    all_results = {mode: [] for mode in modes}

    for symbol in ALL_SYMBOLS:
        print(f"\n[{symbol}]")
        df = load_data(symbol)
        if df is None:
            print("  No data")
            continue

        print(f"  {'Mode':<15} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'Trades':<8} {'WinRate':<10}")
        print(f"  {'-'*65}")

        for mode in modes:
            metrics = run_backtest(df, mode)
            all_results[mode].append({
                "symbol": symbol,
                "sharpe": metrics["sharpe_ratio"],
                "pnl": metrics["total_pnl"],
                "mdd": metrics["max_drawdown"],
                "trades": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
            })
            print(f"  {mode:<15} {metrics['sharpe_ratio']:<10.2f} {metrics['total_pnl']*100:<11.1f}% "
                  f"{metrics['max_drawdown']*100:<9.1f}% {metrics['total_trades']:<8} {metrics['win_rate']*100:<9.1f}%")

    # Summary
    print("\n" + "=" * 140)
    print("Summary: Average across all symbols")
    print("=" * 140)
    print(f"{'Mode':<15} {'Avg Sharpe':<12} {'Avg PnL':<14} {'Avg MDD':<12} {'Positive PnL':<12}")
    print("-" * 65)

    for mode in modes:
        results = all_results[mode]
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        positive_count = sum(1 for r in results if r["pnl"] > 0)

        print(f"{mode:<15} {avg_sharpe:<12.2f} {avg_pnl*100:<13.1f}% {avg_mdd*100:<11.1f}% {positive_count}/{len(results)}")

    print("=" * 140)


if __name__ == "__main__":
    main()
