#!/usr/bin/env python3
"""
Dynamic Stop-Loss 테스트 (KF Residual 기반).

stop_price = kf_trend ± (mult * resid_std)
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


def load_data():
    data_dir = DATA_ROOT / "features-6/futures/BTCUSDT"
    dfs = []
    for year in range(2020, 2026):
        for month in range(1, 13):
            path = data_dir / f"BTCUSDT-features-{year}-{month:02d}.parquet"
            if path.exists():
                dfs.append(pd.read_parquet(path))
    return pd.concat(dfs, ignore_index=True)


def generate_signals_with_dynamic_stop(df_kf, resid_mult=None, resid_window=180):
    """
    Dynamic stop-loss based on KF residuals.

    resid_mult: 잔차 표준편차 배수 (2.0 = 2σ), None이면 비활성화
    resid_window: 잔차 std 계산 윈도우 (180 bars = 30일)
    """
    velocity = df_kf["kf_velocity"].values
    uncertainty = df_kf["kf_uncertainty"].values
    close = df_kf["close"].values
    kf_trend = df_kf["kf_trend"].values
    n = len(df_kf)

    # 잔차 계산 (log space에서 계산 -> % 단위)
    log_price = np.log(close)
    log_trend = np.log(kf_trend)
    log_residuals = log_price - log_trend  # log(price/trend)

    # 잔차의 rolling std
    resid_std = pd.Series(log_residuals).rolling(window=resid_window, min_periods=30).std().values

    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=42, min_periods=5).mean()
    vel_std = vel_series.rolling(window=42, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

    unc_pct = (
        pd.Series(uncertainty)
        .rolling(window=210, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )
    low_uncertainty = unc_pct < 0.5

    entry_long = (vel_zscore > 2.0) & low_uncertainty
    entry_short = (vel_zscore < -2.0) & low_uncertainty

    signals = np.zeros(n, dtype=np.int8)
    position = 0
    stop_hits = 0

    for i in range(210, n):
        if position == 0:
            if entry_long[i]:
                position = 1
                signals[i] = 1
            elif entry_short[i]:
                position = -1
                signals[i] = -1
        elif position == 1:  # Long
            # Dynamic stop: log(price) < log(trend) - mult * resid_std
            if resid_mult is not None and not np.isnan(resid_std[i]):
                stop_threshold = -resid_mult * resid_std[i]
                if log_residuals[i] < stop_threshold:
                    position = 0
                    stop_hits += 1
                    continue

            if vel_zscore[i] < -2.0:
                position = 0
            else:
                signals[i] = 1

        elif position == -1:  # Short
            # Dynamic stop: log(price) > log(trend) + mult * resid_std
            if resid_mult is not None and not np.isnan(resid_std[i]):
                stop_threshold = resid_mult * resid_std[i]
                if log_residuals[i] > stop_threshold:
                    position = 0
                    stop_hits += 1
                    continue

            if vel_zscore[i] > 2.0:
                position = 0
            else:
                signals[i] = -1

    return signals, stop_hits


def main():
    print("Loading data...")
    df = load_data()
    df_kf = calculate_adaptive_kalman(df)

    sizing_config = SizingConfig(method="fixed", min_size=1.0, max_size=1.0)
    bt_config = PyramidBacktestConfig(
        initial_capital=100000.0,
        compounding=True,
        fee_rate=0.001,
        slippage_rate=0.0001,
    )

    # Test configurations
    configs = [
        ("V2.1 (현재)", None, 180),
        ("+ 1.5σ Stop (30일)", 1.5, 180),
        ("+ 2σ Stop (30일)", 2.0, 180),
        ("+ 2.5σ Stop (30일)", 2.5, 180),
        ("+ 3σ Stop (30일)", 3.0, 180),
        ("+ 2σ Stop (14일)", 2.0, 84),
    ]

    print("\n" + "=" * 110)
    print("Dynamic Stop-Loss (KF Residual 기반)")
    print("=" * 110)
    print(f"{'Config':<22} {'Sharpe':<10} {'PnL':<14} {'MDD':<12} {'Trades':<10} {'WinRate':<10} {'StopHits':<10}")
    print("-" * 110)

    for name, mult, window in configs:
        signals, stop_hits = generate_signals_with_dynamic_stop(df_kf, resid_mult=mult, resid_window=window)

        df_test = df_kf.copy()
        df_test["signal"] = signals
        df_test["position_size"] = calculate_position_sizes(df_test, sizing_config, entry_signals=signals)
        df_test["sl_price"] = 0.0

        metrics = run_pyramid_backtest(df_test, bt_config)

        print(f"{name:<22} {metrics['sharpe_ratio']:<10.2f} {metrics['total_pnl']*100:<13.1f}% "
              f"{metrics['max_drawdown']*100:<11.1f}% {metrics['total_trades']:<10} "
              f"{metrics['win_rate']*100:<9.1f}% {stop_hits:<10}")

    print("=" * 110)

    # 잔차 통계
    log_price = np.log(df_kf["close"].values)
    log_trend = np.log(df_kf["kf_trend"].values)
    log_residuals = log_price - log_trend
    resid_std = pd.Series(log_residuals).rolling(window=180, min_periods=30).std()

    print(f"\n[잔차 통계 (log space)]")
    print(f"  평균 resid_std: {resid_std.mean()*100:.2f}%")
    print(f"  2σ 손절선: ±{resid_std.mean()*2*100:.2f}%")
    print(f"  3σ 손절선: ±{resid_std.mean()*3*100:.2f}%")


if __name__ == "__main__":
    main()
