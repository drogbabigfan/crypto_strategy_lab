#!/usr/bin/env python3
"""
Combined Stop-Loss 테스트:
1. 평단 기준 손절: entry_price ± (mult * resid_std) - 진입 직후 하락 방어
2. KF Trend 기준 손절: kf_trend ± (mult * resid_std) - 익절 중 이익 보호 (trailing)

두 손절선 중 더 보수적인 값(Long: 높은 값, Short: 낮은 값) 적용
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


def generate_signals_combined_stop(
    df_kf,
    entry_mult=None,    # 평단 기준 손절 배수
    kf_mult=None,       # KF Trend 기준 손절 배수
    resid_window=180,
):
    """
    Combined stop-loss: 평단 기준 + KF Trend 기준

    entry_mult: 평단 기준 손절 (entry_price ± mult * resid_std)
    kf_mult: KF Trend 기준 손절 (kf_trend ± mult * resid_std)
    """
    velocity = df_kf["kf_velocity"].values
    uncertainty = df_kf["kf_uncertainty"].values
    close = df_kf["close"].values
    kf_trend = df_kf["kf_trend"].values
    n = len(df_kf)

    # 잔차 계산 (log space)
    log_price = np.log(close)
    log_trend = np.log(kf_trend)
    log_residuals = log_price - log_trend

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
    entry_log_price = 0.0  # 진입 시점의 log(price)

    entry_stop_hits = 0
    kf_stop_hits = 0

    for i in range(210, n):
        if position == 0:
            if entry_long[i]:
                position = 1
                entry_log_price = log_price[i]
                signals[i] = 1
            elif entry_short[i]:
                position = -1
                entry_log_price = log_price[i]
                signals[i] = -1

        elif position == 1:  # Long
            stopped = False

            if not np.isnan(resid_std[i]):
                # 1. 평단 기준 손절: log(price) < log(entry) - mult * resid_std
                if entry_mult is not None:
                    entry_stop = entry_log_price - entry_mult * resid_std[i]
                    if log_price[i] < entry_stop:
                        position = 0
                        entry_stop_hits += 1
                        stopped = True

                # 2. KF Trend 기준 손절: log(price) < log(trend) - mult * resid_std
                if not stopped and kf_mult is not None:
                    kf_stop = log_trend[i] - kf_mult * resid_std[i]
                    if log_price[i] < kf_stop:
                        position = 0
                        kf_stop_hits += 1
                        stopped = True

            if stopped:
                continue

            # 일반 청산 조건
            if vel_zscore[i] < -2.0:
                position = 0
            else:
                signals[i] = 1

        elif position == -1:  # Short
            stopped = False

            if not np.isnan(resid_std[i]):
                # 1. 평단 기준 손절: log(price) > log(entry) + mult * resid_std
                if entry_mult is not None:
                    entry_stop = entry_log_price + entry_mult * resid_std[i]
                    if log_price[i] > entry_stop:
                        position = 0
                        entry_stop_hits += 1
                        stopped = True

                # 2. KF Trend 기준 손절: log(price) > log(trend) + mult * resid_std
                if not stopped and kf_mult is not None:
                    kf_stop = log_trend[i] + kf_mult * resid_std[i]
                    if log_price[i] > kf_stop:
                        position = 0
                        kf_stop_hits += 1
                        stopped = True

            if stopped:
                continue

            # 일반 청산 조건
            if vel_zscore[i] > 2.0:
                position = 0
            else:
                signals[i] = -1

    return signals, entry_stop_hits, kf_stop_hits


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

    # Test configurations: (name, entry_mult, kf_mult)
    configs = [
        ("V2.1 (현재)", None, None),
        ("KF 3σ만", None, 3.0),
        ("평단 3σ만", 3.0, None),
        ("평단 3σ + KF 3σ", 3.0, 3.0),
        ("평단 2σ + KF 3σ", 2.0, 3.0),
        ("평단 3σ + KF 2σ", 3.0, 2.0),
        ("평단 2.5σ + KF 2.5σ", 2.5, 2.5),
    ]

    print("\n" + "=" * 130)
    print("Combined Stop-Loss Test: 평단 기준 + KF Trend 기준")
    print("=" * 130)
    print(f"{'Config':<24} {'Sharpe':<10} {'PnL':<14} {'MDD':<12} {'Trades':<10} {'WinRate':<10} {'EntryStop':<12} {'KFStop':<10}")
    print("-" * 130)

    for name, entry_mult, kf_mult in configs:
        signals, entry_stops, kf_stops = generate_signals_combined_stop(
            df_kf, entry_mult=entry_mult, kf_mult=kf_mult
        )

        df_test = df_kf.copy()
        df_test["signal"] = signals
        df_test["position_size"] = calculate_position_sizes(df_test, sizing_config, entry_signals=signals)
        df_test["sl_price"] = 0.0

        metrics = run_pyramid_backtest(df_test, bt_config)

        print(f"{name:<24} {metrics['sharpe_ratio']:<10.2f} {metrics['total_pnl']*100:<13.1f}% "
              f"{metrics['max_drawdown']*100:<11.1f}% {metrics['total_trades']:<10} "
              f"{metrics['win_rate']*100:<9.1f}% {entry_stops:<12} {kf_stops:<10}")

    print("=" * 130)

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
