#!/usr/bin/env python3
"""
Bar 크기별 백테스트 비교 스크립트.

Usage:
    python compare_bar_sizes.py [--min-size 0] [--max-size 2] [--method kf_prob]
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import pandas as pd

from research.features.kalman import calculate_adaptive_kalman
from strategies.akf_v2.common import SizingConfig
from strategies.akf_v2.common.sizing import calculate_position_sizes
from strategies.akf_v2.backtest_pyramid import run_pyramid_backtest, PyramidBacktestConfig

DATA_ROOT = PROJECT_ROOT / "etl/data"


def load_data(bar_size: int, start_year: int = 2020, end_year: int = 2025) -> pd.DataFrame:
    """데이터 로드."""
    data_dir = DATA_ROOT / f"features-{bar_size}/futures/BTCUSDT"
    dfs = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            path = data_dir / f"BTCUSDT-features-{year}-{month:02d}.parquet"
            if path.exists():
                dfs.append(pd.read_parquet(path))
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def generate_signals(df_kf: pd.DataFrame) -> np.ndarray:
    """시그널 생성."""
    velocity = df_kf["kf_velocity"].values
    uncertainty = df_kf["kf_uncertainty"].values
    n = len(df_kf)

    # 코인 기준 고정값 (7일=42 bars, 30일=210 bars)
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

    # WFA 최적 파라미터: entry=2.0, exit=-2.0
    entry_long = (vel_zscore > 2.0) & low_uncertainty
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
            if vel_zscore[i] < -2.0:  # WFA 최적: exit=-2.0
                position = 0
            else:
                signals[i] = 1
        elif position == -1:
            if vel_zscore[i] > 2.0:  # V2.1: short exit=+2.0
                position = 0
            else:
                signals[i] = -1

    return signals


def run_backtest(
    df: pd.DataFrame,
    sizing_method: str = "kf_prob",
    min_size: float = 0.0,
    max_size: float = 2.0,
) -> dict:
    """백테스트 실행."""
    # Kalman Filter
    df_kf = calculate_adaptive_kalman(df)

    # 시그널 생성
    signals = generate_signals(df_kf)
    df_kf["signal"] = signals

    # Sizing
    config = SizingConfig(method=sizing_method, min_size=min_size, max_size=max_size)
    sizes = calculate_position_sizes(df_kf, config, entry_signals=signals)
    df_kf["position_size"] = sizes
    df_kf["sl_price"] = 0.0

    # 백테스트
    bt_config = PyramidBacktestConfig(
        initial_capital=100000.0,
        compounding=True,
        fee_rate=0.001,
        slippage_rate=0.0001,
    )

    result = run_pyramid_backtest(df_kf, bt_config)
    result["bars"] = len(df_kf)
    return result


def main():
    parser = argparse.ArgumentParser(description="Bar 크기별 백테스트 비교")
    parser.add_argument("--min-size", type=float, default=0.0, help="최소 포지션 사이즈")
    parser.add_argument("--max-size", type=float, default=2.0, help="최대 포지션 사이즈")
    parser.add_argument(
        "--method",
        type=str,
        default="kf_prob",
        choices=["fixed", "kelly", "kf_prob"],
        help="사이징 방식",
    )
    parser.add_argument(
        "--bar-sizes",
        type=int,
        nargs="+",
        default=[6, 24, 50, 100],
        help="테스트할 bar 크기들",
    )
    args = parser.parse_args()

    print("=" * 80)
    print(f"Bar 크기별 백테스트 비교")
    print(f"  Method: {args.method}, Size: [{args.min_size}, {args.max_size}]")
    print("=" * 80)

    results = {}

    for bar_size in args.bar_sizes:
        print(f"\n[Bar-{bar_size}] 로딩 중...")
        df = load_data(bar_size)
        if df is None:
            print(f"  데이터 없음")
            continue

        print(f"  {len(df):,} bars 로드됨")
        result = run_backtest(df, args.method, args.min_size, args.max_size)
        results[bar_size] = result
        print(
            f"  완료: Sharpe={result['sharpe_ratio']:.2f}, "
            f"PnL={result['total_pnl']*100:.1f}%, "
            f"MDD={result['max_drawdown']*100:.1f}%"
        )

    # 결과 테이블
    print("\n" + "=" * 80)
    print(
        f"{'Bar':<8} {'Bars':<10} {'Trades':<8} {'PnL':<12} "
        f"{'Sharpe':<10} {'MDD':<10} {'WinRate':<10} {'AvgSize':<10}"
    )
    print("-" * 80)
    for bar_size in args.bar_sizes:
        if bar_size in results:
            r = results[bar_size]
            print(
                f"{bar_size:<8} {r['bars']:<10,} {r['total_trades']:<8} "
                f"{r['total_pnl']*100:>8.1f}%   {r['sharpe_ratio']:<10.2f} "
                f"{r['max_drawdown']*100:<9.1f}% {r['win_rate']*100:<9.1f}% "
                f"{r['avg_size']:<10.2f}"
            )
    print("=" * 80)


if __name__ == "__main__":
    main()
