#!/usr/bin/env python3
"""
Percentile threshold가 실제로 어떤 값인지 분석
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from research.features.kalman import calculate_adaptive_kalman

DATA_ROOT = PROJECT_ROOT / "etl/data"


def load_data(symbol: str, bar_size: int = 6):
    data_dir = DATA_ROOT / f"features-{bar_size}/futures/{symbol}"
    dfs = []
    for year in range(2020, 2026):
        for month in range(1, 13):
            path = data_dir / f"{symbol}-features-{year}-{month:02d}.parquet"
            if path.exists():
                dfs.append(pd.read_parquet(path))
    return pd.concat(dfs, ignore_index=True) if dfs else None


def analyze_thresholds(symbol: str):
    df = load_data(symbol)
    df_kf = calculate_adaptive_kalman(df)

    velocity = df_kf["kf_velocity"].values
    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=42, min_periods=5).mean()
    vel_std = vel_series.rolling(window=42, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10))

    # Rolling 95th percentile (top 5%)
    pct_95 = vel_zscore.rolling(window=540, min_periods=100).quantile(0.95)

    # 유효 구간만
    valid_idx = 540
    pct_95_valid = pct_95.iloc[valid_idx:]

    print(f"\n[{symbol}] 95th Percentile Threshold 통계:")
    print(f"  Mean: {pct_95_valid.mean():.2f}")
    print(f"  Min:  {pct_95_valid.min():.2f}")
    print(f"  Max:  {pct_95_valid.max():.2f}")
    print(f"  Std:  {pct_95_valid.std():.2f}")

    # 고정 z=2.0 대비 percentile이 낮은 비율
    below_2 = (pct_95_valid < 2.0).mean() * 100
    print(f"  Threshold < 2.0 비율: {below_2:.1f}%")

    return {
        "symbol": symbol,
        "mean": pct_95_valid.mean(),
        "min": pct_95_valid.min(),
        "max": pct_95_valid.max(),
        "below_2_pct": below_2,
    }


def main():
    print("=" * 60)
    print("Percentile Threshold Analysis")
    print("고정 z=2.0 vs Rolling 95th Percentile")
    print("=" * 60)

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    results = []

    for symbol in symbols:
        stats = analyze_thresholds(symbol)
        results.append(stats)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Symbol':<12} {'Mean':<8} {'Min':<8} {'Max':<8} {'<2.0 비율':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['symbol']:<12} {r['mean']:<8.2f} {r['min']:<8.2f} {r['max']:<8.2f} {r['below_2_pct']:<9.1f}%")

    print("\n결론:")
    print("  - Percentile threshold < 2.0인 기간 = 약한 신호도 진입")
    print("  - 이 비율이 높을수록 불필요한 매매 증가")


if __name__ == "__main__":
    main()
