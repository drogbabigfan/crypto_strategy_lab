#!/usr/bin/env python3
"""
1% Percentile threshold 분석
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from research.features.kalman import calculate_adaptive_kalman

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


def analyze_thresholds(symbol: str):
    df = load_data(symbol)
    df_kf = calculate_adaptive_kalman(df)

    velocity = df_kf["kf_velocity"].values
    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=42, min_periods=5).mean()
    vel_std = vel_series.rolling(window=42, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10))

    # Rolling percentiles
    valid_idx = 540

    results = {}
    for pct in [1, 2, 3, 5]:
        threshold = vel_zscore.rolling(window=540, min_periods=100).quantile(1 - pct/100)
        valid = threshold.iloc[valid_idx:]
        results[pct] = {
            "mean": valid.mean(),
            "min": valid.min(),
            "max": valid.max(),
        }

    return results


def main():
    print("=" * 80)
    print("Percentile Threshold 비교 (1%, 2%, 3%, 5%)")
    print("=" * 80)

    for symbol in ALL_SYMBOLS[:4]:  # BTC, ETH, XRP, SOL
        stats = analyze_thresholds(symbol)
        print(f"\n[{symbol}]")
        print(f"  {'Pct':<6} {'Mean':<8} {'Min':<8} {'Max':<8}")
        print(f"  {'-'*30}")
        for pct in [1, 2, 3, 5]:
            s = stats[pct]
            print(f"  {pct}%     {s['mean']:<8.2f} {s['min']:<8.2f} {s['max']:<8.2f}")

    print("\n" + "=" * 80)
    print("참고: 고정 z=2.0 ≈ 상위 2.3% (정규분포 기준)")
    print("=" * 80)


if __name__ == "__main__":
    main()
