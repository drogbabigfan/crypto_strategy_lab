#!/usr/bin/env python3
"""
Cross-Asset 분석: 왜 BTC에서만 작동하는가?

1. KF 피처 분포 비교 (velocity, uncertainty, residuals)
2. Z-score 분포 비교
3. 트렌드 품질 비교
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


def analyze_symbol(symbol: str, df: pd.DataFrame):
    """심볼별 KF 피처 분석"""
    df_kf = calculate_adaptive_kalman(df)

    velocity = df_kf["kf_velocity"].values
    uncertainty = df_kf["kf_uncertainty"].values
    close = df_kf["close"].values
    kf_trend = df_kf["kf_trend"].values

    # Log residuals
    log_price = np.log(close)
    log_trend = np.log(kf_trend)
    log_residuals = log_price - log_trend

    # Velocity z-score
    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=42, min_periods=5).mean()
    vel_std = vel_series.rolling(window=42, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

    # Uncertainty percentile
    unc_pct = (
        pd.Series(uncertainty)
        .rolling(window=210, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )

    # 유효 데이터만 (warmup 이후)
    valid_idx = 210
    vel_zscore_valid = vel_zscore[valid_idx:]
    unc_pct_valid = unc_pct[valid_idx:]
    log_resid_valid = log_residuals[valid_idx:]

    # 통계
    stats = {
        "symbol": symbol,
        "bars": len(df),
        # Velocity z-score 분포
        "zscore_mean": np.nanmean(vel_zscore_valid),
        "zscore_std": np.nanstd(vel_zscore_valid),
        "zscore_skew": pd.Series(vel_zscore_valid).skew(),
        "zscore_kurt": pd.Series(vel_zscore_valid).kurtosis(),
        # z > 2.0 비율 (롱 진입 가능 비율)
        "pct_z_gt_2": np.mean(vel_zscore_valid > 2.0) * 100,
        "pct_z_lt_m2": np.mean(vel_zscore_valid < -2.0) * 100,
        # Uncertainty 분포
        "unc_mean": np.nanmean(unc_pct_valid),
        "unc_lt_50": np.mean(unc_pct_valid < 0.5) * 100,
        # Residual (변동성)
        "resid_std_mean": np.nanstd(log_resid_valid) * 100,  # %
        # 트렌드 품질: R² (price vs trend)
        "trend_r2": np.corrcoef(close[valid_idx:], kf_trend[valid_idx:])[0,1]**2,
    }

    return stats, df_kf


def main():
    print("=" * 100)
    print("Cross-Asset KF Feature Analysis")
    print("=" * 100)

    all_stats = []

    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is None:
            continue
        stats, _ = analyze_symbol(symbol, df)
        all_stats.append(stats)

    # 테이블 출력
    print(f"\n[1] Velocity Z-score 분포")
    print(f"{'Symbol':<10} {'Mean':<8} {'Std':<8} {'Skew':<8} {'Kurt':<8} {'z>2.0%':<10} {'z<-2.0%':<10}")
    print("-" * 70)
    for s in all_stats:
        print(f"{s['symbol']:<10} {s['zscore_mean']:<8.2f} {s['zscore_std']:<8.2f} "
              f"{s['zscore_skew']:<8.2f} {s['zscore_kurt']:<8.2f} "
              f"{s['pct_z_gt_2']:<10.1f} {s['pct_z_lt_m2']:<10.1f}")

    print(f"\n[2] Uncertainty & Trend Quality")
    print(f"{'Symbol':<10} {'Unc<50%':<12} {'Resid Std':<12} {'Trend R²':<10}")
    print("-" * 50)
    for s in all_stats:
        print(f"{s['symbol']:<10} {s['unc_lt_50']:<12.1f} {s['resid_std_mean']:<12.2f}% {s['trend_r2']:<10.4f}")

    # 핵심 인사이트
    print("\n" + "=" * 100)
    print("Key Insights")
    print("=" * 100)

    btc = next(s for s in all_stats if s["symbol"] == "BTCUSDT")

    print(f"\n[BTC baseline]")
    print(f"  z>2.0 비율: {btc['pct_z_gt_2']:.1f}%")
    print(f"  Trend R²: {btc['trend_r2']:.4f}")
    print(f"  Resid Std: {btc['resid_std_mean']:.2f}%")

    print(f"\n[다른 코인 vs BTC]")
    for s in all_stats:
        if s["symbol"] != "BTCUSDT":
            z_ratio = s["pct_z_gt_2"] / btc["pct_z_gt_2"]
            r2_diff = s["trend_r2"] - btc["trend_r2"]
            vol_ratio = s["resid_std_mean"] / btc["resid_std_mean"]
            print(f"  {s['symbol']:<10}: z>2 비율 {z_ratio:.2f}x, R² diff {r2_diff:+.4f}, 변동성 {vol_ratio:.2f}x")


if __name__ == "__main__":
    main()
