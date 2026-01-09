#!/usr/bin/env python3
"""
Dual-Eye Kalman Filter 스탑 테스트

Trend Eye (R_close): 추세 업데이트용 - 부드럽게
Risk Eye (R_parkinson): 스탑 계산용 - 봉 내 변동폭 반영

S_risk = P_pred + R_parkinson
Stop = kf_trend_pred × exp(-k × √S_risk)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from research.features.kalman import calculate_adaptive_kalman
from strategies.akf_v2.strategy_v34_final import load_data, calculate_features, V34_PARAMS


def main():
    print("Loading BTC data...")
    df = load_data("BTCUSDT")
    df_kf = calculate_adaptive_kalman(df)

    print(f"Total bars: {len(df_kf):,}")

    # Dual-Eye 시스템 출력 확인
    print("\n" + "="*80)
    print("Dual-Eye Kalman Filter 출력")
    print("="*80)
    print(f"kf_innovation_cov (S_trend):      {df_kf['kf_innovation_cov'].notna().sum():,} rows")
    print(f"kf_innovation_cov_risk (S_risk):  {df_kf['kf_innovation_cov_risk'].notna().sum():,} rows")
    print(f"kf_stop_distance (3×√S_risk):     {df_kf['kf_stop_distance'].notna().sum():,} rows")

    # 값 추출
    S_trend = df_kf["kf_innovation_cov"].values
    S_risk = df_kf["kf_innovation_cov_risk"].values
    stop_dist = df_kf["kf_stop_distance"].values
    kf_trend_pred = df_kf["kf_trend_pred"].values
    close = df_kf["close"].values
    low = df_kf["low"].values
    high = df_kf["high"].values

    sqrt_S_trend = np.sqrt(S_trend)
    sqrt_S_risk = np.sqrt(S_risk)

    # 분포 비교
    print("\n" + "="*80)
    print("S_trend vs S_risk 비교 (Dual-Eye)")
    print("="*80)
    print(f"sqrt(S_trend) 평균: {np.nanmean(sqrt_S_trend)*100:.4f}% (Close 기반)")
    print(f"sqrt(S_risk) 평균:  {np.nanmean(sqrt_S_risk)*100:.4f}% (Parkinson 기반)")
    print(f"S_risk / S_trend:   {np.nanmean(S_risk / (S_trend + 1e-10)):.2f}x")
    print("\n→ Parkinson R이 Close R보다 크면 > 1.0x (봉 내 변동폭 반영)")

    # kf_stop_distance (3×√S_risk) 분포
    print("\n" + "="*80)
    print("kf_stop_distance 분포 (k=3, √S_risk 기반)")
    print("="*80)
    print(f"Mean:   {np.nanmean(stop_dist)*100:.2f}%")
    print(f"Median: {np.nanmedian(stop_dist)*100:.2f}%")
    print(f"Min:    {np.nanmin(stop_dist)*100:.2f}%")
    print(f"Max:    {np.nanmax(stop_dist)*100:.2f}%")

    # 현재 방식 (7 × sigma_hybrid)과 비교
    features = calculate_features(df_kf, V34_PARAMS)
    sigma_hybrid = features["sigma_hybrid"]
    dist_hybrid = 1 - np.exp(-7 * sigma_hybrid)

    print(f"\n현재 (7×σ_hybrid): {np.nanmean(dist_hybrid)*100:.2f}%")
    print(f"신규 (3×√S_risk):  {np.nanmean(stop_dist)*100:.2f}%")

    # k 값별 스탑 거리
    print("\n" + "="*80)
    print("k 값별 스탑 거리 (√S_risk 기반)")
    print("="*80)
    target_dist = np.nanmean(dist_hybrid)
    print(f"Target (현재): {target_dist*100:.2f}%")
    print(f"\n{'k':<5} {'평균 스탑 거리':<20}")
    print("-"*30)

    for k in [2, 3, 4, 5, 6]:
        dist_k = np.nanmean(1 - np.exp(-k * sqrt_S_risk))
        marker = " <-- 현재와 유사" if abs(dist_k - target_dist) < 0.01 else ""
        print(f"{k:<5} {dist_k*100:<19.2f}%{marker}")

    # Mahalanobis Distance (S_risk 기준)
    print("\n" + "="*80)
    print("Mahalanobis Distance (√S_risk 기준)")
    print("="*80)
    log_resid = np.log(close) - np.log(kf_trend_pred)
    mahal = np.abs(log_resid) / (sqrt_S_risk + 1e-10)

    print(f"Mean:   {np.nanmean(mahal):.2f}σ")
    print(f"95%ile: {np.nanpercentile(mahal, 95):.2f}σ")
    print(f"99%ile: {np.nanpercentile(mahal, 99):.2f}σ")

    # Low 기준 스탑 터치 (실제 사용 시나리오)
    print("\n" + "="*80)
    print("실제 스탑 터치 시뮬레이션 (Long 포지션 기준)")
    print("="*80)
    print("Stop = kf_trend_pred × exp(-k × √S_risk)")
    print("터치 조건: low < stop")
    print(f"\n{'k':<5} {'터치 빈도':<15} {'평균 스탑 거리':<20}")
    print("-"*45)

    for k in [2, 3, 4, 5]:
        stop_price = kf_trend_pred * np.exp(-k * sqrt_S_risk)
        touched = low < stop_price
        touch_pct = np.nanmean(touched) * 100
        avg_dist = np.nanmean(1 - np.exp(-k * sqrt_S_risk)) * 100
        print(f"{k:<5} {touch_pct:<14.2f}% {avg_dist:<19.2f}%")

    # 결론
    print("\n" + "="*80)
    print("결론: Dual-Eye Kalman Stop")
    print("="*80)
    print("""
장점:
1. 이론적 근거: Kalman Filter의 Innovation Covariance 사용
2. 봉 내 변동폭 반영: Parkinson R → 휩소 방어
3. 적응형: 시장 상황에 따라 자동 조정
4. 깔끔한 k: k=3 = "3σ 이탈 시 청산"

권장 설정:
- k=3: 타이트 (통계적 의미 있는 최대값)
- k=4~5: 현재 7×σ_hybrid와 유사한 레벨

사용법:
  stop_long = kf_trend_pred × exp(-kf_stop_distance)  # k=3 기본
  stop_short = kf_trend_pred × exp(+kf_stop_distance)
""")


if __name__ == "__main__":
    main()
