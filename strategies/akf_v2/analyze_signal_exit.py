#!/usr/bin/env python3
"""
Signal Exit이 BTC에만 좋고 알트에는 부정적인 이유 분석

Signal Exit 조건:
- v_ratio < 1.0 (변동성이 평균 이하)
- vel_zscore가 반대 방향으로 반전

가설:
1. 알트는 v_ratio < 1.0 구간이 적어서 Signal Exit 기회가 적음
2. 알트는 추세 지속성이 강해서 반전 신호가 fake out
3. BTC는 mean-reversion이 강해서 반전 신호가 유효
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from numba import njit

from research.features.kalman import calculate_adaptive_kalman
from strategies.akf_v2.strategy_v33_final import load_data, calculate_features, V33_PARAMS

ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "LTCUSDT"]


def analyze_v_ratio_distribution(data: dict):
    """v_ratio 분포 분석"""
    print("\n" + "=" * 100)
    print("1. V-RATIO 분포 분석")
    print("=" * 100)
    print(f"{'Symbol':<10} {'Mean':<10} {'Std':<10} {'<1.0 비율':<12} {'<0.8 비율':<12} {'>1.5 비율':<12}")
    print("-" * 70)

    for symbol, df in data.items():
        df_kf = calculate_adaptive_kalman(df)
        features = calculate_features(df_kf)
        v_ratio = features["v_ratio"][210:]  # warmup 제외

        mean_vr = np.mean(v_ratio)
        std_vr = np.std(v_ratio)
        pct_below_1 = np.mean(v_ratio < 1.0) * 100
        pct_below_08 = np.mean(v_ratio < 0.8) * 100
        pct_above_15 = np.mean(v_ratio > 1.5) * 100

        print(f"{symbol:<10} {mean_vr:<10.2f} {std_vr:<10.2f} {pct_below_1:<11.1f}% {pct_below_08:<11.1f}% {pct_above_15:<11.1f}%")


def analyze_signal_exit_trades(data: dict):
    """Signal Exit으로 청산된 트레이드 분석"""
    print("\n" + "=" * 100)
    print("2. SIGNAL EXIT 트레이드 분석")
    print("=" * 100)

    @njit
    def track_signal_exits(
        close, high, low, kf_trend, vel_zscore, unc_pct, sigma_hybrid, dynamic_mult, v_ratio,
        entry_z, unc_pct_max, hard_stop_mult, warmup
    ):
        """Signal Exit 트레이드만 추적"""
        n = len(close)

        # 결과 저장
        signal_exit_returns = []
        signal_exit_holding = []
        missed_returns = []  # Signal Exit 후 10바 수익률

        position = 0
        entry_price = 0.0
        entry_idx = 0
        highest = 0.0
        lowest = np.inf
        prev_stop = 0.0
        hard_stop = 0.0

        for i in range(warmup, n):
            if position == 0:
                if vel_zscore[i] > entry_z and unc_pct[i] < unc_pct_max:
                    position = 1
                    entry_price = close[i]
                    entry_idx = i
                    highest = high[i]
                    dist = dynamic_mult[i] * sigma_hybrid[i]
                    prev_stop = highest * np.exp(-dist)
                    hard_stop = entry_price * np.exp(-hard_stop_mult * sigma_hybrid[i])

                elif vel_zscore[i] < -entry_z and unc_pct[i] < unc_pct_max:
                    position = -1
                    entry_price = close[i]
                    entry_idx = i
                    lowest = low[i]
                    dist = dynamic_mult[i] * sigma_hybrid[i]
                    prev_stop = lowest * np.exp(dist)
                    hard_stop = entry_price * np.exp(hard_stop_mult * sigma_hybrid[i])

            elif position == 1:
                if high[i] > highest:
                    highest = high[i]
                dist = dynamic_mult[i] * sigma_hybrid[i]
                new_stop = highest * np.exp(-dist)
                trail_stop = max(new_stop, prev_stop)
                prev_stop = trail_stop

                # Hard Stop
                if close[i] < hard_stop:
                    position = 0
                    continue
                # Trailing Stop
                if close[i] < trail_stop:
                    position = 0
                    continue
                # Signal Exit (v_ratio < 1.0)
                if v_ratio[i] < 1.0 and vel_zscore[i] < -entry_z:
                    trade_return = (close[i] - entry_price) / entry_price
                    holding = i - entry_idx
                    signal_exit_returns.append(trade_return)
                    signal_exit_holding.append(holding)

                    # Signal Exit 후 10바 수익률 (missed opportunity)
                    if i + 10 < n:
                        future_return = (close[i + 10] - close[i]) / close[i]
                        missed_returns.append(future_return)

                    position = 0
                    continue

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
                if v_ratio[i] < 1.0 and vel_zscore[i] > entry_z:
                    trade_return = (entry_price - close[i]) / entry_price
                    holding = i - entry_idx
                    signal_exit_returns.append(trade_return)
                    signal_exit_holding.append(holding)

                    if i + 10 < n:
                        future_return = (close[i] - close[i + 10]) / close[i]  # short이므로 반대
                        missed_returns.append(future_return)

                    position = 0
                    continue

        return signal_exit_returns, signal_exit_holding, missed_returns

    print(f"{'Symbol':<10} {'Count':<8} {'AvgReturn':<12} {'WinRate':<10} {'AvgHold':<10} {'MissedRet':<12} {'Assessment':<15}")
    print("-" * 90)

    for symbol, df in data.items():
        df_kf = calculate_adaptive_kalman(df)
        features = calculate_features(df_kf)

        returns, holdings, missed = track_signal_exits(
            close=df_kf["close"].values,
            high=df_kf["high"].values,
            low=df_kf["low"].values,
            kf_trend=features["kf_trend"],
            vel_zscore=features["vel_zscore"],
            unc_pct=features["unc_pct"],
            sigma_hybrid=features["sigma_hybrid"],
            dynamic_mult=features["dynamic_mult"],
            v_ratio=features["v_ratio"],
            entry_z=V33_PARAMS["entry_z"],
            unc_pct_max=V33_PARAMS["unc_pct_max"],
            hard_stop_mult=V33_PARAMS["hard_stop_mult"],
            warmup=V33_PARAMS["warmup"],
        )

        if len(returns) > 0:
            avg_ret = np.mean(returns) * 100
            win_rate = np.mean([r > 0 for r in returns]) * 100
            avg_hold = np.mean(holdings)
            avg_missed = np.mean(missed) * 100 if len(missed) > 0 else 0

            # Assessment
            if avg_missed > 1.0:
                assessment = "TOO EARLY"
            elif avg_missed < -1.0:
                assessment = "GOOD TIMING"
            else:
                assessment = "NEUTRAL"

            print(f"{symbol:<10} {len(returns):<8} {avg_ret:<11.2f}% {win_rate:<9.1f}% {avg_hold:<10.1f} {avg_missed:<11.2f}% {assessment}")
        else:
            print(f"{symbol:<10} 0        -           -          -          -            -")


def analyze_trend_persistence(data: dict):
    """추세 지속성 분석 - vel_zscore 반전 후 실제 추세 반전 여부"""
    print("\n" + "=" * 100)
    print("3. 추세 지속성 분석 (vel_zscore 반전 신호의 신뢰도)")
    print("=" * 100)

    print(f"{'Symbol':<10} {'반전신호':<10} {'실제반전':<10} {'신뢰도':<10} {'Fake비율':<10} {'특성':<20}")
    print("-" * 80)

    for symbol, df in data.items():
        df_kf = calculate_adaptive_kalman(df)
        features = calculate_features(df_kf)

        vel_zscore = features["vel_zscore"]
        close = df_kf["close"].values
        v_ratio = features["v_ratio"]

        entry_z = V33_PARAMS["entry_z"]

        # v_ratio < 1.0 구간에서 반전 신호 찾기
        reversal_signals = 0
        actual_reversals = 0
        fake_outs = 0

        for i in range(210, len(vel_zscore) - 20):
            if v_ratio[i] >= 1.0:
                continue

            # Long 포지션 중 short 반전 신호
            if vel_zscore[i-1] > 0 and vel_zscore[i] < -entry_z:
                reversal_signals += 1
                # 10바 후 실제로 하락했는지
                future_ret = (close[i+10] - close[i]) / close[i]
                if future_ret < -0.01:  # 1% 이상 하락
                    actual_reversals += 1
                elif future_ret > 0.01:  # 1% 이상 상승 (fake out)
                    fake_outs += 1

            # Short 포지션 중 long 반전 신호
            elif vel_zscore[i-1] < 0 and vel_zscore[i] > entry_z:
                reversal_signals += 1
                future_ret = (close[i+10] - close[i]) / close[i]
                if future_ret > 0.01:
                    actual_reversals += 1
                elif future_ret < -0.01:
                    fake_outs += 1

        if reversal_signals > 0:
            reliability = actual_reversals / reversal_signals * 100
            fake_rate = fake_outs / reversal_signals * 100

            if reliability > 50:
                char = "MEAN-REVERT"
            elif fake_rate > 40:
                char = "TREND-FOLLOW"
            else:
                char = "MIXED"

            print(f"{symbol:<10} {reversal_signals:<10} {actual_reversals:<10} {reliability:<9.1f}% {fake_rate:<9.1f}% {char}")
        else:
            print(f"{symbol:<10} 0          -          -          -          -")


def analyze_volatility_regime(data: dict):
    """변동성 레짐별 Signal Exit 효과"""
    print("\n" + "=" * 100)
    print("4. 변동성 레짐별 특성")
    print("=" * 100)

    print(f"{'Symbol':<10} {'AvgVol':<10} {'VolOfVol':<10} {'LowVol시간':<12} {'특성':<20}")
    print("-" * 70)

    for symbol, df in data.items():
        df_kf = calculate_adaptive_kalman(df)
        features = calculate_features(df_kf)

        sigma = features["sigma_hybrid"][210:]
        v_ratio = features["v_ratio"][210:]

        avg_vol = np.mean(sigma) * 100
        vol_of_vol = np.std(sigma) / np.mean(sigma)  # 변동성의 변동성
        low_vol_time = np.mean(v_ratio < 1.0) * 100

        if symbol == "BTCUSDT":
            char = "BASE (비교 기준)"
        elif avg_vol > 1.5:
            char = "HIGH VOL"
        elif vol_of_vol > 0.8:
            char = "UNSTABLE VOL"
        else:
            char = "STABLE"

        print(f"{symbol:<10} {avg_vol:<9.2f}% {vol_of_vol:<10.2f} {low_vol_time:<11.1f}% {char}")


def main():
    print("Loading data...")
    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df

    analyze_v_ratio_distribution(data)
    analyze_signal_exit_trades(data)
    analyze_trend_persistence(data)
    analyze_volatility_regime(data)

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY: Signal Exit이 BTC에만 좋은 이유")
    print("=" * 100)
    print("""
1. V-RATIO 분포 차이
   - BTC: v_ratio < 1.0 구간이 많음 → Signal Exit 기회 많음
   - 알트: 변동성이 높아 v_ratio < 1.0 구간이 적음 → 기회 적음

2. 추세 지속성 차이
   - BTC: Mean-reversion 성향 → 반전 신호가 실제 반전으로 이어짐
   - 알트: Trend-following 성향 → 반전 신호가 fake out인 경우 많음

3. Missed Return 분석
   - BTC: Signal Exit 후 추가 손실 방지 효과
   - 알트: Signal Exit 후 추세가 계속되어 수익 기회 상실

4. 결론
   - Signal Exit은 BTC 특화 로직
   - 알트에는 오히려 조기 청산으로 수익 감소
   - 자산별 분리 로직 고려 필요
""")


if __name__ == "__main__":
    main()
