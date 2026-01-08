#!/usr/bin/env python3
"""
Signal Exit 차이의 근본 원인 분석

단순 vol로는 설명 안됨. 더 깊이 분석:
1. 수익률 자기상관 (Autocorrelation) - 모멘텀 vs 평균회귀
2. vel_zscore 반전 신호의 신뢰도
3. 저변동성(v_ratio<1) 구간에서의 추세 특성
4. Signal Exit 발생 시점의 시장 상태
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from research.features.kalman import calculate_adaptive_kalman
from strategies.akf_v2.strategy_v33_final import load_data, calculate_features, V33_PARAMS

ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "LTCUSDT"]


def analyze_return_autocorrelation(data: dict):
    """수익률 자기상관 분석 - 모멘텀 vs 평균회귀"""
    print("\n" + "=" * 100)
    print("1. 수익률 자기상관 (Autocorrelation)")
    print("   > 0: 모멘텀 (추세 지속)")
    print("   < 0: 평균회귀 (반전)")
    print("=" * 100)
    print(f"{'Symbol':<10} {'AC(1)':<10} {'AC(5)':<10} {'AC(10)':<10} {'AC(20)':<10} {'특성':<15}")
    print("-" * 70)

    for symbol, df in data.items():
        close = df["close"].values
        returns = np.diff(np.log(close))

        # 자기상관 계산
        ac1 = np.corrcoef(returns[1:], returns[:-1])[0, 1]
        ac5 = np.corrcoef(returns[5:], returns[:-5])[0, 1]
        ac10 = np.corrcoef(returns[10:], returns[:-10])[0, 1]
        ac20 = np.corrcoef(returns[20:], returns[:-20])[0, 1]

        # 특성 판단
        if ac1 > 0.02 and ac5 > 0.01:
            char = "MOMENTUM"
        elif ac1 < -0.02:
            char = "MEAN-REVERT"
        else:
            char = "RANDOM"

        print(f"{symbol:<10} {ac1:<10.4f} {ac5:<10.4f} {ac10:<10.4f} {ac20:<10.4f} {char}")


def analyze_zscore_reversal_accuracy(data: dict):
    """vel_zscore 반전 신호의 정확도"""
    print("\n" + "=" * 100)
    print("2. vel_zscore 반전 신호 정확도")
    print("   v_ratio < 1.0 구간에서 Z가 반전될 때, 실제로 가격도 반전하는지")
    print("=" * 100)
    print(f"{'Symbol':<10} {'신호수':<8} {'정확':<8} {'틀림':<8} {'정확도':<10} {'Avg이익':<12} {'Avg손실':<12}")
    print("-" * 80)

    entry_z = V33_PARAMS["entry_z"]

    for symbol, df in data.items():
        df_kf = calculate_adaptive_kalman(df)
        features = calculate_features(df_kf)

        vel_zscore = features["vel_zscore"]
        v_ratio = features["v_ratio"]
        close = df_kf["close"].values

        correct = 0
        wrong = 0
        correct_returns = []
        wrong_returns = []

        for i in range(211, len(vel_zscore) - 20):
            if v_ratio[i] >= 1.0:
                continue

            # Long 청산 신호: Z가 +에서 -entry_z 이하로
            if vel_zscore[i-1] > 0 and vel_zscore[i] < -entry_z:
                future_ret = (close[i+10] - close[i]) / close[i]
                if future_ret < 0:  # 실제로 하락 (청산 정확)
                    correct += 1
                    correct_returns.append(abs(future_ret))
                else:  # 오히려 상승 (청산 실수)
                    wrong += 1
                    wrong_returns.append(future_ret)

            # Short 청산 신호: Z가 -에서 +entry_z 이상으로
            elif vel_zscore[i-1] < 0 and vel_zscore[i] > entry_z:
                future_ret = (close[i+10] - close[i]) / close[i]
                if future_ret > 0:  # 실제로 상승 (청산 정확)
                    correct += 1
                    correct_returns.append(abs(future_ret))
                else:  # 오히려 하락 (청산 실수)
                    wrong += 1
                    wrong_returns.append(abs(future_ret))

        total = correct + wrong
        if total > 0:
            accuracy = correct / total * 100
            avg_correct = np.mean(correct_returns) * 100 if correct_returns else 0
            avg_wrong = np.mean(wrong_returns) * 100 if wrong_returns else 0
            print(f"{symbol:<10} {total:<8} {correct:<8} {wrong:<8} {accuracy:<9.1f}% {avg_correct:<11.2f}% {avg_wrong:<11.2f}%")
        else:
            print(f"{symbol:<10} 0        -        -        -          -            -")


def analyze_low_vol_regime_behavior(data: dict):
    """v_ratio < 1.0 구간에서의 가격 행동 분석"""
    print("\n" + "=" * 100)
    print("3. 저변동성 구간(v_ratio < 1.0) 특성")
    print("   이 구간에서 추세가 지속되는지, 반전되는지")
    print("=" * 100)
    print(f"{'Symbol':<10} {'구간수':<8} {'추세지속':<10} {'반전':<10} {'횡보':<10} {'지속률':<10}")
    print("-" * 70)

    for symbol, df in data.items():
        df_kf = calculate_adaptive_kalman(df)
        features = calculate_features(df_kf)

        v_ratio = features["v_ratio"]
        vel_zscore = features["vel_zscore"]
        close = df_kf["close"].values

        trend_continue = 0
        trend_reverse = 0
        sideways = 0

        i = 210
        while i < len(v_ratio) - 20:
            # 저변동성 구간 시작
            if v_ratio[i] < 1.0 and v_ratio[i-1] >= 1.0:
                entry_dir = 1 if vel_zscore[i] > 0 else -1
                entry_price = close[i]

                # 구간 끝 찾기
                j = i + 1
                while j < len(v_ratio) and v_ratio[j] < 1.0:
                    j += 1

                if j < len(v_ratio):
                    exit_price = close[min(j, len(close)-1)]
                    ret = (exit_price - entry_price) / entry_price

                    if abs(ret) < 0.01:  # 1% 미만 변화
                        sideways += 1
                    elif (ret > 0 and entry_dir > 0) or (ret < 0 and entry_dir < 0):
                        trend_continue += 1
                    else:
                        trend_reverse += 1

                i = j
            else:
                i += 1

        total = trend_continue + trend_reverse + sideways
        if total > 0:
            continue_rate = trend_continue / total * 100
            print(f"{symbol:<10} {total:<8} {trend_continue:<10} {trend_reverse:<10} {sideways:<10} {continue_rate:<9.1f}%")


def analyze_signal_exit_context(data: dict):
    """Signal Exit 발생 시점의 컨텍스트 분석"""
    print("\n" + "=" * 100)
    print("4. Signal Exit 발생 시점 분석")
    print("   어떤 상황에서 Signal Exit이 발생하고, 그 후 어떻게 되는지")
    print("=" * 100)
    print(f"{'Symbol':<10} {'진입Z':<8} {'청산Z':<8} {'보유기간':<10} {'진입→청산':<12} {'청산→10바':<12} {'순효과':<10}")
    print("-" * 90)

    entry_z = V33_PARAMS["entry_z"]

    for symbol, df in data.items():
        df_kf = calculate_adaptive_kalman(df)
        features = calculate_features(df_kf)

        vel_zscore = features["vel_zscore"]
        v_ratio = features["v_ratio"]
        unc_pct = features["unc_pct"]
        close = df_kf["close"].values

        entry_zs = []
        exit_zs = []
        holdings = []
        entry_to_exit_rets = []
        exit_to_future_rets = []

        position = 0
        entry_price = 0.0
        entry_idx = 0
        entry_z_val = 0.0

        for i in range(210, len(vel_zscore) - 10):
            if position == 0:
                if vel_zscore[i] > entry_z and unc_pct[i] < 0.5:
                    position = 1
                    entry_price = close[i]
                    entry_idx = i
                    entry_z_val = vel_zscore[i]
                elif vel_zscore[i] < -entry_z and unc_pct[i] < 0.5:
                    position = -1
                    entry_price = close[i]
                    entry_idx = i
                    entry_z_val = vel_zscore[i]

            elif position == 1:
                # Signal Exit 조건
                if v_ratio[i] < 1.0 and vel_zscore[i] < -entry_z:
                    entry_zs.append(entry_z_val)
                    exit_zs.append(vel_zscore[i])
                    holdings.append(i - entry_idx)
                    entry_to_exit_rets.append((close[i] - entry_price) / entry_price * 100)
                    exit_to_future_rets.append((close[i+10] - close[i]) / close[i] * 100)
                    position = 0

            elif position == -1:
                if v_ratio[i] < 1.0 and vel_zscore[i] > entry_z:
                    entry_zs.append(entry_z_val)
                    exit_zs.append(vel_zscore[i])
                    holdings.append(i - entry_idx)
                    entry_to_exit_rets.append((entry_price - close[i]) / entry_price * 100)
                    exit_to_future_rets.append((close[i] - close[i+10]) / close[i] * 100)  # short
                    position = 0

        if len(entry_zs) > 0:
            avg_entry_z = np.mean(np.abs(entry_zs))
            avg_exit_z = np.mean(np.abs(exit_zs))
            avg_hold = np.mean(holdings)
            avg_entry_exit = np.mean(entry_to_exit_rets)
            avg_exit_future = np.mean(exit_to_future_rets)
            net_effect = avg_exit_future  # 음수면 좋음 (손실 회피)

            print(f"{symbol:<10} {avg_entry_z:<8.1f} {avg_exit_z:<8.1f} {avg_hold:<10.1f} "
                  f"{avg_entry_exit:<11.2f}% {avg_exit_future:<11.2f}% {'+' if net_effect > 0 else ''}{net_effect:<9.2f}%")


def analyze_momentum_vs_meanreversion(data: dict):
    """모멘텀 vs 평균회귀 스코어"""
    print("\n" + "=" * 100)
    print("5. 모멘텀 vs 평균회귀 스코어")
    print("   가격이 vel_zscore 방향으로 계속 가는지, 반대로 가는지")
    print("=" * 100)
    print(f"{'Symbol':<10} {'Z방향 지속':<12} {'Z방향 반전':<12} {'지속률':<10} {'특성':<15}")
    print("-" * 60)

    for symbol, df in data.items():
        df_kf = calculate_adaptive_kalman(df)
        features = calculate_features(df_kf)

        vel_zscore = features["vel_zscore"]
        close = df_kf["close"].values

        continue_count = 0
        reverse_count = 0

        for i in range(210, len(vel_zscore) - 10):
            z = vel_zscore[i]
            if abs(z) < 1.0:  # 중립 구간 스킵
                continue

            future_ret = (close[i+10] - close[i]) / close[i]

            # Z 방향과 미래 수익률 방향 비교
            if (z > 0 and future_ret > 0.005) or (z < 0 and future_ret < -0.005):
                continue_count += 1
            elif (z > 0 and future_ret < -0.005) or (z < 0 and future_ret > 0.005):
                reverse_count += 1

        total = continue_count + reverse_count
        if total > 0:
            continue_rate = continue_count / total * 100
            if continue_rate > 55:
                char = "MOMENTUM"
            elif continue_rate < 45:
                char = "MEAN-REVERT"
            else:
                char = "NEUTRAL"
            print(f"{symbol:<10} {continue_count:<12} {reverse_count:<12} {continue_rate:<9.1f}% {char}")


def main():
    print("Loading data...")
    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df

    analyze_return_autocorrelation(data)
    analyze_momentum_vs_meanreversion(data)
    analyze_zscore_reversal_accuracy(data)
    analyze_low_vol_regime_behavior(data)
    analyze_signal_exit_context(data)

    print("\n" + "=" * 100)
    print("CONCLUSION")
    print("=" * 100)


if __name__ == "__main__":
    main()
