#!/usr/bin/env python3
"""
Duration-based Intensity Filter 테스트

1. Bar Duration (Δt_i): 현재 바와 이전 바의 시간 차이
2. Baseline Duration (μ_t): 최근 N개 바의 Duration SMA
3. Flow Intensity (I_t): μ_t / (Δt_i + ε)
   - I > 1: High Velocity (바가 빨리 생성)
   - I < 1: Low Velocity (바가 천천히 생성)
4. Exit 조건: I_t < threshold 일 때만 Signal Exit 실행
   - Flash State (I_t >= threshold)에서는 Signal Exit 보류
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from numba import njit

from research.features.kalman import calculate_adaptive_kalman
from strategies.akf_v2.backtest_pyramid import run_pyramid_backtest, PyramidBacktestConfig
from strategies.akf_v2.strategy_v33_final import load_data, V33_PARAMS, calculate_position_size, calculate_features

DATA_ROOT = PROJECT_ROOT / "etl/data"
ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "LTCUSDT"]


def calculate_intensity(df: pd.DataFrame, baseline_window: int = 20):
    """Flow Intensity 계산"""
    # Bar Duration (log_duration에서 복원)
    duration = np.exp(df["log_duration"].values)

    # Baseline Duration (SMA)
    baseline_duration = pd.Series(duration).rolling(window=baseline_window, min_periods=5).mean().values

    # Flow Intensity = baseline / (current + epsilon)
    epsilon = 1e-6
    intensity = baseline_duration / (duration + epsilon)

    return intensity, duration, baseline_duration


def analyze_intensity_distribution(data: dict, baseline_window: int = 20):
    """Intensity 분포 분석"""
    print("\n" + "=" * 100)
    print(f"1. Flow Intensity 분포 분석 (baseline_window={baseline_window})")
    print("   I > 1: High Velocity (빠름), I < 1: Low Velocity (느림)")
    print("=" * 100)
    print(f"{'Symbol':<10} {'Mean':<10} {'Median':<10} {'Std':<10} {'<1.0':<10} {'>2.0':<10} {'>3.0':<10} {'>5.0':<10}")
    print("-" * 100)

    for symbol, df in data.items():
        intensity, _, _ = calculate_intensity(df, baseline_window)
        intensity = intensity[210:]  # warmup 제외

        print(f"{symbol:<10} {np.nanmean(intensity):<10.2f} {np.nanmedian(intensity):<10.2f} "
              f"{np.nanstd(intensity):<10.2f} {np.nanmean(intensity < 1.0)*100:<9.1f}% "
              f"{np.nanmean(intensity > 2.0)*100:<9.1f}% {np.nanmean(intensity > 3.0)*100:<9.1f}% "
              f"{np.nanmean(intensity > 5.0)*100:<9.1f}%")


def analyze_intensity_vs_reversal(data: dict, baseline_window: int = 20):
    """Intensity 구간별 추세 반전 정확도"""
    print("\n" + "=" * 100)
    print("2. Intensity 구간별 반전 신호 정확도")
    print("   Low Intensity에서 Signal Exit이 더 정확한지 확인")
    print("=" * 100)
    print(f"{'Symbol':<10} {'I<1 (느림)':<20} {'I 1-3':<20} {'I>3 (Flash)':<20}")
    print(f"{'':10} {'반전률':<20} {'반전률':<20} {'반전률':<20}")
    print("-" * 80)

    entry_z = V33_PARAMS["entry_z"]

    for symbol, df in data.items():
        df_kf = calculate_adaptive_kalman(df)
        features = calculate_features(df_kf)
        intensity, _, _ = calculate_intensity(df, baseline_window)

        vel_zscore = features["vel_zscore"]
        close = df_kf["close"].values

        results = {
            "low": {"correct": 0, "wrong": 0},
            "mid": {"correct": 0, "wrong": 0},
            "high": {"correct": 0, "wrong": 0},
        }

        for i in range(210, len(vel_zscore) - 10):
            if np.isnan(intensity[i]):
                continue

            # vel_zscore 반전 신호
            if vel_zscore[i-1] > 0 and vel_zscore[i] < -entry_z:
                future_ret = (close[i+10] - close[i]) / close[i]
                bucket = "low" if intensity[i] < 1.0 else ("mid" if intensity[i] < 3.0 else "high")
                if future_ret < 0:  # 실제 하락 = 정확
                    results[bucket]["correct"] += 1
                else:
                    results[bucket]["wrong"] += 1

            elif vel_zscore[i-1] < 0 and vel_zscore[i] > entry_z:
                future_ret = (close[i+10] - close[i]) / close[i]
                bucket = "low" if intensity[i] < 1.0 else ("mid" if intensity[i] < 3.0 else "high")
                if future_ret > 0:  # 실제 상승 = 정확
                    results[bucket]["correct"] += 1
                else:
                    results[bucket]["wrong"] += 1

        def calc_rate(bucket):
            total = results[bucket]["correct"] + results[bucket]["wrong"]
            if total == 0:
                return "-"
            rate = results[bucket]["correct"] / total * 100
            return f"{rate:.0f}% ({total})"

        print(f"{symbol:<10} {calc_rate('low'):<20} {calc_rate('mid'):<20} {calc_rate('high'):<20}")


@njit
def generate_signals_with_intensity_filter(
    close, high, low, kf_trend, vel_zscore, unc_pct, sigma_hybrid, dynamic_mult,
    v_ratio, resid_std, intensity,
    entry_z, unc_pct_max, v_ratio_threshold, intensity_threshold,
    innov_base_mult, innov_mult_min, innov_mult_max,
    hard_stop_mult, risk_target, conf_lambda, size_min, size_max, warmup
):
    """Intensity 필터 적용 Signal Exit"""
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    position_sizes = np.zeros(n, dtype=np.float64)

    position = 0
    entry_price = 0.0
    current_size = 0.0
    highest = 0.0
    lowest = np.inf
    prev_stop = 0.0
    hard_stop = 0.0

    signal_exits = 0
    blocked_exits = 0  # Flash State로 인해 차단된 exit
    trail_exits = 0
    innov_exits = 0

    for i in range(warmup, n):
        innov_mult = innov_base_mult / (v_ratio[i] + 1e-10)
        innov_mult = max(innov_mult_min, min(innov_mult_max, innov_mult))
        log_resid = np.log(close[i]) - np.log(kf_trend[i])

        if position == 0:
            if vel_zscore[i] > entry_z and unc_pct[i] < unc_pct_max:
                position = 1
                entry_price = close[i]
                highest = high[i]
                current_size = calculate_position_size(
                    sigma_hybrid[i], unc_pct[i], risk_target, hard_stop_mult, conf_lambda, size_min, size_max
                )
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = highest * np.exp(-dist)
                hard_stop = entry_price * np.exp(-hard_stop_mult * sigma_hybrid[i])
                signals[i] = 1
                position_sizes[i] = current_size

            elif vel_zscore[i] < -entry_z and unc_pct[i] < unc_pct_max:
                position = -1
                entry_price = close[i]
                lowest = low[i]
                current_size = calculate_position_size(
                    sigma_hybrid[i], unc_pct[i], risk_target, hard_stop_mult, conf_lambda, size_min, size_max
                )
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = lowest * np.exp(dist)
                hard_stop = entry_price * np.exp(hard_stop_mult * sigma_hybrid[i])
                signals[i] = -1
                position_sizes[i] = current_size

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
                trail_exits += 1
                continue

            # Signal Exit with Intensity Filter
            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] < -entry_z:
                    # Flash State 체크
                    if intensity[i] < intensity_threshold:
                        position = 0
                        signal_exits += 1
                        continue
                    else:
                        blocked_exits += 1
                        # Flash State → Signal Exit 보류

            # Innovation Breaker (v_ratio >= threshold)
            if v_ratio[i] >= v_ratio_threshold:
                if log_resid < -innov_mult * resid_std[i]:
                    position = 0
                    innov_exits += 1
                    continue

            signals[i] = 1
            position_sizes[i] = current_size

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
                trail_exits += 1
                continue

            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] > entry_z:
                    if intensity[i] < intensity_threshold:
                        position = 0
                        signal_exits += 1
                        continue
                    else:
                        blocked_exits += 1

            if v_ratio[i] >= v_ratio_threshold:
                if log_resid > innov_mult * resid_std[i]:
                    position = 0
                    innov_exits += 1
                    continue

            signals[i] = -1
            position_sizes[i] = current_size

    return signals, position_sizes, signal_exits, blocked_exits, trail_exits, innov_exits


def run_backtest_intensity(df: pd.DataFrame, intensity_threshold: float = 3.0, baseline_window: int = 20):
    """Intensity 필터 백테스트"""
    params = V33_PARAMS
    df_kf = calculate_adaptive_kalman(df)
    features = calculate_features(df_kf, params)
    intensity, _, _ = calculate_intensity(df, baseline_window)

    signals, position_sizes, signal_exits, blocked_exits, trail_exits, innov_exits = generate_signals_with_intensity_filter(
        close=df_kf["close"].values,
        high=df_kf["high"].values,
        low=df_kf["low"].values,
        kf_trend=features["kf_trend"],
        vel_zscore=features["vel_zscore"],
        unc_pct=features["unc_pct"],
        sigma_hybrid=features["sigma_hybrid"],
        dynamic_mult=features["dynamic_mult"],
        v_ratio=features["v_ratio"],
        resid_std=features["resid_std"],
        intensity=intensity,
        entry_z=params["entry_z"],
        unc_pct_max=params["unc_pct_max"],
        v_ratio_threshold=params["v_ratio_threshold"],
        intensity_threshold=intensity_threshold,
        innov_base_mult=params["innov_base_mult"],
        innov_mult_min=params["innov_mult_min"],
        innov_mult_max=params["innov_mult_max"],
        hard_stop_mult=params["hard_stop_mult"],
        risk_target=params["risk_target"],
        conf_lambda=params["conf_lambda"],
        size_min=params["size_min"],
        size_max=params["size_max"],
        warmup=params["warmup"],
    )

    df_kf["signal"] = signals
    df_kf["position_size"] = position_sizes
    df_kf["sl_price"] = 0.0

    bt_config = PyramidBacktestConfig(
        initial_capital=100000.0,
        compounding=True,
        fee_rate=0.001,
        slippage_rate=0.0001,
    )

    metrics = run_pyramid_backtest(df_kf, bt_config)
    metrics["signal_exits"] = signal_exits
    metrics["blocked_exits"] = blocked_exits
    metrics["trail_exits"] = trail_exits
    metrics["innov_exits"] = innov_exits

    return metrics


def main():
    print("Loading data...")
    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df

    # Intensity 분포 분석
    analyze_intensity_distribution(data, baseline_window=20)

    # Intensity vs 반전 정확도
    analyze_intensity_vs_reversal(data, baseline_window=20)

    # Intensity threshold 테스트
    print("\n" + "=" * 140)
    print("3. Intensity Filter 테스트")
    print("   I < threshold → Signal Exit 실행")
    print("   I >= threshold (Flash State) → Signal Exit 보류")
    print("=" * 140)

    thresholds = [2.0, 3.0, 4.0, 5.0, 10.0, 999.0]  # 999 = 필터 없음 (baseline)

    all_results = {}

    for threshold in thresholds:
        label = "No Filter" if threshold >= 999 else f"I<{threshold}"
        print(f"\n[{label}]")
        print(f"{'Symbol':<10} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'SigExit':<10} {'Blocked':<10} {'Trail':<10}")
        print("-" * 80)

        results = []
        for symbol, df in data.items():
            m = run_backtest_intensity(df, intensity_threshold=threshold)
            results.append({
                "symbol": symbol,
                "sharpe": m["sharpe_ratio"],
                "pnl": m["total_pnl"],
                "mdd": m["max_drawdown"],
                "signal_exits": m["signal_exits"],
                "blocked_exits": m["blocked_exits"],
                "trail_exits": m["trail_exits"],
            })
            print(f"{symbol:<10} {m['sharpe_ratio']:<10.2f} {m['total_pnl']*100:<11.1f}% "
                  f"{m['max_drawdown']*100:<9.1f}% {m['signal_exits']:<10} {m['blocked_exits']:<10} {m['trail_exits']:<10}")

        all_results[label] = results

        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)
        btc = next(r for r in results if r["symbol"] == "BTCUSDT")
        total_blocked = sum(r["blocked_exits"] for r in results)

        print("-" * 80)
        print(f"Average: Sharpe={avg_sharpe:.2f}, PnL={avg_pnl*100:.1f}%, MDD={avg_mdd*100:.1f}%, "
              f"Positive={positive}/7, Blocked={total_blocked}")
        print(f"BTC: Sharpe={btc['sharpe']:.2f}, PnL={btc['pnl']*100:.1f}%")

    # 비교 테이블
    print("\n" + "=" * 140)
    print("4. 결과 비교")
    print("=" * 140)
    print(f"{'Filter':<15} {'AvgSharpe':<12} {'AvgPnL':<12} {'AvgMDD':<12} {'Positive':<10} {'BTC_PnL':<12} {'TotalBlocked':<12}")
    print("-" * 100)

    for label, results in all_results.items():
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)
        btc = next(r for r in results if r["symbol"] == "BTCUSDT")
        total_blocked = sum(r["blocked_exits"] for r in results)

        print(f"{label:<15} {avg_sharpe:<12.2f} {avg_pnl*100:<11.1f}% {avg_mdd*100:<11.1f}% "
              f"{positive}/7       {btc['pnl']*100:<11.1f}% {total_blocked:<12}")


if __name__ == "__main__":
    main()
