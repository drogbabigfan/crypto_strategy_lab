#!/usr/bin/env python3
"""
Stress Index (Rolling Mean NIS) 기반 이탈 조건 테스트

NIS = (innovation / sqrt(S))² = kf_std_innovation²
Stress = Rolling Mean of NIS over N bars

- Stress ≈ 1.0: 모델 예측 정상 → Signal Exit 신뢰 가능
- Stress >> 1.0: 모델 예측 빗나감 → Trailing Stop에 의존
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
from strategies.akf_v2.strategy_v33_final import load_data, V33_PARAMS, calculate_position_size

DATA_ROOT = PROJECT_ROOT / "etl/data"
ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "LTCUSDT"]


def calculate_features_with_stress(df_kf: pd.DataFrame, stress_window: int = 20, params: dict = None):
    """Stress Index를 포함한 feature 계산"""
    if params is None:
        params = V33_PARAMS

    close = df_kf["close"].values
    kf_trend = df_kf["kf_trend"].values
    kf_uncertainty = df_kf["kf_uncertainty"].values
    velocity = df_kf["kf_velocity"].values
    std_innovation = df_kf["kf_std_innovation"].values

    # 1. Hybrid Volatility
    log_returns = np.log(close[1:] / close[:-1])
    log_returns = np.concatenate([[0], log_returns])
    rv = pd.Series(log_returns).rolling(window=42, min_periods=10).std().values
    sqrt_unc = np.sqrt(kf_uncertainty)
    sigma_hybrid = np.maximum(sqrt_unc, rv)

    # 2. V-Ratio & Dynamic Mult
    baseline = pd.Series(sigma_hybrid).rolling(window=120, min_periods=30).mean().values
    v_ratio = sigma_hybrid / (baseline + 1e-10)
    dynamic_mult = np.clip(5.0 / (v_ratio + 1e-10), 1.25, 7.5)

    # 3. Velocity Z-score
    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=42, min_periods=5).mean()
    vel_std = vel_series.rolling(window=42, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

    # 4. Uncertainty Percentile
    unc_pct = (
        pd.Series(kf_uncertainty)
        .rolling(window=210, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )

    # 5. Residual Std
    log_resid = np.log(close) - np.log(kf_trend)
    resid_std = pd.Series(log_resid).rolling(window=180, min_periods=30).std().fillna(0.01).values

    # 6. NIS & Stress Index (Rolling Mean NIS)
    nis = std_innovation ** 2  # NIS = (innovation / sqrt(S))²
    stress = pd.Series(nis).rolling(window=stress_window, min_periods=5).mean().values

    return {
        "sigma_hybrid": sigma_hybrid,
        "v_ratio": v_ratio,
        "dynamic_mult": dynamic_mult,
        "vel_zscore": vel_zscore,
        "unc_pct": unc_pct,
        "resid_std": resid_std,
        "kf_trend": kf_trend,
        "nis": nis,
        "stress": stress,
    }


def analyze_stress_distribution(data: dict, stress_window: int = 20):
    """Stress 분포 분석"""
    print("\n" + "=" * 100)
    print(f"1. Stress Index 분포 분석 (window={stress_window})")
    print("   Stress ≈ 1.0: 정상, Stress >> 1.0: 이상")
    print("=" * 100)
    print(f"{'Symbol':<10} {'Mean':<10} {'Median':<10} {'Std':<10} {'<1.0':<10} {'<1.5':<10} {'>2.0':<10} {'>3.0':<10}")
    print("-" * 90)

    for symbol, df in data.items():
        df_kf = calculate_adaptive_kalman(df)
        features = calculate_features_with_stress(df_kf, stress_window)
        stress = features["stress"][210:]

        print(f"{symbol:<10} {np.nanmean(stress):<10.2f} {np.nanmedian(stress):<10.2f} {np.nanstd(stress):<10.2f} "
              f"{np.nanmean(stress < 1.0)*100:<9.1f}% {np.nanmean(stress < 1.5)*100:<9.1f}% "
              f"{np.nanmean(stress > 2.0)*100:<9.1f}% {np.nanmean(stress > 3.0)*100:<9.1f}%")


def analyze_stress_vs_trend_reversal(data: dict, stress_window: int = 20):
    """Stress 구간별 추세 반전 확률"""
    print("\n" + "=" * 100)
    print("2. Stress 구간별 추세 반전 분석")
    print("   Stress 낮을 때 Signal Exit이 더 정확한지 확인")
    print("=" * 100)
    print(f"{'Symbol':<10} {'Stress<1':<20} {'Stress 1-2':<20} {'Stress>2':<20}")
    print(f"{'':10} {'반전률 / 지속률':<20} {'반전률 / 지속률':<20} {'반전률 / 지속률':<20}")
    print("-" * 80)

    entry_z = V33_PARAMS["entry_z"]

    for symbol, df in data.items():
        df_kf = calculate_adaptive_kalman(df)
        features = calculate_features_with_stress(df_kf, stress_window)

        vel_zscore = features["vel_zscore"]
        stress = features["stress"]
        close = df_kf["close"].values

        results = {
            "low": {"reverse": 0, "continue": 0},
            "mid": {"reverse": 0, "continue": 0},
            "high": {"reverse": 0, "continue": 0},
        }

        for i in range(210, len(vel_zscore) - 10):
            if np.isnan(stress[i]):
                continue

            # vel_zscore 반전 신호
            if vel_zscore[i-1] > 0 and vel_zscore[i] < -entry_z:
                future_ret = (close[i+10] - close[i]) / close[i]
                bucket = "low" if stress[i] < 1.0 else ("mid" if stress[i] < 2.0 else "high")
                if future_ret < 0:
                    results[bucket]["reverse"] += 1
                else:
                    results[bucket]["continue"] += 1

            elif vel_zscore[i-1] < 0 and vel_zscore[i] > entry_z:
                future_ret = (close[i+10] - close[i]) / close[i]
                bucket = "low" if stress[i] < 1.0 else ("mid" if stress[i] < 2.0 else "high")
                if future_ret > 0:
                    results[bucket]["reverse"] += 1
                else:
                    results[bucket]["continue"] += 1

        def calc_rate(bucket):
            total = results[bucket]["reverse"] + results[bucket]["continue"]
            if total == 0:
                return "-"
            rev_rate = results[bucket]["reverse"] / total * 100
            return f"{rev_rate:.0f}% / {100-rev_rate:.0f}%"

        print(f"{symbol:<10} {calc_rate('low'):<20} {calc_rate('mid'):<20} {calc_rate('high'):<20}")


@njit
def generate_signals_stress_based(
    close, high, low, kf_trend, vel_zscore, unc_pct, sigma_hybrid, dynamic_mult,
    v_ratio, resid_std, stress,
    entry_z, unc_pct_max, stress_threshold, innov_base_mult, innov_mult_min, innov_mult_max,
    hard_stop_mult, risk_target, conf_lambda, size_min, size_max, warmup
):
    """Stress 기반 Signal Exit"""
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
            # Stress 기반 분기
            if stress[i] < stress_threshold:
                # Stress 낮음 = 모델 정상 = Signal Exit 신뢰
                if vel_zscore[i] < -entry_z:
                    position = 0
                    signal_exits += 1
                    continue
            else:
                # Stress 높음 = 모델 불안정 = Innovation Breaker
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
            if stress[i] < stress_threshold:
                if vel_zscore[i] > entry_z:
                    position = 0
                    signal_exits += 1
                    continue
            else:
                if log_resid > innov_mult * resid_std[i]:
                    position = 0
                    innov_exits += 1
                    continue

            signals[i] = -1
            position_sizes[i] = current_size

    return signals, position_sizes, signal_exits, trail_exits, innov_exits


def run_backtest_stress(df: pd.DataFrame, stress_threshold: float = 1.5, stress_window: int = 20):
    """Stress 기반 백테스트"""
    params = V33_PARAMS
    df_kf = calculate_adaptive_kalman(df)
    features = calculate_features_with_stress(df_kf, stress_window, params)

    signals, position_sizes, signal_exits, trail_exits, innov_exits = generate_signals_stress_based(
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
        stress=features["stress"],
        entry_z=params["entry_z"],
        unc_pct_max=params["unc_pct_max"],
        stress_threshold=stress_threshold,
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

    # Stress 분포 분석
    analyze_stress_distribution(data, stress_window=20)

    # Stress vs 추세 반전 분석
    analyze_stress_vs_trend_reversal(data, stress_window=20)

    # Stress threshold 테스트
    print("\n" + "=" * 130)
    print("3. Stress 기반 Signal Exit 테스트")
    print("   Stress < threshold → Signal Exit (모델 신뢰)")
    print("   Stress >= threshold → Innovation Breaker (모델 불신)")
    print("=" * 130)

    thresholds = [1.0, 1.25, 1.5, 2.0, 2.5]

    all_results = {}

    for threshold in thresholds:
        print(f"\n[Stress Threshold = {threshold}]")
        print(f"{'Symbol':<10} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'SigExit':<10} {'InnovExit':<10} {'TrailExit':<10}")
        print("-" * 80)

        results = []
        for symbol, df in data.items():
            m = run_backtest_stress(df, stress_threshold=threshold)
            results.append({
                "symbol": symbol,
                "sharpe": m["sharpe_ratio"],
                "pnl": m["total_pnl"],
                "mdd": m["max_drawdown"],
                "signal_exits": m["signal_exits"],
                "innov_exits": m["innov_exits"],
                "trail_exits": m["trail_exits"],
            })
            print(f"{symbol:<10} {m['sharpe_ratio']:<10.2f} {m['total_pnl']*100:<11.1f}% "
                  f"{m['max_drawdown']*100:<9.1f}% {m['signal_exits']:<10} {m['innov_exits']:<10} {m['trail_exits']:<10}")

        all_results[threshold] = results

        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)
        btc = next(r for r in results if r["symbol"] == "BTCUSDT")

        print("-" * 80)
        print(f"Average: Sharpe={avg_sharpe:.2f}, PnL={avg_pnl*100:.1f}%, MDD={avg_mdd*100:.1f}%, Positive={positive}/7")
        print(f"BTC: Sharpe={btc['sharpe']:.2f}, PnL={btc['pnl']*100:.1f}%")

    # 비교 테이블
    print("\n" + "=" * 130)
    print("4. V-Ratio vs Stress 비교")
    print("=" * 130)
    print(f"{'Method':<30} {'AvgSharpe':<12} {'AvgPnL':<12} {'AvgMDD':<12} {'Positive':<10} {'BTC_PnL':<12}")
    print("-" * 100)

    # Original V-ratio based
    from strategies.akf_v2.strategy_v33_final import run_backtest as run_original
    orig_results = []
    for symbol, df in data.items():
        m, _ = run_original(df)
        orig_results.append({"symbol": symbol, "sharpe": m["sharpe_ratio"], "pnl": m["total_pnl"], "mdd": m["max_drawdown"]})

    avg_sharpe = np.mean([r["sharpe"] for r in orig_results])
    avg_pnl = np.mean([r["pnl"] for r in orig_results])
    avg_mdd = np.mean([r["mdd"] for r in orig_results])
    positive = sum(1 for r in orig_results if r["pnl"] > 0)
    btc = next(r for r in orig_results if r["symbol"] == "BTCUSDT")
    print(f"{'V-Ratio (v<1.0)':<30} {avg_sharpe:<12.2f} {avg_pnl*100:<11.1f}% {avg_mdd*100:<11.1f}% {positive}/7       {btc['pnl']*100:<11.1f}%")

    for threshold, results in all_results.items():
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)
        btc = next(r for r in results if r["symbol"] == "BTCUSDT")
        print(f"{f'Stress (<{threshold})':<30} {avg_sharpe:<12.2f} {avg_pnl*100:<11.1f}% {avg_mdd*100:<11.1f}% {positive}/7       {btc['pnl']*100:<11.1f}%")


if __name__ == "__main__":
    main()
