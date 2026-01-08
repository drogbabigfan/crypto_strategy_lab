#!/usr/bin/env python3
"""
NIS 기반 이탈 조건 테스트

NIS (Normalized Innovation Squared) = innovation² / S
- NIS 낮음 = 예측 정확 = 추세 명확 = Signal Exit 신뢰 가능
- NIS 높음 = 예측 부정확 = 불안정 = Trailing Stop에 의존

기존: v_ratio < 1.0 → Signal Exit
제안: NIS < threshold → Signal Exit
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


def calculate_features_with_nis(df_kf: pd.DataFrame, params: dict = None):
    """NIS를 포함한 feature 계산"""
    if params is None:
        params = V33_PARAMS

    close = df_kf["close"].values
    kf_trend = df_kf["kf_trend"].values
    kf_uncertainty = df_kf["kf_uncertainty"].values
    velocity = df_kf["kf_velocity"].values

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

    # 6. NIS (Normalized Innovation Squared)
    innovation = close - kf_trend
    nis = (innovation ** 2) / (kf_uncertainty + 1e-10)

    # Smooth NIS
    nis_smooth = pd.Series(nis).rolling(window=10, min_periods=1).mean().values

    # NIS percentile (rolling)
    nis_pct = (
        pd.Series(nis_smooth)
        .rolling(window=210, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )

    return {
        "sigma_hybrid": sigma_hybrid,
        "v_ratio": v_ratio,
        "dynamic_mult": dynamic_mult,
        "vel_zscore": vel_zscore,
        "unc_pct": unc_pct,
        "resid_std": resid_std,
        "kf_trend": kf_trend,
        "nis": nis_smooth,
        "nis_pct": nis_pct,
    }


def analyze_nis_distribution(data: dict):
    """NIS 분포 분석"""
    print("\n" + "=" * 100)
    print("1. NIS 분포 분석")
    print("=" * 100)
    print(f"{'Symbol':<10} {'Mean':<10} {'Median':<10} {'Std':<10} {'<0.5 pct':<12} {'<1.0 pct':<12} {'>2.0 pct':<12}")
    print("-" * 80)

    for symbol, df in data.items():
        df_kf = calculate_adaptive_kalman(df)
        features = calculate_features_with_nis(df_kf)
        nis = features["nis"][210:]

        print(f"{symbol:<10} {np.mean(nis):<10.2f} {np.median(nis):<10.2f} {np.std(nis):<10.2f} "
              f"{np.mean(nis < 0.5)*100:<11.1f}% {np.mean(nis < 1.0)*100:<11.1f}% {np.mean(nis > 2.0)*100:<11.1f}%")


def analyze_nis_vs_vratio_correlation(data: dict):
    """NIS와 v_ratio 상관관계"""
    print("\n" + "=" * 100)
    print("2. NIS vs V-Ratio 상관관계")
    print("=" * 100)
    print(f"{'Symbol':<10} {'Correlation':<15} {'NIS<1 & VR<1':<15} {'NIS<1 & VR>=1':<15}")
    print("-" * 60)

    for symbol, df in data.items():
        df_kf = calculate_adaptive_kalman(df)
        features = calculate_features_with_nis(df_kf)
        nis = features["nis"][210:]
        v_ratio = features["v_ratio"][210:]

        corr = np.corrcoef(nis, v_ratio)[0, 1]
        both_low = np.mean((nis < 1.0) & (v_ratio < 1.0)) * 100
        nis_low_vr_high = np.mean((nis < 1.0) & (v_ratio >= 1.0)) * 100

        print(f"{symbol:<10} {corr:<15.3f} {both_low:<14.1f}% {nis_low_vr_high:<14.1f}%")


@njit
def generate_signals_nis_based(
    close, high, low, kf_trend, vel_zscore, unc_pct, sigma_hybrid, dynamic_mult,
    v_ratio, resid_std, nis_pct,
    entry_z, unc_pct_max, nis_threshold, innov_base_mult, innov_mult_min, innov_mult_max,
    hard_stop_mult, risk_target, conf_lambda, size_min, size_max, warmup
):
    """NIS 기반 Signal Exit"""
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
            # NIS 기반 Signal Exit (NIS가 낮을 때 = 예측 정확할 때)
            if nis_pct[i] < nis_threshold:
                if vel_zscore[i] < -entry_z:
                    position = 0
                    signal_exits += 1
                    continue
            # Innovation Breaker (NIS가 높을 때)
            if nis_pct[i] >= nis_threshold:
                if log_resid < -innov_mult * resid_std[i]:
                    position = 0
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
            if nis_pct[i] < nis_threshold:
                if vel_zscore[i] > entry_z:
                    position = 0
                    signal_exits += 1
                    continue
            if nis_pct[i] >= nis_threshold:
                if log_resid > innov_mult * resid_std[i]:
                    position = 0
                    continue

            signals[i] = -1
            position_sizes[i] = current_size

    return signals, position_sizes, signal_exits, trail_exits


def run_backtest_nis(df: pd.DataFrame, nis_threshold: float = 0.5):
    """NIS 기반 백테스트"""
    params = V33_PARAMS
    df_kf = calculate_adaptive_kalman(df)
    features = calculate_features_with_nis(df_kf, params)

    signals, position_sizes, signal_exits, trail_exits = generate_signals_nis_based(
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
        nis_pct=features["nis_pct"],
        entry_z=params["entry_z"],
        unc_pct_max=params["unc_pct_max"],
        nis_threshold=nis_threshold,
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

    return metrics


def main():
    print("Loading data...")
    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df

    analyze_nis_distribution(data)
    analyze_nis_vs_vratio_correlation(data)

    # NIS threshold 테스트
    print("\n" + "=" * 120)
    print("3. NIS 기반 Signal Exit 테스트")
    print("   NIS_pct < threshold → Signal Exit")
    print("   NIS_pct >= threshold → Innovation Breaker")
    print("=" * 120)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    for threshold in thresholds:
        print(f"\n[NIS Threshold = {threshold}]")
        print(f"{'Symbol':<10} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'SigExit':<10} {'TrailExit':<10}")
        print("-" * 70)

        results = []
        for symbol, df in data.items():
            m = run_backtest_nis(df, nis_threshold=threshold)
            results.append({
                "symbol": symbol,
                "sharpe": m["sharpe_ratio"],
                "pnl": m["total_pnl"],
                "mdd": m["max_drawdown"],
                "signal_exits": m["signal_exits"],
                "trail_exits": m["trail_exits"],
            })
            print(f"{symbol:<10} {m['sharpe_ratio']:<10.2f} {m['total_pnl']*100:<11.1f}% "
                  f"{m['max_drawdown']*100:<9.1f}% {m['signal_exits']:<10} {m['trail_exits']:<10}")

        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)
        btc = next(r for r in results if r["symbol"] == "BTCUSDT")

        print("-" * 70)
        print(f"Average: Sharpe={avg_sharpe:.2f}, PnL={avg_pnl*100:.1f}%, MDD={avg_mdd*100:.1f}%, Positive={positive}/7")
        print(f"BTC: Sharpe={btc['sharpe']:.2f}, PnL={btc['pnl']*100:.1f}%")

    # V-ratio 기반과 비교
    print("\n" + "=" * 120)
    print("4. V-Ratio 기반 vs NIS 기반 비교")
    print("=" * 120)
    print(f"{'Method':<25} {'AvgSharpe':<12} {'AvgPnL':<12} {'AvgMDD':<12} {'Positive':<10} {'BTC_PnL':<12}")
    print("-" * 90)

    # Import original for comparison
    from strategies.akf_v2.strategy_v33_final import run_backtest as run_original

    # Original V-ratio based
    orig_results = []
    for symbol, df in data.items():
        m, _ = run_original(df)
        orig_results.append({"symbol": symbol, "sharpe": m["sharpe_ratio"], "pnl": m["total_pnl"], "mdd": m["max_drawdown"]})

    avg_sharpe = np.mean([r["sharpe"] for r in orig_results])
    avg_pnl = np.mean([r["pnl"] for r in orig_results])
    avg_mdd = np.mean([r["mdd"] for r in orig_results])
    positive = sum(1 for r in orig_results if r["pnl"] > 0)
    btc = next(r for r in orig_results if r["symbol"] == "BTCUSDT")
    print(f"{'V-Ratio (v<1.0)':<25} {avg_sharpe:<12.2f} {avg_pnl*100:<11.1f}% {avg_mdd*100:<11.1f}% {positive}/7       {btc['pnl']*100:<11.1f}%")

    # Best NIS based (0.5)
    nis_results = []
    for symbol, df in data.items():
        m = run_backtest_nis(df, nis_threshold=0.5)
        nis_results.append({"symbol": symbol, "sharpe": m["sharpe_ratio"], "pnl": m["total_pnl"], "mdd": m["max_drawdown"]})

    avg_sharpe = np.mean([r["sharpe"] for r in nis_results])
    avg_pnl = np.mean([r["pnl"] for r in nis_results])
    avg_mdd = np.mean([r["mdd"] for r in nis_results])
    positive = sum(1 for r in nis_results if r["pnl"] > 0)
    btc = next(r for r in nis_results if r["symbol"] == "BTCUSDT")
    print(f"{'NIS (pct<0.5)':<25} {avg_sharpe:<12.2f} {avg_pnl*100:<11.1f}% {avg_mdd*100:<11.1f}% {positive}/7       {btc['pnl']*100:<11.1f}%")


if __name__ == "__main__":
    main()
