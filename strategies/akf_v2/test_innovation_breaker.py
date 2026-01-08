#!/usr/bin/env python3
"""
V3.2 + Innovation Breaker 테스트

NIS (Normalized Innovation Squared) 기반 선제적 탈출:
- residual = log(price) - log(predicted_trend)
- NIS = residual^2 / variance
- 조건: |residual| > k * resid_std 이면 즉시 청산

v_ratio > 1.0 (변동성 높을 때)만 적용
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from numba import njit

from research.features.kalman import calculate_adaptive_kalman
from strategies.akf_v2.common import SizingConfig
from strategies.akf_v2.common.sizing import calculate_position_sizes
from strategies.akf_v2.backtest_pyramid import run_pyramid_backtest, PyramidBacktestConfig
from strategies.akf_v2.strategy_v32 import load_data, calculate_v32_features, V32_PARAMS

DATA_ROOT = PROJECT_ROOT / "etl/data"
ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "LTCUSDT"]


@njit
def generate_signals_with_innovation_breaker(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    kf_trend: np.ndarray,
    vel_zscore: np.ndarray,
    unc_pct: np.ndarray,
    sigma_hybrid: np.ndarray,
    dynamic_mult: np.ndarray,
    v_ratio: np.ndarray,
    resid_std: np.ndarray,
    entry_z: float = 2.0,
    unc_pct_max: float = 0.5,
    v_ratio_threshold: float = 1.0,
    innovation_mult: float = 4.0,
    warmup: int = 210,
):
    """
    V3.2 + Innovation Breaker

    Exit conditions:
    1. Trailing Stop (고점 기준)
    2. Signal Exit (v_ratio < threshold)
    3. Innovation Breaker (v_ratio >= threshold): |residual| > k * resid_std
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    stop_prices = np.zeros(n, dtype=np.float64)

    position = 0
    highest_since_entry = 0.0
    lowest_since_entry = np.inf
    prev_trail_stop = 0.0

    trail_exits = 0
    signal_exits = 0
    innovation_exits = 0

    for i in range(warmup, n):
        low_uncertainty = unc_pct[i] < unc_pct_max

        # Innovation (residual)
        log_residual = np.log(close[i]) - np.log(kf_trend[i])

        if position == 0:
            # Long Entry
            if vel_zscore[i] > entry_z and low_uncertainty:
                position = 1
                highest_since_entry = high[i]
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_trail_stop = highest_since_entry * np.exp(-dist)
                signals[i] = 1
                stop_prices[i] = prev_trail_stop

            # Short Entry
            elif vel_zscore[i] < -entry_z and low_uncertainty:
                position = -1
                lowest_since_entry = low[i]
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_trail_stop = lowest_since_entry * np.exp(dist)
                signals[i] = -1
                stop_prices[i] = prev_trail_stop

        elif position == 1:  # Long
            # Update highest
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]

            # Trailing stop 계산
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_trail_stop = highest_since_entry * np.exp(-dist)
            trail_stop = max(new_trail_stop, prev_trail_stop)
            prev_trail_stop = trail_stop
            stop_prices[i] = trail_stop

            # Trailing Stop 체크
            if close[i] < trail_stop:
                position = 0
                trail_exits += 1
                continue

            # Innovation Breaker: v_ratio >= threshold 일 때만
            # Long에서 residual이 음수로 크게 튀면 (가격이 trend보다 훨씬 아래)
            if v_ratio[i] >= v_ratio_threshold:
                if log_residual < -innovation_mult * resid_std[i]:
                    position = 0
                    innovation_exits += 1
                    continue

            # Signal Exit: v_ratio < threshold 일 때만
            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] < -entry_z:
                    position = 0
                    signal_exits += 1
                    continue

            signals[i] = 1

        elif position == -1:  # Short
            # Update lowest
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]

            # Trailing stop 계산
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_trail_stop = lowest_since_entry * np.exp(dist)
            trail_stop = min(new_trail_stop, prev_trail_stop)
            prev_trail_stop = trail_stop
            stop_prices[i] = trail_stop

            # Trailing Stop 체크
            if close[i] > trail_stop:
                position = 0
                trail_exits += 1
                continue

            # Innovation Breaker: v_ratio >= threshold 일 때만
            # Short에서 residual이 양수로 크게 튀면 (가격이 trend보다 훨씬 위)
            if v_ratio[i] >= v_ratio_threshold:
                if log_residual > innovation_mult * resid_std[i]:
                    position = 0
                    innovation_exits += 1
                    continue

            # Signal Exit: v_ratio < threshold 일 때만
            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] > entry_z:
                    position = 0
                    signal_exits += 1
                    continue

            signals[i] = -1

    return signals, stop_prices, trail_exits, signal_exits, innovation_exits


def run_backtest(df: pd.DataFrame, innovation_mult: float = 4.0):
    """Run V3.2 + Innovation Breaker backtest"""
    df_kf = calculate_adaptive_kalman(df)
    features = calculate_v32_features(df_kf)

    # Residual std 계산
    close = df_kf["close"].values
    kf_trend = df_kf["kf_trend"].values
    log_residual = np.log(close) - np.log(kf_trend)
    resid_std = pd.Series(log_residual).rolling(window=180, min_periods=30).std().fillna(0.01).values

    signals, stop_prices, trail_exits, signal_exits, innovation_exits = generate_signals_with_innovation_breaker(
        close=close,
        high=df_kf["high"].values,
        low=df_kf["low"].values,
        kf_trend=kf_trend,
        vel_zscore=features["vel_zscore"],
        unc_pct=features["unc_pct"],
        sigma_hybrid=features["sigma_hybrid"],
        dynamic_mult=features["dynamic_mult"],
        v_ratio=features["v_ratio"],
        resid_std=resid_std,
        entry_z=V32_PARAMS["entry_z"],
        unc_pct_max=V32_PARAMS["unc_pct_max"],
        v_ratio_threshold=V32_PARAMS["v_ratio_threshold"],
        innovation_mult=innovation_mult,
        warmup=V32_PARAMS["warmup"],
    )

    df_kf["signal"] = signals
    df_kf["stop_price"] = stop_prices

    sizing_config = SizingConfig(method="fixed", min_size=1.0, max_size=1.0)
    df_kf["position_size"] = calculate_position_sizes(df_kf, sizing_config, entry_signals=signals)
    df_kf["sl_price"] = 0.0

    bt_config = PyramidBacktestConfig(
        initial_capital=100000.0,
        compounding=True,
        fee_rate=0.001,
        slippage_rate=0.0001,
    )

    metrics = run_pyramid_backtest(df_kf, bt_config)
    metrics["trail_exits"] = trail_exits
    metrics["signal_exits"] = signal_exits
    metrics["innovation_exits"] = innovation_exits
    metrics["avg_resid_std"] = np.nanmean(resid_std[210:])

    return metrics


def main():
    print("Loading data...")
    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars")

    # Test different innovation multipliers
    innovation_mults = [2.0, 3.0, 4.0, 5.0, 6.0]

    print("\n" + "=" * 140)
    print("V3.2 + Innovation Breaker 테스트")
    print("v_ratio >= 1.0 일 때: |residual| > k * resid_std 이면 즉시 청산")
    print("=" * 140)

    summary = []

    for innovation_mult in innovation_mults:
        print(f"\n[innovation_mult = {innovation_mult}]")
        print(f"{'Symbol':<10} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'Trades':<8} {'WinRate':<10} {'Trail':<8} {'Signal':<8} {'Innov':<8}")
        print("-" * 100)

        results = []
        for symbol, df in data.items():
            metrics = run_backtest(df, innovation_mult=innovation_mult)
            results.append({
                "symbol": symbol,
                "sharpe": metrics["sharpe_ratio"],
                "pnl": metrics["total_pnl"],
                "mdd": metrics["max_drawdown"],
                "trades": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
                "trail_exits": metrics["trail_exits"],
                "signal_exits": metrics["signal_exits"],
                "innovation_exits": metrics["innovation_exits"],
            })
            print(f"{symbol:<10} {metrics['sharpe_ratio']:<10.2f} {metrics['total_pnl']*100:<11.1f}% "
                  f"{metrics['max_drawdown']*100:<9.1f}% {metrics['total_trades']:<8} "
                  f"{metrics['win_rate']*100:<9.1f}% {metrics['trail_exits']:<8} "
                  f"{metrics['signal_exits']:<8} {metrics['innovation_exits']:<8}")

        print("-" * 100)
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)
        btc = next(r for r in results if r["symbol"] == "BTCUSDT")

        print(f"Average: Sharpe={avg_sharpe:.2f}, PnL={avg_pnl*100:.1f}%, MDD={avg_mdd*100:.1f}%, Positive={positive}/7")
        print(f"BTC: Sharpe={btc['sharpe']:.2f}, PnL={btc['pnl']*100:.1f}%, MDD={btc['mdd']*100:.1f}%")

        summary.append({
            "innovation_mult": innovation_mult,
            "avg_sharpe": avg_sharpe,
            "avg_pnl": avg_pnl,
            "avg_mdd": avg_mdd,
            "positive": positive,
            "btc_sharpe": btc["sharpe"],
            "btc_pnl": btc["pnl"],
            "btc_mdd": btc["mdd"],
        })

    # Summary
    print("\n" + "=" * 140)
    print("Summary 비교")
    print("=" * 140)
    print(f"{'InnovMult':<12} {'AvgSharpe':<12} {'AvgPnL':<12} {'AvgMDD':<12} {'Positive':<10} {'BTC_Sharpe':<12} {'BTC_PnL':<12} {'BTC_MDD':<12}")
    print("-" * 100)
    for s in summary:
        print(f"{s['innovation_mult']:<12} {s['avg_sharpe']:<12.2f} {s['avg_pnl']*100:<11.1f}% {s['avg_mdd']*100:<11.1f}% "
              f"{s['positive']}/7       {s['btc_sharpe']:<12.2f} {s['btc_pnl']*100:<11.1f}% {s['btc_mdd']*100:<11.1f}%")

    print("\n비교 기준 (V3.2 without Innovation Breaker):")
    print("  Avg Sharpe=0.53, AvgMDD=55.3%, Positive=5/7, BTC: Sharpe=1.02, PnL=464.8%, MDD=25.1%")


if __name__ == "__main__":
    main()
