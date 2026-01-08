#!/usr/bin/env python3
"""
V3.2 + Dynamic Innovation Breaker 테스트

Innovation mult를 v_ratio 기반으로 동적 조절:
- dynamic_innov_mult = base_innov_mult / v_ratio
- v_ratio 높을 때 → 더 민감 (mult 낮음)
- v_ratio 낮을 때 → 덜 민감 (mult 높음)
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
def generate_signals_dynamic_innovation(
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
    base_innov_mult: float = 3.0,
    innov_mult_min: float = 1.5,
    innov_mult_max: float = 5.0,
    warmup: int = 210,
):
    """
    V3.2 + Dynamic Innovation Breaker

    dynamic_innov_mult = clip(base_innov_mult / v_ratio, min, max)
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

        # Dynamic innovation mult
        dynamic_innov_mult = base_innov_mult / (v_ratio[i] + 1e-10)
        dynamic_innov_mult = max(innov_mult_min, min(innov_mult_max, dynamic_innov_mult))

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

            # Dynamic Innovation Breaker: v_ratio >= threshold 일 때만
            if v_ratio[i] >= v_ratio_threshold:
                if log_residual < -dynamic_innov_mult * resid_std[i]:
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

            # Dynamic Innovation Breaker: v_ratio >= threshold 일 때만
            if v_ratio[i] >= v_ratio_threshold:
                if log_residual > dynamic_innov_mult * resid_std[i]:
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


def run_backtest(df: pd.DataFrame, base_innov_mult: float = 3.0, innov_mult_min: float = 1.5, innov_mult_max: float = 5.0):
    """Run V3.2 + Dynamic Innovation Breaker backtest"""
    df_kf = calculate_adaptive_kalman(df)
    features = calculate_v32_features(df_kf)

    # Residual std 계산
    close = df_kf["close"].values
    kf_trend = df_kf["kf_trend"].values
    log_residual = np.log(close) - np.log(kf_trend)
    resid_std = pd.Series(log_residual).rolling(window=180, min_periods=30).std().fillna(0.01).values

    signals, stop_prices, trail_exits, signal_exits, innovation_exits = generate_signals_dynamic_innovation(
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
        base_innov_mult=base_innov_mult,
        innov_mult_min=innov_mult_min,
        innov_mult_max=innov_mult_max,
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

    return metrics


def main():
    print("Loading data...")
    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars")

    # Test different base innovation multipliers
    configs = [
        # (base_mult, min, max)
        (2.0, 1.0, 4.0),
        (2.5, 1.5, 4.0),
        (3.0, 1.5, 5.0),
        (3.0, 2.0, 5.0),
        (4.0, 2.0, 6.0),
    ]

    print("\n" + "=" * 150)
    print("V3.2 + Dynamic Innovation Breaker 테스트")
    print("dynamic_innov_mult = clip(base / v_ratio, min, max)")
    print("v_ratio >= 1.0 일 때만 적용")
    print("=" * 150)

    all_results = {}

    for base_mult, mult_min, mult_max in configs:
        config_name = f"base={base_mult}, range=[{mult_min},{mult_max}]"
        print(f"\n[{config_name}]")
        print(f"{'Symbol':<10} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'Trades':<8} {'WinRate':<10} {'Trail':<8} {'Signal':<8} {'Innov':<8}")
        print("-" * 100)

        results = []
        for symbol, df in data.items():
            metrics = run_backtest(df, base_innov_mult=base_mult, innov_mult_min=mult_min, innov_mult_max=mult_max)
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

        all_results[config_name] = results

        print("-" * 100)
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)
        btc = next(r for r in results if r["symbol"] == "BTCUSDT")
        sol = next(r for r in results if r["symbol"] == "SOLUSDT")

        print(f"Average: Sharpe={avg_sharpe:.2f}, PnL={avg_pnl*100:.1f}%, MDD={avg_mdd*100:.1f}%, Positive={positive}/7")
        print(f"BTC: Sharpe={btc['sharpe']:.2f}, PnL={btc['pnl']*100:.1f}%, MDD={btc['mdd']*100:.1f}%")
        print(f"SOL: Sharpe={sol['sharpe']:.2f}, PnL={sol['pnl']*100:.1f}%, MDD={sol['mdd']*100:.1f}%")

    # 비교 테이블
    print("\n" + "=" * 150)
    print("심볼별 상세 비교 (V3.2 기준 vs Dynamic Innovation)")
    print("=" * 150)

    # V3.2 baseline
    v32_baseline = {
        "BTCUSDT": {"pnl": 464.8, "mdd": 25.1},
        "ETHUSDT": {"pnl": 374.5, "mdd": 42.7},
        "XRPUSDT": {"pnl": 128.2, "mdd": 67.8},
        "SOLUSDT": {"pnl": 763.5, "mdd": 67.0},
        "BNBUSDT": {"pnl": 11.8, "mdd": 44.7},
        "DOGEUSDT": {"pnl": -26.8, "mdd": 68.3},
        "LTCUSDT": {"pnl": -41.0, "mdd": 71.3},
    }

    best_config = "base=3.0, range=[2.0,5.0]"  # 비교용
    if best_config in all_results:
        print(f"\n선택 Config: {best_config}")
        print(f"{'Symbol':<10} {'V3.2 PnL':<12} {'Dyn PnL':<12} {'Δ PnL':<12} {'V3.2 MDD':<12} {'Dyn MDD':<12} {'Δ MDD':<12}")
        print("-" * 80)

        for r in all_results[best_config]:
            symbol = r["symbol"]
            base = v32_baseline[symbol]
            delta_pnl = r["pnl"] * 100 - base["pnl"]
            delta_mdd = r["mdd"] * 100 - base["mdd"]
            print(f"{symbol:<10} {base['pnl']:<11.1f}% {r['pnl']*100:<11.1f}% {delta_pnl:+11.1f}% "
                  f"{base['mdd']:<11.1f}% {r['mdd']*100:<11.1f}% {delta_mdd:+11.1f}%")


if __name__ == "__main__":
    main()
