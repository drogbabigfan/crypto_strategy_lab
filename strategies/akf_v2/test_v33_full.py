#!/usr/bin/env python3
"""
V3.3 Full Test: Dynamic Innovation + Hard Stop

조합:
1. Trailing Stop (기존)
2. Signal Exit (v_ratio < 1.0)
3. Dynamic Innovation Breaker (v_ratio >= 1.0)
4. Hard Stop (진입가 기준)
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
def generate_signals_v33(
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
    # Dynamic Innovation params
    base_innov_mult: float = 2.5,
    innov_mult_min: float = 1.5,
    innov_mult_max: float = 4.0,
    # Hard Stop params
    hard_stop_mult: float = 3.0,
    warmup: int = 210,
):
    """
    V3.3 Signal Generation

    Exit conditions:
    1. Hard Stop (진입가 기준) - 최우선
    2. Trailing Stop (고점 기준)
    3. Dynamic Innovation Breaker (v_ratio >= threshold)
    4. Signal Exit (v_ratio < threshold)
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    stop_prices = np.zeros(n, dtype=np.float64)

    position = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = np.inf
    prev_trail_stop = 0.0
    hard_stop = 0.0

    trail_exits = 0
    signal_exits = 0
    innovation_exits = 0
    hard_exits = 0

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
                entry_price = close[i]
                highest_since_entry = high[i]

                # Trailing stop 초기화
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_trail_stop = highest_since_entry * np.exp(-dist)

                # Hard stop 설정
                hard_stop = entry_price * np.exp(-hard_stop_mult * sigma_hybrid[i])

                signals[i] = 1
                stop_prices[i] = max(prev_trail_stop, hard_stop)

            # Short Entry
            elif vel_zscore[i] < -entry_z and low_uncertainty:
                position = -1
                entry_price = close[i]
                lowest_since_entry = low[i]

                # Trailing stop 초기화
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_trail_stop = lowest_since_entry * np.exp(dist)

                # Hard stop 설정
                hard_stop = entry_price * np.exp(hard_stop_mult * sigma_hybrid[i])

                signals[i] = -1
                stop_prices[i] = min(prev_trail_stop, hard_stop)

        elif position == 1:  # Long
            # Update highest
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]

            # Trailing stop 계산
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_trail_stop = highest_since_entry * np.exp(-dist)
            trail_stop = max(new_trail_stop, prev_trail_stop)
            prev_trail_stop = trail_stop

            effective_stop = max(trail_stop, hard_stop)
            stop_prices[i] = effective_stop

            # 1. Hard Stop 체크 (최우선)
            if close[i] < hard_stop:
                position = 0
                hard_exits += 1
                continue

            # 2. Trailing Stop 체크
            if close[i] < trail_stop:
                position = 0
                trail_exits += 1
                continue

            # 3. Dynamic Innovation Breaker (v_ratio >= threshold)
            if v_ratio[i] >= v_ratio_threshold:
                if log_residual < -dynamic_innov_mult * resid_std[i]:
                    position = 0
                    innovation_exits += 1
                    continue

            # 4. Signal Exit (v_ratio < threshold)
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

            effective_stop = min(trail_stop, hard_stop)
            stop_prices[i] = effective_stop

            # 1. Hard Stop 체크 (최우선)
            if close[i] > hard_stop:
                position = 0
                hard_exits += 1
                continue

            # 2. Trailing Stop 체크
            if close[i] > trail_stop:
                position = 0
                trail_exits += 1
                continue

            # 3. Dynamic Innovation Breaker (v_ratio >= threshold)
            if v_ratio[i] >= v_ratio_threshold:
                if log_residual > dynamic_innov_mult * resid_std[i]:
                    position = 0
                    innovation_exits += 1
                    continue

            # 4. Signal Exit (v_ratio < threshold)
            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] > entry_z:
                    position = 0
                    signal_exits += 1
                    continue

            signals[i] = -1

    return signals, stop_prices, trail_exits, signal_exits, innovation_exits, hard_exits


def run_backtest(df: pd.DataFrame, hard_stop_mult: float = 3.0):
    """Run V3.3 backtest"""
    df_kf = calculate_adaptive_kalman(df)
    features = calculate_v32_features(df_kf)

    # Residual std 계산
    close = df_kf["close"].values
    kf_trend = df_kf["kf_trend"].values
    log_residual = np.log(close) - np.log(kf_trend)
    resid_std = pd.Series(log_residual).rolling(window=180, min_periods=30).std().fillna(0.01).values

    signals, stop_prices, trail_exits, signal_exits, innovation_exits, hard_exits = generate_signals_v33(
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
        hard_stop_mult=hard_stop_mult,
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
    metrics["hard_exits"] = hard_exits

    return metrics


def main():
    print("Loading data...")
    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars")

    # Test different hard stop multipliers
    hard_mults = [2.0, 2.5, 3.0, 4.0, 5.0]

    print("\n" + "=" * 160)
    print("V3.3 Test: Dynamic Innovation (base=2.5, [1.5,4.0]) + Hard Stop")
    print("=" * 160)

    # V3.2 baseline for comparison
    v32_baseline = {
        "BTCUSDT": {"pnl": 464.8, "mdd": 25.1, "sharpe": 1.02},
        "ETHUSDT": {"pnl": 374.5, "mdd": 42.7, "sharpe": 0.85},
        "XRPUSDT": {"pnl": 128.2, "mdd": 67.8, "sharpe": 0.52},
        "SOLUSDT": {"pnl": 763.5, "mdd": 67.0, "sharpe": 0.90},
        "BNBUSDT": {"pnl": 11.8, "mdd": 44.7, "sharpe": 0.25},
        "DOGEUSDT": {"pnl": -26.8, "mdd": 68.3, "sharpe": 0.18},
        "LTCUSDT": {"pnl": -41.0, "mdd": 71.3, "sharpe": -0.02},
    }

    all_results = {}

    for hard_mult in hard_mults:
        print(f"\n[hard_stop_mult = {hard_mult}]")
        print(f"{'Symbol':<10} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'Trades':<8} {'Trail':<8} {'Signal':<8} {'Innov':<8} {'Hard':<8}")
        print("-" * 110)

        results = []
        for symbol, df in data.items():
            metrics = run_backtest(df, hard_stop_mult=hard_mult)
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
                "hard_exits": metrics["hard_exits"],
            })
            print(f"{symbol:<10} {metrics['sharpe_ratio']:<10.2f} {metrics['total_pnl']*100:<11.1f}% "
                  f"{metrics['max_drawdown']*100:<9.1f}% {metrics['total_trades']:<8} "
                  f"{metrics['trail_exits']:<8} {metrics['signal_exits']:<8} "
                  f"{metrics['innovation_exits']:<8} {metrics['hard_exits']:<8}")

        all_results[hard_mult] = results

        print("-" * 110)
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)

        print(f"Average: Sharpe={avg_sharpe:.2f}, PnL={avg_pnl*100:.1f}%, MDD={avg_mdd*100:.1f}%, Positive={positive}/7")

    # Summary comparison
    print("\n" + "=" * 160)
    print("Summary 비교")
    print("=" * 160)
    print(f"{'HardMult':<10} {'AvgSharpe':<12} {'AvgPnL':<12} {'AvgMDD':<12} {'Positive':<10} {'BTC_PnL':<12} {'SOL_PnL':<12} {'BTC_MDD':<12}")
    print("-" * 100)

    for hard_mult, results in all_results.items():
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)
        btc = next(r for r in results if r["symbol"] == "BTCUSDT")
        sol = next(r for r in results if r["symbol"] == "SOLUSDT")

        print(f"{hard_mult:<10} {avg_sharpe:<12.2f} {avg_pnl*100:<11.1f}% {avg_mdd*100:<11.1f}% "
              f"{positive}/7       {btc['pnl']*100:<11.1f}% {sol['pnl']*100:<11.1f}% {btc['mdd']*100:<11.1f}%")

    print("\n비교 기준:")
    print("  V3.2: Avg Sharpe=0.53, AvgMDD=55.3%, Positive=5/7, BTC=464.8%, SOL=763.5%")

    # 최적 config 상세 비교
    best_mult = 3.0
    print(f"\n\n[상세 비교: hard_stop_mult={best_mult}] vs V3.2")
    print("-" * 100)
    print(f"{'Symbol':<10} {'V3.2 PnL':<12} {'V3.3 PnL':<12} {'Δ PnL':<12} {'V3.2 MDD':<12} {'V3.3 MDD':<12} {'Δ MDD':<12}")
    print("-" * 100)

    for r in all_results[best_mult]:
        symbol = r["symbol"]
        base = v32_baseline[symbol]
        delta_pnl = r["pnl"] * 100 - base["pnl"]
        delta_mdd = r["mdd"] * 100 - base["mdd"]
        print(f"{symbol:<10} {base['pnl']:<11.1f}% {r['pnl']*100:<11.1f}% {delta_pnl:+11.1f}% "
              f"{base['mdd']:<11.1f}% {r['mdd']*100:<11.1f}% {delta_mdd:+11.1f}%")


if __name__ == "__main__":
    main()
