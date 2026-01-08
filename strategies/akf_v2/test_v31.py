#!/usr/bin/env python3
"""
AKF V3.1 Strategy

핵심 개선:
1. Hybrid Volatility: max(sqrt(kf_uncertainty), realized_vol) - Lag 해결
2. Adaptive Multiplier: Base_Mult / V_Ratio - 변동성 클수록 타이트
3. Signal Exit 제거: ONLY Trailing Stop + Ratchet
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


@njit
def calculate_trailing_stop_with_ratchet(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    vel_zscore: np.ndarray,
    unc_pct: np.ndarray,
    sigma_hybrid: np.ndarray,
    dynamic_mult: np.ndarray,
    entry_z: float = 2.0,
    warmup: int = 210,
):
    """
    V3.1 Signal Generation with Ratchet Trailing Stop

    Returns: signals, stop_prices
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    stop_prices = np.zeros(n, dtype=np.float64)

    position = 0
    highest_since_entry = 0.0
    lowest_since_entry = np.inf
    prev_stop = 0.0

    for i in range(warmup, n):
        low_uncertainty = unc_pct[i] < 0.5

        if position == 0:
            # Entry
            if vel_zscore[i] > entry_z and low_uncertainty:
                position = 1
                highest_since_entry = high[i]
                # 초기 stop 설정
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = highest_since_entry * np.exp(-dist)
                signals[i] = 1
                stop_prices[i] = prev_stop

            elif vel_zscore[i] < -entry_z and low_uncertainty:
                position = -1
                lowest_since_entry = low[i]
                # 초기 stop 설정
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = lowest_since_entry * np.exp(dist)
                signals[i] = -1
                stop_prices[i] = prev_stop

        elif position == 1:  # Long
            # Update highest
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]

            # Calculate new stop
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_stop = highest_since_entry * np.exp(-dist)

            # Ratchet: Long stop은 절대 내려가면 안됨
            trail_stop = max(new_stop, prev_stop)
            prev_stop = trail_stop
            stop_prices[i] = trail_stop

            # Check stop hit
            if close[i] < trail_stop:
                position = 0
                continue

            signals[i] = 1

        elif position == -1:  # Short
            # Update lowest
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]

            # Calculate new stop
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_stop = lowest_since_entry * np.exp(dist)

            # Ratchet: Short stop은 절대 올라가면 안됨
            trail_stop = min(new_stop, prev_stop)
            prev_stop = trail_stop
            stop_prices[i] = trail_stop

            # Check stop hit
            if close[i] > trail_stop:
                position = 0
                continue

            signals[i] = -1

    return signals, stop_prices


def calculate_v31_features(df_kf: pd.DataFrame, rv_window: int = 42, baseline_window: int = 120, base_mult: float = 20.0):
    """
    V3.1 핵심 피처 계산

    1. Realized Volatility (즉각적)
    2. Hybrid Volatility = max(sqrt(uncertainty), RV)
    3. V-Ratio = sigma_hybrid / baseline
    4. Dynamic Multiplier = Base_Mult / V_Ratio (clipped)
    """
    close = df_kf["close"].values
    kf_uncertainty = df_kf["kf_uncertainty"].values

    # 1. Realized Volatility (log return std, 달러바 보정 sqrt(6))
    log_returns = np.log(close[1:] / close[:-1])
    log_returns = np.concatenate([[0], log_returns])
    rv = pd.Series(log_returns).rolling(window=rv_window, min_periods=10).std().values * np.sqrt(6)

    # 2. Hybrid Volatility = max(sqrt(uncertainty), RV)
    sqrt_uncertainty = np.sqrt(kf_uncertainty)
    sigma_hybrid = np.maximum(sqrt_uncertainty, rv)

    # 3. Baseline = rolling mean of sigma_hybrid
    baseline = pd.Series(sigma_hybrid).rolling(window=baseline_window, min_periods=30).mean().values

    # 4. V-Ratio = sigma_hybrid / baseline
    v_ratio = sigma_hybrid / (baseline + 1e-10)

    # 5. Dynamic Multiplier = Base_Mult / V_Ratio, clipped [5, 30]
    dynamic_mult = np.clip(base_mult / (v_ratio + 1e-10), 5.0, 30.0)

    return {
        "realized_vol": rv,
        "sqrt_uncertainty": sqrt_uncertainty,
        "sigma_hybrid": sigma_hybrid,
        "baseline": baseline,
        "v_ratio": v_ratio,
        "dynamic_mult": dynamic_mult,
    }


def run_v31_backtest(df: pd.DataFrame, base_mult: float = 20.0):
    """Run V3.1 backtest"""
    df_kf = calculate_adaptive_kalman(df)

    # V3.1 features
    features = calculate_v31_features(df_kf, base_mult=base_mult)

    # Velocity z-score
    velocity = df_kf["kf_velocity"].values
    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=42, min_periods=5).mean()
    vel_std = vel_series.rolling(window=42, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

    # Uncertainty percentile
    uncertainty = df_kf["kf_uncertainty"].values
    unc_pct = (
        pd.Series(uncertainty)
        .rolling(window=210, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )

    # Generate signals with Numba
    signals, stop_prices = calculate_trailing_stop_with_ratchet(
        close=df_kf["close"].values,
        high=df_kf["high"].values,
        low=df_kf["low"].values,
        vel_zscore=vel_zscore,
        unc_pct=unc_pct,
        sigma_hybrid=features["sigma_hybrid"],
        dynamic_mult=features["dynamic_mult"],
        entry_z=2.0,
        warmup=210,
    )

    # Backtest
    df_kf["signal"] = signals
    df_kf["stop_price"] = stop_prices
    df_kf["v_ratio"] = features["v_ratio"]
    df_kf["dynamic_mult"] = features["dynamic_mult"]
    df_kf["sigma_hybrid"] = features["sigma_hybrid"]

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

    # 추가 통계
    metrics["avg_v_ratio"] = np.nanmean(features["v_ratio"][210:])
    metrics["avg_dynamic_mult"] = np.nanmean(features["dynamic_mult"][210:])

    return metrics, df_kf


def main():
    print("Loading data...")
    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars")

    # Test V3.1
    print("\n" + "=" * 110)
    print("AKF V3.1 Test: Hybrid Volatility + Adaptive Trailing Stop")
    print("=" * 110)

    print(f"\n{'Symbol':<10} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'Trades':<8} {'WinRate':<10} {'AvgVRatio':<12} {'AvgMult':<10}")
    print("-" * 95)

    results = []
    for symbol, df in data.items():
        metrics, df_result = run_v31_backtest(df)
        results.append({
            "symbol": symbol,
            "sharpe": metrics["sharpe_ratio"],
            "pnl": metrics["total_pnl"],
            "mdd": metrics["max_drawdown"],
            "trades": metrics["total_trades"],
            "win_rate": metrics["win_rate"],
            "avg_v_ratio": metrics["avg_v_ratio"],
            "avg_mult": metrics["avg_dynamic_mult"],
        })
        print(f"{symbol:<10} {metrics['sharpe_ratio']:<10.2f} {metrics['total_pnl']*100:<11.1f}% "
              f"{metrics['max_drawdown']*100:<9.1f}% {metrics['total_trades']:<8} "
              f"{metrics['win_rate']*100:<9.1f}% {metrics['avg_v_ratio']:<12.2f} {metrics['avg_dynamic_mult']:<10.1f}")

    print("-" * 95)
    avg_sharpe = np.mean([r["sharpe"] for r in results])
    avg_pnl = np.mean([r["pnl"] for r in results])
    positive = sum(1 for r in results if r["pnl"] > 0)
    btc = next(r for r in results if r["symbol"] == "BTCUSDT")

    print(f"Average: Sharpe={avg_sharpe:.2f}, PnL={avg_pnl*100:.1f}%, Positive={positive}/7")
    print(f"BTC: Sharpe={btc['sharpe']:.2f}, PnL={btc['pnl']*100:.1f}%")

    # V2.2 baseline 비교
    print("\n" + "=" * 110)
    print("비교: V2.2 (Baseline z=2.0, Signal Exit)")
    print("=" * 110)
    print("  V2.2: Avg Sharpe=0.34, BTC=1657%, 알트 대부분 손실")
    print(f"  V3.1: Avg Sharpe={avg_sharpe:.2f}, BTC={btc['pnl']*100:.0f}%, Positive={positive}/7")


if __name__ == "__main__":
    main()
