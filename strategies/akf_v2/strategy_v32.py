#!/usr/bin/env python3
"""
AKF V3.2 Strategy

핵심 로직:
1. Entry: vel_zscore > 2.0 & unc_pct < 0.5
2. Exit:
   - Trailing Stop: highest * exp(-dynamic_mult * sigma_hybrid) with Ratchet
   - Signal Exit: v_ratio < 1.0 일 때만 허용 (변동성 낮을 때)
3. Hybrid Volatility: max(sqrt(kf_uncertainty), RV)
4. Dynamic Multiplier: clip(base_mult / v_ratio, min, max)

성능:
- Avg Sharpe: 0.53
- Positive: 5/7
- BTC: Sharpe 1.02, PnL 464.8%
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

# ============================================================================
# V3.2 Parameters
# ============================================================================
V32_PARAMS = {
    "entry_z": 2.0,              # 진입 z-score threshold
    "unc_pct_max": 0.5,          # uncertainty percentile 상한
    "base_mult": 5.0,            # trailing stop base multiplier
    "mult_min_ratio": 0.25,      # dynamic_mult 하한 (base * 0.25)
    "mult_max_ratio": 1.5,       # dynamic_mult 상한 (base * 1.5)
    "v_ratio_threshold": 1.0,    # Signal Exit 허용 v_ratio 상한
    "rv_window": 42,             # Realized Vol rolling window
    "baseline_window": 120,      # Baseline rolling window
    "warmup": 210,               # warmup bars
}


# ============================================================================
# Data Loading
# ============================================================================
def load_data(symbol: str, bar_size: int = 6):
    """Load parquet data for symbol"""
    data_dir = DATA_ROOT / f"features-{bar_size}/futures/{symbol}"
    dfs = []
    for year in range(2020, 2026):
        for month in range(1, 13):
            path = data_dir / f"{symbol}-features-{year}-{month:02d}.parquet"
            if path.exists():
                dfs.append(pd.read_parquet(path))
    return pd.concat(dfs, ignore_index=True) if dfs else None


# ============================================================================
# Feature Calculation
# ============================================================================
def calculate_v32_features(df_kf: pd.DataFrame, params: dict = None):
    """
    V3.2 핵심 피처 계산

    Returns:
        dict with sigma_hybrid, v_ratio, dynamic_mult, vel_zscore, unc_pct
    """
    if params is None:
        params = V32_PARAMS

    close = df_kf["close"].values
    kf_uncertainty = df_kf["kf_uncertainty"].values
    velocity = df_kf["kf_velocity"].values

    # 1. Realized Volatility
    log_returns = np.log(close[1:] / close[:-1])
    log_returns = np.concatenate([[0], log_returns])
    rv = pd.Series(log_returns).rolling(
        window=params["rv_window"], min_periods=10
    ).std().values

    # 2. Hybrid Volatility = max(sqrt(uncertainty), RV)
    sqrt_uncertainty = np.sqrt(kf_uncertainty)
    sigma_hybrid = np.maximum(sqrt_uncertainty, rv)

    # 3. Baseline & V-Ratio
    baseline = pd.Series(sigma_hybrid).rolling(
        window=params["baseline_window"], min_periods=30
    ).mean().values
    v_ratio = sigma_hybrid / (baseline + 1e-10)

    # 4. Dynamic Multiplier
    base_mult = params["base_mult"]
    mult_min = base_mult * params["mult_min_ratio"]
    mult_max = base_mult * params["mult_max_ratio"]
    dynamic_mult = np.clip(base_mult / (v_ratio + 1e-10), mult_min, mult_max)

    # 5. Velocity z-score
    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=42, min_periods=5).mean()
    vel_std = vel_series.rolling(window=42, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

    # 6. Uncertainty percentile
    unc_pct = (
        pd.Series(kf_uncertainty)
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
    }


# ============================================================================
# Signal Generation (Numba optimized)
# ============================================================================
@njit
def generate_v32_signals(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    vel_zscore: np.ndarray,
    unc_pct: np.ndarray,
    sigma_hybrid: np.ndarray,
    dynamic_mult: np.ndarray,
    v_ratio: np.ndarray,
    entry_z: float = 2.0,
    unc_pct_max: float = 0.5,
    v_ratio_threshold: float = 1.0,
    warmup: int = 210,
):
    """
    V3.2 Signal Generation

    Entry: vel_zscore > entry_z & unc_pct < unc_pct_max
    Exit:
      - Trailing Stop with Ratchet
      - Signal Exit only when v_ratio < v_ratio_threshold

    Returns: signals, stop_prices, exit_types
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    stop_prices = np.zeros(n, dtype=np.float64)

    position = 0
    highest_since_entry = 0.0
    lowest_since_entry = np.inf
    prev_stop = 0.0

    for i in range(warmup, n):
        low_uncertainty = unc_pct[i] < unc_pct_max

        if position == 0:
            # Long Entry
            if vel_zscore[i] > entry_z and low_uncertainty:
                position = 1
                highest_since_entry = high[i]
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = highest_since_entry * np.exp(-dist)
                signals[i] = 1
                stop_prices[i] = prev_stop

            # Short Entry
            elif vel_zscore[i] < -entry_z and low_uncertainty:
                position = -1
                lowest_since_entry = low[i]
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = lowest_since_entry * np.exp(dist)
                signals[i] = -1
                stop_prices[i] = prev_stop

        elif position == 1:  # Long Position
            # Update highest
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]

            # Calculate trailing stop
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_stop = highest_since_entry * np.exp(-dist)

            # Ratchet: Long stop never goes down
            trail_stop = max(new_stop, prev_stop)
            prev_stop = trail_stop
            stop_prices[i] = trail_stop

            # Check Trailing Stop
            if close[i] < trail_stop:
                position = 0
                continue

            # Signal Exit: only when v_ratio < threshold
            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] < -entry_z:
                    position = 0
                    continue

            signals[i] = 1

        elif position == -1:  # Short Position
            # Update lowest
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]

            # Calculate trailing stop
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_stop = lowest_since_entry * np.exp(dist)

            # Ratchet: Short stop never goes up
            trail_stop = min(new_stop, prev_stop)
            prev_stop = trail_stop
            stop_prices[i] = trail_stop

            # Check Trailing Stop
            if close[i] > trail_stop:
                position = 0
                continue

            # Signal Exit: only when v_ratio < threshold
            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] > entry_z:
                    position = 0
                    continue

            signals[i] = -1

    return signals, stop_prices


# ============================================================================
# Backtest Runner
# ============================================================================
def run_v32_backtest(df: pd.DataFrame, params: dict = None):
    """
    Run V3.2 backtest

    Returns: metrics dict, df_result
    """
    if params is None:
        params = V32_PARAMS

    # Calculate Kalman features
    df_kf = calculate_adaptive_kalman(df)

    # Calculate V3.2 features
    features = calculate_v32_features(df_kf, params)

    # Generate signals
    signals, stop_prices = generate_v32_signals(
        close=df_kf["close"].values,
        high=df_kf["high"].values,
        low=df_kf["low"].values,
        vel_zscore=features["vel_zscore"],
        unc_pct=features["unc_pct"],
        sigma_hybrid=features["sigma_hybrid"],
        dynamic_mult=features["dynamic_mult"],
        v_ratio=features["v_ratio"],
        entry_z=params["entry_z"],
        unc_pct_max=params["unc_pct_max"],
        v_ratio_threshold=params["v_ratio_threshold"],
        warmup=params["warmup"],
    )

    # Add to dataframe
    df_kf["signal"] = signals
    df_kf["stop_price"] = stop_prices
    df_kf["v_ratio"] = features["v_ratio"]
    df_kf["dynamic_mult"] = features["dynamic_mult"]
    df_kf["sigma_hybrid"] = features["sigma_hybrid"]

    # Position sizing
    sizing_config = SizingConfig(method="fixed", min_size=1.0, max_size=1.0)
    df_kf["position_size"] = calculate_position_sizes(df_kf, sizing_config, entry_signals=signals)
    df_kf["sl_price"] = 0.0

    # Run backtest
    bt_config = PyramidBacktestConfig(
        initial_capital=100000.0,
        compounding=True,
        fee_rate=0.001,
        slippage_rate=0.0001,
    )

    metrics = run_pyramid_backtest(df_kf, bt_config)

    # Add feature stats
    warmup = params["warmup"]
    metrics["avg_v_ratio"] = np.nanmean(features["v_ratio"][warmup:])
    metrics["avg_dynamic_mult"] = np.nanmean(features["dynamic_mult"][warmup:])

    return metrics, df_kf


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 100)
    print("AKF V3.2 Strategy")
    print("=" * 100)
    print("\nParameters:")
    for k, v in V32_PARAMS.items():
        print(f"  {k}: {v}")

    print("\nLoading data...")
    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars")

    print("\n" + "-" * 100)
    print(f"{'Symbol':<10} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'Trades':<8} {'WinRate':<10} {'AvgVRatio':<12} {'AvgMult':<10}")
    print("-" * 100)

    results = []
    for symbol, df in data.items():
        metrics, _ = run_v32_backtest(df)
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

    print("-" * 100)

    # Summary
    avg_sharpe = np.mean([r["sharpe"] for r in results])
    avg_pnl = np.mean([r["pnl"] for r in results])
    positive = sum(1 for r in results if r["pnl"] > 0)
    btc = next(r for r in results if r["symbol"] == "BTCUSDT")

    print(f"\nSummary:")
    print(f"  Average Sharpe: {avg_sharpe:.2f}")
    print(f"  Average PnL: {avg_pnl*100:.1f}%")
    print(f"  Positive: {positive}/7")
    print(f"  BTC: Sharpe={btc['sharpe']:.2f}, PnL={btc['pnl']*100:.1f}%")

    # Version comparison
    print("\n" + "=" * 100)
    print("Version Comparison")
    print("=" * 100)
    print(f"{'Version':<15} {'Exit Logic':<40} {'AvgSharpe':<12} {'Positive':<10} {'BTC PnL':<12}")
    print("-" * 90)
    print(f"{'V2.2':<15} {'Signal Exit + KF 3σ':<40} {'0.34':<12} {'4/7':<10} {'1657%':<12}")
    print(f"{'V3.1':<15} {'Trailing Stop Only':<40} {'0.43':<12} {'6/7':<10} {'102%':<12}")
    btc_pnl_str = f"{btc['pnl']*100:.0f}%"
    print(f"{'V3.2':<15} {'Trail + Signal(v_ratio<1.0)':<40} {avg_sharpe:<12.2f} {positive}/7       {btc_pnl_str:<12}")


if __name__ == "__main__":
    main()
