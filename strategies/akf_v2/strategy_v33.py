#!/usr/bin/env python3
"""
AKF V3.3 Strategy

핵심 로직:
1. Entry: vel_zscore > 2.0 & unc_pct < 0.5
2. Exit (4가지):
   - Hard Stop: entry * exp(-hard_mult * sigma) (진입가 기준)
   - Trailing Stop: highest * exp(-dynamic_mult * sigma) with Ratchet
   - Innovation Breaker: |residual| > innov_mult * resid_std (v_ratio >= 1.0)
   - Signal Exit: vel_zscore 반전 (v_ratio < 1.0)
3. Dynamic Multipliers:
   - Trailing: base_mult / v_ratio
   - Innovation: base_innov / v_ratio
   - Z-score 기반: Mult = 30 - 5 * |Z|, clipped [10, 20]

성능:
- Avg Sharpe: 0.55
- Positive: 5/7
- BTC: Sharpe 1.07, PnL 505%, MDD 20.7%
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
# V3.3 Parameters
# ============================================================================
V33_PARAMS = {
    # Entry
    "entry_z": 2.0,
    "unc_pct_max": 0.5,
    "warmup": 210,

    # Trailing Stop
    "base_mult": 5.0,
    "mult_min_ratio": 0.25,
    "mult_max_ratio": 1.5,

    # Innovation Breaker
    "v_ratio_threshold": 1.0,
    "base_innov_mult": 2.5,
    "innov_mult_min": 1.5,
    "innov_mult_max": 4.0,

    # Hard Stop
    "hard_stop_mult": 5.0,

    # Feature calculation
    "rv_window": 42,
    "baseline_window": 120,
    "resid_window": 180,
}


def load_data(symbol: str, bar_size: int = 6):
    data_dir = DATA_ROOT / f"features-{bar_size}/futures/{symbol}"
    dfs = []
    for year in range(2020, 2026):
        for month in range(1, 13):
            path = data_dir / f"{symbol}-features-{year}-{month:02d}.parquet"
            if path.exists():
                dfs.append(pd.read_parquet(path))
    return pd.concat(dfs, ignore_index=True) if dfs else None


def calculate_v33_features(df_kf: pd.DataFrame, params: dict = None):
    """Calculate V3.3 features"""
    if params is None:
        params = V33_PARAMS

    close = df_kf["close"].values
    kf_uncertainty = df_kf["kf_uncertainty"].values
    kf_trend = df_kf["kf_trend"].values
    velocity = df_kf["kf_velocity"].values

    # 1. Realized Volatility
    log_returns = np.log(close[1:] / close[:-1])
    log_returns = np.concatenate([[0], log_returns])
    rv = pd.Series(log_returns).rolling(window=params["rv_window"], min_periods=10).std().values

    # 2. Hybrid Volatility
    sqrt_uncertainty = np.sqrt(kf_uncertainty)
    sigma_hybrid = np.maximum(sqrt_uncertainty, rv)

    # 3. V-Ratio & Dynamic Mult
    baseline = pd.Series(sigma_hybrid).rolling(window=params["baseline_window"], min_periods=30).mean().values
    v_ratio = sigma_hybrid / (baseline + 1e-10)

    base_mult = params["base_mult"]
    mult_min = base_mult * params["mult_min_ratio"]
    mult_max = base_mult * params["mult_max_ratio"]
    dynamic_mult = np.clip(base_mult / (v_ratio + 1e-10), mult_min, mult_max)

    # 4. Velocity z-score
    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=42, min_periods=5).mean()
    vel_std = vel_series.rolling(window=42, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

    # 5. Uncertainty percentile
    unc_pct = (
        pd.Series(kf_uncertainty)
        .rolling(window=210, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )

    # 6. Residual std
    log_residual = np.log(close) - np.log(kf_trend)
    resid_std = pd.Series(log_residual).rolling(window=params["resid_window"], min_periods=30).std().fillna(0.01).values

    return {
        "sigma_hybrid": sigma_hybrid,
        "v_ratio": v_ratio,
        "dynamic_mult": dynamic_mult,
        "vel_zscore": vel_zscore,
        "unc_pct": unc_pct,
        "resid_std": resid_std,
        "kf_trend": kf_trend,
    }


@njit
def generate_v33_signals(
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
    entry_z: float,
    unc_pct_max: float,
    v_ratio_threshold: float,
    base_innov_mult: float,
    innov_mult_min: float,
    innov_mult_max: float,
    hard_stop_mult: float,
    warmup: int,
):
    """V3.3 Signal Generation"""
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    stop_prices = np.zeros(n, dtype=np.float64)

    position = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = np.inf
    prev_trail_stop = 0.0
    hard_stop = 0.0

    for i in range(warmup, n):
        low_uncertainty = unc_pct[i] < unc_pct_max
        log_residual = np.log(close[i]) - np.log(kf_trend[i])

        # Dynamic innovation mult
        dynamic_innov_mult = base_innov_mult / (v_ratio[i] + 1e-10)
        dynamic_innov_mult = max(innov_mult_min, min(innov_mult_max, dynamic_innov_mult))

        if position == 0:
            if vel_zscore[i] > entry_z and low_uncertainty:
                position = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_trail_stop = highest_since_entry * np.exp(-dist)
                hard_stop = entry_price * np.exp(-hard_stop_mult * sigma_hybrid[i])
                signals[i] = 1
                stop_prices[i] = max(prev_trail_stop, hard_stop)

            elif vel_zscore[i] < -entry_z and low_uncertainty:
                position = -1
                entry_price = close[i]
                lowest_since_entry = low[i]
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_trail_stop = lowest_since_entry * np.exp(dist)
                hard_stop = entry_price * np.exp(hard_stop_mult * sigma_hybrid[i])
                signals[i] = -1
                stop_prices[i] = min(prev_trail_stop, hard_stop)

        elif position == 1:
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]

            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_trail_stop = highest_since_entry * np.exp(-dist)
            trail_stop = max(new_trail_stop, prev_trail_stop)
            prev_trail_stop = trail_stop
            stop_prices[i] = max(trail_stop, hard_stop)

            if close[i] < hard_stop:
                position = 0
                continue
            if close[i] < trail_stop:
                position = 0
                continue
            if v_ratio[i] >= v_ratio_threshold:
                if log_residual < -dynamic_innov_mult * resid_std[i]:
                    position = 0
                    continue
            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] < -entry_z:
                    position = 0
                    continue
            signals[i] = 1

        elif position == -1:
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]

            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_trail_stop = lowest_since_entry * np.exp(dist)
            trail_stop = min(new_trail_stop, prev_trail_stop)
            prev_trail_stop = trail_stop
            stop_prices[i] = min(trail_stop, hard_stop)

            if close[i] > hard_stop:
                position = 0
                continue
            if close[i] > trail_stop:
                position = 0
                continue
            if v_ratio[i] >= v_ratio_threshold:
                if log_residual > dynamic_innov_mult * resid_std[i]:
                    position = 0
                    continue
            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] > entry_z:
                    position = 0
                    continue
            signals[i] = -1

    return signals, stop_prices


def run_v33_backtest(df: pd.DataFrame, params: dict = None):
    """Run V3.3 backtest"""
    if params is None:
        params = V33_PARAMS

    df_kf = calculate_adaptive_kalman(df)
    features = calculate_v33_features(df_kf, params)

    signals, stop_prices = generate_v33_signals(
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
        entry_z=params["entry_z"],
        unc_pct_max=params["unc_pct_max"],
        v_ratio_threshold=params["v_ratio_threshold"],
        base_innov_mult=params["base_innov_mult"],
        innov_mult_min=params["innov_mult_min"],
        innov_mult_max=params["innov_mult_max"],
        hard_stop_mult=params["hard_stop_mult"],
        warmup=params["warmup"],
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
    return metrics, df_kf


def main():
    print("=" * 100)
    print("AKF V3.3 Strategy")
    print("=" * 100)

    print("\nLoading data...")
    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars")

    print("\n" + "-" * 100)
    print(f"{'Symbol':<10} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'Trades':<8} {'WinRate':<10}")
    print("-" * 100)

    results = []
    for symbol, df in data.items():
        metrics, _ = run_v33_backtest(df)
        results.append({
            "symbol": symbol,
            "sharpe": metrics["sharpe_ratio"],
            "pnl": metrics["total_pnl"],
            "mdd": metrics["max_drawdown"],
            "trades": metrics["total_trades"],
            "win_rate": metrics["win_rate"],
        })
        print(f"{symbol:<10} {metrics['sharpe_ratio']:<10.2f} {metrics['total_pnl']*100:<11.1f}% "
              f"{metrics['max_drawdown']*100:<9.1f}% {metrics['total_trades']:<8} {metrics['win_rate']*100:<9.1f}%")

    print("-" * 100)
    avg_sharpe = np.mean([r["sharpe"] for r in results])
    avg_pnl = np.mean([r["pnl"] for r in results])
    positive = sum(1 for r in results if r["pnl"] > 0)
    btc = next(r for r in results if r["symbol"] == "BTCUSDT")

    print(f"\nSummary:")
    print(f"  Average Sharpe: {avg_sharpe:.2f}")
    print(f"  Average PnL: {avg_pnl*100:.1f}%")
    print(f"  Positive: {positive}/7")
    print(f"  BTC: Sharpe={btc['sharpe']:.2f}, PnL={btc['pnl']*100:.1f}%")


if __name__ == "__main__":
    main()
