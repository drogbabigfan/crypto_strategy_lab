#!/usr/bin/env python3
"""
V2.2 ~ V3.3 전략 비교
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


# ============================================================================
# V2.2: Signal Exit + KF 3σ Stop
# ============================================================================
def run_v22(df: pd.DataFrame):
    df_kf = calculate_adaptive_kalman(df)

    velocity = df_kf["kf_velocity"].values
    uncertainty = df_kf["kf_uncertainty"].values
    close = df_kf["close"].values
    kf_trend = df_kf["kf_trend"].values
    n = len(df_kf)

    log_price = np.log(close)
    log_trend = np.log(kf_trend)
    log_residuals = log_price - log_trend
    resid_std = pd.Series(log_residuals).rolling(window=180, min_periods=30).std().values

    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=42, min_periods=5).mean()
    vel_std = vel_series.rolling(window=42, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

    unc_pct = (
        pd.Series(uncertainty)
        .rolling(window=210, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )

    signals = np.zeros(n, dtype=np.int8)
    position = 0
    warmup = 210
    entry_z = 2.0

    for i in range(warmup, n):
        low_uncertainty = unc_pct[i] < 0.5

        if position == 0:
            if vel_zscore[i] > entry_z and low_uncertainty:
                position = 1
                signals[i] = 1
            elif vel_zscore[i] < -entry_z and low_uncertainty:
                position = -1
                signals[i] = -1

        elif position == 1:
            # KF 3σ stop
            if not np.isnan(resid_std[i]) and log_residuals[i] < -3.0 * resid_std[i]:
                position = 0
                continue
            # Signal exit
            if vel_zscore[i] < -entry_z:
                position = 0
            else:
                signals[i] = 1

        elif position == -1:
            if not np.isnan(resid_std[i]) and log_residuals[i] > 3.0 * resid_std[i]:
                position = 0
                continue
            if vel_zscore[i] > entry_z:
                position = 0
            else:
                signals[i] = -1

    df_kf["signal"] = signals
    sizing_config = SizingConfig(method="fixed", min_size=1.0, max_size=1.0)
    df_kf["position_size"] = calculate_position_sizes(df_kf, sizing_config, entry_signals=signals)
    df_kf["sl_price"] = 0.0

    bt_config = PyramidBacktestConfig(
        initial_capital=100000.0,
        compounding=True,
        fee_rate=0.001,
        slippage_rate=0.0001,
    )
    return run_pyramid_backtest(df_kf, bt_config)


# ============================================================================
# V3.1: Trailing Stop Only (Hybrid Vol)
# ============================================================================
@njit
def _v31_signals(close, high, low, vel_zscore, unc_pct, sigma_hybrid, dynamic_mult, entry_z=2.0, warmup=210):
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    position = 0
    highest = 0.0
    lowest = np.inf
    prev_stop = 0.0

    for i in range(warmup, n):
        if position == 0:
            if vel_zscore[i] > entry_z and unc_pct[i] < 0.5:
                position = 1
                highest = high[i]
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = highest * np.exp(-dist)
                signals[i] = 1
            elif vel_zscore[i] < -entry_z and unc_pct[i] < 0.5:
                position = -1
                lowest = low[i]
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = lowest * np.exp(dist)
                signals[i] = -1

        elif position == 1:
            if high[i] > highest:
                highest = high[i]
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_stop = highest * np.exp(-dist)
            trail_stop = max(new_stop, prev_stop)
            prev_stop = trail_stop
            if close[i] < trail_stop:
                position = 0
                continue
            signals[i] = 1

        elif position == -1:
            if low[i] < lowest:
                lowest = low[i]
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_stop = lowest * np.exp(dist)
            trail_stop = min(new_stop, prev_stop)
            prev_stop = trail_stop
            if close[i] > trail_stop:
                position = 0
                continue
            signals[i] = -1

    return signals


def run_v31(df: pd.DataFrame):
    df_kf = calculate_adaptive_kalman(df)
    close = df_kf["close"].values
    kf_uncertainty = df_kf["kf_uncertainty"].values
    velocity = df_kf["kf_velocity"].values

    # Hybrid Vol
    log_returns = np.log(close[1:] / close[:-1])
    log_returns = np.concatenate([[0], log_returns])
    rv = pd.Series(log_returns).rolling(window=42, min_periods=10).std().values
    sqrt_unc = np.sqrt(kf_uncertainty)
    sigma_hybrid = np.maximum(sqrt_unc, rv)

    baseline = pd.Series(sigma_hybrid).rolling(window=120, min_periods=30).mean().values
    v_ratio = sigma_hybrid / (baseline + 1e-10)
    dynamic_mult = np.clip(5.0 / (v_ratio + 1e-10), 1.25, 7.5)

    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=42, min_periods=5).mean()
    vel_std = vel_series.rolling(window=42, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

    unc_pct = (
        pd.Series(kf_uncertainty)
        .rolling(window=210, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )

    signals = _v31_signals(close, df_kf["high"].values, df_kf["low"].values,
                           vel_zscore, unc_pct, sigma_hybrid, dynamic_mult)

    df_kf["signal"] = signals
    sizing_config = SizingConfig(method="fixed", min_size=1.0, max_size=1.0)
    df_kf["position_size"] = calculate_position_sizes(df_kf, sizing_config, entry_signals=signals)
    df_kf["sl_price"] = 0.0

    bt_config = PyramidBacktestConfig(
        initial_capital=100000.0,
        compounding=True,
        fee_rate=0.001,
        slippage_rate=0.0001,
    )
    return run_pyramid_backtest(df_kf, bt_config)


# ============================================================================
# V3.2: Trail + Signal Exit (v_ratio < 1.0)
# ============================================================================
@njit
def _v32_signals(close, high, low, vel_zscore, unc_pct, sigma_hybrid, dynamic_mult, v_ratio,
                 entry_z=2.0, v_ratio_threshold=1.0, warmup=210):
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    position = 0
    highest = 0.0
    lowest = np.inf
    prev_stop = 0.0

    for i in range(warmup, n):
        if position == 0:
            if vel_zscore[i] > entry_z and unc_pct[i] < 0.5:
                position = 1
                highest = high[i]
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = highest * np.exp(-dist)
                signals[i] = 1
            elif vel_zscore[i] < -entry_z and unc_pct[i] < 0.5:
                position = -1
                lowest = low[i]
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = lowest * np.exp(dist)
                signals[i] = -1

        elif position == 1:
            if high[i] > highest:
                highest = high[i]
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_stop = highest * np.exp(-dist)
            trail_stop = max(new_stop, prev_stop)
            prev_stop = trail_stop

            if close[i] < trail_stop:
                position = 0
                continue
            if v_ratio[i] < v_ratio_threshold and vel_zscore[i] < -entry_z:
                position = 0
                continue
            signals[i] = 1

        elif position == -1:
            if low[i] < lowest:
                lowest = low[i]
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_stop = lowest * np.exp(dist)
            trail_stop = min(new_stop, prev_stop)
            prev_stop = trail_stop

            if close[i] > trail_stop:
                position = 0
                continue
            if v_ratio[i] < v_ratio_threshold and vel_zscore[i] > entry_z:
                position = 0
                continue
            signals[i] = -1

    return signals


def run_v32(df: pd.DataFrame):
    df_kf = calculate_adaptive_kalman(df)
    close = df_kf["close"].values
    kf_uncertainty = df_kf["kf_uncertainty"].values
    velocity = df_kf["kf_velocity"].values

    log_returns = np.log(close[1:] / close[:-1])
    log_returns = np.concatenate([[0], log_returns])
    rv = pd.Series(log_returns).rolling(window=42, min_periods=10).std().values
    sqrt_unc = np.sqrt(kf_uncertainty)
    sigma_hybrid = np.maximum(sqrt_unc, rv)

    baseline = pd.Series(sigma_hybrid).rolling(window=120, min_periods=30).mean().values
    v_ratio = sigma_hybrid / (baseline + 1e-10)
    dynamic_mult = np.clip(5.0 / (v_ratio + 1e-10), 1.25, 7.5)

    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=42, min_periods=5).mean()
    vel_std = vel_series.rolling(window=42, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

    unc_pct = (
        pd.Series(kf_uncertainty)
        .rolling(window=210, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )

    signals = _v32_signals(close, df_kf["high"].values, df_kf["low"].values,
                           vel_zscore, unc_pct, sigma_hybrid, dynamic_mult, v_ratio)

    df_kf["signal"] = signals
    sizing_config = SizingConfig(method="fixed", min_size=1.0, max_size=1.0)
    df_kf["position_size"] = calculate_position_sizes(df_kf, sizing_config, entry_signals=signals)
    df_kf["sl_price"] = 0.0

    bt_config = PyramidBacktestConfig(
        initial_capital=100000.0,
        compounding=True,
        fee_rate=0.001,
        slippage_rate=0.0001,
    )
    return run_pyramid_backtest(df_kf, bt_config)


# ============================================================================
# V3.3: V3.2 + Innovation Breaker + Hard Stop
# ============================================================================
@njit
def _v33_signals(close, high, low, kf_trend, vel_zscore, unc_pct, sigma_hybrid, dynamic_mult, v_ratio, resid_std,
                 entry_z=2.0, v_ratio_threshold=1.0, base_innov=2.5, innov_min=1.5, innov_max=4.0,
                 hard_mult=5.0, warmup=210):
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    position = 0
    entry_price = 0.0
    highest = 0.0
    lowest = np.inf
    prev_stop = 0.0
    hard_stop = 0.0

    for i in range(warmup, n):
        log_resid = np.log(close[i]) - np.log(kf_trend[i])
        innov_mult = base_innov / (v_ratio[i] + 1e-10)
        innov_mult = max(innov_min, min(innov_max, innov_mult))

        if position == 0:
            if vel_zscore[i] > entry_z and unc_pct[i] < 0.5:
                position = 1
                entry_price = close[i]
                highest = high[i]
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = highest * np.exp(-dist)
                hard_stop = entry_price * np.exp(-hard_mult * sigma_hybrid[i])
                signals[i] = 1
            elif vel_zscore[i] < -entry_z and unc_pct[i] < 0.5:
                position = -1
                entry_price = close[i]
                lowest = low[i]
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = lowest * np.exp(dist)
                hard_stop = entry_price * np.exp(hard_mult * sigma_hybrid[i])
                signals[i] = -1

        elif position == 1:
            if high[i] > highest:
                highest = high[i]
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_stop = highest * np.exp(-dist)
            trail_stop = max(new_stop, prev_stop)
            prev_stop = trail_stop

            if close[i] < hard_stop:
                position = 0
                continue
            if close[i] < trail_stop:
                position = 0
                continue
            if v_ratio[i] >= v_ratio_threshold:
                if log_resid < -innov_mult * resid_std[i]:
                    position = 0
                    continue
            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] < -entry_z:
                    position = 0
                    continue
            signals[i] = 1

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
                continue
            if v_ratio[i] >= v_ratio_threshold:
                if log_resid > innov_mult * resid_std[i]:
                    position = 0
                    continue
            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] > entry_z:
                    position = 0
                    continue
            signals[i] = -1

    return signals


def run_v33(df: pd.DataFrame):
    df_kf = calculate_adaptive_kalman(df)
    close = df_kf["close"].values
    kf_trend = df_kf["kf_trend"].values
    kf_uncertainty = df_kf["kf_uncertainty"].values
    velocity = df_kf["kf_velocity"].values

    log_returns = np.log(close[1:] / close[:-1])
    log_returns = np.concatenate([[0], log_returns])
    rv = pd.Series(log_returns).rolling(window=42, min_periods=10).std().values
    sqrt_unc = np.sqrt(kf_uncertainty)
    sigma_hybrid = np.maximum(sqrt_unc, rv)

    baseline = pd.Series(sigma_hybrid).rolling(window=120, min_periods=30).mean().values
    v_ratio = sigma_hybrid / (baseline + 1e-10)
    dynamic_mult = np.clip(5.0 / (v_ratio + 1e-10), 1.25, 7.5)

    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=42, min_periods=5).mean()
    vel_std = vel_series.rolling(window=42, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

    unc_pct = (
        pd.Series(kf_uncertainty)
        .rolling(window=210, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )

    log_resid = np.log(close) - np.log(kf_trend)
    resid_std = pd.Series(log_resid).rolling(window=180, min_periods=30).std().fillna(0.01).values

    signals = _v33_signals(close, df_kf["high"].values, df_kf["low"].values, kf_trend,
                           vel_zscore, unc_pct, sigma_hybrid, dynamic_mult, v_ratio, resid_std)

    df_kf["signal"] = signals
    sizing_config = SizingConfig(method="fixed", min_size=1.0, max_size=1.0)
    df_kf["position_size"] = calculate_position_sizes(df_kf, sizing_config, entry_signals=signals)
    df_kf["sl_price"] = 0.0

    bt_config = PyramidBacktestConfig(
        initial_capital=100000.0,
        compounding=True,
        fee_rate=0.001,
        slippage_rate=0.0001,
    )
    return run_pyramid_backtest(df_kf, bt_config)


# ============================================================================
# Main
# ============================================================================
def main():
    print("Loading data...")
    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars")

    versions = [
        ("V2.2", run_v22),
        ("V3.1", run_v31),
        ("V3.2", run_v32),
        ("V3.3", run_v33),
    ]

    all_results = {}

    for version_name, run_func in versions:
        print(f"\n{'='*100}")
        print(f"{version_name}")
        print("="*100)
        print(f"{'Symbol':<10} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'Trades':<8} {'WinRate':<10}")
        print("-"*70)

        results = []
        for symbol, df in data.items():
            metrics = run_func(df)
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

        all_results[version_name] = results

        print("-"*70)
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)
        btc = next(r for r in results if r["symbol"] == "BTCUSDT")

        print(f"Average: Sharpe={avg_sharpe:.2f}, PnL={avg_pnl*100:.1f}%, MDD={avg_mdd*100:.1f}%, Positive={positive}/7")
        print(f"BTC: Sharpe={btc['sharpe']:.2f}, PnL={btc['pnl']*100:.1f}%, MDD={btc['mdd']*100:.1f}%")

    # Final Summary
    print("\n" + "="*120)
    print("FINAL SUMMARY")
    print("="*120)
    print(f"{'Version':<10} {'AvgSharpe':<12} {'AvgPnL':<12} {'AvgMDD':<12} {'Positive':<10} {'BTC_Sharpe':<12} {'BTC_PnL':<12} {'BTC_MDD':<12}")
    print("-"*110)

    for version_name, results in all_results.items():
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)
        btc = next(r for r in results if r["symbol"] == "BTCUSDT")

        print(f"{version_name:<10} {avg_sharpe:<12.2f} {avg_pnl*100:<11.1f}% {avg_mdd*100:<11.1f}% "
              f"{positive}/7       {btc['sharpe']:<12.2f} {btc['pnl']*100:<11.1f}% {btc['mdd']*100:<11.1f}%")

    # Symbol-by-symbol comparison
    print("\n" + "="*120)
    print("SYMBOL-BY-SYMBOL COMPARISON")
    print("="*120)

    for symbol in ALL_SYMBOLS:
        print(f"\n[{symbol}]")
        print(f"{'Version':<10} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'RnR':<10}")
        print("-"*50)
        for version_name, results in all_results.items():
            r = next(x for x in results if x["symbol"] == symbol)
            rnr = r["pnl"] / r["mdd"] if r["mdd"] > 0 else 0
            print(f"{version_name:<10} {r['sharpe']:<10.2f} {r['pnl']*100:<11.1f}% {r['mdd']*100:<9.1f}% {rnr:<10.1f}")


if __name__ == "__main__":
    main()
