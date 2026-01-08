#!/usr/bin/env python3
"""
새 청산 로직 테스트:
1. 진입: z=2.5 (강한 신호만)
2. 청산: Uncertainty 기반 trailing stop + Ratchet
   - Long: stop = exp(log(highest) - mult * sqrt(uncertainty))
   - Short: stop = exp(log(lowest) + mult * sqrt(uncertainty))
   - Ratchet: stop은 절대 뒤로 물러나지 않음
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

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


def generate_signals_trailing(
    df_kf,
    entry_z=2.5,
    trail_mult=3.0,        # sqrt(uncertainty) 배수
    kf_stop_mult=3.0,      # KF residual stop (기존)
):
    """
    Entry: z > entry_z (강한 신호만)
    Exit: Uncertainty 기반 trailing stop with Ratchet
    """
    velocity = df_kf["kf_velocity"].values
    uncertainty = df_kf["kf_uncertainty"].values
    close = df_kf["close"].values
    high = df_kf["high"].values
    low = df_kf["low"].values
    kf_trend = df_kf["kf_trend"].values
    n = len(df_kf)

    # Log space
    log_price = np.log(close)
    log_trend = np.log(kf_trend)
    log_residuals = log_price - log_trend
    resid_std = pd.Series(log_residuals).rolling(window=180, min_periods=30).std().values

    # Velocity z-score
    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=42, min_periods=5).mean()
    vel_std = vel_series.rolling(window=42, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

    # Uncertainty percentile (for entry filter)
    unc_pct = (
        pd.Series(uncertainty)
        .rolling(window=210, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )

    signals = np.zeros(n, dtype=np.int8)
    position = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    prev_trail_stop = 0.0  # Ratchet용
    warmup = 210

    trail_stops = 0
    kf_stops = 0

    for i in range(warmup, n):
        low_uncertainty = unc_pct[i] < 0.5

        if position == 0:
            if vel_zscore[i] > entry_z and low_uncertainty:
                position = 1
                highest_since_entry = high[i]
                # 초기 stop 설정
                prev_trail_stop = np.exp(np.log(highest_since_entry) - trail_mult * np.sqrt(uncertainty[i]))
                signals[i] = 1
            elif vel_zscore[i] < -entry_z and low_uncertainty:
                position = -1
                lowest_since_entry = low[i]
                # 초기 stop 설정
                prev_trail_stop = np.exp(np.log(lowest_since_entry) + trail_mult * np.sqrt(uncertainty[i]))
                signals[i] = -1

        elif position == 1:  # Long
            # Update highest
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]

            # Trailing stop 계산: exp(log(highest) - mult * sqrt(uncertainty))
            new_trail_stop = np.exp(np.log(highest_since_entry) - trail_mult * np.sqrt(uncertainty[i]))

            # Ratchet: Long은 stop이 절대 낮아지면 안됨 (max)
            trail_stop = max(new_trail_stop, prev_trail_stop)
            prev_trail_stop = trail_stop

            if close[i] < trail_stop:
                position = 0
                trail_stops += 1
                continue

            # KF residual stop (기존)
            if kf_stop_mult is not None and not np.isnan(resid_std[i]):
                if log_residuals[i] < -kf_stop_mult * resid_std[i]:
                    position = 0
                    kf_stops += 1
                    continue

            signals[i] = 1

        elif position == -1:  # Short
            # Update lowest
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]

            # Trailing stop 계산
            new_trail_stop = np.exp(np.log(lowest_since_entry) + trail_mult * np.sqrt(uncertainty[i]))

            # Ratchet: Short은 stop이 절대 높아지면 안됨 (min)
            trail_stop = min(new_trail_stop, prev_trail_stop)
            prev_trail_stop = trail_stop

            if close[i] > trail_stop:
                position = 0
                trail_stops += 1
                continue

            # KF residual stop
            if kf_stop_mult is not None and not np.isnan(resid_std[i]):
                if log_residuals[i] > kf_stop_mult * resid_std[i]:
                    position = 0
                    kf_stops += 1
                    continue

            signals[i] = -1

    return signals, trail_stops, kf_stops


def run_backtest(df, entry_z, trail_mult):
    df_kf = calculate_adaptive_kalman(df)
    signals, trail_stops, kf_stops = generate_signals_trailing(
        df_kf, entry_z=entry_z, trail_mult=trail_mult
    )

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

    metrics = run_pyramid_backtest(df_kf, bt_config)
    metrics["trail_stops"] = trail_stops
    metrics["kf_stops"] = kf_stops
    return metrics


def run_baseline(df, entry_z):
    """V2.2 style baseline with signal exit"""
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
            # KF stop
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


def main():
    print("Loading data...")
    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df

    # Test configurations: (name, entry_z, trail_mult)
    configs = [
        ("Baseline z=2.0 signal", 2.0, None),
        ("Baseline z=2.5 signal", 2.5, None),
        ("z=2.5 + trail(3)", 2.5, 3.0),
        ("z=2.5 + trail(5)", 2.5, 5.0),
        ("z=2.5 + trail(10)", 2.5, 10.0),
        ("z=2.0 + trail(5)", 2.0, 5.0),
    ]

    print("\n" + "=" * 130)
    print("Trailing Stop Test: sqrt(uncertainty) + Ratchet")
    print("=" * 130)

    for config_name, entry_z, trail_mult in configs:
        print(f"\n[{config_name}]")
        print(f"{'Symbol':<10} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'Trades':<8} {'WinRate':<10} {'TrailStop':<10} {'KFStop':<8}")
        print("-" * 90)

        results = []
        for symbol, df in data.items():
            if trail_mult is None:
                metrics = run_baseline(df, entry_z)
                metrics["trail_stops"] = 0
                metrics["kf_stops"] = 0
            else:
                metrics = run_backtest(df, entry_z, trail_mult)

            results.append({
                "symbol": symbol,
                "sharpe": metrics["sharpe_ratio"],
                "pnl": metrics["total_pnl"],
                "mdd": metrics["max_drawdown"],
                "trades": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
                "trail_stops": metrics["trail_stops"],
                "kf_stops": metrics["kf_stops"],
            })
            print(f"{symbol:<10} {metrics['sharpe_ratio']:<10.2f} {metrics['total_pnl']*100:<11.1f}% "
                  f"{metrics['max_drawdown']*100:<9.1f}% {metrics['total_trades']:<8} "
                  f"{metrics['win_rate']*100:<9.1f}% {metrics['trail_stops']:<10} {metrics['kf_stops']:<8}")

        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)
        btc = next(r for r in results if r["symbol"] == "BTCUSDT")

        print("-" * 90)
        print(f"Average: Sharpe={avg_sharpe:.2f}, PnL={avg_pnl*100:.1f}%, Positive={positive}/7")
        print(f"BTC: Sharpe={btc['sharpe']:.2f}, PnL={btc['pnl']*100:.1f}%")


if __name__ == "__main__":
    main()
