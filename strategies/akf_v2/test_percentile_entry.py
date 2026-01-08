#!/usr/bin/env python3
"""
Percentile 기반 진입 테스트

- 최근 540 bars (90일)의 z-score 분포에서
- 상위 p% (롱) / 하위 p% (숏) 일 때만 진입
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


def generate_signals_percentile(
    df_kf,
    entry_pct=5,          # 상위 5% = 95th percentile
    pct_window=540,       # 90일
    stop_mult=3.0,
):
    """
    Percentile 기반 진입
    - Long: z > (100-entry_pct)th percentile
    - Short: z < entry_pct th percentile
    """
    velocity = df_kf["kf_velocity"].values
    uncertainty = df_kf["kf_uncertainty"].values
    close = df_kf["close"].values
    kf_trend = df_kf["kf_trend"].values
    n = len(df_kf)

    log_price = np.log(close)
    log_trend = np.log(kf_trend)
    log_residuals = log_price - log_trend
    resid_std = pd.Series(log_residuals).rolling(window=180, min_periods=30).std().values

    # Velocity z-score
    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=42, min_periods=5).mean()
    vel_std = vel_series.rolling(window=42, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

    # Rolling percentile thresholds
    vel_zscore_series = pd.Series(vel_zscore)
    upper_threshold = vel_zscore_series.rolling(window=pct_window, min_periods=100).quantile(1 - entry_pct/100).values
    lower_threshold = vel_zscore_series.rolling(window=pct_window, min_periods=100).quantile(entry_pct/100).values

    # Uncertainty filter
    unc_pct = (
        pd.Series(uncertainty)
        .rolling(window=210, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )

    signals = np.zeros(n, dtype=np.int8)
    position = 0
    warmup = max(540, 210)

    for i in range(warmup, n):
        low_uncertainty = unc_pct[i] < 0.5

        if position == 0:
            # Percentile 기반 진입
            if vel_zscore[i] > upper_threshold[i] and low_uncertainty:
                position = 1
                signals[i] = 1
            elif vel_zscore[i] < lower_threshold[i] and low_uncertainty:
                position = -1
                signals[i] = -1

        elif position == 1:
            # Stop
            if not np.isnan(resid_std[i]) and log_residuals[i] < -stop_mult * resid_std[i]:
                position = 0
                continue
            # Exit: 하위 percentile 도달
            if vel_zscore[i] < lower_threshold[i]:
                position = 0
            else:
                signals[i] = 1

        elif position == -1:
            if not np.isnan(resid_std[i]) and log_residuals[i] > stop_mult * resid_std[i]:
                position = 0
                continue
            # Exit: 상위 percentile 도달
            if vel_zscore[i] > upper_threshold[i]:
                position = 0
            else:
                signals[i] = -1

    return signals


def run_backtest(df, entry_pct):
    df_kf = calculate_adaptive_kalman(df)
    signals = generate_signals_percentile(df_kf, entry_pct=entry_pct)

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

    # Test different percentile values
    pct_values = [10, 5, 3, 2, 1]

    print("\n" + "=" * 120)
    print("Percentile-based Entry Test (window=540 bars = 90일)")
    print("=" * 120)

    for entry_pct in pct_values:
        print(f"\n[Entry: Top/Bottom {entry_pct}%]")
        print(f"{'Symbol':<10} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'Trades':<8} {'WinRate':<10}")
        print("-" * 60)

        results = []
        for symbol, df in data.items():
            metrics = run_backtest(df, entry_pct)
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

        # Summary
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)
        btc = next(r for r in results if r["symbol"] == "BTCUSDT")

        print("-" * 60)
        print(f"{'Average':<10} {avg_sharpe:<10.2f} {avg_pnl*100:<11.1f}%")
        print(f"Positive: {positive}/7, BTC: Sharpe={btc['sharpe']:.2f}, PnL={btc['pnl']*100:.1f}%")


if __name__ == "__main__":
    main()
