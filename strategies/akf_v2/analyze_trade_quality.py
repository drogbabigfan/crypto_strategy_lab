#!/usr/bin/env python3
"""
트레이드 품질 분석: 왜 같은 신호가 BTC에서만 잘 작동하는가?

1. 롱/숏 별 승률, 평균수익
2. 신호 강도별 성과 (z=2.0~2.5 vs z>2.5)
3. 포지션 보유 기간별 성과
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from research.features.kalman import calculate_adaptive_kalman

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


def analyze_trades(symbol: str, df: pd.DataFrame):
    """트레이드별 상세 분석"""
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

    # 트레이드 추적
    trades = []
    position = 0
    entry_idx = 0
    entry_price = 0
    entry_zscore = 0
    warmup = 210
    stop_mult = 3.0

    for i in range(warmup, n):
        low_uncertainty = unc_pct[i] < 0.5

        if position == 0:
            if vel_zscore[i] > 2.0 and low_uncertainty:
                position = 1
                entry_idx = i
                entry_price = close[i]
                entry_zscore = vel_zscore[i]
            elif vel_zscore[i] < -2.0 and low_uncertainty:
                position = -1
                entry_idx = i
                entry_price = close[i]
                entry_zscore = vel_zscore[i]

        elif position == 1:
            exit_reason = None
            # Stop check
            if not np.isnan(resid_std[i]) and log_residuals[i] < -stop_mult * resid_std[i]:
                exit_reason = "stop"
            elif vel_zscore[i] < -2.0:
                exit_reason = "signal"

            if exit_reason:
                pnl = (close[i] - entry_price) / entry_price
                trades.append({
                    "direction": "long",
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "duration": i - entry_idx,
                    "entry_zscore": entry_zscore,
                    "pnl": pnl,
                    "exit_reason": exit_reason,
                })
                position = 0

        elif position == -1:
            exit_reason = None
            if not np.isnan(resid_std[i]) and log_residuals[i] > stop_mult * resid_std[i]:
                exit_reason = "stop"
            elif vel_zscore[i] > 2.0:
                exit_reason = "signal"

            if exit_reason:
                pnl = (entry_price - close[i]) / entry_price
                trades.append({
                    "direction": "short",
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "duration": i - entry_idx,
                    "entry_zscore": entry_zscore,
                    "pnl": pnl,
                    "exit_reason": exit_reason,
                })
                position = 0

    return trades


def summarize_trades(symbol: str, trades: list):
    """트레이드 요약 통계"""
    if not trades:
        return None

    df_trades = pd.DataFrame(trades)

    # 전체
    total = len(df_trades)
    win_rate = (df_trades["pnl"] > 0).mean()
    avg_pnl = df_trades["pnl"].mean()
    avg_duration = df_trades["duration"].mean()

    # 롱/숏 분리
    longs = df_trades[df_trades["direction"] == "long"]
    shorts = df_trades[df_trades["direction"] == "short"]

    long_win = (longs["pnl"] > 0).mean() if len(longs) > 0 else 0
    short_win = (shorts["pnl"] > 0).mean() if len(shorts) > 0 else 0
    long_avg = longs["pnl"].mean() if len(longs) > 0 else 0
    short_avg = shorts["pnl"].mean() if len(shorts) > 0 else 0

    # 신호 강도별 (|z| > 2.5 vs 2.0~2.5)
    strong = df_trades[df_trades["entry_zscore"].abs() > 2.5]
    weak = df_trades[df_trades["entry_zscore"].abs() <= 2.5]

    strong_win = (strong["pnl"] > 0).mean() if len(strong) > 0 else 0
    weak_win = (weak["pnl"] > 0).mean() if len(weak) > 0 else 0
    strong_avg = strong["pnl"].mean() if len(strong) > 0 else 0
    weak_avg = weak["pnl"].mean() if len(weak) > 0 else 0

    # Exit reason
    stop_exits = df_trades[df_trades["exit_reason"] == "stop"]
    signal_exits = df_trades[df_trades["exit_reason"] == "signal"]

    stop_avg = stop_exits["pnl"].mean() if len(stop_exits) > 0 else 0
    signal_avg = signal_exits["pnl"].mean() if len(signal_exits) > 0 else 0

    return {
        "symbol": symbol,
        "total_trades": total,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "avg_duration": avg_duration,
        "long_count": len(longs),
        "long_win": long_win,
        "long_avg": long_avg,
        "short_count": len(shorts),
        "short_win": short_win,
        "short_avg": short_avg,
        "strong_count": len(strong),
        "strong_win": strong_win,
        "strong_avg": strong_avg,
        "weak_count": len(weak),
        "weak_win": weak_win,
        "weak_avg": weak_avg,
        "stop_count": len(stop_exits),
        "stop_avg": stop_avg,
        "signal_count": len(signal_exits),
        "signal_avg": signal_avg,
    }


def main():
    print("=" * 110)
    print("Trade Quality Analysis")
    print("=" * 110)

    all_stats = []

    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is None:
            continue
        trades = analyze_trades(symbol, df)
        stats = summarize_trades(symbol, trades)
        if stats:
            all_stats.append(stats)

    # 1. 롱/숏 비교
    print(f"\n[1] Long vs Short Performance")
    print(f"{'Symbol':<10} {'Long#':<8} {'LongWin':<10} {'LongAvg':<12} {'Short#':<8} {'ShortWin':<10} {'ShortAvg':<12}")
    print("-" * 80)
    for s in all_stats:
        print(f"{s['symbol']:<10} {s['long_count']:<8} {s['long_win']*100:<9.1f}% {s['long_avg']*100:<11.2f}% "
              f"{s['short_count']:<8} {s['short_win']*100:<9.1f}% {s['short_avg']*100:<11.2f}%")

    # 2. 신호 강도 비교
    print(f"\n[2] Signal Strength (|z|>2.5 vs |z|<=2.5)")
    print(f"{'Symbol':<10} {'Strong#':<8} {'StrongWin':<10} {'StrongAvg':<12} {'Weak#':<8} {'WeakWin':<10} {'WeakAvg':<12}")
    print("-" * 80)
    for s in all_stats:
        print(f"{s['symbol']:<10} {s['strong_count']:<8} {s['strong_win']*100:<9.1f}% {s['strong_avg']*100:<11.2f}% "
              f"{s['weak_count']:<8} {s['weak_win']*100:<9.1f}% {s['weak_avg']*100:<11.2f}%")

    # 3. Exit reason 비교
    print(f"\n[3] Exit Reason (Stop vs Signal)")
    print(f"{'Symbol':<10} {'Stop#':<8} {'StopAvg':<12} {'Signal#':<8} {'SignalAvg':<12}")
    print("-" * 55)
    for s in all_stats:
        print(f"{s['symbol']:<10} {s['stop_count']:<8} {s['stop_avg']*100:<11.2f}% "
              f"{s['signal_count']:<8} {s['signal_avg']*100:<11.2f}%")

    # 4. 전체 요약
    print(f"\n[4] Overall Summary")
    print(f"{'Symbol':<10} {'Trades':<8} {'WinRate':<10} {'AvgPnL':<12} {'AvgDuration':<12}")
    print("-" * 55)
    for s in all_stats:
        print(f"{s['symbol']:<10} {s['total_trades']:<8} {s['win_rate']*100:<9.1f}% "
              f"{s['avg_pnl']*100:<11.2f}% {s['avg_duration']:<11.1f} bars")


if __name__ == "__main__":
    main()
