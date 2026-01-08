#!/usr/bin/env python3
"""
AKF V2.1 Strategy Runner.

최종 전략 실행 및 성과 분석:
- Buy & Hold 대비 비교
- 연도별 성과
- 월별 성과

Usage:
    python run_v21.py [--start-year 2020] [--end-year 2025]
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from research.features.kalman import calculate_adaptive_kalman
from strategies.akf_v2.common import SizingConfig
from strategies.akf_v2.common.sizing import calculate_position_sizes
from strategies.akf_v2.backtest_pyramid import run_pyramid_backtest, PyramidBacktestConfig

DATA_ROOT = PROJECT_ROOT / "etl/data"


# ============================================================
# V2.1 Parameters
# ============================================================
V21_PARAMS = {
    # Signal Generation
    "zscore_window": 42,        # 7일 (6 bars/day × 7)
    "uncertainty_window": 210,  # 30일
    "warmup": 210,
    "uncertainty_pct_threshold": 0.5,

    # Entry/Exit
    "long_entry": 2.0,
    "long_exit": -2.0,
    "short_entry": -2.0,
    "short_exit": 2.0,

    # Sizing
    "sizing_method": "fixed",
    "position_size": 1.0,

    # Backtest
    "initial_capital": 100000.0,
    "fee_rate": 0.001,
    "slippage_rate": 0.0001,
}


# ============================================================
# Data Loading
# ============================================================
def load_data(bar_size: int = 6, start_year: int = 2020, end_year: int = 2025) -> pd.DataFrame:
    """데이터 로드."""
    data_dir = DATA_ROOT / f"features-{bar_size}/futures/BTCUSDT"
    dfs = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            path = data_dir / f"BTCUSDT-features-{year}-{month:02d}.parquet"
            if path.exists():
                dfs.append(pd.read_parquet(path))
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["start_time"], unit="ms")
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    return df


# ============================================================
# Signal Generation (V2.1)
# ============================================================
def generate_signals_v21(df_kf: pd.DataFrame) -> np.ndarray:
    """V2.1 시그널 생성."""
    p = V21_PARAMS

    velocity = df_kf["kf_velocity"].values
    uncertainty = df_kf["kf_uncertainty"].values
    n = len(df_kf)

    # Velocity Z-score
    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=p["zscore_window"], min_periods=5).mean()
    vel_std = vel_series.rolling(window=p["zscore_window"], min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

    # Uncertainty percentile filter
    unc_pct = (
        pd.Series(uncertainty)
        .rolling(window=p["uncertainty_window"], min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )
    low_uncertainty = unc_pct < p["uncertainty_pct_threshold"]

    # Entry conditions
    entry_long = (vel_zscore > p["long_entry"]) & low_uncertainty
    entry_short = (vel_zscore < p["short_entry"]) & low_uncertainty

    # Signal generation
    signals = np.zeros(n, dtype=np.int8)
    position = 0

    for i in range(p["warmup"], n):
        if position == 0:
            if entry_long[i]:
                position = 1
                signals[i] = 1
            elif entry_short[i]:
                position = -1
                signals[i] = -1
        elif position == 1:
            if vel_zscore[i] < p["long_exit"]:
                position = 0
            else:
                signals[i] = 1
        elif position == -1:
            if vel_zscore[i] > p["short_exit"]:
                position = 0
            else:
                signals[i] = -1

    return signals


# ============================================================
# Backtest
# ============================================================
def run_strategy_backtest(df: pd.DataFrame) -> tuple:
    """전략 백테스트 실행. (df_result, metrics) 반환."""
    p = V21_PARAMS

    # Kalman Filter
    df_kf = calculate_adaptive_kalman(df)

    # Signals
    signals = generate_signals_v21(df_kf)
    df_kf["signal"] = signals

    # Sizing
    sizing_config = SizingConfig(
        method=p["sizing_method"],
        min_size=p["position_size"],
        max_size=p["position_size"],
    )
    sizes = calculate_position_sizes(df_kf, sizing_config, entry_signals=signals)
    df_kf["position_size"] = sizes
    df_kf["sl_price"] = 0.0

    # Backtest
    bt_config = PyramidBacktestConfig(
        initial_capital=p["initial_capital"],
        compounding=True,
        fee_rate=p["fee_rate"],
        slippage_rate=p["slippage_rate"],
    )

    metrics = run_pyramid_backtest(df_kf, bt_config)

    return df_kf, metrics


def calculate_buy_hold(df: pd.DataFrame) -> dict:
    """Buy & Hold 성과 계산."""
    start_price = df["close"].iloc[0]
    end_price = df["close"].iloc[-1]
    pnl = (end_price - start_price) / start_price

    # MDD
    prices = df["close"].values
    peak = prices[0]
    max_dd = 0
    for p in prices:
        if p > peak:
            peak = p
        dd = (peak - p) / peak
        if dd > max_dd:
            max_dd = dd

    # Sharpe (annualized)
    returns = df["close"].pct_change().dropna()
    if len(returns) > 0 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(365 * 6)
    else:
        sharpe = 0

    return {
        "total_pnl": pnl,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
    }


# ============================================================
# Period Analysis
# ============================================================
def analyze_by_period(df: pd.DataFrame, period: str = "year") -> pd.DataFrame:
    """기간별 성과 분석."""
    p = V21_PARAMS

    if period == "year":
        groups = df.groupby("year")
    else:  # month
        df["year_month"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
        groups = df.groupby("year_month")

    results = []

    for name, group_df in groups:
        if len(group_df) < p["warmup"]:
            continue

        # Strategy
        df_kf = calculate_adaptive_kalman(group_df.copy())
        signals = generate_signals_v21(df_kf)
        df_kf["signal"] = signals

        sizing_config = SizingConfig(
            method=p["sizing_method"],
            min_size=p["position_size"],
            max_size=p["position_size"],
        )
        sizes = calculate_position_sizes(df_kf, sizing_config, entry_signals=signals)
        df_kf["position_size"] = sizes
        df_kf["sl_price"] = 0.0

        bt_config = PyramidBacktestConfig(
            initial_capital=p["initial_capital"],
            compounding=True,
            fee_rate=p["fee_rate"],
            slippage_rate=p["slippage_rate"],
        )

        strat_metrics = run_pyramid_backtest(df_kf, bt_config)
        bh_metrics = calculate_buy_hold(group_df)

        results.append({
            "period": name,
            "bars": len(group_df),
            "strat_pnl": strat_metrics["total_pnl"],
            "strat_mdd": strat_metrics["max_drawdown"],
            "strat_sharpe": strat_metrics["sharpe_ratio"],
            "strat_trades": strat_metrics["total_trades"],
            "strat_winrate": strat_metrics["win_rate"],
            "bh_pnl": bh_metrics["total_pnl"],
            "bh_mdd": bh_metrics["max_drawdown"],
            "bh_sharpe": bh_metrics["sharpe_ratio"],
            "alpha": strat_metrics["total_pnl"] - bh_metrics["total_pnl"],
        })

    return pd.DataFrame(results)


# ============================================================
# Reporting
# ============================================================
def print_header():
    """헤더 출력."""
    print("=" * 100)
    print("AKF V2.1 Strategy Report")
    print("=" * 100)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("[Parameters]")
    for k, v in V21_PARAMS.items():
        print(f"  {k}: {v}")
    print()


def print_overall_comparison(strat_metrics: dict, bh_metrics: dict, df: pd.DataFrame):
    """전체 기간 비교."""
    print("=" * 100)
    print("[Overall Performance Comparison]")
    print("=" * 100)

    total_days = (df["datetime"].iloc[-1] - df["datetime"].iloc[0]).days
    total_years = total_days / 365

    print(f"Period: {df['datetime'].iloc[0].strftime('%Y-%m-%d')} ~ {df['datetime'].iloc[-1].strftime('%Y-%m-%d')} ({total_days} days)")
    print(f"Total Bars: {len(df):,}")
    print()

    print(f"{'Metric':<25} {'Strategy':<20} {'Buy & Hold':<20} {'Diff':<15}")
    print("-" * 80)
    print(f"{'Cumulative PnL':<25} {strat_metrics['total_pnl']*100:>15.1f}%    {bh_metrics['total_pnl']*100:>15.1f}%    {(strat_metrics['total_pnl']-bh_metrics['total_pnl'])*100:>+10.1f}%")
    print(f"{'Max Drawdown':<25} {strat_metrics['max_drawdown']*100:>15.1f}%    {bh_metrics['max_drawdown']*100:>15.1f}%    {(strat_metrics['max_drawdown']-bh_metrics['max_drawdown'])*100:>+10.1f}%")
    print(f"{'Sharpe Ratio':<25} {strat_metrics['sharpe_ratio']:>15.2f}     {bh_metrics['sharpe_ratio']:>15.2f}     {strat_metrics['sharpe_ratio']-bh_metrics['sharpe_ratio']:>+10.2f}")
    print(f"{'Total Trades':<25} {strat_metrics['total_trades']:>15}")
    print(f"{'Win Rate':<25} {strat_metrics['win_rate']*100:>15.1f}%")
    print(f"{'Avg Position Size':<25} {strat_metrics['avg_size']:>15.2f}x")
    print(f"{'Trades/Year':<25} {strat_metrics['total_trades']/total_years:>15.1f}")
    print()


def print_yearly_performance(yearly_df: pd.DataFrame):
    """연도별 성과."""
    print("=" * 100)
    print("[Yearly Performance]")
    print("=" * 100)

    print(f"{'Year':<8} {'Strat PnL':<12} {'B&H PnL':<12} {'Alpha':<12} {'Strat MDD':<12} {'B&H MDD':<12} {'Trades':<8} {'WinRate':<10}")
    print("-" * 100)

    for _, row in yearly_df.iterrows():
        print(f"{row['period']:<8} {row['strat_pnl']*100:>8.1f}%    {row['bh_pnl']*100:>8.1f}%    {row['alpha']*100:>+8.1f}%    "
              f"{row['strat_mdd']*100:>8.1f}%    {row['bh_mdd']*100:>8.1f}%    {row['strat_trades']:>6}  {row['strat_winrate']*100:>8.1f}%")

    print("-" * 100)

    # Summary
    avg_strat_pnl = yearly_df["strat_pnl"].mean()
    avg_bh_pnl = yearly_df["bh_pnl"].mean()
    avg_alpha = yearly_df["alpha"].mean()
    positive_alpha_years = (yearly_df["alpha"] > 0).sum()

    print(f"{'Avg':<8} {avg_strat_pnl*100:>8.1f}%    {avg_bh_pnl*100:>8.1f}%    {avg_alpha*100:>+8.1f}%")
    print(f"\nPositive Alpha Years: {positive_alpha_years}/{len(yearly_df)}")
    print()


def print_monthly_performance(monthly_df: pd.DataFrame):
    """월별 성과."""
    print("=" * 100)
    print("[Monthly Performance]")
    print("=" * 100)

    print(f"{'Month':<10} {'Strat PnL':<12} {'B&H PnL':<12} {'Alpha':<12} {'Trades':<8}")
    print("-" * 60)

    for _, row in monthly_df.iterrows():
        print(f"{row['period']:<10} {row['strat_pnl']*100:>8.1f}%    {row['bh_pnl']*100:>8.1f}%    {row['alpha']*100:>+8.1f}%    {row['strat_trades']:>6}")

    print("-" * 60)

    # Monthly statistics
    positive_months = (monthly_df["strat_pnl"] > 0).sum()
    positive_alpha_months = (monthly_df["alpha"] > 0).sum()
    avg_monthly_pnl = monthly_df["strat_pnl"].mean()

    print(f"\nPositive Months: {positive_months}/{len(monthly_df)} ({positive_months/len(monthly_df)*100:.1f}%)")
    print(f"Positive Alpha Months: {positive_alpha_months}/{len(monthly_df)} ({positive_alpha_months/len(monthly_df)*100:.1f}%)")
    print(f"Avg Monthly PnL: {avg_monthly_pnl*100:.2f}%")
    print()


def print_trade_analysis(df_kf: pd.DataFrame):
    """매매 분석."""
    print("=" * 100)
    print("[Trade Analysis]")
    print("=" * 100)

    signals = df_kf["signal"].values

    # Long/Short 분리
    long_entries = 0
    short_entries = 0
    long_bars = 0
    short_bars = 0

    prev_signal = 0
    for s in signals:
        if prev_signal == 0 and s == 1:
            long_entries += 1
        elif prev_signal == 0 and s == -1:
            short_entries += 1

        if s == 1:
            long_bars += 1
        elif s == -1:
            short_bars += 1

        prev_signal = s

    total_entries = long_entries + short_entries
    in_market_bars = long_bars + short_bars
    total_bars = len(signals)

    print(f"{'Metric':<30} {'Value':<20}")
    print("-" * 50)
    print(f"{'Long Entries':<30} {long_entries} ({long_entries/total_entries*100:.1f}%)")
    print(f"{'Short Entries':<30} {short_entries} ({short_entries/total_entries*100:.1f}%)")
    print(f"{'Long Holding (bars)':<30} {long_bars} ({long_bars/in_market_bars*100:.1f}%)")
    print(f"{'Short Holding (bars)':<30} {short_bars} ({short_bars/in_market_bars*100:.1f}%)")
    print(f"{'Market Exposure':<30} {in_market_bars/total_bars*100:.1f}%")
    print(f"{'Avg Holding (bars)':<30} {in_market_bars/total_entries:.1f} ({in_market_bars/total_entries/6:.1f} days)")
    print()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="AKF V2.1 Strategy Runner")
    parser.add_argument("--start-year", type=int, default=2020, help="시작 연도")
    parser.add_argument("--end-year", type=int, default=2025, help="종료 연도")
    parser.add_argument("--bar-size", type=int, default=6, help="Bar 크기")
    parser.add_argument("--no-monthly", action="store_true", help="월별 분석 생략")
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    df = load_data(args.bar_size, args.start_year, args.end_year)
    if df is None:
        print("No data found!")
        return

    print(f"Loaded {len(df):,} bars")
    print()

    # Run backtest
    print("Running backtest...")
    df_kf, strat_metrics = run_strategy_backtest(df)
    bh_metrics = calculate_buy_hold(df)
    print("Done!")
    print()

    # Analyze by period
    print("Analyzing by year...")
    yearly_df = analyze_by_period(df, "year")

    if not args.no_monthly:
        print("Analyzing by month...")
        monthly_df = analyze_by_period(df, "month")

    print()

    # Print report
    print_header()
    print_overall_comparison(strat_metrics, bh_metrics, df)
    print_yearly_performance(yearly_df)

    if not args.no_monthly:
        print_monthly_performance(monthly_df)

    print_trade_analysis(df_kf)

    print("=" * 100)
    print("Report Complete")
    print("=" * 100)


if __name__ == "__main__":
    main()
