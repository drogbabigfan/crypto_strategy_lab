#!/usr/bin/env python3
"""
Binance Cross Margin 포트폴리오 Equity Curve 시뮬레이션

알고리즘:
1. Time Synchronization: Outer Join으로 마스터 타임라인 생성
2. Handling Gaps: Forward Fill + 시작 전 NaN은 Initial Capital
3. PnL Calculation: diff()로 변동 손익금 계산
4. Aggregation: 같은 타임스탬프의 PnL 합산
5. Reconstruction: cumsum()으로 Total Equity Curve 완성
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "research/wfa"))

import numpy as np
import pandas as pd
from strategies.v4.strategy_v35_go import load_data, run_backtest_go, V4_PARAMS, PORTFOLIO_SYMBOLS

RESULT_DIR = Path(__file__).parent / "result"
INITIAL_CAPITAL = 100000.0


def run_backtests_bar_level():
    """각 자산 백테스트 실행 → Bar 단위 Equity Curve 저장 (timestamp 포함)"""
    RESULT_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print("1. 개별 자산 백테스트 (Bar 단위)")
    print("=" * 80)

    for symbol in PORTFOLIO_SYMBOLS:
        print(f"\n[{symbol}]")
        df = load_data(symbol)
        result = run_backtest_go(df, V4_PARAMS)

        # timestamp와 equity
        timestamps = result["timestamps"]
        equity = result["equity_curve"]

        # DataFrame 생성 (bar 단위)
        bar_df = pd.DataFrame({
            "timestamp": pd.to_datetime(timestamps, unit="ms"),
            "equity": equity
        })

        # 저장
        filepath = RESULT_DIR / f"{symbol}_bar_equity.csv"
        bar_df.to_csv(filepath, index=False)

        print(f"  Bars: {len(bar_df)}")
        print(f"  Period: {bar_df['timestamp'].iloc[0]} ~ {bar_df['timestamp'].iloc[-1]}")
        print(f"  Equity: ${equity[0]:,.0f} → ${equity[-1]:,.0f}")
        print(f"  Saved: {filepath}")


def calculate_portfolio_equity():
    """
    4개 자산의 비동기 Equity Curve를 통합하여
    단일 계좌의 Total Portfolio Equity Curve 생성
    """
    print("\n" + "=" * 80)
    print("2. 포트폴리오 Equity Curve 생성")
    print("=" * 80)

    # Step 1: 각 자산 Equity 로드
    asset_dfs = {}
    for symbol in PORTFOLIO_SYMBOLS:
        filepath = RESULT_DIR / f"{symbol}_bar_equity.csv"
        df = pd.read_csv(filepath, parse_dates=["timestamp"])
        df = df.set_index("timestamp")
        df = df.rename(columns={"equity": symbol})
        asset_dfs[symbol] = df
        print(f"\n[{symbol}] Loaded: {len(df)} bars")

    # Step 2: Outer Join으로 마스터 타임라인 생성
    print("\n[Outer Join] 마스터 타임라인 생성...")
    master_df = asset_dfs[PORTFOLIO_SYMBOLS[0]]
    for symbol in PORTFOLIO_SYMBOLS[1:]:
        master_df = master_df.join(asset_dfs[symbol], how="outer")

    master_df = master_df.sort_index()
    print(f"  Total timestamps: {len(master_df)}")
    print(f"  Period: {master_df.index[0]} ~ {master_df.index[-1]}")

    # Step 3: Forward Fill + 시작 전 NaN은 Initial Capital
    print("\n[Forward Fill] 결측치 처리...")
    for symbol in PORTFOLIO_SYMBOLS:
        # 첫 유효값 이전의 NaN → Initial Capital
        first_valid_idx = master_df[symbol].first_valid_index()
        if first_valid_idx is not None:
            master_df.loc[:first_valid_idx, symbol] = master_df.loc[:first_valid_idx, symbol].fillna(INITIAL_CAPITAL)

        # 이후 NaN → Forward Fill
        master_df[symbol] = master_df[symbol].ffill()

        nan_count = master_df[symbol].isna().sum()
        print(f"  {symbol}: NaN remaining = {nan_count}")

    # Step 4: diff()로 변동 손익금(Dollar PnL) 계산
    print("\n[Diff] 변동 손익금 계산...")
    pnl_df = pd.DataFrame(index=master_df.index)
    for symbol in PORTFOLIO_SYMBOLS:
        pnl_df[f"{symbol}_pnl"] = master_df[symbol].diff().fillna(0)

    # Step 5: 같은 타임스탬프의 PnL 합산
    print("\n[Sum] PnL 합산...")
    pnl_df["total_pnl"] = pnl_df[[f"{s}_pnl" for s in PORTFOLIO_SYMBOLS]].sum(axis=1)

    # Step 6: cumsum()으로 Total Equity Curve 완성
    print("\n[Cumsum] Total Equity Curve 생성...")
    pnl_df["total_equity"] = INITIAL_CAPITAL + pnl_df["total_pnl"].cumsum()

    # 결과 저장
    result_df = master_df.copy()
    result_df["total_pnl"] = pnl_df["total_pnl"]
    result_df["total_equity"] = pnl_df["total_equity"]

    filepath = RESULT_DIR / "portfolio_total_equity.csv"
    result_df.to_csv(filepath)
    print(f"\n저장: {filepath}")

    return result_df


def analyze_performance(df: pd.DataFrame):
    """성과 분석"""
    print("\n" + "=" * 80)
    print("3. 포트폴리오 성과 분석")
    print("=" * 80)

    equity = df["total_equity"].values
    initial = INITIAL_CAPITAL
    final = equity[-1]

    # 기간 계산
    start_date = df.index[0]
    end_date = df.index[-1]
    n_days = (end_date - start_date).days
    n_years = n_days / 365.0

    # 성과 지표
    total_return = (final - initial) / initial
    cagr = (final / initial) ** (1 / n_years) - 1 if n_years > 0 else 0

    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    mdd = abs(np.min(drawdown))

    # 일별 수익률로 Sharpe 계산
    daily_df = df["total_equity"].resample("D").last().dropna()
    daily_returns = daily_df.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 1 else 0

    print(f"\n기간: {start_date.date()} ~ {end_date.date()} ({n_years:.2f}년)")
    print(f"초기 자본: ${initial:,.0f}")
    print(f"최종 자본: ${final:,.0f}")
    print(f"\nTotal Return: {total_return*100:.1f}%")
    print(f"CAGR: {cagr*100:.1f}%")
    print(f"MDD: {mdd*100:.1f}%")
    print(f"CAGR/MDD: {cagr/mdd:.2f}" if mdd > 0 else "CAGR/MDD: N/A")
    print(f"Sharpe: {sharpe:.2f}")

    # 연도별 성과
    print("\n[연도별 성과]")
    print(f"{'Year':<8}{'Return':>12}{'MDD':>12}")
    print("-" * 35)

    for year in range(start_date.year, end_date.year + 1):
        year_mask = df.index.year == year
        if year_mask.sum() < 10:
            continue

        year_equity = df.loc[year_mask, "total_equity"].values
        year_start = year_equity[0]
        year_end = year_equity[-1]
        year_return = (year_end - year_start) / year_start

        year_running_max = np.maximum.accumulate(year_equity)
        year_dd = (year_equity - year_running_max) / year_running_max
        year_mdd = abs(np.min(year_dd))

        print(f"{year:<8}{year_return*100:>11.1f}%{year_mdd*100:>11.1f}%")

    return {
        "total_return": total_return,
        "cagr": cagr,
        "mdd": mdd,
        "sharpe": sharpe,
    }


if __name__ == "__main__":
    # 1. 백테스트 실행 및 Bar 단위 Equity 저장
    run_backtests_bar_level()

    # 2. 포트폴리오 Equity Curve 생성
    portfolio_df = calculate_portfolio_equity()

    # 3. 성과 분석
    metrics = analyze_performance(portfolio_df)
