#!/usr/bin/env python3
"""
AKF V4 연도별 성과 분석
- 각 자산 equity curve를 result/ 폴더에 저장
- 일자별 PnL 합산으로 포트폴리오 계산
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


def run_backtests_and_save():
    """각 자산별 백테스트 실행 후 result/ 폴더에 저장"""
    RESULT_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print("1. 백테스트 실행 및 equity curve 저장")
    print("=" * 80)

    for symbol in PORTFOLIO_SYMBOLS:
        print(f"\n[{symbol}] 백테스트 실행...")
        df = load_data(symbol)
        result = run_backtest_go(df, V4_PARAMS)

        # timestamp -> datetime
        timestamps = df["start_time"].values
        dates = pd.to_datetime(timestamps, unit='ms')

        # equity curve
        equity = np.array(result["equity_curve"])

        # DataFrame 생성
        equity_df = pd.DataFrame({
            "datetime": dates,
            "date": dates.date,
            "equity": equity
        })

        # 일자별 마지막 값만 (EOD equity)
        daily_equity = equity_df.groupby("date").last().reset_index()
        daily_equity = daily_equity[["date", "equity"]]

        # 저장
        filepath = RESULT_DIR / f"{symbol}_equity.csv"
        daily_equity.to_csv(filepath, index=False)

        print(f"  저장: {filepath}")
        print(f"  기간: {daily_equity['date'].iloc[0]} ~ {daily_equity['date'].iloc[-1]}")
        print(f"  일수: {len(daily_equity)}")
        print(f"  최종 equity: ${equity[-1]:,.0f}")


def calculate_portfolio():
    """저장된 equity curve로 포트폴리오 계산 (일별 PnL 합산)"""
    print("\n" + "=" * 80)
    print("2. 포트폴리오 계산 (일별 PnL 합산)")
    print("=" * 80)

    # 각 자산 equity 로드
    asset_equity = {}
    for symbol in PORTFOLIO_SYMBOLS:
        filepath = RESULT_DIR / f"{symbol}_equity.csv"
        df = pd.read_csv(filepath, parse_dates=["date"])
        asset_equity[symbol] = df.set_index("date")["equity"]

    # 공통 날짜만 사용
    common_dates = asset_equity[PORTFOLIO_SYMBOLS[0]].index
    for symbol in PORTFOLIO_SYMBOLS[1:]:
        common_dates = common_dates.intersection(asset_equity[symbol].index)
    common_dates = common_dates.sort_values()

    print(f"\n공통 기간: {common_dates[0].date()} ~ {common_dates[-1].date()}")
    print(f"공통 일수: {len(common_dates)}")

    # 각 자산의 일별 PnL 계산
    daily_pnl = pd.DataFrame(index=common_dates)
    for symbol in PORTFOLIO_SYMBOLS:
        eq = asset_equity[symbol].loc[common_dates].values
        # 첫날은 PnL 0, 이후는 전일 대비 차이
        pnl = np.zeros(len(eq))
        pnl[1:] = eq[1:] - eq[:-1]
        daily_pnl[symbol] = pnl

    # 포트폴리오: 각 자산 $100K 시작, 일별 PnL 합산
    initial_capital = 100000 * len(PORTFOLIO_SYMBOLS)  # $400K
    daily_pnl["total_pnl"] = daily_pnl[PORTFOLIO_SYMBOLS].sum(axis=1)
    daily_pnl["portfolio_equity"] = initial_capital + daily_pnl["total_pnl"].cumsum()

    # 저장
    portfolio_path = RESULT_DIR / "portfolio_equity.csv"
    daily_pnl.to_csv(portfolio_path)
    print(f"\n포트폴리오 저장: {portfolio_path}")

    return daily_pnl


def analyze_yearly(daily_pnl: pd.DataFrame):
    """연도별 성과 분석 (공통 기간 기준)"""
    print("\n" + "=" * 80)
    print("3. 개별 자산 연도별 수익률 (%) - 공통 기간 기준")
    print("=" * 80)

    years = [2020, 2021, 2022, 2023, 2024, 2025]

    # 공통 기간 dates
    common_dates = daily_pnl.index

    # 각 자산 equity 로드 (공통 기간만)
    asset_equity = {}
    for symbol in PORTFOLIO_SYMBOLS:
        filepath = RESULT_DIR / f"{symbol}_equity.csv"
        df = pd.read_csv(filepath, parse_dates=["date"])
        eq = df.set_index("date")["equity"]
        asset_equity[symbol] = eq.loc[common_dates]

    # 헤더
    header = f"{'Year':<8}"
    for symbol in PORTFOLIO_SYMBOLS:
        header += f"{symbol:>12}"
    header += f"{'SUM':>12}"
    print(header)
    print("-" * 70)

    yearly_returns = {symbol: {} for symbol in PORTFOLIO_SYMBOLS}

    for year in years:
        row = f"{year:<8}"
        year_sum = 0
        valid_count = 0

        for symbol in PORTFOLIO_SYMBOLS:
            eq = asset_equity[symbol]
            year_mask = eq.index.year == year

            if year_mask.sum() < 10:
                row += f"{'N/A':>12}"
                continue

            year_eq = eq[year_mask]
            start_eq = year_eq.iloc[0]
            end_eq = year_eq.iloc[-1]
            ret = (end_eq - start_eq) / start_eq

            yearly_returns[symbol][year] = ret
            year_sum += ret
            valid_count += 1
            row += f"{ret*100:>11.1f}%"

        if valid_count > 0:
            row += f"{year_sum*100:>11.1f}%"
        print(row)

    # 개별 자산 연도별 MDD (공통 기간 기준)
    print("\n" + "=" * 80)
    print("4. 개별 자산 연도별 MDD (%) - 공통 기간 기준")
    print("=" * 80)

    header = f"{'Year':<8}"
    for symbol in PORTFOLIO_SYMBOLS:
        header += f"{symbol:>12}"
    print(header)
    print("-" * 60)

    for year in years:
        row = f"{year:<8}"
        for symbol in PORTFOLIO_SYMBOLS:
            eq = asset_equity[symbol]
            year_mask = eq.index.year == year

            if year_mask.sum() < 10:
                row += f"{'N/A':>12}"
                continue

            year_eq = eq[year_mask].values
            running_max = np.maximum.accumulate(year_eq)
            drawdown = (year_eq - running_max) / running_max
            mdd = abs(np.min(drawdown))

            row += f"{mdd*100:>11.1f}%"
        print(row)

    # 포트폴리오 연도별 성과
    print("\n" + "=" * 80)
    print("5. 포트폴리오 연도별 성과 (일별 PnL 합산)")
    print("=" * 80)

    print(f"\n{'Year':<8}{'Return':>12}{'MDD':>12}{'Ret/MDD':>12}")
    print("-" * 50)

    portfolio_eq = daily_pnl["portfolio_equity"]

    for year in years:
        year_mask = portfolio_eq.index.year == year
        if year_mask.sum() < 10:
            continue

        year_eq = portfolio_eq[year_mask]
        start_eq = year_eq.iloc[0]
        end_eq = year_eq.iloc[-1]
        ret = (end_eq - start_eq) / start_eq

        # MDD
        running_max = np.maximum.accumulate(year_eq.values)
        drawdown = (year_eq.values - running_max) / running_max
        mdd = abs(np.min(drawdown))

        ratio = ret / mdd if mdd > 0 else 0
        print(f"{year:<8}{ret*100:>11.1f}%{mdd*100:>11.1f}%{ratio:>12.2f}")

    print("-" * 50)

    # 전체 기간
    total_ret = (portfolio_eq.iloc[-1] - portfolio_eq.iloc[0]) / portfolio_eq.iloc[0]
    running_max = np.maximum.accumulate(portfolio_eq.values)
    total_mdd = abs(np.min((portfolio_eq.values - running_max) / running_max))

    print(f"{'TOTAL':<8}{total_ret*100:>11.1f}%{total_mdd*100:>11.1f}%{total_ret/total_mdd:>12.2f}")

    # 검증: 개별 PnL 합 = 포트폴리오 PnL
    print("\n" + "=" * 80)
    print("6. 검증: 개별 PnL 합 = 포트폴리오 PnL")
    print("=" * 80)

    print(f"\n{'Year':<8}{'Sum PnL':>14}{'Port PnL':>14}{'Match':>10}")
    print("-" * 50)

    for year in years:
        year_mask = portfolio_eq.index.year == year
        if year_mask.sum() < 10:
            continue

        # 개별 자산 PnL 합
        sum_pnl = 0
        for symbol in PORTFOLIO_SYMBOLS:
            eq = asset_equity[symbol]
            year_eq = eq[year_mask]
            pnl = year_eq.iloc[-1] - year_eq.iloc[0]
            sum_pnl += pnl

        # 포트폴리오 PnL
        year_eq = portfolio_eq[year_mask]
        port_pnl = year_eq.iloc[-1] - year_eq.iloc[0]

        match = "OK" if abs(sum_pnl - port_pnl) < 1 else "DIFF"
        print(f"{year:<8}${sum_pnl:>12,.0f}${port_pnl:>12,.0f}{match:>10}")


if __name__ == "__main__":
    run_backtests_and_save()
    daily_pnl = calculate_portfolio()
    analyze_yearly(daily_pnl)
