#!/usr/bin/env python3
"""
V5 전략 검증 스크립트

검증 항목:
1. 미래참조 (Look-ahead Bias) - velocity 계산 시점 vs 진입 시점
2. 수수료/슬리피지 설정 확인
3. 연도별 성과 분석 (과적합 체크)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from strategies.v5.strategy_v5 import load_data, run_backtest_go, V5_PARAMS, PORTFOLIO_SYMBOLS
from go_bridge import BacktestConfig


def check_lookahead():
    """미래참조 점검"""
    print("=" * 80)
    print("1. 미래참조 (Look-ahead Bias) 점검")
    print("=" * 80)
    print("""
  ⚠️ 의심 포인트:

  칼만 필터:
  - velocity[t]는 close[t]를 사용해서 계산됨
  - 즉, t봉이 끝나야 velocity[t]를 알 수 있음

  진입 로직:
  - signals[i] = 1 설정 시점: i봉의 close 기준
  - 실제 진입 시점: ???

  Go 백테스터 동작:
  - signal[i] = 1 → i봉의 close에서 진입? 다음 봉 open에서 진입?

  미래참조 발생 조건:
  - signal[i] = 1일 때 i봉의 open 또는 close에서 진입하면 미래참조!
  - 정상: signal[i] = 1 → (i+1)봉의 open에서 진입
""")


def check_fees():
    """수수료/슬리피지 설정 확인"""
    print("\n" + "=" * 80)
    print("2. 수수료/슬리피지 설정")
    print("=" * 80)

    # 현재 설정
    base_fee = 0.001       # 0.1%
    base_slippage = 0.0001 # 0.01%

    print(f"  현재 설정:")
    print(f"    base_fee:      {base_fee * 100:.2f}%")
    print(f"    base_slippage: {base_slippage * 100:.3f}%")
    print(f"    편도 비용:     {(base_fee + base_slippage) * 100:.3f}%")
    print(f"    왕복 비용:     {(base_fee + base_slippage) * 2 * 100:.3f}%")

    # 거래 횟수별 총 비용
    print(f"\n  거래 횟수별 총 비용 (왕복):")
    for trades in [500, 1000, 1500, 2000]:
        total_cost = trades * (base_fee + base_slippage) * 2 * 100
        print(f"    {trades:,} trades: ~{total_cost:.0f}%")

    print(f"\n  ⚠️ 현실적인 비용 (Taker + 슬리피지):")
    realistic_cost = 0.0007  # 0.07%
    print(f"    편도: ~0.07%")
    print(f"    2000 trades 왕복: ~{2000 * realistic_cost * 2 * 100:.0f}%")


def check_yearly_performance():
    """연도별 성과 분석"""
    print("\n" + "=" * 80)
    print("3. 연도별 성과 분석 (과적합 체크)")
    print("=" * 80)

    # 백테스트 실행
    results = {}
    for symbol in PORTFOLIO_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            print(f"  [{symbol}] 백테스트 실행...")
            results[symbol] = run_backtest_go(df)

    # 개별 자산 연도별 성과
    print("\n" + "-" * 80)
    print("개별 자산 연도별 수익률:")
    print("-" * 80)

    for symbol, r in results.items():
        equity = r["equity_curve"]
        timestamps = r["timestamps"]

        dates = pd.to_datetime(timestamps, unit='ms')
        df = pd.DataFrame({"date": dates, "equity": equity})
        df["year"] = df["date"].dt.year

        print(f"\n{symbol} (총 {r['total_trades']} trades):")
        yearly = df.groupby("year")["equity"].agg(["first", "last"])
        yearly["return"] = (yearly["last"] / yearly["first"] - 1) * 100

        for year in sorted(yearly.index):
            if year >= 2020:
                year_df = df[df["year"] == year]
                running_max = year_df["equity"].cummax()
                dd = (year_df["equity"] - running_max) / running_max
                mdd = abs(dd.min()) * 100
                ret = yearly.loc[year, "return"]
                print(f"  {year}: {ret:+8.1f}%  (MDD: {mdd:5.1f}%)")

    # 포트폴리오 연도별 성과
    print("\n" + "-" * 80)
    print("포트폴리오 연도별 성과:")
    print("-" * 80)

    asset_daily = {}
    for symbol, r in results.items():
        timestamps = r["timestamps"]
        equity = r["equity_curve"]
        dates = pd.to_datetime(timestamps, unit='ms')
        df = pd.DataFrame({"datetime": dates, "equity": equity})
        df["date"] = df["datetime"].dt.date
        daily = df.groupby("date")["equity"].last()
        asset_daily[symbol] = daily

    # Union dates
    all_dates = asset_daily[PORTFOLIO_SYMBOLS[0]].index
    for s in PORTFOLIO_SYMBOLS[1:]:
        all_dates = all_dates.union(asset_daily[s].index)
    all_dates = sorted(all_dates)

    for s in PORTFOLIO_SYMBOLS:
        asset_daily[s] = asset_daily[s].reindex(all_dates).ffill()
        if pd.isna(asset_daily[s].iloc[0]):
            asset_daily[s].iloc[0] = 100000.0
            asset_daily[s] = asset_daily[s].ffill()

    initial = 100000.0 * len(PORTFOLIO_SYMBOLS)
    portfolio = np.zeros(len(all_dates))
    portfolio[0] = initial
    for i in range(1, len(all_dates)):
        prev_date = all_dates[i-1]
        curr_date = all_dates[i]
        pnl = sum(asset_daily[s].loc[curr_date] - asset_daily[s].loc[prev_date] for s in PORTFOLIO_SYMBOLS)
        portfolio[i] = portfolio[i-1] + pnl

    df_port = pd.DataFrame({"date": all_dates, "equity": portfolio})
    df_port["date"] = pd.to_datetime(df_port["date"])
    df_port["year"] = df_port["date"].dt.year

    print("\nYear       Return      MDD        Ret/MDD")
    print("-" * 50)

    for year in sorted(df_port["year"].unique()):
        if year < 2020:
            continue
        year_df = df_port[df_port["year"] == year]
        start_eq = year_df["equity"].iloc[0]
        end_eq = year_df["equity"].iloc[-1]
        ret = (end_eq / start_eq - 1) * 100

        running_max = year_df["equity"].cummax()
        dd = (year_df["equity"] - running_max) / running_max
        mdd = abs(dd.min()) * 100

        ret_mdd = ret / mdd if mdd > 0 else 0
        print(f"{year}       {ret:+8.1f}%    {mdd:5.1f}%     {ret_mdd:6.2f}")

    # 전체 기간
    start_eq = df_port["equity"].iloc[0]
    end_eq = df_port["equity"].iloc[-1]
    total_ret = (end_eq / start_eq - 1) * 100
    running_max = df_port["equity"].cummax()
    dd = (df_port["equity"] - running_max) / running_max
    total_mdd = abs(dd.min()) * 100
    print("-" * 50)
    print(f"TOTAL      {total_ret:+8.1f}%    {total_mdd:5.1f}%     {total_ret/total_mdd:.2f}")

    # 과적합 체크
    print("\n" + "-" * 80)
    print("과적합 체크:")
    print("-" * 80)
    print("""
  건강한 전략:
  - 모든 연도에 고루 수익
  - 횡보장(2023)에서도 플러스 또는 소폭 마이너스

  과적합 전략:
  - 특정 연도(2021 상승장, 2022 하락장)에만 폭발적 수익
  - 횡보장(2023)에서 큰 손실
""")


def main():
    check_lookahead()
    check_fees()
    check_yearly_performance()

    print("\n" + "=" * 80)
    print("결론")
    print("=" * 80)


if __name__ == "__main__":
    main()
