#!/usr/bin/env python3
"""
4-Asset Combined Portfolio 백테스트 V2

기존 backtest_pyramid 사용하여 정확한 결과 도출
레버리지 누적 방식: 각 자산의 수익률을 합산
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from strategies.akf_v2.test_duration_filter import run_backtest_intensity
from strategies.akf_v2.strategy_v33_final import load_data

PORTFOLIO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"]


def main():
    print("=" * 100)
    print("4-Asset Combined Portfolio (레버리지 누적)")
    print("=" * 100)

    initial_capital = 100000.0
    intensity_threshold = 4.0

    # 개별 자산 결과
    print("\n[1] 개별 자산 백테스트...")
    individual_results = {}

    for symbol in PORTFOLIO_SYMBOLS:
        df = load_data(symbol)
        if df is None:
            continue

        m = run_backtest_intensity(df, intensity_threshold)
        equity_curve = m["equity_curve"]

        # equity_curve에서 bar별 수익률 계산
        bar_returns = np.zeros(len(equity_curve))
        bar_returns[1:] = np.diff(equity_curve) / equity_curve[:-1]

        individual_results[symbol] = {
            "metrics": m,
            "equity_curve": equity_curve,
            "bar_returns": bar_returns,
        }

        print(f"  {symbol}: PnL={m['total_pnl']*100:.1f}%, Sharpe={m['sharpe_ratio']:.2f}, MDD={m['max_drawdown']*100:.1f}%")

    # 레버리지 누적 합산
    print("\n[2] 레버리지 누적 합산...")

    # 모든 자산의 길이 맞추기 (가장 짧은 것 기준)
    min_len = min(len(r["bar_returns"]) for r in individual_results.values())

    # 각 자산의 수익률 합산 (레버리지 누적)
    combined_returns = np.zeros(min_len)
    for symbol, r in individual_results.items():
        combined_returns += r["bar_returns"][:min_len]

    # Combined Equity Curve (복리)
    combined_equity = np.zeros(min_len)
    combined_equity[0] = initial_capital

    for i in range(1, min_len):
        combined_equity[i] = combined_equity[i-1] * (1 + combined_returns[i])

    # Metrics 계산
    total_pnl = (combined_equity[-1] - initial_capital) / initial_capital

    # MDD
    running_max = np.maximum.accumulate(combined_equity)
    drawdown = (combined_equity - running_max) / running_max
    max_dd = np.abs(np.min(drawdown))

    # Sharpe (일별 기준, 6 bars/day)
    daily_returns = []
    for i in range(0, len(combined_returns), 6):
        chunk = combined_returns[i:i+6]
        if len(chunk) > 0:
            daily_returns.append(np.sum(chunk))

    if len(daily_returns) > 1:
        mean_daily = np.mean(daily_returns)
        std_daily = np.std(daily_returns)
        sharpe = (mean_daily / (std_daily + 1e-10)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # CAGR
    n_years = min_len / (6 * 252)
    if n_years > 0 and combined_equity[-1] > 0:
        cagr = (combined_equity[-1] / initial_capital) ** (1 / n_years) - 1
    else:
        cagr = 0.0

    print("\n[3] Combined Portfolio 성과:")
    print("=" * 60)
    print(f"  Total PnL:    {total_pnl*100:.1f}%")
    print(f"  CAGR:         {cagr*100:.1f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd*100:.1f}%")
    print(f"  Final Equity: ${combined_equity[-1]:,.0f}")

    # 개별 vs 합산 비교
    print("\n[4] 개별 합산 vs 레버리지 누적 비교:")
    print("=" * 60)
    simple_sum_pnl = sum(r["metrics"]["total_pnl"] for r in individual_results.values())
    print(f"  개별 PnL 단순 합: {simple_sum_pnl*100:.1f}%")
    print(f"  레버리지 누적 PnL: {total_pnl*100:.1f}%")

    # 개별 자산 요약
    print("\n[5] 개별 자산 성과 요약:")
    print("=" * 60)
    print(f"{'Symbol':<10} {'PnL':<12} {'Sharpe':<10} {'MDD':<10}")
    print("-" * 45)
    for symbol, r in individual_results.items():
        m = r["metrics"]
        print(f"{symbol:<10} {m['total_pnl']*100:<11.1f}% {m['sharpe_ratio']:<10.2f} {m['max_drawdown']*100:<9.1f}%")

    # 평균 레버리지 추정
    # position_size의 평균값으로 추정
    print("\n[6] 레버리지 추정:")
    print("=" * 60)
    total_avg_size = 0
    for symbol, r in individual_results.items():
        avg_size = r["metrics"].get("avg_size", 1.0)
        total_avg_size += avg_size
        print(f"  {symbol} avg size: {avg_size:.2f}x")
    print(f"  합산 평균 레버리지: {total_avg_size:.2f}x")


if __name__ == "__main__":
    main()
