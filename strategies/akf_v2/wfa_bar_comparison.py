#!/usr/bin/env python3
"""
Bar 크기별 Walk-Forward Analysis 비교 스크립트.

일수 기반 파라미터로 자동 스케일링.

Usage:
    python wfa_bar_comparison.py [--bar-sizes 6 24 50 100] [--min-size 0] [--max-size 2]
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import pandas as pd

from research.features.kalman import calculate_adaptive_kalman
from strategies.akf_v2.common import PyramidConfig, SizingConfig
from strategies.akf_v2.wfa import WFAConfig, WFAEngine, WFAResult, generate_report

DATA_ROOT = PROJECT_ROOT / "etl/data"


def load_data(bar_size: int, start_year: int = 2020, end_year: int = 2025) -> pd.DataFrame:
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
    return pd.concat(dfs, ignore_index=True)


def run_wfa(
    df_kf: pd.DataFrame,
    sizing_method: str = "kf_prob",
    min_size: float = 0.0,
    max_size: float = 2.0,
) -> WFAResult:
    """WFA 실행 (일수 기반 자동 스케일링)."""
    # 사이징 설정
    sizing_config = SizingConfig(
        method=sizing_method,
        min_size=min_size,
        max_size=max_size,
    )

    # 피라미딩 비활성화
    pyramid_config = PyramidConfig(enabled=False)

    # WFA 설정 (일수 기반)
    wfa_config = WFAConfig(
        train_days=365,   # 1년
        test_days=90,     # 3개월
        step_days=90,     # 3개월
        zscore_days=5.0,        # 1주 (거래일)
        uncertainty_days=20.0,  # 1개월 (거래일)
        warmup_days=20.0,       # 1개월
        entry_zscore_range=(0.5, 1.0, 1.5, 2.0),
        exit_zscore_range=(-0.5, -1.0, -1.5, -2.0),
        sizing_config=sizing_config,
        pyramid_config=pyramid_config,
    )

    engine = WFAEngine(wfa_config, use_python_backtest=True)
    folds = engine.run(df_kf)

    return WFAResult(folds=folds)


def main():
    parser = argparse.ArgumentParser(description="Bar 크기별 WFA 비교")
    parser.add_argument("--min-size", type=float, default=0.0, help="최소 포지션 사이즈")
    parser.add_argument("--max-size", type=float, default=2.0, help="최대 포지션 사이즈")
    parser.add_argument(
        "--method",
        type=str,
        default="kf_prob",
        choices=["fixed", "kelly", "kf_prob"],
        help="사이징 방식",
    )
    parser.add_argument(
        "--bar-sizes",
        type=int,
        nargs="+",
        default=[6, 24, 50, 100],
        help="테스트할 bar 크기들",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Bar 크기별 Walk-Forward Analysis (일수 기반 자동 스케일링)")
    print(f"  Method: {args.method}, Size: [{args.min_size}, {args.max_size}]")
    print(f"  zscore: 5일, uncertainty: 20일, warmup: 20일")
    print("=" * 80)

    all_results = {}

    for bar_size in args.bar_sizes:
        print(f"\n{'='*80}")
        print(f"[Bar-{bar_size}] WFA 시작")
        print("=" * 80)

        df = load_data(bar_size)
        if df is None:
            print(f"  데이터 없음")
            continue

        print(f"  {len(df):,} bars 로드됨")

        # Kalman Filter
        print("  Kalman Filter 적용 중...")
        df_kf = calculate_adaptive_kalman(df)

        # WFA 실행
        result = run_wfa(df_kf, args.method, args.min_size, args.max_size)
        all_results[bar_size] = result

        if result.folds:
            summary = result.summary
            print(f"\n  [Bar-{bar_size} 요약]")
            print(f"    Folds: {len(result.folds)}")
            print(f"    Avg Test Sharpe: {summary['avg_test_sharpe']:.2f}")
            print(f"    Cumulative PnL: {summary['cumulative_pnl']*100:.1f}%")
            print(f"    Avg MDD: {summary['avg_mdd']*100:.1f}%")
            print(f"    Overfit Ratio: {summary['overfit_ratio']:.2f}")

    # 최종 비교 테이블
    print("\n" + "=" * 80)
    print("Bar 크기별 WFA 결과 비교")
    print("=" * 80)
    print(
        f"{'Bar':<8} {'Folds':<8} {'AvgSharpe':<12} {'CumPnL':<12} "
        f"{'AvgMDD':<10} {'Overfit':<10} {'AvgTrades':<10}"
    )
    print("-" * 80)

    for bar_size in args.bar_sizes:
        if bar_size in all_results and all_results[bar_size].folds:
            r = all_results[bar_size]
            s = r.summary
            avg_trades = np.mean([f.test_trades for f in r.folds])
            print(
                f"{bar_size:<8} {len(r.folds):<8} {s['avg_test_sharpe']:<12.2f} "
                f"{s['cumulative_pnl']*100:<11.1f}% {s['avg_mdd']*100:<9.1f}% "
                f"{s['overfit_ratio']:<10.2f} {avg_trades:<10.1f}"
            )
    print("=" * 80)


if __name__ == "__main__":
    main()
