#!/usr/bin/env python3
"""
Walk-Forward Analysis + Pyramiding 통합 실행 스크립트.

기능:
1. WFA로 전략 로버스트니스 검증
2. Sigma-Spacing 피라미딩 적용
3. 피라미딩 SL (BEP - n*√P) 지원
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from dataclasses import replace

from research.features.kalman import calculate_adaptive_kalman
from strategies.akf_v2.common import PyramidConfig, SizingConfig
from strategies.akf_v2.wfa import WFAConfig, WFAEngine, WFAResult, generate_report


def load_data(start_year: int = 2020, end_year: int = 2025) -> pd.DataFrame:
    """데이터 로드."""
    DATA_DIR = PROJECT_ROOT / "etl/data/features-6/futures/BTCUSDT"

    months = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            months.append(f"{year}-{month:02d}")

    dfs = []
    for month in months:
        path = DATA_DIR / f"BTCUSDT-features-{month}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))

    if not dfs:
        raise FileNotFoundError(f"No data found in {DATA_DIR}")

    return pd.concat(dfs, ignore_index=True)


def run_wfa_with_pyramiding(
    df_kf: pd.DataFrame,
    pyramid_config: PyramidConfig,
    sizing_config: SizingConfig,
    wfa_config: WFAConfig = None,
) -> WFAResult:
    """WFA + Pyramiding 실행."""
    if wfa_config is None:
        wfa_config = WFAConfig()

    # 설정 주입
    wfa_config = replace(
        wfa_config,
        pyramid_config=pyramid_config,
        sizing_config=sizing_config,
    )

    engine = WFAEngine(wfa_config)
    folds = engine.run(df_kf)

    return WFAResult(folds=folds)


def main():
    print("=" * 70)
    print("Walk-Forward Analysis + Pyramiding")
    print("Sigma-Spacing + SL (BEP ± n*√P)")
    print("=" * 70)

    # 1. 데이터 로드
    print("\n[1/4] Loading data...")
    df = load_data()
    print(f"  Loaded {len(df):,} bars")

    # 2. Kalman Filter 적용
    print("\n[2/4] Applying Kalman filter (PWNA + NIS)...")
    df_kf = calculate_adaptive_kalman(df)
    print(f"  Features: {', '.join([c for c in df_kf.columns if c.startswith('kf_')])}")

    # 3. 설정
    print("\n[3/4] Configuration:")

    # 피라미딩 설정 (비활성화 - 복리 테스트용)
    pyramid_config = PyramidConfig(
        enabled=False,  # 피라미딩 비활성화
        max_pyramid=3,
        pyramid_size=1.0,
        spacing_k=1.5,  # σ 배수 for spacing
        max_total_size=3.0,
        use_sl=False,  # SL도 비활성화
        sl_n=1.5,  # σ 배수 for SL (BEP - 1.5σ)
    )

    print(f"  Pyramiding:")
    print(f"    - enabled: {pyramid_config.enabled}")
    if pyramid_config.enabled:
        print(f"    - max_pyramid: {pyramid_config.max_pyramid}")
        print(f"    - spacing_k: {pyramid_config.spacing_k} (σ)")
        print(f"    - sl_n: {pyramid_config.sl_n} (σ)")
        print(f"    - max_total_size: {pyramid_config.max_total_size}x")

    # 사이징 설정 (피라미딩 비활성화 시 fixed 1.0)
    sizing_config = SizingConfig(
        method="fixed",  # 고정 사이즈
        min_size=1.0,
        max_size=1.0,
        kelly_window=500,
    )

    print(f"  Sizing:")
    print(f"    - method: {sizing_config.method}")
    print(f"    - size: {sizing_config.min_size}x")

    # WFA 설정
    wfa_config = WFAConfig(
        train_bars=3000,
        test_bars=750,
        step_bars=750,
        entry_zscore_range=(0.5, 1.0, 1.5, 2.0),
        exit_zscore_range=(-1.0, -1.5, -2.0, -2.5),
    )

    print(f"  WFA:")
    print(f"    - train: {wfa_config.train_bars} bars (~{wfa_config.train_bars//250} months)")
    print(f"    - test: {wfa_config.test_bars} bars (~{wfa_config.test_bars//250} months)")
    print(f"    - step: {wfa_config.step_bars} bars")

    # 4. WFA 실행
    print("\n[4/4] Running Walk-Forward Analysis...")
    result = run_wfa_with_pyramiding(
        df_kf,
        pyramid_config=pyramid_config,
        sizing_config=sizing_config,
        wfa_config=wfa_config,
    )

    # 5. 결과 출력
    print("\n")
    print(generate_report(result))

    # 6. SL n값 비교 (피라미딩 활성화 시에만)
    if pyramid_config.enabled and pyramid_config.use_sl:
        print("\n" + "=" * 70)
        print("[Bonus] SL n-value comparison")
        print("=" * 70)

        sl_n_values = [0.0, 0.5, 1.0, 1.5, 2.0]
        comparison_results = []

        for sl_n in sl_n_values:
            print(f"\n  Testing sl_n = {sl_n}...")

            test_pyramid_config = replace(pyramid_config, sl_n=sl_n)
            test_result = run_wfa_with_pyramiding(
                df_kf,
                pyramid_config=test_pyramid_config,
                sizing_config=sizing_config,
                wfa_config=wfa_config,
            )

            if test_result.folds:
                summary = test_result.summary
                comparison_results.append({
                    "sl_n": sl_n,
                    "avg_test_sharpe": summary["avg_test_sharpe"],
                    "cumulative_pnl": summary["cumulative_pnl"],
                    "avg_mdd": summary["avg_mdd"],
                    "overfit_ratio": summary["overfit_ratio"],
                })

        if comparison_results:
            print("\n" + "-" * 70)
            print(f"{'sl_n':<8} {'Avg Sharpe':<12} {'Cum PnL':<12} {'Avg MDD':<10} {'Overfit':<10}")
            print("-" * 70)

            for r in comparison_results:
                print(
                    f"{r['sl_n']:<8.1f} {r['avg_test_sharpe']:<12.2f} "
                    f"{r['cumulative_pnl']*100:<11.1f}% {r['avg_mdd']*100:<9.1f}% "
                    f"{r['overfit_ratio']:<10.2f}"
                )

            # 최적 sl_n 추천
            best = max(comparison_results, key=lambda x: x["avg_test_sharpe"])
            print(f"\n  => Best sl_n: {best['sl_n']} (Sharpe: {best['avg_test_sharpe']:.2f})")


if __name__ == "__main__":
    main()
