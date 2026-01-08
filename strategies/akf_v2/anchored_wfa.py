#!/usr/bin/env python3
"""
Anchored Walk-Forward Analysis.

굵직하게 4번 검증하는 구조:
- Fold 1: Train 2020-2021 (2년) → Test 2022 (1년) - 하락장 생존
- Fold 2: Train 2020-2022 (3년) → Test 2023 (1년) - 횡보장 적응
- Fold 3: Train 2020-2023 (4년) → Test 2024 (1년) - 반등장 테스트
- Fold 4: Train 2020-2024 (5년) → Test 2025 (1년) - 최신장 검증

Usage:
    python anchored_wfa.py [--bar-size 6] [--min-size 0] [--max-size 2]
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import pandas as pd
from itertools import product
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

from research.features.kalman import calculate_adaptive_kalman
from strategies.akf_v2.common import PyramidConfig, SizingConfig
from strategies.akf_v2.common.sizing import calculate_position_sizes
from strategies.akf_v2.backtest_pyramid import run_pyramid_backtest, PyramidBacktestConfig

DATA_ROOT = PROJECT_ROOT / "etl/data"


@dataclass
class AnchoredFold:
    """Anchored Fold 정의."""
    fold_id: int
    train_years: Tuple[int, int]  # (start, end) inclusive
    test_year: int
    description: str


# 4개 Fold 정의
ANCHORED_FOLDS = [
    AnchoredFold(1, (2020, 2021), 2022, "하락장 생존 테스트"),
    AnchoredFold(2, (2020, 2022), 2023, "횡보장 적응 테스트"),
    AnchoredFold(3, (2020, 2023), 2024, "반등장 테스트"),
    AnchoredFold(4, (2020, 2024), 2025, "최신장 최종 검증"),
]


@dataclass
class FoldResult:
    """Fold 결과."""
    fold_id: int
    train_period: str
    test_period: str
    description: str

    # 최적 파라미터
    best_entry_zscore: float
    best_exit_zscore: float

    # Train 성과
    train_sharpe: float
    train_pnl: float
    train_mdd: float
    train_trades: int

    # Test 성과
    test_sharpe: float
    test_pnl: float
    test_mdd: float
    test_trades: int
    test_win_rate: float


def load_year_data(bar_size: int, year: int) -> Optional[pd.DataFrame]:
    """특정 연도 데이터 로드."""
    data_dir = DATA_ROOT / f"features-{bar_size}/futures/BTCUSDT"
    dfs = []
    for month in range(1, 13):
        path = data_dir / f"BTCUSDT-features-{year}-{month:02d}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def load_range_data(bar_size: int, start_year: int, end_year: int) -> Optional[pd.DataFrame]:
    """연도 범위 데이터 로드."""
    dfs = []
    for year in range(start_year, end_year + 1):
        df = load_year_data(bar_size, year)
        if df is not None:
            dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def estimate_bars_per_day(df: pd.DataFrame) -> float:
    """bars_per_day 추정."""
    start_ts = df["start_time"].iloc[0]
    end_ts = df["start_time"].iloc[-1]
    total_days = (end_ts - start_ts) / (1000 * 60 * 60 * 24)
    if total_days <= 0:
        return 6.0
    return len(df) / total_days


def generate_signals(
    df: pd.DataFrame,
    entry_zscore: float,
    long_exit_zscore: float,
    short_exit_zscore: float,
    bars_per_day: float,
    sizing_config: SizingConfig,
) -> pd.DataFrame:
    """시그널 생성."""
    velocity = df["kf_velocity"].values
    uncertainty = df["kf_uncertainty"].values
    n = len(df)

    # 코인 기준 고정값 (7일 * 6 bars = 42, 30일 * 7 = 210)
    zscore_window = 42       # 7일 (1주)
    uncertainty_window = 210  # 30일 (1개월)
    warmup = 210

    # Velocity Z-score
    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=zscore_window, min_periods=5).mean()
    vel_std = vel_series.rolling(window=zscore_window, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

    # Uncertainty percentile filter
    p_pct = (
        pd.Series(uncertainty)
        .rolling(window=uncertainty_window, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )
    low_uncertainty = p_pct < 0.5

    # Entry 조건
    entry_long = (vel_zscore > entry_zscore) & low_uncertainty
    entry_short = (vel_zscore < -2.0) & low_uncertainty

    # 시그널 생성
    signals = np.zeros(n, dtype=np.int8)
    position = 0

    for i in range(warmup, n):
        if position == 0:
            if entry_long[i]:
                position = 1
                signals[i] = 1
            elif entry_short[i]:
                position = -1
                signals[i] = -1
        elif position == 1:
            if vel_zscore[i] < long_exit_zscore:
                position = 0
            else:
                signals[i] = 1
        elif position == -1:
            if vel_zscore[i] > short_exit_zscore:
                position = 0
            else:
                signals[i] = -1

    # Sizing
    sizes = calculate_position_sizes(df, sizing_config, entry_signals=signals)

    df = df.copy()
    df["signal"] = signals
    df["position_size"] = sizes
    df["sl_price"] = 0.0

    return df


def run_backtest(df_signals: pd.DataFrame) -> Dict[str, Any]:
    """백테스트 실행."""
    config = PyramidBacktestConfig(
        initial_capital=100000.0,
        compounding=True,
        fee_rate=0.001,
        slippage_rate=0.0001,
    )
    return run_pyramid_backtest(df_signals, config)


def optimize_parameters(
    df_train: pd.DataFrame,
    bars_per_day: float,
    sizing_config: SizingConfig,
    entry_range: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
    exit_range: Tuple[float, ...] = (-0.5, -1.0, -1.5, -2.0),
) -> Tuple[float, float, Dict[str, Any]]:
    """Train 데이터에서 최적 파라미터 탐색."""
    best_sharpe = -np.inf
    best_entry, best_exit = 1.0, -1.5
    best_metrics = {}

    for entry_z, exit_z in product(entry_range, exit_range):
        df_signals = generate_signals(
            df_train, entry_z, exit_z, bars_per_day, sizing_config
        )
        metrics = run_backtest(df_signals)

        if metrics and metrics.get("sharpe_ratio", -np.inf) > best_sharpe:
            best_sharpe = metrics["sharpe_ratio"]
            best_entry, best_exit = entry_z, exit_z
            best_metrics = metrics

    return best_entry, best_exit, best_metrics


def run_anchored_wfa(
    bar_size: int,
    sizing_config: SizingConfig,
) -> List[FoldResult]:
    """Anchored WFA 실행."""
    results = []

    for fold in ANCHORED_FOLDS:
        print(f"\n{'='*70}")
        print(f"Fold {fold.fold_id}: {fold.description}")
        print(f"  Train: {fold.train_years[0]}-{fold.train_years[1]}")
        print(f"  Test:  {fold.test_year}")
        print("=" * 70)

        # 데이터 로드
        df_train = load_range_data(bar_size, fold.train_years[0], fold.train_years[1])
        df_test = load_year_data(bar_size, fold.test_year)

        if df_train is None:
            print(f"  Train 데이터 없음")
            continue
        if df_test is None:
            print(f"  Test 데이터 없음")
            continue

        print(f"  Train: {len(df_train):,} bars")
        print(f"  Test:  {len(df_test):,} bars")

        # Kalman Filter 적용
        print("  Kalman Filter 적용 중...")
        df_train_kf = calculate_adaptive_kalman(df_train)
        df_test_kf = calculate_adaptive_kalman(df_test)

        bars_per_day = estimate_bars_per_day(df_train_kf)
        print(f"  bars_per_day: {bars_per_day:.1f}")

        # Train에서 최적화
        print("  파라미터 최적화 중...")
        best_entry, best_exit, train_metrics = optimize_parameters(
            df_train_kf, bars_per_day, sizing_config
        )
        print(f"  Best: entry={best_entry:.1f}, exit={best_exit:.1f}")
        print(f"  Train Sharpe={train_metrics.get('sharpe_ratio', 0):.2f}, "
              f"PnL={train_metrics.get('total_pnl', 0)*100:.1f}%")

        # Test 평가
        df_test_signals = generate_signals(
            df_test_kf, best_entry, best_exit, bars_per_day, sizing_config
        )
        test_metrics = run_backtest(df_test_signals)

        print(f"  Test  Sharpe={test_metrics.get('sharpe_ratio', 0):.2f}, "
              f"PnL={test_metrics.get('total_pnl', 0)*100:.1f}%, "
              f"MDD={test_metrics.get('max_drawdown', 0)*100:.1f}%")

        results.append(FoldResult(
            fold_id=fold.fold_id,
            train_period=f"{fold.train_years[0]}-{fold.train_years[1]}",
            test_period=str(fold.test_year),
            description=fold.description,
            best_entry_zscore=best_entry,
            best_exit_zscore=best_exit,
            train_sharpe=train_metrics.get("sharpe_ratio", 0),
            train_pnl=train_metrics.get("total_pnl", 0),
            train_mdd=train_metrics.get("max_drawdown", 0),
            train_trades=train_metrics.get("total_trades", 0),
            test_sharpe=test_metrics.get("sharpe_ratio", 0),
            test_pnl=test_metrics.get("total_pnl", 0),
            test_mdd=test_metrics.get("max_drawdown", 0),
            test_trades=test_metrics.get("total_trades", 0),
            test_win_rate=test_metrics.get("win_rate", 0),
        ))

    return results


def print_summary(results: List[FoldResult]):
    """결과 요약 출력."""
    print("\n" + "=" * 90)
    print("Anchored Walk-Forward Analysis 결과 요약")
    print("=" * 90)

    # Fold별 상세
    print(f"\n{'Fold':<6} {'Test':<6} {'Entry':<7} {'Exit':<7} "
          f"{'Train SR':<10} {'Test SR':<10} {'Test PnL':<12} {'Test MDD':<10} {'Trades':<8}")
    print("-" * 90)

    for r in results:
        print(f"{r.fold_id:<6} {r.test_period:<6} {r.best_entry_zscore:<7.1f} {r.best_exit_zscore:<7.1f} "
              f"{r.train_sharpe:<10.2f} {r.test_sharpe:<10.2f} {r.test_pnl*100:<11.1f}% "
              f"{r.test_mdd*100:<9.1f}% {r.test_trades:<8}")

    print("-" * 90)

    # 통계
    if results:
        avg_test_sharpe = np.mean([r.test_sharpe for r in results])
        avg_train_sharpe = np.mean([r.train_sharpe for r in results])
        cumulative_pnl = np.prod([1 + r.test_pnl for r in results]) - 1
        avg_test_mdd = np.mean([r.test_mdd for r in results])
        max_test_mdd = max(r.test_mdd for r in results)
        total_trades = sum(r.test_trades for r in results)
        avg_win_rate = np.mean([r.test_win_rate for r in results])

        overfit_ratio = avg_train_sharpe / avg_test_sharpe if avg_test_sharpe > 0 else float("inf")

        # 파라미터 안정성
        entries = [r.best_entry_zscore for r in results]
        exits = [r.best_exit_zscore for r in results]
        entry_stable = len(set(entries)) == 1
        exit_stable = len(set(exits)) == 1

        print(f"\n{'통계':<30} {'값':<20}")
        print("-" * 50)
        print(f"{'Avg Train Sharpe':<30} {avg_train_sharpe:.2f}")
        print(f"{'Avg Test Sharpe':<30} {avg_test_sharpe:.2f}")
        print(f"{'Overfit Ratio':<30} {overfit_ratio:.2f}")
        print(f"{'Cumulative Test PnL (복리)':<30} {cumulative_pnl*100:.1f}%")
        print(f"{'Avg Test MDD':<30} {avg_test_mdd*100:.1f}%")
        print(f"{'Max Test MDD':<30} {max_test_mdd*100:.1f}%")
        print(f"{'Total Test Trades':<30} {total_trades}")
        print(f"{'Avg Win Rate':<30} {avg_win_rate*100:.1f}%")

        print(f"\n{'파라미터 안정성':<30}")
        print("-" * 50)
        print(f"  Entry zscore: {entries} {'(안정)' if entry_stable else '(변동)'}")
        print(f"  Exit zscore:  {exits} {'(안정)' if exit_stable else '(변동)'}")

        # 등급 판정
        print(f"\n{'등급 판정':<30}")
        print("-" * 50)

        if overfit_ratio < 1.5:
            print("  Overfit: LOW (Good)")
        elif overfit_ratio < 2.0:
            print("  Overfit: MODERATE")
        else:
            print("  Overfit: HIGH (Bad)")

        positive_folds = sum(1 for r in results if r.test_sharpe > 0)
        print(f"  Consistency: {positive_folds}/{len(results)} folds positive")

        if avg_test_sharpe > 0.5 and overfit_ratio < 2.0 and positive_folds >= 3:
            print("\n  => ROBUST: 라이브 트레이딩 가능")
        elif avg_test_sharpe > 0 and positive_folds >= 2:
            print("\n  => ACCEPTABLE: 추가 검증 필요")
        else:
            print("\n  => WEAK: 전략 재검토 필요")

    print("=" * 90)


def run_fixed_params_test(
    bar_size: int,
    sizing_config: SizingConfig,
    fixed_entry: float,
    fixed_exit: float,
) -> List[FoldResult]:
    """고정 파라미터로 동일 Fold 테스트."""
    results = []

    for fold in ANCHORED_FOLDS:
        print(f"\n{'='*70}")
        print(f"Fold {fold.fold_id}: {fold.description} (Fixed: {fixed_entry}, {fixed_exit})")
        print(f"  Test: {fold.test_year}")
        print("=" * 70)

        # Test 데이터만 로드
        df_test = load_year_data(bar_size, fold.test_year)
        if df_test is None:
            print(f"  Test 데이터 없음")
            continue

        print(f"  Test: {len(df_test):,} bars")

        # Kalman Filter
        df_test_kf = calculate_adaptive_kalman(df_test)
        bars_per_day = estimate_bars_per_day(df_test_kf)

        # 고정 파라미터로 시그널 생성
        df_test_signals = generate_signals(
            df_test_kf, fixed_entry, fixed_exit, bars_per_day, sizing_config
        )
        test_metrics = run_backtest(df_test_signals)

        print(f"  Test Sharpe={test_metrics.get('sharpe_ratio', 0):.2f}, "
              f"PnL={test_metrics.get('total_pnl', 0)*100:.1f}%, "
              f"MDD={test_metrics.get('max_drawdown', 0)*100:.1f}%")

        results.append(FoldResult(
            fold_id=fold.fold_id,
            train_period="N/A",
            test_period=str(fold.test_year),
            description=fold.description,
            best_entry_zscore=fixed_entry,
            best_exit_zscore=fixed_exit,
            train_sharpe=0,
            train_pnl=0,
            train_mdd=0,
            train_trades=0,
            test_sharpe=test_metrics.get("sharpe_ratio", 0),
            test_pnl=test_metrics.get("total_pnl", 0),
            test_mdd=test_metrics.get("max_drawdown", 0),
            test_trades=test_metrics.get("total_trades", 0),
            test_win_rate=test_metrics.get("win_rate", 0),
        ))

    return results


def main():
    parser = argparse.ArgumentParser(description="Anchored Walk-Forward Analysis")
    parser.add_argument("--bar-size", type=int, default=6, help="Bar 크기")
    parser.add_argument("--min-size", type=float, default=0.0, help="최소 포지션 사이즈")
    parser.add_argument("--max-size", type=float, default=2.0, help="최대 포지션 사이즈")
    parser.add_argument(
        "--method",
        type=str,
        default="kf_prob",
        choices=["fixed", "kelly", "kf_prob"],
        help="사이징 방식",
    )
    parser.add_argument("--fixed-entry", type=float, default=None, help="고정 entry zscore")
    parser.add_argument("--fixed-exit", type=float, default=None, help="고정 exit zscore")
    args = parser.parse_args()

    sizing_config = SizingConfig(
        method=args.method,
        min_size=args.min_size,
        max_size=args.max_size,
    )

    # 고정 파라미터 모드
    if args.fixed_entry is not None and args.fixed_exit is not None:
        print("=" * 90)
        print("Fixed Parameter Test (동일 Fold)")
        print(f"  Bar Size: {args.bar_size}")
        print(f"  Sizing: {args.method} [{args.min_size}, {args.max_size}]")
        print(f"  Fixed Params: entry={args.fixed_entry}, exit={args.fixed_exit}")
        print("=" * 90)

        results = run_fixed_params_test(
            args.bar_size, sizing_config, args.fixed_entry, args.fixed_exit
        )
    else:
        print("=" * 90)
        print("Anchored Walk-Forward Analysis")
        print(f"  Bar Size: {args.bar_size}")
        print(f"  Sizing: {args.method} [{args.min_size}, {args.max_size}]")
        print(f"  Folds: 4 (Test: 2022, 2023, 2024, 2025)")
        print("=" * 90)

        results = run_anchored_wfa(args.bar_size, sizing_config)

    print_summary(results)


if __name__ == "__main__":
    main()
