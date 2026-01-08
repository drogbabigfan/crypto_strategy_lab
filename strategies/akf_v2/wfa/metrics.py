"""
WFA 결과 분석 및 리포트 모듈.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any
from collections import Counter

from .engine import FoldResult


@dataclass
class WFAResult:
    """WFA 전체 결과."""

    folds: List[FoldResult]
    summary: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.folds:
            self.summary = self._calculate_summary()

    def _calculate_summary(self) -> Dict[str, Any]:
        """요약 통계 계산."""
        if not self.folds:
            return {}

        train_sharpes = [f.train_sharpe for f in self.folds]
        test_sharpes = [f.test_sharpe for f in self.folds]
        test_pnls = [f.test_pnl for f in self.folds]

        avg_train = np.mean(train_sharpes)
        avg_test = np.mean(test_sharpes)
        overfit_ratio = avg_train / avg_test if avg_test > 0 else float("inf")

        # 누적 PnL (복리)
        cumulative_pnl = np.prod([1 + pnl for pnl in test_pnls]) - 1

        # 파라미터 안정성
        entry_counts = Counter([f.best_entry_zscore for f in self.folds])
        exit_counts = Counter([f.best_exit_zscore for f in self.folds])
        most_common_entry = entry_counts.most_common(1)[0] if entry_counts else (1.0, 0)
        most_common_exit = exit_counts.most_common(1)[0] if exit_counts else (-2.0, 0)

        return {
            "n_folds": len(self.folds),
            "avg_train_sharpe": avg_train,
            "avg_test_sharpe": avg_test,
            "std_test_sharpe": np.std(test_sharpes),
            "overfit_ratio": overfit_ratio,
            "cumulative_pnl": cumulative_pnl,
            "avg_test_pnl": np.mean(test_pnls),
            "total_trades": sum(f.test_trades for f in self.folds),
            "avg_win_rate": np.mean([f.test_win_rate for f in self.folds]),
            "avg_mdd": np.mean([f.test_mdd for f in self.folds]),
            "most_common_entry": most_common_entry,
            "most_common_exit": most_common_exit,
            "entry_stability": most_common_entry[1] / len(self.folds),
            "exit_stability": most_common_exit[1] / len(self.folds),
        }


def analyze_robustness(result: WFAResult) -> Dict[str, Any]:
    """
    로버스트니스 분석.

    Returns:
        분석 결과 딕셔너리
    """
    if not result.folds:
        return {"status": "NO_DATA"}

    summary = result.summary
    overfit_ratio = summary["overfit_ratio"]
    entry_stability = summary["entry_stability"]
    exit_stability = summary["exit_stability"]

    # 오버피팅 등급
    if overfit_ratio < 1.3:
        overfit_grade = "LOW"
    elif overfit_ratio < 1.7:
        overfit_grade = "MODERATE"
    else:
        overfit_grade = "HIGH"

    # 파라미터 안정성 등급
    param_stability = (entry_stability + exit_stability) / 2
    if param_stability > 0.6:
        stability_grade = "STABLE"
    elif param_stability > 0.4:
        stability_grade = "MODERATE"
    else:
        stability_grade = "UNSTABLE"

    # 일관성 분석 (Sharpe 분포)
    test_sharpes = [f.test_sharpe for f in result.folds]
    positive_folds = sum(1 for s in test_sharpes if s > 0)
    consistency = positive_folds / len(test_sharpes)

    if consistency > 0.7:
        consistency_grade = "CONSISTENT"
    elif consistency > 0.5:
        consistency_grade = "MODERATE"
    else:
        consistency_grade = "INCONSISTENT"

    # 종합 등급
    grades = {"LOW": 3, "MODERATE": 2, "HIGH": 1, "STABLE": 3, "UNSTABLE": 1, "CONSISTENT": 3, "INCONSISTENT": 1}
    score = (
        grades.get(overfit_grade, 2)
        + grades.get(stability_grade, 2)
        + grades.get(consistency_grade, 2)
    ) / 3

    if score >= 2.5:
        overall_grade = "ROBUST"
    elif score >= 2.0:
        overall_grade = "ACCEPTABLE"
    else:
        overall_grade = "WEAK"

    return {
        "overfit_ratio": overfit_ratio,
        "overfit_grade": overfit_grade,
        "param_stability": param_stability,
        "stability_grade": stability_grade,
        "consistency": consistency,
        "consistency_grade": consistency_grade,
        "positive_folds": positive_folds,
        "total_folds": len(result.folds),
        "overall_grade": overall_grade,
    }


def generate_report(result: WFAResult) -> str:
    """
    콘솔 출력용 리포트 생성.

    Returns:
        포맷된 리포트 문자열
    """
    if not result.folds:
        return "No WFA results to report."

    lines = []
    lines.append("=" * 70)
    lines.append("Walk-Forward Analysis Results (with Pyramiding)")
    lines.append("=" * 70)

    # Fold별 결과
    lines.append("\n[Fold Results]")
    lines.append("-" * 70)
    lines.append(
        f"{'Fold':<5} {'Test Period':<25} {'Entry':<6} {'Exit':<6} "
        f"{'Train SR':<9} {'Test SR':<9} {'Test PnL':<10}"
    )
    lines.append("-" * 70)

    for f in result.folds:
        lines.append(
            f"{f.fold_id:<5} {f.test_period:<25} {f.best_entry_zscore:<6.1f} "
            f"{f.best_exit_zscore:<6.1f} {f.train_sharpe:<9.2f} "
            f"{f.test_sharpe:<9.2f} {f.test_pnl*100:<9.1f}%"
        )

    # 요약
    summary = result.summary
    lines.append("\n" + "=" * 70)
    lines.append("[SUMMARY]")
    lines.append("=" * 70)
    lines.append(f"{'Metric':<30} {'Value':<20}")
    lines.append("-" * 50)
    lines.append(f"{'Number of Folds':<30} {summary['n_folds']:<20}")
    lines.append(f"{'Avg Train Sharpe':<30} {summary['avg_train_sharpe']:<20.2f}")
    lines.append(f"{'Avg Test Sharpe':<30} {summary['avg_test_sharpe']:<20.2f}")
    lines.append(f"{'Std Test Sharpe':<30} {summary['std_test_sharpe']:<20.2f}")
    lines.append(f"{'Overfit Ratio':<30} {summary['overfit_ratio']:<20.2f}")
    lines.append(f"{'Cumulative PnL (OOS)':<30} {summary['cumulative_pnl']*100:<19.1f}%")
    lines.append(f"{'Total Trades':<30} {summary['total_trades']:<20}")
    lines.append(f"{'Avg Win Rate':<30} {summary['avg_win_rate']*100:<19.1f}%")
    lines.append(f"{'Avg MDD':<30} {summary['avg_mdd']*100:<19.1f}%")

    # 파라미터 안정성
    lines.append("\n[Parameter Stability]")
    lines.append("-" * 50)
    entry_val, entry_count = summary["most_common_entry"]
    exit_val, exit_count = summary["most_common_exit"]
    lines.append(
        f"  entry_zscore: {entry_val:.1f} ({entry_count}/{summary['n_folds']} folds, "
        f"{summary['entry_stability']*100:.0f}%)"
    )
    lines.append(
        f"  exit_zscore:  {exit_val:.1f} ({exit_count}/{summary['n_folds']} folds, "
        f"{summary['exit_stability']*100:.0f}%)"
    )

    # 로버스트니스 분석
    robustness = analyze_robustness(result)
    lines.append("\n[Robustness Analysis]")
    lines.append("-" * 50)
    lines.append(f"  Overfit Grade:     {robustness['overfit_grade']}")
    lines.append(f"  Stability Grade:   {robustness['stability_grade']}")
    lines.append(
        f"  Consistency:       {robustness['positive_folds']}/{robustness['total_folds']} "
        f"positive ({robustness['consistency_grade']})"
    )
    lines.append(f"  Overall Grade:     {robustness['overall_grade']}")

    # 결론
    lines.append("\n" + "=" * 70)
    if robustness["overall_grade"] == "ROBUST":
        lines.append("=> Strategy shows ROBUST out-of-sample performance.")
    elif robustness["overall_grade"] == "ACCEPTABLE":
        lines.append("=> Strategy shows ACCEPTABLE performance with some concerns.")
    else:
        lines.append("=> Strategy shows WEAK robustness. Consider revision.")

    return "\n".join(lines)
