"""
Label Optimizer (v4.6)

Main orchestrator for Grid Search + Plateau Search optimization.
"""

from dataclasses import dataclass, field
from typing import Optional
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from .cost_model import SquareRootCostModel
from .tbm import TBMConfig, TripleBarrierLabeler, LabelResult
from .scoring import ScoringMetrics, compute_composite_score
from .plateau import find_plateau_center, grid_idx_to_params, PlateauResult

logger = logging.getLogger(__name__)


@dataclass
class GridConfig:
    """
    Grid search configuration.

    For denser grid (better plateau detection), use finer ranges:
        sl_range = [1.0, 1.5, 2.0, 2.5, 3.0]           # 5 values
        pt_range = [1.5, 1.75, 2.0, 2.25, 2.5, ...]    # 9 values
        time_range = [50, 75, 100, 150, 200, ...]      # 7 values
        → 5×9×7 = 315 grid points
    """

    sl_range: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    pt_range: list[float] = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0, 3.5])
    time_range: list[int] = field(default_factory=lambda: [50, 100, 200, 500, 1000])

    # Scoring weights
    entropy_weight: float = 0.2
    mi_weight: float = 0.4
    rank_ic_weight: float = 0.4

    # MI performance optimization
    mi_sample_ratio: float = 0.1  # Use 10% of samples for MI (speed vs accuracy)

    # Plateau search
    threshold_percentile: float = 90
    fallback_percentile: float = 80

    # Minimum requirements
    min_samples: int = 1000
    min_entropy: float = 1.0  # Avoid degenerate distributions

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return (len(self.sl_range), len(self.pt_range), len(self.time_range))

    @property
    def total_combinations(self) -> int:
        return len(self.sl_range) * len(self.pt_range) * len(self.time_range)

    @property
    def weights(self) -> dict:
        return {
            "entropy": self.entropy_weight,
            "mi": self.mi_weight,
            "rank_ic": self.rank_ic_weight,
        }


@dataclass
class GridSearchResult:
    """Result of a single grid search point."""

    sl: float
    pt: float
    time: int
    metrics: Optional[ScoringMetrics]
    n_filtered: int = 0
    valid: bool = True
    reason: str = ""


@dataclass
class OptimizationResult:
    """Final optimization result."""

    best_params: dict  # {"sl": x, "pt": y, "time": z}
    valid_indices: np.ndarray  # Indices of valid samples
    labels: np.ndarray  # Labels for valid samples
    returns: np.ndarray  # Returns for valid samples
    grid_results: pd.DataFrame  # All grid search results
    plateau: PlateauResult  # Plateau search details
    n_total_samples: int
    n_valid_samples: int
    n_filtered_samples: int


class LabelOptimizer:
    """
    Label Optimizer using Grid Search + Plateau Search.

    Finds optimal Triple Barrier parameters that maximize
    a composite score while avoiding overfitting to peaks.
    """

    # Feature columns to use for scoring
    DEFAULT_FEATURE_COLS = [
        "log_volume",
        "log_tick_count",
        "log_duration",
        "log_trade_intensity",
        "vwap_deviation",
        "volume_imbalance",
        "bar_range",
        "bar_body",
        "garman_klass_vol",
        "realized_vol",
        "shannon_entropy",
        "vol_ratio",
        "vol_zscore",
        "entropy_zscore",
        "skewness",
        "kurtosis",
        "frac_diff_close",
        "detrended_log_price",
        "returns",
        "vw_momentum",
        "momentum_zscore_10",
        "momentum_zscore_50",
        "momentum_zscore_250",
        "momentum_zscore_1000",
        "connors_rsi",
        "sin_time",
        "cos_time",
        "sin_week",
        "cos_week",
    ]

    def __init__(
        self,
        grid_config: Optional[GridConfig] = None,
        cost_model: Optional[SquareRootCostModel] = None,
        feature_cols: Optional[list[str]] = None,
        verbose: bool = True,
    ):
        """
        Initialize optimizer.

        Args:
            grid_config: Grid search configuration
            cost_model: Cost model for Fee Trap Guard
            feature_cols: Feature columns for scoring
            verbose: Show progress bar
        """
        self.grid_config = grid_config or GridConfig()
        self.cost_model = cost_model or SquareRootCostModel()
        self.feature_cols = feature_cols or self.DEFAULT_FEATURE_COLS
        self.verbose = verbose

    def optimize(self, data: pd.DataFrame) -> OptimizationResult:
        """
        Run full optimization pipeline.

        Args:
            data: DataFrame with OHLC, volatility, and features

        Returns:
            OptimizationResult with best params and labels
        """
        logger.info(
            f"Starting grid search: {self.grid_config.total_combinations} combinations"
        )

        # Validate feature columns
        available_cols = [c for c in self.feature_cols if c in data.columns]
        if len(available_cols) < len(self.feature_cols):
            missing = set(self.feature_cols) - set(available_cols)
            logger.warning(f"Missing feature columns: {missing}")
        self.feature_cols = available_cols

        # Run grid search
        grid_results = self._run_grid_search(data)

        # Build results DataFrame
        results_df = self._build_results_df(grid_results)

        # Compute normalized composite scores
        results_df = self._normalize_scores(results_df)

        # Plateau search
        valid_mask = results_df["valid"].values
        scores = results_df["composite"].values.copy()
        scores[~valid_mask] = 0  # Exclude invalid from plateau

        plateau = find_plateau_center(
            scores=scores,
            grid_shape=self.grid_config.grid_shape,
            threshold_percentile=self.grid_config.threshold_percentile,
            fallback_percentile=self.grid_config.fallback_percentile,
        )

        # Get best params
        best_params = grid_idx_to_params(
            plateau.indices,
            {
                "sl": self.grid_config.sl_range,
                "pt": self.grid_config.pt_range,
                "time": self.grid_config.time_range,
            },
        )

        logger.info(f"Best params (plateau center): {best_params}")
        logger.info(f"Plateau size: {plateau.component_size}")

        # Generate final labels with best params
        final_config = TBMConfig(
            sl_mult=best_params["sl"],
            pt_mult=best_params["pt"],
            vertical_bars=best_params["time"],
        )
        final_labeler = TripleBarrierLabeler(final_config, self.cost_model)
        final_result = final_labeler.label(data)

        return OptimizationResult(
            best_params=best_params,
            valid_indices=final_result.valid_indices,
            labels=final_result.labels,
            returns=final_result.returns,
            grid_results=results_df,
            plateau=plateau,
            n_total_samples=len(data),
            n_valid_samples=len(final_result.labels),
            n_filtered_samples=final_result.n_filtered,
        )

    def _run_grid_search(self, data: pd.DataFrame) -> list[GridSearchResult]:
        """Run grid search over all parameter combinations."""
        results = []

        iterator = self._get_iterator()

        for sl in self.grid_config.sl_range:
            for pt in self.grid_config.pt_range:
                for time in self.grid_config.time_range:
                    result = self._evaluate_params(data, sl, pt, time)
                    results.append(result)

                    if self.verbose:
                        next(iterator)

        return results

    def _get_iterator(self):
        """Get progress iterator."""
        if self.verbose:
            return iter(
                tqdm(
                    range(self.grid_config.total_combinations),
                    desc="Grid Search",
                )
            )
        return iter(range(self.grid_config.total_combinations))

    def _evaluate_params(
        self,
        data: pd.DataFrame,
        sl: float,
        pt: float,
        time: int,
    ) -> GridSearchResult:
        """Evaluate a single parameter combination."""
        config = TBMConfig(sl_mult=sl, pt_mult=pt, vertical_bars=time)
        labeler = TripleBarrierLabeler(config, self.cost_model)

        try:
            label_result = labeler.label(data)
        except Exception as e:
            return GridSearchResult(
                sl=sl,
                pt=pt,
                time=time,
                metrics=None,
                valid=False,
                reason=f"Labeling error: {e}",
            )

        # Check minimum samples
        if len(label_result.labels) < self.grid_config.min_samples:
            return GridSearchResult(
                sl=sl,
                pt=pt,
                time=time,
                metrics=None,
                n_filtered=label_result.n_filtered,
                valid=False,
                reason=f"Too few samples: {len(label_result.labels)}",
            )

        # Get features for valid samples
        features = data.iloc[label_result.valid_indices][self.feature_cols].values

        # Compute metrics
        metrics = ScoringMetrics.compute(
            features=features,
            labels=label_result.labels,
            weights=self.grid_config.weights,
            mi_sample_ratio=self.grid_config.mi_sample_ratio,
        )

        # Check minimum entropy
        if metrics.entropy < self.grid_config.min_entropy:
            return GridSearchResult(
                sl=sl,
                pt=pt,
                time=time,
                metrics=metrics,
                n_filtered=label_result.n_filtered,
                valid=False,
                reason=f"Low entropy: {metrics.entropy:.3f}",
            )

        return GridSearchResult(
            sl=sl,
            pt=pt,
            time=time,
            metrics=metrics,
            n_filtered=label_result.n_filtered,
            valid=True,
        )

    def _build_results_df(self, results: list[GridSearchResult]) -> pd.DataFrame:
        """Convert results to DataFrame."""
        rows = []

        for r in results:
            row = {
                "sl": r.sl,
                "pt": r.pt,
                "time": r.time,
                "valid": r.valid,
                "reason": r.reason,
                "n_filtered": r.n_filtered,
            }

            if r.metrics is not None:
                row.update(r.metrics.to_dict())
            else:
                row.update(
                    {
                        "entropy": 0.0,
                        "mi": 0.0,
                        "rank_ic": 0.0,
                        "n_samples": 0,
                        "composite": 0.0,
                    }
                )

            rows.append(row)

        return pd.DataFrame(rows)

    def _normalize_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize metrics across grid and recompute composite."""
        df = df.copy()

        # Only normalize across valid results
        valid_mask = df["valid"].values

        if valid_mask.sum() == 0:
            df["composite"] = 0.0
            return df

        # Get ranges from valid results only
        valid_df = df[valid_mask]

        entropy_range = (valid_df["entropy"].min(), valid_df["entropy"].max())
        mi_range = (valid_df["mi"].min(), valid_df["mi"].max())
        rank_ic_range = (valid_df["rank_ic"].min(), valid_df["rank_ic"].max())

        # Recompute composite with proper normalization
        composites = []
        for _, row in df.iterrows():
            if not row["valid"]:
                composites.append(0.0)
            else:
                composites.append(
                    compute_composite_score(
                        entropy=row["entropy"],
                        mi=row["mi"],
                        rank_ic=row["rank_ic"],
                        entropy_range=entropy_range,
                        mi_range=mi_range,
                        rank_ic_range=rank_ic_range,
                        weights=self.grid_config.weights,
                    )
                )

        df["composite"] = composites
        return df

    def get_summary(self, result: OptimizationResult) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "LABEL OPTIMIZER RESULT",
            "=" * 60,
            "",
            f"Best Parameters:",
            f"  SL Multiplier: {result.best_params['sl']}σ",
            f"  PT Multiplier: {result.best_params['pt']}σ",
            f"  Vertical Bars: {result.best_params['time']}",
            "",
            f"Sample Statistics:",
            f"  Total Samples: {result.n_total_samples:,}",
            f"  Valid Samples: {result.n_valid_samples:,}",
            f"  Filtered (Fee Trap): {result.n_filtered_samples:,}",
            "",
            f"Plateau Search:",
            f"  Component Size: {result.plateau.component_size}",
            f"  Total Components: {result.plateau.n_components}",
            f"  Threshold: {result.plateau.threshold_used}%",
            "",
        ]

        # Label distribution
        unique, counts = np.unique(result.labels, return_counts=True)
        lines.append("Label Distribution:")
        for label, count in zip(unique, counts):
            name = {-1: "Short", 0: "Neutral", 1: "Long"}.get(label, str(label))
            pct = count / len(result.labels) * 100
            lines.append(f"  {name}: {count:,} ({pct:.1f}%)")

        lines.extend(["", "=" * 60])

        return "\n".join(lines)
