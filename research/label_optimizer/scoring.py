"""
Scoring Metrics for Label Optimization (v4.6)

- Entropy: Class distribution balance
- Mutual Information: Feature-Label dependency
- Rank IC: Spearman correlation (ordinal label friendly)
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif


def compute_entropy(labels: np.ndarray) -> float:
    """
    Compute Shannon Entropy of label distribution.

    Higher entropy = more balanced classes.
    Max entropy for 3 classes = log(3) ≈ 1.585

    Args:
        labels: Array of labels

    Returns:
        Shannon entropy value
    """
    if len(labels) == 0:
        return 0.0

    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)

    # Avoid log(0)
    probs = probs[probs > 0]

    return -np.sum(probs * np.log(probs))


def compute_mi(
    features: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 3,
    random_state: int = 42,
    sample_ratio: float = 1.0,
) -> float:
    """
    Compute average Mutual Information between features and labels.

    Higher MI = features are more predictive of labels.

    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Label array
        n_neighbors: Number of neighbors for MI estimation
        random_state: Random seed
        sample_ratio: Fraction of samples to use (0.1 = 10% for speed)

    Returns:
        Average MI across all features
    """
    if len(labels) == 0 or features.shape[0] == 0:
        return 0.0

    # Handle NaN values
    valid_mask = ~np.isnan(features).any(axis=1)
    if valid_mask.sum() < 10:
        return 0.0

    features_clean = features[valid_mask]
    labels_clean = labels[valid_mask]

    # Downsampling for performance (MI is distributional, sampling OK)
    if sample_ratio < 1.0 and len(labels_clean) > 100:
        rng = np.random.RandomState(random_state)
        n_samples = max(100, int(len(labels_clean) * sample_ratio))
        indices = rng.choice(len(labels_clean), size=n_samples, replace=False)
        features_clean = features_clean[indices]
        labels_clean = labels_clean[indices]

    try:
        mi_scores = mutual_info_classif(
            features_clean,
            labels_clean,
            discrete_features=False,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        return float(np.mean(mi_scores))
    except Exception:
        return 0.0


def compute_rank_ic(features: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Rank IC (Information Coefficient).

    Uses Spearman correlation (rank-based) which is suitable for
    ordinal labels like {-1, 0, +1}.

    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Label array

    Returns:
        Mean absolute Spearman correlation across features
    """
    if len(labels) == 0 or features.shape[0] == 0:
        return 0.0

    n_features = features.shape[1]
    ics = []

    for i in range(n_features):
        feature_col = features[:, i]

        # Skip if all same or has NaN
        if np.isnan(feature_col).any():
            continue
        if np.std(feature_col) < 1e-10:
            continue

        try:
            ic, _ = spearmanr(feature_col, labels)
            if not np.isnan(ic):
                ics.append(abs(ic))  # Absolute value (direction agnostic)
        except Exception:
            continue

    return float(np.mean(ics)) if ics else 0.0


def compute_composite_score(
    entropy: float,
    mi: float,
    rank_ic: float,
    entropy_range: tuple[float, float] = (0.0, 1.585),
    mi_range: tuple[float, float] = (0.0, 1.0),
    rank_ic_range: tuple[float, float] = (0.0, 0.5),
    weights: dict | None = None,
) -> float:
    """
    Compute weighted composite score with normalization.

    Args:
        entropy: Entropy value
        mi: Mutual Information value
        rank_ic: Rank IC value
        entropy_range: (min, max) for normalization
        mi_range: (min, max) for normalization
        rank_ic_range: (min, max) for normalization
        weights: Weight dict {"entropy": w1, "mi": w2, "rank_ic": w3}

    Returns:
        Normalized weighted composite score [0, 1]
    """
    if weights is None:
        weights = {"entropy": 0.2, "mi": 0.4, "rank_ic": 0.4}

    def normalize(value: float, min_val: float, max_val: float) -> float:
        if max_val <= min_val:
            return 0.0
        return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)

    norm_entropy = normalize(entropy, *entropy_range)
    norm_mi = normalize(mi, *mi_range)
    norm_rank_ic = normalize(rank_ic, *rank_ic_range)

    return (
        weights["entropy"] * norm_entropy
        + weights["mi"] * norm_mi
        + weights["rank_ic"] * norm_rank_ic
    )


class ScoringMetrics:
    """Container for all scoring metrics."""

    def __init__(
        self,
        entropy: float,
        mi: float,
        rank_ic: float,
        n_samples: int,
        composite: float = 0.0,
    ):
        self.entropy = entropy
        self.mi = mi
        self.rank_ic = rank_ic
        self.n_samples = n_samples
        self.composite = composite

    def to_dict(self) -> dict:
        return {
            "entropy": self.entropy,
            "mi": self.mi,
            "rank_ic": self.rank_ic,
            "n_samples": self.n_samples,
            "composite": self.composite,
        }

    @classmethod
    def compute(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        weights: dict | None = None,
        mi_sample_ratio: float = 1.0,
    ) -> "ScoringMetrics":
        """Compute all metrics at once."""
        entropy = compute_entropy(labels)
        mi = compute_mi(features, labels, sample_ratio=mi_sample_ratio)
        rank_ic = compute_rank_ic(features, labels)

        composite = compute_composite_score(
            entropy=entropy,
            mi=mi,
            rank_ic=rank_ic,
            weights=weights,
        )

        return cls(
            entropy=entropy,
            mi=mi,
            rank_ic=rank_ic,
            n_samples=len(labels),
            composite=composite,
        )
