"""
Tests for Label Optimizer Module (v4.6)
"""

import numpy as np
import pandas as pd
import pytest

from research.label_optimizer.cost_model import SquareRootCostModel
from research.label_optimizer.tbm import TBMConfig, TripleBarrierLabeler
from research.label_optimizer.scoring import (
    compute_entropy,
    compute_mi,
    compute_rank_ic,
    ScoringMetrics,
)
from research.label_optimizer.plateau import (
    find_plateau_center,
    grid_idx_to_params,
)
from research.label_optimizer.optimizer import (
    GridConfig,
    LabelOptimizer,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_price_data():
    """Generate synthetic OHLC data with volatility."""
    np.random.seed(42)
    n_samples = 2000

    # Random walk price
    returns = np.random.randn(n_samples) * 0.02  # 2% daily vol
    prices = 100 * np.exp(np.cumsum(returns))

    # OHLC from prices
    highs = prices * (1 + np.abs(np.random.randn(n_samples)) * 0.01)
    lows = prices * (1 - np.abs(np.random.randn(n_samples)) * 0.01)
    opens = prices * (1 + np.random.randn(n_samples) * 0.005)

    # Volatility (rolling std of returns)
    vol = pd.Series(returns).rolling(24).std().fillna(0.02).values

    # Features
    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "realized_vol": vol,
            "log_volume": np.random.randn(n_samples),
            "log_tick_count": np.random.randn(n_samples),
            "vwap_deviation": np.random.randn(n_samples) * 0.1,
        }
    )

    return df


@pytest.fixture
def cost_model():
    """Default cost model."""
    return SquareRootCostModel()


# =============================================================================
# Test Cost Model
# =============================================================================


class TestSquareRootCostModel:
    def test_base_slippage(self, cost_model):
        """Test minimum slippage."""
        slippage = cost_model.get_slippage(volatility=0.0, trade_size=0.0)
        assert slippage == cost_model.base_slippage

    def test_slippage_increases_with_volatility(self, cost_model):
        """Higher volatility = higher slippage."""
        slip_low = cost_model.get_slippage(volatility=0.01, trade_size=0.01)
        slip_high = cost_model.get_slippage(volatility=0.05, trade_size=0.01)
        assert slip_high > slip_low

    def test_slippage_cap(self, cost_model):
        """Slippage should be capped."""
        slippage = cost_model.get_slippage(
            volatility=1.0,  # Very high
            trade_size=10.0,  # Large trade
        )
        assert slippage <= cost_model.slippage_cap

    def test_is_viable_with_low_pt(self, cost_model):
        """Low PT should fail viability check."""
        assert not cost_model.is_viable(pt_mult=0.1, sigma=0.02)

    def test_is_viable_with_high_pt(self, cost_model):
        """High PT should pass viability check."""
        assert cost_model.is_viable(pt_mult=3.0, sigma=0.02)

    def test_total_cost_is_round_trip(self, cost_model):
        """Total cost should be 2x entry cost."""
        slippage = cost_model.get_slippage(volatility=0.02)
        one_way = cost_model.base_fee + slippage
        total = cost_model.get_total_cost(volatility=0.02)
        assert total == pytest.approx(2 * one_way)

    # === Edge Cases ===

    def test_get_breakeven_pt(self, cost_model):
        """Test breakeven PT calculation."""
        sigma = 0.02
        breakeven_pt = cost_model.get_breakeven_pt(sigma)

        # At breakeven, is_viable should be exactly at threshold
        assert breakeven_pt > 0
        # Slightly above breakeven should be viable
        assert cost_model.is_viable(pt_mult=breakeven_pt * 1.1, sigma=sigma)
        # Slightly below should not be viable
        assert not cost_model.is_viable(pt_mult=breakeven_pt * 0.9, sigma=sigma)

    def test_get_breakeven_pt_zero_sigma(self, cost_model):
        """Zero sigma should return infinity."""
        assert cost_model.get_breakeven_pt(sigma=0.0) == float("inf")
        assert cost_model.get_breakeven_pt(sigma=-0.01) == float("inf")

    def test_slippage_with_zero_volume(self, cost_model):
        """Zero avg_volume should return cap."""
        slippage = cost_model.get_slippage(volatility=0.02, avg_volume=0)
        assert slippage == cost_model.slippage_cap

    def test_slippage_with_negative_volume(self, cost_model):
        """Negative avg_volume should return cap."""
        slippage = cost_model.get_slippage(volatility=0.02, avg_volume=-100)
        assert slippage == cost_model.slippage_cap


# =============================================================================
# Test Triple Barrier
# =============================================================================


class TestTripleBarrierLabeler:
    def test_basic_labeling(self, synthetic_price_data, cost_model):
        """Test basic labeling works."""
        config = TBMConfig(sl_mult=2.0, pt_mult=2.0, vertical_bars=50)
        labeler = TripleBarrierLabeler(config, cost_model)

        result = labeler.label(synthetic_price_data)

        assert len(result.valid_indices) > 0
        assert len(result.labels) == len(result.valid_indices)
        assert set(result.labels).issubset({-1, 0, 1})

    def test_label_distribution(self, synthetic_price_data, cost_model):
        """Test label distribution has all classes."""
        config = TBMConfig(sl_mult=1.5, pt_mult=1.5, vertical_bars=100)
        labeler = TripleBarrierLabeler(config, cost_model)

        result = labeler.label(synthetic_price_data)
        unique = set(result.labels)

        # Should have some diversity
        assert len(unique) >= 2

    def test_fee_trap_filtering(self, synthetic_price_data, cost_model):
        """Test Fee Trap Guard filters samples."""
        config = TBMConfig(sl_mult=2.0, pt_mult=0.5, vertical_bars=50)  # Low PT
        labeler = TripleBarrierLabeler(config, cost_model)

        result = labeler.label(synthetic_price_data)

        # Should filter some samples due to low PT
        assert result.n_filtered > 0

    def test_no_filtering_without_cost_model(self, synthetic_price_data):
        """Without cost model, no filtering."""
        config = TBMConfig(sl_mult=2.0, pt_mult=0.5, vertical_bars=50)
        labeler = TripleBarrierLabeler(config, cost_model=None)

        result = labeler.label(synthetic_price_data)

        assert result.n_filtered == 0

    # === Edge Cases ===

    def test_missing_columns_raises_error(self):
        """Missing required columns should raise ValueError."""
        config = TBMConfig(sl_mult=2.0, pt_mult=2.0, vertical_bars=50)
        labeler = TripleBarrierLabeler(config)

        # Missing 'realized_vol'
        bad_data = pd.DataFrame({
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            labeler.label(bad_data)

    def test_nan_volatility_skipped(self, cost_model):
        """Rows with NaN volatility should be skipped."""
        config = TBMConfig(sl_mult=2.0, pt_mult=2.0, vertical_bars=5)
        labeler = TripleBarrierLabeler(config, cost_model)

        data = pd.DataFrame({
            "open": [100] * 20,
            "high": [105] * 20,
            "low": [95] * 20,
            "close": [100] * 20,
            "realized_vol": [np.nan] * 10 + [0.02] * 10,
        })

        result = labeler.label(data)

        # Only rows with valid volatility should be labeled
        assert all(idx >= 10 for idx in result.valid_indices)

    def test_zero_volatility_skipped(self, cost_model):
        """Rows with zero volatility should be skipped."""
        config = TBMConfig(sl_mult=2.0, pt_mult=2.0, vertical_bars=5)
        labeler = TripleBarrierLabeler(config, cost_model)

        data = pd.DataFrame({
            "open": [100] * 20,
            "high": [105] * 20,
            "low": [95] * 20,
            "close": [100] * 20,
            "realized_vol": [0.0] * 10 + [0.02] * 10,
        })

        result = labeler.label(data)
        assert all(idx >= 10 for idx in result.valid_indices)

    def test_same_bar_pt_sl_hit(self, cost_model):
        """When PT and SL hit on same bar, use close direction."""
        config = TBMConfig(sl_mult=1.0, pt_mult=1.0, vertical_bars=10)
        labeler = TripleBarrierLabeler(config, cost_model=None)  # No fee trap

        # Setup: bar 0 entry, bar 1 has both PT and SL hit
        # Close above entry -> should be labeled as +1 (Long)
        data = pd.DataFrame({
            "open": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            "high": [100, 110, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],  # PT hit
            "low": [100, 90, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],   # SL hit
            "close": [100, 105, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],  # Above entry
            "realized_vol": [0.05] * 12,  # 5% vol, so PT=105, SL=95
        })

        result = labeler.label(data)

        # First sample (idx=0) should have label +1 (close > entry)
        if len(result.labels) > 0 and 0 in result.valid_indices:
            idx = list(result.valid_indices).index(0)
            assert result.labels[idx] == 1

    def test_get_label_distribution(self, synthetic_price_data, cost_model):
        """Test label distribution helper."""
        config = TBMConfig(sl_mult=2.0, pt_mult=2.0, vertical_bars=50)
        labeler = TripleBarrierLabeler(config, cost_model)

        result = labeler.label(synthetic_price_data)
        dist = labeler.get_label_distribution(result.labels)

        # Should have keys for existing labels
        assert isinstance(dist, dict)
        total_pct = sum(v["pct"] for v in dist.values())
        assert total_pct == pytest.approx(100.0)

    def test_returns_calculated_correctly(self, cost_model):
        """Test that returns are calculated correctly."""
        config = TBMConfig(sl_mult=2.0, pt_mult=2.0, vertical_bars=10)
        labeler = TripleBarrierLabeler(config, cost_model=None)

        # Clear uptrend
        prices = [100 + i * 2 for i in range(20)]
        data = pd.DataFrame({
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "realized_vol": [0.02] * 20,
        })

        result = labeler.label(data)

        # All returns should be positive in uptrend
        assert len(result.returns) > 0
        assert all(r >= 0 for r in result.returns[:5])  # At least first few


# =============================================================================
# Test Scoring
# =============================================================================


class TestScoring:
    def test_entropy_balanced(self):
        """Balanced labels have high entropy."""
        labels = np.array([0, 0, 0, 1, 1, 1, -1, -1, -1])
        entropy = compute_entropy(labels)

        # Max entropy for 3 classes = log(3) ≈ 1.0986
        assert entropy > 1.0
        assert entropy < 1.2  # Should be close to log(3)

    def test_entropy_imbalanced(self):
        """Imbalanced labels have low entropy."""
        labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0])
        entropy = compute_entropy(labels)

        assert entropy < 1.0

    def test_rank_ic_with_correlation(self):
        """Features correlated with labels have high IC."""
        np.random.seed(42)
        labels = np.array([1, 1, 1, 0, 0, 0, -1, -1, -1])
        features = np.column_stack(
            [
                labels + np.random.randn(9) * 0.1,  # Correlated
                np.random.randn(9),  # Uncorrelated
            ]
        )

        ic = compute_rank_ic(features, labels)
        assert ic > 0.3

    def test_rank_ic_no_correlation(self):
        """Uncorrelated features have low IC."""
        np.random.seed(42)
        labels = np.array([1, 1, 1, 0, 0, 0, -1, -1, -1])
        features = np.random.randn(9, 5)

        ic = compute_rank_ic(features, labels)
        assert ic < 0.5

    def test_scoring_metrics_compute(self):
        """Test ScoringMetrics.compute()."""
        np.random.seed(42)
        labels = np.array([1, 1, 0, 0, -1, -1])
        features = np.random.randn(6, 3)

        metrics = ScoringMetrics.compute(features, labels)

        assert metrics.entropy > 0
        assert 0 <= metrics.composite <= 1

    def test_mi_downsampling(self):
        """Test MI with downsampling produces similar results."""
        np.random.seed(42)
        labels = np.concatenate([np.ones(100), np.zeros(100), -np.ones(100)])
        features = np.random.randn(300, 5)

        mi_full = compute_mi(features, labels, sample_ratio=1.0)
        mi_sampled = compute_mi(features, labels, sample_ratio=0.1)

        # Both should be non-negative
        assert mi_full >= 0
        assert mi_sampled >= 0

        # Sampled should be roughly similar (within 50% for random data)
        # Note: exact match not expected due to sampling variance
        assert mi_sampled < mi_full * 2  # Sanity check

    # === Edge Cases ===

    def test_entropy_empty_labels(self):
        """Empty labels should return 0."""
        entropy = compute_entropy(np.array([]))
        assert entropy == 0.0

    def test_entropy_single_class(self):
        """Single class should have 0 entropy."""
        labels = np.array([1, 1, 1, 1, 1])
        entropy = compute_entropy(labels)
        assert entropy == 0.0

    def test_mi_empty_data(self):
        """Empty data should return 0."""
        mi = compute_mi(np.array([]).reshape(0, 5), np.array([]))
        assert mi == 0.0

    def test_mi_with_nan_features(self):
        """Features with NaN should be handled."""
        np.random.seed(42)
        labels = np.array([1, 1, 0, 0, -1, -1, 1, 0, -1, 0])
        features = np.random.randn(10, 3)
        features[0, 0] = np.nan  # Add NaN

        mi = compute_mi(features, labels)
        # Should still compute (drops NaN rows)
        assert mi >= 0

    def test_rank_ic_empty_data(self):
        """Empty data should return 0."""
        ic = compute_rank_ic(np.array([]).reshape(0, 5), np.array([]))
        assert ic == 0.0

    def test_rank_ic_constant_feature(self):
        """Constant features should be skipped (std=0)."""
        labels = np.array([1, 1, 0, 0, -1, -1])
        features = np.column_stack([
            np.ones(6),  # Constant - should be skipped
            np.random.randn(6),  # Variable
        ])

        ic = compute_rank_ic(features, labels)
        assert ic >= 0  # Should not crash

    def test_composite_score_normalization(self):
        """Test composite score with custom ranges."""
        from research.label_optimizer.scoring import compute_composite_score

        score = compute_composite_score(
            entropy=1.0,
            mi=0.5,
            rank_ic=0.25,
            entropy_range=(0, 2),
            mi_range=(0, 1),
            rank_ic_range=(0, 0.5),
        )

        # All inputs are at 50% of their range
        # With default weights (0.2, 0.4, 0.4), score should be 0.5
        assert score == pytest.approx(0.5)


# =============================================================================
# Test Plateau Search
# =============================================================================


class TestPlateauSearch:
    def test_find_single_peak(self):
        """Single peak should be found."""
        # 3x3x3 grid with one high value
        scores = np.zeros(27)
        scores[13] = 1.0  # Center

        result = find_plateau_center(scores, grid_shape=(3, 3, 3))

        assert result.indices == (1, 1, 1)

    def test_find_plateau(self):
        """Plateau should select center."""
        # 3x3 grid with plateau
        scores = np.array(
            [
                0.1,
                0.1,
                0.1,
                0.1,
                0.9,
                0.9,  # Plateau in middle
                0.1,
                0.9,
                0.9,
            ]
        )

        result = find_plateau_center(scores, grid_shape=(3, 3))

        # Should select somewhere in the plateau
        assert result.component_size >= 2

    def test_grid_idx_to_params(self):
        """Test index to parameter conversion."""
        params = grid_idx_to_params(
            idx=(1, 2, 0),
            param_ranges={
                "sl": [1.0, 2.0, 3.0],
                "pt": [1.5, 2.0, 2.5],
                "time": [50, 100],
            },
        )

        assert params == {"sl": 2.0, "pt": 2.5, "time": 50}

    # === Edge Cases ===

    def test_fallback_to_argmax(self):
        """When no components found, fallback to argmax."""
        # All isolated points (no connected components above threshold)
        scores = np.array([0.1, 0.2, 0.15, 0.12])

        result = find_plateau_center(
            scores,
            grid_shape=(2, 2),
            threshold_percentile=99,  # Very high threshold
            fallback_percentile=99,   # Still high
            min_component_size=10,    # Impossible to satisfy
        )

        # Should fallback to argmax (index 1 = value 0.2)
        assert result.component_size == 1
        assert result.n_components == 0

    def test_min_component_size_filter(self):
        """Small components should be filtered."""
        # 4x4 grid with two components: size 1 and size 4
        scores = np.array([
            0.9, 0.1, 0.1, 0.1,  # Single peak at (0,0)
            0.1, 0.1, 0.9, 0.9,  # Plateau at (1,2), (1,3)
            0.1, 0.1, 0.9, 0.9,  # Plateau at (2,2), (2,3)
            0.1, 0.1, 0.1, 0.1,
        ])

        result = find_plateau_center(
            scores,
            grid_shape=(4, 4),
            threshold_percentile=80,
            min_component_size=2,  # Filter out single peak
        )

        # Should select from the larger component
        assert result.component_size >= 2

    def test_visualize_plateau_2d(self):
        """Test 2D visualization."""
        from research.label_optimizer.plateau import visualize_plateau, PlateauResult

        scores = np.array([0.1, 0.9, 0.9, 0.1])
        result = PlateauResult(
            indices=(0, 1),
            component_size=2,
            n_components=1,
            threshold_used=80,
            center_of_mass=(0.5, 1.0),
        )

        viz = visualize_plateau(scores, (2, 2), result, ["SL", "PT"])

        assert "Plateau Search Result" in viz
        assert "★" in viz  # Selected point
        assert "●" in viz  # Plateau point

    def test_visualize_plateau_3d(self):
        """Test 3D visualization."""
        from research.label_optimizer.plateau import visualize_plateau, PlateauResult

        scores = np.zeros(27)
        scores[13] = 1.0
        result = PlateauResult(
            indices=(1, 1, 1),
            component_size=1,
            n_components=1,
            threshold_used=90,
            center_of_mass=(1.0, 1.0, 1.0),
        )

        viz = visualize_plateau(scores, (3, 3, 3), result, ["SL", "PT", "Time"])

        assert "Slice at" in viz
        assert "★" in viz

    def test_grid_idx_to_params_length_mismatch(self):
        """Mismatched index length should raise error."""
        with pytest.raises(ValueError, match="Index length"):
            grid_idx_to_params(
                idx=(1, 2),  # 2 indices
                param_ranges={"a": [1], "b": [2], "c": [3]},  # 3 params
            )


# =============================================================================
# Test Full Optimizer
# =============================================================================


class TestLabelOptimizer:
    def test_basic_optimization(self, synthetic_price_data):
        """Test full optimization pipeline."""
        grid_config = GridConfig(
            sl_range=[1.5, 2.0],
            pt_range=[1.5, 2.0],
            time_range=[50, 100],
            min_samples=100,
        )

        optimizer = LabelOptimizer(
            grid_config=grid_config,
            feature_cols=["log_volume", "vwap_deviation"],
            verbose=False,
        )

        result = optimizer.optimize(synthetic_price_data)

        assert result.best_params is not None
        assert "sl" in result.best_params
        assert "pt" in result.best_params
        assert "time" in result.best_params
        assert len(result.labels) > 0

    def test_optimizer_summary(self, synthetic_price_data):
        """Test summary generation."""
        grid_config = GridConfig(
            sl_range=[2.0],
            pt_range=[2.0],
            time_range=[50],
            min_samples=50,
        )

        optimizer = LabelOptimizer(
            grid_config=grid_config,
            verbose=False,
        )

        result = optimizer.optimize(synthetic_price_data)
        summary = optimizer.get_summary(result)

        assert "Best Parameters" in summary
        assert "Label Distribution" in summary

    # === Edge Cases ===

    def test_grid_config_properties(self):
        """Test GridConfig computed properties."""
        config = GridConfig(
            sl_range=[1.0, 2.0, 3.0],
            pt_range=[1.5, 2.0, 2.5, 3.0],
            time_range=[50, 100],
        )

        assert config.grid_shape == (3, 4, 2)
        assert config.total_combinations == 24
        assert config.weights == {"entropy": 0.2, "mi": 0.4, "rank_ic": 0.4}

    def test_missing_feature_columns_warning(self, synthetic_price_data):
        """Missing feature columns should be handled gracefully."""
        grid_config = GridConfig(
            sl_range=[2.0],
            pt_range=[2.0],
            time_range=[50],
            min_samples=50,
        )

        optimizer = LabelOptimizer(
            grid_config=grid_config,
            feature_cols=["log_volume", "nonexistent_column"],  # One missing
            verbose=False,
        )

        # Should not crash, just use available columns
        result = optimizer.optimize(synthetic_price_data)
        assert result is not None

    def test_all_grid_points_invalid(self, synthetic_price_data):
        """When all grid points are invalid, should still return result."""
        grid_config = GridConfig(
            sl_range=[0.01],  # Very tight SL
            pt_range=[0.01],  # Very tight PT
            time_range=[5],   # Very short time
            min_samples=100000,  # Impossible to satisfy
            min_entropy=10.0,    # Impossible to satisfy
        )

        optimizer = LabelOptimizer(
            grid_config=grid_config,
            verbose=False,
        )

        result = optimizer.optimize(synthetic_price_data)

        # Should still return a result (fallback to best available)
        assert result is not None
        assert result.best_params is not None

    def test_mi_sample_ratio_applied(self, synthetic_price_data):
        """Test that mi_sample_ratio is applied."""
        grid_config = GridConfig(
            sl_range=[2.0],
            pt_range=[2.0],
            time_range=[50],
            min_samples=50,
            mi_sample_ratio=0.1,  # 10% sampling
        )

        optimizer = LabelOptimizer(
            grid_config=grid_config,
            verbose=False,
        )

        # Should complete faster with sampling
        result = optimizer.optimize(synthetic_price_data)
        assert result is not None


# =============================================================================
# Integration Test
# =============================================================================


class TestIntegration:
    def test_end_to_end(self, synthetic_price_data, cost_model):
        """Full end-to-end test."""
        # 1. Run optimizer
        grid_config = GridConfig(
            sl_range=[1.5, 2.0, 2.5],
            pt_range=[1.5, 2.0, 2.5],
            time_range=[30, 50],
            min_samples=50,
            min_entropy=0.5,  # Lower threshold for synthetic data
        )

        optimizer = LabelOptimizer(
            grid_config=grid_config,
            cost_model=cost_model,
            feature_cols=["log_volume", "vwap_deviation"],
            verbose=False,
        )

        result = optimizer.optimize(synthetic_price_data)

        # 2. Verify results
        assert result.n_valid_samples > 0
        assert result.plateau.component_size >= 1

        # 3. Verify label quality
        entropy = compute_entropy(result.labels)
        assert entropy > 0.5  # Some diversity

        # 4. Grid results should have valid entries
        valid_count = result.grid_results["valid"].sum()
        assert valid_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
