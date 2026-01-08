"""
Tests for Sniper v4.9 modules.

Tests cover:
- SniperLabeler: TBM labeling with min_expected_return filter
- SniperDataset: Dataset with WeightedRandomSampler
- SniperClassifier: PatchTST + Channel Mixing Head
- SniperTrainer: Training with balanced sampling
- ConfidenceSignalGenerator: Confidence threshold filtering
"""

import pytest
import numpy as np
import pandas as pd
import torch

from research.sniper.labeler import SniperLabeler, SniperTBMConfig, LabelResult
from research.sniper.model import SniperClassifier, SniperModelConfig
from research.sniper.dataset import (
    SniperDataset,
    create_balanced_dataloader,
    create_validation_dataloader,
    train_val_split,
)
from research.sniper.trainer import SniperTrainer, SniperTrainingConfig, FocalLoss
from research.sniper.signal_generator import (
    ConfidenceSignalGenerator,
    count_signals,
    compute_signal_stats,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_bars_df():
    """Create sample bars dataframe for testing."""
    np.random.seed(42)
    n = 500

    # Generate price data with some trends
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    highs = prices + np.abs(np.random.randn(n) * 0.3)
    lows = prices - np.abs(np.random.randn(n) * 0.3)

    # Generate timestamps (1 minute apart)
    start_time = 1609459200000  # 2021-01-01 00:00:00
    times = [start_time + i * 60000 for i in range(n)]

    df = pd.DataFrame({
        "start_time": times,
        "open": prices,
        "high": highs,
        "high_time": [t + 30000 for t in times],  # 30 seconds after start
        "low": lows,
        "low_time": [t + 45000 for t in times],   # 45 seconds after start
        "close": prices + np.random.randn(n) * 0.1,
        "volume": np.random.uniform(100, 1000, n),
        "realized_vol": np.abs(np.random.randn(n) * 0.02),
    })

    return df


@pytest.fixture
def sample_features():
    """Create sample features for testing."""
    np.random.seed(42)
    n = 500
    n_features = 29
    return np.random.randn(n, n_features).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Create sample labels for testing."""
    np.random.seed(42)
    n = 500
    labels = np.random.choice([-1, 0, 1], size=n, p=[0.15, 0.70, 0.15])
    # Set first few to NaN (warmup)
    labels = labels.astype(np.float32)
    labels[:50] = np.nan
    return labels


# =============================================================================
# Labeler Tests
# =============================================================================

class TestSniperLabeler:
    """Tests for SniperLabeler."""

    def test_label_basic(self, sample_bars_df):
        """Test basic labeling functionality."""
        labeler = SniperLabeler(SniperTBMConfig())
        result = labeler.label(sample_bars_df)

        assert isinstance(result, LabelResult)
        assert len(result.labels) == len(sample_bars_df)
        assert result.stats["total"] > 0

    def test_label_distribution(self, sample_bars_df):
        """Test that labels are within expected range."""
        labeler = SniperLabeler(SniperTBMConfig())
        result = labeler.label(sample_bars_df)

        # Check label values
        valid_labels = result.labels[~np.isnan(result.labels)]
        assert set(valid_labels).issubset({-1, 0, 1})

    def test_min_expected_return_filter(self, sample_bars_df):
        """Test that min_expected_return filters low-potential trades."""
        # With high min_expected_return, should have more neutrals
        config_high = SniperTBMConfig(min_expected_return=0.01)  # 1%
        config_low = SniperTBMConfig(min_expected_return=0.001)  # 0.1%

        labeler_high = SniperLabeler(config_high)
        labeler_low = SniperLabeler(config_low)

        result_high = labeler_high.label(sample_bars_df)
        result_low = labeler_low.label(sample_bars_df)

        # High threshold should have more neutrals
        assert result_high.stats["neutral"] >= result_low.stats["neutral"]

    def test_timeout_effect(self, sample_bars_df):
        """Test that max_hold_bars affects labeling."""
        config_short = SniperTBMConfig(max_hold_bars=5)
        config_long = SniperTBMConfig(max_hold_bars=50)

        labeler_short = SniperLabeler(config_short)
        labeler_long = SniperLabeler(config_long)

        result_short = labeler_short.label(sample_bars_df)
        result_long = labeler_long.label(sample_bars_df)

        # Short timeout may result in different distribution
        # At minimum, both should produce valid results
        assert result_short.stats["total"] > 0
        assert result_long.stats["total"] > 0


# =============================================================================
# Dataset Tests
# =============================================================================

class TestSniperDataset:
    """Tests for SniperDataset."""

    def test_dataset_creation(self, sample_features, sample_labels):
        """Test dataset creation."""
        dataset = SniperDataset(
            features=sample_features,
            labels=sample_labels,
            context_len=64,
        )

        assert len(dataset) > 0
        assert dataset.n_features == sample_features.shape[1]

    def test_dataset_getitem(self, sample_features, sample_labels):
        """Test __getitem__ returns correct shapes."""
        context_len = 64
        dataset = SniperDataset(
            features=sample_features,
            labels=sample_labels,
            context_len=context_len,
        )

        x, y = dataset[0]

        # x should be (n_features, context_len)
        assert x.shape == (sample_features.shape[1], context_len)
        assert x.dtype == torch.float32

        # y should be scalar in {0, 1, 2}
        assert y.dim() == 0
        assert y.dtype == torch.long
        assert y.item() in {0, 1, 2}

    def test_label_conversion(self, sample_features, sample_labels):
        """Test that labels are converted from {-1,0,1} to {0,1,2}."""
        dataset = SniperDataset(
            features=sample_features,
            labels=sample_labels,
            context_len=64,
        )

        # Original labels are {-1, 0, 1}
        # Dataset should convert to {0, 1, 2}
        for i in range(min(10, len(dataset))):
            _, y = dataset[i]
            assert y.item() in {0, 1, 2}

    def test_class_counts(self, sample_features, sample_labels):
        """Test get_class_counts."""
        dataset = SniperDataset(
            features=sample_features,
            labels=sample_labels,
            context_len=64,
        )

        counts = dataset.get_class_counts()
        assert len(counts) == 3
        assert counts.sum() == len(dataset)

    def test_balanced_sampler(self, sample_features, sample_labels):
        """Test WeightedRandomSampler creation."""
        dataset = SniperDataset(
            features=sample_features,
            labels=sample_labels,
            context_len=64,
        )

        sampler = dataset.get_balanced_sampler()
        assert sampler is not None
        assert len(sampler) == len(dataset)

    def test_balanced_dataloader(self, sample_features, sample_labels):
        """Test balanced dataloader."""
        dataset = SniperDataset(
            features=sample_features,
            labels=sample_labels,
            context_len=64,
        )

        loader = create_balanced_dataloader(dataset, batch_size=32)
        batch_x, batch_y = next(iter(loader))

        assert batch_x.shape[0] == 32
        assert batch_y.shape[0] == 32

    def test_train_val_split(self, sample_features, sample_labels):
        """Test temporal train/val split."""
        train_ds, val_ds = train_val_split(
            features=sample_features,
            labels=sample_labels,
            context_len=64,
            val_ratio=0.2,
        )

        assert len(train_ds) > 0
        assert len(val_ds) > 0
        # Train should be larger
        assert len(train_ds) > len(val_ds)


# =============================================================================
# Model Tests
# =============================================================================

class TestSniperClassifier:
    """Tests for SniperClassifier."""

    def test_model_creation(self):
        """Test model creation."""
        config = SniperModelConfig(
            n_features=29,
            context_len=128,
            d_model=64,
            n_layers=2,
        )
        model = SniperClassifier(config)

        assert model is not None
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_forward_pass(self):
        """Test forward pass."""
        config = SniperModelConfig(
            n_features=29,
            context_len=128,
            d_model=64,
            n_layers=2,
        )
        model = SniperClassifier(config)
        model.eval()

        # Input: (batch, n_features, context_len)
        x = torch.randn(4, 29, 128)
        logits = model(x)

        # Output: (batch, n_classes)
        assert logits.shape == (4, 3)

    def test_predict_proba(self):
        """Test probability prediction."""
        config = SniperModelConfig(
            n_features=29,
            context_len=128,
        )
        model = SniperClassifier(config)
        model.eval()

        x = torch.randn(4, 29, 128)
        probs = model.predict_proba(x)

        # Probabilities should sum to 1
        assert probs.shape == (4, 3)
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-6)

    def test_predict(self):
        """Test class prediction."""
        config = SniperModelConfig(
            n_features=29,
            context_len=128,
        )
        model = SniperClassifier(config)
        model.eval()

        x = torch.randn(4, 29, 128)
        preds = model.predict(x)

        # Predictions should be in {0, 1, 2}
        assert preds.shape == (4,)
        assert all(p.item() in {0, 1, 2} for p in preds)

    def test_save_load(self, tmp_path):
        """Test model save and load."""
        config = SniperModelConfig(
            n_features=29,
            context_len=128,
        )
        model = SniperClassifier(config)

        # Save
        path = tmp_path / "model.pt"
        model.save(str(path))

        # Load
        loaded = SniperClassifier.load(str(path))

        # Check same config
        assert loaded.config.n_features == config.n_features
        assert loaded.config.context_len == config.context_len


# =============================================================================
# Trainer Tests
# =============================================================================

class TestSniperTrainer:
    """Tests for SniperTrainer."""

    def test_focal_loss(self):
        """Test FocalLoss computation."""
        loss_fn = FocalLoss(gamma=2.0, smoothing=0.05)

        logits = torch.randn(16, 3)
        targets = torch.randint(0, 3, (16,))

        loss = loss_fn(logits, targets)

        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0

    def test_trainer_one_epoch(self, sample_features, sample_labels):
        """Test training for one epoch."""
        # Create dataset
        train_ds = SniperDataset(
            features=sample_features,
            labels=sample_labels,
            context_len=64,
        )

        # Small model for testing
        model = SniperClassifier(SniperModelConfig(
            n_features=sample_features.shape[1],
            context_len=64,
            d_model=32,
            n_layers=1,
        ))

        # Train for 1 epoch
        trainer = SniperTrainer(SniperTrainingConfig(
            epochs=1,
            batch_size=32,
            learning_rate=1e-3,
            device="cpu",
        ))

        trained_model = trainer.train(model, train_ds)

        assert trained_model is not None
        assert len(trainer.history) == 1

    def test_training_summary(self, sample_features, sample_labels):
        """Test get_training_summary."""
        train_ds, val_ds = train_val_split(
            features=sample_features,
            labels=sample_labels,
            context_len=64,
            val_ratio=0.2,
        )

        model = SniperClassifier(SniperModelConfig(
            n_features=sample_features.shape[1],
            context_len=64,
            d_model=32,
            n_layers=1,
        ))

        trainer = SniperTrainer(SniperTrainingConfig(
            epochs=2,
            batch_size=32,
            device="cpu",
        ))

        trainer.train(model, train_ds, val_ds)
        summary = trainer.get_training_summary()

        assert "epochs" in summary
        assert "best_val_f1" in summary
        assert summary["epochs"] == 2


# =============================================================================
# Signal Generator Tests
# =============================================================================

class TestConfidenceSignalGenerator:
    """Tests for ConfidenceSignalGenerator."""

    def test_generate_signals(self, sample_features):
        """Test signal generation."""
        model = SniperClassifier(SniperModelConfig(
            n_features=sample_features.shape[1],
            context_len=64,
            d_model=32,
            n_layers=1,
        ))

        generator = ConfidenceSignalGenerator(
            model=model,
            context_len=64,
            confidence_threshold=0.6,
            device="cpu",
        )

        signals, confidences, probs = generator.generate(sample_features)

        assert signals.shape == (len(sample_features),)
        assert confidences.shape == (len(sample_features),)
        assert probs.shape == (len(sample_features), 3)

        # Signals should be in {-1, 0, 1}
        assert set(signals).issubset({-1, 0, 1})

    def test_warmup_period(self, sample_features):
        """Test that warmup period has no signals."""
        context_len = 64
        model = SniperClassifier(SniperModelConfig(
            n_features=sample_features.shape[1],
            context_len=context_len,
            d_model=32,
            n_layers=1,
        ))

        generator = ConfidenceSignalGenerator(
            model=model,
            context_len=context_len,
            confidence_threshold=0.6,
            device="cpu",
        )

        signals, _, _ = generator.generate(sample_features)

        # First context_len signals should be 0
        assert all(signals[:context_len] == 0)

    def test_confidence_threshold(self, sample_features):
        """Test that higher threshold produces fewer signals."""
        model = SniperClassifier(SniperModelConfig(
            n_features=sample_features.shape[1],
            context_len=64,
            d_model=32,
            n_layers=1,
        ))

        gen_low = ConfidenceSignalGenerator(
            model=model,
            context_len=64,
            confidence_threshold=0.4,
            device="cpu",
        )

        gen_high = ConfidenceSignalGenerator(
            model=model,
            context_len=64,
            confidence_threshold=0.9,
            device="cpu",
        )

        signals_low, _, _ = gen_low.generate(sample_features)
        signals_high, _, _ = gen_high.generate(sample_features)

        # Higher threshold should have fewer or equal non-neutral signals
        n_active_low = (signals_low != 0).sum()
        n_active_high = (signals_high != 0).sum()
        assert n_active_high <= n_active_low


class TestSignalUtils:
    """Tests for signal utility functions."""

    def test_count_signals(self):
        """Test count_signals function."""
        signals = np.array([1, 1, 0, 0, 0, -1, -1, 0])
        counts = count_signals(signals)

        assert counts["long"] == 2
        assert counts["short"] == 2
        assert counts["neutral"] == 4
        assert counts["total"] == 8

    def test_compute_signal_stats(self):
        """Test compute_signal_stats function."""
        signals = np.zeros(200, dtype=np.int8)
        signals[150:160] = 1   # 10 longs
        signals[170:175] = -1  # 5 shorts

        stats = compute_signal_stats(signals, context_len=128)

        assert stats["counts"]["long"] == 10
        assert stats["counts"]["short"] == 5
        assert "ratios" in stats
        assert "transitions" in stats


# =============================================================================
# Integration Tests
# =============================================================================

class TestSniperIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline(self, sample_bars_df, sample_features):
        """Test full labeling -> training -> signal generation."""
        # Step 1: Label
        labeler = SniperLabeler(SniperTBMConfig())
        label_result = labeler.label(sample_bars_df)

        # Step 2: Create dataset
        # Align features with labels (use same length)
        n = min(len(sample_features), len(label_result.labels))
        features = sample_features[:n]
        labels = label_result.labels[:n]

        train_ds, val_ds = train_val_split(
            features=features,
            labels=labels,
            context_len=64,
            val_ratio=0.2,
        )

        # Step 3: Train model
        model = SniperClassifier(SniperModelConfig(
            n_features=features.shape[1],
            context_len=64,
            d_model=32,
            n_layers=1,
        ))

        trainer = SniperTrainer(SniperTrainingConfig(
            epochs=2,
            batch_size=32,
            device="cpu",
        ))

        trained_model = trainer.train(model, train_ds, val_ds)

        # Step 4: Generate signals
        generator = ConfidenceSignalGenerator(
            model=trained_model,
            context_len=64,
            confidence_threshold=0.6,
            device="cpu",
        )

        signals, confidences, probs = generator.generate(features)

        # Verify outputs
        assert len(signals) == n
        assert set(signals).issubset({-1, 0, 1})
        assert trainer.get_training_summary()["best_val_f1"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
