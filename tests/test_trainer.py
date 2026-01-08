"""
Tests for Fine-tuning Trainer Module (v4.6)
"""

import numpy as np
import pytest
import torch
import tempfile
from pathlib import Path

from research.ssl.model import PatchTSTConfig, PatchTSTEncoder, SSLModel
from research.trainer.dataset import (
    FineTuningConfig,
    FineTuningDataset,
)
from research.trainer.model import (
    ResidualBottleneckHead,
    PatchTSTClassifier,
    ClassifierConfig,
)
from research.trainer.loss import (
    FocalLoss,
    CrossEntropyWithSmoothing,
)
from research.trainer.train import (
    FineTuningTrainingConfig,
    FineTuningTrainer,
    EarlyStopping,
    InsufficientDataError,
    finetune,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def encoder_config():
    """Small encoder config for testing."""
    return PatchTSTConfig(
        n_features=8,
        context_len=64,
        patch_len=16,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    )


@pytest.fixture
def encoder(encoder_config):
    """Pre-trained encoder for testing."""
    return PatchTSTEncoder(encoder_config)


@pytest.fixture
def synthetic_features():
    """Generate synthetic features for testing."""
    np.random.seed(42)
    n_samples = 200
    n_features = 8

    features = np.random.randn(n_samples, n_features).astype(np.float32)
    return features


@pytest.fixture
def synthetic_labels():
    """Generate synthetic labels for testing."""
    np.random.seed(42)
    n_samples = 200

    # Indices where labels are valid (after context_len)
    valid_indices = np.arange(64, n_samples)  # context_len = 64

    # Labels: {-1, 0, +1}
    labels = np.random.choice([-1, 0, 1], size=len(valid_indices))

    return valid_indices, labels


@pytest.fixture
def small_dataset(synthetic_features, synthetic_labels):
    """Small dataset for quick tests."""
    valid_indices, labels = synthetic_labels
    return FineTuningDataset(
        features=synthetic_features,
        valid_indices=valid_indices,
        labels=labels,
        context_len=64,
    )


# =============================================================================
# Test FineTuningConfig
# =============================================================================


class TestFineTuningConfig:
    def test_default_config(self):
        """Test default configuration values."""
        config = FineTuningConfig()
        assert config.context_len == 512
        assert config.n_features == 29
        assert len(config.feature_cols) == 29

    def test_feature_cols_content(self):
        """Test feature column names."""
        config = FineTuningConfig()
        assert "log_volume" in config.feature_cols
        assert "realized_vol" in config.feature_cols
        assert "connors_rsi" in config.feature_cols


# =============================================================================
# Test FineTuningDataset
# =============================================================================


class TestFineTuningDataset:
    def test_dataset_creation(self, synthetic_features, synthetic_labels):
        """Test dataset creation."""
        valid_indices, labels = synthetic_labels
        dataset = FineTuningDataset(
            features=synthetic_features,
            valid_indices=valid_indices,
            labels=labels,
            context_len=64,
        )

        assert len(dataset) == len(valid_indices)
        assert dataset.n_features == 8

    def test_getitem_shapes(self, small_dataset):
        """Test item shapes."""
        x, y = small_dataset[0]

        # Channel-first: (n_features, context_len)
        assert x.shape == (8, 64)
        assert x.dtype == torch.float32

        # Label: scalar
        assert y.shape == ()
        assert y.dtype == torch.long

    def test_label_shift(self, small_dataset):
        """Test that labels are shifted from {-1,0,1} to {0,1,2}."""
        for i in range(min(10, len(small_dataset))):
            _, y = small_dataset[i]
            assert y.item() in [0, 1, 2]

    def test_context_window_continuous(self, synthetic_features, synthetic_labels):
        """Test that context windows are continuous (no gaps)."""
        valid_indices, labels = synthetic_labels
        dataset = FineTuningDataset(
            features=synthetic_features,
            valid_indices=valid_indices,
            labels=labels,
            context_len=64,
        )

        # Get a sample
        x, _ = dataset[0]
        target_idx = dataset.usable_indices[0]

        # Reconstruct expected window
        expected = synthetic_features[target_idx - 64:target_idx].T

        assert np.allclose(x.numpy(), expected)

    def test_insufficient_context_filtered(self):
        """Test that indices without enough context are filtered."""
        features = np.random.randn(100, 8).astype(np.float32)

        # Some indices before context_len
        valid_indices = np.array([10, 30, 50, 70, 90])  # 10, 30, 50 < 64
        labels = np.array([1, 0, -1, 1, 0])

        dataset = FineTuningDataset(
            features=features,
            valid_indices=valid_indices,
            labels=labels,
            context_len=64,
        )

        # Only indices >= 64 should remain
        assert len(dataset) == 2  # 70 and 90

    def test_length_mismatch_error(self, synthetic_features):
        """Test error when indices and labels have different lengths."""
        valid_indices = np.arange(64, 100)
        labels = np.random.choice([-1, 0, 1], size=10)  # Wrong length

        with pytest.raises(ValueError, match="length"):
            FineTuningDataset(
                features=synthetic_features,
                valid_indices=valid_indices,
                labels=labels,
                context_len=64,
            )

    def test_no_usable_samples_error(self, synthetic_features):
        """Test error when no samples have enough context."""
        valid_indices = np.arange(0, 50)  # All < 64
        labels = np.random.choice([-1, 0, 1], size=50)

        with pytest.raises(ValueError, match="No usable samples"):
            FineTuningDataset(
                features=synthetic_features,
                valid_indices=valid_indices,
                labels=labels,
                context_len=64,
            )

    def test_class_distribution(self, small_dataset):
        """Test class distribution computation."""
        dist = small_dataset.get_class_distribution()

        # Should have keys for each unique class
        for label in dist:
            assert "count" in dist[label]
            assert "ratio" in dist[label]
            assert dist[label]["ratio"] > 0

    def test_class_weights(self, small_dataset):
        """Test class weights computation."""
        weights = small_dataset.get_class_weights()

        assert weights.shape == (3,)
        assert weights.dtype == torch.float32
        assert torch.all(weights > 0)


# =============================================================================
# Test ResidualBottleneckHead
# =============================================================================


class TestResidualBottleneckHead:
    def test_forward_shape(self):
        """Test bottleneck head output shape."""
        head = ResidualBottleneckHead(
            input_dim=256,
            n_classes=3,
            bottleneck_dim=64,
            dropout=0.0,
        )

        x = torch.randn(4, 256)
        logits = head(x)

        assert logits.shape == (4, 3)

    def test_residual_connection(self):
        """Test that residual connection is applied."""
        head = ResidualBottleneckHead(
            input_dim=256,
            n_classes=3,
            bottleneck_dim=64,
            dropout=0.0,
        )

        # With residual, output should be different from pure bottleneck
        x = torch.ones(1, 256)
        logits = head(x)

        assert not torch.allclose(logits, torch.zeros_like(logits))


# =============================================================================
# Test PatchTSTClassifier
# =============================================================================


class TestPatchTSTClassifier:
    def test_forward_shape(self, encoder):
        """Test classifier output shape."""
        classifier = PatchTSTClassifier(
            encoder=encoder,
            n_classes=3,
            bottleneck_dim=64,
            dropout=0.0,
        )

        x = torch.randn(2, 8, 64)  # (batch, n_features, context_len)
        logits = classifier(x)

        assert logits.shape == (2, 3)

    def test_freeze_encoder(self, encoder):
        """Test encoder freezing."""
        classifier = PatchTSTClassifier(encoder=encoder, n_classes=3)
        classifier.freeze_encoder()

        for param in classifier.encoder.parameters():
            assert not param.requires_grad

        # Classifier params should still be trainable
        for param in classifier.classifier.parameters():
            assert param.requires_grad

    def test_unfreeze_encoder(self, encoder):
        """Test encoder unfreezing."""
        classifier = PatchTSTClassifier(encoder=encoder, n_classes=3)
        classifier.freeze_encoder()
        classifier.unfreeze_encoder()

        for param in classifier.encoder.parameters():
            assert param.requires_grad

    def test_save_load(self, encoder):
        """Test model save and load."""
        classifier = PatchTSTClassifier(encoder=encoder, n_classes=3)

        x = torch.randn(1, 8, 64)
        classifier.eval()
        with torch.no_grad():
            original_output = classifier(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "classifier.pt"
            classifier.save(str(path))

            loaded = PatchTSTClassifier.load(str(path))
            loaded.eval()
            with torch.no_grad():
                loaded_output = loaded(x)

            assert torch.allclose(original_output, loaded_output)

    def test_from_pretrained(self, encoder_config):
        """Test creating classifier from pretrained encoder."""
        # Create and save SSL model
        ssl_model = SSLModel(encoder_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            encoder_path = Path(tmpdir) / "encoder.pt"
            ssl_model.save_encoder(str(encoder_path))

            # Create classifier from pretrained
            classifier = PatchTSTClassifier.from_pretrained(
                str(encoder_path),
                n_classes=3,
            )

            x = torch.randn(1, 8, 64)
            logits = classifier(x)

            assert logits.shape == (1, 3)


# =============================================================================
# Test FocalLoss
# =============================================================================


class TestFocalLoss:
    def test_loss_computation(self):
        """Test focal loss computation."""
        loss_fn = FocalLoss(gamma=2.0, label_smoothing=0.1)

        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])

        loss = loss_fn(logits, targets)

        assert loss.shape == ()
        assert loss.item() > 0

    def test_gamma_effect(self):
        """Test that higher gamma reduces easy example weights."""
        logits = torch.tensor([[5.0, 0.0, 0.0]])  # High confidence for class 0
        targets = torch.tensor([0])

        loss_gamma0 = FocalLoss(gamma=0.0, label_smoothing=0.0)(logits, targets)
        loss_gamma2 = FocalLoss(gamma=2.0, label_smoothing=0.0)(logits, targets)

        # Higher gamma should reduce loss for easy examples
        assert loss_gamma2 < loss_gamma0

    def test_label_smoothing_effect(self):
        """Test label smoothing reduces overconfidence."""
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])

        loss_no_smooth = FocalLoss(gamma=0.0, label_smoothing=0.0)(logits, targets)
        loss_smooth = FocalLoss(gamma=0.0, label_smoothing=0.1)(logits, targets)

        # Both should produce valid losses
        assert loss_no_smooth.item() > 0
        assert loss_smooth.item() > 0

    def test_class_weights(self):
        """Test class weights are applied."""
        weights = torch.tensor([1.0, 2.0, 0.5])

        loss_fn = FocalLoss(gamma=2.0, weight=weights)

        logits = torch.randn(3, 3)
        targets = torch.tensor([0, 1, 2])

        loss = loss_fn(logits, targets)

        assert loss.shape == ()
        assert loss.item() > 0

    def test_reduction_modes(self):
        """Test different reduction modes."""
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])

        loss_mean = FocalLoss(reduction="mean")(logits, targets)
        loss_sum = FocalLoss(reduction="sum")(logits, targets)
        loss_none = FocalLoss(reduction="none")(logits, targets)

        assert loss_mean.shape == ()
        assert loss_sum.shape == ()
        assert loss_none.shape == (4,)


class TestCrossEntropyWithSmoothing:
    def test_loss_computation(self):
        """Test cross entropy with smoothing."""
        loss_fn = CrossEntropyWithSmoothing(label_smoothing=0.1)

        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])

        loss = loss_fn(logits, targets)

        assert loss.shape == ()
        assert loss.item() > 0

    def test_matches_standard_ce_without_smoothing(self):
        """Test that it matches standard CE when smoothing=0."""
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])

        custom_loss = CrossEntropyWithSmoothing(label_smoothing=0.0)(logits, targets)
        standard_loss = torch.nn.functional.cross_entropy(logits, targets)

        assert torch.isclose(custom_loss, standard_loss, atol=1e-5)


# =============================================================================
# Test EarlyStopping
# =============================================================================


class TestEarlyStopping:
    def test_no_stop_improving(self):
        """Test that early stopping doesn't trigger when improving."""
        es = EarlyStopping(patience=3, min_delta=0.01)
        model = torch.nn.Linear(10, 10)

        scores = [0.5, 0.6, 0.7, 0.8]  # Improving (higher is better)
        for score in scores:
            should_stop = es(score, model)
            assert not should_stop

    def test_stop_not_improving(self):
        """Test that early stopping triggers when not improving."""
        es = EarlyStopping(patience=3, min_delta=0.01)
        model = torch.nn.Linear(10, 10)

        # patience=3: stops after 3 non-improving epochs
        # idx 0: best=0.8, counter=0
        # idx 1: 0.7 < 0.79, counter=1
        # idx 2: 0.7 < 0.79, counter=2
        # idx 3: 0.7 < 0.79, counter=3 → stop
        scores = [0.8, 0.7, 0.7, 0.7, 0.7]  # Plateau/declining
        for i, score in enumerate(scores):
            should_stop = es(score, model)
            if i < 3:  # First 3 epochs (idx 0, 1, 2)
                assert not should_stop
            else:
                assert should_stop

    def test_restore_best(self):
        """Test restoring best model state."""
        es = EarlyStopping(patience=3)
        model = torch.nn.Linear(2, 2)

        # Record initial weights
        initial_weight = model.weight.clone()

        # Best score
        es(0.9, model)

        # Change weights and report worse scores
        model.weight.data.fill_(99)
        es(0.7, model)
        es(0.7, model)
        es(0.7, model)

        # Restore
        es.restore_best(model)

        assert torch.allclose(model.weight, initial_weight)


# =============================================================================
# Test FineTuningTrainer
# =============================================================================


class TestFineTuningTrainer:
    def test_insufficient_data_error(self):
        """Test fail-fast for insufficient data."""
        config = FineTuningTrainingConfig(
            min_samples=1000,
            frozen_epochs=1,
            total_epochs=2,
        )
        trainer = FineTuningTrainer(config)

        with pytest.raises(InsufficientDataError):
            trainer.check_data(100)

    def test_training_small_data(self, encoder, small_dataset):
        """Test training loop with small data."""
        config = FineTuningTrainingConfig(
            frozen_epochs=1,
            total_epochs=2,
            batch_size=4,
            min_samples=10,
            patience=10,
            device="cpu",
        )

        classifier = PatchTSTClassifier(
            encoder=encoder,
            n_classes=3,
            bottleneck_dim=32,
            dropout=0.0,
        )

        trainer = FineTuningTrainer(config)
        model = trainer.train(
            classifier,
            small_dataset,
            val_dataset=small_dataset,  # Use same for simplicity
        )

        assert model is not None
        assert len(trainer.loss_history) > 0

    def test_two_phase_training(self, encoder, small_dataset):
        """Test that 2-phase training works correctly."""
        config = FineTuningTrainingConfig(
            frozen_epochs=2,
            total_epochs=4,
            batch_size=4,
            min_samples=10,
            patience=10,
            device="cpu",
        )

        classifier = PatchTSTClassifier(
            encoder=encoder,
            n_classes=3,
            dropout=0.0,
        )

        trainer = FineTuningTrainer(config)
        model = trainer.train(
            classifier,
            small_dataset,
            val_dataset=small_dataset,
        )

        # Should have completed training
        assert len(trainer.loss_history) >= 2

    def test_model_save_during_training(self, encoder, small_dataset):
        """Test that model is saved correctly."""
        config = FineTuningTrainingConfig(
            frozen_epochs=1,
            total_epochs=2,
            batch_size=4,
            min_samples=10,
            patience=10,
            device="cpu",
        )

        classifier = PatchTSTClassifier(
            encoder=encoder,
            n_classes=3,
            dropout=0.0,
        )

        trainer = FineTuningTrainer(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            model = trainer.train(
                classifier,
                small_dataset,
                val_dataset=small_dataset,
                output_path=str(path),
            )

            assert path.exists()

            # Load and verify
            loaded = PatchTSTClassifier.load(str(path))
            assert loaded is not None

    def test_training_summary(self, encoder, small_dataset):
        """Test training summary generation."""
        config = FineTuningTrainingConfig(
            frozen_epochs=1,
            total_epochs=2,
            batch_size=4,
            min_samples=10,
            patience=10,
            device="cpu",
        )

        classifier = PatchTSTClassifier(
            encoder=encoder,
            n_classes=3,
            dropout=0.0,
        )

        trainer = FineTuningTrainer(config)
        trainer.train(
            classifier,
            small_dataset,
            val_dataset=small_dataset,
        )

        summary = trainer.get_training_summary()

        assert "total_epochs" in summary
        assert "final_train_loss" in summary
        assert "final_val_metrics" in summary
        assert "loss_history" in summary


# =============================================================================
# Test finetune convenience function
# =============================================================================


class TestFinetune:
    def test_finetune_function(self, encoder_config, small_dataset):
        """Test finetune convenience function."""
        # Create and save SSL model
        ssl_model = SSLModel(encoder_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            encoder_path = Path(tmpdir) / "encoder.pt"
            ssl_model.save_encoder(str(encoder_path))

            config = FineTuningTrainingConfig(
                frozen_epochs=1,
                total_epochs=2,
                batch_size=4,
                min_samples=10,
                patience=10,
                device="cpu",
            )

            model = finetune(
                str(encoder_path),
                small_dataset,
                val_dataset=small_dataset,
                config=config,
            )

            assert model is not None
            assert isinstance(model, PatchTSTClassifier)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    def test_end_to_end_pipeline(self, encoder_config):
        """Test complete fine-tuning pipeline."""
        # 1. Create synthetic data
        np.random.seed(42)
        n_total = 300
        n_features = 8
        context_len = 64

        features = np.random.randn(n_total, n_features).astype(np.float32)
        valid_indices = np.arange(context_len, n_total - 10)
        labels = np.random.choice([-1, 0, 1], size=len(valid_indices))

        # 2. Create dataset
        dataset = FineTuningDataset(
            features=features,
            valid_indices=valid_indices,
            labels=labels,
            context_len=context_len,
        )

        # 3. Create SSL model and save encoder
        ssl_model = SSLModel(encoder_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            encoder_path = Path(tmpdir) / "encoder.pt"
            model_path = Path(tmpdir) / "classifier.pt"

            ssl_model.save_encoder(str(encoder_path))

            # 4. Create classifier from pretrained
            classifier = PatchTSTClassifier.from_pretrained(
                str(encoder_path),
                n_classes=3,
            )

            # 5. Train
            config = FineTuningTrainingConfig(
                frozen_epochs=1,
                total_epochs=3,
                batch_size=8,
                min_samples=10,
                patience=10,
                device="cpu",
            )

            trainer = FineTuningTrainer(config)
            trained_model = trainer.train(
                classifier,
                dataset,
                val_dataset=dataset,
                class_weights=dataset.get_class_weights(),
                output_path=str(model_path),
            )

            # 6. Verify inference
            x = torch.randn(1, n_features, context_len)
            trained_model.eval()
            with torch.no_grad():
                logits = trained_model(x)

            assert logits.shape == (1, 3)

            # 7. Verify saved model
            loaded = PatchTSTClassifier.load(str(model_path))
            loaded.eval()
            with torch.no_grad():
                loaded_logits = loaded(x)

            assert torch.allclose(logits, loaded_logits)

    def test_gradient_flow(self, encoder):
        """Test that gradients flow correctly."""
        classifier = PatchTSTClassifier(
            encoder=encoder,
            n_classes=3,
            dropout=0.0,
        )

        loss_fn = FocalLoss(gamma=2.0)

        # Forward pass
        x = torch.randn(2, 8, 64)
        y = torch.tensor([0, 1])

        logits = classifier(x)
        loss = loss_fn(logits, y)

        # Backward pass
        loss.backward()

        # Check gradients exist
        for name, param in classifier.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_frozen_phase_no_encoder_gradients(self, encoder):
        """Test that encoder has no gradients when frozen."""
        classifier = PatchTSTClassifier(
            encoder=encoder,
            n_classes=3,
            dropout=0.0,
        )

        # Freeze encoder
        classifier.freeze_encoder()

        loss_fn = FocalLoss(gamma=2.0)

        x = torch.randn(2, 8, 64)
        y = torch.tensor([0, 1])

        logits = classifier(x)
        loss = loss_fn(logits, y)
        loss.backward()

        # Encoder should have no gradients
        for name, param in classifier.encoder.named_parameters():
            assert param.grad is None, f"Encoder param {name} has gradient when frozen"

        # Classifier head should have gradients
        for name, param in classifier.classifier.named_parameters():
            assert param.grad is not None, f"Classifier param {name} missing gradient"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
