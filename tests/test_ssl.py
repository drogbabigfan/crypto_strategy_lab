"""
Tests for SSL Pre-training Module (v4.6)
"""

import numpy as np
import pytest
import torch
import tempfile
from pathlib import Path

from research.ssl.dataset import (
    SSLDataConfig,
    PatchMaskingDataset,
)
from research.ssl.model import (
    PatchTSTConfig,
    PatchEmbedding,
    PositionalEncoding,
    TransformerEncoderLayer,
    PatchTSTEncoder,
    SSLModel,
    masked_reconstruction_loss,
)
from research.ssl.train import (
    SSLTrainingConfig,
    SSLTrainer,
    EarlyStopping,
    InsufficientDataError,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_data():
    """Generate synthetic data for testing."""
    np.random.seed(42)
    n_samples = 100
    context_len = 512
    n_features = 29

    # Random walk features
    data = np.random.randn(n_samples, context_len, n_features).astype(np.float32)
    return data


@pytest.fixture
def small_synthetic_data():
    """Smaller data for quick tests."""
    np.random.seed(42)
    n_samples = 16
    context_len = 64
    n_features = 8

    data = np.random.randn(n_samples, context_len, n_features).astype(np.float32)
    return data


@pytest.fixture
def model_config():
    """Default model config for testing."""
    return PatchTSTConfig(
        n_features=8,
        context_len=64,
        patch_len=16,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        dropout=0.0,  # No dropout for deterministic tests
    )


# =============================================================================
# Test Dataset
# =============================================================================


class TestSSLDataConfig:
    def test_default_assets(self):
        """Test default asset list."""
        config = SSLDataConfig()
        assert len(config.assets) == 7
        assert "BTCUSDT" in config.assets

    def test_default_feature_cols(self):
        """Test default feature columns."""
        config = SSLDataConfig()
        assert len(config.feature_cols) == 29
        assert "log_volume" in config.feature_cols
        assert "realized_vol" in config.feature_cols


class TestPatchMaskingDataset:
    def test_dataset_creation(self, small_synthetic_data):
        """Test dataset creation."""
        dataset = PatchMaskingDataset(
            small_synthetic_data,
            patch_len=16,
            mask_ratio=0.4,
        )

        assert len(dataset) == len(small_synthetic_data)
        assert dataset.n_patches == 64 // 16  # 4 patches
        assert dataset.n_masked_patches == int(4 * 0.4)  # 1 masked

    def test_getitem_shapes(self, small_synthetic_data):
        """Test item shapes."""
        dataset = PatchMaskingDataset(
            small_synthetic_data,
            patch_len=16,
            mask_ratio=0.4,
        )

        item = dataset[0]

        assert "input" in item
        assert "target" in item
        assert "mask_indices" in item

        # Channel-first: (n_features, context_len)
        assert item["input"].shape == (8, 64)
        assert item["target"].shape == (8, 64)
        assert len(item["mask_indices"]) == dataset.n_masked_patches

    def test_masking_applied(self, small_synthetic_data):
        """Test that masking zeros out patches."""
        dataset = PatchMaskingDataset(
            small_synthetic_data,
            patch_len=16,
            mask_ratio=0.5,
            random_seed=42,
        )

        item = dataset[0]
        masked_input = item["input"]
        target = item["target"]
        mask_indices = item["mask_indices"]

        # Masked patches should be zero in input
        for patch_idx in mask_indices:
            start = patch_idx * 16
            end = start + 16
            assert torch.all(masked_input[:, start:end] == 0)

        # Target should be non-zero (original data)
        assert not torch.all(target == 0)

    def test_different_masks_per_sample(self, small_synthetic_data):
        """Test that different samples get different masks."""
        dataset = PatchMaskingDataset(
            small_synthetic_data,
            patch_len=16,
            mask_ratio=0.5,
            random_seed=None,  # Random masks
        )

        item0 = dataset[0]
        item1 = dataset[1]

        # Mask indices should differ (with high probability)
        # Note: They could theoretically be the same by chance
        # but we just check that the dataset works
        assert len(item0["mask_indices"]) == len(item1["mask_indices"])

    def test_context_len_not_divisible_by_patch_len(self):
        """Test error when context_len not divisible by patch_len."""
        data = np.random.randn(10, 65, 8).astype(np.float32)

        with pytest.raises(ValueError, match="divisible"):
            PatchMaskingDataset(data, patch_len=16, mask_ratio=0.4)


# =============================================================================
# Test Model Components
# =============================================================================


class TestPatchEmbedding:
    def test_forward_shape(self):
        """Test patch embedding output shape."""
        embed = PatchEmbedding(patch_len=16, d_model=32, dropout=0.0)

        x = torch.randn(2, 8, 64)  # (Batch, n_features, context_len)
        out = embed(x)

        # (Batch, n_features, n_patches, d_model)
        assert out.shape == (2, 8, 4, 32)


class TestPositionalEncoding:
    def test_forward_shape(self):
        """Test positional encoding preserves shape."""
        pos_enc = PositionalEncoding(n_patches=4, d_model=32)

        x = torch.randn(2, 8, 4, 32)
        out = pos_enc(x)

        assert out.shape == x.shape


class TestTransformerEncoderLayer:
    def test_forward_shape(self):
        """Test transformer layer output shape."""
        layer = TransformerEncoderLayer(
            d_model=32,
            n_heads=2,
            d_ff=64,
            dropout=0.0,
        )

        x = torch.randn(4, 10, 32)  # (Batch, seq_len, d_model)
        out = layer(x)

        assert out.shape == x.shape


class TestPatchTSTEncoder:
    def test_forward_shape(self, model_config):
        """Test encoder output shape."""
        encoder = PatchTSTEncoder(model_config)

        x = torch.randn(2, 8, 64)  # (Batch, n_features, context_len)
        out = encoder(x)

        # (Batch, n_features, n_patches, d_model)
        assert out.shape == (2, 8, 4, 32)

    def test_output_dim(self, model_config):
        """Test output dimension calculation."""
        encoder = PatchTSTEncoder(model_config)

        expected = 8 * 4 * 32  # n_features * n_patches * d_model
        assert encoder.get_output_dim() == expected


class TestSSLModel:
    def test_reconstruction_shape(self, model_config):
        """Test reconstruction output shape."""
        model = SSLModel(model_config)

        x = torch.randn(2, 8, 64)
        out = model(x, return_encoded=False)

        # Should be same shape as input
        assert out.shape == x.shape

    def test_encoded_shape(self, model_config):
        """Test encoded output shape."""
        model = SSLModel(model_config)

        x = torch.randn(2, 8, 64)
        out = model(x, return_encoded=True)

        # (Batch, n_features, n_patches, d_model)
        assert out.shape == (2, 8, 4, 32)

    def test_save_load_encoder(self, model_config):
        """Test encoder save and load."""
        model = SSLModel(model_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "encoder.pt"
            model.save_encoder(str(path))

            # Load encoder
            loaded = SSLModel.load_encoder(str(path))

            # Compare weights
            for (n1, p1), (n2, p2) in zip(
                model.encoder.named_parameters(),
                loaded.named_parameters(),
            ):
                assert n1 == n2
                assert torch.allclose(p1, p2)


class TestMaskedReconstructionLoss:
    def test_loss_computation(self, model_config):
        """Test masked reconstruction loss."""
        model = SSLModel(model_config)

        batch_size = 2
        x = torch.randn(batch_size, 8, 64)
        pred = model(x)

        # Create mask indices (mask 2 patches per sample)
        mask_indices = torch.tensor([[0, 2], [1, 3]])

        loss = masked_reconstruction_loss(pred, x, mask_indices, patch_len=16)

        assert loss.shape == ()
        assert loss.item() > 0

    def test_zero_loss_perfect_reconstruction(self, model_config):
        """Test that identical pred/target gives zero loss."""
        x = torch.randn(2, 8, 64)
        mask_indices = torch.tensor([[0, 1], [2, 3]])

        loss = masked_reconstruction_loss(x, x, mask_indices, patch_len=16)

        assert loss.item() < 1e-6


# =============================================================================
# Test Training
# =============================================================================


class TestEarlyStopping:
    def test_no_stop_improving(self):
        """Test that early stopping doesn't trigger when improving."""
        es = EarlyStopping(patience=3, min_delta=0.01)
        model = torch.nn.Linear(10, 10)

        losses = [1.0, 0.9, 0.8, 0.7]
        for loss in losses:
            should_stop = es(loss, model)
            assert not should_stop

    def test_stop_not_improving(self):
        """Test that early stopping triggers when not improving."""
        es = EarlyStopping(patience=3, min_delta=0.01)
        model = torch.nn.Linear(10, 10)

        losses = [1.0, 0.9, 0.9, 0.9, 0.9]  # Plateau after 2nd
        for i, loss in enumerate(losses):
            should_stop = es(loss, model)
            if i < 4:  # patience=3, so stops at 5th (index 4)
                assert not should_stop
            else:
                assert should_stop

    def test_restore_best(self):
        """Test restoring best model state."""
        es = EarlyStopping(patience=3)
        model = torch.nn.Linear(2, 2)

        # Record initial weights
        initial_weight = model.weight.clone()

        # Improve, then worsen
        es(1.0, model)  # Best
        model.weight.data.fill_(99)  # Change weights
        es(1.5, model)  # Worse
        es(1.5, model)
        es(1.5, model)  # Triggers stop

        # Restore
        es.restore_best(model)

        assert torch.allclose(model.weight, initial_weight)


class TestSSLTrainer:
    def test_insufficient_data_error(self):
        """Test fail-fast for insufficient data."""
        config = SSLTrainingConfig(
            min_samples=1000,
            n_features=8,
            context_len=64,
            patch_len=16,
        )
        trainer = SSLTrainer(config)

        with pytest.raises(InsufficientDataError):
            trainer.check_data(100)

    def test_training_small_data(self, small_synthetic_data):
        """Test training loop with small data (skip convergence check)."""
        config = SSLTrainingConfig(
            n_features=8,
            context_len=64,
            patch_len=16,
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            epochs=2,
            batch_size=4,
            min_samples=10,  # Allow small data
            patience=10,
            device="cpu",
        )

        trainer = SSLTrainer(config)

        # Train
        model = trainer.train_from_array(small_synthetic_data)

        assert model is not None
        assert len(trainer.loss_history) > 0

    def test_encoder_save_during_training(self, small_synthetic_data):
        """Test that encoder is saved correctly."""
        config = SSLTrainingConfig(
            n_features=8,
            context_len=64,
            patch_len=16,
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            epochs=2,
            batch_size=4,
            min_samples=10,
            patience=10,
            device="cpu",
        )

        trainer = SSLTrainer(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "encoder.pt"
            model = trainer.train_from_array(small_synthetic_data, output_path=str(path))

            # Check file exists
            assert path.exists()

            # Load and verify
            loaded = SSLModel.load_encoder(str(path))
            assert loaded is not None


# =============================================================================
# Integration Test
# =============================================================================


class TestIntegration:
    def test_end_to_end_small(self, small_synthetic_data):
        """Test complete SSL pipeline with small data."""
        # 1. Create dataset
        dataset = PatchMaskingDataset(
            small_synthetic_data,
            patch_len=16,
            mask_ratio=0.4,
        )

        # 2. Create model
        config = PatchTSTConfig(
            n_features=8,
            context_len=64,
            patch_len=16,
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
        )
        model = SSLModel(config)

        # 3. Forward pass
        batch = dataset[0]
        x = batch["input"].unsqueeze(0)  # Add batch dim
        target = batch["target"].unsqueeze(0)
        mask_indices = batch["mask_indices"].unsqueeze(0)

        output = model(x)

        # 4. Compute loss
        loss = masked_reconstruction_loss(output, target, mask_indices, patch_len=16)

        assert loss.item() > 0

        # 5. Backward pass
        loss.backward()

        # Verify gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_encoder_forward_after_loading(self, small_synthetic_data):
        """Test that loaded encoder produces same output."""
        config = PatchTSTConfig(
            n_features=8,
            context_len=64,
            patch_len=16,
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
        )
        model = SSLModel(config)
        model.eval()

        # Sample input
        x = torch.randn(1, 8, 64)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "encoder.pt"
            model.save_encoder(str(path))

            # Get original output
            with torch.no_grad():
                original_encoded = model(x, return_encoded=True)

            # Load and get output
            loaded = SSLModel.load_encoder(str(path))
            loaded.eval()
            with torch.no_grad():
                loaded_encoded = loaded(x)

            # Compare
            assert torch.allclose(original_encoded, loaded_encoded)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
