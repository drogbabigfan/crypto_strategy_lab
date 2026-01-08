"""
SSL Training (v4.6)

One-off SSL Pre-training for Foundation Encoder.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .dataset import SSLDataConfig, PatchMaskingDataset, load_ssl_data
from .model import PatchTSTConfig, SSLModel, masked_reconstruction_loss

logger = logging.getLogger(__name__)


class InsufficientDataError(Exception):
    """Raised when there's not enough data for training."""

    pass


class DivergenceError(Exception):
    """Raised when training fails to converge."""

    pass


@dataclass
class SSLTrainingConfig:
    """SSL training configuration."""

    # Architecture
    n_features: int = 29
    context_len: int = 512
    patch_len: int = 16
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1

    # Masking
    mask_ratio: float = 0.4

    # Training
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Early stopping
    patience: int = 5
    min_delta: float = 1e-4

    # Fail-fast
    min_samples: int = 30000
    divergence_check_epochs: int = 5
    min_loss_reduction: float = 0.05

    # Validation
    val_ratio: float = 0.1

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_model_config(self) -> PatchTSTConfig:
        """Convert to model config."""
        return PatchTSTConfig(
            n_features=self.n_features,
            context_len=self.context_len,
            patch_len=self.patch_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
        )


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_state = None
        self.should_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.

        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            return True

        return False

    def restore_best(self, model: nn.Module):
        """Restore best model state."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class SSLTrainer:
    """SSL Pre-training Trainer."""

    def __init__(self, config: SSLTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.loss_history = []

    def check_data(self, n_samples: int):
        """Fail-fast check for data sufficiency."""
        if n_samples < self.config.min_samples:
            raise InsufficientDataError(
                f"Need {self.config.min_samples} samples, got {n_samples}"
            )

    def check_convergence(self):
        """Fail-fast check for training convergence."""
        if len(self.loss_history) < self.config.divergence_check_epochs:
            return

        initial = self.loss_history[0]
        current = self.loss_history[-1]
        reduction = (initial - current) / (initial + 1e-10)

        if reduction < self.config.min_loss_reduction:
            raise DivergenceError(
                f"Loss reduction {reduction:.2%} < {self.config.min_loss_reduction:.0%}"
            )

    def train(
        self,
        features_dir: str,
        data_config: Optional[SSLDataConfig] = None,
        output_path: Optional[str] = None,
    ) -> SSLModel:
        """
        Run SSL pre-training.

        Args:
            features_dir: Directory containing feature parquet files
            data_config: Data loading configuration
            output_path: Path to save trained encoder

        Returns:
            Trained SSLModel
        """
        # Load data
        logger.info("Loading SSL data...")
        data_config = data_config or SSLDataConfig()
        data = load_ssl_data(features_dir, data_config)

        # Update n_features from actual data
        actual_n_features = data.shape[2]
        if actual_n_features != self.config.n_features:
            logger.warning(
                f"Updating n_features: {self.config.n_features} → {actual_n_features}"
            )
            self.config.n_features = actual_n_features

        # Fail-fast check
        self.check_data(len(data))

        # Create dataset
        dataset = PatchMaskingDataset(
            data,
            patch_len=self.config.patch_len,
            mask_ratio=self.config.mask_ratio,
        )

        # Train/Val split
        val_size = int(len(dataset) * self.config.val_ratio)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )

        logger.info(f"Train samples: {train_size}, Val samples: {val_size}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if self.config.device == "cuda" else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.config.device == "cuda" else False,
        )

        # Create model
        model_config = self.config.to_model_config()
        model = SSLModel(model_config).to(self.device)

        logger.info(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Early stopping
        early_stopping = EarlyStopping(
            self.config.patience,
            self.config.min_delta,
        )

        # Training loop
        logger.info("Starting SSL training...")
        start_time = time.time()

        for epoch in range(self.config.epochs):
            # Train
            train_loss = self._train_epoch(model, train_loader, optimizer)
            self.loss_history.append(train_loss)

            # Validate
            val_loss = self._validate(model, val_loader)

            # Log
            elapsed = time.time() - start_time
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Fail-fast convergence check
            self.check_convergence()

            # Early stopping
            if early_stopping(val_loss, model):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        early_stopping.restore_best(model)

        # Save encoder
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_encoder(str(output_path))
            logger.info(f"Encoder saved to {output_path}")

        total_time = time.time() - start_time
        logger.info(f"SSL training completed in {total_time:.1f}s")

        return model

    def _train_epoch(
        self,
        model: SSLModel,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Run one training epoch."""
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)
            mask_indices = batch["mask_indices"].to(self.device)

            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)

            # Loss (only on masked patches)
            loss = masked_reconstruction_loss(
                outputs,
                targets,
                mask_indices,
                self.config.patch_len,
            )

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _validate(self, model: SSLModel, loader: DataLoader) -> float:
        """Run validation."""
        model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)
                mask_indices = batch["mask_indices"].to(self.device)

                outputs = model(inputs)

                loss = masked_reconstruction_loss(
                    outputs,
                    targets,
                    mask_indices,
                    self.config.patch_len,
                )

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def train_from_array(
        self,
        data: np.ndarray,  # (N, context_len, n_features)
        output_path: Optional[str] = None,
    ) -> SSLModel:
        """
        Train from pre-loaded numpy array.

        Useful for testing or when data is already in memory.

        Args:
            data: Feature windows (N, context_len, n_features)
            output_path: Path to save trained encoder

        Returns:
            Trained SSLModel
        """
        # Update n_features from actual data
        actual_n_features = data.shape[2]
        if actual_n_features != self.config.n_features:
            logger.warning(
                f"Updating n_features: {self.config.n_features} → {actual_n_features}"
            )
            self.config.n_features = actual_n_features

        # Fail-fast check
        self.check_data(len(data))

        # Create dataset
        dataset = PatchMaskingDataset(
            data,
            patch_len=self.config.patch_len,
            mask_ratio=self.config.mask_ratio,
        )

        # Train/Val split
        val_size = int(len(dataset) * self.config.val_ratio)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )

        logger.info(f"Train samples: {train_size}, Val samples: {val_size}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Create model
        model_config = self.config.to_model_config()
        model = SSLModel(model_config).to(self.device)

        logger.info(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Early stopping
        early_stopping = EarlyStopping(
            self.config.patience,
            self.config.min_delta,
        )

        # Training loop
        logger.info("Starting SSL training...")
        start_time = time.time()

        for epoch in range(self.config.epochs):
            # Train
            train_loss = self._train_epoch(model, train_loader, optimizer)
            self.loss_history.append(train_loss)

            # Validate
            val_loss = self._validate(model, val_loader)

            # Log
            elapsed = time.time() - start_time
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Fail-fast convergence check (skip for small datasets in tests)
            if len(data) >= self.config.min_samples:
                self.check_convergence()

            # Early stopping
            if early_stopping(val_loss, model):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        early_stopping.restore_best(model)

        # Save encoder
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_encoder(str(output_path))
            logger.info(f"Encoder saved to {output_path}")

        total_time = time.time() - start_time
        logger.info(f"SSL training completed in {total_time:.1f}s")

        return model
