"""
Fine-tuning Training (v4.6)

2-Phase Fine-tuning:
1. Encoder Frozen: Classification Head만 학습
2. Full Unfrozen: 전체 미세조정
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from .model import PatchTSTClassifier
from .loss import FocalLoss

logger = logging.getLogger(__name__)


class InsufficientDataError(Exception):
    """Raised when there's not enough data for training."""
    pass


class DivergenceError(Exception):
    """Raised when training fails to converge."""
    pass


@dataclass
class FineTuningTrainingConfig:
    """Fine-tuning training configuration."""

    # 2-Phase training
    frozen_epochs: int = 5        # Phase 1: encoder frozen
    total_epochs: int = 30        # Total epochs (Phase 1 + 2)

    # Batch size
    batch_size: int = 64

    # Learning rates
    encoder_lr: float = 1e-5      # Phase 2 encoder LR
    head_lr: float = 1e-4         # Classification head LR
    weight_decay: float = 1e-5

    # Focal Loss
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    use_class_weights: bool = True

    # Early stopping
    patience: int = 5
    min_delta: float = 1e-4

    # Fail-fast
    min_samples: int = 1000
    divergence_check_epochs: int = 5
    min_loss_reduction: float = 0.05

    # Validation
    val_ratio: float = 0.1

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class EarlyStopping:
    """Early stopping handler for fine-tuning."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.should_stop = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.

        Args:
            score: Validation metric (higher is better, e.g., F1)

        Returns:
            True if training should stop
        """
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
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


class FineTuningTrainer:
    """
    2-Phase Fine-tuning Trainer.

    Phase 1: Encoder Frozen (frozen_epochs)
        - Only classification head learns
        - Preserves pre-trained representations

    Phase 2: Full Unfrozen (remaining epochs)
        - Entire model fine-tuned
        - Lower LR for encoder than head
    """

    def __init__(self, config: FineTuningTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.loss_history = []
        self.val_metrics_history = []

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
        model: PatchTSTClassifier,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        class_weights: Optional[torch.Tensor] = None,
        output_path: Optional[str] = None,
    ) -> PatchTSTClassifier:
        """
        Run 2-Phase fine-tuning.

        Args:
            model: PatchTSTClassifier (with pre-trained encoder)
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional, uses train split if None)
            class_weights: Optional class weights for loss
            output_path: Path to save trained model

        Returns:
            Trained PatchTSTClassifier
        """
        # Move model to device
        model = model.to(self.device)

        # Fail-fast check
        self.check_data(len(train_dataset))

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.config.device == "cuda" else False,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if self.config.device == "cuda" else False,
            )

        # Loss function
        weights = None
        if self.config.use_class_weights and class_weights is not None:
            weights = class_weights.to(self.device)

        loss_fn = FocalLoss(
            gamma=self.config.focal_gamma,
            label_smoothing=self.config.label_smoothing,
            weight=weights,
        )

        # Early stopping
        early_stopping = EarlyStopping(
            self.config.patience,
            self.config.min_delta,
        )

        # Training loop
        logger.info("Starting 2-Phase Fine-tuning...")
        start_time = time.time()

        for epoch in range(self.config.total_epochs):
            # Phase control
            is_phase1 = epoch < self.config.frozen_epochs
            phase_str = "Phase 1 (Frozen)" if is_phase1 else "Phase 2 (Unfrozen)"

            # Set encoder frozen/unfrozen
            if is_phase1:
                model.freeze_encoder()
                optimizer = torch.optim.AdamW(
                    model.get_classifier_params(),
                    lr=self.config.head_lr,
                    weight_decay=self.config.weight_decay,
                )
            else:
                model.unfreeze_encoder()
                optimizer = torch.optim.AdamW([
                    {"params": model.get_encoder_params(), "lr": self.config.encoder_lr},
                    {"params": model.get_classifier_params(), "lr": self.config.head_lr},
                ], weight_decay=self.config.weight_decay)

            # Train epoch
            train_loss = self._train_epoch(model, train_loader, optimizer, loss_fn)
            self.loss_history.append(train_loss)

            # Validate
            val_metrics = {"loss": train_loss, "f1": 0.0, "accuracy": 0.0}
            if val_loader is not None:
                val_metrics = self._validate(model, val_loader, loss_fn)

            self.val_metrics_history.append(val_metrics)

            # Log
            elapsed = time.time() - start_time
            logger.info(
                f"Epoch {epoch + 1}/{self.config.total_epochs} [{phase_str}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Fail-fast convergence check (skip for small datasets)
            if len(train_dataset) >= self.config.min_samples:
                self.check_convergence()

            # Early stopping (based on validation F1)
            if val_loader is not None:
                if early_stopping(val_metrics['f1'], model):
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Restore best model
        early_stopping.restore_best(model)

        # Save model
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(output_path))
            logger.info(f"Model saved to {output_path}")

        total_time = time.time() - start_time
        logger.info(f"Fine-tuning completed in {total_time:.1f}s")

        return model

    def _train_epoch(
        self,
        model: PatchTSTClassifier,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> float:
        """Run one training epoch."""
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()

            logits = model(x)
            loss = loss_fn(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _validate(
        self,
        model: PatchTSTClassifier,
        loader: DataLoader,
        loss_fn: nn.Module,
    ) -> Dict[str, float]:
        """Run validation and compute metrics."""
        model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = model(x)
                loss = loss_fn(logits, y)

                preds = torch.argmax(logits, dim=-1)

                total_loss += loss.item()
                n_batches += 1

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        # Compute metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        return {
            "loss": total_loss / n_batches,
            "f1": f1_score(all_targets, all_preds, average='macro'),
            "accuracy": accuracy_score(all_targets, all_preds),
            "confusion_matrix": confusion_matrix(all_targets, all_preds).tolist(),
        }

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training run."""
        return {
            "total_epochs": len(self.loss_history),
            "final_train_loss": self.loss_history[-1] if self.loss_history else None,
            "final_val_metrics": self.val_metrics_history[-1] if self.val_metrics_history else None,
            "best_val_f1": max(m['f1'] for m in self.val_metrics_history) if self.val_metrics_history else None,
            "loss_history": self.loss_history,
        }


def finetune(
    encoder_path: str,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    config: Optional[FineTuningTrainingConfig] = None,
    output_path: Optional[str] = None,
) -> PatchTSTClassifier:
    """
    Convenience function for fine-tuning.

    Args:
        encoder_path: Path to pre-trained encoder
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
        output_path: Path to save model

    Returns:
        Trained PatchTSTClassifier
    """
    config = config or FineTuningTrainingConfig()

    # Create model from pre-trained encoder
    model = PatchTSTClassifier.from_pretrained(encoder_path)

    # Get class weights if available
    class_weights = None
    if hasattr(train_dataset, 'get_class_weights'):
        class_weights = train_dataset.get_class_weights()

    # Train
    trainer = FineTuningTrainer(config)
    model = trainer.train(
        model,
        train_dataset,
        val_dataset,
        class_weights=class_weights,
        output_path=output_path,
    )

    return model
