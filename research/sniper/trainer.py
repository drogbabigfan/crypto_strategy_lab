"""
Sniper Trainer (v4.9)

Training loop for SniperClassifier.

Key features:
- Balanced sampling for training
- Unbalanced validation for true metrics
- Focal loss for hard example mining
- Early stopping on validation F1
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from .model import SniperClassifier
from .dataset import (
    SniperDataset,
    create_balanced_dataloader,
    create_validation_dataloader,
)

logger = logging.getLogger(__name__)


@dataclass
class SniperTrainingConfig:
    """Training configuration."""

    # Training
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Loss
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05

    # Early stopping
    patience: int = 10
    min_delta: float = 0.001

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Note: We use balanced sampling, so this is mainly for
    hard example mining (gamma > 0).
    """

    def __init__(self, gamma: float = 2.0, smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)

        # Label smoothing
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.smoothing / n_classes)
            smooth_targets.scatter_(
                1, targets.unsqueeze(1),
                1 - self.smoothing + self.smoothing / n_classes
            )

        # Probabilities
        probs = F.softmax(logits, dim=-1)

        # Focal weight
        pt = (probs * smooth_targets).sum(dim=-1)
        focal_weight = (1 - pt) ** self.gamma

        # Cross entropy
        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss = -(smooth_targets * log_probs).sum(dim=-1)

        # Final loss
        loss = focal_weight * ce_loss

        return loss.mean()


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.should_stop = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        """Check if should stop. Higher score is better."""
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


class SniperTrainer:
    """
    Trainer for SniperClassifier.

    Training uses balanced sampling (1:1:1).
    Validation uses original distribution for true metrics.
    """

    def __init__(self, config: Optional[SniperTrainingConfig] = None):
        self.config = config or SniperTrainingConfig()
        self.device = torch.device(self.config.device)
        self.history = []

    def train(
        self,
        model: SniperClassifier,
        train_dataset: SniperDataset,
        val_dataset: Optional[SniperDataset] = None,
        output_path: Optional[str] = None,
    ) -> SniperClassifier:
        """
        Train the model.

        Args:
            model: SniperClassifier instance
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_path: Path to save best model

        Returns:
            Trained model
        """
        model = model.to(self.device)
        c = self.config

        # Create data loaders
        train_loader = create_balanced_dataloader(
            train_dataset,
            batch_size=c.batch_size,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = create_validation_dataloader(
                val_dataset,
                batch_size=c.batch_size,
            )

        # Loss and optimizer
        loss_fn = FocalLoss(gamma=c.focal_gamma, smoothing=c.label_smoothing)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=c.learning_rate,
            weight_decay=c.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=c.epochs,
            eta_min=c.learning_rate / 10,
        )

        # Early stopping
        early_stopping = EarlyStopping(c.patience, c.min_delta)

        # Training loop
        logger.info("Starting training...")
        start_time = time.time()

        for epoch in range(c.epochs):
            # Train epoch
            train_loss, train_acc = self._train_epoch(
                model, train_loader, optimizer, loss_fn
            )

            # Validate
            val_metrics = {"loss": 0, "acc": 0, "f1": 0}
            if val_loader is not None:
                val_metrics = self._validate(model, val_loader, loss_fn)

            # Update scheduler
            scheduler.step()

            # Log progress
            elapsed = time.time() - start_time
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch+1}/{c.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                f"Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['acc']:.2%} | "
                f"LR: {lr:.2e} | Time: {elapsed:.1f}s"
            )

            # Save history
            self.history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_f1": val_metrics["f1"],
                "lr": lr,
            })

            # Early stopping check
            if val_loader is not None:
                if early_stopping(val_metrics["f1"], model):
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        early_stopping.restore_best(model)

        # Save model
        if output_path:
            model.save(output_path)
            logger.info(f"Model saved to {output_path}")

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s")
        logger.info(f"Best Val F1: {early_stopping.best_score:.4f}")

        return model

    def _train_epoch(
        self,
        model: SniperClassifier,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> tuple:
        """Run one training epoch."""
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()

            logits = model(x)
            loss = loss_fn(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track metrics
            preds = torch.argmax(logits, dim=-1)
            total_loss += loss.item() * x.size(0)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def _validate(
        self,
        model: SniperClassifier,
        loader: DataLoader,
        loss_fn: nn.Module,
    ) -> Dict[str, float]:
        """Run validation."""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = model(x)
                loss = loss_fn(logits, y)

                preds = torch.argmax(logits, dim=-1)

                total_loss += loss.item() * x.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Compute metrics
        avg_loss = total_loss / len(all_targets)
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average="macro")

        # Log per-class metrics
        cm = confusion_matrix(all_targets, all_preds)
        logger.debug(f"Confusion Matrix:\n{cm}")

        # Check for collapse (all predictions same class)
        unique_preds = np.unique(all_preds)
        if len(unique_preds) == 1:
            logger.warning(f"Model collapse! All predictions: {unique_preds[0]}")

        return {
            "loss": avg_loss,
            "acc": accuracy,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
        }

    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        if not self.history:
            return {}

        return {
            "epochs": len(self.history),
            "best_val_f1": max(h["val_f1"] for h in self.history),
            "final_train_loss": self.history[-1]["train_loss"],
            "final_val_f1": self.history[-1]["val_f1"],
        }
