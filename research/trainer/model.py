"""
Fine-tuning Model (v4.6)

PatchTSTClassifier: Pre-trained Encoder + Classification Head
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from research.ssl.model import PatchTSTEncoder, PatchTSTConfig, SSLModel


@dataclass
class ClassifierConfig:
    """Classifier head configuration."""

    n_classes: int = 3  # Short, Neutral, Long
    bottleneck_dim: int = 256
    dropout: float = 0.3


class ResidualBottleneckHead(nn.Module):
    """
    Residual Bottleneck Classification Head.

    Structure:
        Linear(input_dim → bottleneck_dim) → GELU → Dropout
        Linear(bottleneck_dim → input_dim)
        Residual Connection
        Classification: Linear(input_dim → n_classes)
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int = 3,
        bottleneck_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, input_dim),
        )

        # Layer norm for residual
        self.norm = nn.LayerNorm(input_dim)

        # Classification head
        self.classifier = nn.Linear(input_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, input_dim) flattened encoder output

        Returns:
            (Batch, n_classes) logits
        """
        # Residual bottleneck
        residual = x
        x = self.bottleneck(x)
        x = self.norm(x + residual)

        # Classification
        logits = self.classifier(x)

        return logits


class PatchTSTClassifier(nn.Module):
    """
    PatchTST Classifier for Fine-tuning.

    Structure:
        [Pre-trained PatchTST Encoder] ← foundation_encoder.pt
              ↓
        Flatten (n_features × n_patches × d_model)
              ↓
        Residual Bottleneck Head
              ↓
        Classification: Linear(→ n_classes)
              ↓
        Output: Logits [Short, Neutral, Long]
    """

    def __init__(
        self,
        encoder: PatchTSTEncoder,
        n_classes: int = 3,
        bottleneck_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = encoder
        self.config = encoder.config

        # Calculate flattened dimension
        # n_features × n_patches × d_model
        self.flatten_dim = (
            self.config.n_features *
            self.config.n_patches *
            self.config.d_model
        )

        # Classification head
        self.classifier = ResidualBottleneckHead(
            input_dim=self.flatten_dim,
            n_classes=n_classes,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, n_features, context_len)

        Returns:
            (Batch, n_classes) logits
        """
        # Encode: (B, n_features, context_len) → (B, n_features, n_patches, d_model)
        encoded = self.encoder(x)

        # Flatten: (B, n_features, n_patches, d_model) → (B, flatten_dim)
        batch_size = encoded.shape[0]
        flattened = encoded.view(batch_size, -1)

        # Classify
        logits = self.classifier(flattened)

        return logits

    def get_encoder_params(self):
        """Get encoder parameters for separate optimizer group."""
        return self.encoder.parameters()

    def get_classifier_params(self):
        """Get classifier head parameters for separate optimizer group."""
        return self.classifier.parameters()

    def freeze_encoder(self):
        """Freeze encoder weights for Phase 1 training."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder weights for Phase 2 training."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def save(self, path: str):
        """Save the full classifier model."""
        torch.save(
            {
                "config": self.config,
                "classifier_config": {
                    "n_classes": self.classifier.n_classes,
                    "input_dim": self.classifier.input_dim,
                },
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "PatchTSTClassifier":
        """Load a saved classifier model."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Reconstruct encoder
        encoder_config = checkpoint["config"]
        encoder = PatchTSTEncoder(encoder_config)

        # Reconstruct classifier
        classifier_config = checkpoint["classifier_config"]
        model = cls(
            encoder=encoder,
            n_classes=classifier_config["n_classes"],
        )

        # Load weights
        model.load_state_dict(checkpoint["state_dict"])

        return model

    @classmethod
    def from_pretrained(
        cls,
        encoder_path: str,
        n_classes: int = 3,
        bottleneck_dim: int = 256,
        dropout: float = 0.3,
    ) -> "PatchTSTClassifier":
        """
        Create classifier from pre-trained encoder.

        Args:
            encoder_path: Path to pre-trained encoder (from SSL)
            n_classes: Number of output classes
            bottleneck_dim: Bottleneck hidden dimension
            dropout: Dropout rate

        Returns:
            PatchTSTClassifier with pre-trained encoder weights
        """
        encoder = SSLModel.load_encoder(encoder_path)

        return cls(
            encoder=encoder,
            n_classes=n_classes,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
        )
