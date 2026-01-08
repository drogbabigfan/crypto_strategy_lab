"""
Sniper Classifier Model (v4.9)

PatchTST Encoder + Channel Mixing Head for 3-class classification.

Key difference from vanilla PatchTST:
- Vanilla: Each channel predicts independently (Channel Independence)
- Ours: Channels are mixed before classification (Channel Mixing)

This allows learning cross-feature patterns like:
- "High volume + price spike = momentum"
- "Low volume + price change = fake breakout"
"""

from dataclasses import dataclass
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SniperModelConfig:
    """Model configuration."""

    # Input
    n_features: int = 29
    context_len: int = 128

    # Patching
    patch_len: int = 16
    stride: int = 8

    # Transformer (v5.0: lighter model, higher dropout)
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.25

    # Output
    n_classes: int = 3


class PositionalEncoding(nn.Module):
    """Learnable positional encoding."""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """Patch embedding layer."""

    def __init__(self, patch_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_patches, patch_len)

        Returns:
            (B, n_patches, d_model)
        """
        x = self.proj(x)
        return self.dropout(x)


class ChannelMixingHead(nn.Module):
    """
    Channel Mixing Head for classification.

    Takes encoder outputs from all channels and mixes them
    before producing class probabilities.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        n_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_features, n_patches, d_model)

        Returns:
            (B, n_classes)
        """
        return self.head(x)


class SniperClassifier(nn.Module):
    """
    Sniper Classifier: PatchTST + Channel Mixing Head.

    Architecture:
        1. Patching: Split each channel into patches
        2. Embedding: Project patches to d_model
        3. Transformer: Process each channel independently
        4. Channel Mixing: Combine all channels for classification
    """

    def __init__(self, config: Optional[SniperModelConfig] = None):
        super().__init__()

        self.config = config or SniperModelConfig()
        c = self.config

        # Calculate number of patches
        self.n_patches = (c.context_len - c.patch_len) // c.stride + 1

        # Patch embedding (per channel)
        self.patch_embedding = PatchEmbedding(c.patch_len, c.d_model, c.dropout)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(c.d_model, self.n_patches, c.dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.n_heads,
            dim_feedforward=c.d_ff,
            dropout=c.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=c.n_layers,
            enable_nested_tensor=False,
        )

        # Channel Mixing Head
        flat_dim = c.n_features * self.n_patches * c.d_model
        self.classifier = ChannelMixingHead(
            input_dim=flat_dim,
            hidden_dim=512,
            n_classes=c.n_classes,
            dropout=c.dropout,
        )

        # Initialize weights
        self._init_weights()

        # Log model size
        n_params = sum(p.numel() for p in self.parameters())
        print(f"SniperClassifier: {n_params:,} parameters")

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, n_features, context_len) - channel first

        Returns:
            logits: (B, n_classes)
        """
        B, C, L = x.shape
        c = self.config

        # Create patches for each channel
        # (B, C, L) -> (B, C, n_patches, patch_len)
        patches = x.unfold(dimension=2, size=c.patch_len, step=c.stride)

        # Process each channel through patch embedding
        # (B, C, n_patches, patch_len) -> (B, C, n_patches, d_model)
        B, C, N, P = patches.shape
        patches = patches.reshape(B * C, N, P)
        embedded = self.patch_embedding(patches)  # (B*C, N, d_model)

        # Add positional encoding
        embedded = self.pos_encoding(embedded)

        # Transformer encoder
        encoded = self.encoder(embedded)  # (B*C, N, d_model)

        # Reshape back to separate channels
        # (B*C, N, d_model) -> (B, C, N, d_model)
        encoded = encoded.view(B, C, N, -1)

        # Channel Mixing Head
        logits = self.classifier(encoded)  # (B, n_classes)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)

    def save(self, path: str):
        """Save model."""
        torch.save({
            "config": self.config,
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SniperClassifier":
        """Load model."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device)


def create_model(
    n_features: int = 29,
    context_len: int = 128,
    d_model: int = 128,
    n_layers: int = 3,
    dropout: float = 0.2,
) -> SniperClassifier:
    """Create SniperClassifier with custom config."""
    config = SniperModelConfig(
        n_features=n_features,
        context_len=context_len,
        d_model=d_model,
        n_layers=n_layers,
        dropout=dropout,
    )
    return SniperClassifier(config)
