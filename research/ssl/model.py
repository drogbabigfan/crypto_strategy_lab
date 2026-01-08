"""
SSL Model Architecture (v4.6)

PatchTST Encoder for Self-Supervised Pre-training.

Architecture:
    Input: (Batch, n_features, context_len)
        ↓
    Patching: (Batch, n_features, n_patches, patch_len)
        ↓
    Patch Embedding: Linear(patch_len → d_model)
        ↓
    Positional Embedding: Learnable (n_patches positions)
        ↓
    Transformer Encoder: n_heads, n_layers, d_model
        ↓
    Output: Encoded patches (Batch, n_features, n_patches, d_model)
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PatchTSTConfig:
    """PatchTST Encoder configuration."""

    n_features: int = 29  # Number of input features (channels)
    context_len: int = 512
    patch_len: int = 16
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256  # Feed-forward hidden dimension
    dropout: float = 0.1

    @property
    def n_patches(self) -> int:
        return self.context_len // self.patch_len


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer.

    Converts (Batch, n_features, context_len) →
            (Batch, n_features, n_patches, d_model)
    """

    def __init__(self, patch_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model

        # Linear projection for each patch
        self.proj = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, n_features, context_len)

        Returns:
            (Batch, n_features, n_patches, d_model)
        """
        batch_size, n_features, context_len = x.shape
        n_patches = context_len // self.patch_len

        # Reshape to patches: (B, C, L) → (B, C, n_patches, patch_len)
        x = x.view(batch_size, n_features, n_patches, self.patch_len)

        # Project: (B, C, n_patches, patch_len) → (B, C, n_patches, d_model)
        x = self.proj(x)
        x = self.dropout(x)

        return x


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for patches."""

    def __init__(self, n_patches: int, d_model: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, n_features, n_patches, d_model)

        Returns:
            (Batch, n_features, n_patches, d_model) with position added
        """
        return x + self.pos_embed


class TransformerEncoderLayer(nn.Module):
    """Standard Transformer Encoder Layer with pre-norm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, seq_len, d_model)

        Returns:
            (Batch, seq_len, d_model)
        """
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # Pre-norm feed-forward
        normed = self.norm2(x)
        ff_out = self.ff(normed)
        x = x + ff_out

        return x


class PatchTSTEncoder(nn.Module):
    """
    PatchTST Encoder for time series.

    Processes each feature channel independently through shared Transformer.
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            config.patch_len,
            config.d_model,
            config.dropout,
        )

        # Positional encoding
        self.pos_embed = PositionalEncoding(
            config.n_patches,
            config.d_model,
        )

        # Transformer encoder layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, n_features, context_len)

        Returns:
            (Batch, n_features, n_patches, d_model)
        """
        batch_size = x.shape[0]
        n_features = x.shape[1]

        # Patch embedding: (B, C, L) → (B, C, P, D)
        x = self.patch_embed(x)

        # Add positional encoding
        x = self.pos_embed(x)

        # Reshape for Transformer: (B, C, P, D) → (B*C, P, D)
        x = x.view(batch_size * n_features, self.config.n_patches, self.config.d_model)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final norm
        x = self.norm(x)

        # Reshape back: (B*C, P, D) → (B, C, P, D)
        x = x.view(batch_size, n_features, self.config.n_patches, self.config.d_model)

        return x

    def get_output_dim(self) -> int:
        """Get total output dimension for classification head."""
        return self.config.n_features * self.config.n_patches * self.config.d_model


class ReconstructionHead(nn.Module):
    """
    Reconstruction head for SSL pre-training.

    Projects encoded patches back to original patch dimension.
    """

    def __init__(self, d_model: int, patch_len: int):
        super().__init__()
        self.proj = nn.Linear(d_model, patch_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, n_features, n_patches, d_model)

        Returns:
            (Batch, n_features, n_patches, patch_len)
        """
        return self.proj(x)


class SSLModel(nn.Module):
    """
    Complete SSL Model = Encoder + Reconstruction Head.

    For pre-training: uses reconstruction head
    For fine-tuning: uses only encoder (head discarded)
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.config = config
        self.encoder = PatchTSTEncoder(config)
        self.reconstruction_head = ReconstructionHead(config.d_model, config.patch_len)

    def forward(
        self,
        x: torch.Tensor,
        return_encoded: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (Batch, n_features, context_len)
            return_encoded: If True, return encoder output without reconstruction

        Returns:
            If return_encoded=False:
                (Batch, n_features, context_len) - reconstructed sequence
            If return_encoded=True:
                (Batch, n_features, n_patches, d_model) - encoded patches
        """
        # Encode
        encoded = self.encoder(x)  # (B, C, P, D)

        if return_encoded:
            return encoded

        # Reconstruct
        reconstructed = self.reconstruction_head(encoded)  # (B, C, P, patch_len)

        # Reshape to original: (B, C, P, patch_len) → (B, C, context_len)
        batch_size = x.shape[0]
        n_features = x.shape[1]
        reconstructed = reconstructed.view(batch_size, n_features, -1)

        return reconstructed

    def save_encoder(self, path: str):
        """Save only the encoder weights (for fine-tuning)."""
        torch.save(
            {
                "config": self.config,
                "encoder_state_dict": self.encoder.state_dict(),
            },
            path,
        )

    @classmethod
    def load_encoder(cls, path: str) -> "PatchTSTEncoder":
        """Load pre-trained encoder."""
        # weights_only=False needed for loading dataclass config
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        encoder = PatchTSTEncoder(config)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        return encoder


def masked_reconstruction_loss(
    pred: torch.Tensor,  # (Batch, n_features, context_len)
    target: torch.Tensor,  # (Batch, n_features, context_len)
    mask_indices: torch.Tensor,  # (Batch, n_masked)
    patch_len: int,
) -> torch.Tensor:
    """
    Compute MSE loss only on masked patches.

    Args:
        pred: Reconstructed sequence
        target: Original sequence
        mask_indices: Indices of masked patches per sample
        patch_len: Length of each patch

    Returns:
        Scalar loss tensor
    """
    batch_size = pred.shape[0]
    total_loss = 0.0

    for i in range(batch_size):
        for patch_idx in mask_indices[i]:
            start = patch_idx * patch_len
            end = start + patch_len
            loss = F.mse_loss(pred[i, :, start:end], target[i, :, start:end])
            total_loss += loss

    # Average over batch and masked patches
    n_masked = mask_indices.shape[1]
    return total_loss / (batch_size * n_masked)
