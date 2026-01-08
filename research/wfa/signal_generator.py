"""
Signal Generator - Generate trading signals from trained classifier.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class SignalGeneratorConfig:
    """Configuration for signal generation."""

    context_len: int = 512
    batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Signal thresholds (optional confidence-based filtering)
    min_confidence: float = 0.0  # Minimum softmax probability to emit signal
    neutral_band: float = 0.0    # If max prob < neutral_band, emit 0


class SignalGenerator:
    """
    Generate trading signals from a trained classifier.

    The classifier outputs logits for 3 classes:
    - Class 0: Short (-1)
    - Class 1: Neutral (0)
    - Class 2: Long (+1)

    Usage:
        generator = SignalGenerator(model, context_len=512)
        signals = generator.generate(features)  # (N,) int8 array
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[SignalGeneratorConfig] = None,
    ):
        """
        Initialize signal generator.

        Args:
            model: Trained classifier model
            config: Generation configuration
        """
        self.model = model
        self.config = config or SignalGeneratorConfig()
        self.model.to(self.config.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, features: np.ndarray) -> np.ndarray:
        """
        Generate signals for all bars.

        Args:
            features: (N, n_features) array of normalized features

        Returns:
            signals: (N,) int8 array with values in {-1, 0, 1}
        """
        n_samples = len(features)
        context_len = self.config.context_len

        # Initialize all signals as neutral
        signals = np.zeros(n_samples, dtype=np.int8)

        # Need at least context_len samples to generate signals
        if n_samples < context_len:
            return signals

        # Generate signals for each valid position
        valid_indices = list(range(context_len, n_samples))

        # Process in batches for efficiency
        for batch_start in range(0, len(valid_indices), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(valid_indices))
            batch_indices = valid_indices[batch_start:batch_end]

            # Build batch of context windows
            batch_windows = []
            for idx in batch_indices:
                window = features[idx - context_len:idx]  # (context_len, n_features)
                batch_windows.append(window)

            # Stack and convert to tensor
            batch = np.stack(batch_windows)  # (B, context_len, n_features)
            batch = np.transpose(batch, (0, 2, 1))  # (B, n_features, context_len) - channel first

            x = torch.tensor(batch, dtype=torch.float32, device=self.config.device)

            # Forward pass
            logits = self.model(x)  # (B, 3)
            probs = torch.softmax(logits, dim=-1)  # (B, 3)

            # Get predictions
            max_probs, preds = torch.max(probs, dim=-1)  # (B,), (B,)

            # Convert to numpy
            preds = preds.cpu().numpy()
            max_probs = max_probs.cpu().numpy()

            # Apply confidence filtering if configured
            for i, idx in enumerate(batch_indices):
                pred = preds[i]
                prob = max_probs[i]

                # Skip low confidence predictions
                if prob < self.config.min_confidence:
                    continue

                # Apply neutral band
                if prob < self.config.neutral_band:
                    continue

                # Convert {0, 1, 2} -> {-1, 0, +1}
                signals[idx] = pred - 1

        return signals

    @torch.no_grad()
    def generate_with_probs(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate signals with probability scores.

        Args:
            features: (N, n_features) array

        Returns:
            signals: (N,) int8 array
            probs: (N, 3) float32 array of class probabilities
        """
        n_samples = len(features)
        context_len = self.config.context_len

        signals = np.zeros(n_samples, dtype=np.int8)
        probs = np.zeros((n_samples, 3), dtype=np.float32)

        if n_samples < context_len:
            probs[:, 1] = 1.0  # All neutral
            return signals, probs

        valid_indices = list(range(context_len, n_samples))

        for batch_start in range(0, len(valid_indices), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(valid_indices))
            batch_indices = valid_indices[batch_start:batch_end]

            batch_windows = []
            for idx in batch_indices:
                window = features[idx - context_len:idx]
                batch_windows.append(window)

            batch = np.stack(batch_windows)
            batch = np.transpose(batch, (0, 2, 1))

            x = torch.tensor(batch, dtype=torch.float32, device=self.config.device)

            logits = self.model(x)
            batch_probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for i, idx in enumerate(batch_indices):
                probs[idx] = batch_probs[i]
                signals[idx] = np.argmax(batch_probs[i]) - 1

        # Set neutral probability for warmup period
        probs[:context_len, 1] = 1.0

        return signals, probs

    def generate_streaming(self, features: np.ndarray) -> np.ndarray:
        """
        Generate signals one at a time (for live trading simulation).

        Less efficient but more realistic for production.

        Args:
            features: (N, n_features) array

        Returns:
            signals: (N,) int8 array
        """
        n_samples = len(features)
        context_len = self.config.context_len
        signals = np.zeros(n_samples, dtype=np.int8)

        if n_samples < context_len:
            return signals

        for i in range(context_len, n_samples):
            window = features[i - context_len:i]  # (context_len, n_features)
            signal = self._generate_single(window)
            signals[i] = signal

        return signals

    @torch.no_grad()
    def _generate_single(self, window: np.ndarray) -> int:
        """Generate signal for a single context window."""
        # (context_len, n_features) -> (1, n_features, context_len)
        x = window.T[np.newaxis, ...]
        x = torch.tensor(x, dtype=torch.float32, device=self.config.device)

        logits = self.model(x)
        pred = torch.argmax(logits, dim=-1).item()

        return pred - 1  # {0,1,2} -> {-1,0,+1}


def count_signals(signals: np.ndarray) -> dict:
    """
    Count signal distribution.

    Args:
        signals: (N,) int8 array

    Returns:
        dict with counts for each signal type
    """
    unique, counts = np.unique(signals, return_counts=True)
    result = {"short": 0, "neutral": 0, "long": 0, "total": len(signals)}

    for val, count in zip(unique, counts):
        if val == -1:
            result["short"] = int(count)
        elif val == 0:
            result["neutral"] = int(count)
        elif val == 1:
            result["long"] = int(count)

    return result


def compute_signal_stats(signals: np.ndarray) -> dict:
    """
    Compute signal statistics.

    Args:
        signals: (N,) int8 array

    Returns:
        dict with signal statistics
    """
    counts = count_signals(signals)
    total = counts["total"]

    if total == 0:
        return {"counts": counts, "ratios": {}}

    ratios = {
        "short_ratio": counts["short"] / total,
        "neutral_ratio": counts["neutral"] / total,
        "long_ratio": counts["long"] / total,
        "active_ratio": (counts["short"] + counts["long"]) / total,
    }

    # Compute signal transitions
    transitions = {"long_to_short": 0, "short_to_long": 0, "to_neutral": 0, "from_neutral": 0}

    for i in range(1, len(signals)):
        prev, curr = signals[i - 1], signals[i]
        if prev == 1 and curr == -1:
            transitions["long_to_short"] += 1
        elif prev == -1 and curr == 1:
            transitions["short_to_long"] += 1
        elif prev != 0 and curr == 0:
            transitions["to_neutral"] += 1
        elif prev == 0 and curr != 0:
            transitions["from_neutral"] += 1

    return {
        "counts": counts,
        "ratios": ratios,
        "transitions": transitions,
    }
