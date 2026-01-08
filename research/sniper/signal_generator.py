"""
Confidence Signal Generator (v4.9)

Generate trading signals only when model confidence exceeds threshold.

Key concept:
- argmax alone would trade on 51% vs 49%
- With threshold=0.6, we only trade when confident
- This aligns with "Sniper" philosophy
"""

from typing import Tuple, Optional
import logging

import numpy as np
import torch
import torch.nn.functional as F

from .model import SniperClassifier

logger = logging.getLogger(__name__)


class ConfidenceSignalGenerator:
    """
    Generate signals with confidence threshold.

    Only generates non-neutral signals when:
    1. Model predicts Long or Short (not Neutral)
    2. Prediction confidence >= threshold

    Signals:
        +1: Long (confident bullish)
        -1: Short (confident bearish)
         0: Neutral (not confident OR model predicts neutral)
    """

    def __init__(
        self,
        model: SniperClassifier,
        context_len: int = 128,
        confidence_threshold: float = 0.6,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 64,
    ):
        """
        Args:
            model: Trained SniperClassifier
            context_len: Context window length
            confidence_threshold: Minimum confidence to emit signal
            device: Device for inference
            batch_size: Batch size for inference
        """
        self.model = model.to(device).eval()
        self.context_len = context_len
        self.threshold = confidence_threshold
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def generate(
        self,
        features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate signals for all bars.

        Args:
            features: (N, n_features) array of features

        Returns:
            signals: (N,) int8 array with {-1, 0, +1}
            confidences: (N,) float32 array with max probability
            probs: (N, 3) float32 array with all class probabilities
        """
        N = len(features)
        signals = np.zeros(N, dtype=np.int8)
        confidences = np.zeros(N, dtype=np.float32)
        probs = np.zeros((N, 3), dtype=np.float32)

        # Warmup period - no signal
        probs[:self.context_len, 1] = 1.0  # All neutral

        if N <= self.context_len:
            return signals, confidences, probs

        # Valid indices
        valid_indices = list(range(self.context_len, N))

        # Process in batches
        for batch_start in range(0, len(valid_indices), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(valid_indices))
            batch_indices = valid_indices[batch_start:batch_end]

            # Build batch
            batch_windows = []
            for idx in batch_indices:
                window = features[idx - self.context_len:idx]
                batch_windows.append(window)

            # Stack and convert
            batch = np.stack(batch_windows)  # (B, context_len, n_features)
            batch = np.transpose(batch, (0, 2, 1))  # (B, n_features, context_len)

            x = torch.tensor(batch, dtype=torch.float32, device=self.device)

            # Forward pass
            logits = self.model(x)
            batch_probs = F.softmax(logits, dim=-1).cpu().numpy()

            # Process each prediction
            for i, idx in enumerate(batch_indices):
                p = batch_probs[i]
                probs[idx] = p

                # Get predicted class and confidence
                pred_class = np.argmax(p)
                confidence = p[pred_class]
                confidences[idx] = confidence

                # Apply threshold logic
                # Class mapping: 0=Short, 1=Neutral, 2=Long
                if pred_class == 1:
                    # Neutral prediction
                    signals[idx] = 0
                elif confidence >= self.threshold:
                    # Confident Long/Short
                    signals[idx] = pred_class - 1  # {0,2} -> {-1,+1}
                else:
                    # Not confident enough
                    signals[idx] = 0

        # Log summary
        self._log_summary(signals, confidences)

        return signals, confidences, probs

    def _log_summary(self, signals: np.ndarray, confidences: np.ndarray):
        """Log signal generation summary."""
        valid = signals[self.context_len:]
        valid_conf = confidences[self.context_len:]

        n_long = (valid == 1).sum()
        n_short = (valid == -1).sum()
        n_neutral = (valid == 0).sum()
        total = len(valid)

        logger.info(f"Signal generation summary:")
        logger.info(f"  Long: {n_long} ({n_long/total:.1%})")
        logger.info(f"  Short: {n_short} ({n_short/total:.1%})")
        logger.info(f"  Neutral: {n_neutral} ({n_neutral/total:.1%})")
        logger.info(f"  Avg confidence: {valid_conf.mean():.2%}")
        logger.info(f"  Confidence >= {self.threshold:.0%}: {(valid_conf >= self.threshold).sum()}")


def count_signals(signals: np.ndarray) -> dict:
    """Count signal distribution."""
    unique, counts = np.unique(signals, return_counts=True)
    result = {"long": 0, "short": 0, "neutral": 0, "total": len(signals)}

    for val, count in zip(unique, counts):
        if val == 1:
            result["long"] = int(count)
        elif val == -1:
            result["short"] = int(count)
        elif val == 0:
            result["neutral"] = int(count)

    return result


def compute_signal_stats(signals: np.ndarray, context_len: int = 128) -> dict:
    """Compute detailed signal statistics."""
    # Exclude warmup
    valid_signals = signals[context_len:]

    counts = count_signals(valid_signals)
    total_valid = len(valid_signals)

    if total_valid == 0:
        return {"counts": counts, "ratios": {}}

    # Ratios
    ratios = {
        "long_ratio": counts["long"] / total_valid,
        "short_ratio": counts["short"] / total_valid,
        "neutral_ratio": counts["neutral"] / total_valid,
        "active_ratio": (counts["long"] + counts["short"]) / total_valid,
    }

    # Signal transitions
    transitions = {
        "long_to_short": 0,
        "short_to_long": 0,
        "entry_from_neutral": 0,
        "exit_to_neutral": 0,
    }

    for i in range(1, len(valid_signals)):
        prev, curr = valid_signals[i-1], valid_signals[i]
        if prev == 1 and curr == -1:
            transitions["long_to_short"] += 1
        elif prev == -1 and curr == 1:
            transitions["short_to_long"] += 1
        elif prev == 0 and curr != 0:
            transitions["entry_from_neutral"] += 1
        elif prev != 0 and curr == 0:
            transitions["exit_to_neutral"] += 1

    return {
        "counts": counts,
        "ratios": ratios,
        "transitions": transitions,
    }
