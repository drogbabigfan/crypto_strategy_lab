"""
Loss Functions for Fine-tuning (v4.6)

Focal Loss with Label Smoothing for class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Features:
    - Focal modulation: Down-weights easy examples
    - Label smoothing: Prevents overconfidence
    - Optional class weights: Further balance classes

    Args:
        gamma: Focusing parameter (default 2.0)
            - gamma=0: Standard cross-entropy
            - gamma>0: Down-weights easy examples
        label_smoothing: Smoothing factor (default 0.1)
            - 0: No smoothing
            - >0: Distribute probability to other classes
        weight: Optional class weights tensor
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        weight: torch.Tensor = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Focal Loss with label smoothing.

        Args:
            logits: (Batch, n_classes) raw model outputs
            targets: (Batch,) class indices

        Returns:
            Scalar loss tensor (if reduction='mean' or 'sum')
            or (Batch,) tensor (if reduction='none')
        """
        n_classes = logits.size(-1)
        batch_size = logits.size(0)

        # Apply label smoothing to create soft targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.label_smoothing / n_classes)
            smooth_targets.scatter_(
                1,
                targets.unsqueeze(1),
                1 - self.label_smoothing + self.label_smoothing / n_classes
            )

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)

        # Focal weight: (1 - p_t)^gamma
        # p_t = sum of probs weighted by smooth targets
        pt = (probs * smooth_targets).sum(dim=-1)
        focal_weight = (1 - pt) ** self.gamma

        # Cross entropy with soft targets
        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss = -torch.sum(smooth_targets * log_probs, dim=-1)

        # Apply focal modulation
        loss = focal_weight * ce_loss

        # Apply class weights if provided
        if self.weight is not None:
            weight = self.weight.to(logits.device)
            sample_weights = weight[targets]
            loss = loss * sample_weights

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CrossEntropyWithSmoothing(nn.Module):
    """
    Standard Cross Entropy with Label Smoothing.

    For comparison with Focal Loss.
    """

    def __init__(
        self,
        label_smoothing: float = 0.1,
        weight: torch.Tensor = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross entropy with label smoothing."""
        n_classes = logits.size(-1)

        # Soft targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.label_smoothing / n_classes)
            smooth_targets.scatter_(
                1,
                targets.unsqueeze(1),
                1 - self.label_smoothing + self.label_smoothing / n_classes
            )

        # Cross entropy
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(smooth_targets * log_probs, dim=-1)

        # Class weights
        if self.weight is not None:
            weight = self.weight.to(logits.device)
            sample_weights = weight[targets]
            loss = loss * sample_weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
