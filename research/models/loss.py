import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha # Alpha balancing factor (optional)
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (Batch, C)
        targets: (Batch)
        """
        # 1. Compute Cross Entropy (element-wise)
        # Using log_softmax ensures numerical stability
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # 2. Get probabilities of true class (p_t)
        # p_t = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        
        # 3. Compute Focal Term
        # (1 - p_t)^gamma
        focal_term = (1.0 - pt) ** self.gamma
        
        # 4. Apply Loss
        loss = focal_term * ce_loss
        
        # 5. Apply Alpha
        if self.alpha is not None:
            # Alpha typically is weighting per class.
            # If alpha is scalar, just multiply.
            # If tensor, gather.
            # Blueprint said "No Sample Weights", implies standard FL.
            # For simplicity, self.alpha usually not used here if "Class Balancing: Removed" in Blueprint.
            # But kept argument for compatibility.
            if isinstance(self.alpha, (float, int)):
                loss = loss * self.alpha
            elif isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.gather(0, targets)
                loss = loss * alpha_t
        
        # 6. Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
