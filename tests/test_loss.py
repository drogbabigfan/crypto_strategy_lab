import pytest
import torch
import torch.nn.functional as F
from research.models.loss import FocalLoss

def test_focal_loss_reduction():
    """Test Focal Loss computation."""
    # Batch size 5, 3 classes
    logits = torch.randn(5, 3, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 0, 1])
    
    criterion = FocalLoss(gamma=2.0, alpha=None)
    loss = criterion(logits, targets)
    
    # Should be scalar
    assert loss.dim() == 0
    assert loss > 0
    
    # Backward
    loss.backward()
    assert logits.grad is not None

def test_focal_loss_vs_cross_entropy():
    """Focal Loss with gamma=0 should allow recovering approx CE behavior."""
    logits = torch.randn(10, 3)
    targets = torch.randint(0, 3, (10,))
    
    # CE
    ce_loss = F.cross_entropy(logits, targets)
    
    # Focal (gamma=0)
    # Note: Pytorch CE is size_average=True by default. 
    focal = FocalLoss(gamma=0.0)
    fl_loss = focal(logits, targets)
    
    assert torch.isclose(ce_loss, fl_loss, atol=1e-5)
