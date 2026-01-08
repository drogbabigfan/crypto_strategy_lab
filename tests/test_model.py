import pytest
import torch
import torch.nn as nn
from research.models.patchtst import PatchTSTClassifier

def test_patchtst_forward_shape():
    """Verify Forward Pass Output Shape."""
    # Batch=32, Channels=2 (Dynamics, Context), Length=512
    x = torch.randn(32, 2, 512)
    
    model = PatchTSTClassifier(
        seq_len=512,
        n_vars=2,
        patch_len=16,
        stride=16,
        n_classes=3, # Long, Short, Neutral
        d_model=128,
        head_type="residual_bottleneck"
    )
    
    out = model(x)
    
    # Expected: (32, 3)
    assert out.shape == (32, 3)

def test_residual_bottleneck_gradients():
    """Verify Gradients flow through Bottleneck."""
    x = torch.randn(2, 2, 512, requires_grad=True)
    model = PatchTSTClassifier(seq_len=512, n_vars=2, n_classes=3)
    
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None
    assert torch.abs(x.grad).sum() > 0
    
    # Verify Head weights have grads
    # (Checking if 'head' parameters were used)
    for name, param in model.named_parameters():
        if "head" in name and param.requires_grad:
            assert param.grad is not None

def test_fp16_compatibility():
    """Verify FP16 (Half Precision) execution."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for FP16 test")
        
    x = torch.randn(4, 2, 512).cuda().half()
    model = PatchTSTClassifier(seq_len=512, n_vars=2, n_classes=3).cuda().half()
    
    out = model(x)
    assert out.dtype == torch.float16
    assert out.shape == (4, 3)
