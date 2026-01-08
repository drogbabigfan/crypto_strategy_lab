import pytest
import pandas as pd
import numpy as np
from research.utils.optimization import ThresholdOptimizer

def test_confidence_threshold_optimization():
    """Test finding optimal confidence threshold."""
    # Scenario:
    # Model predictions (probabilities for Long class 1)
    # True returns for the period
    
    # Generate synthetic data
    # 3 trades. 
    # Trade A: Prob 0.55 (Weak), Ret -1%
    # Trade B: Prob 0.80 (Strong), Ret +2%
    # Trade C: Prob 0.90 (Strong), Ret +1%
    
    probs = pd.Series([0.55, 0.80, 0.90])
    returns = pd.Series([-0.01, 0.02, 0.01])
    
    optimizer = ThresholdOptimizer(min_thresh=0.5, max_thresh=0.95, step=0.05)
    
    # If Threshold = 0.5 (All trades taken):
    # Sum Ret = -1 + 2 + 1 = +2%. Count=3. Avg = 0.66%.
    
    # If Threshold = 0.6 (Exclude Trade A):
    # Trades B, C.
    # Sum Ret = 2 + 1 = +3%. Count=2. Avg = 1.5%.
    
    # If Threshold = 0.85 (Exclude A, B):
    # Trade C.
    # Sum Ret = 1%. Count=1. Avg = 1.0%.
    
    # Optimizer should pick Threshold ~ 0.6 or 0.8 depending on metric (Total PnL vs Sharpe vs Avg Trade).
    # Let's target "Total PnL" for simplicity in this test.
    
    best_thresh, best_metric = optimizer.optimize(probs, returns, metric="total_pnl")
    
    # Best Total PnL is at Threshold 0.6 (Total 0.03).
    # Note: 0.55 is strictly < 0.6? If step is 0.05.
    # Thresholds checked: 0.5, 0.55, 0.60 ...
    # If >= 0.55 included?
    
    assert best_thresh >= 0.6
    assert best_metric >= 0.03

def test_empty_input():
    optimizer = ThresholdOptimizer()
    t, m = optimizer.optimize(pd.Series([]), pd.Series([]))
    assert t is None
