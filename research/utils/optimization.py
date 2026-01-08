import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

class ThresholdOptimizer:
    def __init__(self, min_thresh=0.5, max_thresh=0.95, step=0.05):
        self.thresholds = np.arange(min_thresh, max_thresh + step/2, step)
        
    def optimize(self, probs, returns, metric="total_pnl"):
        """
        Find optimal threshold.
        probs: Series of confidence scores (0-1).
        returns: Series of realized returns.
        """
        if len(probs) == 0:
            return None, None
            
        best_metric = -np.inf
        best_thresh = None
        
        for thresh in self.thresholds:
            # Filter trades
            mask = probs >= thresh
            selected_returns = returns[mask]
            
            if len(selected_returns) == 0:
                continue
                
            # Calc Metric
            if metric == "total_pnl":
                current_metric = selected_returns.sum()
            elif metric == "avg_pnl":
                current_metric = selected_returns.mean()
            elif metric == "sharpe":
                if selected_returns.std() == 0:
                    current_metric = 0 # or -inf
                else:
                    current_metric = selected_returns.mean() / selected_returns.std()
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            # Update Best
            if current_metric > best_metric:
                best_metric = current_metric
                best_thresh = thresh
                
        return best_thresh, best_metric
