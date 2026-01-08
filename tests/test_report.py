import pytest
import pandas as pd
import numpy as np
from research.utils.report import StrategyReporter

def test_strategy_metrics():
    """Test calculation of detailed strategy metrics."""
    # Equity Curve
    equity = pd.Series([100, 105, 102, 110, 120]) # Returns: 5%, -3%, 8%, 9%
    
    reporter = StrategyReporter(initial_capital=100)
    metrics = reporter.calculate_metrics(equity)
    
    # 1. Total Return
    # 120 / 100 - 1 = 20%
    assert np.isclose(metrics["total_return"], 0.20)
    
    # 2. Max Drawdown
    # Max was 105 -> 102 (-2.8%). Check exact.
    # 105 to 102 is (102-105)/105 = -0.02857
    assert np.isclose(metrics["max_drawdown"], -0.02857, atol=1e-4)
    
    # 3. Sharpe Ratio
    # Daily returns? Let's assume series is daily.
    # Returns: [0.05, -0.028, 0.078, 0.09]
    returns = equity.pct_change().dropna()
    expected_sharpe = returns.mean() / returns.std() * np.sqrt(252) # Annualized
    assert np.isclose(metrics["sharpe_ratio"], expected_sharpe, atol=1e-3)
    
    # 4. Calmar Ratio
    # Annualized Return / Max DD
    # Need Annualized Return calc. Logic usually depends on frequency.
    # Reporter should assume daily or handle freq? Assumed daily for this test.
    pass

def test_html_report_generation():
    """Test logic to create HTML string."""
    reporter = StrategyReporter()
    metrics = {"total_return": 0.20, "sharpe_ratio": 2.5, "max_drawdown": -0.10}
    
    html = reporter.generate_html(metrics, title="Test Strategy")
    
    assert "Test Strategy" in html
    assert "20.00%" in html # Formatted return
    assert "2.50" in html # Formatted Sharpe
