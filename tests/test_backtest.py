import pytest
import pandas as pd
import numpy as np
from research.backtest.backtester import Backtester

def test_backtest_pnl_with_fees():
    """Test basic PnL calculation with transaction costs."""
    # Scenario: 2 Trades.
    # 1. Long at 100, Exit at 105. (5% Gross). Fee 0.1% each side.
    # 2. Short at 100, Exit at 95. (5% Gross). Fee 0.1% each side.
    
    signals = pd.Series([1, 0, 0, -1, 0, 0], index=pd.date_range("2024-01-01", periods=6, freq="D"))
    prices = pd.Series([100, 102, 105, 100, 98, 95], index=signals.index)
    # We need to explicitly signal exits or let the backtester handle it.
    # Let's assume a "Event-Driven" or "Signal-based" input.
    # For Vectorized, usually we convert signals to positions.
    # Position: 1 (Day 0-2), 0 (Day 3), -1 (Day 3-5).
    # Wait, simpler logic: Input 'Events' DataFrame with Entry/Exit prices.
    
    # Or, feed OHLCV and Signal series. Backtester runs loop (Vectorized-Loop Hybrid).
    # Let's test the Engine PnL logic directly.
    
    tester = Backtester(fee_rate=0.001)
    
    # Trade 1: Entry 100, Exit 105.
    # Size 100 USD.
    # Entry Fee: 0.1. Cost 100.
    # Exit Value: 105. Exit Fee: 0.105.
    # Net: 105 - 0.105 - (100 + 0.1) = 104.895 - 100.1 = 4.795.
    # Return: 4.795 / 100 = 4.795%.
    
    net_pnl = tester.calculate_trade_pnl(entry_price=100, exit_price=105, side=1, size=100)
    expected_pnl = 100 * (105/100 - 1) - (100 * 0.001) - (105 * 0.001) # Approx
    
    # Detailed:
    # Qty = Size / Entry. = 1 unit.
    # Exit Val = 1 * 105 = 105.
    # Fees = 100*0.001 + 105*0.001 = 0.1 + 0.105 = 0.205.
    # Gross PnL = 5.
    # Net = 4.795.
    
    assert np.isclose(net_pnl, 4.795, atol=1e-4)

def test_pyramiding_buffered_sl():
    """
    Test Risk-Free Pyramiding Logic.
    Scenario:
    - Entry at 100. Volatility Sigma = 2.0.
    - Initial SL = 100 - 2.0 = 98.
    - Price rises to 102 (Profit = 2.0 = 1 Sigma).
    - Logic triggers Scale-In (Pyramid).
    - New Qty added. New Avg Entry calculated.
    - NEW SL should be BEP - 0.5 * Sigma.
    """
    tester = Backtester(fee_rate=0.0) # Simplify fees
    
    # State 0: Entry 100, Size 1. SL 98.
    position = {
        "avg_entry": 100.0,
        "size": 1.0,
        "sl": 98.0,
        "sigma": 2.0
    }
    
    # Action: Pyramid at 102. Size +0.5.
    current_price = 102.0
    
    new_pos = tester.apply_pyramiding(position, current_price, add_size=0.5)
    
    # Check New Avg Entry
    # (1*100 + 0.5*102) / 1.5 = (100 + 51) / 1.5 = 151 / 1.5 = 100.666...
    expected_avg = 100.6666
    assert np.isclose(new_pos["avg_entry"], expected_avg, atol=1e-3)
    
    # Check Buffered SL
    # Rule: SL = AvgEntry - 0.5 * Sigma
    # SL = 100.666 - 0.5 * 2.0 = 100.666 - 1.0 = 99.666
    # Note: 99.666 > 98 (Original SL). Risk is tightened.
    # Avg Entry is 100.66. SL is 99.66. Distance is 1.0 (0.5 Sigma).
    # If price hits 99.66, we lose 1.0 per unit.
    # Wait, "Risk-Free" usually means SL >= Break Even.
    # "Buffered BEP" implies SL is slightly BELOW BEP to allow noise.
    # User Spec: "Risk-free pyramiding with buffered BEP".
    # And "SL = BEP - 0.5 * Sigma".
    # Since SL < BEP, it's NOT strictly risk-free (you lose 0.5 sigma).
    # But it secures *most* of the profit if the move was large.
    # Let's verify it follows the formula.
    
    expected_sl = new_pos["avg_entry"] - (0.5 * position["sigma"])
    assert np.isclose(new_pos["sl"], expected_sl, atol=1e-3)

def test_equity_curve_drawdown():
    """Test MDD calculation."""
    # Equity: 100, 110, 90, 120.
    # Peak: 100 -> 110 -> 110 -> 120.
    # DD: 0, 0, (90-110)/110 = -18%, 0.
    # MDD = 18.18%
    
    equity = pd.Series([100, 110, 90, 120])
    mdd = Backtester.calculate_mdd(equity)
    
    expected = (90 - 110) / 110
    assert np.isclose(mdd, expected, atol=1e-4) # -0.1818
