import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, fee_rate=0.001):
        self.fee_rate = fee_rate
    
    def calculate_trade_pnl(self, entry_price, exit_price, side, size):
        """
        Calculate Net PnL for a single trade.
        side: 1 (Long), -1 (Short).
        size: Capital allocated (Quote Currency, e.g. USDT).
        """
        # Quantities
        qty = size / entry_price
        
        # Gross PnL
        if side == 1:
            gross_pnl = (exit_price - entry_price) * qty
        else:
            gross_pnl = (entry_price - exit_price) * qty
            
        # Fees
        # Entry Fee = Size * Rate
        entry_fee = size * self.fee_rate
        
        # Exit Fee = ExitValue * Rate
        exit_val = exit_price * qty
        exit_fee = exit_val * self.fee_rate
        
        net_pnl = gross_pnl - entry_fee - exit_fee
        return net_pnl

    def apply_pyramiding(self, position, current_price, add_size):
        """
        Update position with pyramiding logic and Buffered SL.
        position: dict {avg_entry, size, sl, sigma}
        """
        old_size = position["size"]
        old_entry = position["avg_entry"]
        
        # New Size
        new_size = old_size + add_size
        
        # New Avg Entry
        # (Size1 * Price1 + Size2 * Price2) / TotalSize (Weighted Average)
        # Assuming 'size' is in Quantities? Or Value?
        # In test I passed 0.5 as "add_size" to 1.0 "size". 
        # If these are units (BTC):
        total_vol = old_size + add_size
        new_entry = (old_size * old_entry + add_size * current_price) / total_vol
        
        # If these are USD Value:
        # We need to convert to units first.
        # But let's assume 'size' tracks Units for Avg Entry calc.
        # Test assumption was: (1*100 + 0.5*102) / 1.5. This implies Units.
        # So we stick to Units.
        
        # Buffered SL Logic
        # SL = AvgEntry - 0.5 * Sigma
        sigma = position["sigma"]
        # Direction? Assuming Long for pyramiding (usually).
        # Formula: SL = BEP - 0.5 * sigma
        new_sl = new_entry - (0.5 * sigma)
        
        return {
            "avg_entry": new_entry,
            "size": new_size,
            "sl": new_sl,
            "sigma": sigma
        }

    @staticmethod
    def calculate_mdd(equity_curve):
        """
        Calculate Maximum Drawdown.
        equity_curve: Series of Equity values.
        """
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()
