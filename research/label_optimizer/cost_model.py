"""
Square Root Cost Model (v4.6)

Nonlinear market impact model for realistic cost estimation.
Impact = η × σ × √(Q / V)
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SquareRootCostModel:
    """
    Square Root Market Impact Model.

    Attributes:
        base_fee: Exchange fee (0.1% = 0.001)
        base_slippage: Minimum slippage (0.01% = 0.0001)
        impact_coeff: Impact coefficient (η)
        slippage_cap: Maximum slippage cap (0.5% = 0.005)
        min_profit_buffer: Required profit multiplier over cost
    """

    base_fee: float = 0.001  # 0.10%
    base_slippage: float = 0.0001  # 0.01%
    impact_coeff: float = 0.1
    slippage_cap: float = 0.005  # 0.50%
    min_profit_buffer: float = 1.5

    def get_slippage(
        self,
        volatility: float,
        trade_size: float = 0.01,
        avg_volume: float = 1.0,
    ) -> float:
        """
        Calculate slippage using Square Root Law.

        slippage = base + η × σ × √(trade_size / volume)

        Args:
            volatility: Current volatility (σ)
            trade_size: Order size as fraction of volume
            avg_volume: Average volume for normalization

        Returns:
            Total slippage (capped)
        """
        if avg_volume <= 0:
            return self.slippage_cap

        volume_ratio = trade_size / avg_volume
        if volume_ratio < 0:
            volume_ratio = 0
        impact = self.impact_coeff * volatility * np.sqrt(volume_ratio)

        total_slippage = self.base_slippage + impact
        return min(total_slippage, self.slippage_cap)

    def get_total_cost(
        self,
        volatility: float,
        trade_size: float = 0.01,
        avg_volume: float = 1.0,
    ) -> float:
        """
        Calculate round-trip total cost (entry + exit).

        Returns:
            Total cost as fraction of position
        """
        slippage = self.get_slippage(volatility, trade_size, avg_volume)
        return 2 * (self.base_fee + slippage)

    def is_viable(
        self,
        pt_mult: float,
        sigma: float,
        volume_ratio: float = 1.0,
    ) -> bool:
        """
        Fee Trap Guard: Check if trade is profitable.

        Expected Profit = PT × σ
        Required = Total Cost × Buffer

        Args:
            pt_mult: Profit target multiplier
            sigma: Current volatility
            volume_ratio: Trade size / average volume

        Returns:
            True if expected profit exceeds required threshold
        """
        expected_profit = pt_mult * sigma
        total_cost = self.get_total_cost(sigma, trade_size=volume_ratio)
        required = total_cost * self.min_profit_buffer

        return expected_profit > required

    def get_breakeven_pt(self, sigma: float, volume_ratio: float = 1.0) -> float:
        """
        Calculate minimum PT multiplier for breakeven.

        Args:
            sigma: Current volatility
            volume_ratio: Trade size / average volume

        Returns:
            Minimum PT multiplier needed
        """
        if sigma <= 0:
            return float("inf")

        total_cost = self.get_total_cost(sigma, trade_size=volume_ratio)
        return (total_cost * self.min_profit_buffer) / sigma
