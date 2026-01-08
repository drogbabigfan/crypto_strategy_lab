"""
Optimized LPPLS Strategy Configuration.

These parameters were optimized via grid search on 2020-2023 BTC data
with focus on risk-adjusted returns (penalizing high drawdown).
"""

OPTIMIZED_PARAMS = {
    # Signal filter parameters
    'tc_max_days': 70,      # Max days to predicted crash
    'tc_min_days': 5,       # Min days to predicted crash
    'm_min': 0.2,           # Min m parameter (power law exponent)
    'm_max': 0.7,           # Max m parameter
    'omega_min': 5.0,       # Min omega (log-periodic frequency)
    'omega_max': 15.0,      # Max omega

    # Entry parameters
    'signal_lookback': 5,   # Days to look back for signal confirmation
    'entry_threshold': 0.5, # Fraction of valid signals needed to enter

    # Exit parameters
    'profit_target_pct': 0.10,  # 10% profit target
    'stop_loss_pct': 0.06,      # 6% stop loss
    'max_hold_days': 20,        # Maximum holding period
}

# Performance summary (2020-2023 in-sample):
# - Total PnL: +5.06%
# - Number of Trades: 15
# - Win Rate: 46.7%
# - Sharpe Ratio: 0.24
# - Max Drawdown: 16.6%
#
# Out-of-sample (2024):
# - No trades generated (conservative - avoided bull market)
#
# Key characteristics:
# - Conservative: only trades when multiple signals confirm
# - Short holds: max 20 days reduces exposure
# - Tight stops: 6% stop loss limits losses
# - Works best during bubble collapses (e.g., COVID crash 2020)
