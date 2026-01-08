"""
AKF Advanced Strategy Implementation.

Key Differences from Previous Strategies:
1. KF outputs are NOT used directly as signals
2. KF-derived indicators (RSI, Z-Score, Efficiency) used as triggers
3. Regime detection filters out ranging/choppy markets
4. Dynamic risk management with uncertainty-based position sizing

Design Philosophy:
- KF states -> Feature Engineering -> Regime Detection -> Signal Generation
- Only trade in trending regimes with statistically significant impulses
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class AKFAdvancedConfig:
    """Configuration for AKF Advanced Strategy.

    Optimized defaults based on backtest (2020-01 to 2025-11):
    - 168 trades, +31.6% PnL, Sharpe 0.45, PF 1.30
    """

    # Feature parameters
    rsi_period: int = 14

    # Regime detection
    efficiency_threshold: float = 0.4  # Min signal-to-noise for trending (optimized)
    uncertainty_percentile_cap: float = 95  # Avoid extreme volatility

    # Signal thresholds (optimized)
    z_score_long: float = 1.75   # Z-score threshold for long entry
    z_score_short: float = -1.75  # Z-score threshold for short entry
    rsi_overbought: float = 75
    rsi_oversold: float = 25

    # Risk management
    k_sl: float = 2.5  # Stop loss multiplier (uncertainty units)
    target_risk_pct: float = 0.02  # 2% risk per trade
    max_leverage: float = 3.0  # Maximum position size multiplier
    initial_capital: float = 10000.0


class AKFAdvancedStrategy:
    """
    Advanced Kalman Filter Strategy.

    Uses KF outputs to derive trading indicators rather than
    direct price crossovers.

    Flow:
    1. calculate_features() - Generate KF-derived indicators
    2. detect_regime() - Classify market state
    3. generate_signals() - Entry/Exit logic
    4. apply_risk_management() - Position sizing & stops
    """

    REQUIRED_COLUMNS = [
        'close', 'high', 'low',
        'trend', 'velocity', 'trend_pred', 'uncertainty'
    ]

    def __init__(self, config: Optional[AKFAdvancedConfig] = None):
        self.config = config or AKFAdvancedConfig()

    def _validate_input(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    # =========================================================================
    # Module 1: Feature Engineering
    # =========================================================================

    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate RSI from a price series (vectorized)."""
        delta = series.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        # Exponential moving average
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate KF-derived indicators.

        Returns DataFrame with added columns:
        - kf_rsi: RSI calculated from trend (noise-reduced)
        - z_score: Standardized innovation (properly normalized)
        - kf_efficiency: Signal-to-noise ratio (normalized)
        """
        result = df.copy()

        # 1. KF_RSI: RSI using trend instead of close
        result['kf_rsi'] = self._calculate_rsi(
            result['trend'],
            self.config.rsi_period
        )

        # 2. KF_Z_Score: Standardized Innovation
        # Use rolling std of deviation for proper normalization
        deviation = result['close'] - result['trend_pred']
        rolling_std = deviation.rolling(window=20, min_periods=5).std()
        rolling_std = rolling_std.fillna(deviation.std())  # Fill initial NaN
        rolling_std = rolling_std.replace(0, 1e-10)  # Avoid division by zero

        result['z_score'] = deviation / rolling_std

        # Clip extreme values
        result['z_score'] = result['z_score'].clip(-5, 5)

        # 3. KF_Efficiency: Signal-to-Noise Ratio
        # Normalize velocity by its own rolling std for comparability
        velocity_abs = np.abs(result['velocity'])
        velocity_std = velocity_abs.rolling(window=20, min_periods=5).std()
        velocity_std = velocity_std.fillna(velocity_abs.std())
        velocity_std = velocity_std.replace(0, 1e-10)

        # Efficiency = |velocity| / uncertainty, but normalize both
        # Higher efficiency = stronger trend relative to noise
        result['kf_efficiency'] = velocity_abs / (result['uncertainty'] + 1e-10)

        # Normalize efficiency to 0-1 range using percentile
        eff_percentile = result['kf_efficiency'].rolling(
            window=100, min_periods=20
        ).rank(pct=True)
        result['kf_efficiency_norm'] = eff_percentile.fillna(0.5)

        return result

    # =========================================================================
    # Module 2: Market Regime Detection
    # =========================================================================

    def detect_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify market state into Trending (1) or Ranging (0).

        Trending conditions:
        - Normalized KF_Efficiency > threshold (clear trend signal)
        - Uncertainty NOT in extreme percentile (avoid panic/whipsaws)

        Returns DataFrame with 'regime' column.
        """
        result = df.copy()

        # Calculate uncertainty threshold (avoid extreme volatility)
        uncertainty_cap = result['uncertainty'].rolling(
            window=100, min_periods=20
        ).quantile(self.config.uncertainty_percentile_cap / 100)

        # Fill NaN with a large value (conservative)
        uncertainty_cap = uncertainty_cap.fillna(result['uncertainty'].max())

        # Regime conditions using NORMALIZED efficiency (0-1 scale)
        efficiency_ok = result['kf_efficiency_norm'] > self.config.efficiency_threshold
        uncertainty_ok = result['uncertainty'] < uncertainty_cap

        # Trending = both conditions met
        result['regime'] = (efficiency_ok & uncertainty_ok).astype(int)

        # Add debug columns
        result['efficiency_ok'] = efficiency_ok.astype(int)
        result['uncertainty_ok'] = uncertainty_ok.astype(int)

        return result

    # =========================================================================
    # Module 3: Signal Generation
    # =========================================================================

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry/exit signals based on regime and KF indicators.

        Long Entry:
        - Regime == 1 (Trending)
        - velocity > 0 (Upward trend)
        - z_score > threshold (Significant impulse)
        - kf_rsi < overbought (Room to run)

        Short Entry:
        - Regime == 1 (Trending)
        - velocity < 0 (Downward trend)
        - z_score < -threshold (Significant drop)
        - kf_rsi > oversold (Room to fall)

        Exit:
        - Regime flips to 0 (Market turned choppy)
        - Price crosses trend (Mean reversion)

        Returns DataFrame with 'signal' column (1=Long, -1=Short, 0=Flat)
        """
        result = df.copy()

        # Entry conditions
        is_trending = result['regime'] == 1

        # Long conditions
        long_trend = result['velocity'] > 0
        long_impulse = result['z_score'] > self.config.z_score_long
        long_not_overbought = result['kf_rsi'] < self.config.rsi_overbought

        entry_long = is_trending & long_trend & long_impulse & long_not_overbought

        # Short conditions
        short_trend = result['velocity'] < 0
        short_impulse = result['z_score'] < self.config.z_score_short
        short_not_oversold = result['kf_rsi'] > self.config.rsi_oversold

        entry_short = is_trending & short_trend & short_impulse & short_not_oversold

        # Exit conditions (evaluated during position holding)
        regime_exit = result['regime'] == 0
        long_exit = result['close'] < result['trend']  # Price below trend
        short_exit = result['close'] > result['trend']  # Price above trend

        # State machine for signal generation
        n = len(df)
        signal = np.zeros(n, dtype=np.int32)
        position = 0

        for i in range(n):
            if position == 0:
                # Flat: look for entry
                if entry_long.iloc[i]:
                    position = 1
                elif entry_short.iloc[i]:
                    position = -1
            elif position == 1:
                # Long: check exit conditions
                if regime_exit.iloc[i] or long_exit.iloc[i]:
                    position = 0
                # Allow reversal
                elif entry_short.iloc[i]:
                    position = -1
            elif position == -1:
                # Short: check exit conditions
                if regime_exit.iloc[i] or short_exit.iloc[i]:
                    position = 0
                # Allow reversal
                elif entry_long.iloc[i]:
                    position = 1

            signal[i] = position

        result['signal'] = signal

        # Debug columns
        result['entry_long'] = entry_long.astype(int)
        result['entry_short'] = entry_short.astype(int)

        return result

    # =========================================================================
    # Module 4: Dynamic Risk Management
    # =========================================================================

    def apply_risk_management(
        self,
        df: pd.DataFrame,
        capital: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Apply dynamic risk management.

        1. Dynamic Stop Loss:
           - Long SL = close - k_sl * uncertainty
           - Short SL = close + k_sl * uncertainty

        2. Volatility-Targeted Position Sizing:
           - Size = (Capital * Risk%) / (k_sl * uncertainty)
           - Clamped to max_leverage

        Returns DataFrame with:
        - stop_loss_price: Dynamic SL level
        - position_size: Risk-adjusted position size (as fraction of capital)
        """
        result = df.copy()
        capital = capital or self.config.initial_capital

        # Dynamic Stop Loss
        sl_distance = self.config.k_sl * result['uncertainty']

        # Calculate SL price based on signal direction
        result['stop_loss_price'] = np.where(
            result['signal'] == 1,
            result['close'] - sl_distance,  # Long: SL below
            np.where(
                result['signal'] == -1,
                result['close'] + sl_distance,  # Short: SL above
                np.nan  # Flat: no SL
            )
        )

        # Volatility-Targeted Position Sizing
        # Target risk amount
        target_risk = capital * self.config.target_risk_pct

        # Position size (in units of the asset)
        # Risk per unit = k_sl * uncertainty (SL distance)
        risk_per_unit = sl_distance * result['close']  # In dollar terms

        raw_size = target_risk / (risk_per_unit + 1e-10)

        # Convert to leverage (fraction of capital)
        position_value = raw_size * result['close']
        leverage = position_value / capital

        # Clamp to max leverage
        leverage = np.clip(leverage, 0, self.config.max_leverage)

        # Only apply size when in position
        result['position_size'] = np.where(
            result['signal'] != 0,
            leverage,
            0.0
        )

        return result

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    def run_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute complete strategy pipeline.

        Steps:
        1. Validate input
        2. Calculate KF-derived features
        3. Detect market regime
        4. Generate trading signals
        5. Apply risk management

        Returns DataFrame with columns:
        ['signal', 'regime', 'kf_rsi', 'z_score', 'kf_efficiency',
         'position_size', 'stop_loss_price']
        """
        self._validate_input(df)

        # Pipeline
        result = self.calculate_features(df)
        result = self.detect_regime(result)
        result = self.generate_signals(result)
        result = self.apply_risk_management(result)

        return result

    def get_output_columns(self) -> list:
        """Return list of output columns."""
        return [
            'signal',
            'regime',
            'kf_rsi',
            'z_score',
            'kf_efficiency',
            'position_size',
            'stop_loss_price',
        ]


def prepare_kalman_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame with required Kalman columns.

    Maps standard Kalman output columns to expected names:
    - kf_trend -> trend
    - kf_velocity -> velocity
    - kf_trend_pred -> trend_pred
    - kf_uncertainty -> uncertainty (sqrt applied)

    Args:
        df: DataFrame with standard Kalman filter outputs

    Returns:
        DataFrame with renamed columns for AKFAdvancedStrategy
    """
    result = df.copy()

    # Map columns
    column_map = {
        'kf_trend': 'trend',
        'kf_velocity': 'velocity',
        'kf_trend_pred': 'trend_pred',
    }

    for old_name, new_name in column_map.items():
        if old_name in result.columns and new_name not in result.columns:
            result[new_name] = result[old_name]

    # Uncertainty: use sqrt of kf_uncertainty
    if 'kf_uncertainty' in result.columns and 'uncertainty' not in result.columns:
        result['uncertainty'] = np.sqrt(result['kf_uncertainty'])

    return result
