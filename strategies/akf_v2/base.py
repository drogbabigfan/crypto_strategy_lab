"""
Base Strategy Class for AKF V2.

Provides common functionality:
- Feature generation pipeline
- Signal generation interface
- Position sizing
- Risk management
- Backtesting utilities
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from .features import LogKalmanFeatures, FeatureConfig


@dataclass
class StrategyConfig:
    """Base configuration for all strategies."""

    # Feature generation
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)

    # Warmup period (bars to skip at start)
    warmup: int = 50

    # Position sizing
    use_confidence_sizing: bool = True
    base_position_size: float = 1.0
    max_position_size: float = 2.0
    min_position_size: float = 0.25

    # Risk management
    max_drawdown_exit: float = 0.1  # Exit if position DD > 10%
    trailing_stop_atr_mult: float = 2.0


class BaseStrategy(ABC):
    """
    Abstract base class for AKF V2 strategies.

    Subclasses must implement:
    - _generate_raw_signals(): Core signal logic
    """

    REQUIRED_COLUMNS = ['open', 'high', 'low', 'close']

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.feature_generator = LogKalmanFeatures(self.config.feature_config)

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def run_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute complete strategy pipeline.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with signals and all features
        """
        self._validate_input(df)

        # Step 1: Generate features
        result = self.feature_generator.generate(df)

        # Step 2: Generate raw signals
        result = self._generate_raw_signals(result)

        # Step 3: Apply position sizing
        result = self._apply_position_sizing(result)

        # Step 4: Apply warmup (zero out early signals)
        result.loc[:self.config.warmup - 1, 'signal'] = 0

        return result

    @abstractmethod
    def _generate_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate raw trading signals.

        Must add 'signal' column with values in {-1, 0, 1}.

        Args:
            df: DataFrame with all features

        Returns:
            DataFrame with 'signal' column added
        """
        pass

    def _apply_position_sizing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply position sizing based on confidence.

        When confidence is high, increase position size.
        """
        result = df.copy()

        if not self.config.use_confidence_sizing:
            result['position_size'] = self.config.base_position_size
            return result

        # Scale position size by confidence
        if 'confidence' in result.columns:
            # confidence is 0-1, scale to position size range
            size_range = self.config.max_position_size - self.config.min_position_size
            result['position_size'] = (
                self.config.min_position_size +
                result['confidence'] * size_range
            )
        else:
            result['position_size'] = self.config.base_position_size

        return result

    def get_signal_stats(self, df: pd.DataFrame) -> dict:
        """Get statistics about generated signals."""
        if 'signal' not in df.columns:
            return {}

        signals = df['signal']
        total = len(signals)

        return {
            'total_bars': total,
            'long_bars': (signals == 1).sum(),
            'short_bars': (signals == -1).sum(),
            'neutral_bars': (signals == 0).sum(),
            'long_pct': (signals == 1).mean() * 100,
            'short_pct': (signals == -1).mean() * 100,
            'active_pct': (signals != 0).mean() * 100,
        }


class MomentumStrategy(BaseStrategy):
    """
    Momentum-based strategy using velocity signals.

    Entry:
    - Long: velocity_signal > threshold AND regime == 1 (trending)
    - Short: velocity_signal < -threshold AND regime == 1

    Exit:
    - Signal reversal or regime change
    """

    @dataclass
    class Config(StrategyConfig):
        """Momentum strategy specific config."""
        velocity_threshold: float = 0.3  # Entry threshold
        require_acceleration: bool = True  # Require velocity_accel confirmation
        regime_filter: bool = True  # Only trade in trending regime

    def __init__(self, config: Optional['MomentumStrategy.Config'] = None):
        super().__init__(config or MomentumStrategy.Config())

    def _generate_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        cfg = self.config

        # Base conditions
        vel_long = result['velocity_signal'] > cfg.velocity_threshold
        vel_short = result['velocity_signal'] < -cfg.velocity_threshold

        # Acceleration confirmation
        if cfg.require_acceleration:
            accel_long = result['velocity_accel'] > 0
            accel_short = result['velocity_accel'] < 0
            vel_long = vel_long & accel_long
            vel_short = vel_short & accel_short

        # Regime filter
        if cfg.regime_filter:
            trending = result['regime'] == 1
            vel_long = vel_long & trending
            vel_short = vel_short & trending

        # Generate signals
        result['signal'] = 0
        result.loc[vel_long, 'signal'] = 1
        result.loc[vel_short, 'signal'] = -1

        return result


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using deviation signals.

    Entry:
    - Long: mr_signal < -threshold (oversold)
    - Short: mr_signal > threshold (overbought)

    Only in ranging regime (regime == 0).
    """

    @dataclass
    class Config(StrategyConfig):
        """Mean reversion specific config."""
        mr_threshold: float = 2.0  # Z-score threshold
        regime_filter: bool = True  # Only trade in ranging regime
        velocity_filter: bool = True  # Avoid trading against strong trend

    def __init__(self, config: Optional['MeanReversionStrategy.Config'] = None):
        super().__init__(config or MeanReversionStrategy.Config())

    def _generate_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        cfg = self.config

        # Base conditions
        oversold = result['mr_signal'] < -cfg.mr_threshold
        overbought = result['mr_signal'] > cfg.mr_threshold

        # Regime filter (only in ranging market)
        if cfg.regime_filter:
            ranging = result['regime'] == 0
            oversold = oversold & ranging
            overbought = overbought & ranging

        # Velocity filter (don't fight strong trends)
        if cfg.velocity_filter:
            no_strong_down = result['velocity_signal'] > -0.5
            no_strong_up = result['velocity_signal'] < 0.5
            oversold = oversold & no_strong_down
            overbought = overbought & no_strong_up

        # Generate signals
        result['signal'] = 0
        result.loc[oversold, 'signal'] = 1  # Buy oversold
        result.loc[overbought, 'signal'] = -1  # Sell overbought

        return result


class HybridStrategy(BaseStrategy):
    """
    Hybrid strategy combining momentum and mean reversion.

    - In trending regime: Follow momentum
    - In ranging regime: Mean reversion
    - In volatile regime: Reduce exposure
    """

    @dataclass
    class Config(StrategyConfig):
        """Hybrid strategy config."""
        # Momentum params
        velocity_threshold: float = 0.3

        # Mean reversion params
        mr_threshold: float = 1.5

        # Regime-based sizing
        trending_size_mult: float = 1.5
        ranging_size_mult: float = 1.0
        volatile_size_mult: float = 0.5

    def __init__(self, config: Optional['HybridStrategy.Config'] = None):
        super().__init__(config or HybridStrategy.Config())

    def _generate_raw_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        cfg = self.config

        result['signal'] = 0

        # Trending regime: Momentum
        trending = result['regime'] == 1
        mom_long = trending & (result['velocity_signal'] > cfg.velocity_threshold)
        mom_short = trending & (result['velocity_signal'] < -cfg.velocity_threshold)

        # Ranging regime: Mean reversion
        ranging = result['regime'] == 0
        mr_long = ranging & (result['mr_signal'] < -cfg.mr_threshold)
        mr_short = ranging & (result['mr_signal'] > cfg.mr_threshold)

        # Volatile regime: No new entries (but don't override existing)
        # volatile = result['regime'] == -1

        # Combine signals
        result.loc[mom_long | mr_long, 'signal'] = 1
        result.loc[mom_short | mr_short, 'signal'] = -1

        # Adjust position size by regime
        result['regime_size_mult'] = cfg.ranging_size_mult
        result.loc[trending, 'regime_size_mult'] = cfg.trending_size_mult
        result.loc[result['regime'] == -1, 'regime_size_mult'] = cfg.volatile_size_mult

        return result

    def _apply_position_sizing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Override to include regime-based sizing."""
        result = super()._apply_position_sizing(df)

        if 'regime_size_mult' in result.columns:
            result['position_size'] *= result['regime_size_mult']
            result['position_size'] = result['position_size'].clip(
                self.config.min_position_size,
                self.config.max_position_size
            )

        return result
