"""
Hybrid HMM-Hurst Regime Detection Model.

Combines:
- HMM: Identifies market state (Bull/Bear/Sideways)
- Hurst: Determines persistence (Trending vs Mean-Reverting)

Strategy Recommendations:
- Bull + Trending (H>0.5): Momentum long strategies
- Bear + Trending (H>0.5): Momentum short or exit
- Sideways + Mean-Reverting (H<0.5): Mean reversion strategies
- High volatility regimes: Reduce position size
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from .hmm import HMMRegimeDetector, HMMConfig, MarketRegime, RegimeState
from .hurst import HurstCalculator, HurstResult, HurstRegime


class StrategyMode(Enum):
    """Trading strategy mode based on hybrid regime."""
    MOMENTUM_LONG = "momentum_long"         # Bull + Trending
    MOMENTUM_SHORT = "momentum_short"       # Bear + Trending
    MEAN_REVERSION = "mean_reversion"       # Sideways + Mean-Reverting
    CAUTIOUS = "cautious"                   # High uncertainty
    DEFENSIVE = "defensive"                 # Bear + any (capital preservation)
    NEUTRAL = "neutral"                     # No clear signal


@dataclass
class HybridRegimeState:
    """Combined regime state from HMM and Hurst analysis."""
    # HMM state
    market_regime: MarketRegime
    regime_probability: float
    regime_probabilities: Dict[MarketRegime, float]

    # Hurst state
    hurst_value: float
    hurst_regime: HurstRegime
    hurst_confidence: float

    # Combined recommendation
    strategy_mode: StrategyMode
    position_scale: float  # 0.0 to 1.0, how much of normal position size

    # Metadata
    timestamp: Optional[pd.Timestamp] = None

    @property
    def is_trending(self) -> bool:
        return self.hurst_regime == HurstRegime.TRENDING

    @property
    def is_mean_reverting(self) -> bool:
        return self.hurst_regime == HurstRegime.MEAN_REVERTING

    @property
    def is_bull(self) -> bool:
        return self.market_regime == MarketRegime.BULL

    @property
    def is_bear(self) -> bool:
        return self.market_regime == MarketRegime.BEAR


@dataclass
class HybridConfig:
    """Configuration for hybrid regime detector."""
    # HMM config
    hmm_n_states: int = 3
    hmm_n_iter: int = 100
    hmm_vol_window: int = 20

    # Hurst config
    hurst_window: int = 100
    hurst_method: str = 'dfa'
    trending_threshold: float = 0.55
    mean_revert_threshold: float = 0.45

    # Strategy config
    min_confidence: float = 0.6       # Min probability to act on regime
    vol_scaling: bool = True          # Scale position by volatility
    max_position_scale: float = 1.0   # Max position multiplier
    min_position_scale: float = 0.2   # Min position multiplier

    # Feature config
    use_hurst_as_feature: bool = True  # Include Hurst in HMM features


class HybridRegimeDetector:
    """
    Hybrid regime detector combining HMM and Hurst exponent.

    Architecture:
    1. HMM identifies macro regime (Bull/Bear/Sideways)
    2. Hurst exponent identifies micro structure (Trending/Mean-Reverting)
    3. Combined signal determines optimal trading strategy

    This follows the "hierarchical" approach recommended in the research:
    HMM as meta-controller, Hurst as tactical indicator.
    """

    def __init__(self, config: Optional[HybridConfig] = None):
        self.config = config or HybridConfig()

        # Initialize components
        self.hmm_detector = HMMRegimeDetector(
            HMMConfig(
                n_states=self.config.hmm_n_states,
                n_iter=self.config.hmm_n_iter,
                vol_window=self.config.hmm_vol_window,
                features=['returns', 'volatility'],
            )
        )

        self.hurst_calculator = HurstCalculator(
            trending_threshold=self.config.trending_threshold,
            mean_revert_threshold=self.config.mean_revert_threshold,
        )

        self.is_fitted = False
        self._vol_mean: Optional[float] = None
        self._vol_std: Optional[float] = None

    def fit(self, data: pd.DataFrame) -> 'HybridRegimeDetector':
        """
        Fit the hybrid model to historical data.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            self for method chaining
        """
        # Prepare features with Hurst if configured
        if self.config.use_hurst_as_feature:
            data = self._add_hurst_feature(data)
            self.hmm_detector.config.features = ['returns', 'volatility', 'hurst']

        # Fit HMM
        self.hmm_detector.fit(data)

        # Store volatility statistics for position scaling
        returns = data['close'].pct_change()
        volatility = returns.rolling(self.config.hmm_vol_window).std()
        self._vol_mean = volatility.mean()
        self._vol_std = volatility.std()

        self.is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict hybrid regime states for historical data.

        Uses Viterbi (posterior) decoding - looks at full sequence.
        Appropriate for backtesting and analysis.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            DataFrame with regime predictions and strategy recommendations
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Add Hurst feature if needed
        if self.config.use_hurst_as_feature:
            data = self._add_hurst_feature(data)

        # Get HMM predictions
        hmm_results = self.hmm_detector.predict(data)

        # Calculate rolling Hurst
        hurst_results = self.hurst_calculator.calculate_rolling(
            data['close'].pct_change().dropna(),
            window=self.config.hurst_window,
            method=self.config.hurst_method,
        )

        # Align indices
        hurst_results = hurst_results.reindex(data.index)

        # Combine results
        result = pd.DataFrame(index=data.index)

        # HMM columns
        result['market_regime'] = hmm_results['regime']
        result['regime_prob'] = hmm_results['confidence']
        result['prob_bull'] = hmm_results['prob_bull']
        result['prob_bear'] = hmm_results['prob_bear']
        result['prob_sideways'] = hmm_results['prob_sideways']

        # Hurst columns
        result['hurst'] = hurst_results['hurst']
        result['hurst_regime'] = hurst_results['regime']
        result['hurst_confidence'] = hurst_results['confidence']

        # Generate strategy recommendations
        strategies = []
        position_scales = []

        for i in range(len(result)):
            row = result.iloc[i]
            strategy, scale = self._determine_strategy(
                market_regime=row['market_regime'],
                regime_prob=row['regime_prob'],
                hurst=row['hurst'],
                hurst_regime=row['hurst_regime'],
                volatility=self._get_volatility(data, i),
            )
            strategies.append(strategy.value if strategy else 'unknown')
            position_scales.append(scale)

        result['strategy_mode'] = strategies
        result['position_scale'] = position_scales

        return result

    def filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Online filtering: predict using only past data.

        Appropriate for live trading - no look-ahead bias.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            DataFrame with filtered regime predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before filtering")

        # Add Hurst feature if needed
        if self.config.use_hurst_as_feature:
            data = self._add_hurst_feature(data)

        # Get HMM filtered results
        hmm_results = self.hmm_detector.filter(data)

        # Calculate rolling Hurst (already causal)
        hurst_results = self.hurst_calculator.calculate_rolling(
            data['close'].pct_change().dropna(),
            window=self.config.hurst_window,
            method=self.config.hurst_method,
        )
        hurst_results = hurst_results.reindex(data.index)

        # Combine results
        result = pd.DataFrame(index=data.index)

        result['market_regime'] = hmm_results['regime']
        result['regime_prob'] = hmm_results['confidence']
        result['prob_bull'] = hmm_results['prob_bull']
        result['prob_bear'] = hmm_results['prob_bear']
        result['prob_sideways'] = hmm_results['prob_sideways']

        result['hurst'] = hurst_results['hurst']
        result['hurst_regime'] = hurst_results['regime']
        result['hurst_confidence'] = hurst_results['confidence']

        # Generate strategy recommendations
        strategies = []
        position_scales = []

        for i in range(len(result)):
            row = result.iloc[i]
            strategy, scale = self._determine_strategy(
                market_regime=row['market_regime'],
                regime_prob=row['regime_prob'],
                hurst=row['hurst'],
                hurst_regime=row['hurst_regime'],
                volatility=self._get_volatility(data, i),
            )
            strategies.append(strategy.value if strategy else 'unknown')
            position_scales.append(scale)

        result['strategy_mode'] = strategies
        result['position_scale'] = position_scales

        return result

    def get_current_state(self, data: pd.DataFrame) -> HybridRegimeState:
        """
        Get current hybrid regime state.

        Args:
            data: DataFrame with recent OHLCV data

        Returns:
            HybridRegimeState with current regime and recommendations
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Get HMM state
        hmm_state = self.hmm_detector.get_current_regime(data)

        # Calculate current Hurst
        returns = data['close'].pct_change().dropna()
        if len(returns) >= self.config.hurst_window:
            hurst_result = self.hurst_calculator.calculate_dfa(
                returns.iloc[-self.config.hurst_window:].values
            )
        else:
            hurst_result = HurstResult(
                hurst=0.5,
                regime=HurstRegime.RANDOM_WALK,
                confidence=0.0,
                method='dfa'
            )

        # Determine strategy
        strategy, scale = self._determine_strategy(
            market_regime=hmm_state.regime.value,
            regime_prob=hmm_state.confidence,
            hurst=hurst_result.hurst,
            hurst_regime=hurst_result.regime.value,
            volatility=self._get_volatility(data, len(data) - 1),
        )

        return HybridRegimeState(
            market_regime=hmm_state.regime,
            regime_probability=hmm_state.confidence,
            regime_probabilities=hmm_state.probabilities,
            hurst_value=hurst_result.hurst,
            hurst_regime=hurst_result.regime,
            hurst_confidence=hurst_result.confidence,
            strategy_mode=strategy,
            position_scale=scale,
            timestamp=data.index[-1] if hasattr(data.index, '__getitem__') else None,
        )

    def _add_hurst_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add rolling Hurst exponent as feature for HMM."""
        df = data.copy()

        returns = df['close'].pct_change()
        hurst_results = self.hurst_calculator.calculate_rolling(
            returns.dropna(),
            window=self.config.hurst_window,
            method=self.config.hurst_method,
        )

        df['hurst'] = hurst_results['hurst'].reindex(df.index)
        return df

    def _get_volatility(self, data: pd.DataFrame, idx: int) -> float:
        """Get volatility at specific index."""
        if idx < self.config.hmm_vol_window:
            return self._vol_mean or 0.02

        returns = data['close'].pct_change()
        vol = returns.iloc[max(0, idx - self.config.hmm_vol_window):idx + 1].std()
        return vol if not np.isnan(vol) else (self._vol_mean or 0.02)

    def _determine_strategy(
        self,
        market_regime: str,
        regime_prob: float,
        hurst: float,
        hurst_regime: str,
        volatility: float,
    ) -> Tuple[StrategyMode, float]:
        """
        Determine trading strategy based on combined regime state.

        Strategy Matrix:
                        Trending (H>0.5)      Mean-Rev (H<0.5)     Random (H≈0.5)
        Bull            MOMENTUM_LONG         MEAN_REVERSION       CAUTIOUS
        Bear            MOMENTUM_SHORT        MEAN_REVERSION       DEFENSIVE
        Sideways        CAUTIOUS              MEAN_REVERSION       NEUTRAL

        Returns:
            Tuple of (StrategyMode, position_scale)
        """
        # Handle missing data
        if pd.isna(market_regime) or pd.isna(hurst):
            return StrategyMode.NEUTRAL, self.config.min_position_scale

        # Check confidence
        if regime_prob < self.config.min_confidence:
            return StrategyMode.CAUTIOUS, 0.5

        # Strategy matrix
        if market_regime == 'bull':
            if hurst_regime == 'trending':
                strategy = StrategyMode.MOMENTUM_LONG
            elif hurst_regime == 'mean_reverting':
                strategy = StrategyMode.MEAN_REVERSION
            else:
                strategy = StrategyMode.CAUTIOUS
        elif market_regime == 'bear':
            if hurst_regime == 'trending':
                strategy = StrategyMode.MOMENTUM_SHORT
            elif hurst_regime == 'mean_reverting':
                strategy = StrategyMode.MEAN_REVERSION
            else:
                strategy = StrategyMode.DEFENSIVE
        else:  # sideways
            if hurst_regime == 'mean_reverting':
                strategy = StrategyMode.MEAN_REVERSION
            else:
                strategy = StrategyMode.NEUTRAL

        # Calculate position scale
        position_scale = self._calculate_position_scale(
            strategy, regime_prob, hurst, volatility
        )

        return strategy, position_scale

    def _calculate_position_scale(
        self,
        strategy: StrategyMode,
        regime_prob: float,
        hurst: float,
        volatility: float,
    ) -> float:
        """
        Calculate position scaling factor.

        Higher confidence and clearer signals -> larger positions
        Higher volatility -> smaller positions (if vol_scaling enabled)
        """
        # Base scale from strategy type
        strategy_scales = {
            StrategyMode.MOMENTUM_LONG: 1.0,
            StrategyMode.MOMENTUM_SHORT: 0.8,
            StrategyMode.MEAN_REVERSION: 0.7,
            StrategyMode.CAUTIOUS: 0.4,
            StrategyMode.DEFENSIVE: 0.3,
            StrategyMode.NEUTRAL: 0.2,
        }
        base_scale = strategy_scales.get(strategy, 0.5)

        # Adjust by regime probability
        confidence_factor = min(regime_prob / self.config.min_confidence, 1.0)
        scale = base_scale * confidence_factor

        # Adjust by Hurst clarity (how far from 0.5)
        hurst_clarity = abs(hurst - 0.5) * 2  # 0 to 1
        scale *= (0.7 + 0.3 * hurst_clarity)

        # Volatility scaling
        if self.config.vol_scaling and self._vol_mean and self._vol_std:
            vol_zscore = (volatility - self._vol_mean) / (self._vol_std + 1e-10)
            # High volatility -> reduce position
            vol_factor = np.exp(-0.3 * max(0, vol_zscore))
            scale *= vol_factor

        # Clamp to configured range
        return np.clip(
            scale,
            self.config.min_position_scale,
            self.config.max_position_scale
        )


class RegimeStrategySwitch:
    """
    Strategy switcher based on regime detection.

    Manages switching between different trading strategies
    (momentum, mean-reversion, etc.) based on detected regime.
    """

    def __init__(
        self,
        regime_detector: HybridRegimeDetector,
        switch_cooldown: int = 5,  # Min bars between switches
    ):
        self.regime_detector = regime_detector
        self.switch_cooldown = switch_cooldown

        self._current_strategy: Optional[StrategyMode] = None
        self._bars_since_switch: int = 0
        self._switch_history: List[Tuple[pd.Timestamp, StrategyMode]] = []

    def update(
        self,
        data: pd.DataFrame
    ) -> Tuple[StrategyMode, float, bool]:
        """
        Update regime state and potentially switch strategy.

        Args:
            data: Recent OHLCV data

        Returns:
            Tuple of (current_strategy, position_scale, did_switch)
        """
        # Get current regime state
        state = self.regime_detector.get_current_state(data)

        did_switch = False
        self._bars_since_switch += 1

        # Check if we should switch
        if self._should_switch(state):
            self._current_strategy = state.strategy_mode
            self._bars_since_switch = 0
            did_switch = True

            if state.timestamp:
                self._switch_history.append((state.timestamp, state.strategy_mode))

        return (
            self._current_strategy or state.strategy_mode,
            state.position_scale,
            did_switch
        )

    def _should_switch(self, state: HybridRegimeState) -> bool:
        """Determine if strategy should switch."""
        # First time
        if self._current_strategy is None:
            return True

        # Cooldown check
        if self._bars_since_switch < self.switch_cooldown:
            return False

        # Check if new strategy is different and confident
        if state.strategy_mode != self._current_strategy:
            if state.regime_probability >= self.regime_detector.config.min_confidence:
                return True

        return False

    def get_switch_history(self) -> pd.DataFrame:
        """Get history of strategy switches."""
        return pd.DataFrame(
            self._switch_history,
            columns=['timestamp', 'strategy']
        )


def create_default_hybrid_detector() -> HybridRegimeDetector:
    """Create a hybrid detector with sensible defaults."""
    config = HybridConfig(
        hmm_n_states=3,
        hmm_vol_window=20,
        hurst_window=100,
        hurst_method='dfa',
        trending_threshold=0.55,
        mean_revert_threshold=0.45,
        min_confidence=0.6,
        vol_scaling=True,
    )
    return HybridRegimeDetector(config)
