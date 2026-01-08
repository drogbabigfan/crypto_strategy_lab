"""
Regime-Based Trading Strategy.

Uses HMM and Hurst exponent to switch between:
- Momentum strategy in trending regimes
- Mean reversion strategy in mean-reverting regimes
- Reduced exposure in uncertain/bear regimes
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from enum import Enum

from .hurst import HurstCalculator, HurstRegime
from .hmm import MarketRegime


class PositionType(Enum):
    """Position types."""
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class RegimeStrategyConfig:
    """Configuration for regime-based strategy."""
    # Hurst parameters
    hurst_window: int = 100
    hurst_trending_threshold: float = 0.55
    hurst_mean_revert_threshold: float = 0.45

    # Momentum parameters (for trending regime)
    momentum_lookback: int = 20
    momentum_entry_zscore: float = 1.5
    momentum_exit_zscore: float = 0.5

    # Mean reversion parameters
    mean_revert_entry_zscore: float = 2.0
    mean_revert_exit_zscore: float = 0.5
    mean_revert_lookback: int = 20

    # Volatility filter
    vol_window: int = 20
    max_vol_percentile: float = 90  # Skip trades in extreme volatility

    # Risk management
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0
    atr_period: int = 14

    # Position sizing
    base_position_size: float = 1.0
    trending_size_mult: float = 1.2
    mean_revert_size_mult: float = 0.8
    uncertain_size_mult: float = 0.3


class RegimeStrategy:
    """
    Regime-adaptive trading strategy.

    Uses Hurst exponent to determine market structure and
    switches between momentum and mean reversion approaches.
    """

    def __init__(self, config: Optional[RegimeStrategyConfig] = None):
        self.config = config or RegimeStrategyConfig()
        self.hurst_calculator = HurstCalculator(
            trending_threshold=self.config.hurst_trending_threshold,
            mean_revert_threshold=self.config.hurst_mean_revert_threshold,
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on regime detection.

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with signals and regime information
        """
        result = df.copy()

        # Calculate base features
        result = self._calculate_features(result)

        # Calculate Hurst exponent (rolling)
        returns = result['close'].pct_change()
        hurst_df = self.hurst_calculator.calculate_rolling(
            returns.dropna(),
            window=self.config.hurst_window,
            method='dfa'
        )
        result['hurst'] = hurst_df['hurst'].reindex(result.index)
        result['hurst_regime'] = hurst_df['regime'].reindex(result.index)

        # Generate regime-specific signals
        result['momentum_signal'] = self._momentum_signals(result)
        result['mean_revert_signal'] = self._mean_revert_signals(result)

        # Combine signals based on regime
        result['signal'] = self._combine_signals(result)
        result['position_size'] = self._calculate_position_size(result)

        # Calculate stop loss and take profit
        result['stop_loss'] = self._calculate_stop_loss(result)
        result['take_profit'] = self._calculate_take_profit(result)

        return result

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical features."""
        result = df.copy()

        # Returns
        result['returns'] = result['close'].pct_change()

        # Moving averages
        result['sma_20'] = result['close'].rolling(20).mean()
        result['sma_50'] = result['close'].rolling(50).mean()

        # Z-score (deviation from mean)
        rolling_mean = result['close'].rolling(self.config.mean_revert_lookback).mean()
        rolling_std = result['close'].rolling(self.config.mean_revert_lookback).std()
        result['zscore'] = (result['close'] - rolling_mean) / (rolling_std + 1e-10)

        # Volatility (ATR)
        high = result['high']
        low = result['low']
        close = result['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        result['atr'] = tr.rolling(self.config.atr_period).mean()

        # Realized volatility
        result['realized_vol'] = result['returns'].rolling(self.config.vol_window).std()

        # Momentum (ROC)
        result['momentum'] = result['close'].pct_change(self.config.momentum_lookback)

        # RSI
        result['rsi'] = self._calculate_rsi(result['close'], 14)

        return result

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _momentum_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate momentum signals for trending markets.

        Long when: momentum > threshold AND price above SMA
        Short when: momentum < -threshold AND price below SMA
        """
        momentum_zscore = (
            df['momentum'] - df['momentum'].rolling(50).mean()
        ) / (df['momentum'].rolling(50).std() + 1e-10)

        long_signal = (
            (momentum_zscore > self.config.momentum_entry_zscore) &
            (df['close'] > df['sma_20']) &
            (df['rsi'] < 80)
        ).astype(int)

        short_signal = (
            (momentum_zscore < -self.config.momentum_entry_zscore) &
            (df['close'] < df['sma_20']) &
            (df['rsi'] > 20)
        ).astype(int) * -1

        return long_signal + short_signal

    def _mean_revert_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate mean reversion signals for ranging markets.

        Long when: zscore < -threshold (oversold)
        Short when: zscore > threshold (overbought)
        """
        long_signal = (
            (df['zscore'] < -self.config.mean_revert_entry_zscore) &
            (df['rsi'] < 35)
        ).astype(int)

        short_signal = (
            (df['zscore'] > self.config.mean_revert_entry_zscore) &
            (df['rsi'] > 65)
        ).astype(int) * -1

        return long_signal + short_signal

    def _combine_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Combine signals based on detected regime.

        - Trending (H > 0.55): Use momentum signals
        - Mean-reverting (H < 0.45): Use mean reversion signals
        - Random walk (0.45 <= H <= 0.55): No trading or reduced signals
        """
        signals = pd.Series(0, index=df.index)

        # Trending regime - use momentum
        trending_mask = df['hurst_regime'] == 'trending'
        signals.loc[trending_mask] = df.loc[trending_mask, 'momentum_signal']

        # Mean-reverting regime - use mean reversion
        mean_revert_mask = df['hurst_regime'] == 'mean_reverting'
        signals.loc[mean_revert_mask] = df.loc[mean_revert_mask, 'mean_revert_signal']

        # Random walk - no signals (stay flat)
        # Already 0 by default

        # Volatility filter - skip extreme volatility
        vol_threshold = df['realized_vol'].quantile(self.config.max_vol_percentile / 100)
        extreme_vol_mask = df['realized_vol'] > vol_threshold
        signals.loc[extreme_vol_mask] = 0

        return signals

    def _calculate_position_size(self, df: pd.DataFrame) -> pd.Series:
        """Calculate position size based on regime."""
        size = pd.Series(self.config.base_position_size, index=df.index)

        # Adjust by regime
        trending_mask = df['hurst_regime'] == 'trending'
        mean_revert_mask = df['hurst_regime'] == 'mean_reverting'
        random_walk_mask = df['hurst_regime'] == 'random_walk'

        size.loc[trending_mask] *= self.config.trending_size_mult
        size.loc[mean_revert_mask] *= self.config.mean_revert_size_mult
        size.loc[random_walk_mask] *= self.config.uncertain_size_mult

        # Scale by inverse volatility (vol targeting)
        vol_norm = df['realized_vol'] / df['realized_vol'].rolling(100).mean()
        size = size / (vol_norm.clip(lower=0.5, upper=2.0))

        return size.fillna(self.config.base_position_size)

    def _calculate_stop_loss(self, df: pd.DataFrame) -> pd.Series:
        """Calculate stop loss level."""
        return df['atr'] * self.config.stop_loss_atr_mult

    def _calculate_take_profit(self, df: pd.DataFrame) -> pd.Series:
        """Calculate take profit level."""
        return df['atr'] * self.config.take_profit_atr_mult


class RegimeBacktester:
    """
    Simple backtester for regime strategy.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run backtest on signals.

        Args:
            df: DataFrame with 'close', 'signal', 'position_size', 'stop_loss', 'take_profit'

        Returns:
            DataFrame with backtest results
        """
        result = df.copy()

        # Initialize columns
        result['position'] = 0.0
        result['entry_price'] = np.nan
        result['pnl'] = 0.0
        result['cumulative_pnl'] = 0.0
        result['equity'] = float(self.initial_capital)
        result['trade_return'] = 0.0

        # Ensure float dtype for equity column
        result['equity'] = result['equity'].astype(float)

        position = 0.0
        entry_price = 0.0
        equity = self.initial_capital
        trades = []

        for i in range(1, len(result)):
            idx = result.index[i]
            prev_idx = result.index[i - 1]

            signal = result.loc[prev_idx, 'signal']
            close = result.loc[idx, 'close']
            prev_close = result.loc[prev_idx, 'close']

            # Calculate PnL from existing position
            if position != 0:
                price_change = close - prev_close
                pnl = position * price_change * abs(position)

                # Check stop loss / take profit
                stop_loss = result.loc[prev_idx, 'stop_loss']
                take_profit = result.loc[prev_idx, 'take_profit']

                if position > 0:
                    if close <= entry_price - stop_loss:
                        # Stop loss hit
                        pnl = -stop_loss * position
                        position = 0.0
                    elif close >= entry_price + take_profit:
                        # Take profit hit
                        pnl = take_profit * position
                        position = 0.0
                else:
                    if close >= entry_price + stop_loss:
                        # Stop loss hit (short)
                        pnl = -stop_loss * abs(position)
                        position = 0.0
                    elif close <= entry_price - take_profit:
                        # Take profit hit (short)
                        pnl = take_profit * abs(position)
                        position = 0.0

                # Deduct commission
                pnl -= abs(pnl) * self.commission_rate
                equity += pnl
                result.loc[idx, 'pnl'] = pnl
            else:
                result.loc[idx, 'pnl'] = 0.0

            # Process new signals
            if signal != 0 and position == 0:
                # Enter new position
                size = result.loc[prev_idx, 'position_size']
                position = signal * size
                entry_price = close * (1 + self.slippage_rate * np.sign(signal))

                # Commission
                commission = abs(position) * close * self.commission_rate
                equity -= commission

                trades.append({
                    'entry_time': idx,
                    'entry_price': entry_price,
                    'direction': 'long' if signal > 0 else 'short',
                    'size': abs(position),
                })

            elif signal == 0 and position != 0:
                # Exit position (signal flip)
                exit_price = close * (1 - self.slippage_rate * np.sign(position))
                trade_pnl = position * (exit_price - entry_price)
                trade_pnl -= abs(trade_pnl) * self.commission_rate

                equity += trade_pnl
                result.loc[idx, 'pnl'] += trade_pnl

                if trades:
                    trades[-1]['exit_time'] = idx
                    trades[-1]['exit_price'] = exit_price
                    trades[-1]['pnl'] = trade_pnl

                position = 0.0
                entry_price = 0.0

            result.loc[idx, 'position'] = position
            result.loc[idx, 'entry_price'] = entry_price if position != 0 else np.nan
            result.loc[idx, 'equity'] = equity

        result['cumulative_pnl'] = result['pnl'].cumsum()
        result['returns'] = result['equity'].pct_change()

        # Calculate metrics
        self.trades = pd.DataFrame(trades) if trades else pd.DataFrame()
        self.metrics = self._calculate_metrics(result)

        return result

    def _calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate performance metrics."""
        returns = df['returns'].dropna()

        total_return = (df['equity'].iloc[-1] / self.initial_capital - 1) * 100

        # Sharpe ratio (annualized, assuming daily data)
        if len(returns) > 0 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(365)
        else:
            sharpe = 0.0

        # Maximum drawdown
        equity = df['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Win rate
        if len(self.trades) > 0 and 'pnl' in self.trades.columns:
            wins = (self.trades['pnl'] > 0).sum()
            total_trades = len(self.trades)
            win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        else:
            win_rate = 0
            total_trades = 0

        # Profit factor
        if len(self.trades) > 0 and 'pnl' in self.trades.columns:
            gross_profit = self.trades.loc[self.trades['pnl'] > 0, 'pnl'].sum()
            gross_loss = abs(self.trades.loc[self.trades['pnl'] < 0, 'pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        else:
            profit_factor = 0

        return {
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown,
            'win_rate_pct': win_rate,
            'total_trades': total_trades,
            'profit_factor': profit_factor,
            'final_equity': df['equity'].iloc[-1],
        }

    def print_summary(self):
        """Print backtest summary."""
        print("\n" + "=" * 50)
        print("REGIME STRATEGY BACKTEST RESULTS")
        print("=" * 50)
        print(f"Total Return:     {self.metrics['total_return_pct']:.2f}%")
        print(f"Sharpe Ratio:     {self.metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:     {self.metrics['max_drawdown_pct']:.2f}%")
        print(f"Win Rate:         {self.metrics['win_rate_pct']:.1f}%")
        print(f"Total Trades:     {self.metrics['total_trades']}")
        print(f"Profit Factor:    {self.metrics['profit_factor']:.2f}")
        print(f"Final Equity:     ${self.metrics['final_equity']:,.2f}")
        print("=" * 50)
