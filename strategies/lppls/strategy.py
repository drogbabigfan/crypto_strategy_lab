"""
LPPLS Trading Strategy Implementation.

Trading Rules (from research document):
1. Entry (Short): Confidence >= 0.8 with tc converging (std < 5 days)
2. Exit: Target profit (20% drop) or Confidence < 0.1 or tc passed without crash

Supports both:
- Bubble Crash Betting (Short): When positive bubble detected
- Bottom Fishing (Long): When negative bubble detected (B > 0)
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from numpy.typing import NDArray

from .lppls import LPPLSModel, LPPLSResult
from .filter import FilterConfig
from .indicator import (
    LPPLSIndicator,
    WindowConfig,
    ConfidenceResult,
    get_bubble_signal,
)


@dataclass
class LPPLSStrategyConfig:
    """Configuration for LPPLS trading strategy."""
    # LPPLS calculation parameters
    min_window: int = 60       # Minimum window for LPPLS (reduced for faster calc)
    max_window: int = 180      # Maximum window for LPPLS
    window_step: int = 10      # Step size for multi-window
    recalc_interval: int = 5   # Recalculate every N bars

    # Entry thresholds
    confidence_entry: float = 0.6     # Confidence threshold for entry
    tc_std_entry: float = 10.0        # tc std threshold for convergence

    # Exit thresholds
    confidence_exit: float = 0.2      # Exit when confidence drops below
    profit_target_pct: float = 0.15   # Take profit at 15% gain
    stop_loss_pct: float = 0.10       # Stop loss at 10% loss

    # Position sizing
    max_position: float = 1.0         # Maximum position size (1.0 = 100%)
    position_scale: float = 1.0       # Scale position by confidence

    # Filter config
    filter_strict: bool = False       # Use strict parameter bounds
    positive_bubble: bool = True      # True for short (crash betting)


@dataclass
class Trade:
    """Single trade record."""
    entry_idx: int
    entry_price: float
    entry_confidence: float
    direction: int  # 1 for long, -1 for short
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_pct: Optional[float] = None


@dataclass
class BacktestResult:
    """Backtest result summary."""
    trades: List[Trade]
    total_pnl_pct: float
    win_rate: float
    n_trades: int
    sharpe_ratio: float
    max_drawdown: float
    confidence_series: NDArray[np.float64]
    equity_curve: NDArray[np.float64]


class LPPLSStrategy:
    """
    LPPLS-based trading strategy.

    Detects financial bubbles using LPPLS model and trades
    the expected crash/reversal.
    """

    def __init__(self, config: Optional[LPPLSStrategyConfig] = None):
        self.config = config or LPPLSStrategyConfig()
        self._setup_indicator()

    def _setup_indicator(self):
        """Setup LPPLS indicator with config."""
        window_config = WindowConfig(
            min_window=self.config.min_window,
            max_window=self.config.max_window,
            step=self.config.window_step,
        )
        filter_config = FilterConfig(
            use_strict_bounds=self.config.filter_strict,
            positive_bubble=self.config.positive_bubble,
            r_squared_min=0.4,  # Slightly relaxed
            damping_min=0.5,    # Relaxed for more signals
        )
        self.indicator = LPPLSIndicator(
            window_config=window_config,
            filter_config=filter_config,
        )

    def calculate_signals(
        self,
        prices: NDArray[np.float64],
        verbose: bool = False,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Calculate LPPLS signals for price series.

        Args:
            prices: Price array
            verbose: Print progress

        Returns:
            Tuple of (confidence, tc_mean, tc_std) arrays
        """
        n = len(prices)
        log_prices = np.log(prices)

        confidence = np.zeros(n)
        tc_mean = np.full(n, np.nan)
        tc_std = np.full(n, np.nan)

        min_data = self.config.min_window + 20

        for i in range(min_data, n, self.config.recalc_interval):
            if verbose and i % 100 == 0:
                print(f"  Processing bar {i}/{n}...")

            result = self.indicator.calculate(log_prices[:i+1])
            confidence[i] = result.confidence

            if result.tc_mean is not None:
                tc_mean[i] = result.tc_mean
                tc_std[i] = result.tc_std or 0.0

            # Forward fill for bars between recalculations
            for j in range(1, self.config.recalc_interval):
                if i + j < n:
                    confidence[i + j] = confidence[i]
                    tc_mean[i + j] = tc_mean[i]
                    tc_std[i + j] = tc_std[i]

        return confidence, tc_mean, tc_std

    def backtest(
        self,
        prices: NDArray[np.float64],
        verbose: bool = True,
    ) -> BacktestResult:
        """
        Run backtest on price data.

        Args:
            prices: Price array
            verbose: Print progress

        Returns:
            BacktestResult with trades and metrics
        """
        n = len(prices)

        if verbose:
            print("Calculating LPPLS signals...")

        confidence, tc_mean, tc_std = self.calculate_signals(prices, verbose)

        if verbose:
            print("Running backtest...")

        trades: List[Trade] = []
        position = 0  # 0: flat, -1: short, 1: long
        entry_price = 0.0
        entry_idx = 0
        entry_confidence = 0.0

        equity = np.ones(n)
        current_equity = 1.0

        for i in range(self.config.min_window + 20, n):
            price = prices[i]
            conf = confidence[i]
            t_std = tc_std[i] if not np.isnan(tc_std[i]) else float('inf')

            # Update equity
            if position != 0:
                pnl = position * (price / entry_price - 1)
                equity[i] = current_equity * (1 + pnl)
            else:
                equity[i] = current_equity

            # Check exit conditions
            if position != 0:
                pnl_pct = position * (price / entry_price - 1)

                exit_reason = None

                # Take profit
                if pnl_pct >= self.config.profit_target_pct:
                    exit_reason = 'profit_target'
                # Stop loss
                elif pnl_pct <= -self.config.stop_loss_pct:
                    exit_reason = 'stop_loss'
                # Confidence dropped
                elif conf < self.config.confidence_exit:
                    exit_reason = 'confidence_exit'

                if exit_reason:
                    trade = Trade(
                        entry_idx=entry_idx,
                        entry_price=entry_price,
                        entry_confidence=entry_confidence,
                        direction=position,
                        exit_idx=i,
                        exit_price=price,
                        exit_reason=exit_reason,
                        pnl_pct=pnl_pct,
                    )
                    trades.append(trade)
                    current_equity = equity[i]
                    position = 0

            # Check entry conditions (only if flat)
            if position == 0:
                signal = get_bubble_signal(conf, t_std,
                                          confidence_threshold=self.config.confidence_entry,
                                          tc_std_threshold=self.config.tc_std_entry)

                if signal in ['STRONG', 'MODERATE']:
                    # Short for positive bubble (crash betting)
                    if self.config.positive_bubble:
                        position = -1
                    else:
                        position = 1

                    entry_price = price
                    entry_idx = i
                    entry_confidence = conf

        # Close any open position at end
        if position != 0:
            pnl_pct = position * (prices[-1] / entry_price - 1)
            trade = Trade(
                entry_idx=entry_idx,
                entry_price=entry_price,
                entry_confidence=entry_confidence,
                direction=position,
                exit_idx=n - 1,
                exit_price=prices[-1],
                exit_reason='end_of_data',
                pnl_pct=pnl_pct,
            )
            trades.append(trade)

        # Calculate metrics
        if trades:
            pnls = [t.pnl_pct for t in trades if t.pnl_pct is not None]
            total_pnl = sum(pnls)
            wins = sum(1 for p in pnls if p > 0)
            win_rate = wins / len(pnls) if pnls else 0.0

            # Sharpe (simplified - daily returns)
            returns = np.diff(equity) / equity[:-1]
            returns = returns[~np.isnan(returns)]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)

            # Max drawdown
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            max_dd = np.max(drawdown)
        else:
            total_pnl = 0.0
            win_rate = 0.0
            sharpe = 0.0
            max_dd = 0.0

        return BacktestResult(
            trades=trades,
            total_pnl_pct=total_pnl,
            win_rate=win_rate,
            n_trades=len(trades),
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            confidence_series=confidence,
            equity_curve=equity,
        )

    def generate_signals_df(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
    ) -> pd.DataFrame:
        """
        Generate trading signals as DataFrame.

        Args:
            df: DataFrame with price data
            price_col: Name of price column

        Returns:
            DataFrame with added signal columns
        """
        prices = df[price_col].values
        confidence, tc_mean, tc_std = self.calculate_signals(prices)

        result = df.copy()
        result['lppls_confidence'] = confidence
        result['lppls_tc_mean'] = tc_mean
        result['lppls_tc_std'] = tc_std
        result['lppls_signal'] = [
            get_bubble_signal(c, s,
                            confidence_threshold=self.config.confidence_entry,
                            tc_std_threshold=self.config.tc_std_entry)
            for c, s in zip(confidence, tc_std)
        ]

        return result


def load_btc_data(data_dir: str = 'etl/data/bars-24/futures/BTCUSDT') -> pd.DataFrame:
    """Load BTC price data from parquet files."""
    import glob

    files = sorted(glob.glob(f'{data_dir}/*.parquet'))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    df['datetime'] = pd.to_datetime(df['start_time'], unit='ms')
    df = df.sort_values('datetime').reset_index(drop=True)

    return df


def run_lppls_backtest(
    config: Optional[LPPLSStrategyConfig] = None,
    data_dir: str = 'etl/data/bars-24/futures/BTCUSDT',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> BacktestResult:
    """
    Run LPPLS strategy backtest.

    Args:
        config: Strategy configuration
        data_dir: Path to data directory
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)

    Returns:
        BacktestResult
    """
    print("Loading data...")
    df = load_btc_data(data_dir)

    if start_date:
        df = df[df['datetime'] >= start_date]
    if end_date:
        df = df[df['datetime'] <= end_date]

    df = df.reset_index(drop=True)
    print(f"Data: {len(df)} bars from {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")

    prices = df['close'].values
    strategy = LPPLSStrategy(config)

    result = strategy.backtest(prices, verbose=True)

    print("\n" + "="*50)
    print("LPPLS Strategy Backtest Results")
    print("="*50)
    print(f"Total PnL: {result.total_pnl_pct*100:.2f}%")
    print(f"Number of Trades: {result.n_trades}")
    print(f"Win Rate: {result.win_rate*100:.1f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown*100:.1f}%")

    if result.trades:
        print(f"\nTrade Details:")
        for i, trade in enumerate(result.trades[:10], 1):
            direction = "SHORT" if trade.direction == -1 else "LONG"
            print(f"  {i}. {direction}: Entry ${trade.entry_price:.0f} -> "
                  f"Exit ${trade.exit_price:.0f} ({trade.exit_reason}) "
                  f"PnL: {trade.pnl_pct*100:+.1f}%")
        if len(result.trades) > 10:
            print(f"  ... and {len(result.trades) - 10} more trades")

    return result


if __name__ == '__main__':
    # Quick test
    config = LPPLSStrategyConfig(
        min_window=60,
        max_window=120,
        window_step=15,
        recalc_interval=10,
        confidence_entry=0.5,
    )
    result = run_lppls_backtest(
        config=config,
        start_date='2020-01-01',
        end_date='2022-12-31',
    )
