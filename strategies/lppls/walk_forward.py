"""
Walk-Forward Optimization for LPPLS Strategy.

Walk-forward analysis:
1. Split data into train/test windows
2. Optimize parameters on training window
3. Apply to test window
4. Roll forward and repeat

This avoids look-ahead bias and provides realistic performance estimates.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from itertools import product
import warnings

from .strategy import (
    LPPLSStrategy,
    LPPLSStrategyConfig,
    BacktestResult,
    Trade,
    load_btc_data,
)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward optimization."""
    # Window sizes (in bars)
    train_window: int = 2000      # Training window size
    test_window: int = 500        # Test window size
    step_size: int = 500          # Step size for rolling

    # Parameter grid for optimization
    param_grid: Dict[str, List[Any]] = field(default_factory=lambda: {
        'confidence_entry': [0.4, 0.5, 0.6],
        'confidence_exit': [0.15, 0.2, 0.25],
        'profit_target_pct': [0.15, 0.20, 0.25],
        'stop_loss_pct': [0.08, 0.12, 0.15],
    })

    # Fixed LPPLS params (for speed)
    min_window: int = 60
    max_window: int = 120
    window_step: int = 20
    recalc_interval: int = 30


@dataclass
class WalkForwardResult:
    """Result of walk-forward optimization."""
    trades: List[Trade]
    total_pnl_pct: float
    win_rate: float
    n_trades: int
    sharpe_ratio: float
    max_drawdown: float
    windows: List[Dict]  # Details per window
    equity_curve: np.ndarray


class WalkForwardOptimizer:
    """
    Walk-Forward Optimizer for LPPLS Strategy.

    Performs rolling optimization to avoid overfitting.
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig()

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from grid."""
        keys = list(self.config.param_grid.keys())
        values = list(self.config.param_grid.values())

        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def _create_strategy_config(self, params: Dict[str, Any]) -> LPPLSStrategyConfig:
        """Create strategy config from parameter dict."""
        return LPPLSStrategyConfig(
            min_window=self.config.min_window,
            max_window=self.config.max_window,
            window_step=self.config.window_step,
            recalc_interval=self.config.recalc_interval,
            **params,
        )

    def _evaluate_params(
        self,
        prices: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[float, BacktestResult]:
        """
        Evaluate a parameter set on given prices.

        Returns (score, result) where score is used for optimization.
        """
        config = self._create_strategy_config(params)
        strategy = LPPLSStrategy(config)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = strategy.backtest(prices, verbose=False)

        # Score: Sharpe ratio (or PnL if no trades)
        if result.n_trades > 0:
            # Penalize for too few trades
            trade_penalty = max(0, 1 - result.n_trades / 5) * 0.5
            score = result.sharpe_ratio - trade_penalty
        else:
            score = -10.0  # Heavily penalize no trades

        return score, result

    def _optimize_window(
        self,
        prices: np.ndarray,
        verbose: bool = False,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Find best parameters for a training window.

        Returns (best_params, best_score)
        """
        param_combos = self._generate_param_combinations()

        best_params = None
        best_score = -np.inf

        for params in param_combos:
            score, _ = self._evaluate_params(prices, params)
            if score > best_score:
                best_score = score
                best_params = params

        if verbose:
            print(f"    Best params: {best_params} (score: {best_score:.3f})")

        return best_params, best_score

    def run(
        self,
        prices: np.ndarray,
        verbose: bool = True,
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization.

        Args:
            prices: Full price array
            verbose: Print progress

        Returns:
            WalkForwardResult with aggregated results
        """
        n = len(prices)
        train_size = self.config.train_window
        test_size = self.config.test_window
        step = self.config.step_size

        all_trades = []
        windows = []
        equity_pieces = []

        # Starting point after initial training window
        start_idx = train_size

        window_num = 0
        while start_idx + test_size <= n:
            window_num += 1
            train_start = start_idx - train_size
            train_end = start_idx
            test_start = start_idx
            test_end = min(start_idx + test_size, n)

            if verbose:
                print(f"\nWindow {window_num}: Train[{train_start}:{train_end}] "
                      f"Test[{test_start}:{test_end}]")

            # Optimize on training window
            train_prices = prices[train_start:train_end]
            best_params, train_score = self._optimize_window(train_prices, verbose)

            if best_params is None:
                start_idx += step
                continue

            # Apply to test window
            test_prices = prices[test_start:test_end]
            config = self._create_strategy_config(best_params)
            strategy = LPPLSStrategy(config)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test_result = strategy.backtest(test_prices, verbose=False)

            # Adjust trade indices to global
            for trade in test_result.trades:
                trade.entry_idx += test_start
                if trade.exit_idx is not None:
                    trade.exit_idx += test_start
                all_trades.append(trade)

            equity_pieces.append((test_start, test_result.equity_curve))

            window_info = {
                'window_num': window_num,
                'train_range': (train_start, train_end),
                'test_range': (test_start, test_end),
                'best_params': best_params,
                'train_score': train_score,
                'test_pnl': test_result.total_pnl_pct,
                'test_trades': test_result.n_trades,
            }
            windows.append(window_info)

            if verbose:
                print(f"    Test PnL: {test_result.total_pnl_pct*100:+.2f}% "
                      f"({test_result.n_trades} trades)")

            start_idx += step

        # Build full equity curve
        equity_curve = self._build_equity_curve(n, equity_pieces)

        # Calculate aggregated metrics
        if all_trades:
            pnls = [t.pnl_pct for t in all_trades if t.pnl_pct is not None]
            total_pnl = sum(pnls)
            wins = sum(1 for p in pnls if p > 0)
            win_rate = wins / len(pnls) if pnls else 0.0

            returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-10)
            returns = returns[~np.isnan(returns) & (returns != 0)]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252) if len(returns) > 0 else 0.0

            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / (peak + 1e-10)
            max_dd = np.max(drawdown)
        else:
            total_pnl = 0.0
            win_rate = 0.0
            sharpe = 0.0
            max_dd = 0.0

        return WalkForwardResult(
            trades=all_trades,
            total_pnl_pct=total_pnl,
            win_rate=win_rate,
            n_trades=len(all_trades),
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            windows=windows,
            equity_curve=equity_curve,
        )

    def _build_equity_curve(
        self,
        n: int,
        equity_pieces: List[Tuple[int, np.ndarray]],
    ) -> np.ndarray:
        """Build full equity curve from pieces."""
        equity = np.ones(n)

        for start_idx, piece in equity_pieces:
            end_idx = start_idx + len(piece)
            if start_idx > 0:
                # Scale piece to match previous equity level
                scale = equity[start_idx - 1]
                equity[start_idx:end_idx] = piece * scale
            else:
                equity[start_idx:end_idx] = piece

        return equity


def run_walk_forward_backtest(
    data_dir: str = 'etl/data/bars-24/futures/BTCUSDT',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    config: Optional[WalkForwardConfig] = None,
    verbose: bool = True,
) -> WalkForwardResult:
    """
    Run walk-forward optimized backtest.

    Args:
        data_dir: Path to data directory
        start_date: Start date filter
        end_date: End date filter
        config: Walk-forward configuration
        verbose: Print progress

    Returns:
        WalkForwardResult
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

    optimizer = WalkForwardOptimizer(config)
    result = optimizer.run(prices, verbose=verbose)

    print("\n" + "="*60)
    print("Walk-Forward Optimization Results")
    print("="*60)
    print(f"Total PnL: {result.total_pnl_pct*100:.2f}%")
    print(f"Number of Trades: {result.n_trades}")
    print(f"Win Rate: {result.win_rate*100:.1f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown*100:.1f}%")
    print(f"Number of Windows: {len(result.windows)}")

    if result.windows:
        print("\nPer-Window Results:")
        for w in result.windows:
            print(f"  Window {w['window_num']}: "
                  f"Test[{w['test_range'][0]}:{w['test_range'][1]}] "
                  f"PnL: {w['test_pnl']*100:+.1f}% "
                  f"({w['test_trades']} trades)")

    if result.trades:
        print(f"\nSample Trades (first 10):")
        for i, trade in enumerate(result.trades[:10], 1):
            direction = "SHORT" if trade.direction == -1 else "LONG"
            print(f"  {i}. {direction}: ${trade.entry_price:.0f} -> "
                  f"${trade.exit_price:.0f} ({trade.exit_reason}) "
                  f"PnL: {trade.pnl_pct*100:+.1f}%")

    return result


if __name__ == '__main__':
    config = WalkForwardConfig(
        train_window=1500,
        test_window=500,
        step_size=500,
        param_grid={
            'confidence_entry': [0.4, 0.5, 0.6],
            'confidence_exit': [0.15, 0.2],
            'profit_target_pct': [0.15, 0.20],
            'stop_loss_pct': [0.10, 0.15],
        },
    )
    result = run_walk_forward_backtest(
        start_date='2020-01-01',
        end_date='2023-12-31',
        config=config,
    )
