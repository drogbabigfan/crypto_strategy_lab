"""
Parameter optimization for Fine-tuned LPPLS Strategy.

Grid search over strategy parameters to maximize risk-adjusted returns.
"""
import numpy as np
import pandas as pd
from itertools import product
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings

from .deep_lppls import DeepLPPLSPredictor, DeepLPPLSConfig
from .fast_strategy import FastLPPLSStrategy, FastLPPLSConfig
from .strategy import load_btc_data, BacktestResult


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    best_params: Dict
    best_score: float
    best_result: BacktestResult
    all_results: List[Tuple[Dict, float, BacktestResult]]


def load_finetuned_predictor(
    model_path: str = 'strategies/lppls/finetuned_model.pt'
) -> DeepLPPLSPredictor:
    """Load fine-tuned model."""
    config = DeepLPPLSConfig(input_length=252)
    predictor = DeepLPPLSPredictor(
        model_path=model_path,
        config=config,
        use_conv=True,
    )
    return predictor


def calculate_score(result: BacktestResult) -> float:
    """
    Calculate optimization score.

    Prioritizes:
    1. Positive PnL
    2. Good Sharpe ratio
    3. Reasonable number of trades
    4. Low drawdown
    """
    if result.n_trades < 3:
        return -100  # Penalize too few trades

    # Base score from PnL
    score = result.total_pnl_pct * 100

    # Sharpe bonus/penalty
    score += result.sharpe_ratio * 10

    # Drawdown penalty
    score -= result.max_drawdown * 50

    # Win rate bonus
    if result.win_rate > 0.5:
        score += (result.win_rate - 0.5) * 20

    return score


def run_single_backtest(
    predictor: DeepLPPLSPredictor,
    prices: np.ndarray,
    params: Dict,
) -> BacktestResult:
    """Run backtest with specific parameters."""
    config = FastLPPLSConfig(
        input_length=252,
        train_samples=0,
        train_epochs=0,
        **params
    )

    strategy = FastLPPLSStrategy(config)
    strategy.predictor = predictor
    strategy._is_ready = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = strategy.backtest(prices, verbose=False)

    return result


def grid_search(
    predictor: DeepLPPLSPredictor,
    prices: np.ndarray,
    param_grid: Dict[str, List],
    verbose: bool = True,
) -> OptimizationResult:
    """
    Grid search over parameter combinations.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    combinations = list(product(*values))
    n_combos = len(combinations)

    if verbose:
        print(f"Testing {n_combos} parameter combinations...")

    all_results = []
    best_score = -float('inf')
    best_params = None
    best_result = None

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        try:
            result = run_single_backtest(predictor, prices, params)
            score = calculate_score(result)

            all_results.append((params, score, result))

            if score > best_score:
                best_score = score
                best_params = params
                best_result = result

                if verbose:
                    print(f"  [{i+1}/{n_combos}] New best: score={score:.2f}, "
                          f"PnL={result.total_pnl_pct*100:.1f}%, "
                          f"trades={result.n_trades}, "
                          f"win={result.win_rate*100:.0f}%")
        except Exception as e:
            if verbose:
                print(f"  [{i+1}/{n_combos}] Error: {e}")
            continue

        if verbose and (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{n_combos}")

    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        best_result=best_result,
        all_results=all_results,
    )


def run_optimization(
    model_path: str = 'strategies/lppls/finetuned_model.pt',
    start_date: str = '2020-01-01',
    end_date: str = '2023-12-31',
):
    """
    Run full parameter optimization.
    """
    print("="*60)
    print("LPPLS Strategy Parameter Optimization")
    print("="*60)

    # Load data
    print("\nLoading data...")
    df = load_btc_data()
    df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
    df = df.reset_index(drop=True)
    prices = df['close'].values
    print(f"Data: {len(df)} bars from {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")

    # Load model
    print("\nLoading fine-tuned model...")
    predictor = load_finetuned_predictor(model_path)

    # Define parameter grid
    param_grid = {
        # Signal thresholds - focus on stricter tc
        'tc_max_days': [60, 90, 120, 150],
        'tc_min_days': [5, 10, 20],

        # m and omega ranges
        'm_min': [0.2, 0.3],
        'm_max': [0.7, 0.8],
        'omega_min': [5.0, 6.0],
        'omega_max': [15.0, 20.0],

        # Entry thresholds
        'signal_lookback': [3, 5, 7],
        'entry_threshold': [0.5, 0.6, 0.7],

        # Exit parameters
        'profit_target_pct': [0.10, 0.15, 0.20],
        'stop_loss_pct': [0.08, 0.10, 0.12],
        'max_hold_days': [30, 45, 60],
    }

    # Calculate total combinations
    total = 1
    for v in param_grid.values():
        total *= len(v)
    print(f"\nTotal combinations: {total}")

    # First pass: coarse grid on key parameters
    print("\n--- Phase 1: Coarse Grid Search ---")
    coarse_grid = {
        'tc_max_days': [60, 120],
        'tc_min_days': [10],
        'm_min': [0.25],
        'm_max': [0.75],
        'omega_min': [5.0],
        'omega_max': [18.0],
        'signal_lookback': [3, 5],
        'entry_threshold': [0.5, 0.7],
        'profit_target_pct': [0.12, 0.18],
        'stop_loss_pct': [0.08, 0.12],
        'max_hold_days': [30, 60],
    }

    coarse_result = grid_search(predictor, prices, coarse_grid, verbose=True)

    if coarse_result.best_params is None:
        print("No valid results in coarse search!")
        return None

    print(f"\nCoarse search best: score={coarse_result.best_score:.2f}")
    print(f"Best params: {coarse_result.best_params}")

    # Second pass: fine grid around best params
    print("\n--- Phase 2: Fine Grid Search ---")
    best = coarse_result.best_params

    fine_grid = {
        'tc_max_days': [max(30, best['tc_max_days'] - 30), best['tc_max_days'], best['tc_max_days'] + 30],
        'tc_min_days': [best['tc_min_days']],
        'm_min': [best['m_min']],
        'm_max': [best['m_max']],
        'omega_min': [best['omega_min']],
        'omega_max': [best['omega_max']],
        'signal_lookback': [max(2, best['signal_lookback'] - 1), best['signal_lookback'], best['signal_lookback'] + 1],
        'entry_threshold': [max(0.3, best['entry_threshold'] - 0.1), best['entry_threshold'], min(0.9, best['entry_threshold'] + 0.1)],
        'profit_target_pct': [best['profit_target_pct'] - 0.02, best['profit_target_pct'], best['profit_target_pct'] + 0.02],
        'stop_loss_pct': [best['stop_loss_pct'] - 0.02, best['stop_loss_pct'], best['stop_loss_pct'] + 0.02],
        'max_hold_days': [best['max_hold_days']],
    }

    fine_result = grid_search(predictor, prices, fine_grid, verbose=True)

    # Final result
    final_result = fine_result if fine_result.best_score > coarse_result.best_score else coarse_result

    print("\n" + "="*60)
    print("Optimization Complete")
    print("="*60)
    print(f"\nBest Parameters:")
    for k, v in final_result.best_params.items():
        print(f"  {k}: {v}")

    print(f"\nBest Results:")
    print(f"  Total PnL: {final_result.best_result.total_pnl_pct*100:.2f}%")
    print(f"  Number of Trades: {final_result.best_result.n_trades}")
    print(f"  Win Rate: {final_result.best_result.win_rate*100:.1f}%")
    print(f"  Sharpe Ratio: {final_result.best_result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {final_result.best_result.max_drawdown*100:.1f}%")

    # Show trade details
    if final_result.best_result.trades:
        print(f"\nTrade Details:")
        for i, trade in enumerate(final_result.best_result.trades[:15], 1):
            idx = trade.entry_idx
            date = df['datetime'].iloc[idx].strftime('%Y-%m-%d') if idx < len(df) else 'N/A'
            direction = 'SHORT' if trade.direction == -1 else 'LONG'
            print(f"  {i}. [{date}] {direction}: ${trade.entry_price:.0f} -> "
                  f"${trade.exit_price:.0f} ({trade.exit_reason}) "
                  f"PnL: {trade.pnl_pct*100:+.1f}%")

    # Top 5 parameter sets
    print("\n--- Top 5 Parameter Sets ---")
    sorted_results = sorted(final_result.all_results, key=lambda x: x[1], reverse=True)
    for i, (params, score, result) in enumerate(sorted_results[:5], 1):
        print(f"\n{i}. Score={score:.2f}, PnL={result.total_pnl_pct*100:.1f}%, "
              f"Trades={result.n_trades}, Win={result.win_rate*100:.0f}%")
        print(f"   tc_max={params['tc_max_days']}, entry_thresh={params['entry_threshold']}, "
              f"profit={params['profit_target_pct']}, stop={params['stop_loss_pct']}")

    return final_result


def validate_on_oos(
    best_params: Dict,
    model_path: str = 'strategies/lppls/finetuned_model.pt',
    oos_start: str = '2024-01-01',
    oos_end: str = '2024-12-31',
):
    """
    Validate best parameters on out-of-sample data.
    """
    print("\n" + "="*60)
    print("Out-of-Sample Validation")
    print("="*60)

    df = load_btc_data()
    df = df[(df['datetime'] >= oos_start) & (df['datetime'] <= oos_end)]
    df = df.reset_index(drop=True)

    if len(df) < 500:
        print(f"Not enough OOS data: {len(df)} bars")
        return None

    prices = df['close'].values
    print(f"OOS Data: {len(df)} bars from {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")

    predictor = load_finetuned_predictor(model_path)
    result = run_single_backtest(predictor, prices, best_params)

    print(f"\nOOS Results:")
    print(f"  Total PnL: {result.total_pnl_pct*100:.2f}%")
    print(f"  Number of Trades: {result.n_trades}")
    print(f"  Win Rate: {result.win_rate*100:.1f}%")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown*100:.1f}%")

    if result.trades:
        print(f"\nOOS Trade Details:")
        for i, trade in enumerate(result.trades, 1):
            idx = trade.entry_idx
            date = df['datetime'].iloc[idx].strftime('%Y-%m-%d') if idx < len(df) else 'N/A'
            direction = 'SHORT' if trade.direction == -1 else 'LONG'
            print(f"  {i}. [{date}] {direction}: ${trade.entry_price:.0f} -> "
                  f"${trade.exit_price:.0f} ({trade.exit_reason}) "
                  f"PnL: {trade.pnl_pct*100:+.1f}%")

    return result


if __name__ == '__main__':
    # Run optimization
    opt_result = run_optimization(
        start_date='2020-01-01',
        end_date='2023-12-31',
    )

    if opt_result and opt_result.best_params:
        # Validate on 2024 data
        validate_on_oos(
            opt_result.best_params,
            oos_start='2024-01-01',
            oos_end='2024-12-31',
        )
