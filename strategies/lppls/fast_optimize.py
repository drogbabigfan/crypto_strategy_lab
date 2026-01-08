"""
Fast parameter optimization by caching signal calculations.

Key insight: Signal calculation is slow, but trading logic is fast.
Calculate signals once, then test many parameter combinations.
"""
import numpy as np
import pandas as pd
from itertools import product
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from .deep_lppls import DeepLPPLSPredictor, DeepLPPLSConfig
from .fast_strategy import FastLPPLSStrategy, FastLPPLSConfig, FastSignal
from .strategy import load_btc_data, BacktestResult, Trade


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


def calculate_all_signals(
    predictor: DeepLPPLSPredictor,
    prices: np.ndarray,
    input_length: int = 252,
    verbose: bool = True,
) -> List[Dict]:
    """
    Calculate all signals once and cache them.

    Returns list of raw predictions for each bar.
    """
    n = len(prices)
    log_prices = np.log(prices)

    signals = []

    if verbose:
        print(f"Calculating signals for {n - input_length} bars...", flush=True)

    for i in range(input_length, n):
        window = log_prices[i-input_length:i]
        window_norm = (window - window.min()) / (window.max() - window.min() + 1e-8)

        pred = predictor.predict(window_norm)

        signals.append({
            'idx': i,
            'tc_norm': pred['tc_norm'],
            'tc_days': pred['tc_norm'] * input_length,
            'm': pred['m'],
            'omega': pred['omega'],
        })

        if verbose and (i - input_length) % 5000 == 0:
            print(f"  Processed {i - input_length}/{n - input_length}...", flush=True)

    if verbose:
        print(f"  Done. {len(signals)} signals calculated.", flush=True)

    return signals


def filter_signals(
    signals: List[Dict],
    params: Dict,
) -> np.ndarray:
    """
    Apply parameter filters to cached signals.

    Returns boolean array of valid signals.
    """
    n = len(signals)
    valid = np.zeros(n, dtype=bool)

    tc_min = params.get('tc_min_days', 5)
    tc_max = params.get('tc_max_days', 120)
    m_min = params.get('m_min', 0.2)
    m_max = params.get('m_max', 0.8)
    omega_min = params.get('omega_min', 5.0)
    omega_max = params.get('omega_max', 20.0)

    for i, sig in enumerate(signals):
        tc_ok = tc_min <= sig['tc_days'] <= tc_max
        m_ok = m_min <= sig['m'] <= m_max
        omega_ok = omega_min <= sig['omega'] <= omega_max

        valid[i] = tc_ok and m_ok and omega_ok

    return valid


def run_trading_logic(
    prices: np.ndarray,
    valid_signals: np.ndarray,
    params: Dict,
    start_idx: int = 252,
) -> BacktestResult:
    """
    Run trading logic with cached signals.

    This is much faster than recalculating signals each time.
    """
    n = len(prices)

    lookback = params.get('signal_lookback', 5)
    entry_thresh = params.get('entry_threshold', 0.6)
    profit_target = params.get('profit_target_pct', 0.15)
    stop_loss = params.get('stop_loss_pct', 0.10)
    max_hold = params.get('max_hold_days', 45)

    trades: List[Trade] = []
    position = 0
    entry_price = 0.0
    entry_idx = 0
    hold_days = 0

    equity = np.ones(n)
    current_equity = 1.0

    # Pad valid_signals to match prices length
    full_valid = np.zeros(n, dtype=bool)
    full_valid[start_idx:start_idx + len(valid_signals)] = valid_signals

    for i in range(start_idx + lookback, n):
        price = prices[i]

        # Update equity
        if position != 0:
            pnl = position * (price / entry_price - 1)
            equity[i] = current_equity * (1 + pnl)
            hold_days += 1
        else:
            equity[i] = current_equity

        # Exit logic
        if position != 0:
            pnl_pct = position * (price / entry_price - 1)
            exit_reason = None

            if pnl_pct >= profit_target:
                exit_reason = 'profit_target'
            elif pnl_pct <= -stop_loss:
                exit_reason = 'stop_loss'
            elif hold_days >= max_hold:
                exit_reason = 'max_hold'

            if exit_reason:
                trade = Trade(
                    entry_idx=entry_idx,
                    entry_price=entry_price,
                    entry_confidence=0.0,
                    direction=position,
                    exit_idx=i,
                    exit_price=price,
                    exit_reason=exit_reason,
                    pnl_pct=pnl_pct,
                )
                trades.append(trade)
                current_equity = equity[i]
                position = 0
                hold_days = 0

        # Entry logic
        if position == 0:
            recent_valid = full_valid[i-lookback:i]
            valid_ratio = np.mean(recent_valid)

            if valid_ratio >= entry_thresh:
                position = -1  # Short
                entry_price = price
                entry_idx = i
                hold_days = 0

    # Close open position
    if position != 0:
        pnl_pct = position * (prices[-1] / entry_price - 1)
        trade = Trade(
            entry_idx=entry_idx,
            entry_price=entry_price,
            entry_confidence=0.0,
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

        returns = np.diff(equity) / (equity[:-1] + 1e-10)
        returns = returns[~np.isnan(returns) & (returns != 0)]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252) if len(returns) > 0 else 0.0

        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / (peak + 1e-10)
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
        confidence_series=np.array([]),
        equity_curve=equity,
    )


def calculate_score(result: BacktestResult) -> float:
    """Calculate optimization score with focus on risk-adjusted returns."""
    if result.n_trades < 3:
        return -100

    score = result.total_pnl_pct * 100
    score += result.sharpe_ratio * 20  # Reward Sharpe more
    score -= result.max_drawdown * 50  # Penalize drawdown heavily

    if result.win_rate > 0.5:
        score += (result.win_rate - 0.5) * 20

    # Penalize too many trades (overtrading)
    if result.n_trades > 50:
        score -= (result.n_trades - 50) * 0.5

    return score


def fast_grid_search(
    prices: np.ndarray,
    signals: List[Dict],
    param_grid: Dict[str, List],
    verbose: bool = True,
) -> Tuple[Dict, float, BacktestResult]:
    """
    Fast grid search using cached signals.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    if verbose:
        print(f"Testing {len(combinations)} combinations...", flush=True)

    best_score = -float('inf')
    best_params = None
    best_result = None

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        # Filter signals with these params
        valid = filter_signals(signals, params)

        # Run trading
        result = run_trading_logic(prices, valid, params)
        score = calculate_score(result)

        if score > best_score:
            best_score = score
            best_params = params
            best_result = result

            if verbose:
                print(f"  [{i+1}] New best: score={score:.1f}, "
                      f"PnL={result.total_pnl_pct*100:.1f}%, "
                      f"trades={result.n_trades}, win={result.win_rate*100:.0f}%", flush=True)

    return best_params, best_score, best_result


def run_fast_optimization(
    model_path: str = 'strategies/lppls/finetuned_model.pt',
    start_date: str = '2020-01-01',
    end_date: str = '2023-12-31',
):
    """
    Run fast parameter optimization.
    """
    print("="*60, flush=True)
    print("Fast LPPLS Parameter Optimization", flush=True)
    print("="*60, flush=True)

    # Load data
    print("\nLoading data...", flush=True)
    df = load_btc_data()
    df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
    df = df.reset_index(drop=True)
    prices = df['close'].values
    print(f"Data: {len(df)} bars", flush=True)

    # Load model and calculate signals once
    print("\nLoading model and calculating signals (one-time)...", flush=True)
    predictor = load_finetuned_predictor(model_path)
    signals = calculate_all_signals(predictor, prices, verbose=True)

    # Grid search - balanced approach
    print("\n--- Grid Search ---", flush=True)
    param_grid = {
        'tc_max_days': [50, 70, 90],
        'tc_min_days': [5, 10],
        'm_min': [0.2],
        'm_max': [0.7],
        'omega_min': [5.0],
        'omega_max': [15.0, 18.0],
        'signal_lookback': [5, 7],
        'entry_threshold': [0.5, 0.6],
        'profit_target_pct': [0.10, 0.12, 0.15],
        'stop_loss_pct': [0.06, 0.08],
        'max_hold_days': [20, 30],
    }

    best_params, best_score, best_result = fast_grid_search(
        prices, signals, param_grid, verbose=True
    )

    print("\n" + "="*60)
    print("Optimization Results")
    print("="*60)

    print(f"\nBest Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    print(f"\nPerformance:")
    print(f"  Total PnL: {best_result.total_pnl_pct*100:.2f}%")
    print(f"  Number of Trades: {best_result.n_trades}")
    print(f"  Win Rate: {best_result.win_rate*100:.1f}%")
    print(f"  Sharpe Ratio: {best_result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {best_result.max_drawdown*100:.1f}%")

    if best_result.trades:
        print(f"\nTrades:")
        for i, trade in enumerate(best_result.trades[:15], 1):
            idx = trade.entry_idx
            date = df['datetime'].iloc[idx].strftime('%Y-%m-%d') if idx < len(df) else 'N/A'
            direction = 'SHORT' if trade.direction == -1 else 'LONG'
            print(f"  {i}. [{date}] {direction}: ${trade.entry_price:.0f} -> "
                  f"${trade.exit_price:.0f} ({trade.exit_reason}) "
                  f"PnL: {trade.pnl_pct*100:+.1f}%")

    # Out-of-sample test
    print("\n" + "="*60)
    print("Out-of-Sample Test (2024)")
    print("="*60)

    df_oos = load_btc_data()
    df_oos = df_oos[(df_oos['datetime'] >= '2024-01-01') & (df_oos['datetime'] <= '2024-12-31')]
    df_oos = df_oos.reset_index(drop=True)

    if len(df_oos) > 500:
        prices_oos = df_oos['close'].values
        print(f"OOS Data: {len(df_oos)} bars")

        signals_oos = calculate_all_signals(predictor, prices_oos, verbose=True)
        valid_oos = filter_signals(signals_oos, best_params)
        result_oos = run_trading_logic(prices_oos, valid_oos, best_params)

        print(f"\nOOS Performance:")
        print(f"  Total PnL: {result_oos.total_pnl_pct*100:.2f}%")
        print(f"  Number of Trades: {result_oos.n_trades}")
        print(f"  Win Rate: {result_oos.win_rate*100:.1f}%")
        print(f"  Sharpe Ratio: {result_oos.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {result_oos.max_drawdown*100:.1f}%")

        if result_oos.trades:
            print(f"\nOOS Trades:")
            for i, trade in enumerate(result_oos.trades, 1):
                idx = trade.entry_idx
                date = df_oos['datetime'].iloc[idx].strftime('%Y-%m-%d') if idx < len(df_oos) else 'N/A'
                direction = 'SHORT' if trade.direction == -1 else 'LONG'
                print(f"  {i}. [{date}] {direction}: ${trade.entry_price:.0f} -> "
                      f"${trade.exit_price:.0f} ({trade.exit_reason}) "
                      f"PnL: {trade.pnl_pct*100:+.1f}%")

    return best_params, best_result


if __name__ == '__main__':
    best_params, best_result = run_fast_optimization()
