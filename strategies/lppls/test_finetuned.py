"""
Test fine-tuned Deep LPPLS model on BTC data.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from .deep_lppls import DeepLPPLSPredictor, DeepLPPLSConfig
from .fast_strategy import FastLPPLSStrategy, FastLPPLSConfig
from .strategy import load_btc_data, BacktestResult, Trade
from typing import List


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


def analyze_predictions(
    predictor: DeepLPPLSPredictor,
    df: pd.DataFrame,
    sample_points: int = 100,
) -> pd.DataFrame:
    """
    Analyze predictions across the dataset.
    """
    n = len(df)
    step = max(1, (n - 252) // sample_points)

    results = []
    for i in range(252, n, step):
        log_prices = np.log(df['close'].iloc[i-252:i].values)
        pred = predictor.predict(log_prices)

        results.append({
            'date': df['datetime'].iloc[i-1],
            'price': df['close'].iloc[i-1],
            'tc_norm': pred['tc_norm'],
            'tc_days': pred['tc_norm'] * 252,
            'm': pred['m'],
            'omega': pred['omega'],
        })

    return pd.DataFrame(results)


def run_backtest_with_finetuned(
    model_path: str = 'strategies/lppls/finetuned_model.pt',
    start_date: str = '2020-01-01',
    end_date: str = '2023-12-31',
) -> BacktestResult:
    """
    Run backtest using fine-tuned model.
    """
    print("Loading data...")
    df = load_btc_data()
    df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
    df = df.reset_index(drop=True)
    print(f"Data: {len(df)} bars from {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")

    # Load fine-tuned predictor
    print("\nLoading fine-tuned model...")
    predictor = load_finetuned_predictor(model_path)

    # Create strategy with fine-tuned model
    config = FastLPPLSConfig(
        input_length=252,
        # Don't re-train, use loaded model
        train_samples=0,
        train_epochs=0,
        # Signal thresholds - more permissive
        tc_max_days=200,
        tc_min_days=5,
        m_min=0.15,
        m_max=0.85,
        omega_min=4.0,
        omega_max=25.0,
        signal_lookback=3,
        entry_threshold=0.4,
        profit_target_pct=0.15,
        stop_loss_pct=0.10,
        max_hold_days=45,
    )

    strategy = FastLPPLSStrategy(config)
    strategy.predictor = predictor
    strategy._is_ready = True

    # Calculate signals
    print("\nCalculating signals...")
    prices = df['close'].values
    valid_signals, confidences, all_signals = strategy.calculate_signals(prices, verbose=True)

    # Analyze signal distribution
    n_valid = np.sum(valid_signals)
    print(f"\nSignal Analysis:")
    print(f"  Valid signals: {n_valid} / {len(valid_signals)} ({100*n_valid/len(valid_signals):.1f}%)")

    if all_signals:
        tc_days = [s.tc_days_ahead for s in all_signals if s.is_valid]
        ms = [s.m for s in all_signals if s.is_valid]
        omegas = [s.omega for s in all_signals if s.is_valid]

        if tc_days:
            print(f"  tc_days (valid): min={min(tc_days):.0f}, max={max(tc_days):.0f}, mean={np.mean(tc_days):.0f}")
            print(f"  m (valid): min={min(ms):.3f}, max={max(ms):.3f}, mean={np.mean(ms):.3f}")
            print(f"  omega (valid): min={min(omegas):.1f}, max={max(omegas):.1f}, mean={np.mean(omegas):.1f}")

    # Run backtest
    print("\nRunning backtest...")
    result = strategy.backtest(prices, verbose=False)

    print("\n" + "="*60)
    print("Backtest Results (Fine-tuned Model)")
    print("="*60)
    print(f"Total PnL: {result.total_pnl_pct*100:.2f}%")
    print(f"Number of Trades: {result.n_trades}")
    print(f"Win Rate: {result.win_rate*100:.1f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown*100:.1f}%")

    if result.trades:
        print(f"\nTrade Details:")
        for i, trade in enumerate(result.trades[:20], 1):
            idx = trade.entry_idx
            date = df['datetime'].iloc[idx].strftime('%Y-%m-%d') if idx < len(df) else 'N/A'
            direction = 'SHORT' if trade.direction == -1 else 'LONG'
            print(f"  {i}. [{date}] {direction}: ${trade.entry_price:.0f} -> "
                  f"${trade.exit_price:.0f} ({trade.exit_reason}) "
                  f"PnL: {trade.pnl_pct*100:+.1f}%")

    return result


def compare_with_known_crashes(
    model_path: str = 'strategies/lppls/finetuned_model.pt',
):
    """
    Check model predictions around known crash dates.
    """
    print("="*60)
    print("Checking predictions around known crash dates")
    print("="*60)

    predictor = load_finetuned_predictor(model_path)

    # Resample to daily
    df = load_btc_data()
    df['date'] = df['datetime'].dt.date
    daily = df.groupby('date').agg({
        'close': 'last',
        'datetime': 'last',
    }).reset_index()
    daily['datetime'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('datetime').reset_index(drop=True)

    known_crashes = [
        ('2021-04-14', 'April 2021 peak'),
        ('2021-11-10', 'November 2021 ATH'),
        ('2024-03-14', 'March 2024 peak'),
    ]

    for crash_date, description in known_crashes:
        crash_dt = pd.to_datetime(crash_date)

        # Check predictions leading up to crash
        print(f"\n{description} ({crash_date}):")

        for days_before in [30, 20, 10, 5, 0]:
            check_dt = crash_dt - pd.Timedelta(days=days_before)
            mask = daily['datetime'] <= check_dt
            if mask.sum() < 252:
                continue

            idx = mask.sum() - 1
            log_prices = np.log(daily['close'].iloc[idx-251:idx+1].values)

            pred = predictor.predict(log_prices)

            print(f"  {days_before} days before: tc_days={pred['tc_norm']*252:.0f}, "
                  f"m={pred['m']:.3f}, omega={pred['omega']:.1f}")


if __name__ == '__main__':
    # First check predictions around known crashes
    compare_with_known_crashes()

    print("\n")

    # Then run backtest
    result = run_backtest_with_finetuned(
        start_date='2020-01-01',
        end_date='2023-12-31',
    )
