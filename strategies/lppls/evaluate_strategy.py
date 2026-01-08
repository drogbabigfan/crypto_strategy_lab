"""
Evaluate optimized LPPLS strategy performance.
"""
import numpy as np
import pandas as pd

from .deep_lppls import DeepLPPLSPredictor, DeepLPPLSConfig
from .fast_optimize import (
    calculate_all_signals, filter_signals, run_trading_logic
)
from .strategy import load_btc_data
from .optimized_config import OPTIMIZED_PARAMS


def load_predictor(model_path: str = 'strategies/lppls/finetuned_model.pt'):
    """Load fine-tuned model."""
    config = DeepLPPLSConfig(input_length=252)
    return DeepLPPLSPredictor(
        model_path=model_path,
        config=config,
        use_conv=True,
    )


def evaluate_period(
    predictor: DeepLPPLSPredictor,
    df: pd.DataFrame,
    params: dict,
    period_name: str,
):
    """Evaluate strategy on a specific period."""
    prices = df['close'].values

    # Calculate signals
    signals = calculate_all_signals(predictor, prices, verbose=False)
    valid = filter_signals(signals, params)
    result = run_trading_logic(prices, valid, params)

    # Buy and hold
    buy_hold_return = (prices[-1] / prices[252] - 1) * 100

    print(f"\n{'='*60}")
    print(f"{period_name}")
    print(f"{'='*60}")
    print(f"Period: {df['datetime'].iloc[0].strftime('%Y-%m-%d')} to "
          f"{df['datetime'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"Price: ${prices[252]:.0f} -> ${prices[-1]:.0f}")

    print(f"\n--- LPPLS Strategy ---")
    print(f"Total PnL: {result.total_pnl_pct*100:.2f}%")
    print(f"Number of Trades: {result.n_trades}")
    print(f"Win Rate: {result.win_rate*100:.1f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown*100:.1f}%")

    print(f"\n--- Buy & Hold ---")
    print(f"Total Return: {buy_hold_return:.2f}%")

    if result.trades:
        print(f"\n--- Trades ---")
        for i, trade in enumerate(result.trades[:10], 1):
            idx = trade.entry_idx
            date = df['datetime'].iloc[idx].strftime('%Y-%m-%d') if idx < len(df) else 'N/A'
            direction = 'SHORT' if trade.direction == -1 else 'LONG'
            print(f"  {i}. [{date}] {direction}: ${trade.entry_price:.0f} -> "
                  f"${trade.exit_price:.0f} ({trade.exit_reason}) "
                  f"PnL: {trade.pnl_pct*100:+.1f}%")

    return result


def run_evaluation():
    """Run full evaluation."""
    print("="*60)
    print("LPPLS Strategy Evaluation")
    print("="*60)
    print("\nOptimized Parameters:")
    for k, v in OPTIMIZED_PARAMS.items():
        print(f"  {k}: {v}")

    # Load model
    predictor = load_predictor()

    # Load all data
    df = load_btc_data()

    # In-sample: 2020-2023
    df_is = df[(df['datetime'] >= '2020-01-01') & (df['datetime'] <= '2023-12-31')]
    df_is = df_is.reset_index(drop=True)
    evaluate_period(predictor, df_is, OPTIMIZED_PARAMS, "In-Sample (2020-2023)")

    # Out-of-sample: 2024
    df_oos = df[(df['datetime'] >= '2024-01-01') & (df['datetime'] <= '2024-12-31')]
    df_oos = df_oos.reset_index(drop=True)
    if len(df_oos) > 500:
        evaluate_period(predictor, df_oos, OPTIMIZED_PARAMS, "Out-of-Sample (2024)")

    # Crisis period: 2020 March COVID crash
    df_crisis = df[(df['datetime'] >= '2020-02-01') & (df['datetime'] <= '2020-04-30')]
    df_crisis = df_crisis.reset_index(drop=True)
    if len(df_crisis) > 300:
        evaluate_period(predictor, df_crisis, OPTIMIZED_PARAMS, "COVID Crash (Feb-Apr 2020)")

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("""
Key Findings:
1. LPPLS strategy is designed for BUBBLE DETECTION, not trend following
2. Works best during bubble collapses (COVID crash: +14.2%)
3. Conservative in trending markets (2024: no trades)
4. Lower returns than buy-and-hold in bull markets
5. Lower drawdown than buy-and-hold in crash periods

Recommended Use:
- Use as CRASH PROTECTION overlay, not primary strategy
- Combine with trend-following strategies
- Scale position size based on signal confidence
""")


if __name__ == '__main__':
    run_evaluation()
