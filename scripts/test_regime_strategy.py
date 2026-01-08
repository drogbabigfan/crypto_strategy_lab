"""
Test Regime Detection Strategy with Real BTC Data.

Usage:
    python scripts/test_regime_strategy.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
import glob


def load_btc_data(data_dir: str = "etl/data/bars-24/futures/BTCUSDT") -> pd.DataFrame:
    """Load all BTC parquet files and combine them."""
    parquet_files = sorted(glob.glob(f"{data_dir}/*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    dfs = []
    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not dfs:
        raise ValueError("No data loaded")

    data = pd.concat(dfs, ignore_index=True)

    # Sort by timestamp if available
    if 'timestamp' in data.columns:
        data = data.sort_values('timestamp').reset_index(drop=True)
    elif 'open_time' in data.columns:
        data = data.sort_values('open_time').reset_index(drop=True)

    return data


def prepare_ohlc_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare OHLC data from raw data."""
    # Rename columns if needed
    column_mapping = {
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
    }

    result = df.rename(columns=column_mapping).copy()

    # Ensure required columns exist
    required = ['open', 'high', 'low', 'close']
    missing = [c for c in required if c not in result.columns]
    if missing:
        print(f"Available columns: {result.columns.tolist()}")
        raise ValueError(f"Missing required columns: {missing}")

    # Set timestamp as index if available
    if 'timestamp' in result.columns:
        result['timestamp'] = pd.to_datetime(result['timestamp'])
        result = result.set_index('timestamp')
    elif 'open_time' in result.columns:
        result['open_time'] = pd.to_datetime(result['open_time'], unit='ms')
        result = result.set_index('open_time')

    # Remove duplicates
    result = result[~result.index.duplicated(keep='first')]

    return result[['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in result.columns else [])]


def test_hurst_on_btc(data: pd.DataFrame):
    """Test Hurst exponent calculation on BTC data."""
    from strategies.regime_detection import HurstCalculator

    print("\n" + "=" * 50)
    print("HURST EXPONENT ANALYSIS")
    print("=" * 50)

    calculator = HurstCalculator()
    returns = data['close'].pct_change().dropna().values

    # Full series Hurst
    result = calculator.calculate_dfa(returns)
    print(f"\nFull Series Analysis:")
    print(f"  Hurst Exponent: {result.hurst:.4f}")
    print(f"  Regime: {result.regime.value}")
    print(f"  Confidence (R²): {result.confidence:.4f}")

    # Rolling Hurst
    rolling = calculator.calculate_rolling(
        pd.Series(returns),
        window=100,
        method='dfa'
    )

    print(f"\nRolling Hurst Statistics:")
    hurst_values = rolling['hurst'].dropna()
    print(f"  Mean: {hurst_values.mean():.4f}")
    print(f"  Std: {hurst_values.std():.4f}")
    print(f"  Min: {hurst_values.min():.4f}")
    print(f"  Max: {hurst_values.max():.4f}")

    # Regime distribution
    regime_counts = rolling['regime'].value_counts()
    print(f"\nRegime Distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(rolling) * 100
        print(f"  {regime}: {count} ({pct:.1f}%)")

    return rolling


def test_labeling_on_btc(data: pd.DataFrame):
    """Test labeling methods on BTC data."""
    from strategies.regime_detection import ZigZagLabeler, TrendScanningLabeler

    print("\n" + "=" * 50)
    print("ZIGZAG LABELING ANALYSIS")
    print("=" * 50)

    # ZigZag
    zigzag = ZigZagLabeler()
    zigzag_labels = zigzag.label(data)
    pivots = zigzag.get_pivots()

    print(f"\nZigZag Pivots Found: {len(pivots)}")
    if len(pivots) > 0:
        print(f"  Peaks: {(pivots['type'] == 'peak').sum()}")
        print(f"  Troughs: {(pivots['type'] == 'trough').sum()}")

    label_counts = zigzag_labels['label_name'].value_counts()
    print(f"\nLabel Distribution:")
    for label, count in label_counts.items():
        pct = count / len(zigzag_labels) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

    return zigzag_labels


def test_regime_strategy(data: pd.DataFrame):
    """Test the regime-based trading strategy."""
    from strategies.regime_detection.strategy import (
        RegimeStrategy, RegimeStrategyConfig, RegimeBacktester
    )

    print("\n" + "=" * 50)
    print("REGIME STRATEGY BACKTEST")
    print("=" * 50)

    # Configure strategy
    config = RegimeStrategyConfig(
        hurst_window=100,
        momentum_lookback=20,
        momentum_entry_zscore=1.5,
        mean_revert_entry_zscore=2.0,
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=3.0,
    )

    strategy = RegimeStrategy(config)

    # Generate signals
    print("\nGenerating signals...")
    signals = strategy.generate_signals(data)

    # Print signal distribution
    print(f"\nSignal Distribution:")
    signal_counts = signals['signal'].value_counts()
    for sig, count in signal_counts.items():
        pct = count / len(signals) * 100
        label = {1: "Long", -1: "Short", 0: "Flat"}[sig]
        print(f"  {label}: {count} ({pct:.1f}%)")

    # Run backtest
    print("\nRunning backtest...")
    backtester = RegimeBacktester(
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0005,
    )

    results = backtester.run(signals)
    backtester.print_summary()

    # Regime-specific performance
    print("\nRegime-Specific Analysis:")
    for regime in ['trending', 'mean_reverting', 'random_walk']:
        mask = results['hurst_regime'] == regime
        if mask.sum() > 0:
            regime_pnl = results.loc[mask, 'pnl'].sum()
            print(f"  {regime}: ${regime_pnl:,.2f}")

    return results, backtester


def compare_with_buy_and_hold(data: pd.DataFrame, strategy_results: pd.DataFrame):
    """Compare strategy with buy and hold."""
    print("\n" + "=" * 50)
    print("COMPARISON: Strategy vs Buy & Hold")
    print("=" * 50)

    initial = 100000

    # Buy and hold returns
    bh_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
    bh_final = initial * (1 + bh_return / 100)

    # Strategy returns
    strategy_return = (strategy_results['equity'].iloc[-1] / initial - 1) * 100
    strategy_final = strategy_results['equity'].iloc[-1]

    print(f"\nBuy & Hold:")
    print(f"  Return: {bh_return:.2f}%")
    print(f"  Final Equity: ${bh_final:,.2f}")

    print(f"\nRegime Strategy:")
    print(f"  Return: {strategy_return:.2f}%")
    print(f"  Final Equity: ${strategy_final:,.2f}")

    print(f"\nOutperformance: {strategy_return - bh_return:.2f}%")


def main():
    print("=" * 60)
    print("REGIME DETECTION STRATEGY TEST - BTC/USDT")
    print("=" * 60)

    # Load data
    print("\nLoading BTC data...")
    try:
        raw_data = load_btc_data()
        print(f"Loaded {len(raw_data)} rows")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure BTC data is available in etl/data/bars-24/futures/BTCUSDT/")
        return

    # Prepare OHLC
    print("\nPreparing OHLC data...")
    data = prepare_ohlc_data(raw_data)
    print(f"OHLC data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")

    # Subset for faster testing (last 2 years)
    if len(data) > 10000:
        print(f"\nUsing last 10000 bars for testing...")
        data = data.iloc[-10000:]

    # Test Hurst
    hurst_results = test_hurst_on_btc(data)

    # Test labeling
    labels = test_labeling_on_btc(data)

    # Test strategy
    strategy_results, backtester = test_regime_strategy(data)

    # Compare with buy and hold
    compare_with_buy_and_hold(data, strategy_results)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
