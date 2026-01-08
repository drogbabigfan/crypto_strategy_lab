#!/usr/bin/env python3
"""
AKF Strategy Runner.

Generates trading signals using Adaptive Kalman Filter strategies
and outputs them for the Go backtester.

Usage:
    python strategies/akf/run.py --months 2020-01 2020-02 2020-03
    python strategies/akf/run.py --months 2020-01 --strategy strategy_db --k 2.0
    python strategies/akf/run.py --months 2020-01 --backtest  # Run Go backtester
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import subprocess
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from research.features.kalman import calculate_adaptive_kalman
from strategies.akf import AKFStrategies, AKFStrategyConfig
from strategies.akf.strategies_v2 import AKFStrategiesV2, AKFStrategyConfigV2


# Default paths
DATA_DIR = PROJECT_ROOT / "etl/data/features-24/futures/BTCUSDT"
OUTPUT_DIR = PROJECT_ROOT / "strategies/akf/results"
BACKTESTER_PATH = PROJECT_ROOT / "etl/bin/backtester"


def load_data(months: list[str], data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Load feature data for specified months.

    Args:
        months: List of months in YYYY-MM format
        data_dir: Path to features directory

    Returns:
        Concatenated DataFrame
    """
    dfs = []
    for month in months:
        path = data_dir / f"BTCUSDT-features-{month}.parquet"
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        df = pd.read_parquet(path)
        dfs.append(df)
        print(f"Loaded {path.name}: {len(df)} bars")

    if not dfs:
        raise ValueError(f"No data found for months: {months}")

    result = pd.concat(dfs, ignore_index=True)
    print(f"Total: {len(result)} bars")
    return result


def generate_signals(
    df: pd.DataFrame,
    strategy: str = "strategy_db",
    k: float = 2.0,
    warmup: int = 130,
    version: int = 1,
    velocity_threshold: float = 0.0005,
) -> pd.DataFrame:
    """
    Generate AKF trading signals.

    Args:
        df: DataFrame with OHLCV data
        strategy: Strategy name
        k: Band multiplier
        warmup: Warmup period (signals = 0 during warmup)
        version: Strategy version (1 or 2)
        velocity_threshold: Velocity threshold for V2 velocity_scalp

    Returns:
        DataFrame with signal column added
    """
    # Apply Kalman filter
    print(f"Applying Kalman filter...")
    df_kf = calculate_adaptive_kalman(df)

    # Generate strategy signals
    print(f"Generating {strategy} signals (k={k}, version={version})...")

    if version == 1:
        # V1 strategies
        akf = AKFStrategies(AKFStrategyConfig(k=k))
        result = akf.run_strategy(df_kf, strategy, include_bands=True)
    else:
        # V2 strategies
        config = AKFStrategyConfigV2(
            db_k=k,
            benhamou_k=k,
            fusion_k=k,
            velocity_threshold=velocity_threshold,
        )
        akf = AKFStrategiesV2(config)

        if strategy == "strategy_db":
            result = akf.run_strategy_db(df_kf, k=k)
        elif strategy == "strategy_benhamou":
            result = akf.run_strategy_benhamou(df_kf, k=k)
        elif strategy == "strategy_triple_fusion":
            result = akf.run_strategy_triple_fusion(df_kf, k=k)
        elif strategy == "strategy_velocity_scalp":
            result = akf.run_strategy_velocity_scalp(df_kf, threshold=velocity_threshold)
        else:
            raise ValueError(f"Unknown V2 strategy: {strategy}")

        # V2 uses 'position' column, copy to 'signal' for backtester compatibility
        if 'position' in result.columns and 'signal' not in result.columns:
            result['signal'] = result['position']

    # Zero out warmup period
    result.loc[:warmup - 1, 'signal'] = 0

    # Stats
    signal_counts = result['signal'].value_counts().sort_index()
    print(f"Signal distribution:")
    for sig, count in signal_counts.items():
        label = {-1: "Short", 0: "Neutral", 1: "Long"}[sig]
        print(f"  {label:8}: {count:6} ({count/len(result)*100:.1f}%)")

    return result


def save_signals_parquet(
    df: pd.DataFrame,
    output_path: Path,
    timestamp_col: str = "start_time"
) -> None:
    """
    Save signals to parquet for Go backtester.

    Format:
        timestamp: int64
        signal: int8 (-1, 0, 1)
    """
    signals_df = pd.DataFrame({
        'timestamp': df[timestamp_col].astype(np.int64),
        'signal': df['signal'].astype(np.int8),
    })

    schema = pa.schema([
        ('timestamp', pa.int64()),
        ('signal', pa.int8()),
    ])

    table = pa.Table.from_pandas(signals_df, schema=schema)
    pq.write_table(table, output_path)
    print(f"Saved signals to {output_path}")


def save_features_parquet(
    df: pd.DataFrame,
    output_path: Path,
    timestamp_col: str = "start_time"
) -> None:
    """
    Save features (OHLCV) to parquet for Go backtester.

    Format:
        timestamp, open, high, low, close, volume, realized_vol
    """
    # Volume is stored as log_volume, need to convert back
    if 'log_volume' in df.columns and 'volume' not in df.columns:
        volume = np.exp(df['log_volume'])
    elif 'volume' in df.columns:
        volume = df['volume']
    else:
        volume = np.ones(len(df)) * 1000  # Default

    features_df = pd.DataFrame({
        'timestamp': df[timestamp_col].astype(np.int64),
        'open': df['open'].astype(np.float64),
        'high': df['high'].astype(np.float64),
        'low': df['low'].astype(np.float64),
        'close': df['close'].astype(np.float64),
        'volume': volume.astype(np.float64),
        'realized_vol': df['realized_vol'].astype(np.float64),
    })

    schema = pa.schema([
        ('timestamp', pa.int64()),
        ('open', pa.float64()),
        ('high', pa.float64()),
        ('low', pa.float64()),
        ('close', pa.float64()),
        ('volume', pa.float64()),
        ('realized_vol', pa.float64()),
    ])

    table = pa.Table.from_pandas(features_df, schema=schema)
    pq.write_table(table, output_path)
    print(f"Saved features to {output_path}")


def run_go_backtester(
    signals_path: Path,
    features_path: Path,
    exit_mode: str = "signal",
    compounding: bool = False,
    output_path: Path = None,
) -> dict:
    """
    Run Go backtester and return results.

    Args:
        signals_path: Path to signals parquet
        features_path: Path to features parquet
        exit_mode: "signal" (exit on opposite signal) or "barrier" (TP/SL/Timeout)
        compounding: Use compounding returns (reinvest profits)
        output_path: Optional path to save results JSON

    Returns:
        Backtest results dict
    """
    if not BACKTESTER_PATH.exists():
        raise FileNotFoundError(
            f"Go backtester not found at {BACKTESTER_PATH}. "
            f"Build it with: cd etl && go build -o bin/backtester ./cmd/backtester"
        )

    cmd = [
        str(BACKTESTER_PATH),
        '-signals', str(signals_path),
        '-features', str(features_path),
        '-exit-mode', exit_mode,
        '-equity',
        '-quiet',  # Suppress console summary, output JSON only
    ]

    if compounding:
        cmd.append('-compounding')

    # If output_path specified, write to file and read back
    # Otherwise, parse stdout directly
    if output_path:
        cmd.extend(['-output', str(output_path)])

    print(f"Running Go backtester...")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Backtester stderr: {result.stderr}")
        raise RuntimeError(f"Backtester failed with code {result.returncode}")

    # If output file was specified, read from file
    if output_path and output_path.exists():
        with open(output_path, 'r') as f:
            return json.load(f)

    # Otherwise parse stdout
    return json.loads(result.stdout)


def print_results(results: dict) -> None:
    """Print backtest results summary."""
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Total Trades:   {results.get('total_trades', 0)}")
    print(f"Win Rate:       {results.get('win_rate', 0)*100:.1f}%")
    print(f"Avg PnL:        {results.get('avg_pnl', 0)*100:.3f}%")
    print(f"Total PnL:      {results.get('total_pnl', 0)*100:.2f}%")
    print(f"Sharpe Ratio:   {results.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown:   {results.get('max_drawdown', 0)*100:.2f}%")
    print(f"Profit Factor:  {results.get('profit_factor', 0):.2f}")
    print(f"Avg Hold Bars:  {results.get('avg_hold_bars', 0):.1f}")
    print("-" * 50)
    print(f"TP Count:       {results.get('tp_count', 0)}")
    print(f"SL Count:       {results.get('sl_count', 0)}")
    print(f"Timeout Count:  {results.get('timeout_count', 0)}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="AKF Strategy Runner")
    parser.add_argument(
        '--months', nargs='+', required=True,
        help='Months to process (YYYY-MM format)'
    )
    parser.add_argument(
        '--strategy', default='strategy_db',
        choices=[
            # V1 strategies
            'strategy_db', 'strategy_benhamou', 'strategy_hybrid',
            # V2 strategies
            'strategy_triple_fusion', 'strategy_velocity_scalp',
        ],
        help='Strategy to use'
    )
    parser.add_argument(
        '--version', type=int, default=1, choices=[1, 2],
        help='Strategy version (1=original, 2=new with velocity filter)'
    )
    parser.add_argument(
        '--k', type=float, default=2.0,
        help='Band multiplier k'
    )
    parser.add_argument(
        '--velocity-threshold', type=float, default=0.0005,
        help='Velocity threshold for V2 velocity_scalp strategy'
    )
    parser.add_argument(
        '--warmup', type=int, default=130,
        help='Warmup period (bars)'
    )
    parser.add_argument(
        '--data-dir', type=Path, default=DATA_DIR,
        help='Path to features directory'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=OUTPUT_DIR,
        help='Output directory for signals'
    )
    parser.add_argument(
        '--backtest', action='store_true',
        help='Run Go backtester after generating signals'
    )
    parser.add_argument(
        '--exit-mode', default='signal',
        choices=['signal', 'barrier'],
        help='Backtester exit mode'
    )
    parser.add_argument(
        '--compounding', action='store_true',
        help='Use compounding returns (reinvest profits)'
    )

    args = parser.parse_args()

    # Auto-detect version based on strategy name
    if args.strategy in ['strategy_triple_fusion', 'strategy_velocity_scalp']:
        args.version = 2
    # V2 version of existing strategies
    elif args.version == 2 and args.strategy == 'strategy_hybrid':
        args.strategy = 'strategy_triple_fusion'

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\n{'='*50}")
    print(f"Loading data for: {args.months}")
    print(f"{'='*50}")
    df = load_data(args.months, args.data_dir)

    # Generate signals
    print(f"\n{'='*50}")
    print(f"Strategy: {args.strategy}, k={args.k}, version={args.version}")
    print(f"{'='*50}")
    result = generate_signals(
        df,
        args.strategy,
        args.k,
        args.warmup,
        version=args.version,
        velocity_threshold=args.velocity_threshold,
    )

    # Generate output filename
    month_range = f"{args.months[0]}_to_{args.months[-1]}" if len(args.months) > 1 else args.months[0]
    version_suffix = f"_v{args.version}" if args.version > 1 else ""
    base_name = f"akf_{args.strategy}_{month_range}_k{args.k}{version_suffix}"

    signals_path = args.output_dir / f"{base_name}_signals.parquet"
    features_path = args.output_dir / f"{base_name}_features.parquet"

    # Save outputs
    print(f"\n{'='*50}")
    print("Saving outputs")
    print(f"{'='*50}")
    save_signals_parquet(result, signals_path)
    save_features_parquet(result, features_path)

    # Run backtester if requested
    if args.backtest:
        print(f"\n{'='*50}")
        print("Running Backtester")
        print(f"{'='*50}")

        results_path = args.output_dir / f"{base_name}_results.json"
        mode_str = "compounding" if args.compounding else "simple"
        print(f"Return mode: {mode_str}")
        try:
            bt_results = run_go_backtester(
                signals_path,
                features_path,
                args.exit_mode,
                args.compounding,
                results_path,
            )
            print_results(bt_results)
            print(f"\nResults saved to: {results_path}")
        except Exception as e:
            print(f"Backtest failed: {e}")
            return 1

    print(f"\nDone! Signals: {signals_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
