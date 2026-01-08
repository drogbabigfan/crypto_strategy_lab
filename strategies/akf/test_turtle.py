#!/usr/bin/env python3
"""
Test AKF Turtle Strategy (Donchian-Kalman Hybrid).

Tests the classic Donchian Channel breakout filtered by Kalman regime detection.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import subprocess
import tempfile
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from research.features.kalman import calculate_adaptive_kalman
from strategies.akf.strategies_turtle import (
    AKFTurtleStrategy,
    TurtleConfig,
    prepare_kalman_data
)


DATA_DIR = PROJECT_ROOT / "etl/data/features-6/futures/BTCUSDT"
BACKTESTER_PATH = PROJECT_ROOT / "etl/bin/backtester"


def generate_months(start_year: int, start_month: int, end_year: int, end_month: int) -> list[str]:
    start = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)
    months = []
    current = start
    while current <= end:
        months.append(current.strftime('%Y-%m'))
        current += relativedelta(months=1)
    return months


def load_data(months: list[str]) -> pd.DataFrame:
    dfs = []
    for month in months:
        path = DATA_DIR / f"BTCUSDT-features-{month}.parquet"
        if path.exists():
            dfs.append(pd.read_parquet(path))
    if not dfs:
        raise ValueError(f"No data found")
    return pd.concat(dfs, ignore_index=True)


def run_backtest(df_signals: pd.DataFrame, warmup: int = 130, max_loss: float = 0.5) -> dict:
    """Run Go backtester and return metrics."""
    df_signals = df_signals.copy()
    df_signals.loc[:warmup - 1, 'signal'] = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        signals_path = Path(tmpdir) / "signals.parquet"
        features_path = Path(tmpdir) / "features.parquet"

        # Save signals
        signals_df = pd.DataFrame({
            'timestamp': df_signals['start_time'].astype(np.int64),
            'signal': df_signals['signal'].astype(np.int8),
        })
        schema = pa.schema([('timestamp', pa.int64()), ('signal', pa.int8())])
        table = pa.Table.from_pandas(signals_df, schema=schema)
        pq.write_table(table, signals_path)

        # Save features
        if 'log_volume' in df_signals.columns:
            volume = np.exp(df_signals['log_volume'])
        elif 'volume' in df_signals.columns:
            volume = df_signals['volume']
        else:
            volume = np.ones(len(df_signals)) * 1000

        features_df = pd.DataFrame({
            'timestamp': df_signals['start_time'].astype(np.int64),
            'open': df_signals['open'].astype(np.float64),
            'high': df_signals['high'].astype(np.float64),
            'low': df_signals['low'].astype(np.float64),
            'close': df_signals['close'].astype(np.float64),
            'volume': volume.astype(np.float64),
            'realized_vol': df_signals['realized_vol'].astype(np.float64),
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
        pq.write_table(table, features_path)

        # Run backtester
        cmd = [
            str(BACKTESTER_PATH),
            '-signals', str(signals_path),
            '-features', str(features_path),
            '-exit-mode', 'signal',
            '-max-loss', str(max_loss),
            '-quiet',
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            print(f"Backtester error: {proc.stderr}")
            return None

        return json.loads(proc.stdout)


def analyze_strategy_behavior(df: pd.DataFrame):
    """Analyze strategy signal characteristics."""
    print("\n--- Strategy Behavior Analysis ---")

    # Regime distribution
    regime_counts = df['regime'].value_counts()
    print(f"Regime Distribution:")
    print(f"  Tradeable (1): {regime_counts.get(1, 0)} bars ({regime_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"  Extreme (0):   {regime_counts.get(0, 0)} bars ({regime_counts.get(0, 0)/len(df)*100:.1f}%)")

    # Signal distribution
    signal_counts = df['signal'].value_counts()
    print(f"\nSignal Distribution:")
    print(f"  Long (1):   {signal_counts.get(1, 0)} bars")
    print(f"  Short (-1): {signal_counts.get(-1, 0)} bars")
    print(f"  Flat (0):   {signal_counts.get(0, 0)} bars")

    # Donchian channel stats
    print(f"\nDonchian Channel Stats:")
    channel_width = df['donchian_upper'] - df['donchian_lower']
    print(f"  Avg Width: {channel_width.mean():.2f}")
    print(f"  Breakout (Upper): {(df['close'] > df['donchian_upper']).sum()} bars")
    print(f"  Breakdown (Lower): {(df['close'] < df['donchian_lower']).sum()} bars")

    # Kalman filter stats
    print(f"\nKalman Filter Stats:")
    print(f"  Velocity > 0: {(df['velocity'] > 0).sum()} bars ({(df['velocity'] > 0).mean()*100:.1f}%)")
    print(f"  Velocity < 0: {(df['velocity'] < 0).sum()} bars ({(df['velocity'] < 0).mean()*100:.1f}%)")
    print(f"  Uncertainty: mean={df['uncertainty'].mean():.4f}, std={df['uncertainty'].std():.4f}")


def main():
    # Full period
    months = generate_months(2020, 1, 2025, 11)
    print(f"Period: {months[0]} to {months[-1]} ({len(months)} months)")

    # Load data
    print("Loading data...")
    df = load_data(months)
    print(f"Loaded {len(df)} bars")

    # Apply Kalman filter
    print("Applying Kalman filter...")
    df_kf = calculate_adaptive_kalman(df)

    # Prepare columns
    df_prepared = prepare_kalman_data(df_kf)

    # Verify columns
    required = ['close', 'high', 'low', 'trend', 'velocity', 'uncertainty']
    print(f"Required columns present: {all(c in df_prepared.columns for c in required)}")

    # Test configurations - Focus on Exit Modes
    configs = [
        # =====================================================================
        # Exit Mode Comparison (55-bar entry, best from previous test)
        # =====================================================================

        # Donchian Exit (Classic Turtle: 55 entry, 20 exit)
        (TurtleConfig(
            donchian_period=55,
            exit_mode='donchian',
            exit_donchian_period=20
        ), 'Donchian55_Exit20'),

        # Donchian Exit (55 entry, 10 exit - faster)
        (TurtleConfig(
            donchian_period=55,
            exit_mode='donchian',
            exit_donchian_period=10
        ), 'Donchian55_Exit10'),

        # Trend Cross Exit
        (TurtleConfig(
            donchian_period=55,
            exit_mode='trend_cross'
        ), 'Donchian55_TrendCross'),

        # ATR Exit (2x ATR)
        (TurtleConfig(
            donchian_period=55,
            exit_mode='atr',
            exit_atr_mult=2.0
        ), 'Donchian55_ATR2.0'),

        # ATR Exit (3x ATR - wider)
        (TurtleConfig(
            donchian_period=55,
            exit_mode='atr',
            exit_atr_mult=3.0
        ), 'Donchian55_ATR3.0'),

        # Kalman Exit (original - for comparison)
        (TurtleConfig(
            donchian_period=55,
            exit_mode='kalman',
            exit_buffer_mult=1.5
        ), 'Donchian55_Kalman1.5'),

        # =====================================================================
        # Best Exit Mode with Different Entry Periods
        # =====================================================================

        # 40-bar entry with Donchian exit
        (TurtleConfig(
            donchian_period=40,
            exit_mode='donchian',
            exit_donchian_period=15
        ), 'Donchian40_Exit15'),

        # 40-bar entry with Trend Cross
        (TurtleConfig(
            donchian_period=40,
            exit_mode='trend_cross'
        ), 'Donchian40_TrendCross'),

        # 20-bar entry with Donchian exit
        (TurtleConfig(
            donchian_period=20,
            exit_mode='donchian',
            exit_donchian_period=10
        ), 'Donchian20_Exit10'),

        # 20-bar entry with Trend Cross
        (TurtleConfig(
            donchian_period=20,
            exit_mode='trend_cross'
        ), 'Donchian20_TrendCross'),
    ]

    results = []

    print("\n" + "=" * 80)
    print("TESTING AKF TURTLE STRATEGY (Donchian-Kalman Hybrid)")
    print("=" * 80)

    for config, name in configs:
        print(f"\n>>> {name}...")
        print(f"    donchian={config.donchian_period}, exit_mode={config.exit_mode}")

        try:
            strategy = AKFTurtleStrategy(config)
            df_signals = strategy.run_strategy(df_prepared.copy())

            # Analyze behavior for baseline
            if name == 'Baseline':
                analyze_strategy_behavior(df_signals)

            metrics = run_backtest(df_signals)

            if metrics:
                result = {
                    'name': name,
                    'trades': metrics.get('total_trades', 0),
                    'win_rate': metrics.get('win_rate', 0) * 100,
                    'avg_pnl': metrics.get('avg_pnl', 0) * 100,
                    'total_pnl': metrics.get('total_pnl', 0) * 100,
                    'sharpe': metrics.get('sharpe_ratio', 0),
                    'mdd': metrics.get('max_drawdown', 0) * 100,
                    'profit_factor': metrics.get('profit_factor', 0),
                    'avg_hold': metrics.get('avg_hold_bars', 0),
                }
                results.append(result)

                status = "STOPPED" if result['total_pnl'] <= -50 else "OK"
                print(f"    [{status}] Trades: {result['trades']}, PnL: {result['total_pnl']:.1f}%, "
                      f"WinRate: {result['win_rate']:.1f}%, Sharpe: {result['sharpe']:.2f}")
            else:
                print(f"    Failed to run backtest")

        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if results:
        print("\n" + "=" * 100)
        print("SUMMARY - AKF Turtle Strategy (Donchian-Kalman Hybrid)")
        print("=" * 100)

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('total_pnl', ascending=False)

        print(df_results[['name', 'trades', 'win_rate', 'avg_pnl', 'total_pnl',
                          'sharpe', 'mdd', 'profit_factor', 'avg_hold']].to_string(index=False))

        # Stats
        profitable = (df_results['total_pnl'] > 0).sum()
        stopped = (df_results['total_pnl'] <= -50).sum()

        print(f"\nProfitable: {profitable}/{len(results)}")
        print(f"Early Stop (-50%): {stopped}/{len(results)}")

        # Best performer
        best = df_results.iloc[0]
        print(f"\nBest: {best['name']} with {best['total_pnl']:.1f}% PnL, Sharpe: {best['sharpe']:.2f}")

        # Save results
        output_path = PROJECT_ROOT / 'strategies/akf/results/turtle_results.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
