#!/usr/bin/env python3
"""
Test AKF Advanced Strategy.

Key differences from previous strategies:
1. KF-derived indicators (RSI, Z-Score, Efficiency)
2. Regime detection (only trade in trending markets)
3. Dynamic risk management (uncertainty-based sizing)
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
from strategies.akf.strategies_advanced import (
    AKFAdvancedStrategy,
    AKFAdvancedConfig,
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
    print(f"  Trending (1): {regime_counts.get(1, 0)} bars ({regime_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"  Ranging (0):  {regime_counts.get(0, 0)} bars ({regime_counts.get(0, 0)/len(df)*100:.1f}%)")

    # Signal distribution
    signal_counts = df['signal'].value_counts()
    print(f"\nSignal Distribution:")
    print(f"  Long (1):  {signal_counts.get(1, 0)} bars")
    print(f"  Short (-1): {signal_counts.get(-1, 0)} bars")
    print(f"  Flat (0):  {signal_counts.get(0, 0)} bars")

    # Entry conditions analysis
    print(f"\nEntry Condition Stats:")
    print(f"  entry_long triggers:  {df['entry_long'].sum()}")
    print(f"  entry_short triggers: {df['entry_short'].sum()}")

    # Feature statistics
    print(f"\nFeature Statistics:")
    print(f"  z_score: mean={df['z_score'].mean():.3f}, std={df['z_score'].std():.3f}")
    print(f"  kf_efficiency: mean={df['kf_efficiency'].mean():.3f}, median={df['kf_efficiency'].median():.3f}")
    print(f"  kf_rsi: mean={df['kf_rsi'].mean():.1f}, min={df['kf_rsi'].min():.1f}, max={df['kf_rsi'].max():.1f}")


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

    # Prepare for advanced strategy (rename columns)
    df_prepared = prepare_kalman_data(df_kf)

    # Verify columns
    required = ['close', 'high', 'low', 'trend', 'velocity', 'trend_pred', 'uncertainty']
    print(f"Required columns present: {all(c in df_prepared.columns for c in required)}")

    # Test configurations
    configs = [
        # Baseline
        (AKFAdvancedConfig(), 'Baseline'),

        # Lower efficiency threshold (more trades)
        (AKFAdvancedConfig(efficiency_threshold=0.3), 'LowEfficiency(0.3)'),

        # Higher efficiency threshold (fewer, cleaner trades)
        (AKFAdvancedConfig(efficiency_threshold=0.7), 'HighEfficiency(0.7)'),

        # Tighter Z-score (more signals)
        (AKFAdvancedConfig(z_score_long=0.5, z_score_short=-0.5), 'TightZ(0.5)'),

        # Wider Z-score (fewer signals)
        (AKFAdvancedConfig(z_score_long=1.5, z_score_short=-1.5), 'WideZ(1.5)'),

        # Relaxed RSI filters
        (AKFAdvancedConfig(rsi_overbought=80, rsi_oversold=20), 'RelaxedRSI'),

        # Combined: more signals
        (AKFAdvancedConfig(
            efficiency_threshold=0.3,
            z_score_long=0.5,
            z_score_short=-0.5,
            rsi_overbought=80,
            rsi_oversold=20
        ), 'Aggressive'),

        # Combined: fewer, high-quality signals
        (AKFAdvancedConfig(
            efficiency_threshold=0.7,
            z_score_long=1.5,
            z_score_short=-1.5,
            rsi_overbought=65,
            rsi_oversold=35
        ), 'Conservative'),
    ]

    results = []

    print("\n" + "=" * 80)
    print("TESTING AKF ADVANCED STRATEGY (with -50% early stop)")
    print("=" * 80)

    for config, name in configs:
        print(f"\n>>> {name}...")
        print(f"    efficiency_threshold={config.efficiency_threshold}, "
              f"z_score={config.z_score_long}, rsi={config.rsi_overbought}/{config.rsi_oversold}")

        try:
            strategy = AKFAdvancedStrategy(config)
            df_signals = strategy.run_strategy(df_prepared.copy())

            # Analyze behavior for first config
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
        print("SUMMARY - AKF Advanced Strategy")
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
        print(f"\nBest: {best['name']} with {best['total_pnl']:.1f}% PnL")

        # Save results
        output_path = PROJECT_ROOT / 'strategies/akf/results/advanced_results.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
