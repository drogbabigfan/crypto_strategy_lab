#!/usr/bin/env python3
"""
Compare Exit Modes: V1 (Fixed) vs V3 (Reversal).

V1 Fixed: signal=0 now triggers exit (Go backtester updated)
V3 Reversal: Always-in-market, exit = reverse position
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
from strategies.akf import AKFStrategies, AKFStrategyConfig
from strategies.akf.strategies_v3 import AKFStrategiesV3, AKFStrategyConfigV3


DATA_DIR = PROJECT_ROOT / "etl/data/features-24/futures/BTCUSDT"
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


def run_backtest(df_signals: pd.DataFrame, warmup: int = 130) -> dict:
    """Run Go backtester and return metrics."""
    # Zero out warmup
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
            '-quiet',
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            print(f"Backtester error: {proc.stderr}")
            return None

        return json.loads(proc.stdout)


def analyze_signals(df: pd.DataFrame, name: str) -> dict:
    """Analyze signal distribution."""
    signals = df['signal'].values
    total = len(signals)
    long_count = (signals == 1).sum()
    short_count = (signals == -1).sum()
    flat_count = (signals == 0).sum()

    # Count transitions
    transitions = np.diff(signals)
    long_entries = ((signals[:-1] != 1) & (signals[1:] == 1)).sum()
    short_entries = ((signals[:-1] != -1) & (signals[1:] == -1)).sum()
    exits_to_flat = ((signals[:-1] != 0) & (signals[1:] == 0)).sum()

    return {
        'name': name,
        'total_bars': total,
        'long_bars': long_count,
        'short_bars': short_count,
        'flat_bars': flat_count,
        'long_pct': long_count / total * 100,
        'short_pct': short_count / total * 100,
        'flat_pct': flat_count / total * 100,
        'long_entries': long_entries,
        'short_entries': short_entries,
        'exits_to_flat': exits_to_flat,
    }


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

    k = 2.0
    results = []

    # Test configurations
    configs = [
        ('V1 Fixed (signal=0 exit)', 'v1', 'strategy_db'),
        ('V3 Reversal (always-in-market)', 'v3', 'strategy_db_reversal'),
    ]

    for name, version, strategy in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")

        # Generate signals
        if version == 'v1':
            akf = AKFStrategies(AKFStrategyConfig(k=k))
            df_signals = akf.strategy_db(df_kf.copy(), k=k)
        else:
            akf = AKFStrategiesV3(AKFStrategyConfigV3(k=k))
            df_signals = akf.strategy_db_reversal(df_kf.copy(), k=k)

        # Analyze signals
        signal_stats = analyze_signals(df_signals, name)
        print(f"  Long bars: {signal_stats['long_pct']:.1f}%")
        print(f"  Short bars: {signal_stats['short_pct']:.1f}%")
        print(f"  Flat bars: {signal_stats['flat_pct']:.1f}%")
        print(f"  Exits to flat: {signal_stats['exits_to_flat']}")

        # Run backtest
        metrics = run_backtest(df_signals)

        if metrics:
            result = {
                'name': name,
                'version': version,
                'trades': metrics.get('total_trades', 0),
                'win_rate': metrics.get('win_rate', 0) * 100,
                'avg_pnl': metrics.get('avg_pnl', 0) * 100,
                'total_pnl': metrics.get('total_pnl', 0) * 100,
                'sharpe': metrics.get('sharpe_ratio', 0),
                'mdd': metrics.get('max_drawdown', 0) * 100,
                'profit_factor': metrics.get('profit_factor', 0),
                'avg_hold': metrics.get('avg_hold_bars', 0),
                'flat_pct': signal_stats['flat_pct'],
            }
            results.append(result)

            print(f"\n  Trades: {result['trades']}")
            print(f"  Win Rate: {result['win_rate']:.1f}%")
            print(f"  Avg PnL: {result['avg_pnl']:.3f}%")
            print(f"  Total PnL: {result['total_pnl']:.1f}%")
            print(f"  Sharpe: {result['sharpe']:.2f}")
            print(f"  MDD: {result['mdd']:.1f}%")
            print(f"  Profit Factor: {result['profit_factor']:.2f}")
            print(f"  Avg Hold: {result['avg_hold']:.1f} bars")

    # Summary table
    if results:
        print("\n" + "="*100)
        print("COMPARISON SUMMARY")
        print("="*100)

        df_results = pd.DataFrame(results)
        print(df_results[['name', 'trades', 'win_rate', 'avg_pnl', 'total_pnl',
                          'sharpe', 'mdd', 'profit_factor', 'avg_hold', 'flat_pct']].to_string(index=False))

        # Save results
        output_path = PROJECT_ROOT / 'strategies/akf/results/exit_mode_comparison.csv'
        df_results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
