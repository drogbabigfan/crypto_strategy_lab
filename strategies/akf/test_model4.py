#!/usr/bin/env python3
"""
Test Benhamou's Model 4 strategies.

Model 4 = Kalman Filter + Stochastic Oscillator
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

from research.features.kalman import calculate_kalman_model4
from strategies.akf.strategies_model4 import BenhamouModel4, BenhamouModel4Config


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

        # Run backtester with max-loss
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


def main():
    # Full period
    months = generate_months(2020, 1, 2025, 11)
    print(f"Period: {months[0]} to {months[-1]} ({len(months)} months)")

    # Load data
    print("Loading data...")
    df = load_data(months)
    print(f"Loaded {len(df)} bars")

    # Apply Kalman Model 4 (with oscillator)
    print("Applying Kalman Model 4 (Kalman + Stochastic Oscillator)...")
    df_kf = calculate_kalman_model4(df, osc_period=14)

    # Verify columns
    print(f"Columns: {[c for c in df_kf.columns if 'kf_' in c or 'stoch' in c or 'osc_' in c]}")

    # Test configurations
    configs = [
        # Model 4 Basic (Kalman + Oscillator filter)
        ('strategy_model4_basic', 2.0, 'M4 Basic k=2.0'),
        ('strategy_model4_basic', 1.5, 'M4 Basic k=1.5'),
        ('strategy_model4_basic', 1.0, 'M4 Basic k=1.0'),

        # Model 4 with exit (trend reversion exit)
        ('strategy_model4_with_exit', 2.0, 'M4 WithExit k=2.0'),
        ('strategy_model4_with_exit', 1.5, 'M4 WithExit k=1.5'),

        # Model 4 Oscillator Reversal (mean reversion at extremes)
        ('strategy_model4_oscillator_reversal', 2.0, 'M4 OscReversal k=2.0'),
        ('strategy_model4_oscillator_reversal', 1.0, 'M4 OscReversal k=1.0'),

        # Model 4 Combo (trend + MR adaptive)
        ('strategy_model4_combo', 2.0, 'M4 Combo k=2.0'),
        ('strategy_model4_combo', 1.5, 'M4 Combo k=1.5'),
    ]

    results = []

    print("\n" + "=" * 80)
    print("TESTING BENHAMOU MODEL 4 STRATEGIES (with -50% early stop)")
    print("=" * 80)

    for strategy, k, name in configs:
        print(f"\n>>> {name}...")

        try:
            model4 = BenhamouModel4(BenhamouModel4Config(k=k))
            df_signals = model4.run_strategy(df_kf.copy(), strategy, k=k)

            metrics = run_backtest(df_signals)

            if metrics:
                result = {
                    'name': name,
                    'strategy': strategy,
                    'k': k,
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
                print(f"  [{status}] Trades: {result['trades']}, PnL: {result['total_pnl']:.1f}%, "
                      f"WinRate: {result['win_rate']:.1f}%, Sharpe: {result['sharpe']:.2f}")
            else:
                print(f"  Failed to run backtest")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Test without oscillator filter (pure Kalman)
    print("\n" + "-" * 80)
    print("TESTING WITHOUT OSCILLATOR FILTER (Pure Kalman)")
    print("-" * 80)

    for strategy, k, name in [
        ('strategy_model4_basic', 2.0, 'Pure Kalman k=2.0'),
        ('strategy_model4_basic', 1.5, 'Pure Kalman k=1.5'),
    ]:
        print(f"\n>>> {name}...")

        try:
            config = BenhamouModel4Config(k=k, use_osc_filter=False)
            model4 = BenhamouModel4(config)
            df_signals = model4.run_strategy(df_kf.copy(), strategy, k=k)

            metrics = run_backtest(df_signals)

            if metrics:
                result = {
                    'name': name,
                    'strategy': strategy,
                    'k': k,
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
                print(f"  [{status}] Trades: {result['trades']}, PnL: {result['total_pnl']:.1f}%, "
                      f"WinRate: {result['win_rate']:.1f}%, Sharpe: {result['sharpe']:.2f}")

        except Exception as e:
            print(f"  Error: {e}")

    # Summary
    if results:
        print("\n" + "=" * 100)
        print("SUMMARY - Benhamou Model 4")
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

        # Save results
        output_path = PROJECT_ROOT / 'strategies/akf/results/model4_results.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
