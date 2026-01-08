#!/usr/bin/env python3
"""
Test all AKF strategies with fixed look-ahead bias backtester.

Tests V1, V2, V3 strategies with early stop at -50% loss.
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
from strategies.akf.strategies_v2 import AKFStrategiesV2, AKFStrategyConfigV2
from strategies.akf.strategies_v3 import AKFStrategiesV3, AKFStrategyConfigV3


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


def generate_signals(df_kf: pd.DataFrame, strategy: str, version: int, k: float) -> pd.DataFrame:
    """Generate signals for a given strategy."""
    if version == 1:
        akf = AKFStrategies(AKFStrategyConfig(k=k))
        if strategy == 'strategy_db':
            return akf.strategy_db(df_kf.copy(), k=k)
        elif strategy == 'strategy_benhamou':
            return akf.strategy_benhamou(df_kf.copy(), k=k)
        elif strategy == 'strategy_hybrid':
            return akf.strategy_hybrid(df_kf.copy(), k=k)
    elif version == 2:
        config = AKFStrategyConfigV2(db_k=k, benhamou_k=k, fusion_k=k)
        akf = AKFStrategiesV2(config)
        if strategy == 'strategy_db':
            return akf.run_strategy_db(df_kf.copy(), k=k)
        elif strategy == 'strategy_benhamou':
            return akf.run_strategy_benhamou(df_kf.copy(), k=k)
        elif strategy == 'strategy_triple_fusion':
            return akf.run_strategy_triple_fusion(df_kf.copy(), k=k)
        elif strategy == 'strategy_velocity_scalp':
            return akf.run_strategy_velocity_scalp(df_kf.copy())
    elif version == 3:
        akf = AKFStrategiesV3(AKFStrategyConfigV3(k=k))
        if strategy == 'strategy_db_reversal':
            return akf.strategy_db_reversal(df_kf.copy(), k=k)
        elif strategy == 'strategy_benhamou_reversal':
            return akf.strategy_benhamou_reversal(df_kf.copy(), k=k)
        elif strategy == 'strategy_hybrid_reversal':
            return akf.strategy_hybrid_reversal(df_kf.copy(), k=k)

    return None


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

    # Test configurations: (strategy, version, k, name)
    configs = [
        # V1 strategies
        ('strategy_db', 1, 2.0, 'V1 DB k=2.0'),
        ('strategy_db', 1, 1.5, 'V1 DB k=1.5'),
        ('strategy_benhamou', 1, 2.0, 'V1 Benhamou k=2.0'),
        ('strategy_hybrid', 1, 2.0, 'V1 Hybrid k=2.0'),

        # V2 strategies
        ('strategy_db', 2, 2.0, 'V2 DB k=2.0'),
        ('strategy_benhamou', 2, 2.0, 'V2 Benhamou k=2.0'),
        ('strategy_triple_fusion', 2, 2.0, 'V2 TripleFusion k=2.0'),
        ('strategy_velocity_scalp', 2, 2.0, 'V2 VelocityScalp'),

        # V3 strategies (reversal/always-in-market)
        ('strategy_db_reversal', 3, 2.0, 'V3 DB Reversal k=2.0'),
        ('strategy_benhamou_reversal', 3, 2.0, 'V3 Benhamou Reversal k=2.0'),
        ('strategy_hybrid_reversal', 3, 2.0, 'V3 Hybrid Reversal k=2.0'),
    ]

    results = []

    print("\n" + "=" * 80)
    print("TESTING ALL STRATEGIES (with -50% early stop)")
    print("=" * 80)

    for strategy, version, k, name in configs:
        print(f"\n>>> {name}...")

        try:
            df_signals = generate_signals(df_kf, strategy, version, k)

            if df_signals is None:
                print(f"  Failed to generate signals")
                continue

            # Handle position/signal column naming
            if 'position' in df_signals.columns and 'signal' not in df_signals.columns:
                df_signals['signal'] = df_signals['position']

            metrics = run_backtest(df_signals)

            if metrics:
                result = {
                    'name': name,
                    'strategy': strategy,
                    'version': version,
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

                # Check if early stopped
                status = "STOPPED" if result['total_pnl'] <= -50 else "OK"
                print(f"  [{status}] Trades: {result['trades']}, PnL: {result['total_pnl']:.1f}%, Sharpe: {result['sharpe']:.2f}")
            else:
                print(f"  Failed to run backtest")

        except Exception as e:
            print(f"  Error: {e}")

    # Summary table
    if results:
        print("\n" + "=" * 100)
        print("SUMMARY (Look-ahead bias FIXED)")
        print("=" * 100)

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('total_pnl', ascending=False)

        print(df_results[['name', 'trades', 'win_rate', 'avg_pnl', 'total_pnl',
                          'sharpe', 'mdd', 'profit_factor', 'avg_hold']].to_string(index=False))

        # Stats
        profitable = (df_results['total_pnl'] > 0).sum()
        stopped = (df_results['total_pnl'] <= -50).sum()

        print(f"\n수익 전략: {profitable}/{len(results)}")
        print(f"Early Stop (-50%): {stopped}/{len(results)}")

        # Save results
        output_path = PROJECT_ROOT / 'strategies/akf/results/all_strategies_fixed.csv'
        df_results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
