#!/usr/bin/env python3
"""
Live Trading Feature Generator (V9)

Dollar bar에서 V9 전략에 필요한 피처를 생성합니다.

생성되는 피처:
- OHLCV 기본 데이터
- Dual Kalman Filter (fast=20, slow=120)
  - fast_velocity, fast_zscore, fast_uncertainty
  - slow_velocity, slow_zscore, slow_uncertainty
  - spread, spread_std
- Duration features
  - log_duration
  - duration_ma (rolling mean, window=50)
- Volatility features
  - parkinson_vol (rolling Parkinson volatility)
  - compression (bar activity factor)
  - kf_innovation_cov_risk (for stop distance)

사용법:
    python scripts/generate_live_trading_features.py
    python scripts/generate_live_trading_features.py --bar-size 6
    python scripts/generate_live_trading_features.py --symbol BTCUSDT
    python scripts/generate_live_trading_features.py --force
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from research.features.kalman import KalmanFeatureGenerator, KalmanConfig

# Paths
ETL_DATA_ROOT = PROJECT_ROOT / "etl/data"

# Symbols to process
ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"]

# V9 Parameters
V9_PARAMS = {
    # Dual Kalman
    "fast_window": 20,
    "slow_window": 120,
    # Duration
    "duration_ma_window": 50,
    # Volatility
    "parkinson_window": 20,
    "compression_window": 100,
    # Spread
    "spread_std_window": 50,
    # Tick (for short signal)
    "tick_p25_window": 100,
}


def load_bars(symbol: str, bar_size: int) -> pd.DataFrame:
    """Load dollar bar data for a symbol."""
    bars_dir = ETL_DATA_ROOT / f"bars-{bar_size}/futures/{symbol}"

    if not bars_dir.exists():
        print(f"  [WARN] Bars directory not found: {bars_dir}")
        return None

    parquet_files = sorted(bars_dir.glob(f"{symbol}-bars-*.parquet"))

    if not parquet_files:
        print(f"  [WARN] No parquet files found in {bars_dir}")
        return None

    dfs = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            dfs.append(df)
        except Exception as e:
            print(f"  [WARN] Failed to read {pf.name}: {e}")

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)

    # Sort by timestamp
    if "start_time" in df.columns:
        df = df.sort_values("start_time").reset_index(drop=True)
    elif "open_time" in df.columns:
        df = df.sort_values("open_time").reset_index(drop=True)

    return df


def calculate_log_duration(df: pd.DataFrame) -> np.ndarray:
    """Calculate log duration for intensity computation."""
    if "duration" in df.columns:
        duration = df["duration"].values
    elif "start_time" in df.columns and "end_time" in df.columns:
        duration = (df["end_time"] - df["start_time"]).values / 1000.0  # ms to seconds
    elif "start_time" in df.columns:
        # Estimate from time differences
        times = df["start_time"].values
        duration = np.zeros(len(times))
        duration[1:] = np.diff(times) / 1000.0  # ms to seconds
        duration[0] = duration[1] if len(duration) > 1 else 14400  # default 4 hours
    else:
        # Default duration (4 hours for 6 bars/day)
        duration = np.ones(len(df)) * 14400

    # Avoid log(0)
    duration = np.maximum(duration, 1.0)
    return np.log(duration)


def calculate_dual_kalman(
    df: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 120
) -> pd.DataFrame:
    """
    Dual Kalman Filter 계산

    Args:
        df: DataFrame with OHLC data
        fast_window: Fast Kalman window (default 20)
        slow_window: Slow Kalman window (default 120)

    Returns:
        DataFrame with Dual Kalman features:
        - fast_velocity, fast_zscore, fast_uncertainty
        - slow_velocity, slow_zscore, slow_uncertainty
        - spread, spread_std
        - kf_innovation_cov_risk (from fast Kalman, for stop distance)
    """
    # Fast Kalman (window=20)
    fast_kf = KalmanFeatureGenerator(r_window=fast_window, q_window=fast_window)
    df_fast = fast_kf.generate(df)

    # Slow Kalman (window=120)
    slow_kf = KalmanFeatureGenerator(r_window=slow_window, q_window=slow_window)
    df_slow = slow_kf.generate(df)

    # Combine results
    result = df.copy()

    # Fast Kalman features
    result["fast_velocity"] = df_fast["kf_velocity"].values
    result["fast_velocity_var"] = df_fast["kf_velocity_var"].values
    result["fast_zscore"] = df_fast["kf_velocity_zscore"].values
    result["fast_uncertainty"] = df_fast["kf_uncertainty"].values
    result["fast_trend"] = df_fast["kf_trend"].values

    # Slow Kalman features
    result["slow_velocity"] = df_slow["kf_velocity"].values
    result["slow_velocity_var"] = df_slow["kf_velocity_var"].values
    result["slow_zscore"] = df_slow["kf_velocity_zscore"].values
    result["slow_uncertainty"] = df_slow["kf_uncertainty"].values
    result["slow_trend"] = df_slow["kf_trend"].values

    # Spread (fast - slow)
    result["spread"] = result["fast_velocity"] - result["slow_velocity"]
    result["spread_std"] = result["spread"].rolling(
        V9_PARAMS["spread_std_window"], min_periods=10
    ).std()

    # Innovation covariance for stop distance (from fast Kalman)
    result["kf_innovation_cov_risk"] = df_fast["kf_innovation_cov_risk"].values

    return result


def calculate_parkinson_vol(
    high: np.ndarray,
    low: np.ndarray,
    window: int = 20
) -> np.ndarray:
    """
    Parkinson volatility 계산

    Parkinson Volatility: σ = sqrt((1/4ln2) × mean[(ln(H/L))²])
    """
    log_hl = np.log(high / low)
    parkinson_sq = log_hl ** 2 / (4 * np.log(2))
    return pd.Series(np.sqrt(parkinson_sq)).rolling(window, min_periods=5).mean().values


def calculate_compression(
    log_duration: np.ndarray,
    window: int = 100
) -> np.ndarray:
    """
    Compression factor 계산

    compression = current_bars_per_day / rolling_mean(bars_per_day)
    Clipped to [0.5, 2.0]
    """
    duration_hours = np.exp(log_duration) / 3600
    bars_per_day = 24 / duration_hours
    rolling_bars = pd.Series(bars_per_day).rolling(window, min_periods=20).mean().values
    return np.clip(bars_per_day / (rolling_bars + 1e-10), 0.5, 2.0)


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate V9 live trading features from dollar bar data."""

    # Ensure required columns exist
    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # 1. Calculate log_duration
    log_duration = calculate_log_duration(df)

    # 2. Prepare DataFrame with log_duration for Kalman
    df_with_duration = df.copy()
    df_with_duration["log_duration"] = log_duration

    # 3. Apply Dual Kalman Filter
    result = calculate_dual_kalman(
        df_with_duration,
        fast_window=V9_PARAMS["fast_window"],
        slow_window=V9_PARAMS["slow_window"]
    )

    # 4. Duration features
    result["log_duration"] = log_duration
    duration = np.exp(log_duration)
    result["duration"] = duration
    result["duration_ma"] = pd.Series(duration).rolling(
        V9_PARAMS["duration_ma_window"], min_periods=10
    ).mean().values

    # 5. Parkinson volatility
    result["parkinson_vol"] = calculate_parkinson_vol(
        df["high"].values,
        df["low"].values,
        window=V9_PARAMS["parkinson_window"]
    )

    # 6. Compression factor
    result["compression"] = calculate_compression(
        log_duration,
        window=V9_PARAMS["compression_window"]
    )

    # 7. Tick count features (for short signal)
    if "tick_count" in df.columns:
        tick = df["tick_count"].values
    elif "log_tick_count" in df.columns:
        tick = np.exp(df["log_tick_count"].values)
    else:
        # Estimate from volume
        tick = np.ones(len(df)) * 1000  # default

    result["tick_count"] = tick
    result["tick_p25"] = pd.Series(tick).rolling(
        V9_PARAMS["tick_p25_window"], min_periods=20
    ).quantile(0.25).shift(1).values  # shift(1) to prevent lookahead

    # 8. Ensure timestamp columns are preserved
    timestamp_cols = ["start_time", "end_time", "open_time", "close_time", "timestamp"]
    for col in timestamp_cols:
        if col in df.columns and col not in result.columns:
            result[col] = df[col].values

    # 9. Ensure volume columns are preserved
    volume_cols = ["dollar_volume", "volume", "buy_volume", "sell_volume"]
    for col in volume_cols:
        if col in df.columns and col not in result.columns:
            result[col] = df[col].values

    return result


def save_features_by_month(df: pd.DataFrame, symbol: str, bar_size: int, output_dir: Path):
    """Save features split by month."""

    # Determine timestamp column
    ts_col = None
    for col in ["start_time", "open_time", "timestamp"]:
        if col in df.columns:
            ts_col = col
            break

    if ts_col is None:
        # Save as single file if no timestamp
        output_path = output_dir / f"{symbol}-live-features-all.parquet"
        df.to_parquet(output_path, index=False)
        print(f"    Saved: {output_path.name} ({len(df):,} rows)")
        return

    # Convert to datetime
    df["_datetime"] = pd.to_datetime(df[ts_col], unit="ms")
    df["_year"] = df["_datetime"].dt.year
    df["_month"] = df["_datetime"].dt.month

    # Group by year-month and save
    saved_count = 0
    for (year, month), group in df.groupby(["_year", "_month"]):
        # Drop temporary columns
        group = group.drop(columns=["_datetime", "_year", "_month"])

        output_path = output_dir / f"{symbol}-live-features-{year}-{month:02d}.parquet"
        group.to_parquet(output_path, index=False)
        saved_count += 1

    # Clean up temp columns from original df
    df.drop(columns=["_datetime", "_year", "_month"], inplace=True)

    print(f"    Saved {saved_count} monthly files")


def process_symbol(symbol: str, bar_size: int, force: bool = False, output_base: Path = None):
    """Process a single symbol."""
    print(f"\n[{symbol}] Processing...")

    # Load bars
    df = load_bars(symbol, bar_size)
    if df is None or len(df) == 0:
        print(f"  [SKIP] No data available")
        return False

    print(f"  Loaded {len(df):,} bars")

    # Check output directory
    if output_base is None:
        output_base = ETL_DATA_ROOT / f"live-features-{bar_size}"
    output_dir = output_base / f"futures/{symbol}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed (unless force)
    existing_files = list(output_dir.glob(f"{symbol}-live-features-*.parquet"))
    if existing_files and not force:
        print(f"  [SKIP] {len(existing_files)} feature files already exist (use --force to regenerate)")
        return True

    # Generate features
    print(f"  Generating V9 live trading features...")
    try:
        df_features = generate_features(df)
    except Exception as e:
        print(f"  [ERROR] Feature generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"  Generated {len(df_features.columns)} columns")

    # Print feature summary
    print(f"  Key features:")
    for col in ["fast_zscore", "slow_zscore", "duration_ma", "parkinson_vol", "spread"]:
        if col in df_features.columns:
            valid = df_features[col].dropna()
            if len(valid) > 0:
                print(f"    {col}: mean={valid.mean():.4f}, std={valid.std():.4f}")

    # Save
    print(f"  Saving to {output_dir}...")
    save_features_by_month(df_features, symbol, bar_size, output_dir)

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate V9 live trading features from dollar bars")
    parser.add_argument("--bar-size", type=int, default=6, choices=[6, 24],
                        help="Bars per day (6 or 24)")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Process single symbol (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if files exist")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom output directory (default: live-features-{bar_size})")
    args = parser.parse_args()

    # Determine output base directory
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = ETL_DATA_ROOT / f"live-features-{args.bar_size}"

    print("=" * 80)
    print(f"V9 Live Trading Feature Generator")
    print(f"Bar size: {args.bar_size} bars/day")
    print(f"Output: {output_base}/futures/")
    print("=" * 80)
    print(f"\nV9 Parameters:")
    for k, v in V9_PARAMS.items():
        print(f"  {k}: {v}")

    # Determine symbols to process
    if args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = ALL_SYMBOLS

    # Process each symbol
    results = {}
    for symbol in symbols:
        success = process_symbol(symbol, args.bar_size, args.force, output_base)
        results[symbol] = "OK" if success else "FAILED"

    # Summary
    print("\n" + "=" * 80)
    print("Summary:")
    for symbol, status in results.items():
        print(f"  {symbol}: {status}")
    print("=" * 80)


if __name__ == "__main__":
    main()
