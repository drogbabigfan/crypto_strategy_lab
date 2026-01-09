#!/usr/bin/env python3
"""
AKF V4 Strategy with Go Backtester

V3.5에서 발전한 최종 버전:
- 미래참조 완전 제거 (horizon 지연 적용)
- 롱/숏 분리 MAE 및 K 계산
- 로그 스케일 통일 (가격 기반 계산 전체)
- 가격 변동성 기반 Dynamic K (P99)

아키텍처:
- Python: 모든 시그널 생성 (진입 + 청산)
- Go: custom_stop 모드로 시그널 실행 (1=롱, -1=숏, 0=청산)
- 정확한 mark-to-market equity curve
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "research/wfa"))

import numpy as np
import pandas as pd
from numba import njit

from research.features.kalman import calculate_adaptive_kalman
from go_bridge import GoBridge, BacktestConfig

DATA_ROOT = PROJECT_ROOT / "etl/data"
GO_BACKTESTER_PATH = PROJECT_ROOT / "etl/bin/backtester"

# Full universe
ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "LTCUSDT"]
PORTFOLIO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"]


# ============================================================================
# V3.5 Parameters
# ============================================================================
V4_PARAMS = {
    # Entry
    "entry_z": 2.0,
    "unc_pct_max": 0.5,
    "warmup": 210,

    # Trail Stop - Rolling K (가격 변동성 기반)
    "trail_k_horizon": 6,      # K 계산 horizon (MAE와 동일)
    "trail_k_window": 180,     # Rolling window
    "trail_k_quantile": 0.99,  # P99

    # Hard Stop (Rolling MAE)
    "mae_horizon": 6,      # 진입 후 6봉 (1일) 동안의 MAE
    "mae_window": 180,     # 과거 180봉 (30일) 참고
    "mae_quantile": 0.99,  # P99

    # Innovation Breaker (v_ratio 기반 동적)
    "v_ratio_threshold": 1.0,
    "innov_base_mult": 2.5,
    "innov_mult_min": 1.5,
    "innov_mult_max": 4.0,

    # Dynamic Sizing (Gaussian)
    "risk_target": 0.03,        # 3%
    "gauss_max_mult": 2.0,      # 최대 배율
    "gauss_sigma": 0.5,         # Gaussian sigma
    "size_min": 0.1,
    "size_max": 3.0,

    # Feature calculation
    "rv_window": 42,
    "baseline_window": 120,
    "resid_window": 180,

    # Intensity Filter (Signal Exit용)
    "intensity_threshold": 4.0,
    "intensity_baseline_window": 20,
}


# ============================================================================
# Data Loading
# ============================================================================
def load_data(symbol: str, bar_size: int = 6):
    """Load feature data for a symbol."""
    data_dir = DATA_ROOT / f"features-{bar_size}/futures/{symbol}"
    dfs = []
    for year in range(2020, 2026):
        for month in range(1, 13):
            path = data_dir / f"{symbol}-features-{year}-{month:02d}.parquet"
            if path.exists():
                dfs.append(pd.read_parquet(path))
    return pd.concat(dfs, ignore_index=True) if dfs else None


# ============================================================================
# Rolling MAE (Maximum Adverse Excursion) - Long/Short 분리, 미래참조 방지
# ============================================================================
def calculate_rolling_mae(
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    horizon: int = 6,
    window: int = 180,
    quantile: float = 0.95
) -> tuple:
    """
    Rolling MAE (Maximum Adverse Excursion) 계산 - Long/Short 분리

    Long MAE: (Open_t - min(Low_{t+1:t+H})) / Open_t  (하방 리스크)
    Short MAE: (max(High_{t+1:t+H}) - Open_t) / Open_t  (상방 리스크)

    미래참조 방지: t 시점에서는 mae[t-horizon-1] 까지만 사용 가능
    → rolling_mae[t] = percentile(mae[t-window-horizon : t-horizon])
    """
    n = len(open_prices)
    mae_long = np.full(n, np.nan)
    mae_short = np.full(n, np.nan)

    # 각 시점에서 horizon 봉 후의 MAE 계산
    for t in range(n - horizon):
        entry_price = open_prices[t]
        if entry_price <= 0 or np.isnan(entry_price):
            continue

        future_lows = low_prices[t+1:t+1+horizon]
        future_highs = high_prices[t+1:t+1+horizon]

        if len(future_lows) == 0 or np.any(np.isnan(future_lows)):
            continue
        if len(future_highs) == 0 or np.any(np.isnan(future_highs)):
            continue

        min_low = np.min(future_lows)
        max_high = np.max(future_highs)

        # 로그 스케일 (hard stop이 exp(-mae) 형태이므로)
        mae_long[t] = np.log(entry_price / min_low)    # ln(entry/min_low) > 0
        mae_short[t] = np.log(max_high / entry_price)  # ln(max_high/entry) > 0

    rolling_mae_long = np.full(n, np.nan)
    rolling_mae_short = np.full(n, np.nan)

    # 미래참조 방지: t 시점에서 사용 가능한 MAE는 t-horizon-1까지
    # mae[t-horizon]은 prices[t-horizon+1:t+1]을 사용 → t 시점 포함!
    # mae[t-horizon-1]은 prices[t-horizon:t]를 사용 → t-1까지만 (OK)
    min_start = window + horizon + 1

    for t in range(min_start, n):
        # t-horizon-1 까지의 MAE만 사용 (미래참조 방지)
        start_idx = t - window - horizon - 1
        end_idx = t - horizon - 1
        if start_idx < 0:
            start_idx = 0

        past_mae_long = mae_long[start_idx:end_idx]
        past_mae_short = mae_short[start_idx:end_idx]

        valid_long = past_mae_long[~np.isnan(past_mae_long)]
        valid_short = past_mae_short[~np.isnan(past_mae_short)]

        if len(valid_long) >= 10:
            rolling_mae_long[t] = np.percentile(valid_long, quantile * 100)
        if len(valid_short) >= 10:
            rolling_mae_short[t] = np.percentile(valid_short, quantile * 100)

    return rolling_mae_long, rolling_mae_short


# ============================================================================
# Rolling K (Trail Stop 계수) - 가격 변동성 기반, 미래참조 방지
# ============================================================================
def calculate_rolling_k(
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    sqrt_s_risk: np.ndarray,
    horizon: int = 6,
    window: int = 180,
    quantile: float = 0.95
) -> tuple:
    """
    Rolling K 계산 - Long/Short 분리, 미래참조 방지

    K_t = MAE_t / avg(sqrt(S_risk_{t+1:t+H}))

    Long K: 하방 MAE 기반
    Short K: 상방 MAE 기반

    미래참조 방지: t 시점에서는 k[t-horizon-1] 까지만 사용 가능
    """
    n = len(open_prices)
    k_long = np.full(n, np.nan)
    k_short = np.full(n, np.nan)

    # 각 시점에서 horizon 봉 후의 K 계산
    for t in range(n - horizon):
        entry_price = open_prices[t]
        if entry_price <= 0 or np.isnan(entry_price):
            continue

        future_lows = low_prices[t+1:t+1+horizon]
        future_highs = high_prices[t+1:t+1+horizon]
        future_sqrt_s = sqrt_s_risk[t+1:t+1+horizon]

        if len(future_lows) == 0 or np.any(np.isnan(future_lows)):
            continue
        if len(future_highs) == 0 or np.any(np.isnan(future_highs)):
            continue
        if len(future_sqrt_s) == 0 or np.any(np.isnan(future_sqrt_s)):
            continue

        min_low = np.min(future_lows)
        max_high = np.max(future_highs)
        avg_sqrt_s = np.mean(future_sqrt_s)

        if avg_sqrt_s > 1e-10:
            # 로그 스케일 (trail stop이 exp(-k * sqrt(S)) 형태이므로)
            log_mae_long = np.log(entry_price / min_low)   # ln(entry/min_low)
            log_mae_short = np.log(max_high / entry_price)  # ln(max_high/entry)
            k_long[t] = log_mae_long / avg_sqrt_s
            k_short[t] = log_mae_short / avg_sqrt_s

    rolling_k_long = np.full(n, np.nan)
    rolling_k_short = np.full(n, np.nan)

    # 미래참조 방지
    min_start = window + horizon + 1

    for t in range(min_start, n):
        start_idx = t - window - horizon - 1
        end_idx = t - horizon - 1
        if start_idx < 0:
            start_idx = 0

        past_k_long = k_long[start_idx:end_idx]
        past_k_short = k_short[start_idx:end_idx]

        valid_long = past_k_long[~np.isnan(past_k_long)]
        valid_short = past_k_short[~np.isnan(past_k_short)]

        if len(valid_long) >= 10:
            rolling_k_long[t] = np.percentile(valid_long, quantile * 100)
        if len(valid_short) >= 10:
            rolling_k_short[t] = np.percentile(valid_short, quantile * 100)

    return rolling_k_long, rolling_k_short


# ============================================================================
# Feature Calculation
# ============================================================================
def calculate_features(df_kf: pd.DataFrame, params: dict = None):
    """Calculate trading features from Kalman filter output."""
    if params is None:
        params = V4_PARAMS

    close = df_kf["close"].values
    kf_trend = df_kf["kf_trend"].values
    kf_uncertainty = df_kf["kf_uncertainty"].values
    velocity = df_kf["kf_velocity"].values

    # 1. Hybrid Volatility = max(sqrt(uncertainty), RV)
    log_returns = np.log(close[1:] / close[:-1])
    log_returns = np.concatenate([[0], log_returns])
    rv = pd.Series(log_returns).rolling(window=params["rv_window"], min_periods=10).std().values
    sqrt_unc = np.sqrt(kf_uncertainty)
    sigma_hybrid = np.maximum(sqrt_unc, rv)

    # 2. V-Ratio
    baseline = pd.Series(sigma_hybrid).rolling(window=params["baseline_window"], min_periods=30).mean().values
    v_ratio = sigma_hybrid / (baseline + 1e-10)

    # 3. Velocity Z-score
    vel_series = pd.Series(velocity)
    vel_mean = vel_series.rolling(window=42, min_periods=5).mean()
    vel_std = vel_series.rolling(window=42, min_periods=5).std()
    vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

    # 4. Uncertainty Percentile
    unc_pct = (
        pd.Series(kf_uncertainty)
        .rolling(window=210, min_periods=20)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        .fillna(0.5)
        .values
    )

    # 5. Residual Std
    log_resid = np.log(close) - np.log(kf_trend)
    resid_std = pd.Series(log_resid).rolling(window=params["resid_window"], min_periods=30).std().fillna(0.01).values

    return {
        "sigma_hybrid": sigma_hybrid,
        "v_ratio": v_ratio,
        "vel_zscore": vel_zscore,
        "unc_pct": unc_pct,
        "resid_std": resid_std,
        "kf_trend": kf_trend,
    }


def calculate_intensity(df: pd.DataFrame, baseline_window: int = 20):
    """Calculate Flow Intensity."""
    duration = np.exp(df["log_duration"].values)
    baseline_duration = pd.Series(duration).rolling(window=baseline_window, min_periods=5).mean().values
    epsilon = 1e-6
    intensity = baseline_duration / (duration + epsilon)
    return intensity


# ============================================================================
# Position Sizing (Gaussian)
# ============================================================================
@njit
def calculate_position_size(
    unc_pct: float,
    stop_distance: float,
    risk_target: float,
    gauss_max_mult: float,
    gauss_sigma: float,
    size_min: float,
    size_max: float
):
    """Gaussian Position Sizing with MAE-based stop distance."""
    base_size = risk_target / (stop_distance + 1e-10)
    gauss_scale = gauss_max_mult * np.exp(-unc_pct**2 / (2 * gauss_sigma**2))
    final_size = base_size * gauss_scale
    final_size = max(size_min, min(size_max, final_size))
    return final_size


# ============================================================================
# Signal Generation with Rolling K (가격 변동성 기반)
# ============================================================================
@njit
def generate_signals(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    kf_trend: np.ndarray,
    sqrt_s_risk: np.ndarray,
    rolling_mae_long: np.ndarray,
    rolling_mae_short: np.ndarray,
    rolling_k_long: np.ndarray,
    rolling_k_short: np.ndarray,
    vel_zscore: np.ndarray,
    unc_pct: np.ndarray,
    v_ratio: np.ndarray,
    resid_std: np.ndarray,
    intensity: np.ndarray,
    entry_z: float,
    unc_pct_max: float,
    v_ratio_threshold: float,
    intensity_threshold: float,
    innov_base_mult: float,
    innov_mult_min: float,
    innov_mult_max: float,
    risk_target: float,
    gauss_max_mult: float,
    gauss_sigma: float,
    size_min: float,
    size_max: float,
    warmup: int,
):
    """
    V3.5 Signal Generation with Rolling K (가격 변동성 기반)

    Rolling K: 과거 가격 변동성 기반 P95 (미래참조 방지)
    - Long: rolling_k_long (하방 MAE / avg(sqrt(S)))
    - Short: rolling_k_short (상방 MAE / avg(sqrt(S)))

    Trail Stop: highest × exp(-k × sqrt(S_risk))
    Hard Stop: entry × (1 - rolling_mae)
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    position_sizes = np.zeros(n, dtype=np.float64)
    stop_prices = np.zeros(n, dtype=np.float64)
    used_k = np.zeros(n, dtype=np.float64)

    # Trading state
    position = 0
    entry_price = 0.0
    current_size = 0.0
    highest = 0.0
    lowest = np.inf
    prev_stop = 0.0
    hard_stop = 0.0
    entry_mae = 0.0
    current_trail_k = 0.0

    for i in range(warmup, n):
        # Innovation mult
        innov_mult = innov_base_mult / (v_ratio[i] + 1e-10)
        innov_mult = max(innov_mult_min, min(innov_mult_max, innov_mult))
        log_resid = np.log(close[i]) - np.log(kf_trend[i])
        stop_dist = sqrt_s_risk[i]

        if position == 0:
            # Check entry conditions + rolling K/MAE availability
            long_signal = (vel_zscore[i] > entry_z and unc_pct[i] < unc_pct_max
                          and not np.isnan(rolling_k_long[i]) and not np.isnan(rolling_mae_long[i]))
            short_signal = (vel_zscore[i] < -entry_z and unc_pct[i] < unc_pct_max
                           and not np.isnan(rolling_k_short[i]) and not np.isnan(rolling_mae_short[i]))

            if long_signal:
                position = 1
                entry_price = close[i]
                highest = high[i]
                entry_mae = rolling_mae_long[i]
                current_trail_k = rolling_k_long[i]

                # Position sizing: mae를 raw 퍼센트로 변환 (1 - exp(-mae_log))
                mae_pct = 1.0 - np.exp(-entry_mae)
                current_size = calculate_position_size(
                    unc_pct[i], mae_pct, risk_target,
                    gauss_max_mult, gauss_sigma, size_min, size_max
                )

                prev_stop = highest * np.exp(-current_trail_k * stop_dist)
                hard_stop = entry_price * np.exp(-entry_mae)

                signals[i] = 1
                position_sizes[i] = current_size
                stop_prices[i] = max(prev_stop, hard_stop)
                used_k[i] = current_trail_k

            elif short_signal:
                position = -1
                entry_price = close[i]
                lowest = low[i]
                entry_mae = rolling_mae_short[i]
                current_trail_k = rolling_k_short[i]

                # Position sizing: mae를 raw 퍼센트로 변환 (exp(mae_log) - 1)
                mae_pct = np.exp(entry_mae) - 1.0
                current_size = calculate_position_size(
                    unc_pct[i], mae_pct, risk_target,
                    gauss_max_mult, gauss_sigma, size_min, size_max
                )

                prev_stop = lowest * np.exp(current_trail_k * stop_dist)
                hard_stop = entry_price * np.exp(entry_mae)

                signals[i] = -1
                position_sizes[i] = current_size
                stop_prices[i] = min(prev_stop, hard_stop)
                used_k[i] = current_trail_k

        elif position == 1:  # LONG
            if high[i] > highest:
                highest = high[i]

            new_stop = highest * np.exp(-current_trail_k * stop_dist)
            trail_stop = max(new_stop, prev_stop)
            prev_stop = trail_stop

            # Exit checks
            exited = False
            exit_price = 0.0

            if low[i] < hard_stop:
                exited = True
                exit_price = hard_stop
            elif low[i] < trail_stop:
                exited = True
                exit_price = trail_stop
            elif v_ratio[i] >= v_ratio_threshold and log_resid < -innov_mult * resid_std[i]:
                exited = True
            elif v_ratio[i] < v_ratio_threshold and vel_zscore[i] < -entry_z and intensity[i] < intensity_threshold:
                exited = True

            if exited:
                signals[i] = 0
                if exit_price > 0:
                    stop_prices[i] = exit_price
                position = 0
                continue

            # Still in long position
            signals[i] = 1
            position_sizes[i] = current_size
            stop_prices[i] = 0
            used_k[i] = current_trail_k

        elif position == -1:  # SHORT
            if low[i] < lowest:
                lowest = low[i]

            new_stop = lowest * np.exp(current_trail_k * stop_dist)
            trail_stop = min(new_stop, prev_stop)
            prev_stop = trail_stop

            # Exit checks
            exited = False
            exit_price = 0.0

            if high[i] > hard_stop:
                exited = True
                exit_price = hard_stop
            elif high[i] > trail_stop:
                exited = True
                exit_price = trail_stop
            elif v_ratio[i] >= v_ratio_threshold and log_resid > innov_mult * resid_std[i]:
                exited = True
            elif v_ratio[i] < v_ratio_threshold and vel_zscore[i] > entry_z and intensity[i] < intensity_threshold:
                exited = True

            if exited:
                signals[i] = 0
                if exit_price > 0:
                    stop_prices[i] = exit_price
                position = 0
                continue

            # Still in short position
            signals[i] = -1
            position_sizes[i] = current_size
            stop_prices[i] = 1e18
            used_k[i] = current_trail_k

    return signals, position_sizes, stop_prices, used_k


# ============================================================================
# Prepare Features for Go Backtester
# ============================================================================
def prepare_go_features(df: pd.DataFrame) -> np.ndarray:
    """
    Prepare features array for Go backtester.

    Expected columns: timestamp, open, high, high_time, low, low_time, close, volume, realized_vol
    """
    n = len(df)
    features = np.zeros((n, 9))

    # Helper to get column values
    def get_col(name, default=None):
        if name in df.columns:
            return df[name].values
        return default

    # Get timestamp (ms)
    ts = get_col("timestamp") or get_col("open_time")
    if ts is None:
        ts = np.arange(n, dtype=np.int64) * 60000
    features[:, 0] = ts

    features[:, 1] = df["open"].values
    features[:, 2] = df["high"].values

    # high_time - use midpoint of bar if not available
    high_time = get_col("high_time")
    if high_time is not None:
        features[:, 3] = high_time
    else:
        features[:, 3] = features[:, 0] + 30000

    features[:, 4] = df["low"].values

    # low_time
    low_time = get_col("low_time")
    if low_time is not None:
        features[:, 5] = low_time
    else:
        features[:, 5] = features[:, 0] + 45000

    features[:, 6] = df["close"].values

    # volume
    vol = get_col("dollar_volume") or get_col("volume")
    if vol is not None:
        features[:, 7] = vol
    else:
        features[:, 7] = np.ones(n) * 1e6

    # realized_vol - use log returns std if not available
    if "realized_vol" in df.columns:
        features[:, 8] = df["realized_vol"].values
    else:
        log_ret = np.log(df["close"].values[1:] / df["close"].values[:-1])
        rv = pd.Series(np.concatenate([[0], log_ret])).rolling(20, min_periods=5).std().values
        features[:, 8] = np.nan_to_num(rv, nan=0.02)

    return features


# ============================================================================
# Run Backtest with Go Backtester
# ============================================================================
def run_backtest_go(df: pd.DataFrame, params: dict = None):
    """Run backtest using Go backtester."""
    if params is None:
        params = V4_PARAMS

    # Ensure df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected DataFrame, got {type(df)}")

    # Calculate Kalman filter
    df_kf = calculate_adaptive_kalman(df)

    # Ensure df_kf is still a DataFrame
    if not isinstance(df_kf, pd.DataFrame):
        raise ValueError(f"calculate_adaptive_kalman returned {type(df_kf)}, expected DataFrame")
    features = calculate_features(df_kf, params)
    intensity = calculate_intensity(df, params["intensity_baseline_window"])

    # Get S_risk from Kalman filter (Dual-Eye)
    sqrt_s_risk = np.sqrt(df_kf["kf_innovation_cov_risk"].values)

    # Calculate Rolling MAE for Hard Stop (Long/Short 분리, 미래참조 방지)
    rolling_mae_long, rolling_mae_short = calculate_rolling_mae(
        open_prices=df_kf["open"].values,
        high_prices=df_kf["high"].values,
        low_prices=df_kf["low"].values,
        horizon=params["mae_horizon"],
        window=params["mae_window"],
        quantile=params["mae_quantile"]
    )

    # Calculate Rolling K for Trail Stop (가격 변동성 기반, 미래참조 방지)
    rolling_k_long, rolling_k_short = calculate_rolling_k(
        open_prices=df_kf["open"].values,
        high_prices=df_kf["high"].values,
        low_prices=df_kf["low"].values,
        sqrt_s_risk=sqrt_s_risk,
        horizon=params["trail_k_horizon"],
        window=params["trail_k_window"],
        quantile=params["trail_k_quantile"]
    )

    # Generate signals with Rolling K
    signals, position_sizes, stop_prices, used_k = generate_signals(
        close=df_kf["close"].values,
        high=df_kf["high"].values,
        low=df_kf["low"].values,
        kf_trend=features["kf_trend"],
        sqrt_s_risk=sqrt_s_risk,
        rolling_mae_long=rolling_mae_long,
        rolling_mae_short=rolling_mae_short,
        rolling_k_long=rolling_k_long,
        rolling_k_short=rolling_k_short,
        vel_zscore=features["vel_zscore"],
        unc_pct=features["unc_pct"],
        v_ratio=features["v_ratio"],
        resid_std=features["resid_std"],
        intensity=intensity,
        entry_z=params["entry_z"],
        unc_pct_max=params["unc_pct_max"],
        v_ratio_threshold=params["v_ratio_threshold"],
        intensity_threshold=params["intensity_threshold"],
        innov_base_mult=params["innov_base_mult"],
        innov_mult_min=params["innov_mult_min"],
        innov_mult_max=params["innov_mult_max"],
        risk_target=params["risk_target"],
        gauss_max_mult=params["gauss_max_mult"],
        gauss_sigma=params["gauss_sigma"],
        size_min=params["size_min"],
        size_max=params["size_max"],
        warmup=params["warmup"],
    )

    # Prepare features for Go backtester
    go_features = prepare_go_features(df_kf)

    # Get timestamps
    if "timestamp" in df_kf.columns:
        timestamps = df_kf["timestamp"].values.astype(np.int64)
    elif "open_time" in df_kf.columns:
        timestamps = df_kf["open_time"].values.astype(np.int64)
    else:
        timestamps = np.arange(len(df_kf), dtype=np.int64) * 60000

    # Initialize Go bridge
    bridge = GoBridge(str(GO_BACKTESTER_PATH))

    if not bridge.is_available():
        raise RuntimeError(f"Go backtester not found at {GO_BACKTESTER_PATH}")

    # Configure backtest - custom_stop mode (neutral signal triggers exit)
    # Python handles ALL exits: stops, innovation breaker, signal exit
    # Go just follows signals: 1=long, -1=short, 0=exit
    config = BacktestConfig(
        exit_mode="custom_stop",  # Neutral signal (0) triggers exit in this mode
        initial_capital=100000.0,
        risk_per_trade=1.0,  # Size already includes risk scaling
        compounding=True,
        base_fee=0.001,
        base_slippage=0.0001,
        max_leverage=10.0,
    )

    # Run backtest
    # stop_prices are 0 for exit bars (Python already exited, no stop needed)
    result = bridge.run_backtest_with_data(
        signals=signals,
        features=go_features,
        config=config,
        timestamps=timestamps,
        sizes=position_sizes,
        stop_prices=stop_prices,
        include_equity=True,
    )

    # Calculate additional metrics
    n_bars = len(df_kf) - params["warmup"]
    bars_per_year = 6 * 365  # 6 bars per day

    years = n_bars / bars_per_year
    if years > 0 and result.total_pnl > -1:
        cagr = ((1 + result.total_pnl) ** (1 / years)) - 1
    else:
        cagr = 0.0

    active_sizes = position_sizes[position_sizes > 0]
    avg_size = np.mean(active_sizes) if len(active_sizes) > 0 else 0
    min_size = np.min(active_sizes) if len(active_sizes) > 0 else 0
    max_size = np.max(active_sizes) if len(active_sizes) > 0 else 0

    # Calculate K statistics
    active_k = used_k[used_k > 0]
    avg_k = np.mean(active_k) if len(active_k) > 0 else 0
    min_k = np.min(active_k) if len(active_k) > 0 else 0
    max_k = np.max(active_k) if len(active_k) > 0 else 0

    # Calculate bar returns for portfolio
    equity = np.array(result.equity_curve) if result.equity_curve else np.ones(len(df_kf)) * 100000
    bar_returns = np.zeros(len(equity))
    bar_returns[1:] = np.diff(equity) / (equity[:-1] + 1e-10)

    return {
        "total_trades": result.total_trades,
        "total_pnl": result.total_pnl,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "win_rate": result.win_rate,
        "avg_pnl": result.avg_pnl,
        "avg_hold_bars": result.avg_hold_bars,
        "tp_count": result.tp_count,
        "sl_count": result.sl_count,
        "timeout_count": result.timeout_count,
        "cagr": cagr,
        "avg_size": avg_size,
        "min_size": min_size,
        "max_size": max_size,
        "avg_k": avg_k,
        "min_k": min_k,
        "max_k": max_k,
        "equity_curve": equity,
        "bar_returns": bar_returns,
    }


# ============================================================================
# Portfolio Metrics
# ============================================================================
def calculate_portfolio_metrics(individual_results: dict):
    """Calculate combined portfolio metrics."""
    min_len = min(len(m["bar_returns"]) for m in individual_results.values())

    combined_returns = np.zeros(min_len)
    for symbol, m in individual_results.items():
        combined_returns += m["bar_returns"][:min_len]

    initial_capital = 100000.0
    combined_equity = np.zeros(min_len)
    combined_equity[0] = initial_capital

    for i in range(1, min_len):
        combined_equity[i] = combined_equity[i-1] * (1 + combined_returns[i])

    total_pnl = (combined_equity[-1] - initial_capital) / initial_capital

    running_max = np.maximum.accumulate(combined_equity)
    drawdown = (combined_equity - running_max) / running_max
    max_dd = np.abs(np.min(drawdown))

    daily_returns = []
    for i in range(0, len(combined_returns), 6):
        chunk = combined_returns[i:i+6]
        if len(chunk) > 0:
            daily_returns.append(np.sum(chunk))

    if len(daily_returns) > 1:
        sharpe = (np.mean(daily_returns) / (np.std(daily_returns) + 1e-10)) * np.sqrt(252)
    else:
        sharpe = 0.0

    n_years = min_len / (6 * 365)
    if n_years > 0 and combined_equity[-1] > 0:
        cagr = (combined_equity[-1] / initial_capital) ** (1 / n_years) - 1
    else:
        cagr = 0.0

    total_avg_size = sum(m["avg_size"] for m in individual_results.values())

    return {
        "total_pnl": total_pnl,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "avg_leverage": total_avg_size,
        "cagr_mdd": cagr / max_dd if max_dd > 0 else 0,
        "equity_curve": combined_equity,
    }


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 120)
    print("AKF V4 Strategy with Go Backtester")
    print("=" * 120)

    print("\n[Parameters]")
    print(f"  Entry Z: {V4_PARAMS['entry_z']}")
    print(f"  Trail Stop: Rolling K P{V4_PARAMS['trail_k_quantile']*100:.0f} (H={V4_PARAMS['trail_k_horizon']}, W={V4_PARAMS['trail_k_window']})")
    print(f"  Hard Stop: MAE P{V4_PARAMS['mae_quantile']*100:.0f} (H={V4_PARAMS['mae_horizon']}, W={V4_PARAMS['mae_window']})")
    print(f"  Risk Target: {V4_PARAMS['risk_target']*100}%")
    print(f"  Gaussian: M={V4_PARAMS['gauss_max_mult']}, sigma={V4_PARAMS['gauss_sigma']}")
    print(f"  Size Range: [{V4_PARAMS['size_min']}, {V4_PARAMS['size_max']}]")

    print(f"\n[Go Backtester: {GO_BACKTESTER_PATH}]")

    # Load data for all symbols
    print("\nLoading data for full universe...")
    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars")
        else:
            print(f"  {symbol}: NO DATA")

    print("\n" + "-" * 130)
    print(f"{'Symbol':<10} {'Sharpe':<10} {'CAGR':<12} {'MDD':<10} {'CAGR/MDD':<10} {'Trades':<8} {'WinRate':<10} {'AvgK':<8} {'AvgSize':<10} {'SL/TP':<10}")
    print("-" * 130)

    individual_results = {}
    for symbol, df in data.items():
        try:
            metrics = run_backtest_go(df)
            individual_results[symbol] = metrics

            cagr_mdd = metrics["cagr"] / metrics["max_drawdown"] if metrics["max_drawdown"] > 0 else 0
            sl_tp = f"{metrics['sl_count']}/{metrics['tp_count']}"

            print(f"{symbol:<10} {metrics['sharpe_ratio']:<10.2f} {metrics['cagr']*100:<11.1f}% "
                  f"{metrics['max_drawdown']*100:<9.1f}% {cagr_mdd:<10.2f} "
                  f"{metrics['total_trades']:<8} {metrics['win_rate']*100:<9.1f}% "
                  f"{metrics['avg_k']:<8.1f} {metrics['avg_size']:<10.2f} {sl_tp:<10}")
        except Exception as e:
            print(f"{symbol:<10} ERROR: {str(e)[:80]}")

    print("-" * 130)

    if len(individual_results) >= 4:
        # Portfolio metrics (first 4 symbols)
        portfolio_symbols = [s for s in PORTFOLIO_SYMBOLS if s in individual_results]
        if len(portfolio_symbols) >= 4:
            portfolio_results = {s: individual_results[s] for s in portfolio_symbols}
            portfolio = calculate_portfolio_metrics(portfolio_results)

            print(f"\n[4-Asset Portfolio: {', '.join(portfolio_symbols)}]")
            print(f"  CAGR:       {portfolio['cagr']*100:.1f}%")
            print(f"  MDD:        {portfolio['max_drawdown']*100:.1f}%")
            print(f"  CAGR/MDD:   {portfolio['cagr_mdd']:.2f}")
            print(f"  Sharpe:     {portfolio['sharpe']:.2f}")
            print(f"  Leverage:   {portfolio['avg_leverage']:.2f}x")

    # Full universe portfolio
    if len(individual_results) > 4:
        full_portfolio = calculate_portfolio_metrics(individual_results)
        print(f"\n[Full Universe Portfolio: {len(individual_results)} assets]")
        print(f"  CAGR:       {full_portfolio['cagr']*100:.1f}%")
        print(f"  MDD:        {full_portfolio['max_drawdown']*100:.1f}%")
        print(f"  CAGR/MDD:   {full_portfolio['cagr_mdd']:.2f}")
        print(f"  Sharpe:     {full_portfolio['sharpe']:.2f}")
        print(f"  Leverage:   {full_portfolio['avg_leverage']:.2f}x")


if __name__ == "__main__":
    main()
