#!/usr/bin/env python3
"""
AKF V3.3 Final Strategy (with Dynamic Sizing + Intensity Filter)

=== 진입 ===
- vel_zscore > 2.0 & unc_pct < 0.5

=== 청산 (우선순위) ===
1. Hard Stop: entry × exp(-5 × σ_entry)
2. Trailing Stop: highest × exp(-dynamic_mult × σ) with Ratchet
3. Innovation Breaker: |residual| > innov_mult × resid_std (v_ratio ≥ 1.0)
4. Signal Exit: vel_zscore 반전 (v_ratio < 1.0 AND Intensity < 4.0)
   - Intensity = Baseline_Duration / Current_Duration
   - Flash State (I >= 4.0)에서는 Signal Exit 보류

=== Dynamic Sizing ===
1. Base Size = Risk_Target / Hard_Stop_Distance
   - Risk Target: 3%
   - Hard Stop Distance: 5 × σ_hybrid

2. Confidence Scaling (Sigmoid)
   - M = 1.0 + 0.5 × tanh(2.5 × (0.5 - unc_pct))
   - unc_pct 낮을수록 → 최대 1.5배
   - unc_pct 높을수록 → 최소 0.5배

3. Final Size = clip(Base × M, 0.1, 3.0)

=== 성능 (I<4.0 적용) ===
개별 자산:
- BTC: PnL 489%, Sharpe 1.09, MDD 17.1%
- ETH: PnL 186%, Sharpe 0.91, MDD 23.9%
- XRP: PnL 138%, Sharpe 0.70, MDD 42.3%
- SOL: PnL 413%, Sharpe 0.88, MDD 22.9%

4-Asset Combined (레버리지 누적):
- Total PnL: 7,998%
- CAGR: 79.3%
- Sharpe: 1.01
- MDD: 29.6%
- 평균 레버리지: 2.61x
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from numba import njit

from research.features.kalman import calculate_adaptive_kalman
from strategies.akf_v2.backtest_pyramid import run_pyramid_backtest, PyramidBacktestConfig

DATA_ROOT = PROJECT_ROOT / "etl/data"
ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "LTCUSDT"]
PORTFOLIO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"]


# ============================================================================
# V3.3 Final Parameters
# ============================================================================
V33_PARAMS = {
    # Entry
    "entry_z": 2.0,
    "unc_pct_max": 0.5,
    "warmup": 210,

    # Trailing Stop (v_ratio 기반 동적)
    "trail_base_mult": 5.0,
    "trail_mult_min": 1.25,
    "trail_mult_max": 7.5,

    # Innovation Breaker (v_ratio 기반 동적)
    "v_ratio_threshold": 1.0,
    "innov_base_mult": 2.5,
    "innov_mult_min": 1.5,
    "innov_mult_max": 4.0,

    # Hard Stop
    "hard_stop_mult": 5.0,

    # Dynamic Sizing
    "risk_target": 0.03,        # 3%
    "conf_lambda": 2.5,
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
    data_dir = DATA_ROOT / f"features-{bar_size}/futures/{symbol}"
    dfs = []
    for year in range(2020, 2026):
        for month in range(1, 13):
            path = data_dir / f"{symbol}-features-{year}-{month:02d}.parquet"
            if path.exists():
                dfs.append(pd.read_parquet(path))
    return pd.concat(dfs, ignore_index=True) if dfs else None


# ============================================================================
# Feature Calculation
# ============================================================================
def calculate_features(df_kf: pd.DataFrame, params: dict = None):
    if params is None:
        params = V33_PARAMS

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

    # 2. V-Ratio & Dynamic Trail Mult
    baseline = pd.Series(sigma_hybrid).rolling(window=params["baseline_window"], min_periods=30).mean().values
    v_ratio = sigma_hybrid / (baseline + 1e-10)
    dynamic_mult = np.clip(
        params["trail_base_mult"] / (v_ratio + 1e-10),
        params["trail_mult_min"],
        params["trail_mult_max"]
    )

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
        "dynamic_mult": dynamic_mult,
        "vel_zscore": vel_zscore,
        "unc_pct": unc_pct,
        "resid_std": resid_std,
        "kf_trend": kf_trend,
    }


def calculate_intensity(df: pd.DataFrame, baseline_window: int = 20):
    """
    Flow Intensity 계산

    I = Baseline_Duration / Current_Duration
    - I > 1: High Velocity (바가 빨리 생성)
    - I < 1: Low Velocity (바가 천천히 생성)
    """
    duration = np.exp(df["log_duration"].values)
    baseline_duration = pd.Series(duration).rolling(window=baseline_window, min_periods=5).mean().values
    epsilon = 1e-6
    intensity = baseline_duration / (duration + epsilon)
    return intensity


# ============================================================================
# Position Sizing
# ============================================================================
@njit
def calculate_position_size(
    sigma_hybrid: float,
    unc_pct: float,
    risk_target: float,
    hard_stop_mult: float,
    conf_lambda: float,
    size_min: float,
    size_max: float
):
    """
    Dynamic Position Sizing

    1. Base Size = Risk_Target / Hard_Stop_Distance
    2. Confidence Scaling = 1.0 + 0.5 × tanh(λ × (0.5 - unc_pct))
    3. Final Size = clip(Base × Conf, min, max)
    """
    # Hard Stop Distance
    hard_stop_dist = hard_stop_mult * sigma_hybrid

    # Base Size
    base_size = risk_target / (hard_stop_dist + 1e-10)

    # Confidence Scaling (Sigmoid)
    conf_scale = 1.0 + 0.5 * np.tanh(conf_lambda * (0.5 - unc_pct))

    # Final Size
    final_size = base_size * conf_scale
    final_size = max(size_min, min(size_max, final_size))

    return final_size


# ============================================================================
# Signal Generation
# ============================================================================
@njit
def generate_signals(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    kf_trend: np.ndarray,
    vel_zscore: np.ndarray,
    unc_pct: np.ndarray,
    sigma_hybrid: np.ndarray,
    dynamic_mult: np.ndarray,
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
    hard_stop_mult: float,
    risk_target: float,
    conf_lambda: float,
    size_min: float,
    size_max: float,
    warmup: int,
):
    """V3.3 Signal Generation with Dynamic Sizing + Intensity Filter"""
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    position_sizes = np.zeros(n, dtype=np.float64)
    stop_prices = np.zeros(n, dtype=np.float64)

    position = 0
    entry_price = 0.0
    current_size = 0.0
    highest = 0.0
    lowest = np.inf
    prev_stop = 0.0
    hard_stop = 0.0

    for i in range(warmup, n):
        # Innovation mult (v_ratio 기반 동적)
        innov_mult = innov_base_mult / (v_ratio[i] + 1e-10)
        innov_mult = max(innov_mult_min, min(innov_mult_max, innov_mult))

        log_resid = np.log(close[i]) - np.log(kf_trend[i])

        if position == 0:
            # === LONG ENTRY ===
            if vel_zscore[i] > entry_z and unc_pct[i] < unc_pct_max:
                position = 1
                entry_price = close[i]
                highest = high[i]

                # Dynamic Sizing
                current_size = calculate_position_size(
                    sigma_hybrid[i], unc_pct[i], risk_target,
                    hard_stop_mult, conf_lambda, size_min, size_max
                )

                # Stops
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = highest * np.exp(-dist)
                hard_stop = entry_price * np.exp(-hard_stop_mult * sigma_hybrid[i])

                signals[i] = 1
                position_sizes[i] = current_size
                stop_prices[i] = max(prev_stop, hard_stop)

            # === SHORT ENTRY ===
            elif vel_zscore[i] < -entry_z and unc_pct[i] < unc_pct_max:
                position = -1
                entry_price = close[i]
                lowest = low[i]

                # Dynamic Sizing
                current_size = calculate_position_size(
                    sigma_hybrid[i], unc_pct[i], risk_target,
                    hard_stop_mult, conf_lambda, size_min, size_max
                )

                # Stops
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = lowest * np.exp(dist)
                hard_stop = entry_price * np.exp(hard_stop_mult * sigma_hybrid[i])

                signals[i] = -1
                position_sizes[i] = current_size
                stop_prices[i] = min(prev_stop, hard_stop)

        elif position == 1:  # === LONG POSITION ===
            if high[i] > highest:
                highest = high[i]

            # Trailing Stop
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_stop = highest * np.exp(-dist)
            trail_stop = max(new_stop, prev_stop)  # Ratchet
            prev_stop = trail_stop
            stop_prices[i] = max(trail_stop, hard_stop)

            # Exit checks (우선순위)
            # 1. Hard Stop
            if close[i] < hard_stop:
                position = 0
                continue
            # 2. Trailing Stop
            if close[i] < trail_stop:
                position = 0
                continue
            # 3. Innovation Breaker (v_ratio >= threshold)
            if v_ratio[i] >= v_ratio_threshold:
                if log_resid < -innov_mult * resid_std[i]:
                    position = 0
                    continue
            # 4. Signal Exit (v_ratio < threshold AND intensity < threshold)
            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] < -entry_z:
                    if intensity[i] < intensity_threshold:
                        position = 0
                        continue
                    # else: Flash State → Signal Exit 보류

            signals[i] = 1
            position_sizes[i] = current_size

        elif position == -1:  # === SHORT POSITION ===
            if low[i] < lowest:
                lowest = low[i]

            # Trailing Stop
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_stop = lowest * np.exp(dist)
            trail_stop = min(new_stop, prev_stop)  # Ratchet
            prev_stop = trail_stop
            stop_prices[i] = min(trail_stop, hard_stop)

            # Exit checks (우선순위)
            # 1. Hard Stop
            if close[i] > hard_stop:
                position = 0
                continue
            # 2. Trailing Stop
            if close[i] > trail_stop:
                position = 0
                continue
            # 3. Innovation Breaker (v_ratio >= threshold)
            if v_ratio[i] >= v_ratio_threshold:
                if log_resid > innov_mult * resid_std[i]:
                    position = 0
                    continue
            # 4. Signal Exit (v_ratio < threshold AND intensity < threshold)
            if v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] > entry_z:
                    if intensity[i] < intensity_threshold:
                        position = 0
                        continue
                    # else: Flash State → Signal Exit 보류

            signals[i] = -1
            position_sizes[i] = current_size

    return signals, position_sizes, stop_prices


# ============================================================================
# Backtest Runner
# ============================================================================
def calculate_cagr(total_pnl: float, n_bars: int, bars_per_year: float = 365 * 4):
    years = n_bars / bars_per_year
    if years <= 0 or total_pnl <= -1:
        return 0.0
    return ((1 + total_pnl) ** (1 / years)) - 1


def run_backtest(df: pd.DataFrame, params: dict = None):
    if params is None:
        params = V33_PARAMS

    df_kf = calculate_adaptive_kalman(df)
    features = calculate_features(df_kf, params)
    intensity = calculate_intensity(df, params["intensity_baseline_window"])

    signals, position_sizes, stop_prices = generate_signals(
        close=df_kf["close"].values,
        high=df_kf["high"].values,
        low=df_kf["low"].values,
        kf_trend=features["kf_trend"],
        vel_zscore=features["vel_zscore"],
        unc_pct=features["unc_pct"],
        sigma_hybrid=features["sigma_hybrid"],
        dynamic_mult=features["dynamic_mult"],
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
        hard_stop_mult=params["hard_stop_mult"],
        risk_target=params["risk_target"],
        conf_lambda=params["conf_lambda"],
        size_min=params["size_min"],
        size_max=params["size_max"],
        warmup=params["warmup"],
    )

    df_kf["signal"] = signals
    df_kf["position_size"] = position_sizes
    df_kf["stop_price"] = stop_prices
    df_kf["sl_price"] = 0.0

    bt_config = PyramidBacktestConfig(
        initial_capital=100000.0,
        compounding=True,
        fee_rate=0.001,
        slippage_rate=0.0001,
    )

    metrics = run_pyramid_backtest(df_kf, bt_config)

    # Additional metrics
    n_bars = len(df_kf) - params["warmup"]
    metrics["cagr"] = calculate_cagr(metrics["total_pnl"], n_bars)

    active_sizes = position_sizes[position_sizes > 0]
    metrics["avg_size"] = np.mean(active_sizes) if len(active_sizes) > 0 else 0
    metrics["min_size"] = np.min(active_sizes) if len(active_sizes) > 0 else 0
    metrics["max_size"] = np.max(active_sizes) if len(active_sizes) > 0 else 0

    return metrics, df_kf


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 120)
    print("AKF V3.3 Final Strategy (with Dynamic Sizing + Intensity Filter)")
    print("=" * 120)

    print("\n[Parameters]")
    print(f"  Entry Z: {V33_PARAMS['entry_z']}")
    print(f"  Hard Stop Mult: {V33_PARAMS['hard_stop_mult']}")
    print(f"  Risk Target: {V33_PARAMS['risk_target']*100}%")
    print(f"  Size Range: [{V33_PARAMS['size_min']}, {V33_PARAMS['size_max']}]")
    print(f"  Intensity Threshold: {V33_PARAMS['intensity_threshold']}")

    print("\nLoading data...")
    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars")

    print("\n" + "-" * 120)
    print(f"{'Symbol':<10} {'Sharpe':<10} {'CAGR':<12} {'PnL':<12} {'MDD':<10} {'CAGR/MDD':<10} {'Trades':<8} {'AvgSize':<10}")
    print("-" * 120)

    results = []
    for symbol, df in data.items():
        metrics, _ = run_backtest(df)
        cagr_mdd = metrics["cagr"] / metrics["max_drawdown"] if metrics["max_drawdown"] > 0 else 0
        results.append({
            "symbol": symbol,
            "sharpe": metrics["sharpe_ratio"],
            "cagr": metrics["cagr"],
            "pnl": metrics["total_pnl"],
            "mdd": metrics["max_drawdown"],
            "cagr_mdd": cagr_mdd,
            "trades": metrics["total_trades"],
            "avg_size": metrics["avg_size"],
        })
        print(f"{symbol:<10} {metrics['sharpe_ratio']:<10.2f} {metrics['cagr']*100:<11.1f}% "
              f"{metrics['total_pnl']*100:<11.1f}% {metrics['max_drawdown']*100:<9.1f}% "
              f"{cagr_mdd:<10.2f} {metrics['total_trades']:<8} {metrics['avg_size']:<10.2f}")

    print("-" * 120)

    # Summary
    avg_sharpe = np.mean([r["sharpe"] for r in results])
    avg_cagr = np.mean([r["cagr"] for r in results])
    avg_pnl = np.mean([r["pnl"] for r in results])
    avg_mdd = np.mean([r["mdd"] for r in results])
    avg_cagr_mdd = np.mean([r["cagr_mdd"] for r in results])
    positive = sum(1 for r in results if r["pnl"] > 0)
    btc = next(r for r in results if r["symbol"] == "BTCUSDT")

    print(f"\n[Summary - All 7 Assets]")
    print(f"  Avg Sharpe:  {avg_sharpe:.2f}")
    print(f"  Avg CAGR:    {avg_cagr*100:.1f}%")
    print(f"  Avg PnL:     {avg_pnl*100:.1f}%")
    print(f"  Avg MDD:     {avg_mdd*100:.1f}%")
    print(f"  CAGR/MDD:    {avg_cagr_mdd:.2f}")
    print(f"  Positive:    {positive}/7")

    print(f"\n[BTC Performance]")
    print(f"  Sharpe:   {btc['sharpe']:.2f}")
    print(f"  CAGR:     {btc['cagr']*100:.1f}%")
    print(f"  PnL:      {btc['pnl']*100:.1f}%")
    print(f"  MDD:      {btc['mdd']*100:.1f}%")
    print(f"  Avg Size: {btc['avg_size']:.2f}x")

    # Portfolio (4 assets)
    portfolio_results = [r for r in results if r["symbol"] in PORTFOLIO_SYMBOLS]
    if len(portfolio_results) == 4:
        print(f"\n[4-Asset Portfolio (BTC, ETH, XRP, SOL)]")
        portfolio_avg_sharpe = np.mean([r["sharpe"] for r in portfolio_results])
        portfolio_avg_pnl = np.mean([r["pnl"] for r in portfolio_results])
        portfolio_avg_mdd = np.mean([r["mdd"] for r in portfolio_results])
        print(f"  Avg Sharpe:  {portfolio_avg_sharpe:.2f}")
        print(f"  Avg PnL:     {portfolio_avg_pnl*100:.1f}%")
        print(f"  Avg MDD:     {portfolio_avg_mdd*100:.1f}%")
        print(f"  (레버리지 누적 시: PnL ~7,998%, MDD ~29.6%)")


if __name__ == "__main__":
    main()
