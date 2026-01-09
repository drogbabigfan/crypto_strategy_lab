#!/usr/bin/env python3
"""
AKF V3.5 Strategy (Dual-Eye Kalman + Rolling MAE Stops)

=== V3.4 → V3.5 변경사항 ===

스탑 시스템 전면 개편:
- 기존 (V3.4): 7 × σ_hybrid 기반 (그리드 서치로 찾은 값)
- 신규 (V3.5): 이론적 근거가 있는 Kalman + MAE 기반

1. Trail Stop: k × √S_risk (Dual-Eye Kalman)
   - S_risk = P_pred + R_parkinson (Innovation Covariance)
   - Parkinson R: 봉 내 변동폭 (high-low) 반영 → 휩소 방어
   - k=5: 5σ 이탈 시 청산 (통계적 의미)

2. Hard Stop: Rolling MAE 95th percentile
   - MAE_t = (Open_t - min(Low_{t+1}...Low_{t+H})) / Open_t
   - 과거 180봉 중 95th percentile 사용
   - "지금 진입하면 1일(6봉) 동안 최악의 경우 얼마나 빠질까?"

장점:
- Trail: Kalman Filter의 Innovation Covariance 활용 (이론적 근거)
- Hard: 실제 데이터 기반 최대 역행폭 (통계적 근거)
- 두 스탑 모두 시장 상황에 적응

=== 성능 (V3.5 - Trail k=5 + MAE Q95) ===

개별 자산:
              CAGR        MDD         CAGR/MDD    Sharpe
- BTC:        25.4%       21.3%       1.19        0.94
- ETH:        7.4%        39.0%       0.19        0.46
- XRP:        10.5%       33.4%       0.31        0.50
- SOL:        13.9%       32.1%       0.43        0.61

4-Asset Combined:
- CAGR:       66.3%
- MDD:        27.8%
- CAGR/MDD:   2.38 (V3.4: 1.66, 43% 개선)
- Sharpe:     1.23
- Avg Lev:    2.61x

거래 통계:
- 평균 18-21 trades/year
- 평균 보유 기간: 4-5일
- 시장 참여율: ~22%
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from numba import njit

from research.features.kalman import calculate_adaptive_kalman
from strategies.v4.backtest_pyramid import run_pyramid_backtest, PyramidBacktestConfig

DATA_ROOT = PROJECT_ROOT / "etl/data"
ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "LTCUSDT"]
PORTFOLIO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"]


# ============================================================================
# V3.5 Parameters
# ============================================================================
V35_PARAMS = {
    # Entry
    "entry_z": 2.0,
    "unc_pct_max": 0.5,
    "warmup": 210,

    # Trail Stop (Kalman Dual-Eye)
    "trail_k": 5.0,  # k × √S_risk

    # Hard Stop (Rolling MAE)
    "mae_horizon": 6,      # 진입 후 6봉 (1일) 동안의 MAE
    "mae_window": 180,     # 과거 180봉 (30일) 참고
    "mae_quantile": 0.95,  # 95th percentile

    # Innovation Breaker (v_ratio 기반 동적)
    "v_ratio_threshold": 1.0,
    "innov_base_mult": 2.5,
    "innov_mult_min": 1.5,
    "innov_mult_max": 4.0,

    # Dynamic Sizing (Gaussian)
    "risk_target": 0.03,        # 3%
    "gauss_max_mult": 2.0,      # 최대 배율
    "gauss_sigma": 0.5,         # Gaussian σ
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
# Rolling MAE (Maximum Adverse Excursion)
# ============================================================================
def calculate_rolling_mae(
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    horizon: int = 6,
    window: int = 180,
    quantile: float = 0.95
) -> np.ndarray:
    """
    Rolling MAE (Maximum Adverse Excursion) 계산

    "지금 진입하면 향후 H봉 동안 재수 없으면 어디까지 빠질까?"

    Long 기준:
    MAE_t = (Open_t - min(Low_{t+1} ... Low_{t+H})) / Open_t

    Returns:
        rolling_mae_pct: 각 시점의 quantile percentile MAE (%)
    """
    n = len(open_prices)
    mae = np.full(n, np.nan)

    # Step 1: 각 봉의 MAE 계산 (Long 기준)
    for t in range(n - horizon):
        entry_price = open_prices[t]
        if entry_price <= 0 or np.isnan(entry_price):
            continue

        future_lows = low_prices[t+1:t+1+horizon]
        if len(future_lows) == 0 or np.any(np.isnan(future_lows)):
            continue

        min_low = np.min(future_lows)
        mae[t] = (entry_price - min_low) / entry_price

    # Step 2: Rolling percentile
    rolling_mae_pct = np.full(n, np.nan)

    for t in range(window, n):
        past_mae = mae[t-window:t]
        valid_mae = past_mae[~np.isnan(past_mae)]
        if len(valid_mae) >= 10:
            rolling_mae_pct[t] = np.percentile(valid_mae, quantile * 100)

    # 초기 구간은 전체 평균으로 채움
    valid_all = mae[~np.isnan(mae)]
    if len(valid_all) > 0:
        default_mae = np.percentile(valid_all, quantile * 100)
        rolling_mae_pct[:window] = default_mae

    return rolling_mae_pct


# ============================================================================
# Feature Calculation
# ============================================================================
def calculate_features(df_kf: pd.DataFrame, params: dict = None):
    if params is None:
        params = V35_PARAMS

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
    """
    Gaussian Position Sizing with MAE-based stop distance

    1. Base Size = Risk_Target / Stop_Distance (MAE)
    2. Gaussian Scaling = max_mult × exp(-unc_pct² / (2σ²))
    3. Final Size = clip(Base × Gaussian, min, max)
    """
    # Base Size (MAE 기반)
    base_size = risk_target / (stop_distance + 1e-10)

    # Gaussian Scaling
    gauss_scale = gauss_max_mult * np.exp(-unc_pct**2 / (2 * gauss_sigma**2))

    # Final Size
    final_size = base_size * gauss_scale
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
    sqrt_s_risk: np.ndarray,
    rolling_mae: np.ndarray,
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
    trail_k: float,
    risk_target: float,
    gauss_max_mult: float,
    gauss_sigma: float,
    size_min: float,
    size_max: float,
    warmup: int,
):
    """
    V3.5 Signal Generation

    Trail Stop: k × √S_risk (Dual-Eye Kalman)
    Hard Stop: Rolling MAE percentile
    """
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
    entry_mae = 0.0

    for i in range(warmup, n):
        # Innovation mult (v_ratio 기반 동적)
        innov_mult = innov_base_mult / (v_ratio[i] + 1e-10)
        innov_mult = max(innov_mult_min, min(innov_mult_max, innov_mult))

        log_resid = np.log(close[i]) - np.log(kf_trend[i])

        # Trail stop distance (Kalman S_risk)
        stop_dist = sqrt_s_risk[i]

        if position == 0:
            # === LONG ENTRY ===
            if vel_zscore[i] > entry_z and unc_pct[i] < unc_pct_max:
                position = 1
                entry_price = close[i]
                highest = high[i]
                entry_mae = rolling_mae[i]

                # Gaussian Sizing (MAE 기반)
                current_size = calculate_position_size(
                    unc_pct[i], entry_mae, risk_target,
                    gauss_max_mult, gauss_sigma, size_min, size_max
                )

                # Trail Stop (Kalman)
                prev_stop = highest * np.exp(-trail_k * stop_dist)
                # Hard Stop (MAE)
                hard_stop = entry_price * (1.0 - entry_mae)

                signals[i] = 1
                position_sizes[i] = current_size
                stop_prices[i] = max(prev_stop, hard_stop)

            # === SHORT ENTRY ===
            elif vel_zscore[i] < -entry_z and unc_pct[i] < unc_pct_max:
                position = -1
                entry_price = close[i]
                lowest = low[i]
                entry_mae = rolling_mae[i]

                # Gaussian Sizing (MAE 기반)
                current_size = calculate_position_size(
                    unc_pct[i], entry_mae, risk_target,
                    gauss_max_mult, gauss_sigma, size_min, size_max
                )

                # Trail Stop (Kalman)
                prev_stop = lowest * np.exp(trail_k * stop_dist)
                # Hard Stop (MAE)
                hard_stop = entry_price * (1.0 + entry_mae)

                signals[i] = -1
                position_sizes[i] = current_size
                stop_prices[i] = min(prev_stop, hard_stop)

        elif position == 1:  # === LONG POSITION ===
            if high[i] > highest:
                highest = high[i]

            # Trailing Stop (Kalman based, dynamic)
            new_stop = highest * np.exp(-trail_k * stop_dist)
            trail_stop = max(new_stop, prev_stop)  # Ratchet
            prev_stop = trail_stop
            stop_prices[i] = max(trail_stop, hard_stop)

            # Exit checks (우선순위)
            # 1. Hard Stop (MAE based, fixed at entry)
            if low[i] < hard_stop:
                position = 0
                continue
            # 2. Trailing Stop (Kalman based)
            if low[i] < trail_stop:
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

            signals[i] = 1
            position_sizes[i] = current_size

        elif position == -1:  # === SHORT POSITION ===
            if low[i] < lowest:
                lowest = low[i]

            # Trailing Stop (Kalman based, dynamic)
            new_stop = lowest * np.exp(trail_k * stop_dist)
            trail_stop = min(new_stop, prev_stop)  # Ratchet
            prev_stop = trail_stop
            stop_prices[i] = min(trail_stop, hard_stop)

            # Exit checks (우선순위)
            # 1. Hard Stop (MAE based, fixed at entry)
            if high[i] > hard_stop:
                position = 0
                continue
            # 2. Trailing Stop (Kalman based)
            if high[i] > trail_stop:
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
        params = V35_PARAMS

    df_kf = calculate_adaptive_kalman(df)
    features = calculate_features(df_kf, params)
    intensity = calculate_intensity(df, params["intensity_baseline_window"])

    # Get S_risk from Kalman filter (Dual-Eye)
    sqrt_s_risk = np.sqrt(df_kf["kf_innovation_cov_risk"].values)

    # Calculate Rolling MAE for Hard Stop
    rolling_mae = calculate_rolling_mae(
        open_prices=df_kf["open"].values,
        high_prices=df_kf["high"].values,
        low_prices=df_kf["low"].values,
        horizon=params["mae_horizon"],
        window=params["mae_window"],
        quantile=params["mae_quantile"]
    )

    signals, position_sizes, stop_prices = generate_signals(
        close=df_kf["close"].values,
        high=df_kf["high"].values,
        low=df_kf["low"].values,
        kf_trend=features["kf_trend"],
        sqrt_s_risk=sqrt_s_risk,
        rolling_mae=rolling_mae,
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
        trail_k=params["trail_k"],
        risk_target=params["risk_target"],
        gauss_max_mult=params["gauss_max_mult"],
        gauss_sigma=params["gauss_sigma"],
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

    # Bar returns for portfolio calculation
    equity = metrics["equity_curve"]
    bar_returns = np.zeros(len(equity))
    bar_returns[1:] = np.diff(equity) / equity[:-1]
    metrics["bar_returns"] = bar_returns

    return metrics, df_kf


# ============================================================================
# Portfolio Metrics
# ============================================================================
def calculate_portfolio_metrics(individual_results):
    """4-asset portfolio metrics"""
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

    n_years = min_len / (6 * 252)
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
    }


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 120)
    print("AKF V3.5 Strategy (Dual-Eye Kalman + Rolling MAE Stops)")
    print("=" * 120)

    print("\n[Parameters]")
    print(f"  Entry Z: {V35_PARAMS['entry_z']}")
    print(f"  Trail Stop: k={V35_PARAMS['trail_k']} × sqrt(S_risk)")
    print(f"  Hard Stop: MAE Q{V35_PARAMS['mae_quantile']*100:.0f}% (H={V35_PARAMS['mae_horizon']}, W={V35_PARAMS['mae_window']})")
    print(f"  Risk Target: {V35_PARAMS['risk_target']*100}%")
    print(f"  Gaussian: M={V35_PARAMS['gauss_max_mult']}, σ={V35_PARAMS['gauss_sigma']}")
    print(f"  Size Range: [{V35_PARAMS['size_min']}, {V35_PARAMS['size_max']}]")

    print("\nLoading data...")
    data = {}
    for symbol in PORTFOLIO_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars")

    print("\n" + "-" * 120)
    print(f"{'Symbol':<10} {'Sharpe':<10} {'CAGR':<12} {'MDD':<10} {'CAGR/MDD':<10} {'Trades':<8} {'WinRate':<10} {'AvgSize':<10}")
    print("-" * 120)

    individual_results = {}
    for symbol, df in data.items():
        metrics, _ = run_backtest(df)
        individual_results[symbol] = metrics

        cagr_mdd = metrics["cagr"] / metrics["max_drawdown"] if metrics["max_drawdown"] > 0 else 0
        print(f"{symbol:<10} {metrics['sharpe_ratio']:<10.2f} {metrics['cagr']*100:<11.1f}% "
              f"{metrics['max_drawdown']*100:<9.1f}% {cagr_mdd:<10.2f} "
              f"{metrics['total_trades']:<8} {metrics['win_rate']*100:<9.1f}% {metrics['avg_size']:<10.2f}")

    print("-" * 120)

    # Portfolio metrics
    portfolio = calculate_portfolio_metrics(individual_results)
    print(f"\n[4-Asset Portfolio]")
    print(f"  CAGR:       {portfolio['cagr']*100:.1f}%")
    print(f"  MDD:        {portfolio['max_drawdown']*100:.1f}%")
    print(f"  CAGR/MDD:   {portfolio['cagr_mdd']:.2f}")
    print(f"  Sharpe:     {portfolio['sharpe']:.2f}")
    print(f"  Leverage:   {portfolio['avg_leverage']:.2f}x")

    print(f"\n[Comparison with V3.4]")
    print(f"  V3.4: CAGR=60.2%, MDD=36.2%, CAGR/MDD=1.66")
    print(f"  V3.5: CAGR={portfolio['cagr']*100:.1f}%, MDD={portfolio['max_drawdown']*100:.1f}%, CAGR/MDD={portfolio['cagr_mdd']:.2f}")
    improvement = (portfolio['cagr_mdd'] / 1.66 - 1) * 100
    print(f"  Improvement: {improvement:+.0f}%")


if __name__ == "__main__":
    main()
