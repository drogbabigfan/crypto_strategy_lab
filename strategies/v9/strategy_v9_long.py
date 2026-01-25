#!/usr/bin/env python3
"""
V9 Strategy - Dual Kalman Long Only

Phase 1-2: 롱 온리 전략, 단일 포지션
- Dual Kalman Filter (fast=20, slow=120)
- 4가지 시그널 비교 + 조합 테스트
- V7 청산 로직 (Parkinson band 스탑/TP)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from research.features.kalman import KalmanFeatureGenerator, KalmanConfig

DATA_ROOT = PROJECT_ROOT / "etl/data"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"]


class SignalType(Enum):
    """롱 시그널 타입"""
    VEL_VOLUME = "VelVolume"      # fast_z > 2 AND duration < ma*0.5
    TREND_ALIGN = "TrendAlign"    # slow_z > 1 AND fast_z > 2
    FAST_ONLY = "FastOnly"        # fast_z > 2
    SLOW_Z = "SlowZ"              # slow_z > 2
    # AND 조합
    VEL_TREND = "VelTrend"        # VelVolume AND TrendAlign
    VEL_SLOW = "VelSlow"          # VelVolume AND SlowZ
    ALL_THREE = "AllThree"        # fast_z > 2 AND slow_z > 1 AND duration < ma*0.5
    # OR 조합
    OR_VEL_TREND = "OR_VelTrend"      # VelVolume OR TrendAlign
    OR_VEL_SLOW = "OR_VelSlow"        # VelVolume OR SlowZ
    OR_FAST_SLOW = "OR_FastSlow"      # FastOnly OR SlowZ
    OR_ALL = "OR_All"                  # VelVolume OR TrendAlign OR SlowZ


class ShortSignalType(Enum):
    """숏 시그널 타입"""
    SPREAD = "Spread"             # fast_z < -2 AND spread < -1.0 * spread_std
    SLOW_TICK = "SlowTick"        # slow_z < -2 AND tick < tick_p25
    FAST_ONLY = "FastOnly"        # fast_z < -2
    SPREAD_TIGHT = "SpreadTight"  # fast_z < -2 AND spread < -1.5 * spread_std


PARAMS = {
    "warmup": 210,
    # Dual Kalman
    "fast_window": 20,
    "slow_window": 120,
    # Entry
    "fast_zscore_threshold": 2.0,
    "slow_zscore_threshold": 1.0,  # TrendAlign용
    "slow_zscore_threshold_high": 2.0,  # SlowZ용
    "duration_ratio": 0.5,  # VelVolume용
    "duration_ma_window": 50,
    # Exit
    "K": 1.5,
    "horizon": 10,
    "parkinson_window": 20,
    "compression_window": 100,
    # Position Sizing
    "risk_per_trade": 0.07,
    "max_position_size": 1.0,
    "min_position_size": 0.1,
    # Pyramid
    "pyramid_threshold": 0.75,  # 0.75x SD 움직일 때마다 피라미드
    "max_pyramids": 5,
    "pyramid_size_ratio": 0.5,  # 피라미드 사이즈 = 최초 × ratio
    "max_total_size": 2.5,
    # Costs
    "fee": 0.001,
    "slippage": 0.0005,
}


@dataclass
class PyramidEntry:
    bar: int
    price: float
    size: float


@dataclass
class Trade:
    direction: int
    entry_bar: int
    entry_price: float
    exit_bar: int
    exit_price: float
    exit_reason: str
    size: float
    gross_return: float = 0.0
    net_return: float = 0.0
    entries: List["PyramidEntry"] = None  # 피라미딩용


def load_data(symbol: str, bar_size: int = 6) -> Optional[pd.DataFrame]:
    """심볼별 데이터 로드"""
    data_dir = DATA_ROOT / f"features-{bar_size}/futures/{symbol}"
    dfs = []
    for year in range(2020, 2026):
        for month in range(1, 13):
            path = data_dir / f"{symbol}-features-{year}-{month:02d}.parquet"
            if path.exists():
                dfs.append(pd.read_parquet(path))
    return pd.concat(dfs, ignore_index=True) if dfs else None


def calculate_dual_kalman(df: pd.DataFrame, fast_window: int = 20, slow_window: int = 120) -> pd.DataFrame:
    """
    Dual Kalman Filter 계산

    Returns:
        DataFrame with:
        - fast_zscore: fast kalman velocity z-score
        - slow_zscore: slow kalman velocity z-score
        - spread: fast_velocity - slow_velocity
        - spread_std: rolling std of spread
    """
    # Fast Kalman (window=20)
    fast_config = KalmanConfig(r_window=fast_window, q_window=fast_window)
    fast_kf = KalmanFeatureGenerator(r_window=fast_window, q_window=fast_window)
    df_fast = fast_kf.generate(df)

    # Slow Kalman (window=120)
    slow_config = KalmanConfig(r_window=slow_window, q_window=slow_window)
    slow_kf = KalmanFeatureGenerator(r_window=slow_window, q_window=slow_window)
    df_slow = slow_kf.generate(df)

    # Combine
    result = df.copy()
    result["fast_velocity"] = df_fast["kf_velocity"].values
    result["fast_zscore"] = df_fast["kf_velocity_zscore"].values
    result["fast_uncertainty"] = df_fast["kf_uncertainty"].values

    result["slow_velocity"] = df_slow["kf_velocity"].values
    result["slow_zscore"] = df_slow["kf_velocity_zscore"].values
    result["slow_uncertainty"] = df_slow["kf_uncertainty"].values

    # Spread
    result["spread"] = result["fast_velocity"] - result["slow_velocity"]
    result["spread_std"] = result["spread"].rolling(50, min_periods=10).std()

    # Parkinson volatility for stop distance
    result["kf_innovation_cov_risk"] = df_fast["kf_innovation_cov_risk"].values

    return result


def calculate_parkinson_vol(high: np.ndarray, low: np.ndarray, window: int = 20) -> np.ndarray:
    """Parkinson volatility 계산"""
    log_hl = np.log(high / low)
    parkinson_sq = log_hl ** 2 / (4 * np.log(2))
    return pd.Series(np.sqrt(parkinson_sq)).rolling(window, min_periods=5).mean().values


def calculate_compression(log_duration: np.ndarray, window: int = 100) -> np.ndarray:
    """Compression factor 계산"""
    duration_hours = np.exp(log_duration) / 3600
    bars_per_day = 24 / duration_hours
    rolling_bars = pd.Series(bars_per_day).rolling(window, min_periods=20).mean().values
    return np.clip(bars_per_day / (rolling_bars + 1e-10), 0.5, 2.0)


def get_short_signal(
    signal_type: ShortSignalType,
    fast_z: float,
    slow_z: float,
    spread: float,
    spread_std: float,
    tick: float,
    tick_p25: float,
    params: dict
) -> bool:
    """시그널 타입에 따른 숏 진입 조건 체크"""

    fast_thresh = params["fast_zscore_threshold"]
    slow_thresh_high = params["slow_zscore_threshold_high"]

    if np.isnan(fast_z) or np.isnan(slow_z):
        return False

    if signal_type == ShortSignalType.SPREAD:
        if np.isnan(spread) or np.isnan(spread_std) or spread_std <= 0:
            return False
        return fast_z < -fast_thresh and spread < -1.0 * spread_std

    elif signal_type == ShortSignalType.SPREAD_TIGHT:
        if np.isnan(spread) or np.isnan(spread_std) or spread_std <= 0:
            return False
        return fast_z < -fast_thresh and spread < -1.5 * spread_std

    elif signal_type == ShortSignalType.SLOW_TICK:
        if np.isnan(tick) or np.isnan(tick_p25):
            return False
        return slow_z < -slow_thresh_high and tick < tick_p25

    elif signal_type == ShortSignalType.FAST_ONLY:
        return fast_z < -fast_thresh

    return False


def get_long_signal(
    signal_type: SignalType,
    fast_z: float,
    slow_z: float,
    duration: float,
    duration_ma: float,
    params: dict
) -> bool:
    """시그널 타입에 따른 롱 진입 조건 체크"""

    fast_thresh = params["fast_zscore_threshold"]
    slow_thresh = params["slow_zscore_threshold"]
    slow_thresh_high = params["slow_zscore_threshold_high"]
    dur_ratio = params["duration_ratio"]

    if np.isnan(fast_z) or np.isnan(slow_z) or np.isnan(duration) or np.isnan(duration_ma):
        return False

    # 개별 시그널 계산
    vel_volume = fast_z > fast_thresh and duration < duration_ma * dur_ratio
    trend_align = slow_z > slow_thresh and fast_z > fast_thresh
    fast_only = fast_z > fast_thresh
    slow_z_sig = slow_z > slow_thresh_high

    if signal_type == SignalType.VEL_VOLUME:
        return vel_volume

    elif signal_type == SignalType.TREND_ALIGN:
        return trend_align

    elif signal_type == SignalType.FAST_ONLY:
        return fast_only

    elif signal_type == SignalType.SLOW_Z:
        return slow_z_sig

    elif signal_type == SignalType.VEL_TREND:
        # VelVolume AND TrendAlign
        return vel_volume and slow_z > slow_thresh

    elif signal_type == SignalType.VEL_SLOW:
        # VelVolume AND SlowZ
        return vel_volume and slow_z_sig

    elif signal_type == SignalType.ALL_THREE:
        return vel_volume and slow_z > slow_thresh

    # OR 조합
    elif signal_type == SignalType.OR_VEL_TREND:
        # VelVolume OR TrendAlign
        return vel_volume or trend_align

    elif signal_type == SignalType.OR_VEL_SLOW:
        # VelVolume OR SlowZ
        return vel_volume or slow_z_sig

    elif signal_type == SignalType.OR_FAST_SLOW:
        # FastOnly OR SlowZ
        return fast_only or slow_z_sig

    elif signal_type == SignalType.OR_ALL:
        # VelVolume OR TrendAlign OR SlowZ
        return vel_volume or trend_align or slow_z_sig

    return False


def run_backtest(
    symbol: str,
    signal_type: SignalType,
    params: dict = None,
    enable_pyramid: bool = False,
) -> Optional[dict]:
    """
    백테스트 실행

    Args:
        symbol: 심볼
        signal_type: 시그널 타입
        params: 파라미터
        enable_pyramid: 피라미딩 활성화 여부
    """
    if params is None:
        params = PARAMS

    df = load_data(symbol)
    if df is None:
        return None

    # Dual Kalman 계산
    df_kf = calculate_dual_kalman(df, params["fast_window"], params["slow_window"])
    n = len(df_kf)

    # 필요한 데이터 추출
    fast_zscore = df_kf["fast_zscore"].values
    slow_zscore = df_kf["slow_zscore"].values
    open_prices = df_kf["open"].values
    high = df_kf["high"].values
    low = df_kf["low"].values
    close = df_kf["close"].values

    high_times = df_kf["high_time"].values if "high_time" in df_kf.columns else np.arange(n)
    low_times = df_kf["low_time"].values if "low_time" in df_kf.columns else np.arange(n)
    bar_timestamps = df_kf["start_time"].values if "start_time" in df_kf.columns else np.arange(n) * 60000

    # Duration 계산
    log_duration = df_kf["log_duration"].values if "log_duration" in df_kf.columns else np.zeros(n)
    duration = np.exp(log_duration)
    duration_ma = pd.Series(duration).rolling(params["duration_ma_window"], min_periods=10).mean().values

    # Stop distance 계산
    parkinson = calculate_parkinson_vol(high, low, params["parkinson_window"])
    compression = calculate_compression(log_duration, params["compression_window"])
    stop_distance = params["K"] * parkinson * np.sqrt(params["horizon"]) * np.sqrt(compression)

    # Parameters
    warmup = params["warmup"]
    fee = params["fee"]
    slippage = params["slippage"]
    risk_per_trade = params["risk_per_trade"]
    max_position_size = params["max_position_size"]
    min_position_size = params["min_position_size"]

    # Pyramid parameters
    pyramid_threshold = params.get("pyramid_threshold", 0.75)
    max_pyramids = params.get("max_pyramids", 5)
    pyramid_size_ratio = params.get("pyramid_size_ratio", 0.5)
    max_total_size = params.get("max_total_size", 2.5)
    signal_pyramid = params.get("signal_pyramid", False)  # 시그널 기반 피라미딩

    # State
    trades: List[Trade] = []
    initial_capital = params.get("initial_capital", 100000.0)
    equity = initial_capital
    mtm_equity = np.full(n, np.nan)

    in_position = False
    current_entries: List[PyramidEntry] = []
    current_stop = 0.0
    current_tp = 0.0
    highest = 0.0
    initial_sd = 0.0
    initial_entry_size = 0.0
    pyramid_count = 0
    next_pyramid_price = 0.0
    last_signal_bar = -999  # 시그널 피라미딩용: 마지막 시그널 bar

    def get_avg_entry():
        if not current_entries:
            return 0.0, 0.0
        total_cost = sum(e.price * e.size for e in current_entries)
        total_size = sum(e.size for e in current_entries)
        return total_cost / total_size if total_size > 0 else 0.0, total_size

    for bar in range(warmup, n):
        sd = stop_distance[bar]

        # 롱 시그널 체크
        long_signal = get_long_signal(
            signal_type,
            fast_zscore[bar],
            slow_zscore[bar],
            duration[bar],
            duration_ma[bar],
            params
        )

        if not in_position:
            mtm_equity[bar] = equity

            if not long_signal or np.isnan(sd) or sd <= 0:
                continue

            # 다음 bar open에서 진입
            entry_bar = bar + 1
            if entry_bar >= n:
                continue

            entry_price = open_prices[entry_bar] * (1 + slippage)

            # Position sizing
            position_size = risk_per_trade / (sd + 1e-10)
            position_size = np.clip(position_size, min_position_size, max_position_size)

            # Initialize position
            current_entries = [PyramidEntry(bar=entry_bar, price=entry_price, size=position_size)]
            in_position = True
            initial_sd = sd
            initial_entry_size = position_size
            pyramid_count = 1
            highest = high[entry_bar] if entry_bar < n else entry_price

            # Stop/TP 설정
            current_stop = entry_price * (1 - sd)
            current_tp = entry_price * (1 + sd)
            next_pyramid_price = entry_price * (1 + pyramid_threshold * sd)

        else:
            # In position

            # 1. Check for pyramid (if enabled)
            if enable_pyramid:
                total_size = sum(e.size for e in current_entries)
                can_pyramid = (
                    pyramid_count < max_pyramids and
                    total_size < max_total_size
                )

                pyramid_triggered = False
                pyramid_price = None

                # 1a. 가격 기반 피라미딩
                if can_pyramid and high[bar] >= next_pyramid_price:
                    pyramid_triggered = True
                    pyramid_price = next_pyramid_price

                # 1b. 시그널 기반 피라미딩 (새로운 시그널 발생시)
                if signal_pyramid and can_pyramid and not pyramid_triggered:
                    # 시그널이 발생하고, 마지막 피라미딩 후 최소 1 bar 경과
                    if long_signal and bar > last_signal_bar + 1:
                        # 현재 가격이 평균 진입가보다 높을 때만
                        avg_entry, _ = get_avg_entry()
                        if close[bar] > avg_entry:
                            pyramid_triggered = True
                            pyramid_price = close[bar]

                if pyramid_triggered and pyramid_price is not None:
                    # Add pyramid entry
                    pyr_size = initial_entry_size * pyramid_size_ratio
                    remaining = max_total_size - total_size
                    pyr_size = min(pyr_size, remaining)

                    if pyr_size > 0:
                        actual_pyr_price = pyramid_price * (1 + slippage)
                        current_entries.append(PyramidEntry(bar=bar, price=actual_pyr_price, size=pyr_size))
                        pyramid_count += 1
                        last_signal_bar = bar

                        # Trail stop and TP
                        trail_amount = pyramid_threshold * initial_sd
                        current_stop *= (1 + trail_amount)
                        current_tp *= (1 + trail_amount)
                        next_pyramid_price = pyramid_price * (1 + pyramid_threshold * initial_sd)

            # 2. Trail stop (high 갱신시) - 피라미딩 없을 때만
            if not enable_pyramid:
                if high[bar] > highest:
                    highest = high[bar]
                    new_stop = highest * (1 - initial_sd)
                    current_stop = max(current_stop, new_stop)

            # 3. MTM tracking
            avg_entry, total_size = get_avg_entry()
            unrealized_pnl = (low[bar] - avg_entry) / avg_entry
            mtm_equity[bar] = equity * (1 + unrealized_pnl * total_size)

            # 4. Check exit
            exit_price = None
            exit_reason = None

            stop_hit = low[bar] <= current_stop
            tp_hit = high[bar] >= current_tp

            if stop_hit and tp_hit:
                if low_times[bar] <= high_times[bar]:
                    exit_reason, exit_price = "stop", current_stop
                else:
                    exit_reason, exit_price = "tp", current_tp
            elif stop_hit:
                exit_reason, exit_price = "stop", current_stop
            elif tp_hit:
                exit_reason, exit_price = "tp", current_tp

            # 5. Process exit
            if exit_price is not None:
                avg_entry, total_size = get_avg_entry()
                actual_exit = exit_price * (1 - slippage)
                gross_return = (exit_price - avg_entry) / avg_entry
                net_return = (actual_exit - avg_entry) / avg_entry - fee * (len(current_entries) + 1)

                trade = Trade(
                    direction=1,
                    entry_bar=current_entries[0].bar,
                    entry_price=avg_entry,
                    exit_bar=bar,
                    exit_price=actual_exit,
                    exit_reason=exit_reason,
                    size=total_size,
                    gross_return=gross_return,
                    net_return=net_return,
                    entries=current_entries.copy() if enable_pyramid else None,
                )
                trades.append(trade)

                equity *= (1 + net_return * total_size)

                # Reset
                in_position = False
                current_entries = []
                current_stop = 0.0
                current_tp = 0.0
                highest = 0.0
                pyramid_count = 0

    # Calculate metrics
    valid_mtm = mtm_equity[~np.isnan(mtm_equity)]
    equity_arr = valid_mtm if len(valid_mtm) > 0 else np.array([initial_capital, equity])

    total_trades = len(trades)
    if total_trades == 0:
        return None

    wins = sum(1 for t in trades if t.net_return > 0)
    win_rate = wins / total_trades

    bars_per_year = 365 * 24 / 4
    total_bars = n - warmup
    years = total_bars / bars_per_year
    total_return = equity_arr[-1] / equity_arr[0]
    cagr = total_return ** (1 / max(years, 0.01)) - 1

    running_max = np.maximum.accumulate(equity_arr)
    drawdown = (running_max - equity_arr) / running_max
    max_dd = np.max(drawdown)

    returns = np.diff(equity_arr) / equity_arr[:-1]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(bars_per_year) if len(returns) > 0 else 0

    stop_count = sum(1 for t in trades if t.exit_reason == "stop")
    tp_count = sum(1 for t in trades if t.exit_reason == "tp")

    avg_return = np.mean([t.net_return for t in trades])

    # Pyramid stats
    if enable_pyramid:
        avg_entries = np.mean([len(t.entries) if t.entries else 1 for t in trades])
        pyramid_rate = sum(1 for t in trades if t.entries and len(t.entries) > 1) / total_trades
    else:
        avg_entries = 1.0
        pyramid_rate = 0.0

    return {
        "symbol": symbol,
        "signal_type": signal_type.value,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "cagr": cagr,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "calmar": cagr / max_dd if max_dd > 0 else 0,
        "avg_return": avg_return,
        "mtm_equity": equity_arr,
        "timestamps": bar_timestamps[~np.isnan(mtm_equity)],
        "trades": trades,
        "years": years,
        "stop_pct": stop_count / total_trades * 100,
        "tp_pct": tp_count / total_trades * 100,
        "avg_size": np.mean([t.size for t in trades]),
        "avg_entries": avg_entries,
        "pyramid_rate": pyramid_rate,
    }


def calculate_portfolio_metrics(results: List[dict]) -> Optional[dict]:
    """포트폴리오 메트릭 계산"""
    if not results:
        return None

    initial_capital = 100000.0
    asset_daily_equity = {}

    for r in results:
        symbol = r["symbol"]
        timestamps = r["timestamps"]
        equity = r["mtm_equity"]
        dates = pd.to_datetime(timestamps, unit='ms')
        df_eq = pd.DataFrame({"equity": equity}, index=dates)
        daily_eq = df_eq.resample('D').last()["equity"]
        asset_daily_equity[symbol] = daily_eq

    all_dates = sorted(set().union(*[set(eq.index) for eq in asset_daily_equity.values()]))
    if len(all_dates) < 2:
        return None

    all_dates = pd.DatetimeIndex(all_dates)
    symbols = list(asset_daily_equity.keys())

    for symbol in symbols:
        asset_daily_equity[symbol] = asset_daily_equity[symbol].reindex(all_dates).ffill()
        if pd.isna(asset_daily_equity[symbol].iloc[0]):
            asset_daily_equity[symbol].iloc[0] = initial_capital
            asset_daily_equity[symbol] = asset_daily_equity[symbol].ffill()

    total_initial = initial_capital * len(symbols)
    portfolio_equity = np.zeros(len(all_dates))
    portfolio_equity[0] = total_initial

    for i in range(1, len(all_dates)):
        date = all_dates[i]
        prev_date = all_dates[i - 1]
        daily_pnl = sum(asset_daily_equity[s].loc[date] - asset_daily_equity[s].loc[prev_date] for s in symbols)
        portfolio_equity[i] = portfolio_equity[i - 1] + daily_pnl

    running_max = np.maximum.accumulate(portfolio_equity)
    max_dd = np.max((running_max - portfolio_equity) / running_max)

    daily_returns = np.diff(portfolio_equity) / (portfolio_equity[:-1] + 1e-10)
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)

    n_years = len(all_dates) / 365.0
    cagr = (portfolio_equity[-1] / total_initial) ** (1 / n_years) - 1

    total_trades = sum(r["total_trades"] for r in results)
    avg_wr = np.mean([r["win_rate"] for r in results])

    return {
        "cagr": cagr,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "calmar": cagr / max_dd if max_dd > 0 else 0,
        "years": n_years,
        "total_trades": total_trades,
        "avg_win_rate": avg_wr,
    }


def run_signal_comparison():
    """모든 시그널 타입 비교 테스트"""
    print("=" * 130)
    print("V9 Long Only - Signal Comparison (No Pyramid)")
    print("=" * 130)
    print(f"\nDual Kalman: fast={PARAMS['fast_window']}, slow={PARAMS['slow_window']}")
    print(f"Risk: {PARAMS['risk_per_trade']*100:.0f}%, K={PARAMS['K']}")

    all_results = {}

    for signal_type in SignalType:
        print(f"\n--- {signal_type.value} ---")
        print(f"{'Symbol':<10} {'Trades':>7} {'WR':>7} {'CAGR':>10} {'MDD':>8} {'Sharpe':>8} {'Calmar':>8} {'AvgRet':>10}")
        print("-" * 85)

        results = []
        for symbol in SYMBOLS:
            r = run_backtest(symbol, signal_type)
            if r:
                results.append(r)
                print(f"{r['symbol']:<10} {r['total_trades']:>7} {r['win_rate']*100:>6.1f}% "
                      f"{r['cagr']*100:>9.1f}% {r['max_dd']*100:>7.1f}% "
                      f"{r['sharpe']:>8.2f} {r['calmar']:>8.2f} {r['avg_return']*100:>9.2f}%")

        if len(results) >= 2:
            portfolio = calculate_portfolio_metrics(results)
            if portfolio:
                print("-" * 85)
                print(f"{'Portfolio':<10} {portfolio['total_trades']:>7} {portfolio['avg_win_rate']*100:>6.1f}% "
                      f"{portfolio['cagr']*100:>9.1f}% {portfolio['max_dd']*100:>7.1f}% "
                      f"{portfolio['sharpe']:>8.2f} {portfolio['calmar']:>8.2f}")
                all_results[signal_type.value] = portfolio

    # Summary
    print("\n" + "=" * 130)
    print("SUMMARY: Portfolio Results by Signal Type")
    print("=" * 130)
    print(f"{'Signal':<15} {'Trades':>8} {'WR':>7} {'CAGR':>10} {'MDD':>8} {'Sharpe':>8} {'Calmar':>8}")
    print("-" * 75)

    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["calmar"], reverse=True)
    for signal_name, metrics in sorted_results:
        print(f"{signal_name:<15} {metrics['total_trades']:>8} {metrics['avg_win_rate']*100:>6.1f}% "
              f"{metrics['cagr']*100:>9.1f}% {metrics['max_dd']*100:>7.1f}% "
              f"{metrics['sharpe']:>8.2f} {metrics['calmar']:>8.2f}")

    return all_results


def run_asymmetric_backtest(
    symbol: str,
    long_signal_type: SignalType,
    short_signal_type: ShortSignalType,
    params: dict = None,
    enable_pyramid: bool = True,
) -> Optional[dict]:
    """
    비대칭 백테스트 실행 (롱/숏 다른 시그널)

    Args:
        symbol: 심볼
        long_signal_type: 롱 시그널 타입
        short_signal_type: 숏 시그널 타입
        params: 파라미터
        enable_pyramid: 피라미딩 활성화 여부
    """
    if params is None:
        params = PARAMS

    df = load_data(symbol)
    if df is None:
        return None

    # Dual Kalman 계산
    df_kf = calculate_dual_kalman(df, params["fast_window"], params["slow_window"])
    n = len(df_kf)

    # 필요한 데이터 추출
    fast_zscore = df_kf["fast_zscore"].values
    slow_zscore = df_kf["slow_zscore"].values
    spread = df_kf["spread"].values
    spread_std = df_kf["spread_std"].values
    open_prices = df_kf["open"].values
    high = df_kf["high"].values
    low = df_kf["low"].values
    close = df_kf["close"].values

    high_times = df_kf["high_time"].values if "high_time" in df_kf.columns else np.arange(n)
    low_times = df_kf["low_time"].values if "low_time" in df_kf.columns else np.arange(n)
    bar_timestamps = df_kf["start_time"].values if "start_time" in df_kf.columns else np.arange(n) * 60000

    # Duration 계산
    log_duration = df_kf["log_duration"].values if "log_duration" in df_kf.columns else np.zeros(n)
    duration = np.exp(log_duration)
    duration_ma = pd.Series(duration).rolling(params["duration_ma_window"], min_periods=10).mean().values

    # Tick count
    if "log_tick_count" in df_kf.columns:
        tick_count = np.exp(df_kf["log_tick_count"].values)
        tick_p25 = pd.Series(tick_count).rolling(100, min_periods=20).quantile(0.25).shift(1).bfill().values
    else:
        tick_count = np.ones(n)
        tick_p25 = np.ones(n)

    # Stop distance 계산
    parkinson = calculate_parkinson_vol(high, low, params["parkinson_window"])
    compression = calculate_compression(log_duration, params["compression_window"])
    stop_distance = params["K"] * parkinson * np.sqrt(params["horizon"]) * np.sqrt(compression)

    # Parameters
    warmup = params["warmup"]
    fee = params["fee"]
    slippage = params["slippage"]
    risk_per_trade = params["risk_per_trade"]
    max_position_size = params["max_position_size"]
    min_position_size = params["min_position_size"]

    # Pyramid parameters
    pyramid_threshold = params.get("pyramid_threshold", 0.75)
    max_pyramids = params.get("max_pyramids", 5)
    pyramid_size_ratio = params.get("pyramid_size_ratio", 0.5)
    max_total_size = params.get("max_total_size", 2.5)

    # State
    trades: List[Trade] = []
    initial_capital = params.get("initial_capital", 100000.0)
    equity = initial_capital
    mtm_equity = np.full(n, np.nan)

    in_position = False
    current_direction = 0  # 1=Long, -1=Short
    current_entries: List[PyramidEntry] = []
    current_stop = 0.0
    current_tp = 0.0
    highest = 0.0
    lowest = float('inf')
    initial_sd = 0.0
    initial_entry_size = 0.0
    pyramid_count = 0
    next_pyramid_price = 0.0

    def get_avg_entry():
        if not current_entries:
            return 0.0, 0.0
        total_cost = sum(e.price * e.size for e in current_entries)
        total_size = sum(e.size for e in current_entries)
        return total_cost / total_size if total_size > 0 else 0.0, total_size

    for bar in range(warmup, n):
        sd = stop_distance[bar]

        # 시그널 체크
        long_signal = get_long_signal(
            long_signal_type,
            fast_zscore[bar],
            slow_zscore[bar],
            duration[bar],
            duration_ma[bar],
            params
        )

        short_signal = get_short_signal(
            short_signal_type,
            fast_zscore[bar],
            slow_zscore[bar],
            spread[bar],
            spread_std[bar],
            tick_count[bar],
            tick_p25[bar],
            params
        )

        if not in_position:
            mtm_equity[bar] = equity

            if np.isnan(sd) or sd <= 0:
                continue

            # 다음 bar open에서 진입
            entry_bar = bar + 1
            if entry_bar >= n:
                continue

            new_direction = 0
            if long_signal:
                new_direction = 1
            elif short_signal:
                new_direction = -1

            if new_direction == 0:
                continue

            if new_direction == 1:
                entry_price = open_prices[entry_bar] * (1 + slippage)
            else:
                entry_price = open_prices[entry_bar] * (1 - slippage)

            # Position sizing
            position_size = risk_per_trade / (sd + 1e-10)
            position_size = np.clip(position_size, min_position_size, max_position_size)

            # Initialize position
            current_entries = [PyramidEntry(bar=entry_bar, price=entry_price, size=position_size)]
            in_position = True
            current_direction = new_direction
            initial_sd = sd
            initial_entry_size = position_size
            pyramid_count = 1

            if new_direction == 1:  # Long
                highest = high[entry_bar] if entry_bar < n else entry_price
                current_stop = entry_price * (1 - sd)
                current_tp = entry_price * (1 + sd)
                next_pyramid_price = entry_price * (1 + pyramid_threshold * sd)
            else:  # Short
                lowest = low[entry_bar] if entry_bar < n else entry_price
                current_stop = entry_price * (1 + sd)
                current_tp = entry_price * (1 - sd)
                next_pyramid_price = entry_price * (1 - pyramid_threshold * sd)

        else:
            # In position

            # 1. Check for pyramid (if enabled)
            if enable_pyramid:
                total_size = sum(e.size for e in current_entries)
                can_pyramid = (
                    pyramid_count < max_pyramids and
                    total_size < max_total_size
                )

                pyramid_triggered = False
                if can_pyramid:
                    if current_direction == 1 and high[bar] >= next_pyramid_price:
                        pyramid_triggered = True
                        pyramid_price = next_pyramid_price
                    elif current_direction == -1 and low[bar] <= next_pyramid_price:
                        pyramid_triggered = True
                        pyramid_price = next_pyramid_price

                if pyramid_triggered:
                    pyr_size = initial_entry_size * pyramid_size_ratio
                    remaining = max_total_size - total_size
                    pyr_size = min(pyr_size, remaining)

                    if pyr_size > 0:
                        if current_direction == 1:
                            actual_pyr_price = pyramid_price * (1 + slippage)
                        else:
                            actual_pyr_price = pyramid_price * (1 - slippage)

                        current_entries.append(PyramidEntry(bar=bar, price=actual_pyr_price, size=pyr_size))
                        pyramid_count += 1

                        trail_amount = pyramid_threshold * initial_sd
                        if current_direction == 1:
                            current_stop *= (1 + trail_amount)
                            current_tp *= (1 + trail_amount)
                            next_pyramid_price = pyramid_price * (1 + pyramid_threshold * initial_sd)
                        else:
                            current_stop *= (1 - trail_amount)
                            current_tp *= (1 - trail_amount)
                            next_pyramid_price = pyramid_price * (1 - pyramid_threshold * initial_sd)

            # 2. MTM tracking
            avg_entry, total_size = get_avg_entry()
            if current_direction == 1:
                unrealized_pnl = (low[bar] - avg_entry) / avg_entry
            else:
                unrealized_pnl = (avg_entry - high[bar]) / avg_entry
            mtm_equity[bar] = equity * (1 + unrealized_pnl * total_size)

            # 3. Check exit
            exit_price = None
            exit_reason = None

            if current_direction == 1:  # Long
                stop_hit = low[bar] <= current_stop
                tp_hit = high[bar] >= current_tp

                if stop_hit and tp_hit:
                    if low_times[bar] <= high_times[bar]:
                        exit_reason, exit_price = "stop", current_stop
                    else:
                        exit_reason, exit_price = "tp", current_tp
                elif stop_hit:
                    exit_reason, exit_price = "stop", current_stop
                elif tp_hit:
                    exit_reason, exit_price = "tp", current_tp
            else:  # Short
                stop_hit = high[bar] >= current_stop
                tp_hit = low[bar] <= current_tp

                if stop_hit and tp_hit:
                    if high_times[bar] <= low_times[bar]:
                        exit_reason, exit_price = "stop", current_stop
                    else:
                        exit_reason, exit_price = "tp", current_tp
                elif stop_hit:
                    exit_reason, exit_price = "stop", current_stop
                elif tp_hit:
                    exit_reason, exit_price = "tp", current_tp

            # 4. Process exit
            if exit_price is not None:
                avg_entry, total_size = get_avg_entry()

                if current_direction == 1:
                    actual_exit = exit_price * (1 - slippage)
                    gross_return = (exit_price - avg_entry) / avg_entry
                else:
                    actual_exit = exit_price * (1 + slippage)
                    gross_return = (avg_entry - exit_price) / avg_entry

                net_return = gross_return - fee * (len(current_entries) + 1) - slippage * 2

                trade = Trade(
                    direction=current_direction,
                    entry_bar=current_entries[0].bar,
                    entry_price=avg_entry,
                    exit_bar=bar,
                    exit_price=actual_exit,
                    exit_reason=exit_reason,
                    size=total_size,
                    gross_return=gross_return,
                    net_return=net_return,
                    entries=current_entries.copy() if enable_pyramid else None,
                )
                trades.append(trade)

                equity *= (1 + net_return * total_size)

                # Reset
                in_position = False
                current_direction = 0
                current_entries = []
                current_stop = 0.0
                current_tp = 0.0
                highest = 0.0
                lowest = float('inf')
                pyramid_count = 0

    # Calculate metrics
    valid_mtm = mtm_equity[~np.isnan(mtm_equity)]
    equity_arr = valid_mtm if len(valid_mtm) > 0 else np.array([initial_capital, equity])

    total_trades = len(trades)
    if total_trades == 0:
        return None

    long_trades = [t for t in trades if t.direction == 1]
    short_trades = [t for t in trades if t.direction == -1]

    wins = sum(1 for t in trades if t.net_return > 0)
    win_rate = wins / total_trades

    bars_per_year = 365 * 24 / 4
    total_bars = n - warmup
    years = total_bars / bars_per_year
    total_return = equity_arr[-1] / equity_arr[0]
    cagr = total_return ** (1 / max(years, 0.01)) - 1

    running_max = np.maximum.accumulate(equity_arr)
    drawdown = (running_max - equity_arr) / running_max
    max_dd = np.max(drawdown)

    returns = np.diff(equity_arr) / equity_arr[:-1]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(bars_per_year) if len(returns) > 0 else 0

    avg_return = np.mean([t.net_return for t in trades])

    # Long/Short win rates
    long_wr = sum(1 for t in long_trades if t.net_return > 0) / len(long_trades) if long_trades else 0
    short_wr = sum(1 for t in short_trades if t.net_return > 0) / len(short_trades) if short_trades else 0

    # Pyramid stats
    if enable_pyramid:
        avg_entries = np.mean([len(t.entries) if t.entries else 1 for t in trades])
    else:
        avg_entries = 1.0

    return {
        "symbol": symbol,
        "long_signal": long_signal_type.value,
        "short_signal": short_signal_type.value,
        "total_trades": total_trades,
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "win_rate": win_rate,
        "long_wr": long_wr,
        "short_wr": short_wr,
        "cagr": cagr,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "calmar": cagr / max_dd if max_dd > 0 else 0,
        "avg_return": avg_return,
        "mtm_equity": equity_arr,
        "timestamps": bar_timestamps[~np.isnan(mtm_equity)],
        "trades": trades,
        "years": years,
        "avg_entries": avg_entries,
    }


def run_asymmetric_test():
    """비대칭 전략 테스트 (Long: VelTrend, Short: Spread)"""
    print("=" * 130)
    print("V9 Asymmetric Strategy - Long + Short with Pyramid")
    print("=" * 130)
    print(f"\nLong Signal: VelTrend (fast_z > 2 AND duration < ma*0.5 AND slow_z > 1)")
    print(f"Short Signal: Spread (fast_z < -2 AND spread < -1.0 * spread_std)")

    long_signal = SignalType.VEL_TREND
    short_signal = ShortSignalType.SPREAD

    print(f"\n{'Symbol':<10} {'Total':>6} {'Long':>5} {'Short':>5} {'WR':>6} {'L_WR':>6} {'S_WR':>6} "
          f"{'CAGR':>9} {'MDD':>7} {'Sharpe':>7} {'Calmar':>7}")
    print("-" * 100)

    results = []
    for symbol in SYMBOLS:
        r = run_asymmetric_backtest(symbol, long_signal, short_signal, enable_pyramid=True)
        if r:
            results.append(r)
            print(f"{r['symbol']:<10} {r['total_trades']:>6} {r['long_trades']:>5} {r['short_trades']:>5} "
                  f"{r['win_rate']*100:>5.1f}% {r['long_wr']*100:>5.1f}% {r['short_wr']*100:>5.1f}% "
                  f"{r['cagr']*100:>8.1f}% {r['max_dd']*100:>6.1f}% "
                  f"{r['sharpe']:>7.2f} {r['calmar']:>7.2f}")

    if len(results) >= 2:
        portfolio = calculate_portfolio_metrics(results)
        if portfolio:
            total_long = sum(r['long_trades'] for r in results)
            total_short = sum(r['short_trades'] for r in results)
            avg_long_wr = np.mean([r['long_wr'] for r in results])
            avg_short_wr = np.mean([r['short_wr'] for r in results])
            print("-" * 100)
            print(f"{'Portfolio':<10} {portfolio['total_trades']:>6} {total_long:>5} {total_short:>5} "
                  f"{portfolio['avg_win_rate']*100:>5.1f}% {avg_long_wr*100:>5.1f}% {avg_short_wr*100:>5.1f}% "
                  f"{portfolio['cagr']*100:>8.1f}% {portfolio['max_dd']*100:>6.1f}% "
                  f"{portfolio['sharpe']:>7.2f} {portfolio['calmar']:>7.2f}")

    # Compare with Long Only
    print("\n" + "=" * 130)
    print("COMPARISON: Long Only vs Asymmetric (Long + Short)")
    print("=" * 130)

    results_long = []
    for symbol in SYMBOLS:
        r = run_backtest(symbol, SignalType.VEL_TREND, enable_pyramid=True)
        if r:
            results_long.append(r)

    portfolio_long = calculate_portfolio_metrics(results_long) if results_long else None
    portfolio_asym = calculate_portfolio_metrics(results) if results else None

    if portfolio_long and portfolio_asym:
        print(f"\n{'Metric':<15} {'Long Only':>15} {'Asymmetric':>15} {'Change':>15}")
        print("-" * 60)
        print(f"{'CAGR':<15} {portfolio_long['cagr']*100:>14.1f}% {portfolio_asym['cagr']*100:>14.1f}% "
              f"{(portfolio_asym['cagr']-portfolio_long['cagr'])*100:>+14.1f}%")
        print(f"{'MDD':<15} {portfolio_long['max_dd']*100:>14.1f}% {portfolio_asym['max_dd']*100:>14.1f}% "
              f"{(portfolio_asym['max_dd']-portfolio_long['max_dd'])*100:>+14.1f}%")
        print(f"{'Sharpe':<15} {portfolio_long['sharpe']:>15.2f} {portfolio_asym['sharpe']:>15.2f} "
              f"{portfolio_asym['sharpe']-portfolio_long['sharpe']:>+15.2f}")
        print(f"{'Calmar':<15} {portfolio_long['calmar']:>15.2f} {portfolio_asym['calmar']:>15.2f} "
              f"{portfolio_asym['calmar']-portfolio_long['calmar']:>+15.2f}")
        print(f"{'Trades':<15} {portfolio_long['total_trades']:>15} {portfolio_asym['total_trades']:>15} "
              f"{portfolio_asym['total_trades']-portfolio_long['total_trades']:>+15}")

    return results


def run_pyramid_comparison():
    """피라미딩 적용 전후 비교 (VelTrend 시그널)"""
    print("=" * 130)
    print("V9 Long Only - Pyramid Comparison (VelTrend Signal)")
    print("=" * 130)
    print(f"\nPyramid: threshold={PARAMS['pyramid_threshold']}x SD, max={PARAMS['max_pyramids']}, size_ratio={PARAMS['pyramid_size_ratio']}")

    signal_type = SignalType.VEL_TREND

    # No Pyramid
    print("\n--- No Pyramid ---")
    print(f"{'Symbol':<10} {'Trades':>7} {'WR':>7} {'CAGR':>10} {'MDD':>8} {'Sharpe':>8} {'Calmar':>8}")
    print("-" * 70)

    results_no_pyr = []
    for symbol in SYMBOLS:
        r = run_backtest(symbol, signal_type, enable_pyramid=False)
        if r:
            results_no_pyr.append(r)
            print(f"{r['symbol']:<10} {r['total_trades']:>7} {r['win_rate']*100:>6.1f}% "
                  f"{r['cagr']*100:>9.1f}% {r['max_dd']*100:>7.1f}% "
                  f"{r['sharpe']:>8.2f} {r['calmar']:>8.2f}")

    if len(results_no_pyr) >= 2:
        portfolio_no_pyr = calculate_portfolio_metrics(results_no_pyr)
        if portfolio_no_pyr:
            print("-" * 70)
            print(f"{'Portfolio':<10} {portfolio_no_pyr['total_trades']:>7} {portfolio_no_pyr['avg_win_rate']*100:>6.1f}% "
                  f"{portfolio_no_pyr['cagr']*100:>9.1f}% {portfolio_no_pyr['max_dd']*100:>7.1f}% "
                  f"{portfolio_no_pyr['sharpe']:>8.2f} {portfolio_no_pyr['calmar']:>8.2f}")

    # With Pyramid
    print("\n--- With Pyramid ---")
    print(f"{'Symbol':<10} {'Trades':>7} {'WR':>7} {'CAGR':>10} {'MDD':>8} {'Sharpe':>8} {'Calmar':>8} {'AvgEnt':>7} {'Pyr%':>6}")
    print("-" * 95)

    results_pyr = []
    for symbol in SYMBOLS:
        r = run_backtest(symbol, signal_type, enable_pyramid=True)
        if r:
            results_pyr.append(r)
            print(f"{r['symbol']:<10} {r['total_trades']:>7} {r['win_rate']*100:>6.1f}% "
                  f"{r['cagr']*100:>9.1f}% {r['max_dd']*100:>7.1f}% "
                  f"{r['sharpe']:>8.2f} {r['calmar']:>8.2f} "
                  f"{r['avg_entries']:>6.2f} {r['pyramid_rate']*100:>5.1f}%")

    if len(results_pyr) >= 2:
        portfolio_pyr = calculate_portfolio_metrics(results_pyr)
        if portfolio_pyr:
            avg_entries = np.mean([r['avg_entries'] for r in results_pyr])
            avg_pyr_rate = np.mean([r['pyramid_rate'] for r in results_pyr])
            print("-" * 95)
            print(f"{'Portfolio':<10} {portfolio_pyr['total_trades']:>7} {portfolio_pyr['avg_win_rate']*100:>6.1f}% "
                  f"{portfolio_pyr['cagr']*100:>9.1f}% {portfolio_pyr['max_dd']*100:>7.1f}% "
                  f"{portfolio_pyr['sharpe']:>8.2f} {portfolio_pyr['calmar']:>8.2f} "
                  f"{avg_entries:>6.2f} {avg_pyr_rate*100:>5.1f}%")

    # Summary comparison
    if portfolio_no_pyr and portfolio_pyr:
        print("\n" + "=" * 130)
        print("COMPARISON: No Pyramid vs With Pyramid")
        print("=" * 130)
        print(f"{'Metric':<15} {'No Pyramid':>15} {'With Pyramid':>15} {'Change':>15}")
        print("-" * 60)
        print(f"{'CAGR':<15} {portfolio_no_pyr['cagr']*100:>14.1f}% {portfolio_pyr['cagr']*100:>14.1f}% "
              f"{(portfolio_pyr['cagr']-portfolio_no_pyr['cagr'])*100:>+14.1f}%")
        print(f"{'MDD':<15} {portfolio_no_pyr['max_dd']*100:>14.1f}% {portfolio_pyr['max_dd']*100:>14.1f}% "
              f"{(portfolio_pyr['max_dd']-portfolio_no_pyr['max_dd'])*100:>+14.1f}%")
        print(f"{'Sharpe':<15} {portfolio_no_pyr['sharpe']:>15.2f} {portfolio_pyr['sharpe']:>15.2f} "
              f"{portfolio_pyr['sharpe']-portfolio_no_pyr['sharpe']:>+15.2f}")
        print(f"{'Calmar':<15} {portfolio_no_pyr['calmar']:>15.2f} {portfolio_pyr['calmar']:>15.2f} "
              f"{portfolio_pyr['calmar']-portfolio_no_pyr['calmar']:>+15.2f}")

    return {"no_pyramid": portfolio_no_pyr, "with_pyramid": portfolio_pyr}


def run_signal_pyramid_test():
    """시그널 기반 피라미딩 테스트"""
    print("=" * 130)
    print("V9 Long Only - Signal Pyramid Comparison")
    print("=" * 130)

    test_cases = [
        (False, "Price Only (현재)"),
        (True, "Price + Signal"),
    ]

    all_results = {}

    for signal_pyr, name in test_cases:
        test_params = PARAMS.copy()
        test_params["signal_pyramid"] = signal_pyr

        print(f"\n--- {name} ---")
        print(f"{'Symbol':<10} {'Trades':>7} {'WR':>7} {'CAGR':>10} {'MDD':>8} {'Sharpe':>8} {'Calmar':>8} {'AvgEnt':>7}")
        print("-" * 80)

        results = []
        for symbol in SYMBOLS:
            r = run_backtest(symbol, SignalType.VEL_TREND, params=test_params, enable_pyramid=True)
            if r:
                results.append(r)
                print(f"{r['symbol']:<10} {r['total_trades']:>7} {r['win_rate']*100:>6.1f}% "
                      f"{r['cagr']*100:>9.1f}% {r['max_dd']*100:>7.1f}% "
                      f"{r['sharpe']:>8.2f} {r['calmar']:>8.2f} {r['avg_entries']:>6.2f}")

        if len(results) >= 2:
            portfolio = calculate_portfolio_metrics(results)
            if portfolio:
                avg_entries = np.mean([r['avg_entries'] for r in results])
                print("-" * 80)
                print(f"{'Portfolio':<10} {portfolio['total_trades']:>7} {portfolio['avg_win_rate']*100:>6.1f}% "
                      f"{portfolio['cagr']*100:>9.1f}% {portfolio['max_dd']*100:>7.1f}% "
                      f"{portfolio['sharpe']:>8.2f} {portfolio['calmar']:>8.2f} {avg_entries:>6.2f}")
                all_results[name] = {**portfolio, "avg_entries": avg_entries}

    # Summary
    if len(all_results) >= 2:
        print("\n" + "=" * 130)
        print("COMPARISON")
        print("=" * 130)
        p1 = all_results.get("Price Only (현재)", {})
        p2 = all_results.get("Price + Signal", {})
        if p1 and p2:
            print(f"{'Metric':<15} {'Price Only':>15} {'Price+Signal':>15} {'Change':>15}")
            print("-" * 60)
            print(f"{'CAGR':<15} {p1['cagr']*100:>14.1f}% {p2['cagr']*100:>14.1f}% "
                  f"{(p2['cagr']-p1['cagr'])*100:>+14.1f}%")
            print(f"{'MDD':<15} {p1['max_dd']*100:>14.1f}% {p2['max_dd']*100:>14.1f}% "
                  f"{(p2['max_dd']-p1['max_dd'])*100:>+14.1f}%")
            print(f"{'Sharpe':<15} {p1['sharpe']:>15.2f} {p2['sharpe']:>15.2f} "
                  f"{p2['sharpe']-p1['sharpe']:>+15.2f}")
            print(f"{'Calmar':<15} {p1['calmar']:>15.2f} {p2['calmar']:>15.2f} "
                  f"{p2['calmar']-p1['calmar']:>+15.2f}")
            print(f"{'Trades':<15} {p1['total_trades']:>15} {p2['total_trades']:>15} "
                  f"{p2['total_trades']-p1['total_trades']:>+15}")
            print(f"{'Avg Entries':<15} {p1['avg_entries']:>15.2f} {p2['avg_entries']:>15.2f} "
                  f"{p2['avg_entries']-p1['avg_entries']:>+15.2f}")

    return all_results


def run_threshold_sweep():
    """진입 조건 완화 테스트"""
    print("=" * 130)
    print("V9 Long Only - Threshold Sweep (With Pyramid)")
    print("=" * 130)

    # 테스트할 파라미터 조합
    test_cases = [
        # (fast_z, slow_z, dur_ratio, name)
        (2.0, 1.0, 0.5, "Strict (현재)"),
        (1.5, 1.0, 0.5, "Fast 1.5"),
        (2.0, 0.5, 0.5, "Slow 0.5"),
        (1.5, 0.5, 0.5, "Fast 1.5 + Slow 0.5"),
        (2.0, 1.0, 0.7, "Duration 0.7"),
        (1.5, 0.5, 0.7, "Relaxed All"),
        (1.5, 0.0, 0.7, "No Slow Filter"),
        (1.0, 0.0, 1.0, "Very Loose"),
    ]

    all_results = {}

    for fast_z, slow_z, dur_ratio, name in test_cases:
        test_params = PARAMS.copy()
        test_params["fast_zscore_threshold"] = fast_z
        test_params["slow_zscore_threshold"] = slow_z
        test_params["duration_ratio"] = dur_ratio

        print(f"\n--- {name} (fast>{fast_z}, slow>{slow_z}, dur<{dur_ratio}) ---")
        print(f"{'Symbol':<10} {'Trades':>7} {'WR':>7} {'CAGR':>10} {'MDD':>8} {'Sharpe':>8} {'Calmar':>8}")
        print("-" * 70)

        results = []
        for symbol in SYMBOLS:
            r = run_backtest(symbol, SignalType.VEL_TREND, params=test_params, enable_pyramid=True)
            if r:
                results.append(r)
                print(f"{r['symbol']:<10} {r['total_trades']:>7} {r['win_rate']*100:>6.1f}% "
                      f"{r['cagr']*100:>9.1f}% {r['max_dd']*100:>7.1f}% "
                      f"{r['sharpe']:>8.2f} {r['calmar']:>8.2f}")

        if len(results) >= 2:
            portfolio = calculate_portfolio_metrics(results)
            if portfolio:
                print("-" * 70)
                print(f"{'Portfolio':<10} {portfolio['total_trades']:>7} {portfolio['avg_win_rate']*100:>6.1f}% "
                      f"{portfolio['cagr']*100:>9.1f}% {portfolio['max_dd']*100:>7.1f}% "
                      f"{portfolio['sharpe']:>8.2f} {portfolio['calmar']:>8.2f}")
                all_results[name] = portfolio

    # Summary
    print("\n" + "=" * 130)
    print("SUMMARY: Threshold Comparison")
    print("=" * 130)
    print(f"{'Setting':<25} {'Trades':>8} {'WR':>7} {'CAGR':>10} {'MDD':>8} {'Sharpe':>8} {'Calmar':>8}")
    print("-" * 85)

    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["calmar"], reverse=True)
    for name, metrics in sorted_results:
        print(f"{name:<25} {metrics['total_trades']:>8} {metrics['avg_win_rate']*100:>6.1f}% "
              f"{metrics['cagr']*100:>9.1f}% {metrics['max_dd']*100:>7.1f}% "
              f"{metrics['sharpe']:>8.2f} {metrics['calmar']:>8.2f}")

    return all_results


def run_or_signal_test():
    """OR 조합 시그널 테스트 (피라미딩 적용)"""
    print("=" * 130)
    print("V9 Long Only - OR Signal Combinations (With Pyramid)")
    print("=" * 130)

    or_signals = [
        SignalType.OR_VEL_TREND,   # VelVolume OR TrendAlign
        SignalType.OR_VEL_SLOW,    # VelVolume OR SlowZ
        SignalType.OR_FAST_SLOW,   # FastOnly OR SlowZ
        SignalType.OR_ALL,         # VelVolume OR TrendAlign OR SlowZ
    ]

    # 비교용: AND 조합 최고 성과
    and_signals = [
        SignalType.VEL_TREND,      # VelVolume AND TrendAlign (기존 최고)
    ]

    all_results = {}

    for signal_type in and_signals + or_signals:
        print(f"\n--- {signal_type.value} ---")
        print(f"{'Symbol':<10} {'Trades':>7} {'WR':>7} {'CAGR':>10} {'MDD':>8} {'Sharpe':>8} {'Calmar':>8}")
        print("-" * 70)

        results = []
        for symbol in SYMBOLS:
            r = run_backtest(symbol, signal_type, enable_pyramid=True)
            if r:
                results.append(r)
                print(f"{r['symbol']:<10} {r['total_trades']:>7} {r['win_rate']*100:>6.1f}% "
                      f"{r['cagr']*100:>9.1f}% {r['max_dd']*100:>7.1f}% "
                      f"{r['sharpe']:>8.2f} {r['calmar']:>8.2f}")

        if len(results) >= 2:
            portfolio = calculate_portfolio_metrics(results)
            if portfolio:
                print("-" * 70)
                print(f"{'Portfolio':<10} {portfolio['total_trades']:>7} {portfolio['avg_win_rate']*100:>6.1f}% "
                      f"{portfolio['cagr']*100:>9.1f}% {portfolio['max_dd']*100:>7.1f}% "
                      f"{portfolio['sharpe']:>8.2f} {portfolio['calmar']:>8.2f}")
                all_results[signal_type.value] = portfolio

    # Summary
    print("\n" + "=" * 130)
    print("SUMMARY: AND vs OR Signal Comparison (Portfolio, With Pyramid)")
    print("=" * 130)
    print(f"{'Signal':<20} {'Trades':>8} {'WR':>7} {'CAGR':>10} {'MDD':>8} {'Sharpe':>8} {'Calmar':>8}")
    print("-" * 80)

    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["calmar"], reverse=True)
    for signal_name, metrics in sorted_results:
        marker = "**" if "OR" in signal_name else ""
        print(f"{marker}{signal_name:<18} {metrics['total_trades']:>8} {metrics['avg_win_rate']*100:>6.1f}% "
              f"{metrics['cagr']*100:>9.1f}% {metrics['max_dd']*100:>7.1f}% "
              f"{metrics['sharpe']:>8.2f} {metrics['calmar']:>8.2f}")

    return all_results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "pyramid":
            run_pyramid_comparison()
        elif sys.argv[1] == "asymmetric":
            run_asymmetric_test()
        elif sys.argv[1] == "or":
            run_or_signal_test()
        elif sys.argv[1] == "threshold":
            run_threshold_sweep()
        elif sys.argv[1] == "signal":
            run_signal_pyramid_test()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python strategy_v9_long.py [pyramid|asymmetric|or|threshold|signal]")
    else:
        run_signal_comparison()
