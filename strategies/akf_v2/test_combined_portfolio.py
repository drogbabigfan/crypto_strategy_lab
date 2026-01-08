#!/usr/bin/env python3
"""
4-Asset Combined Portfolio 백테스트

레버리지 누적 방식으로 계산:
1. 각 자산별 PnL ($) 계산
2. 시간축 정렬
3. PnL 합산 → Equity 갱신
4. 레버리지 체크
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from numba import njit

from research.features.kalman import calculate_adaptive_kalman
from strategies.akf_v2.strategy_v33_final import load_data, V33_PARAMS, calculate_position_size, calculate_features

DATA_ROOT = PROJECT_ROOT / "etl/data"
PORTFOLIO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"]


def calculate_intensity(df: pd.DataFrame, baseline_window: int = 20):
    """Flow Intensity 계산"""
    duration = np.exp(df["log_duration"].values)
    baseline_duration = pd.Series(duration).rolling(window=baseline_window, min_periods=5).mean().values
    epsilon = 1e-6
    intensity = baseline_duration / (duration + epsilon)
    return intensity


@njit
def run_backtest_with_pnl_tracking(
    close, high, low, open_prices, kf_trend, vel_zscore, unc_pct, sigma_hybrid, dynamic_mult,
    v_ratio, resid_std, intensity,
    entry_z, unc_pct_max, v_ratio_threshold, intensity_threshold,
    innov_base_mult, innov_mult_min, innov_mult_max,
    hard_stop_mult, risk_target, conf_lambda, size_min, size_max, warmup,
    initial_capital, fee_rate, slippage_rate
):
    """
    PnL ($) 및 Position Size ($) 추적 백테스트

    Returns:
        bar_pnl: 각 바의 실현 PnL ($)
        bar_notional: 각 바의 포지션 Notional Size ($)
        equity_curve: 자산 곡선
    """
    n = len(close)

    # 결과 배열
    bar_pnl = np.zeros(n)          # 각 바의 PnL ($)
    bar_notional = np.zeros(n)     # 각 바의 Notional Size ($)
    equity_curve = np.zeros(n)

    # 상태
    equity = initial_capital
    position = 0
    entry_price = 0.0
    current_size = 0.0  # 비율
    highest = 0.0
    lowest = np.inf
    prev_stop = 0.0
    hard_stop = 0.0

    for i in range(warmup, n):
        equity_curve[i] = equity

        innov_mult = innov_base_mult / (v_ratio[i] + 1e-10)
        innov_mult = max(innov_mult_min, min(innov_mult_max, innov_mult))
        log_resid = np.log(close[i]) - np.log(kf_trend[i])

        if position == 0:
            # Entry 체크
            if vel_zscore[i] > entry_z and unc_pct[i] < unc_pct_max:
                position = 1
                entry_price = open_prices[i] * (1 + slippage_rate)
                highest = high[i]
                current_size = calculate_position_size(
                    sigma_hybrid[i], unc_pct[i], risk_target, hard_stop_mult, conf_lambda, size_min, size_max
                )
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = highest * np.exp(-dist)
                hard_stop = entry_price * np.exp(-hard_stop_mult * sigma_hybrid[i])

                # Notional Size 기록
                bar_notional[i] = equity * current_size

            elif vel_zscore[i] < -entry_z and unc_pct[i] < unc_pct_max:
                position = -1
                entry_price = open_prices[i] * (1 - slippage_rate)
                lowest = low[i]
                current_size = calculate_position_size(
                    sigma_hybrid[i], unc_pct[i], risk_target, hard_stop_mult, conf_lambda, size_min, size_max
                )
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = lowest * np.exp(dist)
                hard_stop = entry_price * np.exp(hard_stop_mult * sigma_hybrid[i])

                bar_notional[i] = equity * current_size

        elif position == 1:
            # Notional Size 기록
            notional = equity * current_size
            bar_notional[i] = notional

            if high[i] > highest:
                highest = high[i]
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_stop = highest * np.exp(-dist)
            trail_stop = max(new_stop, prev_stop)
            prev_stop = trail_stop

            # Exit 체크
            exit_price = 0.0
            should_exit = False

            # Hard Stop
            if close[i] < hard_stop:
                exit_price = hard_stop * (1 - slippage_rate)
                should_exit = True
            # Trailing Stop
            elif close[i] < trail_stop:
                exit_price = trail_stop * (1 - slippage_rate)
                should_exit = True
            # Signal Exit with Intensity Filter
            elif v_ratio[i] < v_ratio_threshold and vel_zscore[i] < -entry_z:
                if intensity[i] < intensity_threshold:
                    exit_price = close[i] * (1 - slippage_rate)
                    should_exit = True
            # Innovation Breaker
            elif v_ratio[i] >= v_ratio_threshold:
                if log_resid < -innov_mult * resid_std[i]:
                    exit_price = close[i] * (1 - slippage_rate)
                    should_exit = True

            if should_exit:
                # PnL 계산 ($)
                price_return = (exit_price - entry_price) / entry_price
                gross_pnl = notional * price_return
                fee = notional * fee_rate * 2  # entry + exit
                net_pnl = gross_pnl - fee

                bar_pnl[i] = net_pnl
                equity += net_pnl
                position = 0
            else:
                # 미실현 손익은 PnL에 포함 안함 (실현 기준)
                pass

        elif position == -1:
            notional = equity * current_size
            bar_notional[i] = notional

            if low[i] < lowest:
                lowest = low[i]
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_stop = lowest * np.exp(dist)
            trail_stop = min(new_stop, prev_stop)
            prev_stop = trail_stop

            exit_price = 0.0
            should_exit = False

            if close[i] > hard_stop:
                exit_price = hard_stop * (1 + slippage_rate)
                should_exit = True
            elif close[i] > trail_stop:
                exit_price = trail_stop * (1 + slippage_rate)
                should_exit = True
            elif v_ratio[i] < v_ratio_threshold and vel_zscore[i] > entry_z:
                if intensity[i] < intensity_threshold:
                    exit_price = close[i] * (1 + slippage_rate)
                    should_exit = True
            elif v_ratio[i] >= v_ratio_threshold:
                if log_resid > innov_mult * resid_std[i]:
                    exit_price = close[i] * (1 + slippage_rate)
                    should_exit = True

            if should_exit:
                price_return = (entry_price - exit_price) / entry_price
                gross_pnl = notional * price_return
                fee = notional * fee_rate * 2
                net_pnl = gross_pnl - fee

                bar_pnl[i] = net_pnl
                equity += net_pnl
                position = 0

    # 마지막 equity 업데이트
    equity_curve[n-1] = equity

    return bar_pnl, bar_notional, equity_curve


def run_single_asset_backtest(symbol: str, initial_capital: float = 100000.0, intensity_threshold: float = 4.0):
    """단일 자산 백테스트 실행, 타임스탬프와 함께 PnL/Notional 반환"""
    df = load_data(symbol)
    if df is None:
        return None

    df_kf = calculate_adaptive_kalman(df)
    features = calculate_features(df_kf)
    intensity = calculate_intensity(df)

    params = V33_PARAMS

    bar_pnl, bar_notional, equity_curve = run_backtest_with_pnl_tracking(
        close=df_kf["close"].values,
        high=df_kf["high"].values,
        low=df_kf["low"].values,
        open_prices=df_kf["open"].values,
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
        intensity_threshold=intensity_threshold,
        innov_base_mult=params["innov_base_mult"],
        innov_mult_min=params["innov_mult_min"],
        innov_mult_max=params["innov_mult_max"],
        hard_stop_mult=params["hard_stop_mult"],
        risk_target=params["risk_target"],
        conf_lambda=params["conf_lambda"],
        size_min=params["size_min"],
        size_max=params["size_max"],
        warmup=params["warmup"],
        initial_capital=initial_capital,
        fee_rate=0.001,
        slippage_rate=0.0001,
    )

    # DataFrame으로 반환 (타임스탬프 포함)
    result_df = pd.DataFrame({
        "timestamp": df_kf["open_time"].values if "open_time" in df_kf.columns else range(len(bar_pnl)),
        "pnl": bar_pnl,
        "notional": bar_notional,
        "equity": equity_curve,
        "close": df_kf["close"].values,
    })

    return result_df


def combine_portfolios(asset_data: dict, initial_capital: float = 100000.0, max_leverage: float = 10.0):
    """
    여러 자산 포트폴리오 합산

    Args:
        asset_data: {symbol: DataFrame with pnl, notional, timestamp}
        initial_capital: 초기 자본
        max_leverage: 최대 허용 레버리지
    """
    # Step 1: 시간축 정렬
    # 모든 자산의 timestamp를 합쳐서 유니온 생성
    all_timestamps = set()
    for symbol, df in asset_data.items():
        if "timestamp" in df.columns:
            all_timestamps.update(df["timestamp"].values)

    if not all_timestamps:
        # timestamp가 없으면 인덱스 기반으로
        max_len = max(len(df) for df in asset_data.values())
        all_timestamps = list(range(max_len))
    else:
        all_timestamps = sorted(all_timestamps)

    # Step 2 & 3: PnL 합산 및 Equity 갱신
    combined_pnl = np.zeros(len(all_timestamps))
    combined_notional = np.zeros(len(all_timestamps))

    # 각 자산별 데이터를 timestamp 기준으로 매핑
    for symbol, df in asset_data.items():
        if "timestamp" in df.columns:
            ts_to_idx = {ts: i for i, ts in enumerate(all_timestamps)}
            for _, row in df.iterrows():
                if row["timestamp"] in ts_to_idx:
                    idx = ts_to_idx[row["timestamp"]]
                    combined_pnl[idx] += row["pnl"]
                    combined_notional[idx] += abs(row["notional"])
        else:
            # 인덱스 기반
            for i in range(len(df)):
                if i < len(combined_pnl):
                    combined_pnl[i] += df["pnl"].iloc[i]
                    combined_notional[i] += abs(df["notional"].iloc[i])

    # Equity curve 계산
    equity_curve = np.zeros(len(all_timestamps))
    equity = initial_capital

    leverage_history = np.zeros(len(all_timestamps))

    for i in range(len(all_timestamps)):
        equity += combined_pnl[i]
        equity_curve[i] = equity

        # Step 4: 레버리지 체크
        if equity > 0:
            leverage_history[i] = combined_notional[i] / equity

    return {
        "timestamps": all_timestamps,
        "pnl": combined_pnl,
        "notional": combined_notional,
        "equity": equity_curve,
        "leverage": leverage_history,
    }


def calculate_metrics(equity_curve: np.ndarray, initial_capital: float = 100000.0):
    """성과 지표 계산"""
    # Total PnL
    total_pnl = (equity_curve[-1] - initial_capital) / initial_capital

    # Max Drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_dd = np.abs(np.min(drawdown))

    # Sharpe (일별 기준, 6 bars/day 가정)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    daily_returns = []
    for i in range(0, len(returns), 6):
        chunk = returns[i:i+6]
        if len(chunk) > 0:
            daily_returns.append(np.sum(chunk))

    if len(daily_returns) > 1:
        mean_daily = np.mean(daily_returns)
        std_daily = np.std(daily_returns)
        sharpe = (mean_daily / (std_daily + 1e-10)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # CAGR
    n_years = len(equity_curve) / (6 * 252)  # 4시간봉 기준
    if n_years > 0:
        cagr = (equity_curve[-1] / initial_capital) ** (1 / n_years) - 1
    else:
        cagr = 0.0

    return {
        "total_pnl": total_pnl,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "cagr": cagr,
    }


def main():
    print("=" * 100)
    print("4-Asset Combined Portfolio Backtest (레버리지 누적)")
    print("=" * 100)

    initial_capital = 100000.0
    intensity_threshold = 4.0

    # Step 1: 각 자산별 백테스트
    print("\n[1] 개별 자산 백테스트...")
    asset_data = {}
    individual_metrics = {}

    for symbol in PORTFOLIO_SYMBOLS:
        print(f"  {symbol}...", end=" ")
        result = run_single_asset_backtest(symbol, initial_capital, intensity_threshold)
        if result is not None:
            asset_data[symbol] = result

            # 개별 지표
            final_equity = result["equity"].iloc[-1] if result["equity"].iloc[-1] > 0 else result["equity"].max()
            pnl = (final_equity - initial_capital) / initial_capital

            # MDD
            eq = result["equity"].values
            eq = np.where(eq > 0, eq, initial_capital)  # 0 제거
            running_max = np.maximum.accumulate(eq)
            dd = (eq - running_max) / running_max
            mdd = np.abs(np.min(dd))

            individual_metrics[symbol] = {"pnl": pnl, "mdd": mdd, "final_equity": final_equity}
            print(f"PnL: {pnl*100:.1f}%, MDD: {mdd*100:.1f}%")
        else:
            print("FAILED")

    print("\n[2] 개별 자산 성과:")
    print(f"{'Symbol':<10} {'PnL':<12} {'MDD':<12} {'Final Equity':<15}")
    print("-" * 50)
    for symbol, m in individual_metrics.items():
        print(f"{symbol:<10} {m['pnl']*100:<11.1f}% {m['mdd']*100:<11.1f}% ${m['final_equity']:,.0f}")

    # Step 2: 포트폴리오 합산
    print("\n[3] 포트폴리오 합산 (레버리지 누적)...")
    combined = combine_portfolios(asset_data, initial_capital)

    # Step 3: 성과 지표
    metrics = calculate_metrics(combined["equity"], initial_capital)

    print("\n[4] Combined Portfolio 성과:")
    print("=" * 60)
    print(f"  Total PnL:    {metrics['total_pnl']*100:.1f}%")
    print(f"  CAGR:         {metrics['cagr']*100:.1f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']*100:.1f}%")
    print(f"  Final Equity: ${combined['equity'][-1]:,.0f}")

    # Step 4: 레버리지 통계
    lev = combined["leverage"]
    lev_nonzero = lev[lev > 0]

    print("\n[5] 레버리지 통계:")
    print("=" * 60)
    if len(lev_nonzero) > 0:
        print(f"  평균 레버리지: {np.mean(lev_nonzero):.2f}x")
        print(f"  최대 레버리지: {np.max(lev_nonzero):.2f}x")
        print(f"  레버리지 > 3x 비율: {np.mean(lev_nonzero > 3) * 100:.1f}%")
        print(f"  레버리지 > 5x 비율: {np.mean(lev_nonzero > 5) * 100:.1f}%")
        print(f"  레버리지 > 10x 비율: {np.mean(lev_nonzero > 10) * 100:.1f}%")

    # Buy & Hold 비교
    print("\n[6] Buy & Hold 비교:")
    print("=" * 60)
    print(f"{'Asset':<12} {'Strategy':<15} {'Buy&Hold':<15} {'Outperform':<12}")
    print("-" * 55)

    for symbol in PORTFOLIO_SYMBOLS:
        if symbol in asset_data:
            df = asset_data[symbol]
            first_close = df["close"].iloc[0]
            last_close = df["close"].iloc[-1]
            bh_return = (last_close - first_close) / first_close
            strat_return = individual_metrics[symbol]["pnl"]
            outperform = "✓" if strat_return > bh_return else ""
            print(f"{symbol:<12} {strat_return*100:<14.1f}% {bh_return*100:<14.1f}% {outperform}")


if __name__ == "__main__":
    main()
