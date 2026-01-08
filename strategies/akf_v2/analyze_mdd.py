#!/usr/bin/env python3
"""
MDD 분석: 왜 알트가 MDD가 큰가?
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from research.features.kalman import calculate_adaptive_kalman
from strategies.akf_v2.strategy_v32 import load_data, calculate_v32_features, generate_v32_signals, V32_PARAMS

DATA_ROOT = PROJECT_ROOT / "etl/data"
ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "LTCUSDT"]


def analyze_trades(df: pd.DataFrame, symbol: str):
    """개별 트레이드 분석"""
    df_kf = calculate_adaptive_kalman(df)
    features = calculate_v32_features(df_kf)

    signals, stop_prices = generate_v32_signals(
        close=df_kf["close"].values,
        high=df_kf["high"].values,
        low=df_kf["low"].values,
        vel_zscore=features["vel_zscore"],
        unc_pct=features["unc_pct"],
        sigma_hybrid=features["sigma_hybrid"],
        dynamic_mult=features["dynamic_mult"],
        v_ratio=features["v_ratio"],
        entry_z=V32_PARAMS["entry_z"],
        unc_pct_max=V32_PARAMS["unc_pct_max"],
        v_ratio_threshold=V32_PARAMS["v_ratio_threshold"],
        warmup=V32_PARAMS["warmup"],
    )

    df_kf["signal"] = signals
    df_kf["stop_price"] = stop_prices

    # 트레이드 추출
    trades = []
    position = 0
    entry_price = 0
    entry_idx = 0

    for i in range(len(df_kf)):
        sig = signals[i]

        if position == 0 and sig != 0:
            # Entry
            position = sig
            entry_price = df_kf["close"].iloc[i]
            entry_idx = i

        elif position != 0 and sig == 0:
            # Exit
            exit_price = df_kf["close"].iloc[i]
            if position == 1:
                pnl = (exit_price / entry_price - 1)
            else:
                pnl = (entry_price / exit_price - 1)

            trades.append({
                "entry_idx": entry_idx,
                "exit_idx": i,
                "direction": position,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "bars_held": i - entry_idx,
            })
            position = 0

    return trades, df_kf


def analyze_drawdown_periods(df_kf: pd.DataFrame, signals: np.ndarray):
    """드로우다운 발생 구간 분석"""
    close = df_kf["close"].values

    # 간단한 equity curve (compounding)
    equity = [1.0]
    position = 0
    entry_price = 0

    for i in range(1, len(close)):
        if position == 0:
            equity.append(equity[-1])
            if signals[i] != 0:
                position = signals[i]
                entry_price = close[i]
        else:
            if signals[i] == 0:
                # Exit
                if position == 1:
                    ret = close[i] / entry_price - 1
                else:
                    ret = entry_price / close[i] - 1
                equity.append(equity[-1] * (1 + ret))
                position = 0
            elif signals[i] == position:
                equity.append(equity[-1])
            else:
                equity.append(equity[-1])

    equity = np.array(equity)

    # Drawdown 계산
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak

    # Max drawdown 시점 찾기
    mdd_idx = np.argmin(dd)
    mdd_value = dd[mdd_idx]

    # Peak 시점
    peak_idx = np.argmax(equity[:mdd_idx+1]) if mdd_idx > 0 else 0

    return {
        "mdd": abs(mdd_value),
        "mdd_idx": mdd_idx,
        "peak_idx": peak_idx,
        "peak_equity": equity[peak_idx] if peak_idx < len(equity) else 1.0,
        "trough_equity": equity[mdd_idx],
        "equity": equity,
        "dd": dd,
    }


def main():
    print("=" * 100)
    print("MDD 분석: 알트 vs BTC")
    print("=" * 100)

    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df

    # 1. 기본 변동성 비교
    print("\n[1] 기본 변동성 비교 (연간화 volatility)")
    print("-" * 60)
    print(f"{'Symbol':<12} {'Daily Vol':<12} {'Avg True Range':<15}")
    print("-" * 60)

    for symbol, df in data.items():
        log_ret = np.log(df["close"].values[1:] / df["close"].values[:-1])
        daily_vol = np.std(log_ret) * np.sqrt(252 * 4)  # 6시간봉 → 일간 → 연간

        atr = (df["high"] - df["low"]).mean() / df["close"].mean() * 100

        print(f"{symbol:<12} {daily_vol*100:<11.1f}% {atr:<14.2f}%")

    # 2. 트레이드별 손실 분석
    print("\n[2] 트레이드 손실 분석")
    print("-" * 80)
    print(f"{'Symbol':<12} {'Trades':<8} {'Winners':<10} {'Losers':<10} {'AvgWin':<12} {'AvgLoss':<12} {'MaxLoss':<12}")
    print("-" * 80)

    all_trades = {}
    for symbol, df in data.items():
        trades, df_kf = analyze_trades(df, symbol)
        all_trades[symbol] = trades

        winners = [t for t in trades if t["pnl"] > 0]
        losers = [t for t in trades if t["pnl"] <= 0]

        avg_win = np.mean([t["pnl"] for t in winners]) if winners else 0
        avg_loss = np.mean([t["pnl"] for t in losers]) if losers else 0
        max_loss = min([t["pnl"] for t in losers]) if losers else 0

        print(f"{symbol:<12} {len(trades):<8} {len(winners):<10} {len(losers):<10} "
              f"{avg_win*100:<11.1f}% {avg_loss*100:<11.1f}% {max_loss*100:<11.1f}%")

    # 3. 연속 손실 분석
    print("\n[3] 연속 손실 분석")
    print("-" * 60)
    print(f"{'Symbol':<12} {'MaxConsecLoss':<15} {'ConsecLossPnL':<15}")
    print("-" * 60)

    for symbol, trades in all_trades.items():
        max_consec = 0
        max_consec_pnl = 0

        consec = 0
        consec_pnl = 0

        for t in trades:
            if t["pnl"] <= 0:
                consec += 1
                consec_pnl += t["pnl"]
            else:
                if consec > max_consec:
                    max_consec = consec
                    max_consec_pnl = consec_pnl
                consec = 0
                consec_pnl = 0

        if consec > max_consec:
            max_consec = consec
            max_consec_pnl = consec_pnl

        print(f"{symbol:<12} {max_consec:<15} {max_consec_pnl*100:<14.1f}%")

    # 4. MDD 발생 시점 분석
    print("\n[4] MDD 발생 시점")
    print("-" * 80)

    for symbol, df in data.items():
        df_kf = calculate_adaptive_kalman(df)
        features = calculate_v32_features(df_kf)
        signals, _ = generate_v32_signals(
            close=df_kf["close"].values,
            high=df_kf["high"].values,
            low=df_kf["low"].values,
            vel_zscore=features["vel_zscore"],
            unc_pct=features["unc_pct"],
            sigma_hybrid=features["sigma_hybrid"],
            dynamic_mult=features["dynamic_mult"],
            v_ratio=features["v_ratio"],
            entry_z=V32_PARAMS["entry_z"],
            unc_pct_max=V32_PARAMS["unc_pct_max"],
            v_ratio_threshold=V32_PARAMS["v_ratio_threshold"],
            warmup=V32_PARAMS["warmup"],
        )

        dd_info = analyze_drawdown_periods(df_kf, signals)

        peak_date = df_kf["open_time"].iloc[dd_info["peak_idx"]] if "open_time" in df_kf.columns else dd_info["peak_idx"]
        trough_date = df_kf["open_time"].iloc[dd_info["mdd_idx"]] if "open_time" in df_kf.columns else dd_info["mdd_idx"]

        print(f"{symbol}: MDD={dd_info['mdd']*100:.1f}%")
        print(f"  Peak: idx={dd_info['peak_idx']}, equity={dd_info['peak_equity']:.2f}")
        print(f"  Trough: idx={dd_info['mdd_idx']}, equity={dd_info['trough_equity']:.2f}")
        if "open_time" in df_kf.columns:
            print(f"  Period: {peak_date} → {trough_date}")
        print()

    # 5. 알트 특성: 급락장 분석
    print("\n[5] 급락장 (일일 -10% 이상) 횟수")
    print("-" * 60)

    for symbol, df in data.items():
        log_ret = np.log(df["close"].values[1:] / df["close"].values[:-1])
        # 6시간봉 4개 = 1일
        daily_ret = []
        for i in range(0, len(log_ret)-3, 4):
            daily_ret.append(sum(log_ret[i:i+4]))

        daily_ret = np.array(daily_ret)
        crash_days = np.sum(daily_ret < -0.10)
        worst_day = np.min(daily_ret)

        print(f"{symbol:<12} Crash days: {crash_days:<5} Worst: {worst_day*100:.1f}%")


if __name__ == "__main__":
    main()
