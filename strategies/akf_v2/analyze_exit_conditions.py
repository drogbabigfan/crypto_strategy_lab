#!/usr/bin/env python3
"""
이탈 조건별 영향력 분석

1. Exit Count: 각 조건으로 이탈한 횟수
2. Ablation Study: 각 조건 제거 시 성능 변화
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
from strategies.akf_v2.strategy_v33_final import load_data, calculate_features, calculate_position_size, V33_PARAMS

DATA_ROOT = PROJECT_ROOT / "etl/data"
ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "LTCUSDT"]


@njit
def generate_signals_with_exit_tracking(
    close, high, low, kf_trend, vel_zscore, unc_pct, sigma_hybrid, dynamic_mult, v_ratio, resid_std,
    entry_z, unc_pct_max, v_ratio_threshold, innov_base_mult, innov_mult_min, innov_mult_max,
    hard_stop_mult, risk_target, conf_lambda, size_min, size_max, warmup,
    # Ablation flags
    use_hard_stop=True, use_trail_stop=True, use_innovation=True, use_signal_exit=True
):
    """Signal generation with exit condition tracking"""
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    position_sizes = np.zeros(n, dtype=np.float64)

    position = 0
    entry_price = 0.0
    current_size = 0.0
    highest = 0.0
    lowest = np.inf
    prev_stop = 0.0
    hard_stop = 0.0

    # Exit counters
    hard_exits = 0
    trail_exits = 0
    innov_exits = 0
    signal_exits = 0

    for i in range(warmup, n):
        innov_mult = innov_base_mult / (v_ratio[i] + 1e-10)
        innov_mult = max(innov_mult_min, min(innov_mult_max, innov_mult))
        log_resid = np.log(close[i]) - np.log(kf_trend[i])

        if position == 0:
            if vel_zscore[i] > entry_z and unc_pct[i] < unc_pct_max:
                position = 1
                entry_price = close[i]
                highest = high[i]
                current_size = calculate_position_size(
                    sigma_hybrid[i], unc_pct[i], risk_target, hard_stop_mult, conf_lambda, size_min, size_max
                )
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = highest * np.exp(-dist)
                hard_stop = entry_price * np.exp(-hard_stop_mult * sigma_hybrid[i])
                signals[i] = 1
                position_sizes[i] = current_size

            elif vel_zscore[i] < -entry_z and unc_pct[i] < unc_pct_max:
                position = -1
                entry_price = close[i]
                lowest = low[i]
                current_size = calculate_position_size(
                    sigma_hybrid[i], unc_pct[i], risk_target, hard_stop_mult, conf_lambda, size_min, size_max
                )
                dist = dynamic_mult[i] * sigma_hybrid[i]
                prev_stop = lowest * np.exp(dist)
                hard_stop = entry_price * np.exp(hard_stop_mult * sigma_hybrid[i])
                signals[i] = -1
                position_sizes[i] = current_size

        elif position == 1:
            if high[i] > highest:
                highest = high[i]
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_stop = highest * np.exp(-dist)
            trail_stop = max(new_stop, prev_stop)
            prev_stop = trail_stop

            # 1. Hard Stop
            if use_hard_stop and close[i] < hard_stop:
                position = 0
                hard_exits += 1
                continue
            # 2. Trailing Stop
            if use_trail_stop and close[i] < trail_stop:
                position = 0
                trail_exits += 1
                continue
            # 3. Innovation Breaker
            if use_innovation and v_ratio[i] >= v_ratio_threshold:
                if log_resid < -innov_mult * resid_std[i]:
                    position = 0
                    innov_exits += 1
                    continue
            # 4. Signal Exit
            if use_signal_exit and v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] < -entry_z:
                    position = 0
                    signal_exits += 1
                    continue

            signals[i] = 1
            position_sizes[i] = current_size

        elif position == -1:
            if low[i] < lowest:
                lowest = low[i]
            dist = dynamic_mult[i] * sigma_hybrid[i]
            new_stop = lowest * np.exp(dist)
            trail_stop = min(new_stop, prev_stop)
            prev_stop = trail_stop

            # 1. Hard Stop
            if use_hard_stop and close[i] > hard_stop:
                position = 0
                hard_exits += 1
                continue
            # 2. Trailing Stop
            if use_trail_stop and close[i] > trail_stop:
                position = 0
                trail_exits += 1
                continue
            # 3. Innovation Breaker
            if use_innovation and v_ratio[i] >= v_ratio_threshold:
                if log_resid > innov_mult * resid_std[i]:
                    position = 0
                    innov_exits += 1
                    continue
            # 4. Signal Exit
            if use_signal_exit and v_ratio[i] < v_ratio_threshold:
                if vel_zscore[i] > entry_z:
                    position = 0
                    signal_exits += 1
                    continue

            signals[i] = -1
            position_sizes[i] = current_size

    return signals, position_sizes, hard_exits, trail_exits, innov_exits, signal_exits


def run_backtest_with_ablation(df: pd.DataFrame, use_hard=True, use_trail=True, use_innov=True, use_signal=True):
    params = V33_PARAMS
    df_kf = calculate_adaptive_kalman(df)
    features = calculate_features(df_kf, params)

    signals, position_sizes, hard_exits, trail_exits, innov_exits, signal_exits = generate_signals_with_exit_tracking(
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
        entry_z=params["entry_z"],
        unc_pct_max=params["unc_pct_max"],
        v_ratio_threshold=params["v_ratio_threshold"],
        innov_base_mult=params["innov_base_mult"],
        innov_mult_min=params["innov_mult_min"],
        innov_mult_max=params["innov_mult_max"],
        hard_stop_mult=params["hard_stop_mult"],
        risk_target=params["risk_target"],
        conf_lambda=params["conf_lambda"],
        size_min=params["size_min"],
        size_max=params["size_max"],
        warmup=params["warmup"],
        use_hard_stop=use_hard,
        use_trail_stop=use_trail,
        use_innovation=use_innov,
        use_signal_exit=use_signal,
    )

    df_kf["signal"] = signals
    df_kf["position_size"] = position_sizes
    df_kf["sl_price"] = 0.0

    bt_config = PyramidBacktestConfig(
        initial_capital=100000.0,
        compounding=True,
        fee_rate=0.001,
        slippage_rate=0.0001,
    )

    metrics = run_pyramid_backtest(df_kf, bt_config)
    metrics["hard_exits"] = hard_exits
    metrics["trail_exits"] = trail_exits
    metrics["innov_exits"] = innov_exits
    metrics["signal_exits"] = signal_exits
    metrics["total_exits"] = hard_exits + trail_exits + innov_exits + signal_exits

    return metrics


def main():
    print("Loading data...")
    data = {}
    for symbol in ALL_SYMBOLS:
        df = load_data(symbol)
        if df is not None:
            data[symbol] = df
            print(f"  {symbol}: {len(df):,} bars")

    # ========================================
    # Part 1: Exit Count Analysis
    # ========================================
    print("\n" + "=" * 140)
    print("Part 1: EXIT COUNT ANALYSIS (각 조건으로 이탈한 횟수)")
    print("=" * 140)
    print(f"{'Symbol':<10} {'Hard':<8} {'Trail':<8} {'Innov':<8} {'Signal':<8} {'Total':<8} | {'Hard%':<8} {'Trail%':<8} {'Innov%':<8} {'Signal%':<8}")
    print("-" * 120)

    total_exits = {"hard": 0, "trail": 0, "innov": 0, "signal": 0}
    exit_data = []

    for symbol, df in data.items():
        m = run_backtest_with_ablation(df)
        total = m["total_exits"]
        h_pct = m["hard_exits"] / total * 100 if total > 0 else 0
        t_pct = m["trail_exits"] / total * 100 if total > 0 else 0
        i_pct = m["innov_exits"] / total * 100 if total > 0 else 0
        s_pct = m["signal_exits"] / total * 100 if total > 0 else 0

        print(f"{symbol:<10} {m['hard_exits']:<8} {m['trail_exits']:<8} {m['innov_exits']:<8} {m['signal_exits']:<8} {total:<8} | "
              f"{h_pct:<7.1f}% {t_pct:<7.1f}% {i_pct:<7.1f}% {s_pct:<7.1f}%")

        total_exits["hard"] += m["hard_exits"]
        total_exits["trail"] += m["trail_exits"]
        total_exits["innov"] += m["innov_exits"]
        total_exits["signal"] += m["signal_exits"]

        exit_data.append({
            "symbol": symbol,
            "hard": m["hard_exits"],
            "trail": m["trail_exits"],
            "innov": m["innov_exits"],
            "signal": m["signal_exits"],
        })

    print("-" * 120)
    total_all = sum(total_exits.values())
    print(f"{'TOTAL':<10} {total_exits['hard']:<8} {total_exits['trail']:<8} {total_exits['innov']:<8} {total_exits['signal']:<8} {total_all:<8} | "
          f"{total_exits['hard']/total_all*100:<7.1f}% {total_exits['trail']/total_all*100:<7.1f}% "
          f"{total_exits['innov']/total_all*100:<7.1f}% {total_exits['signal']/total_all*100:<7.1f}%")

    # ========================================
    # Part 2: Ablation Study
    # ========================================
    print("\n" + "=" * 140)
    print("Part 2: ABLATION STUDY (각 조건 제거 시 성능 변화)")
    print("=" * 140)

    ablation_configs = [
        (True, True, True, True, "Full (Baseline)"),
        (False, True, True, True, "No Hard Stop"),
        (True, False, True, True, "No Trailing"),
        (True, True, False, True, "No Innovation"),
        (True, True, True, False, "No Signal Exit"),
        (True, False, False, False, "Hard Only"),
        (False, True, False, False, "Trail Only"),
    ]

    print(f"\n{'Config':<20} {'Sharpe':<10} {'PnL':<12} {'MDD':<10} {'Trades':<8} {'Positive':<10}")
    print("-" * 80)

    ablation_results = {}

    for use_hard, use_trail, use_innov, use_signal, desc in ablation_configs:
        results = []
        for symbol, df in data.items():
            m = run_backtest_with_ablation(df, use_hard, use_trail, use_innov, use_signal)
            results.append({
                "symbol": symbol,
                "sharpe": m["sharpe_ratio"],
                "pnl": m["total_pnl"],
                "mdd": m["max_drawdown"],
                "trades": m["total_trades"],
            })

        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_pnl = np.mean([r["pnl"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        avg_trades = np.mean([r["trades"] for r in results])
        positive = sum(1 for r in results if r["pnl"] > 0)

        ablation_results[desc] = {
            "sharpe": avg_sharpe,
            "pnl": avg_pnl,
            "mdd": avg_mdd,
            "trades": avg_trades,
            "positive": positive,
            "results": results,
        }

        print(f"{desc:<20} {avg_sharpe:<10.2f} {avg_pnl*100:<11.1f}% {avg_mdd*100:<9.1f}% {avg_trades:<8.0f} {positive}/7")

    # ========================================
    # Part 3: Impact Analysis
    # ========================================
    print("\n" + "=" * 140)
    print("Part 3: IMPACT ANALYSIS (각 조건의 기여도)")
    print("=" * 140)

    baseline = ablation_results["Full (Baseline)"]

    print(f"\n{'Condition':<20} {'Sharpe Δ':<12} {'PnL Δ':<12} {'MDD Δ':<12} {'Assessment':<20}")
    print("-" * 80)

    impact_map = {
        "Hard Stop": "No Hard Stop",
        "Trailing Stop": "No Trailing",
        "Innovation": "No Innovation",
        "Signal Exit": "No Signal Exit",
    }

    for cond, no_cond_key in impact_map.items():
        no_cond = ablation_results[no_cond_key]
        sharpe_delta = baseline["sharpe"] - no_cond["sharpe"]
        pnl_delta = baseline["pnl"] - no_cond["pnl"]
        mdd_delta = baseline["mdd"] - no_cond["mdd"]  # negative is better

        # Assessment
        if sharpe_delta > 0.05:
            assessment = "IMPORTANT (Sharpe+)"
        elif sharpe_delta < -0.05:
            assessment = "HARMFUL (Sharpe-)"
        elif mdd_delta < -0.05:
            assessment = "HELPFUL (MDD-)"
        elif abs(pnl_delta) < 0.1 and abs(mdd_delta) < 0.02:
            assessment = "NEGLIGIBLE"
        else:
            assessment = "MIXED"

        print(f"{cond:<20} {sharpe_delta:+.2f}        {pnl_delta*100:+.1f}%       {mdd_delta*100:+.1f}%       {assessment}")

    # ========================================
    # Part 4: Per-Symbol Ablation
    # ========================================
    print("\n" + "=" * 140)
    print("Part 4: PER-SYMBOL IMPACT (BTC vs ALT)")
    print("=" * 140)

    print(f"\n[BTC Impact]")
    print(f"{'Config':<20} {'Sharpe':<10} {'PnL':<12} {'MDD':<10}")
    print("-" * 50)

    for desc, data_dict in ablation_results.items():
        btc = next(r for r in data_dict["results"] if r["symbol"] == "BTCUSDT")
        print(f"{desc:<20} {btc['sharpe']:<10.2f} {btc['pnl']*100:<11.1f}% {btc['mdd']*100:<9.1f}%")

    print(f"\n[ALT Average (excluding BTC)]")
    print(f"{'Config':<20} {'Sharpe':<10} {'PnL':<12} {'MDD':<10}")
    print("-" * 50)

    for desc, data_dict in ablation_results.items():
        alts = [r for r in data_dict["results"] if r["symbol"] != "BTCUSDT"]
        avg_sharpe = np.mean([r["sharpe"] for r in alts])
        avg_pnl = np.mean([r["pnl"] for r in alts])
        avg_mdd = np.mean([r["mdd"] for r in alts])
        print(f"{desc:<20} {avg_sharpe:<10.2f} {avg_pnl*100:<11.1f}% {avg_mdd*100:<9.1f}%")


if __name__ == "__main__":
    main()
