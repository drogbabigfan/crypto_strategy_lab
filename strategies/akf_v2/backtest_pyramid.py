"""
피라미딩 지원 백테스터.

기존 Go 백테스터와 다른 점:
- 바 단위로 position_size 변경 반영
- 피라미딩 시 누적 사이즈로 수익 계산
- Next-bar entry 방식 유지
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class PyramidBacktestConfig:
    """백테스트 설정."""
    initial_capital: float = 100000.0
    compounding: bool = True       # True: 복리, False: 단리
    fee_rate: float = 0.001        # 0.1% 수수료
    slippage_rate: float = 0.0001  # 0.01% 슬리피지
    leverage_mode: str = "notional"  # "notional" or "margin"
    # notional: size는 자본 대비 포지션 비율 (1.0 = 100% 자본 투입)
    # margin: size는 레버리지 배수 (3.0 = 3x 레버리지, 실제 마진 = 자본/3)


@dataclass
class Trade:
    """완료된 거래."""
    entry_bar: int
    exit_bar: int
    direction: int  # 1: Long, -1: Short
    entry_price: float
    exit_price: float
    avg_size: float      # 평균 포지션 사이즈
    max_size: float      # 최대 포지션 사이즈
    pnl: float           # 순수익률
    gross_pnl: float     # 총수익률
    holding_bars: int


def run_pyramid_backtest(
    df: pd.DataFrame,
    config: Optional[PyramidBacktestConfig] = None
) -> dict:
    """
    피라미딩 지원 백테스트 실행.

    Args:
        df: DataFrame with columns:
            - open, high, low, close
            - signal: -1, 0, 1
            - position_size: 바 단위 포지션 사이즈 (비율)
            - sl_price (optional): 손절가
        config: 백테스트 설정

    Returns:
        dict with metrics
    """
    if config is None:
        config = PyramidBacktestConfig()

    signals = df['signal'].values
    sizes = df['position_size'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    # 손절가 (있으면 사용)
    sl_prices = df['sl_price'].values if 'sl_price' in df.columns else np.zeros(len(df))

    n = len(signals)

    # 상태 변수
    position = 0          # 현재 포지션 방향
    entry_bar = 0         # 진입 바
    entry_price = 0.0     # 진입가

    # 바별 수익 추적
    bar_pnls = np.zeros(n)  # 각 바의 수익률 (size 반영)

    trades: List[Trade] = []

    # 거래별 추적
    trade_sizes = []      # 거래 중 각 바의 사이즈
    trade_returns = []    # 거래 중 각 바의 수익률

    for i in range(1, n):
        # signal[i-1]은 bar[i-1].close에서 결정된 시그널
        # 이를 bar[i].open에서 실행 (Next Bar Entry)
        prev_signal = signals[i - 1]
        prev_size = sizes[i - 1] if sizes[i - 1] > 0 else 1.0

        # === 포지션 없음 → 진입 체크 ===
        if position == 0:
            if prev_signal != 0:
                # Entry at bar[i].open (signal[i-1]에서 결정됨)
                position = int(prev_signal)
                entry_bar = i
                entry_price = opens[i] * (1 + config.slippage_rate * position)  # 슬리피지
                trade_sizes = [prev_size]
                trade_returns = []

        # === 포지션 있음 ===
        elif position != 0:
            # === 청산 체크 (바 시작 시점) ===
            # signal[i-1]이 0이거나 반대 방향이면 bar[i].open에서 청산
            should_exit = (prev_signal == 0) or (prev_signal == -position)

            if should_exit:
                # bar[i].open에서 청산 (signal[i-1]에서 결정됨)
                exit_price = opens[i] * (1 - config.slippage_rate * position)

                # 마지막 바 수익 (prev_close → open[i])
                if i > entry_bar:
                    final_return = (opens[i] - closes[i-1]) / closes[i-1] * position
                    trade_returns.append(final_return)
                    trade_sizes.append(prev_size)

                # 거래 통계
                avg_size = np.mean(trade_sizes) if trade_sizes else prev_size
                max_size = np.max(trade_sizes) if trade_sizes else prev_size

                # 총 수익률 계산
                if trade_returns:
                    weighted_returns = np.array(trade_returns) * np.array(trade_sizes[:len(trade_returns)])
                    gross_pnl = np.sum(weighted_returns)
                else:
                    gross_pnl = 0.0

                total_fee = config.fee_rate * 2 * avg_size
                net_pnl = gross_pnl - total_fee

                trades.append(Trade(
                    entry_bar=entry_bar,
                    exit_bar=i,
                    direction=position,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    avg_size=avg_size,
                    max_size=max_size,
                    pnl=net_pnl,
                    gross_pnl=gross_pnl,
                    holding_bars=i - entry_bar
                ))

                # 리셋
                position = 0
                entry_bar = 0
                entry_price = 0.0
                trade_sizes = []
                trade_returns = []
                continue

            # 손절 체크 (Dollar bar의 low/high 사용)
            # sl_prices[i-1]은 bar[i-1].close에서 결정된 손절가
            sl_hit = False
            sl_exit_price = 0.0
            current_sl = sl_prices[i - 1] if sl_prices[i - 1] > 0 else 0.0

            if current_sl > 0:
                if position == 1 and lows[i] <= current_sl:
                    # Long: low가 SL에 도달
                    sl_hit = True
                    sl_exit_price = current_sl
                elif position == -1 and highs[i] >= current_sl:
                    # Short: high가 SL에 도달
                    sl_hit = True
                    sl_exit_price = current_sl

            if sl_hit:
                # 손절 청산
                exit_price = sl_exit_price * (1 - config.slippage_rate * position)

                # 손절 바 수익률 계산 (prev_close → SL price)
                if i == entry_bar:
                    bar_return = (sl_exit_price - entry_price) / entry_price * position
                else:
                    bar_return = (sl_exit_price - closes[i-1]) / closes[i-1] * position

                weighted_return = bar_return * prev_size
                bar_pnls[i] = weighted_return

                trade_sizes.append(prev_size)
                trade_returns.append(bar_return)

                # 거래 기록
                avg_size = np.mean(trade_sizes)
                max_size = np.max(trade_sizes)
                weighted_returns = np.array(trade_returns) * np.array(trade_sizes[:len(trade_returns)])
                gross_pnl = np.sum(weighted_returns)
                total_fee = config.fee_rate * 2 * avg_size
                net_pnl = gross_pnl - total_fee

                trades.append(Trade(
                    entry_bar=entry_bar,
                    exit_bar=i,
                    direction=position,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    avg_size=avg_size,
                    max_size=max_size,
                    pnl=net_pnl,
                    gross_pnl=gross_pnl,
                    holding_bars=i - entry_bar
                ))

                # 리셋
                position = 0
                entry_bar = 0
                entry_price = 0.0
                trade_sizes = []
                trade_returns = []
                continue

            # 현재 바 수익률 계산 (prev_close to close)
            if i == entry_bar:
                # 진입 바: entry_price → close
                bar_return = (closes[i] - entry_price) / entry_price * position
            else:
                # 유지 바: prev_close → close
                bar_return = (closes[i] - closes[i-1]) / closes[i-1] * position

            # 사이즈 가중 수익률
            weighted_return = bar_return * prev_size
            bar_pnls[i] = weighted_return

            trade_sizes.append(prev_size)
            trade_returns.append(bar_return)

    # === 마지막 포지션 강제 청산 ===
    if position != 0 and len(trade_sizes) > 0:
        # 마지막 바의 close에서 청산
        exit_price = closes[-1] * (1 - config.slippage_rate * position)

        avg_size = np.mean(trade_sizes)
        max_size = np.max(trade_sizes)

        if trade_returns:
            weighted_returns = np.array(trade_returns) * np.array(trade_sizes[:len(trade_returns)])
            gross_pnl = np.sum(weighted_returns)
        else:
            gross_pnl = 0.0

        total_fee = config.fee_rate * 2 * avg_size
        net_pnl = gross_pnl - total_fee

        trades.append(Trade(
            entry_bar=entry_bar,
            exit_bar=n - 1,
            direction=position,
            entry_price=entry_price,
            exit_price=exit_price,
            avg_size=avg_size,
            max_size=max_size,
            pnl=net_pnl,
            gross_pnl=gross_pnl,
            holding_bars=n - 1 - entry_bar
        ))

    # === Equity Curve 계산 ===
    # 거래 단위 복리 (바 단위가 아님)
    if config.compounding:
        equity_curve = np.ones(n) * config.initial_capital
        current_equity = config.initial_capital

        for trade in trades:
            # 거래 완료 시점에 복리 적용
            # PnL은 이미 수익률로 계산됨
            trade_return = trade.pnl  # 수수료 포함 순수익률
            current_equity = current_equity * (1 + trade_return)

            # 해당 거래 기간 동안 equity 업데이트
            if trade.exit_bar < n:
                equity_curve[trade.exit_bar:] = current_equity
    else:
        position_size = config.initial_capital
        equity_curve = config.initial_capital + np.cumsum(bar_pnls * position_size)

    # === Metrics 계산 ===
    if len(trades) == 0:
        return {
            "total_trades": 0,
            "total_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_size": 1.0,
            "max_size": 1.0,
        }

    # Total PnL
    total_pnl = (equity_curve[-1] - config.initial_capital) / config.initial_capital

    # Win rate
    winning_trades = [t for t in trades if t.pnl > 0]
    win_rate = len(winning_trades) / len(trades)

    # Sharpe ratio (일별 수익률 기준, 연환산)
    # 4시간봉 기준: 6 bars/day, 252 trading days
    bars_per_year = 6 * 252
    daily_returns = []
    for i in range(0, len(bar_pnls), 6):
        chunk = bar_pnls[i:i+6]
        if len(chunk) > 0:
            daily_returns.append(np.sum(chunk))

    if len(daily_returns) > 1:
        mean_daily = np.mean(daily_returns)
        std_daily = np.std(daily_returns)
        sharpe = (mean_daily / (std_daily + 1e-10)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max Drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_dd = np.abs(np.min(drawdown))

    # Average/Max size
    avg_size = np.mean([t.avg_size for t in trades])
    max_size = np.max([t.max_size for t in trades])

    return {
        "total_trades": len(trades),
        "total_pnl": total_pnl,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "avg_pnl": np.mean([t.pnl for t in trades]),
        "avg_hold_bars": np.mean([t.holding_bars for t in trades]),
        "avg_size": avg_size,
        "max_size": max_size,
        "equity_curve": equity_curve,
        "trades": trades,
    }


def compare_with_go_backtest(df: pd.DataFrame):
    """Go 백테스터와 결과 비교 (디버깅용)."""
    # 피라미딩 백테스터 (복리)
    result_pyramid = run_pyramid_backtest(df, PyramidBacktestConfig(compounding=True))

    # 피라미딩 백테스터 (단리)
    result_simple = run_pyramid_backtest(df, PyramidBacktestConfig(compounding=False))

    print("=== Pyramid Backtester Results ===")
    print(f"복리: PnL={result_pyramid['total_pnl']*100:.1f}%, Sharpe={result_pyramid['sharpe_ratio']:.2f}, MDD={result_pyramid['max_drawdown']*100:.1f}%")
    print(f"단리: PnL={result_simple['total_pnl']*100:.1f}%, Sharpe={result_simple['sharpe_ratio']:.2f}, MDD={result_simple['max_drawdown']*100:.1f}%")

    return result_pyramid, result_simple
