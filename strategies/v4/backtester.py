"""
V4 Pyramiding Backtester

이전 버전 문제점 해결:
1. Mark-to-Market: 매 봉 미실현 손익 반영
2. Layer별 수수료: 진입마다 개별 수수료 차감
3. Position 객체: 다중 레이어 관리 (평단가, 개별 진입가)
4. State Machine: 명확한 상태 전이 로직
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class PositionState(Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass
class Layer:
    """개별 진입 레이어."""
    entry_bar: int
    entry_price: float
    size: float
    entry_fee: float


@dataclass
class Position:
    """
    다중 레이어 포지션 관리.

    피라미딩 시 각 레이어를 개별 추적하여:
    - 정확한 평단가 계산
    - 레이어별 수수료 추적
    - 부분 청산 지원 (향후)
    """
    direction: int = 0  # 1: Long, -1: Short, 0: Flat
    layers: List[Layer] = field(default_factory=list)
    total_fees: float = 0.0

    @property
    def total_size(self) -> float:
        return sum(layer.size for layer in self.layers)

    @property
    def avg_entry_price(self) -> float:
        if not self.layers:
            return 0.0
        total_value = sum(layer.entry_price * layer.size for layer in self.layers)
        total_size = self.total_size
        return total_value / total_size if total_size > 0 else 0.0

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def add_layer(self, bar: int, price: float, size: float, fee_rate: float, slippage_rate: float) -> float:
        """
        레이어 추가 (피라미딩).

        Returns:
            실제 진입가 (슬리피지 반영)
        """
        actual_price = price * (1 + slippage_rate * self.direction)
        entry_fee = fee_rate * size

        self.layers.append(Layer(
            entry_bar=bar,
            entry_price=actual_price,
            size=size,
            entry_fee=entry_fee
        ))
        self.total_fees += entry_fee

        return actual_price

    def close_all(self, price: float, fee_rate: float, slippage_rate: float) -> tuple:
        """
        전량 청산.

        Returns:
            (청산가, 총 수수료, Gross PnL, Net PnL)
        """
        if not self.layers:
            return 0.0, 0.0, 0.0, 0.0

        actual_price = price * (1 - slippage_rate * self.direction)
        exit_fee = fee_rate * self.total_size
        self.total_fees += exit_fee

        # Gross PnL 계산 (레이어별)
        gross_pnl = 0.0
        for layer in self.layers:
            layer_return = (actual_price - layer.entry_price) / layer.entry_price * self.direction
            gross_pnl += layer_return * layer.size

        net_pnl = gross_pnl - self.total_fees

        return actual_price, self.total_fees, gross_pnl, net_pnl

    def unrealized_pnl(self, current_price: float) -> float:
        """
        미실현 손익 계산 (MTM).

        Returns:
            현재가 기준 미실현 수익률 (size 가중)
        """
        if not self.layers:
            return 0.0

        total_pnl = 0.0
        for layer in self.layers:
            layer_return = (current_price - layer.entry_price) / layer.entry_price * self.direction
            total_pnl += layer_return * layer.size

        return total_pnl

    def reset(self):
        """포지션 초기화."""
        self.direction = 0
        self.layers = []
        self.total_fees = 0.0


@dataclass
class TradeRecord:
    """완료된 거래 기록."""
    entry_bar: int
    exit_bar: int
    direction: int
    avg_entry_price: float
    exit_price: float
    num_layers: int
    total_size: float
    max_size: float
    gross_pnl: float
    net_pnl: float
    total_fees: float
    holding_bars: int


@dataclass
class BacktestConfig:
    """백테스트 설정."""
    initial_capital: float = 100000.0
    fee_rate: float = 0.001         # 0.1%
    slippage_rate: float = 0.0001   # 0.01%
    max_layers: int = 3             # 최대 피라미딩 횟수
    max_leverage: float = 3.0       # 최대 총 레버리지


@dataclass
class BacktestResult:
    """백테스트 결과."""
    trades: List[TradeRecord]
    equity_curve: np.ndarray
    drawdown_curve: np.ndarray
    bar_pnls: np.ndarray

    # Metrics
    total_pnl: float = 0.0
    cagr: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_holding_bars: float = 0.0
    total_trades: int = 0
    avg_layers: float = 0.0


def run_backtest(
    df: pd.DataFrame,
    config: Optional[BacktestConfig] = None
) -> BacktestResult:
    """
    State Machine 기반 피라미딩 백테스터.

    매 봉 처리 순서:
    1. MTM 평가 (미실현 손익 → Equity 업데이트)
    2. 청산 체크 (Stop Loss, Exit Signal)
    3. 진입/추가진입 체크

    Args:
        df: DataFrame with columns:
            - open, high, low, close
            - signal: -1, 0, 1 (방향)
            - position_size: 진입 사이즈 (비율)
            - stop_price (optional): 스탑 가격

    Returns:
        BacktestResult
    """
    if config is None:
        config = BacktestConfig()

    # Data extraction
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    signals = df['signal'].values
    sizes = df['position_size'].values

    stop_prices = df['stop_price'].values if 'stop_price' in df.columns else np.zeros(len(df))

    n = len(df)

    # State variables
    position = Position()
    balance = config.initial_capital

    # Tracking arrays
    equity_curve = np.zeros(n)
    equity_curve[0] = balance
    bar_pnls = np.zeros(n)

    trades: List[TradeRecord] = []

    # Track max size during trade
    trade_max_size = 0.0
    trade_entry_bar = 0

    for i in range(1, n):
        prev_signal = signals[i - 1]
        prev_size = sizes[i - 1] if sizes[i - 1] > 0 else 0.0
        prev_stop = stop_prices[i - 1] if stop_prices[i - 1] > 0 else 0.0

        # ============================================================
        # Step 1: Mark-to-Market (미실현 손익 계산)
        # ============================================================
        if position.direction != 0:
            unrealized = position.unrealized_pnl(closes[i])
            current_equity = balance * (1 + unrealized)
        else:
            current_equity = balance
            unrealized = 0.0

        equity_curve[i] = current_equity
        bar_pnls[i] = unrealized - (bar_pnls[i-1] if i > 1 else 0)  # Delta

        # ============================================================
        # Step 2: 청산 체크
        # ============================================================
        should_exit = False
        exit_price = 0.0
        exit_reason = ""

        if position.direction != 0:
            # 2a. Stop Loss 체크 (장중 터치)
            if prev_stop > 0:
                if position.direction == 1 and lows[i] <= prev_stop:
                    should_exit = True
                    exit_price = prev_stop
                    exit_reason = "stop_loss"
                elif position.direction == -1 and highs[i] >= prev_stop:
                    should_exit = True
                    exit_price = prev_stop
                    exit_reason = "stop_loss"

            # 2b. Signal Exit (신호가 0 또는 반대)
            if not should_exit:
                if prev_signal == 0 or prev_signal == -position.direction:
                    should_exit = True
                    exit_price = opens[i]
                    exit_reason = "signal_exit"

            # 2c. 강제청산 체크 (Margin Call) - 자산이 0 이하
            if not should_exit and current_equity <= 0:
                should_exit = True
                exit_price = closes[i]
                exit_reason = "margin_call"

        # 청산 실행
        if should_exit:
            actual_exit, total_fees, gross_pnl, net_pnl = position.close_all(
                exit_price, config.fee_rate, config.slippage_rate
            )

            # Balance 업데이트 (실현 손익)
            balance = balance * (1 + net_pnl)

            # Trade 기록
            trades.append(TradeRecord(
                entry_bar=trade_entry_bar,
                exit_bar=i,
                direction=position.direction,
                avg_entry_price=position.avg_entry_price,
                exit_price=actual_exit,
                num_layers=position.num_layers,
                total_size=position.total_size,
                max_size=trade_max_size,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                total_fees=total_fees,
                holding_bars=i - trade_entry_bar
            ))

            # Reset
            position.reset()
            trade_max_size = 0.0
            trade_entry_bar = 0

            equity_curve[i] = balance
            continue

        # ============================================================
        # Step 3: 진입/추가진입 체크
        # ============================================================
        if prev_signal != 0 and prev_size > 0:
            # 3a. 신규 진입
            if position.direction == 0:
                position.direction = int(prev_signal)
                position.add_layer(
                    bar=i,
                    price=opens[i],
                    size=prev_size,
                    fee_rate=config.fee_rate,
                    slippage_rate=config.slippage_rate
                )
                trade_entry_bar = i
                trade_max_size = prev_size

            # 3b. 피라미딩 (같은 방향, 레이어 제한 내)
            elif position.direction == prev_signal:
                if position.num_layers < config.max_layers:
                    # 레버리지 제한 체크
                    new_total_size = position.total_size + prev_size
                    if new_total_size <= config.max_leverage:
                        position.add_layer(
                            bar=i,
                            price=opens[i],
                            size=prev_size,
                            fee_rate=config.fee_rate,
                            slippage_rate=config.slippage_rate
                        )
                        trade_max_size = max(trade_max_size, new_total_size)

        # Update max size tracking
        if position.direction != 0:
            trade_max_size = max(trade_max_size, position.total_size)

    # ============================================================
    # 마지막 포지션 강제 청산
    # ============================================================
    if position.direction != 0:
        actual_exit, total_fees, gross_pnl, net_pnl = position.close_all(
            closes[-1], config.fee_rate, config.slippage_rate
        )

        balance = balance * (1 + net_pnl)

        trades.append(TradeRecord(
            entry_bar=trade_entry_bar,
            exit_bar=n - 1,
            direction=position.direction,
            avg_entry_price=position.avg_entry_price,
            exit_price=actual_exit,
            num_layers=position.num_layers,
            total_size=position.total_size,
            max_size=trade_max_size,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            total_fees=total_fees,
            holding_bars=n - 1 - trade_entry_bar
        ))

        equity_curve[-1] = balance

    # ============================================================
    # Metrics 계산
    # ============================================================
    # Drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdown_curve = (equity_curve - running_max) / running_max
    max_drawdown = np.abs(np.min(drawdown_curve))

    # Total PnL
    total_pnl = (equity_curve[-1] - config.initial_capital) / config.initial_capital

    # CAGR (4시간봉 기준)
    n_years = n / (365 * 6)  # 6 bars per day
    if n_years > 0 and total_pnl > -1:
        cagr = ((1 + total_pnl) ** (1 / n_years)) - 1
    else:
        cagr = 0.0

    # Sharpe (일별 수익률)
    daily_returns = []
    for j in range(0, n, 6):
        if j + 6 <= n:
            day_return = (equity_curve[j + 5] - equity_curve[j]) / equity_curve[j]
            daily_returns.append(day_return)

    if len(daily_returns) > 1:
        sharpe = (np.mean(daily_returns) / (np.std(daily_returns) + 1e-10)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Trade metrics
    if trades:
        win_rate = len([t for t in trades if t.net_pnl > 0]) / len(trades)
        avg_trade_pnl = np.mean([t.net_pnl for t in trades])
        avg_holding = np.mean([t.holding_bars for t in trades])
        avg_layers = np.mean([t.num_layers for t in trades])
    else:
        win_rate = 0.0
        avg_trade_pnl = 0.0
        avg_holding = 0.0
        avg_layers = 0.0

    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        bar_pnls=bar_pnls,
        total_pnl=total_pnl,
        cagr=cagr,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        win_rate=win_rate,
        avg_trade_pnl=avg_trade_pnl,
        avg_holding_bars=avg_holding,
        total_trades=len(trades),
        avg_layers=avg_layers
    )


def test_backtester():
    """백테스터 검증 테스트."""
    print("=" * 80)
    print("Backtester Verification Tests")
    print("=" * 80)

    # Test 1: Simple Long Trade (단일 진입)
    # signal[0]=1 → bar[1].open=100에서 진입
    # signal[1,2]=1 유지 (피라미딩은 max_layers=1로 방지)
    # signal[3]=0 → bar[4].open=110에서 청산
    # 기대: (110-100)/100 = 10% - 수수료
    print("\n[Test 1] Simple Long Trade (10% gain)")
    df1 = pd.DataFrame({
        'open': [100, 100, 105, 108, 110],
        'high': [100, 105, 110, 112, 115],
        'low': [100, 95, 100, 105, 108],
        'close': [100, 102, 107, 109, 112],
        'signal': [1, 1, 1, 0, 0],  # signal 유지 후 청산
        'position_size': [1.0, 1.0, 1.0, 0, 0],
        'stop_price': [0, 0, 0, 0, 0]
    })

    # max_layers=1로 피라미딩 비활성화
    result1 = run_backtest(df1, BacktestConfig(fee_rate=0.001, slippage_rate=0.0, max_layers=1))
    print(f"  Trades: {result1.total_trades}")
    print(f"  PnL: {result1.total_pnl*100:.2f}%")
    print(f"  Expected: (110-100)/100 = 10% - 0.2% fees = 9.8%")

    if result1.trades:
        t = result1.trades[0]
        print(f"  Entry: {t.avg_entry_price:.2f}, Exit: {t.exit_price:.2f}")
        print(f"  Layers: {t.num_layers}")
        print(f"  Gross PnL: {t.gross_pnl*100:.2f}%, Net PnL: {t.net_pnl*100:.2f}%")
        expected_gross = (110 - 100) / 100 * 1.0  # 10%
        print(f"  [검증] 예상 Gross: {expected_gross*100:.2f}%")

    # Test 2: Stop Loss Hit
    # signal[0]=1 → bar[1].open=100에서 진입
    # bar[2].low=90 < stop=95 → bar[2]에서 손절 청산
    # 기대: (95-100)/100 = -5% - 0.2% fees = -5.2%
    print("\n[Test 2] Stop Loss Hit")
    df2 = pd.DataFrame({
        'open': [100, 100, 95, 90],
        'high': [100, 105, 100, 95],
        'low': [100, 95, 90, 85],  # bar[2].low=90 < stop=95
        'close': [100, 100, 92, 90],
        'signal': [1, 1, 0, 0],  # 손절 후 signal=0
        'position_size': [1.0, 1.0, 0, 0],
        'stop_price': [0, 95, 95, 0]  # Stop at 95
    })

    result2 = run_backtest(df2, BacktestConfig(fee_rate=0.001, slippage_rate=0.0, max_layers=1))
    print(f"  Trades: {result2.total_trades}")
    print(f"  Expected: -5% - 0.2% = -5.2%")
    for idx, t in enumerate(result2.trades):
        print(f"  Trade {idx+1}: Entry={t.avg_entry_price:.2f}, Exit={t.exit_price:.2f} (Stop=95), PnL={t.net_pnl*100:.2f}%")

    # Test 3: Pyramiding
    # signal[0]=1 → bar[1].open=100에서 1차 진입 (0.5x)
    # signal[1]=1 → bar[2].open=105에서 2차 진입 (0.5x) → 총 1.0x
    # signal[4]=0 → bar[5].open=120에서 청산
    print("\n[Test 3] Pyramiding (2 layers)")
    df3 = pd.DataFrame({
        'open': [100, 100, 105, 110, 115, 120],
        'high': [100, 105, 110, 115, 120, 125],
        'low': [100, 95, 100, 105, 110, 115],
        'close': [100, 102, 108, 112, 118, 120],
        'signal': [1, 1, 1, 1, 0, 0],  # 피라미딩 후 청산
        'position_size': [0.5, 0.5, 0.5, 0.5, 0, 0],
        'stop_price': [0, 0, 0, 0, 0, 0]
    })

    result3 = run_backtest(df3, BacktestConfig(fee_rate=0.001, slippage_rate=0.0, max_layers=3))
    print(f"  Trades: {result3.total_trades}")
    if result3.trades:
        t = result3.trades[0]
        print(f"  Layers: {t.num_layers}")
        print(f"  Avg Entry: {t.avg_entry_price:.2f}")
        print(f"  Max Size: {t.max_size:.2f}")
        print(f"  Net PnL: {t.net_pnl*100:.2f}%")

    # Test 4: MTM Equity Check
    print("\n[Test 4] Mark-to-Market Equity")
    df4 = pd.DataFrame({
        'open': [100, 100, 110, 100, 110],
        'high': [100, 110, 115, 110, 115],
        'low': [100, 95, 105, 95, 105],
        'close': [100, 105, 110, 100, 110],
        'signal': [0, 1, 1, 1, 0],
        'position_size': [0, 1.0, 1.0, 1.0, 0],
        'stop_price': [0, 0, 0, 0, 0]
    })

    result4 = run_backtest(df4, BacktestConfig(fee_rate=0.0, slippage_rate=0.0))
    print(f"  Equity Curve: {result4.equity_curve}")
    print(f"  Max DD during trade (bar 3): should show drawdown")
    print(f"  Max Drawdown: {result4.max_drawdown*100:.2f}%")

    print("\n" + "=" * 80)
    print("All tests completed!")


if __name__ == "__main__":
    test_backtester()
