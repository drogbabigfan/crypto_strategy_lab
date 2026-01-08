"""
포지션 사이징 모듈.

지원 방식:
- fixed: 고정 사이즈 (1.0)
- p_inverse: P-역수 기반 (uncertainty 낮을수록 큰 사이즈)
- kelly: Kelly Criterion 기반 (Trade 단위)
- kelly_kf: Kelly + KF 분포 기반 확률 (Probabilistic Approach)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional


@dataclass
class SizingConfig:
    """포지션 사이징 설정."""

    method: str = "kf_prob"  # "fixed", "kelly", "kf_prob"
    min_size: float = 0.25
    max_size: float = 3.0
    kelly_trades: int = 50  # Kelly용 이동평균 트레이드 수
    kelly_window: int = 1000  # Kelly용 트레이드 추출 바 윈도우


def calculate_position_sizes(
    df: pd.DataFrame,
    config: SizingConfig,
    entry_signals: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    포지션 사이즈 계산.

    Args:
        df: DataFrame with 'close', 'kf_uncertainty' columns
        config: 사이징 설정
        entry_signals: 진입 시그널 (Kelly 방식에서 사용)

    Returns:
        position_size 배열
    """
    n = len(df)

    if config.method == "fixed":
        return np.ones(n)

    elif config.method == "kelly":
        if entry_signals is None:
            return np.ones(n) * config.min_size
        return _calculate_kelly_sizing(df, config, entry_signals)

    elif config.method == "kf_prob":
        return _calculate_kf_prob_sizing(df, config)

    else:
        return np.ones(n)


def _calculate_kf_prob_sizing(
    df: pd.DataFrame, config: SizingConfig
) -> np.ndarray:
    """
    KF 분포 기반 확률 사이징 (The Probabilistic Approach).

    칼만 필터의 확률분포를 이용해 "다음 스텝에서 가격이 오를 확률"을 직접 계산.

    수식:
        Z = Velocity / √P_vel
        p_win = CDF(Z)

    예시:
        - v=0.005, √P=0.002 → Z=2.5 → p≈99.3% (풀베팅)
        - v=0.005, √P=0.01  → Z=0.5 → p≈69%  (비중 축소)
    """
    velocity = df["kf_velocity"].values
    uncertainty = df["kf_uncertainty"].values
    n = len(df)

    position_sizes = np.ones(n) * config.min_size

    for i in range(100, n):  # warmup
        v = velocity[i]
        p_val = uncertainty[i]

        # Z = Velocity / √P
        sigma = np.sqrt(p_val) + 1e-10
        z_score = v / sigma

        # p_win = Φ(|Z|) - 방향 무관하게 신뢰도 측정
        # |Z|가 클수록 velocity가 확실 → 큰 사이즈
        p_win = norm.cdf(abs(z_score))

        # p_win을 사이즈로 스케일링
        # p_win ∈ [0.5, 1.0] → size ∈ [min_size, max_size]
        # p_win = 0.5 (Z=0, 불확실) → min_size
        # p_win = 1.0 (Z→∞, 확실) → max_size
        size_ratio = (p_win - 0.5) * 2  # [0, 1]로 정규화
        size_ratio = np.clip(size_ratio, 0, 1)

        size = config.min_size + (config.max_size - config.min_size) * size_ratio
        position_sizes[i] = size

    return position_sizes


def _extract_completed_trades(
    signals: np.ndarray,
    prices: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> np.ndarray:
    """
    완료된 트레이드의 수익률 추출 (Trade 단위).

    Args:
        signals: 시그널 배열 (-1, 0, 1)
        prices: 가격 배열
        start_idx: 시작 인덱스
        end_idx: 종료 인덱스 (미포함)

    Returns:
        트레이드별 수익률 배열
    """
    trade_returns = []
    position = 0
    entry_price = 0.0

    for i in range(start_idx, end_idx):
        signal = signals[i]
        price = prices[i]

        if position == 0:
            # Flat → Entry
            if signal == 1:
                position = 1
                entry_price = price
            elif signal == -1:
                position = -1
                entry_price = price

        elif position == 1:
            # Long position
            if signal != 1:  # Exit (signal changed to 0 or -1)
                pnl = (price - entry_price) / entry_price
                trade_returns.append(pnl)
                position = 0

                # Reversal: immediately enter short
                if signal == -1:
                    position = -1
                    entry_price = price

        elif position == -1:
            # Short position
            if signal != -1:  # Exit (signal changed to 0 or 1)
                pnl = (entry_price - price) / entry_price
                trade_returns.append(pnl)
                position = 0

                # Reversal: immediately enter long
                if signal == 1:
                    position = 1
                    entry_price = price

    return np.array(trade_returns)


def _calculate_kelly_sizing(
    df: pd.DataFrame,
    config: SizingConfig,
    entry_signals: np.ndarray,
) -> np.ndarray:
    """
    Kelly Criterion 기반 사이징 (Trade 단위).

    롤링 윈도우에서 완료된 트레이드의 승률과 손익비를 계산하여 Kelly fraction 적용.
    P 조정으로 uncertainty 반영.
    """
    close = df["close"].values
    uncertainty = df["kf_uncertainty"].values
    n = len(df)

    window = config.kelly_window
    mid_size = (config.min_size + config.max_size) / 2
    position_sizes = np.ones(n) * mid_size

    for i in range(window, n):
        idx_start = max(0, i - window)

        # Trade 단위로 완료된 거래 수익률 추출
        trade_returns = _extract_completed_trades(
            entry_signals, close, idx_start, i
        )

        if len(trade_returns) < 5:
            # 트레이드 수가 너무 적으면 보수적으로
            kelly_raw = 0.25
        else:
            wins = trade_returns[trade_returns > 0]
            losses = trade_returns[trade_returns < 0]

            if len(wins) == 0 or len(losses) == 0:
                kelly_raw = 0.25
            else:
                win_rate = len(wins) / len(trade_returns)
                avg_win = np.mean(wins)
                avg_loss = -np.mean(losses)

                # Kelly formula: f = p - (1-p)/b, where b = avg_win/avg_loss
                b = avg_win / (avg_loss + 1e-10)
                kelly_raw = win_rate - (1 - win_rate) / (b + 1e-10)
                kelly_raw = np.clip(kelly_raw, 0, 1)

                # Half-Kelly for safety
                kelly_raw *= 0.5

        # P 조정 (uncertainty 낮을수록 사이즈 증가)
        p_val = uncertainty[i]
        p_mean_local = np.mean(uncertainty[max(0, i - 100) : i + 1])
        p_ratio = p_mean_local / (p_val + 1e-10)
        p_ratio = np.clip(p_ratio, 0.5, 2.0)

        kelly_adjusted = kelly_raw * p_ratio
        size = config.min_size + (config.max_size - config.min_size) * kelly_adjusted
        position_sizes[i] = np.clip(size, config.min_size, config.max_size)

    return position_sizes


def calculate_kelly_fraction(
    trade_returns: np.ndarray, half_kelly: bool = True
) -> float:
    """
    Kelly fraction 계산.

    Args:
        trade_returns: 거래별 수익률 배열
        half_kelly: True면 절반 Kelly 적용 (보수적)

    Returns:
        Kelly fraction (0~1)
    """
    if len(trade_returns) < 10:
        return 0.5

    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    if len(wins) == 0 or len(losses) == 0:
        return 0.5

    win_rate = len(wins) / len(trade_returns)
    avg_win = np.mean(wins)
    avg_loss = -np.mean(losses)

    b = avg_win / (avg_loss + 1e-10)
    kelly = win_rate - (1 - win_rate) / (b + 1e-10)
    kelly = np.clip(kelly, 0, 1)

    if half_kelly:
        kelly *= 0.5

    return kelly
