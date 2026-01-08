"""
Sigma-Spacing 기반 피라미딩 모듈.

Log-Price 모델에서 √P가 변동성 비율(%)이므로:
- Long:  NextEntry = LastEntry + (k × √P)
- Short: NextEntry = LastEntry - (k × √P)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class PyramidConfig:
    """피라미딩 설정."""

    enabled: bool = True
    max_pyramid: int = 3  # 최대 피라미딩 횟수 (base + 2 add)
    pyramid_size: float = 1.0  # 각 피라미딩 추가 사이즈
    spacing_k: float = 1.5  # σ 배수 (1.0~2.0)
    max_total_size: float = 3.0  # 최대 총 사이즈

    # 손절 설정: SL = avg_entry ± sl_n * √P
    use_sl: bool = True
    sl_n: float = 1.5  # σ 배수 (0~2, 0이면 BEP에서 손절)


@dataclass
class PyramidState:
    """피라미딩 상태 추적."""

    position: int = 0  # 1: long, -1: short, 0: flat
    pyramid_count: int = 0  # 현재 피라미딩 횟수
    last_entry_price: float = 0.0  # 마지막 진입가 (spacing 기준)
    avg_entry_price: float = 0.0  # 평균 진입가 (PnL 계산용)
    total_units: float = 0.0  # 총 진입 유닛 (가중평균용)
    total_size: float = 0.0  # 총 포지션 사이즈

    def reset(self):
        """상태 초기화."""
        self.position = 0
        self.pyramid_count = 0
        self.last_entry_price = 0.0
        self.avg_entry_price = 0.0
        self.total_units = 0.0
        self.total_size = 0.0

    def enter(self, direction: int, price: float, size: float):
        """신규 진입."""
        self.position = direction
        self.pyramid_count = 1
        self.last_entry_price = price
        self.avg_entry_price = price
        self.total_units = size
        self.total_size = size

    def add_pyramid(self, price: float, size: float):
        """피라미딩 추가."""
        self.pyramid_count += 1
        self.last_entry_price = price

        # 평균 진입가 업데이트 (가중 평균)
        new_total = self.total_units + size
        self.avg_entry_price = (
            self.total_units * self.avg_entry_price + size * price
        ) / new_total
        self.total_units = new_total
        self.total_size += size


def check_pyramid_condition(
    state: PyramidState,
    current_price: float,
    kf_uncertainty: float,
    config: PyramidConfig,
) -> bool:
    """
    Sigma-Spacing 피라미딩 조건 체크.

    Log-price 모델에서:
    - Long:  log(current) >= log(last_entry) + k*σ
            => current >= last_entry * exp(k*σ)
    - Short: log(current) <= log(last_entry) - k*σ
            => current <= last_entry * exp(-k*σ)

    Args:
        state: 현재 피라미딩 상태
        current_price: 현재 가격
        kf_uncertainty: Kalman Filter의 P (error covariance)
        config: 피라미딩 설정

    Returns:
        피라미딩 조건 충족 여부
    """
    if not config.enabled:
        return False

    if state.pyramid_count >= config.max_pyramid:
        return False

    if state.total_size >= config.max_total_size:
        return False

    # σ = √P (Log-price 모델에서 변동성 비율)
    sigma = np.sqrt(kf_uncertainty)
    spacing = config.spacing_k * sigma

    if state.position == 1:  # Long
        # log(current) >= log(last_entry) + spacing
        # => current >= last_entry * exp(spacing)
        target_price = state.last_entry_price * np.exp(spacing)
        return current_price >= target_price
    elif state.position == -1:  # Short
        # log(current) <= log(last_entry) - spacing
        # => current <= last_entry * exp(-spacing)
        target_price = state.last_entry_price * np.exp(-spacing)
        return current_price <= target_price

    return False


def calculate_pyramid_sl(
    avg_entry_price: float,
    direction: int,
    kf_uncertainty: float,
    sl_n: float,
) -> float:
    """
    피라미딩 손절가 계산 (Log-price 모델).

    Log-price에서:
    - Long:  log(SL) = log(avg_entry) - sl_n * √P
            => SL = avg_entry * exp(-sl_n * √P)
    - Short: log(SL) = log(avg_entry) + sl_n * √P
            => SL = avg_entry * exp(sl_n * √P)

    Args:
        avg_entry_price: 평균 진입가 (BEP)
        direction: 포지션 방향 (1: long, -1: short)
        kf_uncertainty: Kalman Filter의 P (error covariance)
        sl_n: σ 배수 (0~2)

    Returns:
        손절가
    """
    sigma = np.sqrt(kf_uncertainty)
    sl_distance = sl_n * sigma

    if direction == 1:  # Long
        # log(SL) = log(entry) - sl_distance
        # SL = entry * exp(-sl_distance)
        return avg_entry_price * np.exp(-sl_distance)
    elif direction == -1:  # Short
        # log(SL) = log(entry) + sl_distance
        # SL = entry * exp(sl_distance)
        return avg_entry_price * np.exp(sl_distance)
    return 0.0


def apply_pyramiding(
    df: pd.DataFrame,
    base_signals: np.ndarray,
    base_sizes: np.ndarray,
    config: PyramidConfig,
    warmup: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sigma-Spacing 피라미딩 적용.

    Args:
        df: DataFrame with 'close', 'kf_uncertainty' columns
        base_signals: 기본 시그널 배열 (-1, 0, 1)
        base_sizes: 기본 포지션 사이즈 배열
        config: 피라미딩 설정
        warmup: 워밍업 기간

    Returns:
        (final_signals, final_sizes, sl_prices) 튜플
    """
    n = len(df)
    close = df["close"].values
    uncertainty = df["kf_uncertainty"].values

    final_signals = np.zeros(n, dtype=np.int8)
    final_sizes = np.zeros(n)
    sl_prices = np.zeros(n)

    state = PyramidState()

    for i in range(warmup, n):
        current_price = close[i]
        current_p = uncertainty[i]
        base_signal = base_signals[i]
        base_size = base_sizes[i] if base_sizes[i] > 0 else 1.0

        # === 포지션 없음 ===
        if state.position == 0:
            if base_signal != 0:
                # 신규 진입
                state.enter(int(base_signal), current_price, base_size)
                final_signals[i] = state.position
                final_sizes[i] = state.total_size

                # 손절가 설정
                if config.use_sl:
                    sl_prices[i] = calculate_pyramid_sl(
                        state.avg_entry_price,
                        state.position,
                        current_p,
                        config.sl_n,
                    )

        # === Long 포지션 ===
        elif state.position == 1:
            # 손절 체크
            sl_hit = config.use_sl and sl_prices[i - 1] > 0 and current_price <= sl_prices[i - 1]

            if base_signal == 0 or base_signal == -1 or sl_hit:
                # 청산 (시그널 0, 반전, 또는 손절)
                final_signals[i] = 0
                final_sizes[i] = 0
                sl_prices[i] = 0
                state.reset()
            else:
                # 유지
                final_signals[i] = 1

                # 피라미딩 체크
                if check_pyramid_condition(state, current_price, current_p, config):
                    add_size = min(
                        config.pyramid_size,
                        config.max_total_size - state.total_size,
                    )
                    if add_size > 0:
                        state.add_pyramid(current_price, add_size)

                        # 손절가 업데이트 (새 평균가 기준)
                        if config.use_sl:
                            sl_prices[i] = calculate_pyramid_sl(
                                state.avg_entry_price,
                                state.position,
                                current_p,
                                config.sl_n,
                            )
                        else:
                            sl_prices[i] = sl_prices[i - 1]
                    else:
                        sl_prices[i] = sl_prices[i - 1]
                else:
                    sl_prices[i] = sl_prices[i - 1]

                final_sizes[i] = min(state.total_size, config.max_total_size)

        # === Short 포지션 ===
        elif state.position == -1:
            # 손절 체크
            sl_hit = config.use_sl and sl_prices[i - 1] > 0 and current_price >= sl_prices[i - 1]

            if base_signal == 0 or base_signal == 1 or sl_hit:
                # 청산 (시그널 0, 반전, 또는 손절)
                final_signals[i] = 0
                final_sizes[i] = 0
                sl_prices[i] = 0
                state.reset()
            else:
                # 유지
                final_signals[i] = -1

                # 피라미딩 체크
                if check_pyramid_condition(state, current_price, current_p, config):
                    add_size = min(
                        config.pyramid_size,
                        config.max_total_size - state.total_size,
                    )
                    if add_size > 0:
                        state.add_pyramid(current_price, add_size)

                        # 손절가 업데이트 (새 평균가 기준)
                        if config.use_sl:
                            sl_prices[i] = calculate_pyramid_sl(
                                state.avg_entry_price,
                                state.position,
                                current_p,
                                config.sl_n,
                            )
                        else:
                            sl_prices[i] = sl_prices[i - 1]
                    else:
                        sl_prices[i] = sl_prices[i - 1]
                else:
                    sl_prices[i] = sl_prices[i - 1]

                final_sizes[i] = min(state.total_size, config.max_total_size)

    # 시그널은 해당 바에서 생성 (shift 안 함)
    # 백테스터가 signal[i]를 보고 bar[i+1].open에서 매매 처리

    return final_signals, final_sizes, sl_prices


def calculate_pyramid_spacing(
    kf_uncertainty: float, k: float = 1.5
) -> float:
    """
    피라미딩 간격 계산 (Log-price 스케일).

    Log-price에서 k*σ 만큼 떨어진 가격을 계산할 때:
    - next_price = current_price * exp(k * σ)
    - 실제 가격 비율 = exp(k * σ) - 1

    Args:
        kf_uncertainty: Kalman Filter의 P (error covariance)
        k: σ 배수 (기본 1.5)

    Returns:
        간격 (Log-price 스케일, 예: 0.02 ≈ 2% price move)
    """
    sigma = np.sqrt(kf_uncertainty)
    return k * sigma  # Log-price 스케일


def calculate_price_spacing(
    current_price: float, kf_uncertainty: float, k: float = 1.5
) -> float:
    """
    실제 가격 스케일에서 피라미딩 간격 계산.

    Args:
        current_price: 현재 가격
        kf_uncertainty: Kalman Filter의 P (error covariance)
        k: σ 배수 (기본 1.5)

    Returns:
        다음 피라미딩 목표 가격 (Long 기준)
    """
    sigma = np.sqrt(kf_uncertainty)
    return current_price * np.exp(k * sigma)
