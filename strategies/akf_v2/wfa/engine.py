"""
Walk-Forward Analysis 엔진.

피라미딩 지원 + Sigma-Spacing.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import json
import subprocess
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from itertools import product

from strategies.akf_v2.common.pyramiding import PyramidConfig, apply_pyramiding
from strategies.akf_v2.common.sizing import SizingConfig, calculate_position_sizes
from strategies.akf_v2.backtest_pyramid import run_pyramid_backtest, PyramidBacktestConfig


@dataclass
class WFAConfig:
    """Walk-Forward Analysis 설정."""

    # Window 설정 (일수 기반 - bars_per_day로 자동 스케일링)
    train_days: int = 365  # 1년
    test_days: int = 90    # 3개월
    step_days: int = 90    # 3개월

    # 최적화 범위
    entry_zscore_range: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    exit_zscore_range: Tuple[float, ...] = (-1.0, -1.5, -2.0, -2.5)

    # 시그널 파라미터 (일수 기반)
    zscore_days: float = 5.0      # 1주 (거래일)
    uncertainty_days: float = 20.0  # 1개월 (거래일)
    warmup_days: float = 20.0     # 1개월

    use_uncertainty_filter: bool = True
    uncertainty_percentile: float = 0.5

    # 사이징 및 피라미딩 설정 (고정)
    sizing_config: SizingConfig = field(default_factory=SizingConfig)
    pyramid_config: PyramidConfig = field(default_factory=PyramidConfig)


@dataclass
class FoldResult:
    """단일 Fold 결과."""

    fold_id: int
    train_period: str
    test_period: str

    # 최적 파라미터
    best_entry_zscore: float
    best_exit_zscore: float

    # 성과
    train_sharpe: float
    train_pnl: float
    test_sharpe: float
    test_pnl: float
    test_trades: int
    test_win_rate: float
    test_mdd: float


def estimate_bars_per_day(df: pd.DataFrame) -> float:
    """데이터에서 하루당 bar 수 추정."""
    start_ts = df["start_time"].iloc[0]
    end_ts = df["start_time"].iloc[-1]
    total_days = (end_ts - start_ts) / (1000 * 60 * 60 * 24)  # ms to days
    if total_days <= 0:
        return 6.0  # fallback
    return len(df) / total_days


class WFAEngine:
    """Walk-Forward Analysis 엔진."""

    def __init__(self, config: WFAConfig, use_python_backtest: bool = False):
        self.config = config
        self.use_python_backtest = use_python_backtest
        self._backtester_path = PROJECT_ROOT / "etl/bin/backtester"
        self._bars_per_day = 6.0  # default, updated in run()

    def run(self, df: pd.DataFrame) -> List[FoldResult]:
        """
        Walk-Forward Analysis 실행.

        Args:
            df: Kalman 필터가 적용된 DataFrame

        Returns:
            FoldResult 리스트
        """
        # bars_per_day 추정
        self._bars_per_day = estimate_bars_per_day(df)

        # 일수 기반 → bar 수 변환
        train_bars = int(self.config.train_days * self._bars_per_day)
        test_bars = int(self.config.test_days * self._bars_per_day)
        step_bars = int(self.config.step_days * self._bars_per_day)

        # 최소값 보장
        train_bars = max(train_bars, 500)
        test_bars = max(test_bars, 100)
        step_bars = max(step_bars, 100)

        print(f"  bars_per_day: {self._bars_per_day:.1f}")
        print(f"  train/test/step: {train_bars}/{test_bars}/{step_bars} bars")

        results = []
        n = len(df)
        fold_id = 0
        start_idx = 0

        total_window = train_bars + test_bars

        while start_idx + total_window <= n:
            train_end = start_idx + train_bars
            test_end = train_end + test_bars

            df_train = df.iloc[start_idx:train_end].copy()
            df_test = df.iloc[train_end:test_end].copy()

            # 기간 문자열
            train_start_t = pd.to_datetime(df_train["start_time"].iloc[0], unit="ms")
            train_end_t = pd.to_datetime(df_train["start_time"].iloc[-1], unit="ms")
            test_start_t = pd.to_datetime(df_test["start_time"].iloc[0], unit="ms")
            test_end_t = pd.to_datetime(df_test["start_time"].iloc[-1], unit="ms")

            train_period = f"{train_start_t:%Y-%m-%d}~{train_end_t:%Y-%m-%d}"
            test_period = f"{test_start_t:%Y-%m-%d}~{test_end_t:%Y-%m-%d}"

            print(f"\nFold {fold_id}: Train {train_period}")
            print(f"         Test  {test_period}")

            # 1. Train에서 최적 파라미터 탐색
            best_entry, best_exit, train_metrics = self._optimize_fold(df_train)
            print(
                f"  Best params: entry={best_entry:.1f}, exit={best_exit:.1f}, "
                f"Train Sharpe={train_metrics.get('sharpe_ratio', 0):.2f}"
            )

            # 2. Test에서 평가
            test_metrics = self._test_fold(df_test, best_entry, best_exit)

            if test_metrics:
                print(
                    f"  Test: Sharpe={test_metrics.get('sharpe_ratio', 0):.2f}, "
                    f"PnL={test_metrics.get('total_pnl', 0)*100:.1f}%, "
                    f"Trades={test_metrics.get('total_trades', 0)}"
                )

                results.append(
                    FoldResult(
                        fold_id=fold_id,
                        train_period=train_period,
                        test_period=test_period,
                        best_entry_zscore=best_entry,
                        best_exit_zscore=best_exit,
                        train_sharpe=train_metrics.get("sharpe_ratio", 0),
                        train_pnl=train_metrics.get("total_pnl", 0),
                        test_sharpe=test_metrics.get("sharpe_ratio", 0),
                        test_pnl=test_metrics.get("total_pnl", 0),
                        test_trades=test_metrics.get("total_trades", 0),
                        test_win_rate=test_metrics.get("win_rate", 0),
                        test_mdd=test_metrics.get("max_drawdown", 0),
                    )
                )

            fold_id += 1
            start_idx += step_bars

        return results

    def _optimize_fold(
        self, df_train: pd.DataFrame
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Train 데이터에서 최적 파라미터 탐색."""
        best_sharpe = -np.inf
        best_entry, best_exit = 1.0, -2.0
        best_metrics = {}

        for entry_z, exit_z in product(
            self.config.entry_zscore_range, self.config.exit_zscore_range
        ):
            df_signals = self._generate_signals(df_train.copy(), entry_z, exit_z)
            metrics = self._run_backtest(df_signals)

            if metrics and metrics.get("sharpe_ratio", -np.inf) > best_sharpe:
                best_sharpe = metrics["sharpe_ratio"]
                best_entry, best_exit = entry_z, exit_z
                best_metrics = metrics

        return best_entry, best_exit, best_metrics

    def _test_fold(
        self, df_test: pd.DataFrame, entry_zscore: float, exit_zscore: float
    ) -> Optional[Dict[str, Any]]:
        """Test 데이터에서 평가."""
        df_signals = self._generate_signals(df_test.copy(), entry_zscore, exit_zscore)
        return self._run_backtest(df_signals)

    def _generate_signals(
        self, df: pd.DataFrame, entry_zscore: float, exit_zscore: float
    ) -> pd.DataFrame:
        """시그널 생성 (피라미딩 적용)."""
        velocity = df["kf_velocity"].values
        uncertainty = df["kf_uncertainty"].values
        close = df["close"].values
        n = len(df)

        # 일수 기반 → bar 수 변환
        zscore_window = max(5, int(self.config.zscore_days * self._bars_per_day))
        uncertainty_window = max(20, int(self.config.uncertainty_days * self._bars_per_day))
        warmup = max(20, int(self.config.warmup_days * self._bars_per_day))

        # Velocity Z-score
        vel_series = pd.Series(velocity)
        vel_mean = vel_series.rolling(
            window=zscore_window, min_periods=5
        ).mean()
        vel_std = vel_series.rolling(
            window=zscore_window, min_periods=5
        ).std()
        vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

        # Entry 조건
        entry_long_cond = vel_zscore > entry_zscore
        entry_short_cond = vel_zscore < -2.0  # Short은 고정 threshold

        # Uncertainty filter
        if self.config.use_uncertainty_filter:
            p_pct = (
                pd.Series(uncertainty)
                .rolling(window=uncertainty_window, min_periods=20)
                .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
                .fillna(0.5)
                .values
            )
            low_uncertainty = p_pct < self.config.uncertainty_percentile
            entry_long_cond = entry_long_cond & low_uncertainty
            entry_short_cond = entry_short_cond & low_uncertainty

        # 기본 시그널 생성 (exit 포함)
        base_signals = np.zeros(n, dtype=np.int8)
        position = 0

        for i in range(warmup, n):
            if position == 0:
                if entry_long_cond[i]:
                    position = 1
                    base_signals[i] = 1
                elif entry_short_cond[i]:
                    position = -1
                    base_signals[i] = -1
            elif position == 1:
                if vel_zscore[i] < exit_zscore:
                    position = 0
                    base_signals[i] = 0
                else:
                    base_signals[i] = 1
            elif position == -1:
                if vel_zscore[i] > 0:
                    position = 0
                    base_signals[i] = 0
                else:
                    base_signals[i] = -1

        # 포지션 사이징
        base_sizes = calculate_position_sizes(
            df, self.config.sizing_config, entry_signals=base_signals
        )

        # 피라미딩 적용
        if self.config.pyramid_config.enabled:
            final_signals, final_sizes, sl_prices = apply_pyramiding(
                df,
                base_signals,
                base_sizes,
                self.config.pyramid_config,
                warmup=warmup,
            )
        else:
            # 시그널은 해당 바에서 생성 (shift 안 함)
            # 백테스터가 signal[i]를 보고 bar[i+1].open에서 매매 처리
            final_signals = base_signals
            final_sizes = base_sizes
            sl_prices = np.zeros(n)

        df["signal"] = final_signals
        df["position_size"] = final_sizes
        df["sl_price"] = sl_prices

        return df

    def _run_backtest(self, df_signals: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """백테스트 실행 (Python 또는 Go)."""
        if self.use_python_backtest:
            return self._run_python_backtest(df_signals)
        else:
            return self._run_go_backtest(df_signals)

    def _run_python_backtest(self, df_signals: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Python 백테스터 실행."""
        config = PyramidBacktestConfig(
            initial_capital=100000.0,
            compounding=True,  # 복리 모드
            fee_rate=0.001,
            slippage_rate=0.0001,
        )

        result = run_pyramid_backtest(df_signals, config)
        return result

    def _run_go_backtest(self, df_signals: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Go 백테스터 실행."""
        if not self._backtester_path.exists():
            print(f"Warning: Backtester not found at {self._backtester_path}")
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            signals_path = Path(tmpdir) / "signals.parquet"
            features_path = Path(tmpdir) / "features.parquet"

            # Signals 파일
            sl_price_col = (
                df_signals["sl_price"].astype(np.float64)
                if "sl_price" in df_signals.columns
                else np.zeros(len(df_signals), dtype=np.float64)
            )
            signals_df = pd.DataFrame(
                {
                    "timestamp": df_signals["start_time"].astype(np.int64),
                    "signal": df_signals["signal"].astype(np.int8),
                    "size": df_signals["position_size"].astype(np.float64),
                    "sl_price": sl_price_col,
                }
            )
            table = pa.Table.from_pandas(
                signals_df,
                schema=pa.schema(
                    [
                        ("timestamp", pa.int64()),
                        ("signal", pa.int8()),
                        ("size", pa.float64()),
                        ("sl_price", pa.float64()),
                    ]
                ),
            )
            pq.write_table(table, signals_path)

            # Features 파일
            volume = (
                np.exp(df_signals["log_volume"])
                if "log_volume" in df_signals.columns
                else np.ones(len(df_signals)) * 1000
            )
            high_time = (
                df_signals["high_time"].astype(np.int64)
                if "high_time" in df_signals.columns
                else df_signals["start_time"].astype(np.int64)
            )
            low_time = (
                df_signals["low_time"].astype(np.int64)
                if "low_time" in df_signals.columns
                else df_signals["start_time"].astype(np.int64)
            )
            realized_vol = (
                df_signals["realized_vol"].astype(np.float64)
                if "realized_vol" in df_signals.columns
                else np.ones(len(df_signals)) * 0.01
            )

            features_df = pd.DataFrame(
                {
                    "timestamp": df_signals["start_time"].astype(np.int64),
                    "open": df_signals["open"].astype(np.float64),
                    "high": df_signals["high"].astype(np.float64),
                    "high_time": high_time,
                    "low": df_signals["low"].astype(np.float64),
                    "low_time": low_time,
                    "close": df_signals["close"].astype(np.float64),
                    "volume": volume.astype(np.float64),
                    "realized_vol": realized_vol,
                }
            )
            table = pa.Table.from_pandas(
                features_df,
                schema=pa.schema(
                    [
                        ("timestamp", pa.int64()),
                        ("open", pa.float64()),
                        ("high", pa.float64()),
                        ("high_time", pa.int64()),
                        ("low", pa.float64()),
                        ("low_time", pa.int64()),
                        ("close", pa.float64()),
                        ("volume", pa.float64()),
                        ("realized_vol", pa.float64()),
                    ]
                ),
            )
            pq.write_table(table, features_path)

            cmd = [
                str(self._backtester_path),
                "-signals",
                str(signals_path),
                "-features",
                str(features_path),
                "-exit-mode",
                "signal",
                # "-compounding",  # 단리 모드로 검증 (Look-ahead bias 수정 후)
                "-quiet",
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                return None

            try:
                return json.loads(proc.stdout)
            except json.JSONDecodeError:
                return None
