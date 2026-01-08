"""
AKF V2 Strategy Lab - Step-by-Step Verification.

4단계 검증 프로세스:
- Step 1: Velocity 기본 엔진 (방향성 확인)
- Step 2: Dynamic Band (P 활용, 노이즈 필터링)
- Step 3: Gain Filter (K 활용, 급변 구간 회피)
- Step 4: Position Sizing (P 활용, 리스크 조절)

Usage:
    from strategies.akf_v2.lab import StrategyLab
    lab = StrategyLab()
    lab.load_data()
    lab.run_step(1)  # Step 1 실행
    lab.run_step(2)  # Step 2 실행
    ...
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from research.features.kalman import calculate_adaptive_kalman


@dataclass
class StepConfig:
    """Configuration for each step."""
    # Step 1: Velocity (Z-score 기반 동적 파라미터)
    entry_zscore: float = 1.0  # 진입: velocity_zscore > entry_zscore
    exit_zscore: float = -2.0  # 청산: velocity_zscore < exit_zscore
    zscore_window: int = 20  # z-score 계산 윈도우
    long_only: bool = False  # Long + Short

    # Exit Conditions (청산 조건)
    exit_trend_reversal: bool = True  # z-score 기반 청산
    exit_trailing_stop: bool = False  # 트레일링 스탑: close < kf_trend
    exit_innovation_stop: bool = True  # 이노베이션 손절: std_innovation < -3.0
    innovation_threshold: float = -3.0  # 이노베이션 손절 임계값

    # P 기반 진입 필터 (MDD 개선용)
    use_uncertainty_filter: bool = False  # P가 낮을 때만 진입
    uncertainty_percentile: float = 0.5  # P가 이 백분위수 이하일 때 진입

    # Step 2: Dynamic Band
    use_dynamic_band: bool = False
    band_base_k: float = 1.5  # 기본 밴드 배수
    band_p_scale: float = 1.0  # P에 의한 밴드 확장 비율

    # 동적밴드 청산 (P 기반) - 효과 미미, 비활성화 권장
    exit_dynamic_band: bool = False  # 효과 없음 (z-score 청산으로 충분)
    exit_band_k: float = 2.0  # 사용 안 함

    # Step 3: Gain Filter
    use_gain_filter: bool = False
    gain_threshold: float = 0.8  # K > threshold면 진입 금지

    # Step 4: Position Sizing
    use_position_sizing: bool = False
    base_size: float = 1.0
    min_size: float = 0.25
    max_size: float = 2.0


class StrategyLab:
    """
    단계별 전략 검증 실험실.

    칼만 필터의 각 출력(velocity, P, K)을 순차적으로 활용하여
    전략 성능 개선을 검증합니다.
    """

    DATA_DIR = PROJECT_ROOT / "etl/data/features-6/futures/BTCUSDT"
    BACKTESTER = PROJECT_ROOT / "etl/bin/backtester"

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.df_kf: Optional[pd.DataFrame] = None
        self.results: Dict[int, dict] = {}

    def load_data(
        self,
        start_year: int = 2020,
        end_year: int = 2025,
        end_month: int = 12
    ) -> None:
        """Load BTCUSDT data."""
        months = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                if year == end_year and month > end_month:
                    break
                months.append(f"{year}-{month:02d}")

        dfs = []
        for month in months:
            path = self.DATA_DIR / f"BTCUSDT-features-{month}.parquet"
            if path.exists():
                dfs.append(pd.read_parquet(path))

        self.df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(self.df):,} bars ({months[0]} ~ {months[-1]})")
        print(f"Price: ${self.df['close'].min():,.0f} ~ ${self.df['close'].max():,.0f}")

        # Apply Kalman Filter
        print("Applying Kalman Filter (log-price mode)...")
        self.df_kf = calculate_adaptive_kalman(self.df)
        print(f"Kalman features added: kf_trend, kf_velocity, kf_uncertainty, kf_gain")

    def run_step(self, step: int, config: Optional[StepConfig] = None) -> dict:
        """
        Run strategy for specified step.

        Args:
            step: 1, 2, 3, or 4
            config: Optional custom config (defaults based on step)

        Returns:
            Backtest metrics
        """
        if self.df_kf is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Get default config for step
        if config is None:
            config = self._get_step_config(step)

        print(f"\n{'='*60}")
        print(f"STEP {step}: {self._get_step_name(step)}")
        print(f"{'='*60}")

        # Generate signals
        df_signals = self._generate_signals(self.df_kf.copy(), config, step)

        # Signal stats
        signals = df_signals['signal']
        n_long = (signals == 1).sum()
        n_short = (signals == -1).sum()
        n_trades = n_long + n_short
        print(f"Signals: Long={n_long:,}, Short={n_short:,}, Total={n_trades:,}")

        # Run backtest
        metrics = self._run_backtest(df_signals)

        if metrics:
            self._print_metrics(metrics, step)
            self.results[step] = metrics

        return metrics

    def _get_step_config(self, step: int) -> StepConfig:
        """Get default config for each step."""
        # Z-score 기반 동적 파라미터

        if step == 1:
            # Step 1: Trend Following (Z-score 기반)
            # 진입: velocity z-score > 1.0 (1σ)
            # 청산: velocity z-score < -2.0 (2σ)
            return StepConfig(
                entry_zscore=1.0,
                exit_zscore=-2.0,
                zscore_window=20,
                exit_trend_reversal=True,
                exit_trailing_stop=False,
                exit_innovation_stop=True,
                innovation_threshold=-3.0,
                use_dynamic_band=False,
                use_gain_filter=False,
                use_position_sizing=False,
            )
        elif step == 2:
            # Step 2: P 기반 진입 필터 (Uncertainty Filter)
            return StepConfig(
                entry_zscore=1.0,
                exit_zscore=-2.0,
                zscore_window=20,
                exit_trend_reversal=True,
                exit_trailing_stop=False,
                exit_innovation_stop=True,
                innovation_threshold=-3.0,
                use_uncertainty_filter=True,
                uncertainty_percentile=0.5,
                use_dynamic_band=False,
                use_gain_filter=False,
                use_position_sizing=False,
            )
        elif step == 3:
            # Step 3: Gain Filter 추가
            return StepConfig(
                entry_zscore=1.0,
                exit_zscore=-2.0,
                zscore_window=20,
                exit_trend_reversal=True,
                exit_trailing_stop=False,
                exit_innovation_stop=True,
                innovation_threshold=-3.0,
                use_uncertainty_filter=True,
                uncertainty_percentile=0.5,
                use_dynamic_band=False,
                use_gain_filter=True,
                gain_threshold=0.5,
                use_position_sizing=False,
            )
        elif step == 4:
            # Step 4: Position Sizing 추가
            return StepConfig(
                entry_zscore=1.0,
                exit_zscore=-2.0,
                zscore_window=20,
                exit_trend_reversal=True,
                exit_trailing_stop=False,
                exit_innovation_stop=True,
                innovation_threshold=-3.0,
                use_uncertainty_filter=True,
                uncertainty_percentile=0.5,
                use_dynamic_band=False,
                use_gain_filter=True,
                gain_threshold=0.5,
                use_position_sizing=True,
                base_size=1.0,
                min_size=0.25,
                max_size=2.0,
            )
        else:
            raise ValueError(f"Invalid step: {step}. Must be 1-4.")

    def _get_step_name(self, step: int) -> str:
        """Get step description."""
        names = {
            1: "Velocity 기본 엔진 (방향성 검증)",
            2: "Uncertainty Filter (P 활용 - MDD 개선)",
            3: "Gain Filter (K 활용 - 급변 구간 회피)",
            4: "Position Sizing (P 활용 - 리스크 조절)",
        }
        return names.get(step, "Unknown")

    def _generate_signals(
        self,
        df: pd.DataFrame,
        config: StepConfig,
        step: int
    ) -> pd.DataFrame:
        """
        Generate trading signals based on step configuration.

        핵심 발견: Velocity는 Mean Reversion 특성을 보임
        - Velocity가 크게 음수 → 이후 반등 (Long 기회)
        - Velocity가 크게 양수 → 이후 조정 (Short 기회)

        청산 조건:
        1. 추세 반전: velocity 방향 전환
        2. 트레일링 스탑: price < kf_trend
        3. 이노베이션 손절: std_innovation < -3.0
        """
        result = df.copy()

        # === 데이터 준비 ===
        velocity = result['kf_velocity'].values
        close = result['close'].values
        kf_trend = result['kf_trend'].values
        std_innovation = result['kf_std_innovation'].values if 'kf_std_innovation' in result.columns else np.zeros(len(result))
        kf_uncertainty = result['kf_uncertainty'].values

        # === 동적밴드 계산 (P 기반) ===
        # P는 log-space의 분산, kf_trend는 price space
        # 밴드 = kf_trend * exp(±k * sqrt(P))
        # P가 클수록 밴드가 넓어짐 (불확실할 때 여유)
        sqrt_p = np.sqrt(kf_uncertainty)
        upper_band = kf_trend * np.exp(config.exit_band_k * sqrt_p)
        lower_band = kf_trend * np.exp(-config.exit_band_k * sqrt_p)

        n = len(result)

        # === Velocity Z-score 계산 (동적 파라미터) ===
        vel_series = pd.Series(velocity)
        vel_mean = vel_series.rolling(window=config.zscore_window, min_periods=5).mean()
        vel_std = vel_series.rolling(window=config.zscore_window, min_periods=5).std()
        vel_zscore = ((vel_series - vel_mean) / (vel_std + 1e-10)).values

        # === 진입 조건 계산 ===
        # Z-score 기반 Trend Following
        # Long: z-score > 1.0 (1σ 이상 상승)
        # Short: z-score < -2.0 (2σ 이상 하락) - 더 엄격하게
        entry_long_cond = vel_zscore > config.entry_zscore   # z-score > entry_zscore → Long
        entry_short_cond = vel_zscore < -2.0  # Short는 더 엄격하게 (2σ)

        # === P 기반 진입 필터 (MDD 개선용) ===
        if config.use_uncertainty_filter:
            # P가 낮을 때만 진입 (확신 높을 때)
            p = result['kf_uncertainty'].values
            p_pct = pd.Series(p).rolling(window=100, min_periods=20).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            ).fillna(0.5).values

            low_uncertainty = p_pct < config.uncertainty_percentile
            entry_long_cond = entry_long_cond & low_uncertainty
            entry_short_cond = entry_short_cond & low_uncertainty

        # === STEP 2: Dynamic Band (P 활용) ===
        if config.use_dynamic_band:
            p = result['kf_uncertainty']
            p_normalized = p / p.rolling(window=100, min_periods=20).mean()
            p_normalized = p_normalized.fillna(1.0).values

            band_width = config.band_base_k * (1 + config.band_p_scale * (p_normalized - 1))
            band_width = np.clip(band_width, 0.5, 5.0)

            deviation = close - kf_trend
            dev_std = pd.Series(deviation).rolling(window=20, min_periods=5).std().values
            z_score = deviation / (dev_std + 1e-10)

            # Mean Reversion: 하단 밴드 이탈 → Long, 상단 밴드 이탈 → Short
            oversold = z_score < -band_width
            overbought = z_score > band_width

            entry_long_cond = entry_long_cond & oversold
            entry_short_cond = entry_short_cond & overbought

        # === STEP 3: Gain Filter (K 활용) ===
        if config.use_gain_filter:
            k = result['kf_gain'].values
            stable_market = k < config.gain_threshold

            entry_long_cond = entry_long_cond & stable_market
            entry_short_cond = entry_short_cond & stable_market

        # Long Only 모드
        if config.long_only:
            entry_short_cond = np.zeros(n, dtype=bool)

        # === 상태 기반 시그널 생성 (진입 + 청산) ===
        signals = np.zeros(n, dtype=np.int8)
        position = 0  # 0: flat, 1: long, -1: short

        for i in range(100, n):  # Warmup: 100바 이후부터
            if position == 0:
                # === FLAT → 진입 체크 ===
                if entry_long_cond[i]:
                    position = 1
                    signals[i] = 1
                elif entry_short_cond[i]:
                    position = -1
                    signals[i] = -1

            elif position == 1:
                # === LONG 포지션 → 청산 체크 ===
                # Trend Following Long: 상승 추세 진입 → 추세 꺾이면 청산
                should_exit = False

                # 1. 추세 반전 청산: velocity z-score < exit_zscore (강한 하락)
                if config.exit_trend_reversal and vel_zscore[i] < config.exit_zscore:
                    should_exit = True

                # 2. 트레일링 스탑: price < kf_trend (추세선 이탈)
                if config.exit_trailing_stop and close[i] < kf_trend[i]:
                    should_exit = True

                # 3. 이노베이션 손절: std_innovation < threshold (급락)
                if config.exit_innovation_stop and std_innovation[i] < config.innovation_threshold:
                    should_exit = True

                # 4. 동적밴드 청산: price < lower_band (하단 밴드 이탈)
                if config.exit_dynamic_band and close[i] < lower_band[i]:
                    should_exit = True

                if should_exit:
                    position = 0
                    signals[i] = 0
                else:
                    signals[i] = 1

            elif position == -1:
                # === SHORT 포지션 → 청산 체크 ===
                # Trend Following Short: 하락 추세 진입 → 추세 꺾이면 청산
                should_exit = False

                # 1. 추세 반전 청산: velocity z-score > 0 (상승 전환 시 바로 청산)
                # Short는 빨리 청산해서 손실 제한
                if config.exit_trend_reversal and vel_zscore[i] > 0:
                    should_exit = True

                # 2. 트레일링 스탑: price > kf_trend (추세선 돌파)
                if config.exit_trailing_stop and close[i] > kf_trend[i]:
                    should_exit = True

                # 3. 이노베이션 손절: std_innovation > threshold (급등)
                if config.exit_innovation_stop and std_innovation[i] > -config.innovation_threshold:
                    should_exit = True

                # 4. 동적밴드 청산: price > upper_band (상단 밴드 돌파)
                if config.exit_dynamic_band and close[i] > upper_band[i]:
                    should_exit = True

                if should_exit:
                    position = 0
                    signals[i] = 0
                else:
                    signals[i] = -1

        result['signal'] = signals

        # === STEP 4: Position Sizing (P 활용) ===
        if config.use_position_sizing:
            p = result['kf_uncertainty']
            p_inv = 1.0 / (np.sqrt(p) + 1e-10)
            p_inv_pct = p_inv.rolling(window=100, min_periods=20).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )
            p_inv_pct = p_inv_pct.fillna(0.5)

            size_range = config.max_size - config.min_size
            result['position_size'] = config.min_size + size_range * p_inv_pct
        else:
            result['position_size'] = config.base_size

        return result

    def _run_backtest(self, df_signals: pd.DataFrame) -> Optional[dict]:
        """Run Go backtester."""
        if not self.BACKTESTER.exists():
            print(f"Warning: Backtester not found at {self.BACKTESTER}")
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            signals_path = Path(tmpdir) / "signals.parquet"
            features_path = Path(tmpdir) / "features.parquet"

            # Save signals
            signals_df = pd.DataFrame({
                'timestamp': df_signals['start_time'].astype(np.int64),
                'signal': df_signals['signal'].astype(np.int8),
            })
            table = pa.Table.from_pandas(signals_df, schema=pa.schema([
                ('timestamp', pa.int64()), ('signal', pa.int8())
            ]))
            pq.write_table(table, signals_path)

            # Save features
            volume = np.exp(df_signals['log_volume']) if 'log_volume' in df_signals.columns else np.ones(len(df_signals)) * 1000

            features_df = pd.DataFrame({
                'timestamp': df_signals['start_time'].astype(np.int64),
                'open': df_signals['open'].astype(np.float64),
                'high': df_signals['high'].astype(np.float64),
                'low': df_signals['low'].astype(np.float64),
                'close': df_signals['close'].astype(np.float64),
                'volume': volume.astype(np.float64),
                'realized_vol': df_signals['realized_vol'].astype(np.float64),
            })
            table = pa.Table.from_pandas(features_df, schema=pa.schema([
                ('timestamp', pa.int64()),
                ('open', pa.float64()), ('high', pa.float64()),
                ('low', pa.float64()), ('close', pa.float64()),
                ('volume', pa.float64()), ('realized_vol', pa.float64()),
            ]))
            pq.write_table(table, features_path)

            # Run backtester
            cmd = [
                str(self.BACKTESTER),
                '-signals', str(signals_path),
                '-features', str(features_path),
                '-exit-mode', 'signal',
                '-quiet',
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"Backtest error: {proc.stderr}")
                return None

            return json.loads(proc.stdout)

    def _print_metrics(self, metrics: dict, step: int) -> None:
        """Print backtest metrics."""
        print(f"\n📊 Results:")
        print(f"  Trades:      {metrics.get('total_trades', 0):,}")
        print(f"  Win Rate:    {metrics.get('win_rate', 0)*100:.1f}%")
        print(f"  Avg PnL:     {metrics.get('avg_pnl', 0)*100:.3f}%")
        print(f"  Total PnL:   {metrics.get('total_pnl', 0)*100:.1f}%")
        print(f"  Sharpe:      {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  MDD:         {metrics.get('max_drawdown', 0)*100:.1f}%")
        print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")

        # Step-specific commentary
        if step == 1:
            pnl = metrics.get('total_pnl', 0)
            if pnl > 0:
                print(f"\n✅ 엔진 방향성 OK - 수익 곡선 우상향")
            else:
                print(f"\n⚠️ 엔진 방향성 문제 - Q, R 튜닝 필요")
        elif step == 2:
            trades = metrics.get('total_trades', 0)
            prev_trades = self.results.get(1, {}).get('total_trades', trades * 5)
            reduction = (1 - trades / prev_trades) * 100 if prev_trades > 0 else 0
            print(f"\n📉 매매 횟수 감소: {reduction:.0f}%")
            if reduction > 50:
                print(f"✅ 노이즈 필터링 효과 확인")
        elif step == 3:
            mdd = metrics.get('max_drawdown', 0)
            prev_mdd = self.results.get(2, {}).get('max_drawdown', mdd)
            if mdd < prev_mdd:
                print(f"\n✅ MDD 개선: {prev_mdd*100:.1f}% → {mdd*100:.1f}%")
            else:
                print(f"\n⚠️ MDD 개선 미미 - gain_threshold 조정 필요")
        elif step == 4:
            sharpe = metrics.get('sharpe_ratio', 0)
            prev_sharpe = self.results.get(3, {}).get('sharpe_ratio', sharpe)
            if sharpe > prev_sharpe:
                print(f"\n✅ Sharpe 개선: {prev_sharpe:.2f} → {sharpe:.2f}")
            else:
                print(f"\n⚠️ Sharpe 개선 미미 - sizing 파라미터 조정 필요")

    def compare_steps(self) -> None:
        """Compare all completed steps."""
        if not self.results:
            print("No results to compare. Run steps first.")
            return

        print(f"\n{'='*70}")
        print("STEP COMPARISON")
        print(f"{'='*70}")
        print(f"{'Step':<8} {'Trades':>8} {'Win%':>8} {'PnL%':>10} {'Sharpe':>8} {'MDD%':>8}")
        print("-" * 70)

        for step in sorted(self.results.keys()):
            m = self.results[step]
            print(f"Step {step:<3} {m.get('total_trades',0):>8,} "
                  f"{m.get('win_rate',0)*100:>7.1f}% "
                  f"{m.get('total_pnl',0)*100:>9.1f}% "
                  f"{m.get('sharpe_ratio',0):>8.2f} "
                  f"{m.get('max_drawdown',0)*100:>7.1f}%")


def main():
    """Quick test."""
    lab = StrategyLab()
    lab.load_data()

    for step in range(1, 5):
        lab.run_step(step)

    lab.compare_steps()


if __name__ == "__main__":
    main()
