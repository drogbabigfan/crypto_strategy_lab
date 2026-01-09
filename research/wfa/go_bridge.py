"""
Go Bridge - Python to Go Backtester interface.

Calls the Go backtester via subprocess and parses JSON results.
"""

import json
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class BacktestConfig:
    """Configuration for Go backtester."""

    # Exit mode: "tbm" (Triple Barrier), "signal" (opposite signal), "custom_stop" (per-bar stop prices)
    exit_mode: str = "tbm"

    sl_mult: float = 2.0
    pt_mult: float = 2.5
    max_hold_bars: int = 100

    base_fee: float = 0.001
    base_slippage: float = 0.0001
    impact_coeff: float = 0.1
    slippage_cap: float = 0.005

    initial_capital: float = 100000.0
    risk_per_trade: float = 1.0  # 1.0 = 100% capital, used with Size for leverage
    max_leverage: float = 10.0  # Maximum allowed leverage
    compounding: bool = True  # True for compound returns

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization (Go-compatible keys)."""
        # Go expects PascalCase keys
        return {
            "ExitMode": self.exit_mode,
            "SLMult": self.sl_mult,
            "PTMult": self.pt_mult,
            "MaxHoldBars": self.max_hold_bars,
            "BaseFee": self.base_fee,
            "BaseSlippage": self.base_slippage,
            "ImpactCoeff": self.impact_coeff,
            "SlippageCap": self.slippage_cap,
            "InitialCapital": self.initial_capital,
            "RiskPerTrade": self.risk_per_trade,
            "MaxLeverage": self.max_leverage,
            "Compounding": self.compounding,
        }


@dataclass
class BacktestResult:
    """Result from Go backtester."""

    total_trades: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    avg_hold_bars: float = 0.0
    tp_count: int = 0
    sl_count: int = 0
    timeout_count: int = 0
    equity_curve: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "BacktestResult":
        """Create from JSON dict."""
        return cls(
            total_trades=data.get("total_trades", 0),
            win_rate=data.get("win_rate", 0.0),
            avg_pnl=data.get("avg_pnl", 0.0),
            total_pnl=data.get("total_pnl", 0.0),
            sharpe_ratio=data.get("sharpe_ratio", 0.0),
            max_drawdown=data.get("max_drawdown", 0.0),
            profit_factor=data.get("profit_factor", 0.0),
            avg_hold_bars=data.get("avg_hold_bars", 0.0),
            tp_count=data.get("tp_count", 0),
            sl_count=data.get("sl_count", 0),
            timeout_count=data.get("timeout_count", 0),
            equity_curve=data.get("equity_curve", []),
        )

    def is_valid(self) -> bool:
        """Check if result has meaningful trades."""
        return self.total_trades > 0


class GoBridge:
    """
    Bridge between Python and Go backtester.

    Usage:
        bridge = GoBridge("./bin/backtester")
        result = bridge.run_backtest(
            signals=signals_array,
            features_path="features.parquet",
            config=BacktestConfig(sl_mult=2.0, pt_mult=2.5)
        )
    """

    def __init__(self, backtester_path: str = "./etl/bin/backtester"):
        """
        Initialize Go bridge.

        Args:
            backtester_path: Path to Go backtester binary
        """
        self.backtester_path = Path(backtester_path)

    def _check_binary(self) -> None:
        """Verify backtester binary exists."""
        if not self.backtester_path.exists():
            raise FileNotFoundError(
                f"Go backtester not found: {self.backtester_path}\n"
                f"Build it with: cd etl && go build -o bin/backtester ./cmd/backtester"
            )

    def run_backtest(
        self,
        signals: np.ndarray,
        features_path: str,
        config: Optional[BacktestConfig] = None,
        timestamps: Optional[np.ndarray] = None,
        sizes: Optional[np.ndarray] = None,
        stop_prices: Optional[np.ndarray] = None,
        include_equity: bool = False,
    ) -> BacktestResult:
        """
        Run Go backtester with given signals and features.

        Args:
            signals: (N,) int8 array with values in {-1, 0, 1}
            features_path: Path to features parquet file
            config: Backtest configuration
            timestamps: Optional timestamps for signals (uses indices if None)
            sizes: Optional (N,) float64 array with position sizes (1.0 = 100%, 2.0 = 2x leverage)
            stop_prices: Optional (N,) float64 array with per-bar stop prices (for custom_stop mode)
            include_equity: Whether to include equity curve in result

        Returns:
            BacktestResult with metrics
        """
        self._check_binary()

        if config is None:
            config = BacktestConfig()

        # Validate signals
        signals = np.asarray(signals, dtype=np.int8)
        if not np.all(np.isin(signals, [-1, 0, 1])):
            raise ValueError("Signals must be in {-1, 0, 1}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # 1. Write signals to parquet (with optional sizes and stop_prices)
            signals_path = tmpdir / "signals.parquet"
            self._write_signals(signals, signals_path, timestamps, sizes, stop_prices)

            # 2. Write config to JSON
            config_path = tmpdir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config.to_dict(), f)

            # 3. Prepare output path
            output_path = tmpdir / "result.json"

            # 4. Build command
            cmd = [
                str(self.backtester_path),
                "-signals", str(signals_path),
                "-features", str(features_path),
                "-config", str(config_path),
                "-output", str(output_path),
                "-quiet",
            ]
            if include_equity:
                cmd.append("-equity")

            # 5. Run Go backtester
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"Go backtester failed (code {result.returncode}):\n"
                    f"stderr: {result.stderr}\n"
                    f"stdout: {result.stdout}"
                )

            # 6. Parse results
            with open(output_path) as f:
                data = json.load(f)

            return BacktestResult.from_dict(data)

    def run_backtest_with_data(
        self,
        signals: np.ndarray,
        features: np.ndarray,
        config: Optional[BacktestConfig] = None,
        timestamps: Optional[np.ndarray] = None,
        sizes: Optional[np.ndarray] = None,
        stop_prices: Optional[np.ndarray] = None,
        include_equity: bool = False,
    ) -> BacktestResult:
        """
        Run backtest with in-memory features data.

        Args:
            signals: (N,) int8 array with values in {-1, 0, 1}
            features: (N, 9) array with [timestamp, open, high, high_time, low, low_time, close, volume, realized_vol]
            config: Backtest configuration
            timestamps: Optional timestamps for signals
            sizes: Optional (N,) float64 array with position sizes (1.0 = 100%, 2.0 = 2x leverage)
            stop_prices: Optional (N,) float64 array with per-bar stop prices (for custom_stop mode)
            include_equity: Whether to include equity curve

        Returns:
            BacktestResult with metrics
        """
        self._check_binary()

        if config is None:
            config = BacktestConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write features to parquet
            features_path = tmpdir / "features.parquet"
            self._write_features(features, features_path)

            # Delegate to file-based method
            return self.run_backtest(
                signals=signals,
                features_path=str(features_path),
                config=config,
                timestamps=timestamps,
                sizes=sizes,
                stop_prices=stop_prices,
                include_equity=include_equity,
            )

    def _write_signals(
        self,
        signals: np.ndarray,
        path: Path,
        timestamps: Optional[np.ndarray] = None,
        sizes: Optional[np.ndarray] = None,
        stop_prices: Optional[np.ndarray] = None,
    ) -> None:
        """Write signals to parquet format with optional sizes and stop prices."""
        n = len(signals)

        if timestamps is None:
            timestamps = np.arange(n, dtype=np.int64)
        else:
            timestamps = np.asarray(timestamps, dtype=np.int64)

        # Build table columns
        columns = {
            "timestamp": pa.array(timestamps, type=pa.int64()),
            "signal": pa.array(signals, type=pa.int8()),
        }

        # Add optional size column
        if sizes is not None:
            sizes = np.asarray(sizes, dtype=np.float64)
            if len(sizes) != n:
                raise ValueError(f"sizes length ({len(sizes)}) != signals length ({n})")
            columns["size"] = pa.array(sizes, type=pa.float64())

        # Add optional stop price column
        if stop_prices is not None:
            stop_prices = np.asarray(stop_prices, dtype=np.float64)
            if len(stop_prices) != n:
                raise ValueError(f"stop_prices length ({len(stop_prices)}) != signals length ({n})")
            columns["sl_price"] = pa.array(stop_prices, type=pa.float64())

        table = pa.table(columns)
        pq.write_table(table, path)

    def _write_features(self, features: np.ndarray, path: Path) -> None:
        """
        Write features array to parquet.

        Expected columns: timestamp, open, high, high_time, low, low_time, close, volume, realized_vol
        """
        if features.shape[1] != 9:
            raise ValueError(
                f"Features must have 9 columns, got {features.shape[1]}. "
                "Expected: [timestamp, open, high, high_time, low, low_time, close, volume, realized_vol]"
            )

        table = pa.table({
            "timestamp": pa.array(features[:, 0].astype(np.int64), type=pa.int64()),
            "open": pa.array(features[:, 1], type=pa.float64()),
            "high": pa.array(features[:, 2], type=pa.float64()),
            "high_time": pa.array(features[:, 3].astype(np.int64), type=pa.int64()),
            "low": pa.array(features[:, 4], type=pa.float64()),
            "low_time": pa.array(features[:, 5].astype(np.int64), type=pa.int64()),
            "close": pa.array(features[:, 6], type=pa.float64()),
            "volume": pa.array(features[:, 7], type=pa.float64()),
            "realized_vol": pa.array(features[:, 8], type=pa.float64()),
        })
        pq.write_table(table, path)

    def is_available(self) -> bool:
        """Check if Go backtester binary is available."""
        return self.backtester_path.exists()

    def get_version(self) -> Optional[str]:
        """Get Go backtester version (if available)."""
        if not self.is_available():
            return None

        try:
            result = subprocess.run(
                [str(self.backtester_path), "-version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return "unknown"
