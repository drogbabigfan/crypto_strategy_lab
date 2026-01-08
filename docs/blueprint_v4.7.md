# Deep Learning Bitcoin Trading Strategy Blueprint v4.7

## 변경 이력

| 버전 | 날짜 | 주요 변경 |
|------|------|----------|
| v1 | - | 초기 설계 (TIB 기반) |
| v2 | - | 실제 구현 반영 (Adaptive Dollar Bar 기반) |
| v3 | - | ETL + Feature Engineering 완료 (Go 통합 파이프라인) |
| v4 | - | SSL Pre-training + Dynamic Label Optimization 도입 |
| v4.1 | - | Fee Trap, Dirty Zero, Overlap, Dynamic Slippage 수정 |
| v4.2 | - | Profitability Proxy, Nonlinear Cost, Execution Lag 추가 |
| v4.3 | - | Multi-Asset Support 추가 |
| v4.5 | - | 복잡도 제거, Plateau Search, Fail-Fast, One-off SSL |
| v4.6 | 2026-01 | Re-structured - 데이터 의존성 기반 Stage 재정렬 |
| **v4.7** | 2026-01 | **Hybrid Architecture** - Go Backtester + Python WFA Engine |

---

## 1. Goal Description

### 1.1 목표
**Crypto Futures Mid-Frequency Trading Strategy 개발**

- **전략**: Trend Following (Long & Short)
- **핵심 제약**: Avg Profit per Trade > 0.15% (수수료/슬리피지 후)
- **설계 철학**: Modular, Config-Driven, File-Centric, TDD, Fail-Fast

### 1.2 대상 자산

| 자산 | 용도 | 우선순위 |
|------|------|----------|
| BTCUSDT | SSL + Fine-tuning + Trading | Primary |
| ETHUSDT | SSL Pre-training | Foundation |
| SOLUSDT | SSL Pre-training | Foundation |
| XRPUSDT | SSL Pre-training | Foundation |
| DOGEUSDT | SSL Pre-training | Foundation |
| BNBUSDT | SSL Pre-training | Foundation |
| LTCUSDT | SSL Pre-training | Foundation |

### 1.3 v4.7 핵심 원칙

```
1. One-off SSL: WFA 진입 전 1회만 Foundation Model 학습
2. Plateau Search: Grid Search에서 Peak 금지, 안정적 영역 중심 선택
3. Fail-Fast: 데이터 부족/학습 실패 시 즉시 중단, Fallback 없음
4. Next Bar Entry: 신호(t) → 진입(t+1 Open), 미래 정보 사용 금지
5. Sample-level Drop: Context Window 내부가 아닌 샘플 단위로 제외
6. Hybrid Architecture: Python Master (WFA) + Go Slave (Backtester)
```

---

## 2. Architecture Overview

### 2.1 Hybrid Python/Go Architecture (v4.7 신규)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          v4.7 Hybrid Architecture                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  Python Master (WFA Engine)                                                │ │
│  │                                                                            │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │ │
│  │  │  SSL Train   │  │ Label Optim  │  │  Fine-tune   │  │   Signal     │   │ │
│  │  │  (Stage 2)   │  │  (Step 3.1)  │  │  (Step 3.2)  │  │  Generator   │   │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────┬───────┘   │ │
│  │                                                                │           │ │
│  │  ┌─────────────────────────────────────────────────────────────▼─────────┐ │ │
│  │  │  WFA Orchestrator                                                     │ │ │
│  │  │  - Window splitting                                                   │ │ │
│  │  │  - Optuna HPO                                                         │ │ │
│  │  │  - Results aggregation                                                │ │ │
│  │  └─────────────────────────────────────────────┬─────────────────────────┘ │ │
│  └────────────────────────────────────────────────│─────────────────────────────┘ │
│                                                   │ subprocess                    │
│                                                   ▼                               │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  Go Slave (Backtester)                                                     │ │
│  │                                                                            │ │
│  │  signals.parquet ──▶ ┌─────────────────┐ ──▶ result.json                  │ │
│  │  features.parquet    │  Go Backtester  │     {sharpe, pnl, trades...}     │ │
│  │  config.json    ──▶ │  (Fast, Exact)  │                                   │ │
│  │                      └─────────────────┘                                   │ │
│  │                                                                            │ │
│  │  Key Features:                                                             │ │
│  │  - Next Bar Entry (t+1 Open)                                              │ │
│  │  - Square Root Cost Model                                                  │ │
│  │  - HighTime/LowTime for precise SL/TP ordering                            │ │
│  │  - Parquet I/O                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 왜 Hybrid인가?

| 관점 | Python | Go | 선택 |
|------|--------|-----|------|
| 학습 (Training) | PyTorch 필수 | 불가 | Python |
| 최적화 (Optuna) | Native | Binding 필요 | Python |
| Backtest 속도 | ~1000 bars/s | ~100K bars/s | Go |
| WFA 루프 | 90% 시간 = 학습 | 10% = Backtest | Python Master |

**결론**: Python이 전체 파이프라인 제어, Go는 Backtest만 담당

---

## 3. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     v4.7 Pipeline Architecture                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  STAGE 1: ETL & Feature Engineering (Go) ✅ 완료                           │ │
│  │                                                                            │ │
│  │  Raw Trades (ZIP) ──▶ Dollar Bars ──▶ Feature Parquet (30 features)       │ │
│  │  (Multi-Asset)         Streaming        data/features/futures/            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                        │                                         │
│                                        ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  STAGE 2: SSL Pre-training (Python) ✅ 완료 (25 tests)                     │ │
│  │                                                                            │ │
│  │  Multi-Asset Features ──▶ Patch Masking (40%) ──▶ Foundation Encoder      │ │
│  │  (BTC,ETH,SOL,XRP,         Reconstruction          artifacts/ssl/         │ │
│  │   DOGE,BNB,LTC)            Loss                    foundation_encoder.pt  │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                        │                                         │
│                                        ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  STAGE 3: Walk-Forward Analysis Loop                                       │ │
│  │  ┌──────────────────────────────────────────────────────────────────────┐  │ │
│  │  │  For each (Train Window, Test Window):                               │  │ │
│  │  │                                                                      │  │ │
│  │  │  [STEP 3.1]         [STEP 3.2]          [STEP 3.3]                  │  │ │
│  │  │  Label          ──▶  Fine-tuning   ──▶   Signal Gen                 │  │ │
│  │  │  Optimization        (Python)            + Backtest                  │  │ │
│  │  │  (Python) ✅         ✅ 완료              (Python→Go)                │  │ │
│  │  │  48 tests            37 tests                                        │  │ │
│  │  │                                                                      │  │ │
│  │  │  Grid Search    ──▶  Load Encoder  ──▶  Generate Signals (Python)   │  │ │
│  │  │  Plateau Search      + Classifier       Call Go Backtester           │  │ │
│  │  │  Best Params         Focal Loss         Parse JSON Results           │  │ │
│  │  └──────────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 단계별 요약

| Stage | 이름 | 언어 | 입력 | 출력 | 상태 |
|-------|------|------|------|------|------|
| 1 | ETL & Features | Go | Raw Trades | Feature Parquet | ✅ 완료 |
| 2 | SSL Pre-training | Python | Multi-Asset Features | Foundation Encoder | ✅ 완료 (25 tests) |
| 3 | WFA Loop | Python | Encoder + Features | Trades + Metrics | 🔄 진행중 |
| 3.1 | └ Label Optimizer | Python | Train Features | Best Params + Labels | ✅ 완료 (48 tests) |
| 3.2 | └ Fine-tuning | Python | Encoder + Labels | Classifier Model | ✅ 완료 (37 tests) |
| 3.3 | └ Backtester | **Go** | Signals + Features | Metrics JSON | ✅ 완료 (51 tests) |
| 3.3 | └ WFA Engine | Python | - | Orchestration | ⬜ 다음 |

---

## 4. Go Backtester (Step 3.3) - v4.7 신규

### 4.1 개요

```
Purpose: 빠르고 정확한 Backtest 실행
Input:   signals.parquet, features.parquet, config.json
Output:  result.json (metrics + trades)
Speed:   ~100K bars/second
```

### 4.2 패키지 구조

```
etl/internal/backtest/
├── types.go        # Signal, Direction, Position, Trade, Config, Result
├── cost_model.go   # Square Root Cost Model (slippage, fees)
├── executor.go     # Next Bar Entry logic (SL/TP/Timeout)
├── metrics.go      # Sharpe, MaxDD, ProfitFactor, Sortino, Calmar
├── runner.go       # Parquet I/O, backtest orchestration
├── cost_model_test.go
├── executor_test.go
├── metrics_test.go
└── runner_test.go
Total: 51 tests
```

### 4.3 핵심 구조체

```go
// Bar with precise High/Low timing
type Bar struct {
    Timestamp   int64   // Bar timestamp (ms)
    Open        float64
    High        float64
    HighTime    int64   // When High occurred - for precise SL/TP ordering
    Low         float64
    LowTime     int64   // When Low occurred - for precise SL/TP ordering
    Close       float64
    Volume      float64
    RealizedVol float64 // For barrier calculation
}

// Trade result
type Trade struct {
    EntryBar    int
    ExitBar     int
    Direction   Direction  // Long (+1) or Short (-1)
    EntryPrice  float64
    ExitPrice   float64
    PnL         float64    // Net P&L (after costs)
    GrossPnL    float64
    TotalCost   float64
    ExitReason  ExitReason // TP, SL, TIMEOUT
    HoldingBars int
}

// Backtest configuration
type Config struct {
    SLMult         float64 // Stop Loss σ multiplier
    PTMult         float64 // Profit Target σ multiplier
    MaxHoldBars    int     // Vertical barrier

    BaseFee        float64 // Exchange fee (0.001 = 0.1%)
    BaseSlippage   float64 // Min slippage
    ImpactCoeff    float64 // Square root impact
    SlippageCap    float64 // Max slippage

    InitialCapital float64
    RiskPerTrade   float64
}
```

### 4.4 Square Root Cost Model

```go
// Impact = η × σ × √(Q / V)
func (c *CostModel) GetSlippage(volatility, tradeSize, avgVolume float64) float64 {
    if avgVolume <= 0 {
        return c.SlippageCap
    }

    volumeRatio := tradeSize / avgVolume
    impact := c.ImpactCoeff * volatility * math.Sqrt(volumeRatio)

    totalSlippage := c.BaseSlippage + impact
    return min(totalSlippage, c.SlippageCap)
}

// Round-trip cost (entry + exit)
func (c *CostModel) GetTotalCost(volatility, tradeSize, avgVolume float64) float64 {
    slippage := c.GetSlippage(volatility, tradeSize, avgVolume)
    return 2 * (c.BaseFee + slippage)
}
```

### 4.5 Time-based SL/TP Ordering (v4.7 핵심)

```go
// determineLongExit uses HighTime/LowTime for precise ordering
func (e *Executor) determineLongExit(bar Bar, pos *Position, holdingBars int) (float64, ExitReason) {
    slHit := bar.Low <= pos.StopLoss
    tpHit := bar.High >= pos.TakeProfit

    if slHit && tpHit {
        // Both hit in same bar - use actual timestamps
        if bar.LowTime <= bar.HighTime {
            return pos.StopLoss, ExitReasonSL
        }
        return pos.TakeProfit, ExitReasonTP
    }

    if slHit {
        return pos.StopLoss, ExitReasonSL
    }
    if tpHit {
        return pos.TakeProfit, ExitReasonTP
    }
    if holdingBars >= pos.MaxHoldBars {
        return bar.Close, ExitReasonTimeout
    }

    return 0, ""
}
```

### 4.6 CLI 사용법

```bash
# Standalone binary
./bin/backtester \
    -signals signals.parquet \
    -features features.parquet \
    -sl 2.0 -pt 2.5 -max-hold 100 \
    -output result.json

# Via etl CLI
./bin/etl backtest \
    -signals signals.parquet \
    -features features.parquet \
    -config config.json \
    -output result.json \
    -equity  # Include equity curve
```

### 4.7 JSON Output Format

```json
{
  "total_trades": 150,
  "win_rate": 0.52,
  "avg_pnl": 0.0018,
  "total_pnl": 0.27,
  "sharpe_ratio": 1.85,
  "max_drawdown": 0.12,
  "profit_factor": 1.65,
  "avg_hold_bars": 45.3,
  "tp_count": 65,
  "sl_count": 55,
  "timeout_count": 30,
  "equity_curve": [100000, 100180, ...]
}
```

---

## 5. Python WFA Engine (Step 3.3) - 다음 구현

### 5.1 개요

```python
class WFAEngine:
    """
    Walk-Forward Analysis Engine.

    Python controls the entire WFA loop:
    1. Window splitting
    2. Label optimization (Step 3.1)
    3. Fine-tuning (Step 3.2)
    4. Signal generation
    5. Call Go backtester
    6. Aggregate results
    """
```

### 5.2 Go Bridge

```python
# research/wfa/go_bridge.py

import subprocess
import json
import tempfile
from pathlib import Path

class GoBridge:
    """Python → Go Backtester interface."""

    def __init__(self, backtester_path: str = "./bin/backtester"):
        self.backtester_path = Path(backtester_path)
        if not self.backtester_path.exists():
            raise FileNotFoundError(f"Go backtester not found: {backtester_path}")

    def run_backtest(
        self,
        signals: np.ndarray,      # (N,) int8 array [-1, 0, 1]
        features_path: str,       # Path to features parquet
        config: dict,             # Backtest config
        include_equity: bool = False
    ) -> dict:
        """
        Run Go backtester and return results.

        Returns:
            dict with metrics: total_trades, win_rate, sharpe_ratio, etc.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Write signals to parquet
            signals_path = Path(tmpdir) / "signals.parquet"
            self._write_signals(signals, signals_path)

            # 2. Write config to JSON
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f)

            # 3. Run Go backtester
            output_path = Path(tmpdir) / "result.json"
            cmd = [
                str(self.backtester_path),
                "-signals", str(signals_path),
                "-features", features_path,
                "-config", str(config_path),
                "-output", str(output_path),
                "-quiet"
            ]
            if include_equity:
                cmd.append("-equity")

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Backtester failed: {result.stderr}")

            # 4. Parse results
            with open(output_path) as f:
                return json.load(f)

    def _write_signals(self, signals: np.ndarray, path: Path):
        """Write signals to parquet format."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({
            'timestamp': pa.array(range(len(signals)), type=pa.int64()),
            'signal': pa.array(signals, type=pa.int8())
        })
        pq.write_table(table, path)
```

### 5.3 Signal Generator

```python
# research/wfa/signal_generator.py

class SignalGenerator:
    """Generate trading signals from trained classifier."""

    def __init__(self, model: nn.Module, context_len: int = 512):
        self.model = model
        self.context_len = context_len
        self.model.eval()

    @torch.no_grad()
    def generate(self, features: np.ndarray) -> np.ndarray:
        """
        Generate signals for all bars.

        Args:
            features: (N, n_features) array

        Returns:
            signals: (N,) int8 array with values in {-1, 0, 1}
        """
        signals = np.zeros(len(features), dtype=np.int8)

        for i in range(self.context_len, len(features)):
            window = features[i - self.context_len:i]  # (512, 28)
            x = torch.tensor(window.T, dtype=torch.float32).unsqueeze(0)

            logits = self.model(x)
            pred = torch.argmax(logits, dim=-1).item()

            # {0, 1, 2} → {-1, 0, +1}
            signals[i] = pred - 1

        return signals
```

### 5.4 WFA Main Loop

```python
# research/wfa/engine.py

class WFAEngine:
    """Walk-Forward Analysis orchestrator."""

    def __init__(
        self,
        config: WFAConfig,
        encoder_path: str,
        go_bridge: GoBridge
    ):
        self.config = config
        self.encoder_path = encoder_path
        self.go_bridge = go_bridge
        self.results = []

    def run(self, features_dir: str) -> list[dict]:
        """Execute full WFA loop."""

        # Load all features
        features = self._load_features(features_dir)

        # Generate folds
        folds = self._generate_folds(features)

        for fold_id, fold in enumerate(folds):
            print(f"\n{'='*60}")
            print(f"Fold {fold_id}: Train [{fold.train_start} ~ {fold.train_end}]")
            print(f"           Test  [{fold.test_start} ~ {fold.test_end}]")

            # Step 3.1: Label Optimization
            opt_result = self._optimize_labels(fold.train_data)

            # Step 3.2: Fine-tuning
            model = self._finetune(fold.train_data, opt_result)

            # Step 3.3a: Generate Signals
            signals = SignalGenerator(model).generate(fold.test_features)

            # Step 3.3b: Run Go Backtester
            backtest_config = {
                "sl_mult": opt_result.best_params["sl"],
                "pt_mult": opt_result.best_params["pt"],
                "max_hold_bars": opt_result.best_params["time"],
                "base_fee": 0.001,
                "impact_coeff": 0.1,
                # ... other config
            }

            metrics = self.go_bridge.run_backtest(
                signals=signals,
                features_path=fold.test_features_path,
                config=backtest_config
            )

            self.results.append({
                "fold_id": fold_id,
                "train_period": (fold.train_start, fold.train_end),
                "test_period": (fold.test_start, fold.test_end),
                "best_params": opt_result.best_params,
                "metrics": metrics
            })

            print(f"Results: Sharpe={metrics['sharpe_ratio']:.2f}, "
                  f"WinRate={metrics['win_rate']:.1%}")

        return self.results
```

---

## 6. Implementation Status

### 완료된 항목

| Stage | Component | Language | Tests | Status |
|-------|-----------|----------|-------|--------|
| 1 | ETL & Dollar Bars | Go | 100+ | ✅ |
| 1 | Feature Engineering | Go | 50+ | ✅ |
| 2 | SSL Dataset | Python | 5 | ✅ |
| 2 | PatchTST Encoder | Python | 8 | ✅ |
| 2 | SSL Training | Python | 12 | ✅ |
| 3.1 | Cost Model | Python | 12 | ✅ |
| 3.1 | Triple Barrier | Python | 15 | ✅ |
| 3.1 | Scoring (Entropy, MI, RankIC) | Python | 10 | ✅ |
| 3.1 | Plateau Search | Python | 6 | ✅ |
| 3.1 | Label Optimizer | Python | 5 | ✅ |
| 3.2 | Fine-tuning Dataset | Python | 8 | ✅ |
| 3.2 | Classifier Model | Python | 10 | ✅ |
| 3.2 | Focal Loss | Python | 5 | ✅ |
| 3.2 | Trainer | Python | 14 | ✅ |
| **3.3** | **Go Backtester** | **Go** | **51** | ✅ |

**Total Tests: 256+**

### 다음 구현

| Component | Language | Description |
|-----------|----------|-------------|
| Signal Generator | Python | Model → Signals |
| Go Bridge | Python | subprocess wrapper |
| WFA Engine | Python | Main orchestrator |
| Integration Tests | Python | End-to-end |

---

## 7. Directory Structure (Updated)

```
/home/kimhoyeon/dev/dl_rl_btc/
├── config/
│   └── config_v47.yaml
│
├── data/
│   ├── features/futures/
│   │   ├── BTCUSDT/
│   │   ├── ETHUSDT/
│   │   └── ...
│   └── state/futures/
│
├── etl/                              # Go
│   ├── cmd/
│   │   ├── main.go                   # etl CLI (bars, features, backtest)
│   │   └── backtester/main.go        # Standalone backtester
│   ├── bin/
│   │   ├── etl                       # Main CLI
│   │   └── backtester                # Backtester binary
│   └── internal/
│       ├── bars/                     # Dollar Bar generation
│       ├── features/                 # Feature engineering
│       ├── backtest/                 # ✅ NEW: Backtester package
│       │   ├── types.go
│       │   ├── cost_model.go
│       │   ├── executor.go
│       │   ├── metrics.go
│       │   ├── runner.go
│       │   └── *_test.go
│       └── storage/                  # Parquet I/O
│
├── research/                         # Python
│   ├── ssl/                          # Stage 2 ✅
│   ├── label_optimizer/              # Step 3.1 ✅
│   ├── trainer/                      # Step 3.2 ✅
│   └── wfa/                          # Step 3.3 (다음)
│       ├── __init__.py
│       ├── go_bridge.py              # Python → Go interface
│       ├── signal_generator.py       # Model → Signals
│       └── engine.py                 # WFA orchestrator
│
├── artifacts/
│   ├── ssl/foundation_encoder.pt
│   └── classifier/
│
├── tests/
│   ├── test_ssl.py                   # 25 tests ✅
│   ├── test_label_optimizer.py       # 48 tests ✅
│   ├── test_trainer.py               # 37 tests ✅
│   └── test_wfa.py                   # (다음)
│
└── docs/
    ├── blueprint_v4.6.md
    ├── blueprint_v4.7.md             # ✅ 현재 버전
    └── go_backtester.md              # Go Backtester 상세
```

---

## 8. Success Metrics

| 지표 | 기준 | 의미 |
|------|------|------|
| Avg PnL/Trade | > 0.15% | 순수익 (비용 후) |
| Win Rate | > 45% | 승률 |
| Sharpe Ratio | > 1.5 | 위험 조정 수익 |
| Max Drawdown | < 20% | 최대 손실폭 |
| Profit Factor | > 1.5 | 총이익/총손실 |

---

## Appendix A: Go Backtester Quick Reference

### A.1 Build

```bash
cd etl
go build -o bin/etl ./cmd/main.go
go build -o bin/backtester ./cmd/backtester
```

### A.2 Test

```bash
go test ./internal/backtest/... -v
```

### A.3 CLI Usage

```bash
# Full options
./bin/backtester \
    -signals signals.parquet \
    -features features.parquet \
    -config config.json \
    -output result.json \
    -sl 2.0 \
    -pt 2.5 \
    -max-hold 100 \
    -equity \
    -quiet
```

### A.4 Config JSON Format

```json
{
    "sl_mult": 2.0,
    "pt_mult": 2.5,
    "max_hold_bars": 100,
    "base_fee": 0.001,
    "base_slippage": 0.0001,
    "impact_coeff": 0.1,
    "slippage_cap": 0.005,
    "initial_capital": 100000,
    "risk_per_trade": 0.02
}
```

---

**Document Version**: v4.7 (Hybrid Architecture)
**Last Updated**: 2026-01
**Status**: Go Backtester Complete, WFA Engine Next
