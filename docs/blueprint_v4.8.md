# Deep Learning Bitcoin Trading Strategy Blueprint v4.8

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
| v4.7 | 2026-01 | Hybrid Architecture - Go Backtester + Python WFA Engine |
| **v4.8** | 2026-01 | **Core Implementation Complete** - 모든 핵심 모듈 구현 완료 |

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

### 1.3 v4.8 핵심 원칙

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

### 2.1 Hybrid Python/Go Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          v4.8 Hybrid Architecture                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  Python Master (WFA Engine) ✅ 완료                                        │ │
│  │                                                                            │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │ │
│  │  │  SSL Train   │  │ Label Optim  │  │  Fine-tune   │  │   Signal     │   │ │
│  │  │  (Stage 2)   │  │  (Step 3.1)  │  │  (Step 3.2)  │  │  Generator   │   │ │
│  │  │  ✅ 25 tests │  │  ✅ 48 tests │  │  ✅ 37 tests │  │  ✅ 36 tests │   │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────┬───────┘   │ │
│  │                                                                │           │ │
│  │  ┌─────────────────────────────────────────────────────────────▼─────────┐ │ │
│  │  │  WFA Orchestrator ✅                                                  │ │ │
│  │  │  - Window splitting (generate_folds)                                  │ │ │
│  │  │  - Per-fold processing (_process_fold)                                │ │ │
│  │  │  - Results aggregation (_compute_summary)                             │ │ │
│  │  └─────────────────────────────────────────────┬─────────────────────────┘ │ │
│  └────────────────────────────────────────────────│─────────────────────────────┘ │
│                                                   │ subprocess                    │
│                                                   ▼                               │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  Go Slave (Backtester) ✅ 51 tests                                        │ │
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
│                     v4.8 Pipeline Architecture                                   │
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
│  │  STAGE 3: Walk-Forward Analysis Loop ✅ 완료                               │ │
│  │  ┌──────────────────────────────────────────────────────────────────────┐  │ │
│  │  │  For each (Train Window, Test Window):                               │  │ │
│  │  │                                                                      │  │ │
│  │  │  [STEP 3.1]         [STEP 3.2]          [STEP 3.3]                  │  │ │
│  │  │  Label          ──▶  Fine-tuning   ──▶   Signal Gen                 │  │ │
│  │  │  Optimization        (Python)            + Backtest                  │  │ │
│  │  │  (Python) ✅         ✅ 37 tests         (Python→Go)                 │  │ │
│  │  │  48 tests                                ✅ 36+51 tests              │  │ │
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
| 3 | WFA Loop | Python | Encoder + Features | Trades + Metrics | ✅ 완료 |
| 3.1 | └ Label Optimizer | Python | Train Features | Best Params + Labels | ✅ 완료 (48 tests) |
| 3.2 | └ Fine-tuning | Python | Encoder + Labels | Classifier Model | ✅ 완료 (37 tests) |
| 3.3 | └ Backtester | **Go** | Signals + Features | Metrics JSON | ✅ 완료 (51 tests) |
| 3.3 | └ WFA Engine | Python | - | Orchestration | ✅ 완료 (36 tests) |

---

## 4. Go Backtester (Step 3.3)

### 4.1 개요

```
Purpose: 빠르고 정확한 Backtest 실행
Input:   signals.parquet, features.parquet, config.json
Output:  result.json (metrics + trades)
Speed:   ~100K bars/second
Tests:   51
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
    volumeRatio := tradeSize / avgVolume
    impact := c.ImpactCoeff * volatility * math.Sqrt(volumeRatio)
    totalSlippage := c.BaseSlippage + impact
    return min(totalSlippage, c.SlippageCap)
}
```

### 4.5 Time-based SL/TP Ordering

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
    // ...
}
```

---

## 5. Python WFA Engine (Step 3.3) ✅ 완료

### 5.1 패키지 구조

```
research/wfa/
├── __init__.py           # Public API exports
├── go_bridge.py          # Python → Go subprocess interface
├── signal_generator.py   # Model → Trading signals
└── engine.py             # WFA orchestrator
Total: 36 tests
```

### 5.2 GoBridge - Python → Go Interface

```python
@dataclass
class BacktestConfig:
    sl_mult: float = 2.0
    pt_mult: float = 2.5
    max_hold_bars: int = 100
    base_fee: float = 0.001
    # ...

    def to_dict(self) -> dict:
        # Go expects PascalCase keys
        return {
            "SLMult": self.sl_mult,
            "PTMult": self.pt_mult,
            # ...
        }

class GoBridge:
    def run_backtest(
        self,
        signals: np.ndarray,      # (N,) int8 array [-1, 0, 1]
        features_path: str,
        config: BacktestConfig,
    ) -> BacktestResult:
        # 1. Write signals to temp parquet
        # 2. Write config to JSON
        # 3. subprocess.run(backtester, ...)
        # 4. Parse result.json
        ...
```

### 5.3 SignalGenerator

```python
class SignalGenerator:
    """Generate trading signals from trained classifier."""

    def __init__(self, model: nn.Module, config: SignalGeneratorConfig):
        self.model = model
        self.config = config
        self.model.eval()

    @torch.no_grad()
    def generate(self, features: np.ndarray) -> np.ndarray:
        """
        Generate signals for all bars.

        Args:
            features: (N, n_features) array

        Returns:
            signals: (N,) int8 array with values in {-1, 0, 1}
                     -1 = Short, 0 = Neutral, 1 = Long
        """
        # Batch processing for efficiency
        # Context window sliding
        # Softmax → argmax → {0,1,2} → {-1,0,+1}
```

### 5.4 WFAEngine

```python
class WFAEngine:
    """Walk-Forward Analysis orchestrator."""

    def run(
        self,
        features_df: pd.DataFrame,
        label_optimizer: Callable,
        trainer: Callable,
        feature_cols: list[str],
    ) -> list[FoldResult]:
        """
        Execute full WFA loop.

        For each fold:
        1. Split data (train/test)
        2. Step 3.1: Label Optimization
        3. Step 3.2: Fine-tuning
        4. Step 3.3: Signal Generation + Go Backtest
        5. Aggregate results
        """
        folds = self.generate_folds(features_df)

        for fold in folds:
            result = self._process_fold(fold, ...)
            self.results.append(result)

        self._save_results()
        return self.results
```

---

## 6. Implementation Status

### 완료된 항목 ✅

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
| 3.3 | Go Backtester | Go | 51 | ✅ |
| **3.3** | **Signal Generator** | **Python** | **8** | ✅ |
| **3.3** | **Go Bridge** | **Python** | **13** | ✅ |
| **3.3** | **WFA Engine** | **Python** | **15** | ✅ |

**Total Tests: ~350**

### 다음 단계

| Component | Description | Priority |
|-----------|-------------|----------|
| Integration Test | 실제 데이터로 End-to-end 검증 | High |
| run_wfa.py | 전체 WFA 실행 스크립트 | High |
| Hyperparameter Tuning | 실제 데이터로 최적 파라미터 탐색 | Medium |

---

## 7. Directory Structure

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
│       ├── backtest/                 # ✅ Backtester package (51 tests)
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
│   └── wfa/                          # Step 3.3 ✅
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
│   └── test_wfa.py                   # 36 tests ✅
│
└── docs/
    ├── blueprint_v4.7.md
    ├── blueprint_v4.8.md             # ✅ 현재 버전
    └── GO_BACKTESTER.md              # Go Backtester 상세
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

## Appendix A: Quick Reference

### A.1 Run Tests

```bash
# Python tests
python -m pytest tests/test_ssl.py -v           # 25 tests
python -m pytest tests/test_label_optimizer.py -v  # 48 tests
python -m pytest tests/test_trainer.py -v       # 37 tests
python -m pytest tests/test_wfa.py -v           # 36 tests

# Go tests
cd etl && go test ./internal/backtest/... -v    # 51 tests
```

### A.2 Build Go Backtester

```bash
cd etl
go build -o bin/etl ./cmd/main.go
go build -o bin/backtester ./cmd/backtester
```

### A.3 Go Backtester CLI

```bash
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

### A.4 Config JSON Format (Go expects PascalCase)

```json
{
    "SLMult": 2.0,
    "PTMult": 2.5,
    "MaxHoldBars": 100,
    "BaseFee": 0.001,
    "BaseSlippage": 0.0001,
    "ImpactCoeff": 0.1,
    "SlippageCap": 0.005,
    "InitialCapital": 100000,
    "RiskPerTrade": 0.02
}
```

---

**Document Version**: v4.8 (Core Implementation Complete)
**Last Updated**: 2026-01
**Status**: All core modules complete, ready for integration testing
