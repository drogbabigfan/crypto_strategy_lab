# Go Backtester Documentation

## Overview

고성능 백테스트 엔진. Python WFA Engine에서 subprocess로 호출되어 시그널과 피처를 받아 백테스트를 수행하고 결과를 JSON으로 반환.

## Architecture

```
Python WFA Engine
    │
    │  subprocess call
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Go Backtester                                              │
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │   Runner    │──▶│  Executor   │──▶│   Metrics   │       │
│  │  (Parquet)  │   │ (Next Bar)  │   │ (Sharpe..)  │       │
│  └─────────────┘   └──────┬──────┘   └─────────────┘       │
│                           │                                 │
│                    ┌──────▼──────┐                         │
│                    │ Cost Model  │                         │
│                    │ (√ Impact)  │                         │
│                    └─────────────┘                         │
└─────────────────────────────────────────────────────────────┘
    │
    │  JSON output
    ▼
Python (parse results)
```

## Package Structure

```
internal/backtest/
├── types.go          # Core types: Signal, Bar, Trade, Config, Result
├── cost_model.go     # Square Root Cost Model
├── executor.go       # Next Bar Entry execution logic
├── metrics.go        # Performance metrics calculation
├── runner.go         # Parquet I/O, orchestration
└── *_test.go         # 51 comprehensive tests
```

## Key Components

### 1. Types (`types.go`)

```go
// Bar with precise High/Low timing for accurate SL/TP ordering
type Bar struct {
    Timestamp   int64
    Open        float64
    High        float64
    HighTime    int64   // When High occurred (ms)
    Low         float64
    LowTime     int64   // When Low occurred (ms)
    Close       float64
    Volume      float64
    RealizedVol float64
}

// Trade result
type Trade struct {
    EntryBar    int
    ExitBar     int
    Direction   Direction  // Long (+1) or Short (-1)
    EntryPrice  float64
    ExitPrice   float64
    PnL         float64    // Net (after costs)
    GrossPnL    float64    // Before costs
    TotalCost   float64
    ExitReason  ExitReason // TP, SL, TIMEOUT
    HoldingBars int
}

// Configuration
type Config struct {
    SLMult         float64 // Stop Loss (σ multiplier)
    PTMult         float64 // Profit Target (σ multiplier)
    MaxHoldBars    int     // Vertical barrier
    BaseFee        float64 // Exchange fee (0.001 = 0.1%)
    BaseSlippage   float64 // Min slippage
    ImpactCoeff    float64 // Square root impact
    SlippageCap    float64 // Max slippage
    InitialCapital float64
    RiskPerTrade   float64
}
```

### 2. Cost Model (`cost_model.go`)

Square Root Law 기반 시장 충격 모델:

```
Impact = η × σ × √(Q / V)

η: Impact coefficient (default 0.1)
σ: Realized volatility
Q: Trade size
V: Average volume
```

**Key Functions:**
- `GetSlippage(volatility, tradeSize, avgVolume)` - 슬리피지 계산
- `GetTotalCost(...)` - Round-trip 비용 (진입 + 청산)
- `ApplyEntrySlippage(price, direction, ...)` - 진입 가격에 슬리피지 적용
- `ApplyExitSlippage(price, direction, ...)` - 청산 가격에 슬리피지 적용
- `CalculatePnL(entry, exit, direction)` - 손익 계산
- `IsViable(ptMult, sigma, ...)` - Fee Trap 체크

### 3. Executor (`executor.go`)

Next Bar Entry 로직:

```
Signal at bar t → Entry at bar t+1 Open
```

**핵심 규칙:**
1. 신호는 Close 확정 후 생성 가정
2. 진입은 다음 바 Open에서만 가능
3. 슬리피지는 Entry/Exit 모두 적용

**SL/TP Ordering (v4.7):**
```go
// 같은 바에서 SL과 TP 모두 터치 시, 실제 시간으로 판정
if slHit && tpHit {
    if bar.LowTime <= bar.HighTime {  // Long position
        return StopLoss
    }
    return TakeProfit
}
```

### 4. Metrics (`metrics.go`)

계산되는 지표:
- `TotalTrades` - 총 거래 수
- `WinRate` - 승률
- `AvgPnL` - 평균 손익
- `TotalPnL` - 누적 손익
- `SharpeRatio` - 샤프 비율 (연간화)
- `MaxDrawdown` - 최대 낙폭
- `ProfitFactor` - 이익/손실 비율
- `AvgHoldBars` - 평균 보유 기간
- `TPCount`, `SLCount`, `TimeoutCount` - 청산 사유별 카운트
- `EquityCurve` - 자산 곡선

**추가 함수:**
- `Sortino(pnls)` - Sortino ratio
- `Calmar(return, maxDD)` - Calmar ratio
- `ConsecutiveLosses(trades)` - 연속 손실
- `ExpectancyPerBar(trades)` - 바당 기대값

### 5. Runner (`runner.go`)

Parquet I/O 및 백테스트 오케스트레이션:

```go
// 파일 기반 실행
func (r *Runner) Run(signalsPath, featuresPath string) (Result, error)

// 인메모리 데이터 실행
func (r *Runner) RunWithData(bars []Bar, signals []Signal) Result
```

**Parquet 스키마:**

Signals:
| Field | Type | Description |
|-------|------|-------------|
| timestamp | int64 | Bar timestamp |
| signal | int8 | -1, 0, +1 |

Features:
| Field | Type | Description |
|-------|------|-------------|
| timestamp | int64 | Bar timestamp |
| open | float64 | Open price |
| high | float64 | High price |
| high_time | int64 | When high occurred |
| low | float64 | Low price |
| low_time | int64 | When low occurred |
| close | float64 | Close price |
| volume | float64 | Volume |
| realized_vol | float64 | Realized volatility |

## CLI Usage

### Standalone Binary

```bash
./bin/backtester \
    -signals signals.parquet \
    -features features.parquet \
    -sl 2.0 \
    -pt 2.5 \
    -max-hold 100 \
    -output result.json \
    -equity \
    -quiet
```

### Via ETL CLI

```bash
./bin/etl backtest \
    -signals signals.parquet \
    -features features.parquet \
    -config config.json \
    -output result.json
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-signals` | Signals parquet path | Required |
| `-features` | Features parquet path | Required |
| `-config` | Config JSON path | Use defaults |
| `-output` | Output JSON path | stdout |
| `-sl` | Stop loss multiplier | 2.0 |
| `-pt` | Profit target multiplier | 2.5 |
| `-max-hold` | Max holding bars | 100 |
| `-equity` | Include equity curve | false |
| `-quiet` | Suppress summary | false |

## JSON Output

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

## Config JSON

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

## Testing

```bash
# Run all tests
go test ./internal/backtest/... -v

# Run specific test
go test ./internal/backtest/... -run TestSLHitBeforeTP -v

# Benchmark
go test ./internal/backtest/... -bench=.
```

**Test Coverage:** 51 tests
- Cost Model: 10 tests
- Executor: 15 tests
- Metrics: 14 tests
- Runner: 12 tests

## Performance

| Dataset Size | Time |
|--------------|------|
| 1,000 bars | ~0.1ms |
| 10,000 bars | ~1ms |
| 100,000 bars | ~10ms |
| 1,000,000 bars | ~100ms |

## Python Integration

```python
from research.wfa.go_bridge import GoBridge

bridge = GoBridge("./bin/backtester")

result = bridge.run_backtest(
    signals=signals_array,      # np.ndarray int8
    features_path="features.parquet",
    config={
        "sl_mult": 2.0,
        "pt_mult": 2.5,
        "max_hold_bars": 100,
        # ...
    }
)

print(f"Sharpe: {result['sharpe_ratio']:.2f}")
```

## Design Decisions

### 1. Why Go?
- 100x faster than Python for loops
- No GIL, true parallelism
- Easy to build single binary
- Parquet support via `segmentio/parquet-go`

### 2. Why HighTime/LowTime?
- Dollar bars can have both SL and TP hit in same bar
- Need precise ordering based on actual occurrence time
- Prevents systematic bias towards SL or TP

### 3. Why Next Bar Entry?
- Prevents look-ahead bias
- Signal generated using Close[t]
- Entry only possible at Open[t+1]
- More realistic than same-bar entry

### 4. Why Square Root Impact?
- Academic standard (Kyle, Almgren-Chriss)
- `Impact ∝ √(Volume)` is empirically validated
- Penalizes large trades appropriately

## Changelog

### v1.0.0 (2026-01)
- Initial implementation
- Next Bar Entry logic
- Square Root Cost Model
- HighTime/LowTime for precise SL/TP
- Parquet I/O
- 51 comprehensive tests
