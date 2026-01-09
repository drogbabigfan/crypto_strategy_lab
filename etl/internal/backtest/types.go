// Package backtest provides realistic backtesting with Next Bar Entry logic.
// Designed as a pure function: signals in → metrics out.
package backtest

import "fmt"

// Signal represents a trading signal at a specific bar.
type Signal int8

const (
	SignalShort   Signal = -1
	SignalNeutral Signal = 0
	SignalLong    Signal = 1
)

// Direction represents the position direction.
type Direction int8

const (
	DirectionShort Direction = -1
	DirectionFlat  Direction = 0
	DirectionLong  Direction = 1
)

// ExitReason indicates why a trade was closed.
type ExitReason string

const (
	ExitReasonTP      ExitReason = "TP"      // Take Profit hit
	ExitReasonSL      ExitReason = "SL"      // Stop Loss hit
	ExitReasonTimeout ExitReason = "TIMEOUT" // Vertical barrier (max hold)
	ExitReasonSignal  ExitReason = "SIGNAL"  // Opposite signal
)

// ExitMode determines how positions are closed.
type ExitMode string

const (
	// ExitModeTBM uses Triple Barrier Method (TP/SL/Timeout based on volatility)
	ExitModeTBM ExitMode = "tbm"
	// ExitModeSignal exits on opposite signal (Long->Short or Short->Long)
	ExitModeSignal ExitMode = "signal"
	// ExitModeCustomStop uses per-bar stop prices from Python (trailing stop, MAE, etc.)
	// Stop price is checked against bar's high/low for immediate exit at stop price
	ExitModeCustomStop ExitMode = "custom_stop"
)

// Bar represents a single OHLCV bar with features.
// Maps to the parquet structure from Python signal generator.
type Bar struct {
	Timestamp   int64   // Bar timestamp (ms)
	Open        float64 // Open price
	High        float64 // High price
	HighTime    int64   // When High occurred (ms) - for precise SL/TP ordering
	Low         float64 // Low price
	LowTime     int64   // When Low occurred (ms) - for precise SL/TP ordering
	Close       float64 // Close price
	Volume      float64 // Volume
	RealizedVol float64 // Realized volatility (for barrier calculation)
}

// SignalBar represents a bar with its signal.
type SignalBar struct {
	Bar    Bar
	Signal Signal
}

// Position represents an open position.
type Position struct {
	Direction   Direction // Long or Short
	EntryPrice  float64   // Actual entry price (with slippage)
	EntryBar    int       // Bar index when entered
	TakeProfit  float64   // TP price level
	StopLoss    float64   // SL price level
	MaxHoldBars int       // Maximum bars to hold (vertical barrier)
	Size        float64   // Position size ratio (1.0 = 100%, 1.5 = 150%)
}

// Trade represents a completed trade.
type Trade struct {
	EntryBar    int        // Entry bar index
	ExitBar     int        // Exit bar index
	Direction   Direction  // Long or Short
	EntryPrice  float64    // Entry price (with slippage)
	ExitPrice   float64    // Exit price (with slippage)
	PnL         float64    // Net P&L ratio (after costs)
	GrossPnL    float64    // Gross P&L ratio (before costs)
	TotalCost   float64    // Total cost (fees + slippage)
	ExitReason  ExitReason // Why the trade was closed
	HoldingBars int        // Number of bars held
	Size        float64    // Position size ratio (1.0 = 100%, 1.5 = 150%)
}

// Config holds backtester configuration.
type Config struct {
	// Exit mode
	ExitMode ExitMode // "tbm" (Triple Barrier) or "signal" (opposite signal)

	// Barrier parameters (in volatility multiples) - used in TBM mode
	SLMult      float64 // Stop Loss multiplier (e.g., 2.0 means 2σ)
	PTMult      float64 // Profit Target multiplier
	MaxHoldBars int     // Maximum holding period (vertical barrier)

	// Cost model parameters
	BaseFee      float64 // Base exchange fee (e.g., 0.001 = 0.1%)
	BaseSlippage float64 // Minimum slippage (e.g., 0.0001 = 0.01%)
	ImpactCoeff  float64 // Market impact coefficient
	SlippageCap  float64 // Maximum slippage cap

	// Position sizing
	InitialCapital float64 // Starting capital
	RiskPerTrade   float64 // Risk per trade (e.g., 0.02 = 2%)
	MaxLeverage    float64 // Maximum leverage (e.g., 10.0 = 10x)

	// Equity calculation mode
	Compounding bool // true: compound returns, false: simple interest (additive)

	// Early stop
	MaxLossPct float64 // Stop backtest if loss exceeds this (e.g., 0.5 = 50%)
}

// DefaultConfig returns sensible default configuration.
func DefaultConfig() Config {
	return Config{
		ExitMode:       ExitModeTBM,
		SLMult:         2.0,
		PTMult:         2.5,
		MaxHoldBars:    100,
		BaseFee:        0.001,  // 0.10%
		BaseSlippage:   0.0001, // 0.01%
		ImpactCoeff:    0.1,
		SlippageCap:    0.005,  // 0.50%
		InitialCapital: 100000,
		RiskPerTrade:   1.0,    // 100% - full capital
		MaxLeverage:    10.0,   // 10x max leverage
		Compounding:    false,  // simple interest by default
		MaxLossPct:     0.5,    // 50% max loss - early stop
	}
}

// Validate checks if the configuration is valid.
func (c Config) Validate() error {
	// Validate exit mode
	if c.ExitMode != ExitModeTBM && c.ExitMode != ExitModeSignal && c.ExitMode != ExitModeCustomStop {
		return fmt.Errorf("ExitMode must be 'tbm', 'signal', or 'custom_stop', got %q", c.ExitMode)
	}

	// TBM mode requires barrier parameters
	if c.ExitMode == ExitModeTBM {
		if c.SLMult <= 0 {
			return fmt.Errorf("SLMult must be positive, got %f", c.SLMult)
		}
		if c.PTMult <= 0 {
			return fmt.Errorf("PTMult must be positive, got %f", c.PTMult)
		}
		if c.MaxHoldBars <= 0 {
			return fmt.Errorf("MaxHoldBars must be positive, got %d", c.MaxHoldBars)
		}
	}
	// CustomStop mode: no special validation needed, stop prices come from signal file

	// Cost model parameters (always validated)
	if c.BaseFee < 0 {
		return fmt.Errorf("BaseFee cannot be negative, got %f", c.BaseFee)
	}
	if c.ImpactCoeff < 0 {
		return fmt.Errorf("ImpactCoeff cannot be negative, got %f", c.ImpactCoeff)
	}
	if c.InitialCapital <= 0 {
		return fmt.Errorf("InitialCapital must be positive, got %f", c.InitialCapital)
	}
	if c.RiskPerTrade <= 0 || c.RiskPerTrade > 1 {
		return fmt.Errorf("RiskPerTrade must be in (0, 1], got %f", c.RiskPerTrade)
	}
	return nil
}

// Result holds the complete backtest results.
type Result struct {
	// Summary metrics
	TotalTrades  int     // Total number of trades
	WinRate      float64 // Winning trade ratio
	AvgPnL       float64 // Average P&L per trade
	TotalPnL     float64 // Total cumulative P&L
	SharpeRatio  float64 // Risk-adjusted return
	MaxDrawdown  float64 // Maximum drawdown ratio
	ProfitFactor float64 // Gross profit / Gross loss
	AvgHoldBars  float64 // Average holding period

	// Breakdown by exit reason
	TPCount      int // Take Profit exits
	SLCount      int // Stop Loss exits
	TimeoutCount int // Timeout exits

	// Detailed data
	Trades      []Trade   // All completed trades
	EquityCurve []float64 // Equity at each bar
}
