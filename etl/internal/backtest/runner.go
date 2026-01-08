package backtest

import (
	"encoding/json"
	"fmt"
	"math"
	"os"

	"github.com/segmentio/parquet-go"
)

// Runner orchestrates the backtest execution.
// It reads signals and features from parquet files, runs the simulation,
// and outputs metrics.
type Runner struct {
	config    Config
	executor  *Executor
	metrics   *MetricsCalculator
}

// NewRunner creates a new backtest runner.
func NewRunner(cfg Config) (*Runner, error) {
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	return &Runner{
		config:    cfg,
		executor:  NewExecutor(cfg),
		metrics:   NewMetricsCalculator(cfg.InitialCapital, cfg.RiskPerTrade, cfg.Compounding),
	}, nil
}

// SignalRecord represents a row in the signals parquet file.
type SignalRecord struct {
	Timestamp int64   `parquet:"timestamp"`
	Signal    int8    `parquet:"signal"`            // -1, 0, 1
	Size      float64 `parquet:"size,optional"`     // Position size ratio (0.25 = 25%, 1.0 = 100%, 1.5 = 150%)
	SLPrice   float64 `parquet:"sl_price,optional"` // Stop loss price (0 = no SL)
}

// FeatureRecord represents a row in the features parquet file.
type FeatureRecord struct {
	Timestamp   int64   `parquet:"timestamp"`
	Open        float64 `parquet:"open"`
	High        float64 `parquet:"high"`
	HighTime    int64   `parquet:"high_time"`
	Low         float64 `parquet:"low"`
	LowTime     int64   `parquet:"low_time"`
	Close       float64 `parquet:"close"`
	Volume      float64 `parquet:"volume"`
	RealizedVol float64 `parquet:"realized_vol"`
}

// Run executes the backtest with data from files.
func (r *Runner) Run(signalsPath, featuresPath string) (Result, error) {
	// Load signals and sizes
	signals, sizes, err := r.loadSignals(signalsPath)
	if err != nil {
		return Result{}, fmt.Errorf("failed to load signals: %w", err)
	}

	// Load features (bars)
	bars, err := r.loadFeatures(featuresPath)
	if err != nil {
		return Result{}, fmt.Errorf("failed to load features: %w", err)
	}

	// Validate lengths
	if len(signals) != len(bars) {
		return Result{}, fmt.Errorf("signal count (%d) != bar count (%d)", len(signals), len(bars))
	}

	// Run simulation
	return r.RunWithData(bars, signals, sizes), nil
}

// RunWithData executes the backtest with in-memory data.
// sizes is optional - if nil or empty, all positions use size 1.0.
func (r *Runner) RunWithData(bars []Bar, signals []Signal, sizes []float64) Result {
	r.executor.Reset()

	trades := make([]Trade, 0)
	cumulativePnL := 0.0
	stoppedEarly := false
	stopBar := len(bars)

	for i := 0; i < len(bars); i++ {
		signal := SignalNeutral
		if i < len(signals) {
			signal = signals[i]
		}

		// Get position size (default 1.0 if not provided)
		size := 1.0
		if sizes != nil && i < len(sizes) && sizes[i] > 0 {
			size = sizes[i]
		}

		trade := r.executor.ProcessBar(i, bars[i], signal, size)
		if trade != nil {
			trades = append(trades, *trade)
			cumulativePnL += trade.PnL * trade.Size // Weighted by size

			// Early stop if loss exceeds threshold
			if r.config.MaxLossPct > 0 && cumulativePnL < -r.config.MaxLossPct {
				stoppedEarly = true
				stopBar = i
				break
			}
		}
	}

	// Force close any open position at the end (if not stopped early)
	if r.executor.HasPosition() && len(bars) > 0 {
		var lastBar Bar
		if stoppedEarly {
			lastBar = bars[stopBar]
		} else {
			lastBar = bars[len(bars)-1]
		}
		avgVol := r.executor.getAverageVolume()
		trade := r.executor.ForceClose(stopBar, lastBar.Close, avgVol, lastBar.RealizedVol)
		if trade != nil {
			trades = append(trades, *trade)
		}
	}

	// Calculate metrics with mark-to-market equity curve
	// Use only bars up to stop point
	barsToUse := bars
	if stoppedEarly {
		barsToUse = bars[:stopBar+1]
	}
	return r.metrics.Calculate(trades, barsToUse)
}

// loadSignals reads signals and sizes from a parquet file.
// Returns signals, sizes (nil if not present in file), and error.
func (r *Runner) loadSignals(path string) ([]Signal, []float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := parquet.NewReader(file)
	defer reader.Close()

	signals := make([]Signal, 0, reader.NumRows())
	sizes := make([]float64, 0, reader.NumRows())
	hasSizes := false

	for {
		var record SignalRecord
		err := reader.Read(&record)
		if err != nil {
			break // EOF or error
		}
		signals = append(signals, Signal(record.Signal))

		// Check if size column is present (non-zero value indicates it's set)
		if record.Size > 0 {
			hasSizes = true
		}
		sizes = append(sizes, record.Size)
	}

	// If no sizes were set, return nil to indicate default sizing
	if !hasSizes {
		return signals, nil, nil
	}

	// Replace 0 sizes with 1.0 (default)
	for i := range sizes {
		if sizes[i] <= 0 {
			sizes[i] = 1.0
		}
	}

	return signals, sizes, nil
}

// loadFeatures reads bars/features from a parquet file.
func (r *Runner) loadFeatures(path string) ([]Bar, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := parquet.NewReader(file)
	defer reader.Close()

	bars := make([]Bar, 0, reader.NumRows())
	for {
		var record FeatureRecord
		err := reader.Read(&record)
		if err != nil {
			break // EOF or error
		}
		bars = append(bars, Bar{
			Timestamp:   record.Timestamp,
			Open:        record.Open,
			High:        record.High,
			HighTime:    record.HighTime,
			Low:         record.Low,
			LowTime:     record.LowTime,
			Close:       record.Close,
			Volume:      record.Volume,
			RealizedVol: record.RealizedVol,
		})
	}

	return bars, nil
}

// ResultJSON is the JSON-serializable result format.
type ResultJSON struct {
	TotalTrades  int     `json:"total_trades"`
	WinRate      float64 `json:"win_rate"`
	AvgPnL       float64 `json:"avg_pnl"`
	TotalPnL     float64 `json:"total_pnl"`
	SharpeRatio  float64 `json:"sharpe_ratio"`
	MaxDrawdown  float64 `json:"max_drawdown"`
	ProfitFactor float64 `json:"profit_factor"`
	AvgHoldBars  float64 `json:"avg_hold_bars"`
	TPCount      int     `json:"tp_count"`
	SLCount      int     `json:"sl_count"`
	TimeoutCount int     `json:"timeout_count"`
	EquityCurve  []float64 `json:"equity_curve,omitempty"`
}

// sanitizeFloat converts Inf/NaN to JSON-safe values.
func sanitizeFloat(v float64) float64 {
	if math.IsInf(v, 1) {
		return 1e308 // Max safe JSON float
	}
	if math.IsInf(v, -1) {
		return -1e308
	}
	if math.IsNaN(v) {
		return 0
	}
	return v
}

// ToJSON converts Result to JSON-serializable format.
func (r Result) ToJSON(includeEquityCurve bool) ResultJSON {
	res := ResultJSON{
		TotalTrades:  r.TotalTrades,
		WinRate:      sanitizeFloat(r.WinRate),
		AvgPnL:       sanitizeFloat(r.AvgPnL),
		TotalPnL:     sanitizeFloat(r.TotalPnL),
		SharpeRatio:  sanitizeFloat(r.SharpeRatio),
		MaxDrawdown:  sanitizeFloat(r.MaxDrawdown),
		ProfitFactor: sanitizeFloat(r.ProfitFactor),
		AvgHoldBars:  sanitizeFloat(r.AvgHoldBars),
		TPCount:      r.TPCount,
		SLCount:      r.SLCount,
		TimeoutCount: r.TimeoutCount,
	}

	if includeEquityCurve {
		res.EquityCurve = r.EquityCurve
	}

	return res
}

// WriteJSON writes the result to a JSON file.
func (r Result) WriteJSON(path string, includeEquityCurve bool) error {
	data, err := json.MarshalIndent(r.ToJSON(includeEquityCurve), "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}

// ConfigFromJSON reads config from a JSON file.
func ConfigFromJSON(path string) (Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, err
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Config{}, err
	}

	return cfg, nil
}

// PrintSummary prints a human-readable summary of the result.
func (r Result) PrintSummary() {
	fmt.Println("=== Backtest Results ===")
	fmt.Printf("Total Trades:   %d\n", r.TotalTrades)
	fmt.Printf("Win Rate:       %.2f%%\n", r.WinRate*100)
	fmt.Printf("Avg PnL:        %.4f%%\n", r.AvgPnL*100)
	fmt.Printf("Total PnL:      %.4f%%\n", r.TotalPnL*100)
	fmt.Printf("Sharpe Ratio:   %.2f\n", r.SharpeRatio)
	fmt.Printf("Max Drawdown:   %.2f%%\n", r.MaxDrawdown*100)
	fmt.Printf("Profit Factor:  %.2f\n", r.ProfitFactor)
	fmt.Printf("Avg Hold Bars:  %.1f\n", r.AvgHoldBars)
	fmt.Println("--- Exit Reasons ---")
	fmt.Printf("Take Profit:    %d\n", r.TPCount)
	fmt.Printf("Stop Loss:      %d\n", r.SLCount)
	fmt.Printf("Timeout:        %d\n", r.TimeoutCount)
}
