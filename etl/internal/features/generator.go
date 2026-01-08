package features

import (
	"math"
	"time"
)

// Generator is the main orchestrator for streaming feature generation.
// It combines L1, Regime, Stationarity, and Momentum features into a single
// output row per bar.
//
// Key design principles:
// - Single-pass streaming: O(1) per bar, O(1) memory per normalizer
// - State continuity: State persists across file boundaries
// - No batch operations: Pure streaming, no look-ahead
// - Deterministic: Same input sequence = same output
type Generator struct {
	config Config

	// Feature generators
	l1           *L1
	regime       *Regime
	stationarity *Stationarity
	momentum     *Momentum
	connorsRSI   *ConnorsRSI

	// Count
	count int64
}

// NewGenerator creates a new feature generator with the given configuration.
func NewGenerator(config Config) *Generator {
	return &Generator{
		config: config,
		l1:     NewL1(),
		regime: NewRegime(RegimeConfig{
			ParkinsonWindow: config.ParkinsonWindow,
			EntropyWindow:   config.EntropyWindow,
			EntropyBins:     config.EntropyBins,
			VolHalflife:     config.VolHalflife,
			EntropyHalflife: config.EntropyHalflife,
		}),
		stationarity: NewStationarity(StationarityConfig{
			FracDiffD:       config.FracDiffD,
			FracDiffThresh:  config.FracDiffThresh,
			DetrendHalflife: config.DetrendHalflife,
		}),
		momentum:   NewMomentum(DefaultMomentumConfig()),
		connorsRSI: NewConnorsRSI(DefaultConnorsRSIConfig()),
	}
}

// Process computes all features for a single bar.
// This is the main streaming entry point.
func (g *Generator) Process(bar InputBar) FeatureRow {
	g.count++

	var row FeatureRow

	// === Time & Price (passthrough) ===
	row.StartTime = bar.StartTime
	row.EndTime = bar.EndTime
	row.Open = bar.Open
	row.High = bar.High
	row.Low = bar.Low
	row.Close = bar.Close

	// === L1 Features (stateless) ===
	l1 := g.l1.Compute(bar)
	row.LogVolume = l1.LogVolume
	row.LogTickCount = l1.LogTickCount
	row.LogDuration = l1.LogDuration
	row.LogTradeIntensity = l1.LogTradeIntensity
	row.VWAPDeviation = l1.VWAPDeviation
	row.VolumeImbalance = l1.VolumeImbalance
	row.BarRange = l1.BarRange
	row.BarBody = l1.BarBody

	// === Regime Features (stateful) ===
	regime := g.regime.Update(bar)
	row.GarmanKlassVol = regime.GarmanKlassVol
	row.RealizedVol = regime.RealizedVol
	row.ShannonEntropy = regime.ShannonEntropy
	row.VolRatio = regime.VolRatio
	row.VolZScore = regime.VolZScore
	row.EntropyZScore = regime.EntropyZScore
	row.Skewness = regime.Skewness
	row.Kurtosis = regime.Kurtosis

	// === Stationarity Features (stateful) ===
	stat := g.stationarity.Update(bar)
	row.FracDiffClose = stat.FracDiffClose
	row.DetrendedLogPrice = stat.DetrendedLogPrice
	row.Returns = stat.Returns

	// === Momentum Features (stateful) ===
	mom := g.momentum.Update(stat.Returns, bar.Volume)
	row.VWMomentum = mom.VWMomentum
	row.MomentumZScore10 = mom.MomentumZScore10
	row.MomentumZScore50 = mom.MomentumZScore50
	row.MomentumZScore250 = mom.MomentumZScore250
	row.MomentumZScore1000 = mom.MomentumZScore1000

	// === Technical Indicators ===
	row.ConnorsRSI = g.connorsRSI.Update(bar.Close)

	// === Cyclical Time Features ===
	// Sin/Cos encoding for hour of day and day of week (UTC)
	// Preserves cyclical continuity: 23:00 is close to 00:00, Sunday close to Monday
	row.SinTime, row.CosTime, row.SinWeek, row.CosWeek = cyclicalTimeFeatures(bar.StartTime)

	// === Primed Status ===
	row.IsPrimed = g.regime.IsPrimed() && g.stationarity.IsPrimed() && g.momentum.IsPrimed() && g.connorsRSI.IsPrimed()

	return row
}

// IsPrimed returns true if all normalizers are warmed up.
func (g *Generator) IsPrimed() bool {
	return g.regime.IsPrimed() && g.stationarity.IsPrimed() && g.momentum.IsPrimed() && g.connorsRSI.IsPrimed()
}

// Count returns the number of bars processed.
func (g *Generator) Count() int64 {
	return g.count
}

// Reset clears all state.
func (g *Generator) Reset() {
	g.regime.Reset()
	g.stationarity.Reset()
	g.momentum.Reset()
	g.connorsRSI.Reset()
	g.count = 0
}

// GeneratorState represents serializable state for the entire generator.
type GeneratorState struct {
	Regime       RegimeState
	Stationarity StationarityState
	Momentum     MomentumState
	ConnorsRSI   ConnorsRSIState
	Count        int64
}

// State returns a serializable state snapshot.
func (g *Generator) State() GeneratorState {
	return GeneratorState{
		Regime:       g.regime.State(),
		Stationarity: g.stationarity.State(),
		Momentum:     g.momentum.State(),
		ConnorsRSI:   g.connorsRSI.State(),
		Count:        g.count,
	}
}

// LoadState restores state from a snapshot.
func (g *Generator) LoadState(state GeneratorState) {
	g.regime.LoadState(state.Regime)
	g.stationarity.LoadState(state.Stationarity)
	g.momentum.LoadState(state.Momentum)
	g.connorsRSI.LoadState(state.ConnorsRSI)
	g.count = state.Count
}

// FeatureNames returns all feature column names in order.
func (g *Generator) FeatureNames() []string {
	return []string{
		// Time
		"start_time", "end_time",
		// Price
		"open", "high", "low", "close",
		// L1
		"log_volume", "log_tick_count", "log_duration", "log_trade_intensity",
		"vwap_deviation", "volume_imbalance", "bar_range", "bar_body",
		// Regime
		"garman_klass_vol", "realized_vol", "shannon_entropy", "vol_ratio",
		"vol_zscore", "entropy_zscore", "skewness", "kurtosis",
		// Stationarity
		"frac_diff_close", "detrended_log_price", "returns",
		// Momentum
		"vw_momentum", "momentum_zscore_10", "momentum_zscore_50", "momentum_zscore_250", "momentum_zscore_1000",
		// Technical
		"connors_rsi",
		// Cyclical Time
		"sin_time", "cos_time", "sin_week", "cos_week",
		// Meta
		"is_primed",
	}
}

// cyclicalTimeFeatures computes sin/cos encoded time features from Unix timestamp.
// Uses UTC timezone for consistency across global crypto markets.
//
// Returns:
//   - sinTime, cosTime: Hour of day encoded as sin(2π*h/24), cos(2π*h/24)
//   - sinWeek, cosWeek: Day of week encoded as sin(2π*d/7), cos(2π*d/7)
//
// The sin/cos encoding preserves cyclical continuity:
//   - 23:00 and 00:00 are close in the encoded space
//   - Sunday and Monday are close in the encoded space
func cyclicalTimeFeatures(unixMs int64) (sinTime, cosTime, sinWeek, cosWeek float64) {
	// Convert milliseconds to time.Time (UTC)
	t := time.UnixMilli(unixMs).UTC()

	// Hour of day: 0-23 → sin/cos
	hour := float64(t.Hour())
	hourAngle := 2 * math.Pi * hour / 24.0
	sinTime = math.Sin(hourAngle)
	cosTime = math.Cos(hourAngle)

	// Day of week: 0=Sunday, 6=Saturday → sin/cos
	dayOfWeek := float64(t.Weekday())
	weekAngle := 2 * math.Pi * dayOfWeek / 7.0
	sinWeek = math.Sin(weekAngle)
	cosWeek = math.Cos(weekAngle)

	return
}
