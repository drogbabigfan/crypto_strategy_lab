// Package features provides streaming feature generation for financial time series.
// All features are computed in a single pass with O(1) memory per update.
package features

import "dl-rl-btc-etl/internal/bars"

// FeatureRow contains all computed features for a single bar.
// This is the output that gets written to Parquet for model training.
type FeatureRow struct {
	// === Time ===
	StartTime int64 `parquet:"start_time"`
	EndTime   int64 `parquet:"end_time"`

	// === Price ===
	Open  float64 `parquet:"open"`
	High  float64 `parquet:"high"`
	Low   float64 `parquet:"low"`
	Close float64 `parquet:"close"`

	// === L1 Features (Stateless transforms) ===
	LogVolume         float64 `parquet:"log_volume"`
	LogTickCount      float64 `parquet:"log_tick_count"`
	LogDuration       float64 `parquet:"log_duration"`
	LogTradeIntensity float64 `parquet:"log_trade_intensity"`
	VWAPDeviation     float64 `parquet:"vwap_deviation"`
	VolumeImbalance   float64 `parquet:"volume_imbalance"`
	BarRange          float64 `parquet:"bar_range"`
	BarBody           float64 `parquet:"bar_body"`

	// === Regime Features (Rolling statistics) ===
	GarmanKlassVol float64 `parquet:"garman_klass_vol"` // GK volatility (replaces Parkinson)
	RealizedVol    float64 `parquet:"realized_vol"`
	ShannonEntropy float64 `parquet:"shannon_entropy"`
	VolRatio       float64 `parquet:"vol_ratio"`
	VolZScore      float64 `parquet:"vol_zscore"`
	EntropyZScore  float64 `parquet:"entropy_zscore"`
	Skewness       float64 `parquet:"skewness"`  // 3rd moment: distribution asymmetry
	Kurtosis       float64 `parquet:"kurtosis"`  // 4th moment: tail thickness

	// === Stationarity Features ===
	FracDiffClose     float64 `parquet:"frac_diff_close"`
	DetrendedLogPrice float64 `parquet:"detrended_log_price"`
	Returns           float64 `parquet:"returns"`

	// === Momentum Features ===
	VWMomentum         float64 `parquet:"vw_momentum"`           // Volume-Weighted Momentum
	MomentumZScore10   float64 `parquet:"momentum_zscore_10"`
	MomentumZScore50   float64 `parquet:"momentum_zscore_50"`
	MomentumZScore250  float64 `parquet:"momentum_zscore_250"`
	MomentumZScore1000 float64 `parquet:"momentum_zscore_1000"`

	// === Technical Indicators ===
	ConnorsRSI float64 `parquet:"connors_rsi"` // Composite RSI indicator

	// === Cyclical Time Features ===
	// Sin/Cos encoding preserves cyclical continuity (23:00 is close to 00:00)
	SinTime float64 `parquet:"sin_time"` // sin(2π * hour / 24) - Hour of day
	CosTime float64 `parquet:"cos_time"` // cos(2π * hour / 24) - Hour of day
	SinWeek float64 `parquet:"sin_week"` // sin(2π * day_of_week / 7) - Day of week
	CosWeek float64 `parquet:"cos_week"` // cos(2π * day_of_week / 7) - Day of week

	// === Meta ===
	IsPrimed bool `parquet:"is_primed"` // True if all normalizers are warmed up
}

// Config holds configuration for feature generation.
type Config struct {
	// Regime
	ParkinsonWindow int     `yaml:"parkinson_window"` // Window for Parkinson volatility
	EntropyWindow   int     `yaml:"entropy_window"`   // Window for Shannon entropy
	EntropyBins     int     `yaml:"entropy_bins"`     // Number of bins for entropy histogram
	VolHalflife     float64 `yaml:"vol_halflife"`     // EWM halflife for vol z-score
	EntropyHalflife float64 `yaml:"entropy_halflife"` // EWM halflife for entropy z-score

	// Stationarity
	FracDiffD      float64 `yaml:"frac_diff_d"`       // Fractional differentiation order
	FracDiffThresh float64 `yaml:"frac_diff_thresh"`  // Weight threshold for FracDiff
	DetrendHalflife float64 `yaml:"detrend_halflife"` // EWM halflife for detrending

	// Momentum
	MomentumWindows  []int   `yaml:"momentum_windows"`  // Windows for momentum calculation
	MomentumHalflife float64 `yaml:"momentum_halflife"` // EWM halflife for momentum z-score
}

// DefaultConfig returns sensible default configuration.
func DefaultConfig() Config {
	return Config{
		// Regime
		ParkinsonWindow: 24,
		EntropyWindow:   24,
		EntropyBins:     10,
		VolHalflife:     50,
		EntropyHalflife: 50,

		// Stationarity
		FracDiffD:       0.4,
		FracDiffThresh:  1e-5,
		DetrendHalflife: 100,

		// Momentum
		MomentumWindows:  []int{10, 50, 250, 1000},
		MomentumHalflife: 50,
	}
}

// InputBar is an alias for the bar type from the bars package.
type InputBar = bars.DynamicDollarBar

// Epsilon is a small value to prevent division by zero.
const Epsilon = 1e-10
