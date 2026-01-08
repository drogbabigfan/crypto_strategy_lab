package features

import (
	"math"
)

// L1Features contains the first-level (stateless) features.
// These are simple transformations that don't require historical state.
type L1Features struct {
	LogVolume         float64
	LogTickCount      float64
	LogDuration       float64
	LogTradeIntensity float64
	VWAPDeviation     float64
	VolumeImbalance   float64
	BarRange          float64
	BarBody           float64
	KyleLambda        float64 // Amihud illiquidity measure
}

// L1 computes stateless L1 features from bar data.
// All computations are O(1) with no memory allocation.
type L1 struct{}

// NewL1 creates a new L1 feature generator.
func NewL1() *L1 {
	return &L1{}
}

// Compute calculates all L1 features for a single bar.
// This is a pure function with no side effects.
func (l *L1) Compute(bar InputBar) L1Features {
	var f L1Features

	// === Log Transforms ===
	// Log1p(x) = log(1 + x), handles x=0 gracefully
	f.LogVolume = math.Log1p(bar.Volume)
	f.LogTickCount = math.Log1p(float64(bar.TickCount))
	f.LogDuration = math.Log1p(bar.Duration)

	// Trade Intensity = DollarValue / Duration
	// Measures how fast money is flowing through the market
	duration := bar.Duration
	if duration < Epsilon {
		duration = Epsilon
	}
	tradeIntensity := bar.DollarValue / duration
	f.LogTradeIntensity = math.Log1p(tradeIntensity)

	// === VWAP Deviation ===
	// VWAP = DollarValue / Volume (Volume-Weighted Average Price)
	// Deviation = (Close - VWAP) / Close
	// Positive = Close above VWAP (bullish), Negative = below (bearish)
	volume := bar.Volume
	if volume < Epsilon {
		volume = Epsilon
	}
	vwap := bar.DollarValue / volume
	f.VWAPDeviation = (bar.Close - vwap) / (bar.Close + Epsilon)

	// === Volume Imbalance ===
	// NetImbalance = BuyDollarVol - SellDollarVol
	// Imbalance = NetImbalance / TotalDollarValue
	// Range: [-1, 1], Positive = buy pressure, Negative = sell pressure
	dollarValue := bar.DollarValue
	if dollarValue < Epsilon {
		dollarValue = Epsilon
	}
	f.VolumeImbalance = bar.NetImbalance / dollarValue

	// === Bar Range ===
	// Range = (High - Low) / Close
	// Measures volatility within the bar, normalized by price
	close := bar.Close
	if close < Epsilon {
		close = Epsilon
	}
	f.BarRange = (bar.High - bar.Low) / close

	// === Bar Body ===
	// Body = (Close - Open) / Close
	// Positive = bullish (closed higher), Negative = bearish
	f.BarBody = (bar.Close - bar.Open) / close

	return f
}

// FeatureNames returns the names of all L1 features.
func (l *L1) FeatureNames() []string {
	return []string{
		"log_volume",
		"log_tick_count",
		"log_duration",
		"log_trade_intensity",
		"vwap_deviation",
		"volume_imbalance",
		"bar_range",
		"bar_body",
	}
}
