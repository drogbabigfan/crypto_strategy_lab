package features

import (
	"dl-rl-btc-etl/internal/features/normalizer"
)

// MomentumFeatures contains momentum-related features.
type MomentumFeatures struct {
	VWMomentum         float64 // Volume-Weighted Momentum (raw)
	MomentumZScore10   float64 // Z-score with EWM halflife 10
	MomentumZScore50   float64 // Z-score with EWM halflife 50
	MomentumZScore250  float64 // Z-score with EWM halflife 250
	MomentumZScore1000 float64 // Z-score with EWM halflife 1000
}

// MomentumConfig holds configuration for momentum features.
type MomentumConfig struct {
	Halflife10   float64
	Halflife50   float64
	Halflife250  float64
	Halflife1000 float64
}

// DefaultMomentumConfig returns default configuration.
func DefaultMomentumConfig() MomentumConfig {
	return MomentumConfig{
		Halflife10:   10,
		Halflife50:   50,
		Halflife250:  250,
		Halflife1000: 1000,
	}
}

// Momentum computes momentum features using streaming algorithms.
type Momentum struct {
	config MomentumConfig

	// EWM normalizers for different time horizons
	ewm10   *normalizer.EWM
	ewm50   *normalizer.EWM
	ewm250  *normalizer.EWM
	ewm1000 *normalizer.EWM

	// Previous close for return calculation
	prevClose float64

	// Current features
	current MomentumFeatures
	count   int64
}

// NewMomentum creates a new momentum feature generator.
func NewMomentum(config MomentumConfig) *Momentum {
	return &Momentum{
		config:  config,
		ewm10:   normalizer.NewEWMWithHalflife(config.Halflife10),
		ewm50:   normalizer.NewEWMWithHalflife(config.Halflife50),
		ewm250:  normalizer.NewEWMWithHalflife(config.Halflife250),
		ewm1000: normalizer.NewEWMWithHalflife(config.Halflife1000),
	}
}

// Update processes a new bar and returns updated features.
// Returns is the bar's return (from stationarity or computed externally).
// Volume is the bar's trading volume.
func (m *Momentum) Update(returns, volume float64) MomentumFeatures {
	m.count++

	// === Volume-Weighted Momentum ===
	// VW Momentum = Return * Volume
	// Captures both direction and conviction (volume as proxy for conviction)
	m.current.VWMomentum = returns * volume

	// === Multi-Horizon Z-Scores ===
	// Each EWM normalizes the momentum signal at different time scales
	// Short horizons (10) = responsive, noisy
	// Long horizons (1000) = smooth, lagging
	m.current.MomentumZScore10 = m.ewm10.Update(m.current.VWMomentum)
	m.current.MomentumZScore50 = m.ewm50.Update(m.current.VWMomentum)
	m.current.MomentumZScore250 = m.ewm250.Update(m.current.VWMomentum)
	m.current.MomentumZScore1000 = m.ewm1000.Update(m.current.VWMomentum)

	return m.current
}

// CurrentFeatures returns the current feature values.
func (m *Momentum) CurrentFeatures() MomentumFeatures {
	return m.current
}

// IsPrimed returns true if all EWM normalizers are warmed up.
// Uses the longest horizon as the criterion.
func (m *Momentum) IsPrimed() bool {
	return m.ewm1000.IsPrimed()
}

// Reset clears all state.
func (m *Momentum) Reset() {
	m.ewm10.Reset()
	m.ewm50.Reset()
	m.ewm250.Reset()
	m.ewm1000.Reset()
	m.prevClose = 0
	m.current = MomentumFeatures{}
	m.count = 0
}

// MomentumState represents serializable state.
type MomentumState struct {
	EWM10State   normalizer.EWMState
	EWM50State   normalizer.EWMState
	EWM250State  normalizer.EWMState
	EWM1000State normalizer.EWMState
	PrevClose    float64
	Count        int64
}

// State returns serializable state snapshot.
func (m *Momentum) State() MomentumState {
	return MomentumState{
		EWM10State:   m.ewm10.State(),
		EWM50State:   m.ewm50.State(),
		EWM250State:  m.ewm250.State(),
		EWM1000State: m.ewm1000.State(),
		PrevClose:    m.prevClose,
		Count:        m.count,
	}
}

// LoadState restores from serialized state.
func (m *Momentum) LoadState(state MomentumState) {
	m.ewm10.LoadState(state.EWM10State)
	m.ewm50.LoadState(state.EWM50State)
	m.ewm250.LoadState(state.EWM250State)
	m.ewm1000.LoadState(state.EWM1000State)
	m.prevClose = state.PrevClose
	m.count = state.Count
}
