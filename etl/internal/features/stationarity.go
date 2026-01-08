package features

import (
	"math"

	"dl-rl-btc-etl/internal/features/normalizer"
)

// StationarityFeatures contains stationarity-related features.
type StationarityFeatures struct {
	FracDiffClose     float64
	DetrendedLogPrice float64
	Returns           float64
}

// StationarityConfig holds configuration for stationarity features.
type StationarityConfig struct {
	FracDiffD       float64 // Fractional differentiation order (0 < d < 1)
	FracDiffThresh  float64 // Weight threshold for FracDiff window
	DetrendHalflife float64 // EWM halflife for detrending
}

// DefaultStationarityConfig returns default configuration.
func DefaultStationarityConfig() StationarityConfig {
	return StationarityConfig{
		FracDiffD:       0.4,
		FracDiffThresh:  1e-5,
		DetrendHalflife: 100,
	}
}

// Stationarity computes stationarity features using streaming algorithms.
type Stationarity struct {
	config StationarityConfig

	// FracDiff: precomputed weights and ring buffer for log prices
	fracDiffWeights []float64
	logPriceBuffer  *ringBuffer

	// Detrending: EWM for trend estimation
	detrendEWM *normalizer.EWM

	// Returns: previous close for return calculation
	prevClose float64

	// Current features
	current StationarityFeatures
	count   int64
}

// NewStationarity creates a new stationarity feature generator.
func NewStationarity(config StationarityConfig) *Stationarity {
	// Compute fractional differentiation weights
	weights := computeFracDiffWeights(config.FracDiffD, config.FracDiffThresh)

	return &Stationarity{
		config:          config,
		fracDiffWeights: weights,
		logPriceBuffer:  newRingBuffer(len(weights)),
		detrendEWM:      normalizer.NewEWMWithHalflife(config.DetrendHalflife),
	}
}

// computeFracDiffWeights computes the weights for fractional differentiation.
// Formula: w_k = -w_{k-1} * (d - k + 1) / k
// Where d is the differentiation order and k is the lag.
func computeFracDiffWeights(d, threshold float64) []float64 {
	weights := []float64{1.0} // w_0 = 1

	k := 1
	for {
		// w_k = -w_{k-1} * (d - k + 1) / k
		w := -weights[k-1] * (d - float64(k) + 1) / float64(k)
		if math.Abs(w) < threshold {
			break
		}
		weights = append(weights, w)
		k++

		// Safety limit
		if k > 10000 {
			break
		}
	}

	return weights
}

// Update processes a new bar and returns updated features.
func (s *Stationarity) Update(bar InputBar) StationarityFeatures {
	s.count++

	// === Returns ===
	if s.prevClose > Epsilon {
		s.current.Returns = (bar.Close - s.prevClose) / s.prevClose
	} else {
		s.current.Returns = 0
	}
	s.prevClose = bar.Close

	// === Log Price for FracDiff and Detrending ===
	close := bar.Close
	if close < Epsilon {
		close = Epsilon
	}
	logPrice := math.Log(close)

	// Add to log price buffer
	s.logPriceBuffer.Push(logPrice)

	// === Fractional Differentiation ===
	// FracDiff(log(P)) = sum(w_k * log(P_{t-k}))
	if s.logPriceBuffer.count >= len(s.fracDiffWeights) {
		s.current.FracDiffClose = s.computeFracDiff()
	} else if s.logPriceBuffer.count > 0 {
		// Partial computation with available data
		s.current.FracDiffClose = s.computeFracDiffPartial()
	}

	// === Detrended Log Price ===
	// Detrended = log(P) - EWM(log(P))
	// First update EWM to get trend estimate
	_ = s.detrendEWM.Update(logPrice)
	trend := s.detrendEWM.Mean()
	s.current.DetrendedLogPrice = logPrice - trend

	return s.current
}

// computeFracDiff computes fractional differentiation when buffer is full.
func (s *Stationarity) computeFracDiff() float64 {
	values := s.logPriceBuffer.Values()
	n := len(values)
	numWeights := len(s.fracDiffWeights)

	if n < numWeights {
		return 0
	}

	var result float64
	// Current index is (buffer.index - 1 + window) % window for most recent value
	// We need to access values in reverse chronological order

	for k := 0; k < numWeights; k++ {
		// Get the (k)th lag value
		// Most recent is at (index - 1 + window) % window
		idx := (s.logPriceBuffer.index - 1 - k + s.logPriceBuffer.window) % s.logPriceBuffer.window
		result += s.fracDiffWeights[k] * values[idx]
	}

	return result
}

// computeFracDiffPartial computes partial FracDiff with available data.
func (s *Stationarity) computeFracDiffPartial() float64 {
	values := s.logPriceBuffer.Values()
	n := s.logPriceBuffer.count

	if n == 0 {
		return 0
	}

	var result float64
	numWeights := min(n, len(s.fracDiffWeights))

	for k := 0; k < numWeights; k++ {
		// Get the (k)th lag value from the buffer
		idx := (s.logPriceBuffer.index - 1 - k + s.logPriceBuffer.window) % s.logPriceBuffer.window
		if idx < 0 {
			idx += s.logPriceBuffer.window
		}
		if idx < len(values) {
			result += s.fracDiffWeights[k] * values[idx]
		}
	}

	return result
}

// CurrentFeatures returns the current feature values.
func (s *Stationarity) CurrentFeatures() StationarityFeatures {
	return s.current
}

// IsPrimed returns true if enough data has been processed.
func (s *Stationarity) IsPrimed() bool {
	// Need full FracDiff window and primed EWM
	return s.logPriceBuffer.IsFull() && s.detrendEWM.IsPrimed()
}

// Reset clears all state.
func (s *Stationarity) Reset() {
	s.logPriceBuffer.Reset()
	s.detrendEWM.Reset()
	s.prevClose = 0
	s.current = StationarityFeatures{}
	s.count = 0
}

// StationarityState represents serializable state.
type StationarityState struct {
	LogPriceBuffer []float64
	LogPriceIndex  int
	LogPriceCount  int
	DetrendState   normalizer.EWMState
	PrevClose      float64
	Count          int64
}

// State returns serializable state snapshot.
func (s *Stationarity) State() StationarityState {
	return StationarityState{
		LogPriceBuffer: append([]float64{}, s.logPriceBuffer.data...),
		LogPriceIndex:  s.logPriceBuffer.index,
		LogPriceCount:  s.logPriceBuffer.count,
		DetrendState:   s.detrendEWM.State(),
		PrevClose:      s.prevClose,
		Count:          s.count,
	}
}

// LoadState restores from serialized state.
func (s *Stationarity) LoadState(state StationarityState) {
	copy(s.logPriceBuffer.data, state.LogPriceBuffer)
	s.logPriceBuffer.index = state.LogPriceIndex
	s.logPriceBuffer.count = state.LogPriceCount
	s.detrendEWM.LoadState(state.DetrendState)
	s.prevClose = state.PrevClose
	s.count = state.Count
}

// min returns the smaller of two integers.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
