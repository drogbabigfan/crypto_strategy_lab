package features

import (
	"math"

	"dl-rl-btc-etl/internal/features/normalizer"
)

// RegimeFeatures contains regime detection features.
type RegimeFeatures struct {
	GarmanKlassVol float64 // Garman-Klass volatility (more efficient than Parkinson)
	RealizedVol    float64
	ShannonEntropy float64
	VolRatio       float64
	VolZScore      float64
	EntropyZScore  float64
	Skewness       float64 // 3rd moment: distribution asymmetry
	Kurtosis       float64 // 4th moment: tail thickness (excess)
}

// RegimeConfig holds configuration for regime detection.
type RegimeConfig struct {
	ParkinsonWindow int     // Window for Parkinson volatility
	EntropyWindow   int     // Window for Shannon entropy
	EntropyBins     int     // Number of bins for entropy histogram
	VolHalflife     float64 // EWM halflife for volatility z-score
	EntropyHalflife float64 // EWM halflife for entropy z-score
}

// DefaultRegimeConfig returns default regime configuration.
func DefaultRegimeConfig() RegimeConfig {
	return RegimeConfig{
		ParkinsonWindow: 24,
		EntropyWindow:   24,
		EntropyBins:     10,
		VolHalflife:     50,
		EntropyHalflife: 50,
	}
}

// Regime computes market regime features using streaming algorithms.
// It maintains state across bar updates for rolling calculations.
type Regime struct {
	config RegimeConfig

	// Garman-Klass volatility: ring buffer for GK terms
	gkBuffer *ringBuffer
	gkSum    float64

	// Realized volatility: ring buffer for returns
	returnBuffer *ringBuffer
	prevClose    float64
	prevOpen     float64

	// Shannon entropy: ring buffer for returns
	entropyBuffer *ringBuffer

	// Higher moments: Welford accumulator for skewness/kurtosis
	momentAccum *normalizer.Welford

	// EWM normalizers for z-scores
	volNormalizer     *normalizer.EWM
	entropyNormalizer *normalizer.EWM

	// Current features
	current RegimeFeatures
	count   int64
}

// ringBuffer is a simple fixed-size circular buffer.
type ringBuffer struct {
	data   []float64
	index  int
	count  int
	window int
}

func newRingBuffer(size int) *ringBuffer {
	return &ringBuffer{
		data:   make([]float64, size),
		window: size,
	}
}

func (r *ringBuffer) Push(value float64) (old float64, wasFull bool) {
	wasFull = r.count >= r.window
	if wasFull {
		old = r.data[r.index]
	}

	r.data[r.index] = value
	r.index = (r.index + 1) % r.window

	if r.count < r.window {
		r.count++
	}

	return old, wasFull
}

func (r *ringBuffer) IsFull() bool {
	return r.count >= r.window
}

func (r *ringBuffer) Values() []float64 {
	if r.count < r.window {
		return r.data[:r.count]
	}
	return r.data
}

func (r *ringBuffer) Reset() {
	r.index = 0
	r.count = 0
	for i := range r.data {
		r.data[i] = 0
	}
}

// NewRegime creates a new regime feature generator.
func NewRegime(config RegimeConfig) *Regime {
	return &Regime{
		config:            config,
		gkBuffer:          newRingBuffer(config.ParkinsonWindow),
		returnBuffer:      newRingBuffer(config.ParkinsonWindow),
		entropyBuffer:     newRingBuffer(config.EntropyWindow),
		momentAccum:       normalizer.NewWelfordWithMinSamples(30),
		volNormalizer:     normalizer.NewEWMWithHalflife(config.VolHalflife),
		entropyNormalizer: normalizer.NewEWMWithHalflife(config.EntropyHalflife),
	}
}

// Update processes a new bar and returns updated features.
func (r *Regime) Update(bar InputBar) RegimeFeatures {
	r.count++

	// === Garman-Klass Volatility ===
	// GK formula: σ² = 0.5*ln(H/L)² - (2ln2-1)*ln(C/O)²
	// More efficient estimator than Parkinson (uses all OHLC data)
	high := bar.High
	low := bar.Low
	open := bar.Open
	close := bar.Close

	// Safe division guards
	if low < Epsilon {
		low = Epsilon
	}
	if high < low {
		high = low
	}
	if open < Epsilon {
		open = Epsilon
	}
	if close < Epsilon {
		close = Epsilon
	}

	logHL := math.Log(high / low)
	logCO := math.Log(close / open)

	// GK term for this bar
	gkTerm := 0.5*logHL*logHL - (2*math.Ln2-1)*logCO*logCO

	// Update rolling sum
	old, wasFull := r.gkBuffer.Push(gkTerm)
	if wasFull {
		r.gkSum -= old
	}
	r.gkSum += gkTerm

	// GK volatility: sqrt(mean of GK terms)
	n := float64(r.gkBuffer.count)
	if n > 0 {
		// Clamp to avoid negative values from numerical errors
		meanGK := r.gkSum / n
		if meanGK < 0 {
			meanGK = 0
		}
		r.current.GarmanKlassVol = math.Sqrt(meanGK)
	}

	// === Realized Volatility (close-to-close) ===
	var ret float64
	if r.prevClose > Epsilon {
		ret = (bar.Close - r.prevClose) / r.prevClose
	}
	r.prevClose = bar.Close
	r.prevOpen = bar.Open

	// Push to return buffer for realized vol
	r.returnBuffer.Push(ret)

	// Calculate realized vol (std of returns)
	if r.returnBuffer.count >= 2 {
		returns := r.returnBuffer.Values()
		r.current.RealizedVol = stdDev(returns)
	}

	// === Vol Ratio ===
	if r.current.RealizedVol > Epsilon {
		r.current.VolRatio = r.current.GarmanKlassVol / r.current.RealizedVol
	}

	// === Shannon Entropy (Vol-Normalized) ===
	// Normalize returns by realized vol to measure distribution "shape"
	// rather than spread. This decouples entropy from volatility.
	scaledReturn := ret
	if r.current.RealizedVol > 1e-8 {
		scaledReturn = ret / r.current.RealizedVol
	}
	r.entropyBuffer.Push(scaledReturn)
	if r.entropyBuffer.IsFull() {
		r.current.ShannonEntropy = r.calcEntropy(r.entropyBuffer.Values())
	}

	// === Higher Moments (Skewness & Kurtosis) ===
	r.momentAccum.Update(ret)
	r.current.Skewness = r.momentAccum.Skewness()
	r.current.Kurtosis = r.momentAccum.Kurtosis()

	// === Z-Scores (EWM) ===
	r.current.VolZScore = r.volNormalizer.Update(r.current.GarmanKlassVol)
	r.current.EntropyZScore = r.entropyNormalizer.Update(r.current.ShannonEntropy)

	return r.current
}

// calcEntropy calculates normalized Shannon entropy of returns.
func (r *Regime) calcEntropy(returns []float64) float64 {
	if len(returns) < 2 {
		return 0
	}

	// Find min/max for binning
	minVal, maxVal := returns[0], returns[0]
	for _, v := range returns {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	// Handle case where all values are the same
	rangeVal := maxVal - minVal
	if rangeVal < Epsilon {
		return 0 // All same values = zero entropy
	}

	// Create histogram
	bins := make([]int, r.config.EntropyBins)
	binWidth := rangeVal / float64(r.config.EntropyBins)

	for _, v := range returns {
		binIdx := int((v - minVal) / binWidth)
		if binIdx >= r.config.EntropyBins {
			binIdx = r.config.EntropyBins - 1
		}
		if binIdx < 0 {
			binIdx = 0
		}
		bins[binIdx]++
	}

	// Calculate entropy: -sum(p * log(p))
	n := float64(len(returns))
	var entropy float64
	for _, count := range bins {
		if count > 0 {
			p := float64(count) / n
			entropy -= p * math.Log(p)
		}
	}

	// Normalize by max entropy (log(bins))
	maxEntropy := math.Log(float64(r.config.EntropyBins))
	if maxEntropy > 0 {
		entropy /= maxEntropy
	}

	return entropy
}

// stdDev calculates sample standard deviation.
func stdDev(values []float64) float64 {
	if len(values) < 2 {
		return 0
	}

	// Calculate mean
	var sum float64
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))

	// Calculate variance
	var sumSq float64
	for _, v := range values {
		diff := v - mean
		sumSq += diff * diff
	}

	variance := sumSq / float64(len(values)-1)
	return math.Sqrt(variance)
}

// CurrentFeatures returns the current feature values.
func (r *Regime) CurrentFeatures() RegimeFeatures {
	return r.current
}

// IsPrimed returns true if all normalizers are warmed up.
func (r *Regime) IsPrimed() bool {
	return r.volNormalizer.IsPrimed() && r.entropyNormalizer.IsPrimed()
}

// Reset clears all state.
func (r *Regime) Reset() {
	r.gkBuffer.Reset()
	r.returnBuffer.Reset()
	r.entropyBuffer.Reset()
	r.gkSum = 0
	r.prevClose = 0
	r.prevOpen = 0
	r.momentAccum.Reset()
	r.volNormalizer.Reset()
	r.entropyNormalizer.Reset()
	r.current = RegimeFeatures{}
	r.count = 0
}

// RegimeState represents serializable state.
type RegimeState struct {
	GKBuffer []float64
	GKSum    float64
	GKIndex  int
	GKCount  int

	ReturnBuffer []float64
	ReturnIndex  int
	ReturnCount  int
	PrevClose    float64
	PrevOpen     float64

	EntropyBuffer []float64
	EntropyIndex  int
	EntropyCount  int

	MomentState      normalizer.WelfordState
	VolNormState     normalizer.EWMState
	EntropyNormState normalizer.EWMState

	Count int64
}

// State returns serializable state snapshot.
func (r *Regime) State() RegimeState {
	return RegimeState{
		GKBuffer: append([]float64{}, r.gkBuffer.data...),
		GKSum:    r.gkSum,
		GKIndex:  r.gkBuffer.index,
		GKCount:  r.gkBuffer.count,

		ReturnBuffer: append([]float64{}, r.returnBuffer.data...),
		ReturnIndex:  r.returnBuffer.index,
		ReturnCount:  r.returnBuffer.count,
		PrevClose:    r.prevClose,
		PrevOpen:     r.prevOpen,

		EntropyBuffer: append([]float64{}, r.entropyBuffer.data...),
		EntropyIndex:  r.entropyBuffer.index,
		EntropyCount:  r.entropyBuffer.count,

		MomentState:      r.momentAccum.State(),
		VolNormState:     r.volNormalizer.State(),
		EntropyNormState: r.entropyNormalizer.State(),

		Count: r.count,
	}
}

// LoadState restores from serialized state.
func (r *Regime) LoadState(state RegimeState) {
	copy(r.gkBuffer.data, state.GKBuffer)
	r.gkSum = state.GKSum
	r.gkBuffer.index = state.GKIndex
	r.gkBuffer.count = state.GKCount

	copy(r.returnBuffer.data, state.ReturnBuffer)
	r.returnBuffer.index = state.ReturnIndex
	r.returnBuffer.count = state.ReturnCount
	r.prevClose = state.PrevClose
	r.prevOpen = state.PrevOpen

	copy(r.entropyBuffer.data, state.EntropyBuffer)
	r.entropyBuffer.index = state.EntropyIndex
	r.entropyBuffer.count = state.EntropyCount

	r.momentAccum.LoadState(state.MomentState)
	r.volNormalizer.LoadState(state.VolNormState)
	r.entropyNormalizer.LoadState(state.EntropyNormState)

	r.count = state.Count
}
