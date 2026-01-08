package normalizer

import "math"

// Welford implements Welford's online algorithm for computing
// running mean, variance, skewness, and kurtosis in a numerically stable way.
//
// The algorithm updates statistics incrementally with each new value,
// avoiding the need to store all values and preventing catastrophic
// cancellation that occurs with naive algorithms.
//
// This is the preferred algorithm for streaming statistics because:
// - Single pass: O(1) memory, O(1) per update
// - Numerically stable: No catastrophic cancellation
// - Accurate: Mathematically equivalent to batch computation
// - Higher moments: Skewness (3rd) and Kurtosis (4th) included
//
// Reference:
// - Welford, B. P. (1962). "Note on a method for calculating corrected sums of squares and products"
// - Terriberry (2007). "Computing Higher-Order Moments Online"
type Welford struct {
	count      int64
	mean       float64
	m2         float64 // sum of squared differences from mean
	m3         float64 // sum of cubed differences from mean (for skewness)
	m4         float64 // sum of 4th power differences (for kurtosis)
	minSamples int64   // minimum samples before considered "primed"
}

// NewWelford creates a new Welford accumulator.
func NewWelford() *Welford {
	return &Welford{
		minSamples: 2, // default: need at least 2 for variance
	}
}

// NewWelfordWithMinSamples creates a Welford accumulator with a
// custom minimum sample count for the IsPrimed check.
func NewWelfordWithMinSamples(minSamples int64) *Welford {
	if minSamples < 2 {
		minSamples = 2
	}
	return &Welford{
		minSamples: minSamples,
	}
}

// Update adds a new value and updates the running statistics.
// Uses Terriberry's algorithm for higher moments.
func (w *Welford) Update(value float64) {
	n1 := w.count
	w.count++
	n := w.count

	delta := value - w.mean
	deltaN := delta / float64(n)
	deltaN2 := deltaN * deltaN
	term1 := delta * deltaN * float64(n1)

	w.mean += deltaN

	// Update M4 before M3 and M2 (order matters for numerical stability)
	w.m4 += term1*deltaN2*(float64(n*n)-3*float64(n)+3) +
		6*deltaN2*w.m2 - 4*deltaN*w.m3

	w.m3 += term1*deltaN*(float64(n)-2) - 3*deltaN*w.m2

	w.m2 += term1
}

// UpdateAndGet adds a new value and returns the z-score of that value.
// This is the primary method for streaming normalization.
func (w *Welford) UpdateAndGet(value float64) float64 {
	w.Update(value)
	return w.ZScore(value)
}

// Count returns the number of values seen so far.
func (w *Welford) Count() int64 {
	return w.count
}

// Mean returns the current running mean.
func (w *Welford) Mean() float64 {
	return w.mean
}

// Variance returns the sample variance (Bessel's correction: n-1).
// Returns 0 if count < 2.
func (w *Welford) Variance() float64 {
	if w.count < 2 {
		return 0
	}
	return w.m2 / float64(w.count-1)
}

// PopulationVariance returns the population variance (divided by n).
// Returns 0 if count < 1.
func (w *Welford) PopulationVariance() float64 {
	if w.count < 1 {
		return 0
	}
	return w.m2 / float64(w.count)
}

// Std returns the sample standard deviation.
func (w *Welford) Std() float64 {
	return math.Sqrt(w.Variance())
}

// Skewness returns the sample skewness (Fisher's definition).
// Positive skewness = right tail is longer (more extreme positive values)
// Negative skewness = left tail is longer (more extreme negative values)
// Returns 0 if count < 3.
func (w *Welford) Skewness() float64 {
	if w.count < 3 {
		return 0
	}
	if w.m2 < 1e-14 {
		return 0 // No variance, no skewness
	}
	n := float64(w.count)
	// Fisher's sample skewness with bias correction
	return math.Sqrt(n*(n-1)) / (n - 2) * w.m3 / math.Pow(w.m2, 1.5) * math.Sqrt(n)
}

// Kurtosis returns the excess kurtosis (Fisher's definition).
// Excess kurtosis = kurtosis - 3 (so normal distribution has excess kurtosis of 0)
// Positive = fat tails (leptokurtic), Negative = thin tails (platykurtic)
// Returns 0 if count < 4.
func (w *Welford) Kurtosis() float64 {
	if w.count < 4 {
		return 0
	}
	if w.m2 < 1e-14 {
		return 0 // No variance
	}
	n := float64(w.count)
	// Fisher's sample excess kurtosis with bias correction
	kurtosis := (n*n-1)/((n-2)*(n-3))*(n*w.m4/(w.m2*w.m2)-3) + 3*(n-1)*(n-1)/((n-2)*(n-3))
	return kurtosis - 3 // excess kurtosis
}

// ZScore computes the z-score of a value using current statistics.
// Returns 0 if standard deviation is effectively zero.
func (w *Welford) ZScore(value float64) float64 {
	std := w.Std()
	if std < 1e-10 {
		return 0
	}
	return (value - w.mean) / std
}

// IsPrimed returns true if enough samples have been collected
// for stable statistics.
func (w *Welford) IsPrimed() bool {
	return w.count >= w.minSamples
}

// Reset clears all accumulated statistics.
func (w *Welford) Reset() {
	w.count = 0
	w.mean = 0
	w.m2 = 0
	w.m3 = 0
	w.m4 = 0
}

// State returns a snapshot of the current state for serialization.
func (w *Welford) State() WelfordState {
	return WelfordState{
		Count:      w.count,
		Mean:       w.mean,
		M2:         w.m2,
		M3:         w.m3,
		M4:         w.m4,
		MinSamples: w.minSamples,
	}
}

// LoadState restores state from a snapshot.
func (w *Welford) LoadState(state WelfordState) {
	w.count = state.Count
	w.mean = state.Mean
	w.m2 = state.M2
	w.m3 = state.M3
	w.m4 = state.M4
	w.minSamples = state.MinSamples
}

// WelfordState represents a serializable snapshot of Welford state.
type WelfordState struct {
	Count      int64   `json:"count"`
	Mean       float64 `json:"mean"`
	M2         float64 `json:"m2"`
	M3         float64 `json:"m3"`
	M4         float64 `json:"m4"`
	MinSamples int64   `json:"min_samples"`
}
