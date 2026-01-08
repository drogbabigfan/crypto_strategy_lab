package normalizer

import "math"

const (
	// DefaultRecalcInterval is the default number of updates between
	// full recalculations to prevent floating-point drift.
	DefaultRecalcInterval = 10000
)

// Rolling implements a rolling window z-score normalizer using a ring buffer.
//
// This normalizer maintains statistics over a fixed-size sliding window,
// which is useful when you want equal weight to all recent observations
// and a hard cutoff for older data.
//
// Key features:
// - Fixed memory: O(window) regardless of total data processed
// - Kahan summation: Prevents floating-point accumulation errors
// - Periodic recalculation: Eliminates any remaining drift
// - Ring buffer: Efficient O(1) updates
//
// Use cases:
// - When you want exactly N recent observations
// - When older data should have zero influence
// - When you need strict lookback guarantees
type Rolling struct {
	window        int
	buffer        []float64
	index         int // current write position in ring buffer
	count         int // number of elements currently in buffer (up to window)
	updateCount   int64
	recalcCount   int64
	recalcInterval int64

	// Kahan summation for precision
	sum   *KahanSum
	sumSq *KahanSum
}

// NewRolling creates a new rolling normalizer with the specified window size.
func NewRolling(window int) *Rolling {
	return NewRollingWithRecalcInterval(window, DefaultRecalcInterval)
}

// NewRollingWithRecalcInterval creates a rolling normalizer with custom
// recalculation interval.
func NewRollingWithRecalcInterval(window int, recalcInterval int64) *Rolling {
	if window < 1 {
		window = 1
	}
	if recalcInterval < 100 {
		recalcInterval = 100
	}

	return &Rolling{
		window:         window,
		buffer:         make([]float64, window),
		recalcInterval: recalcInterval,
		sum:            NewKahanSum(),
		sumSq:          NewKahanSum(),
	}
}

// Update adds a new value and returns its z-score.
// This is the primary streaming method.
func (r *Rolling) Update(value float64) float64 {
	r.updateCount++

	// Periodic recalculation to eliminate floating-point drift
	if r.updateCount%r.recalcInterval == 0 {
		r.recalculate()
	}

	// Remove oldest value if window is full
	if r.count >= r.window {
		old := r.buffer[r.index]
		r.sum.Sub(old)
		r.sumSq.Sub(old * old)
	} else {
		r.count++
	}

	// Add new value
	r.buffer[r.index] = value
	r.sum.Add(value)
	r.sumSq.Add(value * value)

	// Advance ring buffer index
	r.index = (r.index + 1) % r.window

	return r.zscore(value)
}

// recalculate performs a full recalculation of sum and sumSq
// from the buffer to eliminate accumulated floating-point errors.
func (r *Rolling) recalculate() {
	r.recalcCount++

	var sum, sumSq float64
	for i := 0; i < r.count; i++ {
		v := r.buffer[i]
		sum += v
		sumSq += v * v
	}

	r.sum.Reset(sum)
	r.sumSq.Reset(sumSq)
}

// zscore computes z-score using current statistics.
func (r *Rolling) zscore(value float64) float64 {
	if r.count < 2 {
		return 0
	}

	std := r.Std()
	if std < 1e-10 {
		return 0
	}

	return (value - r.Mean()) / std
}

// ZScore computes z-score for an arbitrary value using current statistics.
func (r *Rolling) ZScore(value float64) float64 {
	return r.zscore(value)
}

// Window returns the window size.
func (r *Rolling) Window() int {
	return r.window
}

// Count returns the current number of elements in the window.
func (r *Rolling) Count() int {
	return r.count
}

// Mean returns the mean of values in the current window.
func (r *Rolling) Mean() float64 {
	if r.count == 0 {
		return 0
	}
	return r.sum.Value() / float64(r.count)
}

// Variance returns the sample variance of values in the current window.
func (r *Rolling) Variance() float64 {
	if r.count < 2 {
		return 0
	}

	n := float64(r.count)
	mean := r.Mean()

	// Var = E[X^2] - E[X]^2 (population variance)
	// Sample variance = n/(n-1) * population variance
	variance := (r.sumSq.Value() / n) - (mean * mean)

	// Handle numerical issues that could make variance slightly negative
	if variance < 0 {
		variance = 0
	}

	// Convert to sample variance (Bessel's correction)
	return variance * n / (n - 1)
}

// PopulationVariance returns the population variance (no Bessel correction).
func (r *Rolling) PopulationVariance() float64 {
	if r.count < 1 {
		return 0
	}

	n := float64(r.count)
	mean := r.Mean()
	variance := (r.sumSq.Value() / n) - (mean * mean)

	if variance < 0 {
		variance = 0
	}

	return variance
}

// Std returns the sample standard deviation.
func (r *Rolling) Std() float64 {
	return math.Sqrt(r.Variance())
}

// IsPrimed returns true if the window is full.
func (r *Rolling) IsPrimed() bool {
	return r.count >= r.window
}

// DriftError returns the current accumulated floating-point error.
// This is useful for debugging and monitoring.
func (r *Rolling) DriftError() float64 {
	// Calculate true sum from buffer
	var trueSum float64
	for i := 0; i < r.count; i++ {
		trueSum += r.buffer[i]
	}

	return math.Abs(trueSum - r.sum.Value())
}

// RecalcCount returns the number of recalculations performed.
func (r *Rolling) RecalcCount() int64 {
	return r.recalcCount
}

// Reset clears all state.
func (r *Rolling) Reset() {
	r.index = 0
	r.count = 0
	r.updateCount = 0
	r.recalcCount = 0
	r.sum.Clear()
	r.sumSq.Clear()

	// Clear buffer
	for i := range r.buffer {
		r.buffer[i] = 0
	}
}

// State returns a snapshot for serialization.
func (r *Rolling) State() RollingState {
	// Copy buffer
	bufCopy := make([]float64, len(r.buffer))
	copy(bufCopy, r.buffer)

	return RollingState{
		Window:        r.window,
		Buffer:        bufCopy,
		Index:         r.index,
		Count:         r.count,
		UpdateCount:   r.updateCount,
		RecalcCount:   r.recalcCount,
		Sum:           r.sum.Value(),
		SumSq:         r.sumSq.Value(),
	}
}

// LoadState restores from a snapshot.
func (r *Rolling) LoadState(state RollingState) {
	// Resize buffer if needed
	if len(r.buffer) != state.Window {
		r.buffer = make([]float64, state.Window)
	}
	r.window = state.Window

	copy(r.buffer, state.Buffer)
	r.index = state.Index
	r.count = state.Count
	r.updateCount = state.UpdateCount
	r.recalcCount = state.RecalcCount
	r.sum.Reset(state.Sum)
	r.sumSq.Reset(state.SumSq)
}

// RollingState represents a serializable snapshot of Rolling state.
type RollingState struct {
	Window        int       `json:"window"`
	Buffer        []float64 `json:"buffer"`
	Index         int       `json:"index"`
	Count         int       `json:"count"`
	UpdateCount   int64     `json:"update_count"`
	RecalcCount   int64     `json:"recalc_count"`
	Sum           float64   `json:"sum"`
	SumSq         float64   `json:"sum_sq"`
}
