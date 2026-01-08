package normalizer

import "math"

// EWM implements Exponentially Weighted Moving statistics for
// streaming z-score normalization.
//
// EWM gives more weight to recent observations, with older values
// decaying exponentially. This is ideal for financial time series
// where recent data is more relevant than distant history.
//
// Key properties:
// - Adapts to changing distributions (regime shifts)
// - No fixed lookback window (infinite memory with decay)
// - Single-pass O(1) computation per update
// - Compatible with pandas ewm(adjust=False)
//
// The decay rate is controlled by alpha (0 < alpha <= 1):
// - Higher alpha = faster decay, more responsive
// - Lower alpha = slower decay, smoother
type EWM struct {
	alpha float64 // decay factor
	mean  float64 // exponentially weighted mean
	var_  float64 // exponentially weighted variance
	count int64
}

// NewEWMWithHalflife creates an EWM normalizer using halflife parameter.
// Halflife is the number of periods for the weight to decay by half.
//
// alpha = 1 - exp(-ln(2) / halflife)
//
// Example halflife values:
// - 10: Fast adaptation, weight halves every 10 periods
// - 50: Medium adaptation (common choice)
// - 100: Slow adaptation, smoother statistics
func NewEWMWithHalflife(halflife float64) *EWM {
	if halflife <= 0 {
		halflife = 1 // minimum halflife
	}
	alpha := 1 - math.Exp(-math.Ln2/halflife)
	return &EWM{alpha: alpha}
}

// NewEWMWithSpan creates an EWM normalizer using span parameter.
// Span is the "center of mass" decay parameter.
//
// alpha = 2 / (span + 1)
//
// This matches pandas ewm(span=N) behavior.
func NewEWMWithSpan(span float64) *EWM {
	if span < 1 {
		span = 1
	}
	alpha := 2.0 / (span + 1)
	return &EWM{alpha: alpha}
}

// NewEWMWithAlpha creates an EWM normalizer with explicit alpha.
// Alpha must be in range (0, 1].
func NewEWMWithAlpha(alpha float64) *EWM {
	if alpha <= 0 {
		alpha = 0.01
	}
	if alpha > 1 {
		alpha = 1
	}
	return &EWM{alpha: alpha}
}

// Update adds a new value and returns its z-score.
// This is the primary streaming method.
func (e *EWM) Update(value float64) float64 {
	e.count++

	if e.count == 1 {
		// First value: initialize mean, variance = 0
		e.mean = value
		e.var_ = 0
		return 0 // z-score undefined for first value
	}

	// Update EWM mean
	delta := value - e.mean
	e.mean += e.alpha * delta

	// Update EWM variance using Finch's algorithm
	// This is numerically stable and matches the recursive formula:
	// var_t = (1-alpha) * (var_{t-1} + alpha * delta^2)
	e.var_ = (1 - e.alpha) * (e.var_ + e.alpha*delta*delta)

	// Compute z-score
	return e.zscore(value)
}

// zscore computes z-score using current statistics.
// Applies soft clipping (Tanh-based) to prevent gradient explosion in ML models.
// Output range is approximately [-5, 5] with smooth compression at tails.
func (e *EWM) zscore(value float64) float64 {
	std := e.Std()
	if std < 1e-10 {
		return 0
	}
	rawZ := (value - e.mean) / std
	return softClip(rawZ)
}

// softClip applies Tanh-based soft clipping to constrain z-scores.
// Properties:
// - For |x| < 3: nearly linear (preserves information)
// - For |x| > 5: smoothly approaches ±5 (prevents outliers)
// - Differentiable everywhere (gradient-friendly)
func softClip(x float64) float64 {
	const bound = 5.0
	return bound * math.Tanh(x/bound)
}

// Alpha returns the decay factor.
func (e *EWM) Alpha() float64 {
	return e.alpha
}

// Mean returns the current exponentially weighted mean.
func (e *EWM) Mean() float64 {
	return e.mean
}

// Var returns the current exponentially weighted variance.
func (e *EWM) Var() float64 {
	return e.var_
}

// Std returns the current exponentially weighted standard deviation.
func (e *EWM) Std() float64 {
	if e.var_ < 0 {
		return 0 // numerical safety
	}
	return math.Sqrt(e.var_)
}

// Count returns the number of values processed.
func (e *EWM) Count() int64 {
	return e.count
}

// IsPrimed returns true if enough samples have been processed
// for stable statistics. Rule of thumb: need ~2/alpha samples.
func (e *EWM) IsPrimed() bool {
	minSamples := int64(2.0 / e.alpha)
	if minSamples < 10 {
		minSamples = 10
	}
	return e.count >= minSamples
}

// Reset clears all state.
func (e *EWM) Reset() {
	e.mean = 0
	e.var_ = 0
	e.count = 0
}

// State returns a snapshot for serialization.
func (e *EWM) State() EWMState {
	return EWMState{
		Alpha: e.alpha,
		Mean:  e.mean,
		Var:   e.var_,
		Count: e.count,
	}
}

// LoadState restores from a snapshot.
func (e *EWM) LoadState(state EWMState) {
	e.alpha = state.Alpha
	e.mean = state.Mean
	e.var_ = state.Var
	e.count = state.Count
}

// EWMState represents a serializable snapshot of EWM state.
type EWMState struct {
	Alpha float64 `json:"alpha"`
	Mean  float64 `json:"mean"`
	Var   float64 `json:"var"`
	Count int64   `json:"count"`
}
