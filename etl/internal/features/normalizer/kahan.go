// Package normalizer provides stateful normalization algorithms
// for streaming financial data processing.
package normalizer

// KahanSum implements the Kahan summation algorithm for accurate
// floating-point accumulation. It compensates for the loss of
// significance that occurs with naive summation.
//
// The algorithm maintains a running compensation for accumulated
// rounding errors, providing much better precision than naive
// summation especially when:
// - Summing many small values
// - Adding/subtracting values repeatedly (catastrophic cancellation)
// - Mixing values of very different magnitudes
//
// Reference: https://en.wikipedia.org/wiki/Kahan_summation_algorithm
type KahanSum struct {
	sum        float64
	correction float64 // running compensation for lost low-order bits
}

// NewKahanSum creates a new KahanSum initialized to zero.
func NewKahanSum() *KahanSum {
	return &KahanSum{}
}

// Add adds a value to the sum with error compensation.
func (k *KahanSum) Add(value float64) {
	// Apply correction to the value being added
	y := value - k.correction

	// sum is big, y is small, so low-order bits of y are lost
	t := k.sum + y

	// (t - k.sum) recovers the high-order part of y;
	// subtracting y recovers -(low part of y)
	k.correction = (t - k.sum) - y

	k.sum = t
}

// Sub subtracts a value from the sum with error compensation.
func (k *KahanSum) Sub(value float64) {
	k.Add(-value)
}

// Value returns the current sum.
func (k *KahanSum) Value() float64 {
	return k.sum
}

// Reset resets the sum to a specific value and clears the correction.
func (k *KahanSum) Reset(value float64) {
	k.sum = value
	k.correction = 0
}

// Clear resets the sum to zero.
func (k *KahanSum) Clear() {
	k.sum = 0
	k.correction = 0
}
