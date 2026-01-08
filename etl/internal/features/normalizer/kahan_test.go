package normalizer

import (
	"math"
	"testing"
)

func TestKahanSum_Basic(t *testing.T) {
	t.Run("single value", func(t *testing.T) {
		k := NewKahanSum()
		k.Add(42.0)

		if k.Value() != 42.0 {
			t.Errorf("expected 42.0, got %v", k.Value())
		}
	})

	t.Run("multiple values", func(t *testing.T) {
		k := NewKahanSum()
		k.Add(1.0)
		k.Add(2.0)
		k.Add(3.0)

		if k.Value() != 6.0 {
			t.Errorf("expected 6.0, got %v", k.Value())
		}
	})

	t.Run("subtraction", func(t *testing.T) {
		k := NewKahanSum()
		k.Add(10.0)
		k.Sub(3.0)

		if k.Value() != 7.0 {
			t.Errorf("expected 7.0, got %v", k.Value())
		}
	})
}

func TestKahanSum_Reset(t *testing.T) {
	k := NewKahanSum()
	k.Add(100.0)
	k.Add(200.0)
	k.Reset(50.0)

	if k.Value() != 50.0 {
		t.Errorf("expected 50.0 after reset, got %v", k.Value())
	}

	// Correction should also be reset
	k.Add(0.1)
	if math.Abs(k.Value()-50.1) > 1e-10 {
		t.Errorf("expected 50.1, got %v", k.Value())
	}
}

func TestKahanSum_FloatingPointPrecision(t *testing.T) {
	// Classic example: 0.1 + 0.2 + 0.3 != 0.6 in naive floating point
	t.Run("small fractions", func(t *testing.T) {
		k := NewKahanSum()
		for i := 0; i < 10; i++ {
			k.Add(0.1)
		}

		// Should be very close to 1.0
		if math.Abs(k.Value()-1.0) > 1e-14 {
			t.Errorf("expected ~1.0, got %v, diff: %e", k.Value(), math.Abs(k.Value()-1.0))
		}
	})

	t.Run("large number of additions", func(t *testing.T) {
		k := NewKahanSum()
		n := 1000000
		val := 0.1

		for i := 0; i < n; i++ {
			k.Add(val)
		}

		expected := float64(n) * val
		diff := math.Abs(k.Value() - expected)

		// Kahan should maintain precision much better than naive sum
		if diff > 1e-9 {
			t.Errorf("precision loss too high: expected %v, got %v, diff: %e", expected, k.Value(), diff)
		}
	})
}

func TestKahanSum_CatastrophicCancellation(t *testing.T) {
	// Test the scenario that causes problems in naive rolling sums
	// Add large values then subtract them
	k := NewKahanSum()

	// Add 1 million, then subtract 1 million small amounts
	largeVal := 1e10
	smallVal := 1e-5
	n := 1000000

	k.Add(largeVal)
	for i := 0; i < n; i++ {
		k.Add(smallVal)
	}
	for i := 0; i < n; i++ {
		k.Sub(smallVal)
	}

	// Should be back to largeVal
	diff := math.Abs(k.Value() - largeVal)
	if diff > 1e-4 {
		t.Errorf("catastrophic cancellation: expected %v, got %v, diff: %e", largeVal, k.Value(), diff)
	}
}

func TestKahanSum_NegativeValues(t *testing.T) {
	k := NewKahanSum()
	k.Add(-5.0)
	k.Add(-3.0)
	k.Add(10.0)

	if k.Value() != 2.0 {
		t.Errorf("expected 2.0, got %v", k.Value())
	}
}

func TestKahanSum_ZeroValue(t *testing.T) {
	k := NewKahanSum()

	if k.Value() != 0.0 {
		t.Errorf("expected 0.0 for new KahanSum, got %v", k.Value())
	}

	k.Add(0.0)
	if k.Value() != 0.0 {
		t.Errorf("expected 0.0 after adding 0, got %v", k.Value())
	}
}

func TestKahanSum_EdgeCases(t *testing.T) {
	t.Run("very small values", func(t *testing.T) {
		k := NewKahanSum()
		for i := 0; i < 1000; i++ {
			k.Add(1e-15)
		}
		expected := 1e-12
		if math.Abs(k.Value()-expected) > 1e-20 {
			t.Errorf("expected %e, got %e", expected, k.Value())
		}
	})

	t.Run("alternating signs", func(t *testing.T) {
		k := NewKahanSum()
		for i := 0; i < 10000; i++ {
			if i%2 == 0 {
				k.Add(1.0)
			} else {
				k.Sub(1.0)
			}
		}
		if k.Value() != 0.0 {
			t.Errorf("expected 0.0 for alternating sum, got %v", k.Value())
		}
	})

	t.Run("infinity handling", func(t *testing.T) {
		k := NewKahanSum()
		k.Add(math.Inf(1))

		if !math.IsInf(k.Value(), 1) {
			t.Errorf("expected +Inf, got %v", k.Value())
		}
	})

	t.Run("NaN handling", func(t *testing.T) {
		k := NewKahanSum()
		k.Add(math.NaN())

		if !math.IsNaN(k.Value()) {
			t.Errorf("expected NaN, got %v", k.Value())
		}
	})
}

// Benchmark to compare Kahan vs naive sum
func BenchmarkKahanSum(b *testing.B) {
	for i := 0; i < b.N; i++ {
		k := NewKahanSum()
		for j := 0; j < 10000; j++ {
			k.Add(0.1)
		}
		_ = k.Value()
	}
}

func BenchmarkNaiveSum(b *testing.B) {
	for i := 0; i < b.N; i++ {
		var sum float64
		for j := 0; j < 10000; j++ {
			sum += 0.1
		}
		_ = sum
	}
}
