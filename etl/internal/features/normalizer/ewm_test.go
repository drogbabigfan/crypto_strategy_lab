package normalizer

import (
	"math"
	"math/rand"
	"testing"
)

func TestEWM_NewWithHalflife(t *testing.T) {
	t.Run("halflife 10", func(t *testing.T) {
		e := NewEWMWithHalflife(10)

		// alpha = 1 - exp(-ln(2)/halflife) = 1 - exp(-0.0693) ≈ 0.067
		expectedAlpha := 1 - math.Exp(-math.Ln2/10)
		if math.Abs(e.Alpha()-expectedAlpha) > 1e-10 {
			t.Errorf("expected alpha %v, got %v", expectedAlpha, e.Alpha())
		}
	})

	t.Run("halflife 50", func(t *testing.T) {
		e := NewEWMWithHalflife(50)

		expectedAlpha := 1 - math.Exp(-math.Ln2/50)
		if math.Abs(e.Alpha()-expectedAlpha) > 1e-10 {
			t.Errorf("expected alpha %v, got %v", expectedAlpha, e.Alpha())
		}
	})
}

func TestEWM_NewWithSpan(t *testing.T) {
	t.Run("span 10", func(t *testing.T) {
		e := NewEWMWithSpan(10)

		// alpha = 2 / (span + 1)
		expectedAlpha := 2.0 / 11.0
		if math.Abs(e.Alpha()-expectedAlpha) > 1e-10 {
			t.Errorf("expected alpha %v, got %v", expectedAlpha, e.Alpha())
		}
	})
}

func TestEWM_Basic(t *testing.T) {
	t.Run("first value", func(t *testing.T) {
		e := NewEWMWithHalflife(10)

		z := e.Update(100.0)

		// First value: mean = value, var = 0, zscore = 0
		if z != 0 {
			t.Errorf("first zscore should be 0, got %v", z)
		}
		if e.Mean() != 100.0 {
			t.Errorf("mean should be 100.0, got %v", e.Mean())
		}
		if e.Count() != 1 {
			t.Errorf("count should be 1, got %d", e.Count())
		}
	})

	t.Run("second value", func(t *testing.T) {
		e := NewEWMWithHalflife(10)
		e.Update(100.0)
		z := e.Update(110.0)

		// After 2 values, mean moves toward 110, zscore should be positive
		if e.Mean() <= 100 || e.Mean() >= 110 {
			t.Errorf("mean should be between 100 and 110, got %v", e.Mean())
		}
		// 110 > mean, so zscore should be positive
		if z <= 0 {
			t.Errorf("zscore for above-mean value should be positive, got %v", z)
		}
	})

	t.Run("convergence to constant", func(t *testing.T) {
		e := NewEWMWithHalflife(10)

		// Feed many constant values
		for i := 0; i < 1000; i++ {
			e.Update(50.0)
		}

		// Mean should converge to 50
		if math.Abs(e.Mean()-50.0) > 1e-6 {
			t.Errorf("mean should converge to 50, got %v", e.Mean())
		}

		// Variance should converge to 0
		if e.Var() > 1e-6 {
			t.Errorf("variance should converge to 0, got %v", e.Var())
		}
	})
}

func TestEWM_ZScore(t *testing.T) {
	t.Run("stable zscore distribution", func(t *testing.T) {
		e := NewEWMWithHalflife(50)
		rng := rand.New(rand.NewSource(42))

		// Feed standard normal values
		var zscores []float64
		for i := 0; i < 10000; i++ {
			z := e.Update(rng.NormFloat64())
			if i > 200 { // After warmup
				zscores = append(zscores, z)
			}
		}

		// Calculate mean and std of zscores
		var sum, sumSq float64
		for _, z := range zscores {
			sum += z
			sumSq += z * z
		}
		n := float64(len(zscores))
		zMean := sum / n
		zVar := sumSq/n - zMean*zMean

		// Z-scores should have mean ~0 and variance ~1
		// (with some tolerance due to EWM adaptation)
		if math.Abs(zMean) > 0.1 {
			t.Errorf("zscore mean should be ~0, got %v", zMean)
		}
		if zVar < 0.5 || zVar > 2.0 {
			t.Errorf("zscore variance should be ~1, got %v", zVar)
		}
	})
}

func TestEWM_IsPrimed(t *testing.T) {
	e := NewEWMWithHalflife(50) // needs ~100 samples to prime (2/alpha)

	// Initially not primed
	if e.IsPrimed() {
		t.Error("should not be primed initially")
	}

	// Feed values
	for i := 0; i < 50; i++ {
		e.Update(float64(i))
	}
	if e.IsPrimed() {
		t.Error("should not be primed with only 50 samples for halflife=50")
	}

	// After enough samples
	for i := 0; i < 100; i++ {
		e.Update(float64(i))
	}
	if !e.IsPrimed() {
		t.Error("should be primed with 150 samples")
	}
}

func TestEWM_Responsiveness(t *testing.T) {
	t.Run("shorter halflife is more responsive", func(t *testing.T) {
		e1 := NewEWMWithHalflife(10)  // fast
		e2 := NewEWMWithHalflife(100) // slow

		// Initialize with same value
		for i := 0; i < 100; i++ {
			e1.Update(100.0)
			e2.Update(100.0)
		}

		// Sudden change
		e1.Update(200.0)
		e2.Update(200.0)

		// Faster EWM should adapt more
		if e1.Mean() <= e2.Mean() {
			t.Errorf("shorter halflife should be more responsive: e1.Mean=%v, e2.Mean=%v",
				e1.Mean(), e2.Mean())
		}
	})
}

func TestEWM_EdgeCases(t *testing.T) {
	t.Run("zero halflife panics or uses minimum", func(t *testing.T) {
		// Depending on implementation, this should either panic or use minimum
		defer func() {
			if r := recover(); r != nil {
				// Expected panic, test passes
			}
		}()
		e := NewEWMWithHalflife(0)
		// If no panic, alpha should be capped
		if e.Alpha() > 1.0 || e.Alpha() < 0 {
			t.Error("invalid alpha for zero halflife")
		}
	})

	t.Run("very large values", func(t *testing.T) {
		e := NewEWMWithHalflife(10)

		for i := 0; i < 100; i++ {
			e.Update(1e15 + float64(i))
		}

		// Should still compute valid mean
		if math.IsNaN(e.Mean()) || math.IsInf(e.Mean(), 0) {
			t.Error("mean should be valid for large values")
		}
	})

	t.Run("alternating values", func(t *testing.T) {
		e := NewEWMWithHalflife(10)

		for i := 0; i < 1000; i++ {
			if i%2 == 0 {
				e.Update(100.0)
			} else {
				e.Update(-100.0)
			}
		}

		// Mean should be close to 0
		if math.Abs(e.Mean()) > 10 {
			t.Errorf("mean should be ~0 for alternating values, got %v", e.Mean())
		}
	})

	t.Run("infinity input", func(t *testing.T) {
		e := NewEWMWithHalflife(10)
		e.Update(100.0)
		e.Update(math.Inf(1))

		if !math.IsInf(e.Mean(), 1) {
			t.Errorf("expected +Inf mean after +Inf input, got %v", e.Mean())
		}
	})

	t.Run("NaN input", func(t *testing.T) {
		e := NewEWMWithHalflife(10)
		e.Update(100.0)
		e.Update(math.NaN())

		if !math.IsNaN(e.Mean()) {
			t.Errorf("expected NaN mean after NaN input, got %v", e.Mean())
		}
	})
}

func TestEWM_Reset(t *testing.T) {
	e := NewEWMWithHalflife(10)

	for i := 0; i < 100; i++ {
		e.Update(float64(i))
	}

	e.Reset()

	if e.Count() != 0 {
		t.Errorf("expected count 0 after reset, got %d", e.Count())
	}
	if e.Mean() != 0 {
		t.Errorf("expected mean 0 after reset, got %v", e.Mean())
	}
	if e.Var() != 0 {
		t.Errorf("expected var 0 after reset, got %v", e.Var())
	}
	if e.IsPrimed() {
		t.Error("should not be primed after reset")
	}
}

func TestEWM_StateSerialize(t *testing.T) {
	e1 := NewEWMWithHalflife(50)

	// Feed some values
	for i := 0; i < 100; i++ {
		e1.Update(float64(i))
	}

	// Save state
	state := e1.State()

	// Create new instance and load state
	e2 := NewEWMWithHalflife(50)
	e2.LoadState(state)

	// Both should produce same results
	z1 := e1.Update(200.0)
	z2 := e2.Update(200.0)

	if z1 != z2 {
		t.Errorf("zscore mismatch after state restore: %v vs %v", z1, z2)
	}
	if e1.Mean() != e2.Mean() {
		t.Errorf("mean mismatch after state restore: %v vs %v", e1.Mean(), e2.Mean())
	}
}

// Verify against pandas ewm formula
func TestEWM_PandasCompatibility(t *testing.T) {
	// pandas ewm uses adjust=True by default which gives different results
	// Our implementation matches adjust=False for streaming
	t.Run("matches pandas ewm adjust=False", func(t *testing.T) {
		e := NewEWMWithSpan(3) // alpha = 2/(3+1) = 0.5
		alpha := 0.5

		values := []float64{1, 2, 3, 4, 5}
		var expectedMean float64

		for i, v := range values {
			_ = e.Update(v)

			if i == 0 {
				expectedMean = v
			} else {
				expectedMean = alpha*v + (1-alpha)*expectedMean
			}
		}

		if math.Abs(e.Mean()-expectedMean) > 1e-10 {
			t.Errorf("mean mismatch with pandas formula: expected %v, got %v",
				expectedMean, e.Mean())
		}
	})
}

// Benchmark
func BenchmarkEWM_Update(b *testing.B) {
	e := NewEWMWithHalflife(50)
	for i := 0; i < b.N; i++ {
		_ = e.Update(float64(i))
	}
}
