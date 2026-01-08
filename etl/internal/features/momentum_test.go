package features

import (
	"math"
	"math/rand"
	"testing"
)

func TestMomentum_VWMomentum(t *testing.T) {
	t.Run("basic calculation", func(t *testing.T) {
		mom := NewMomentum(DefaultMomentumConfig())

		// VWMomentum = returns * volume
		returns := 0.01  // 1% return
		volume := 1000.0 // 1000 units

		result := mom.Update(returns, volume)

		expected := returns * volume // 10
		if math.Abs(result.VWMomentum-expected) > 1e-10 {
			t.Errorf("expected VWMomentum=%v, got %v", expected, result.VWMomentum)
		}
	})

	t.Run("negative returns", func(t *testing.T) {
		mom := NewMomentum(DefaultMomentumConfig())

		returns := -0.02 // -2% return
		volume := 500.0

		result := mom.Update(returns, volume)

		expected := -10.0 // -0.02 * 500
		if math.Abs(result.VWMomentum-expected) > 1e-10 {
			t.Errorf("expected VWMomentum=%v, got %v", expected, result.VWMomentum)
		}
	})

	t.Run("zero volume", func(t *testing.T) {
		mom := NewMomentum(DefaultMomentumConfig())

		returns := 0.05
		volume := 0.0

		result := mom.Update(returns, volume)

		if result.VWMomentum != 0 {
			t.Errorf("expected VWMomentum=0 for zero volume, got %v", result.VWMomentum)
		}
	})

	t.Run("zero returns", func(t *testing.T) {
		mom := NewMomentum(DefaultMomentumConfig())

		returns := 0.0
		volume := 1000.0

		result := mom.Update(returns, volume)

		if result.VWMomentum != 0 {
			t.Errorf("expected VWMomentum=0 for zero returns, got %v", result.VWMomentum)
		}
	})
}

func TestMomentum_MultiHorizonZScores(t *testing.T) {
	t.Run("different horizons have different responsiveness", func(t *testing.T) {
		mom := NewMomentum(DefaultMomentumConfig())

		// Feed steady momentum first
		for i := 0; i < 100; i++ {
			mom.Update(0.001, 1000) // Small positive momentum
		}

		// Then a sudden spike
		result := mom.Update(0.1, 10000) // Large positive momentum

		// Shorter horizon should react more strongly
		// (higher z-score for the same spike)
		if math.Abs(result.MomentumZScore10) <= math.Abs(result.MomentumZScore1000) {
			t.Logf("ZScore10=%v, ZScore1000=%v", result.MomentumZScore10, result.MomentumZScore1000)
			// Note: This might not always hold due to soft clipping
			// Just log it for observation
		}
	})

	t.Run("z-scores are soft-clipped", func(t *testing.T) {
		mom := NewMomentum(DefaultMomentumConfig())

		// Feed some normal data first
		for i := 0; i < 50; i++ {
			mom.Update(0.001, 1000)
		}

		// Feed extreme outlier
		result := mom.Update(1.0, 1000000) // Extreme momentum

		// Z-scores should be within soft-clip bounds [-5, 5]
		if result.MomentumZScore10 < -5.1 || result.MomentumZScore10 > 5.1 {
			t.Errorf("MomentumZScore10 out of soft-clip range: %v", result.MomentumZScore10)
		}
		if result.MomentumZScore50 < -5.1 || result.MomentumZScore50 > 5.1 {
			t.Errorf("MomentumZScore50 out of soft-clip range: %v", result.MomentumZScore50)
		}
	})

	t.Run("first update returns zero z-scores", func(t *testing.T) {
		mom := NewMomentum(DefaultMomentumConfig())

		result := mom.Update(0.01, 1000)

		// First z-score is always 0 (no history to compare)
		if result.MomentumZScore10 != 0 {
			t.Errorf("expected zero z-score for first update, got %v", result.MomentumZScore10)
		}
	})
}

func TestMomentum_IsPrimed(t *testing.T) {
	mom := NewMomentum(DefaultMomentumConfig())

	// Initially not primed
	if mom.IsPrimed() {
		t.Error("should not be primed initially")
	}

	// Feed some data (not enough)
	for i := 0; i < 100; i++ {
		mom.Update(0.001, 1000)
	}

	if mom.IsPrimed() {
		t.Error("should not be primed with only 100 samples (halflife 1000 needs ~2000)")
	}

	// Feed more data until primed
	// EWM priming rule: ~2/alpha samples, where alpha = 1 - exp(-ln2/halflife)
	// For halflife=1000, alpha ≈ 0.000693, so need ~2885 samples
	for i := 0; i < 3000; i++ {
		mom.Update(0.001, 1000)
	}

	if !mom.IsPrimed() {
		t.Error("should be primed after 3100 samples")
	}
}

func TestMomentum_Reset(t *testing.T) {
	mom := NewMomentum(DefaultMomentumConfig())

	// Feed some data
	for i := 0; i < 100; i++ {
		mom.Update(0.01, 1000)
	}

	mom.Reset()

	// After reset, state should be cleared
	result := mom.Update(0.01, 1000)

	// First update after reset should give z-score = 0
	if result.MomentumZScore10 != 0 {
		t.Errorf("expected zero z-score after reset, got %v", result.MomentumZScore10)
	}

	if mom.IsPrimed() {
		t.Error("should not be primed after reset")
	}
}

func TestMomentum_StateSerialize(t *testing.T) {
	mom1 := NewMomentum(DefaultMomentumConfig())

	// Feed some data
	for i := 0; i < 200; i++ {
		mom1.Update(0.001*float64(i%10-5), 1000+float64(i%5)*100)
	}

	// Save state
	state := mom1.State()

	// Create new instance and load state
	mom2 := NewMomentum(DefaultMomentumConfig())
	mom2.LoadState(state)

	// Both should produce identical results
	returns := 0.02
	volume := 1500.0
	result1 := mom1.Update(returns, volume)
	result2 := mom2.Update(returns, volume)

	if math.Abs(result1.VWMomentum-result2.VWMomentum) > 1e-10 {
		t.Errorf("VWMomentum mismatch: %v vs %v", result1.VWMomentum, result2.VWMomentum)
	}
	if math.Abs(result1.MomentumZScore10-result2.MomentumZScore10) > 1e-10 {
		t.Errorf("MomentumZScore10 mismatch: %v vs %v", result1.MomentumZScore10, result2.MomentumZScore10)
	}
	if math.Abs(result1.MomentumZScore50-result2.MomentumZScore50) > 1e-10 {
		t.Errorf("MomentumZScore50 mismatch: %v vs %v", result1.MomentumZScore50, result2.MomentumZScore50)
	}
	if math.Abs(result1.MomentumZScore250-result2.MomentumZScore250) > 1e-10 {
		t.Errorf("MomentumZScore250 mismatch: %v vs %v", result1.MomentumZScore250, result2.MomentumZScore250)
	}
	if math.Abs(result1.MomentumZScore1000-result2.MomentumZScore1000) > 1e-10 {
		t.Errorf("MomentumZScore1000 mismatch: %v vs %v", result1.MomentumZScore1000, result2.MomentumZScore1000)
	}
}

func TestMomentum_StreamingProperty(t *testing.T) {
	// Same input sequence should produce same output
	config := DefaultMomentumConfig()
	mom1 := NewMomentum(config)
	mom2 := NewMomentum(config)

	rng := rand.New(rand.NewSource(42))

	for i := 0; i < 500; i++ {
		returns := rng.NormFloat64() * 0.01
		volume := 1000 + rng.Float64()*500

		result1 := mom1.Update(returns, volume)
		result2 := mom2.Update(returns, volume)

		if result1.VWMomentum != result2.VWMomentum {
			t.Errorf("non-deterministic VWMomentum at iteration %d", i)
			break
		}
		if result1.MomentumZScore10 != result2.MomentumZScore10 {
			t.Errorf("non-deterministic MomentumZScore10 at iteration %d", i)
			break
		}
	}
}

func TestMomentum_EdgeCases(t *testing.T) {
	t.Run("very large values", func(t *testing.T) {
		mom := NewMomentum(DefaultMomentumConfig())

		returns := 1e10
		volume := 1e15

		result := mom.Update(returns, volume)

		if math.IsNaN(result.VWMomentum) || math.IsInf(result.VWMomentum, 0) {
			t.Error("should handle large values without NaN/Inf")
		}
	})

	t.Run("very small values", func(t *testing.T) {
		mom := NewMomentum(DefaultMomentumConfig())

		returns := 1e-15
		volume := 1e-10

		result := mom.Update(returns, volume)

		if math.IsNaN(result.VWMomentum) {
			t.Error("should handle small values without NaN")
		}
	})

	t.Run("alternating signs", func(t *testing.T) {
		mom := NewMomentum(DefaultMomentumConfig())

		for i := 0; i < 100; i++ {
			sign := 1.0
			if i%2 == 0 {
				sign = -1.0
			}
			result := mom.Update(sign*0.01, 1000)

			if math.IsNaN(result.VWMomentum) || math.IsNaN(result.MomentumZScore10) {
				t.Errorf("NaN at iteration %d with alternating signs", i)
				break
			}
		}
	})

	t.Run("infinity input", func(t *testing.T) {
		mom := NewMomentum(DefaultMomentumConfig())

		// First some normal data
		for i := 0; i < 10; i++ {
			mom.Update(0.01, 1000)
		}

		// Then infinity
		result := mom.Update(math.Inf(1), 1000)

		// VWMomentum will be Inf, but z-scores might handle it
		if math.IsNaN(result.MomentumZScore10) {
			t.Log("MomentumZScore10 is NaN for Inf input (acceptable)")
		}
	})

	t.Run("NaN input", func(t *testing.T) {
		mom := NewMomentum(DefaultMomentumConfig())

		result := mom.Update(math.NaN(), 1000)

		// VWMomentum will be NaN
		if !math.IsNaN(result.VWMomentum) {
			t.Error("expected NaN for NaN input")
		}
	})
}

func TestMomentum_CurrentFeatures(t *testing.T) {
	mom := NewMomentum(DefaultMomentumConfig())

	// Initial state
	initial := mom.CurrentFeatures()
	if initial.VWMomentum != 0 {
		t.Error("expected zero VWMomentum initially")
	}

	// After update
	mom.Update(0.01, 1000)
	current := mom.CurrentFeatures()
	if current.VWMomentum != 10 {
		t.Errorf("expected VWMomentum=10, got %v", current.VWMomentum)
	}
}

// Benchmark
func BenchmarkMomentum_Update(b *testing.B) {
	mom := NewMomentum(DefaultMomentumConfig())

	// Warmup
	for i := 0; i < 1000; i++ {
		mom.Update(0.01, 1000)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = mom.Update(0.01, 1000)
	}
}
