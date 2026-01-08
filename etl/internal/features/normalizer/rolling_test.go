package normalizer

import (
	"math"
	"math/rand"
	"testing"
)

func TestRolling_Basic(t *testing.T) {
	t.Run("window size", func(t *testing.T) {
		r := NewRolling(10)
		if r.Window() != 10 {
			t.Errorf("expected window 10, got %d", r.Window())
		}
	})

	t.Run("single value", func(t *testing.T) {
		r := NewRolling(10)
		z := r.Update(5.0)

		// First value: zscore = 0 (no variance yet)
		if z != 0 {
			t.Errorf("expected zscore 0 for single value, got %v", z)
		}
		if r.Count() != 1 {
			t.Errorf("expected count 1, got %d", r.Count())
		}
	})

	t.Run("fills window", func(t *testing.T) {
		r := NewRolling(5)
		values := []float64{1, 2, 3, 4, 5}

		for _, v := range values {
			r.Update(v)
		}

		// Mean = 3, variance = 2.5
		expectedMean := 3.0
		if math.Abs(r.Mean()-expectedMean) > 1e-10 {
			t.Errorf("expected mean %v, got %v", expectedMean, r.Mean())
		}

		if !r.IsPrimed() {
			t.Error("should be primed when window is full")
		}
	})

	t.Run("sliding window", func(t *testing.T) {
		r := NewRolling(3)

		// Fill window
		r.Update(1.0)
		r.Update(2.0)
		r.Update(3.0)

		// Mean should be 2
		if math.Abs(r.Mean()-2.0) > 1e-10 {
			t.Errorf("expected mean 2.0, got %v", r.Mean())
		}

		// Add new value, oldest (1.0) should drop
		r.Update(4.0)

		// New window: [2, 3, 4], mean = 3
		if math.Abs(r.Mean()-3.0) > 1e-10 {
			t.Errorf("expected mean 3.0 after slide, got %v", r.Mean())
		}

		// Add another
		r.Update(5.0)

		// New window: [3, 4, 5], mean = 4
		if math.Abs(r.Mean()-4.0) > 1e-10 {
			t.Errorf("expected mean 4.0 after second slide, got %v", r.Mean())
		}
	})
}

func TestRolling_ZScore(t *testing.T) {
	t.Run("zscore calculation", func(t *testing.T) {
		r := NewRolling(5)
		values := []float64{10, 20, 30, 40, 50}

		for _, v := range values {
			r.Update(v)
		}

		// Mean = 30, should have meaningful variance
		mean := r.Mean()
		std := r.Std()

		// Z-score of mean should be 0
		z := r.ZScore(mean)
		if math.Abs(z) > 1e-10 {
			t.Errorf("zscore of mean should be 0, got %v", z)
		}

		// Z-score of mean+std should be 1
		z = r.ZScore(mean + std)
		if math.Abs(z-1.0) > 1e-10 {
			t.Errorf("zscore should be 1.0, got %v", z)
		}
	})

	t.Run("zscore with zero variance", func(t *testing.T) {
		r := NewRolling(5)

		// All same values
		for i := 0; i < 5; i++ {
			r.Update(100.0)
		}

		// Std = 0, zscore should return 0 to avoid division by zero
		z := r.ZScore(150.0)
		if z != 0 {
			t.Errorf("expected 0 for zero variance, got %v", z)
		}
	})
}

func TestRolling_FloatingPointDrift(t *testing.T) {
	t.Run("drift after many updates", func(t *testing.T) {
		r := NewRolling(100)
		rng := rand.New(rand.NewSource(42))

		// Run many updates - this is where naive sum would accumulate error
		for i := 0; i < 1000000; i++ {
			r.Update(rng.Float64() * 100)
		}

		// Check drift error
		drift := r.DriftError()
		// With Kahan + periodic recalc, drift should be minimal
		if drift > 1e-6 {
			t.Errorf("drift error too high: %e", drift)
		}
	})

	t.Run("drift with catastrophic cancellation scenario", func(t *testing.T) {
		r := NewRolling(100)

		// Pattern that causes catastrophic cancellation in naive sums
		// Large value + many small additions/removals
		base := 1e10
		for i := 0; i < 100; i++ {
			r.Update(base + float64(i)*1e-5)
		}

		// Now cycle many times
		for i := 0; i < 100000; i++ {
			r.Update(base + float64(i%100)*1e-5)
		}

		drift := r.DriftError()
		if drift > 1e-3 {
			t.Errorf("drift error too high in cancellation scenario: %e", drift)
		}
	})
}

func TestRolling_PeriodicRecalculation(t *testing.T) {
	t.Run("recalculation triggers", func(t *testing.T) {
		r := NewRollingWithRecalcInterval(10, 100) // recalc every 100 updates

		for i := 0; i < 150; i++ {
			r.Update(float64(i))
		}

		// Should have triggered at least one recalculation
		if r.RecalcCount() < 1 {
			t.Errorf("expected at least 1 recalculation, got %d", r.RecalcCount())
		}
	})

	t.Run("recalculation corrects drift", func(t *testing.T) {
		// Create rolling without Kahan (hypothetically for testing)
		// Here we just verify that recalculation doesn't break anything
		r := NewRollingWithRecalcInterval(50, 1000)

		for i := 0; i < 10000; i++ {
			r.Update(float64(i % 100))
		}

		// After recalculations, mean should still be accurate
		// Window should contain values 50-99 (last 50 of 0-99 cycle)
		// Expected mean of [50..99] = 74.5
		expectedMean := 74.5
		if math.Abs(r.Mean()-expectedMean) > 0.1 {
			t.Errorf("expected mean ~%v, got %v", expectedMean, r.Mean())
		}
	})
}

func TestRolling_NumericalStability(t *testing.T) {
	t.Run("standard normal distribution", func(t *testing.T) {
		r := NewRolling(1000)
		rng := rand.New(rand.NewSource(42))

		// Feed standard normal values
		for i := 0; i < 10000; i++ {
			r.Update(rng.NormFloat64())
		}

		// Mean should be close to 0
		if math.Abs(r.Mean()) > 0.1 {
			t.Errorf("expected mean ~0, got %v", r.Mean())
		}

		// Variance should be close to 1
		if math.Abs(r.Variance()-1.0) > 0.1 {
			t.Errorf("expected variance ~1, got %v", r.Variance())
		}
	})

	t.Run("large values", func(t *testing.T) {
		r := NewRolling(10)
		base := 1e12

		for i := 0; i < 10; i++ {
			r.Update(base + float64(i))
		}

		// Mean should be base + 4.5
		expectedMean := base + 4.5
		if math.Abs(r.Mean()-expectedMean) > 1 {
			t.Errorf("expected mean %v, got %v", expectedMean, r.Mean())
		}
	})
}

func TestRolling_EdgeCases(t *testing.T) {
	t.Run("window size 1", func(t *testing.T) {
		r := NewRolling(1)

		r.Update(100.0)
		// Single element, variance = 0
		if r.Variance() != 0 {
			t.Errorf("expected variance 0 for window=1, got %v", r.Variance())
		}

		r.Update(200.0)
		// Still single element after slide
		if r.Mean() != 200.0 {
			t.Errorf("expected mean 200, got %v", r.Mean())
		}
	})

	t.Run("negative values", func(t *testing.T) {
		r := NewRolling(5)
		values := []float64{-10, -20, -30, -40, -50}

		for _, v := range values {
			r.Update(v)
		}

		if r.Mean() != -30 {
			t.Errorf("expected mean -30, got %v", r.Mean())
		}
	})

	t.Run("mixed signs", func(t *testing.T) {
		r := NewRolling(4)
		values := []float64{-10, 10, -10, 10}

		for _, v := range values {
			r.Update(v)
		}

		if r.Mean() != 0 {
			t.Errorf("expected mean 0, got %v", r.Mean())
		}
	})

	t.Run("infinity", func(t *testing.T) {
		r := NewRolling(5)
		r.Update(1.0)
		r.Update(2.0)
		r.Update(math.Inf(1))

		if !math.IsInf(r.Mean(), 1) {
			t.Errorf("expected +Inf mean, got %v", r.Mean())
		}
	})

	t.Run("NaN", func(t *testing.T) {
		r := NewRolling(5)
		r.Update(1.0)
		r.Update(math.NaN())

		if !math.IsNaN(r.Mean()) {
			t.Errorf("expected NaN mean, got %v", r.Mean())
		}
	})

	t.Run("zero window size uses minimum", func(t *testing.T) {
		r := NewRolling(0)
		// Should use minimum window size (e.g., 1)
		if r.Window() < 1 {
			t.Errorf("expected minimum window >= 1, got %d", r.Window())
		}
	})
}

func TestRolling_Reset(t *testing.T) {
	r := NewRolling(10)

	for i := 0; i < 20; i++ {
		r.Update(float64(i))
	}

	r.Reset()

	if r.Count() != 0 {
		t.Errorf("expected count 0 after reset, got %d", r.Count())
	}
	if r.Mean() != 0 {
		t.Errorf("expected mean 0 after reset, got %v", r.Mean())
	}
	if r.IsPrimed() {
		t.Error("should not be primed after reset")
	}
}

func TestRolling_StateSerialize(t *testing.T) {
	r1 := NewRolling(10)

	// Feed values
	for i := 0; i < 15; i++ {
		r1.Update(float64(i))
	}

	// Save state
	state := r1.State()

	// Create new instance and load state
	r2 := NewRolling(10)
	r2.LoadState(state)

	// Both should produce same results for same input
	z1 := r1.Update(100.0)
	z2 := r2.Update(100.0)

	if math.Abs(z1-z2) > 1e-10 {
		t.Errorf("zscore mismatch after state restore: %v vs %v", z1, z2)
	}
	if math.Abs(r1.Mean()-r2.Mean()) > 1e-10 {
		t.Errorf("mean mismatch after state restore: %v vs %v", r1.Mean(), r2.Mean())
	}
}

func TestRolling_IsPrimed(t *testing.T) {
	r := NewRolling(10)

	for i := 0; i < 9; i++ {
		r.Update(float64(i))
		if r.IsPrimed() {
			t.Errorf("should not be primed with %d samples", i+1)
		}
	}

	r.Update(9.0)
	if !r.IsPrimed() {
		t.Error("should be primed when window is full")
	}
}

// Test comparison with simple batch calculation
func TestRolling_MatchesBatchCalculation(t *testing.T) {
	r := NewRolling(5)
	values := []float64{2, 4, 4, 4, 5, 5, 7, 9}

	for i, v := range values {
		r.Update(v)

		if i >= 4 { // Window is full
			// Get the last 5 values
			start := i - 4
			window := values[start : i+1]

			// Calculate batch mean
			var sum float64
			for _, x := range window {
				sum += x
			}
			batchMean := sum / 5.0

			if math.Abs(r.Mean()-batchMean) > 1e-10 {
				t.Errorf("at index %d: rolling mean %v != batch mean %v",
					i, r.Mean(), batchMean)
			}
		}
	}
}

// Benchmark
func BenchmarkRolling_Update(b *testing.B) {
	r := NewRolling(100)
	for i := 0; i < b.N; i++ {
		_ = r.Update(float64(i))
	}
}

func BenchmarkRolling_UpdateLargeWindow(b *testing.B) {
	r := NewRolling(10000)
	for i := 0; i < b.N; i++ {
		_ = r.Update(float64(i))
	}
}
