package normalizer

import (
	"math"
	"math/rand"
	"testing"
)

func TestWelford_Basic(t *testing.T) {
	t.Run("single value", func(t *testing.T) {
		w := NewWelford()
		w.Update(5.0)

		if w.Count() != 1 {
			t.Errorf("expected count 1, got %d", w.Count())
		}
		if w.Mean() != 5.0 {
			t.Errorf("expected mean 5.0, got %v", w.Mean())
		}
		// Variance undefined for n=1
		if w.Variance() != 0 {
			t.Errorf("expected variance 0 for n=1, got %v", w.Variance())
		}
	})

	t.Run("two values", func(t *testing.T) {
		w := NewWelford()
		w.Update(2.0)
		w.Update(4.0)

		if w.Count() != 2 {
			t.Errorf("expected count 2, got %d", w.Count())
		}
		if w.Mean() != 3.0 {
			t.Errorf("expected mean 3.0, got %v", w.Mean())
		}
		// Variance of [2, 4] = 2 (sample variance)
		if w.Variance() != 2.0 {
			t.Errorf("expected variance 2.0, got %v", w.Variance())
		}
	})

	t.Run("multiple values", func(t *testing.T) {
		w := NewWelford()
		values := []float64{2, 4, 4, 4, 5, 5, 7, 9}

		for _, v := range values {
			w.Update(v)
		}

		// Mean = 40/8 = 5
		if w.Mean() != 5.0 {
			t.Errorf("expected mean 5.0, got %v", w.Mean())
		}

		// Sample variance = sum((x-mean)^2) / (n-1) = 32/7 ≈ 4.571
		expectedVar := 32.0 / 7.0
		if math.Abs(w.Variance()-expectedVar) > 1e-10 {
			t.Errorf("expected variance %v, got %v", expectedVar, w.Variance())
		}

		// Std = sqrt(4.571) ≈ 2.138
		expectedStd := math.Sqrt(expectedVar)
		if math.Abs(w.Std()-expectedStd) > 1e-10 {
			t.Errorf("expected std %v, got %v", expectedStd, w.Std())
		}
	})
}

func TestWelford_ZScore(t *testing.T) {
	t.Run("basic zscore", func(t *testing.T) {
		w := NewWelford()
		// Create a known distribution
		values := []float64{10, 20, 30, 40, 50}
		for _, v := range values {
			w.Update(v)
		}

		// Mean = 30, Std ≈ 15.81
		mean := w.Mean()
		std := w.Std()

		// Z-score of mean should be 0
		zscore := w.ZScore(mean)
		if math.Abs(zscore) > 1e-10 {
			t.Errorf("expected zscore of mean to be 0, got %v", zscore)
		}

		// Z-score of (mean + std) should be 1
		zscore = w.ZScore(mean + std)
		if math.Abs(zscore-1.0) > 1e-10 {
			t.Errorf("expected zscore to be 1.0, got %v", zscore)
		}

		// Z-score of (mean - std) should be -1
		zscore = w.ZScore(mean - std)
		if math.Abs(zscore-(-1.0)) > 1e-10 {
			t.Errorf("expected zscore to be -1.0, got %v", zscore)
		}
	})

	t.Run("zscore with zero std", func(t *testing.T) {
		w := NewWelford()
		// All same values -> std = 0
		for i := 0; i < 10; i++ {
			w.Update(5.0)
		}

		// Should return 0 when std is 0 to avoid division by zero
		zscore := w.ZScore(10.0)
		if zscore != 0 {
			t.Errorf("expected 0 for zero std, got %v", zscore)
		}
	})
}

func TestWelford_NumericalStability(t *testing.T) {
	t.Run("large values", func(t *testing.T) {
		w := NewWelford()
		base := 1e10
		values := []float64{base + 1, base + 2, base + 3, base + 4, base + 5}

		for _, v := range values {
			w.Update(v)
		}

		// Mean should be base + 3
		expectedMean := base + 3
		if math.Abs(w.Mean()-expectedMean) > 1e-5 {
			t.Errorf("expected mean %v, got %v", expectedMean, w.Mean())
		}

		// Sample variance of [1,2,3,4,5] = 2.5
		expectedVar := 2.5
		if math.Abs(w.Variance()-expectedVar) > 1e-5 {
			t.Errorf("expected variance %v, got %v", expectedVar, w.Variance())
		}
	})

	t.Run("streaming large count", func(t *testing.T) {
		w := NewWelford()
		n := 1000000
		// Standard normal distribution values
		rng := rand.New(rand.NewSource(42))

		for i := 0; i < n; i++ {
			w.Update(rng.NormFloat64())
		}

		// Mean should be close to 0
		if math.Abs(w.Mean()) > 0.01 {
			t.Errorf("expected mean ~0, got %v", w.Mean())
		}

		// Variance should be close to 1
		if math.Abs(w.Variance()-1.0) > 0.01 {
			t.Errorf("expected variance ~1, got %v", w.Variance())
		}
	})
}

func TestWelford_UpdateAndGet(t *testing.T) {
	t.Run("returns current zscore", func(t *testing.T) {
		w := NewWelford()

		// First few values need warmup
		z1 := w.UpdateAndGet(10.0)
		if z1 != 0 {
			t.Errorf("first value zscore should be 0, got %v", z1)
		}

		_ = w.UpdateAndGet(20.0)
		// Only 2 values, zscore might be unstable but should be computed

		// After enough values, zscore should be meaningful
		for i := 0; i < 100; i++ {
			w.UpdateAndGet(float64(15 + i%10))
		}

		z := w.UpdateAndGet(w.Mean())
		if math.Abs(z) > 0.5 {
			t.Errorf("zscore of mean should be ~0, got %v", z)
		}
	})
}

func TestWelford_EdgeCases(t *testing.T) {
	t.Run("empty state", func(t *testing.T) {
		w := NewWelford()

		if w.Count() != 0 {
			t.Errorf("expected count 0, got %d", w.Count())
		}
		if w.Mean() != 0 {
			t.Errorf("expected mean 0 for empty, got %v", w.Mean())
		}
		if w.Variance() != 0 {
			t.Errorf("expected variance 0 for empty, got %v", w.Variance())
		}
		if w.Std() != 0 {
			t.Errorf("expected std 0 for empty, got %v", w.Std())
		}
	})

	t.Run("negative values", func(t *testing.T) {
		w := NewWelford()
		values := []float64{-10, -20, -30}

		for _, v := range values {
			w.Update(v)
		}

		if w.Mean() != -20 {
			t.Errorf("expected mean -20, got %v", w.Mean())
		}
	})

	t.Run("mixed positive and negative", func(t *testing.T) {
		w := NewWelford()
		values := []float64{-5, 5, -5, 5}

		for _, v := range values {
			w.Update(v)
		}

		if w.Mean() != 0 {
			t.Errorf("expected mean 0, got %v", w.Mean())
		}
	})

	t.Run("infinity", func(t *testing.T) {
		w := NewWelford()
		w.Update(1.0)
		w.Update(math.Inf(1))

		if !math.IsInf(w.Mean(), 1) {
			t.Errorf("expected +Inf mean, got %v", w.Mean())
		}
	})

	t.Run("NaN", func(t *testing.T) {
		w := NewWelford()
		w.Update(1.0)
		w.Update(math.NaN())

		if !math.IsNaN(w.Mean()) {
			t.Errorf("expected NaN mean, got %v", w.Mean())
		}
	})
}

func TestWelford_Reset(t *testing.T) {
	w := NewWelford()
	for i := 0; i < 100; i++ {
		w.Update(float64(i))
	}

	w.Reset()

	if w.Count() != 0 {
		t.Errorf("expected count 0 after reset, got %d", w.Count())
	}
	if w.Mean() != 0 {
		t.Errorf("expected mean 0 after reset, got %v", w.Mean())
	}
	if w.Variance() != 0 {
		t.Errorf("expected variance 0 after reset, got %v", w.Variance())
	}
}

func TestWelford_IsPrimed(t *testing.T) {
	w := NewWelfordWithMinSamples(10)

	for i := 0; i < 9; i++ {
		w.Update(float64(i))
		if w.IsPrimed() {
			t.Errorf("should not be primed with %d samples", i+1)
		}
	}

	w.Update(9.0)
	if !w.IsPrimed() {
		t.Error("should be primed with 10 samples")
	}
}

// Benchmark
func BenchmarkWelford_Update(b *testing.B) {
	w := NewWelford()
	for i := 0; i < b.N; i++ {
		w.Update(float64(i))
	}
}

func BenchmarkWelford_UpdateAndGet(b *testing.B) {
	w := NewWelford()
	for i := 0; i < b.N; i++ {
		_ = w.UpdateAndGet(float64(i))
	}
}
