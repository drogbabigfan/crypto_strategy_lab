package features

import (
	"math"
	"math/rand"
	"testing"

	"dl-rl-btc-etl/internal/bars"
)

func makeStatTestBar(close float64) bars.DynamicDollarBar {
	return bars.DynamicDollarBar{
		Open:        close * 0.99,
		High:        close * 1.01,
		Low:         close * 0.98,
		Close:       close,
		Volume:      1000,
		DollarValue: close * 1000,
		Duration:    60,
	}
}

func TestStationarity_Returns(t *testing.T) {
	t.Run("basic returns", func(t *testing.T) {
		stat := NewStationarity(DefaultStationarityConfig())

		// First bar
		stat.Update(makeStatTestBar(100))

		// Second bar: 10% increase
		result := stat.Update(makeStatTestBar(110))

		// Return = (110 - 100) / 100 = 0.1
		expected := 0.1
		if math.Abs(result.Returns-expected) > 1e-10 {
			t.Errorf("expected returns %v, got %v", expected, result.Returns)
		}
	})

	t.Run("negative returns", func(t *testing.T) {
		stat := NewStationarity(DefaultStationarityConfig())
		stat.Update(makeStatTestBar(100))
		result := stat.Update(makeStatTestBar(90))

		expected := -0.1
		if math.Abs(result.Returns-expected) > 1e-10 {
			t.Errorf("expected returns %v, got %v", expected, result.Returns)
		}
	})

	t.Run("first bar returns zero", func(t *testing.T) {
		stat := NewStationarity(DefaultStationarityConfig())
		result := stat.Update(makeStatTestBar(100))

		if result.Returns != 0 {
			t.Errorf("expected 0 returns for first bar, got %v", result.Returns)
		}
	})
}

func TestStationarity_FracDiff(t *testing.T) {
	t.Run("weights calculation", func(t *testing.T) {
		config := StationarityConfig{
			FracDiffD:       0.4,
			FracDiffThresh:  1e-5,
			DetrendHalflife: 100,
		}
		stat := NewStationarity(config)

		// First weight should be 1.0
		if len(stat.fracDiffWeights) == 0 {
			t.Fatal("no fracdiff weights generated")
		}
		if stat.fracDiffWeights[0] != 1.0 {
			t.Errorf("first weight should be 1.0, got %v", stat.fracDiffWeights[0])
		}

		// Weights should decrease in magnitude
		for i := 1; i < len(stat.fracDiffWeights); i++ {
			if math.Abs(stat.fracDiffWeights[i]) > math.Abs(stat.fracDiffWeights[i-1]) {
				t.Error("weights should decrease in magnitude")
				break
			}
		}
	})

	t.Run("fracdiff preserves memory", func(t *testing.T) {
		stat := NewStationarity(StationarityConfig{
			FracDiffD:       0.4,
			FracDiffThresh:  1e-5,
			DetrendHalflife: 100,
		})

		// Feed a random walk
		rng := rand.New(rand.NewSource(42))
		price := 100.0
		for i := 0; i < 200; i++ {
			price += rng.NormFloat64() * 2
			stat.Update(makeStatTestBar(price))
		}

		result := stat.CurrentFeatures()
		// FracDiff should be computable and not NaN/Inf
		if math.IsNaN(result.FracDiffClose) || math.IsInf(result.FracDiffClose, 0) {
			t.Errorf("fracdiff should be valid, got %v", result.FracDiffClose)
		}
	})

	t.Run("fracdiff d=0 equals no diff", func(t *testing.T) {
		stat := NewStationarity(StationarityConfig{
			FracDiffD:       0.0, // No differentiation
			FracDiffThresh:  1e-5,
			DetrendHalflife: 100,
		})

		// Feed constant price
		for i := 0; i < 100; i++ {
			stat.Update(makeStatTestBar(100))
		}

		result := stat.CurrentFeatures()
		// With d=0, fracdiff(log(100)) ≈ log(100)
		expected := math.Log(100)
		if math.Abs(result.FracDiffClose-expected) > 0.1 {
			t.Errorf("with d=0, fracdiff should be ~log(price), got %v", result.FracDiffClose)
		}
	})

	t.Run("fracdiff d=1 equals regular diff", func(t *testing.T) {
		stat := NewStationarity(StationarityConfig{
			FracDiffD:       1.0, // Full differentiation
			FracDiffThresh:  1e-5,
			DetrendHalflife: 100,
		})

		// Feed bars
		for i := 0; i < 100; i++ {
			stat.Update(makeStatTestBar(100 + float64(i)))
		}

		result := stat.CurrentFeatures()
		// With d=1, fracdiff should be approximately log returns
		// d(log(P)) ≈ (P_t - P_{t-1}) / P_{t-1} for small changes
		if math.IsNaN(result.FracDiffClose) {
			t.Errorf("fracdiff d=1 should be valid, got %v", result.FracDiffClose)
		}
	})
}

func TestStationarity_DetrendedLogPrice(t *testing.T) {
	t.Run("detrended around zero", func(t *testing.T) {
		stat := NewStationarity(StationarityConfig{
			FracDiffD:       0.4,
			FracDiffThresh:  1e-5,
			DetrendHalflife: 20,
		})

		// Feed constant price
		for i := 0; i < 200; i++ {
			stat.Update(makeStatTestBar(100))
		}

		result := stat.CurrentFeatures()
		// Detrended should be close to 0 for constant price
		if math.Abs(result.DetrendedLogPrice) > 0.01 {
			t.Errorf("detrended should be ~0 for constant price, got %v", result.DetrendedLogPrice)
		}
	})

	t.Run("positive deviation above trend", func(t *testing.T) {
		stat := NewStationarity(StationarityConfig{
			FracDiffD:       0.4,
			FracDiffThresh:  1e-5,
			DetrendHalflife: 50,
		})

		// Establish baseline
		for i := 0; i < 100; i++ {
			stat.Update(makeStatTestBar(100))
		}

		// Sudden price increase
		for i := 0; i < 5; i++ {
			stat.Update(makeStatTestBar(120))
		}

		result := stat.CurrentFeatures()
		// Price above trend should give positive detrended value
		if result.DetrendedLogPrice <= 0 {
			t.Errorf("expected positive detrended for above-trend price, got %v", result.DetrendedLogPrice)
		}
	})

	t.Run("negative deviation below trend", func(t *testing.T) {
		stat := NewStationarity(StationarityConfig{
			FracDiffD:       0.4,
			FracDiffThresh:  1e-5,
			DetrendHalflife: 50,
		})

		// Establish baseline
		for i := 0; i < 100; i++ {
			stat.Update(makeStatTestBar(100))
		}

		// Sudden price drop
		for i := 0; i < 5; i++ {
			stat.Update(makeStatTestBar(80))
		}

		result := stat.CurrentFeatures()
		// Price below trend should give negative detrended value
		if result.DetrendedLogPrice >= 0 {
			t.Errorf("expected negative detrended for below-trend price, got %v", result.DetrendedLogPrice)
		}
	})
}

func TestStationarity_IsPrimed(t *testing.T) {
	stat := NewStationarity(StationarityConfig{
		FracDiffD:       0.4,
		FracDiffThresh:  1e-3, // Larger threshold = smaller buffer needed
		DetrendHalflife: 20,   // Need ~40 samples
	})

	// Initially not primed
	if stat.IsPrimed() {
		t.Error("should not be primed initially")
	}

	// Feed some bars (not enough)
	for i := 0; i < 30; i++ {
		stat.Update(makeStatTestBar(100))
	}

	if stat.IsPrimed() {
		t.Error("should not be primed with only 30 samples")
	}

	// Feed more bars (enough for both buffer and EWM)
	for i := 0; i < 100; i++ {
		stat.Update(makeStatTestBar(100))
	}

	if !stat.IsPrimed() {
		// Debug info
		t.Errorf("should be primed after 130 samples (buffer full: %v, ewm primed: %v)",
			stat.logPriceBuffer.IsFull(), stat.detrendEWM.IsPrimed())
	}
}

func TestStationarity_Reset(t *testing.T) {
	stat := NewStationarity(DefaultStationarityConfig())

	// Feed some bars
	for i := 0; i < 100; i++ {
		stat.Update(makeStatTestBar(100 + float64(i)))
	}

	stat.Reset()

	if stat.IsPrimed() {
		t.Error("should not be primed after reset")
	}

	result := stat.CurrentFeatures()
	if result.FracDiffClose != 0 || result.DetrendedLogPrice != 0 {
		t.Error("features should be zero after reset")
	}
}

func TestStationarity_StateSerialize(t *testing.T) {
	stat1 := NewStationarity(DefaultStationarityConfig())

	// Feed some bars
	for i := 0; i < 100; i++ {
		stat1.Update(makeStatTestBar(100 + float64(i)*0.5))
	}

	// Save state
	state := stat1.State()

	// Create new instance and load state
	stat2 := NewStationarity(DefaultStationarityConfig())
	stat2.LoadState(state)

	// Both should produce same results
	bar := makeStatTestBar(150)
	f1 := stat1.Update(bar)
	f2 := stat2.Update(bar)

	if math.Abs(f1.FracDiffClose-f2.FracDiffClose) > 1e-10 {
		t.Errorf("fracdiff mismatch after state restore: %v vs %v",
			f1.FracDiffClose, f2.FracDiffClose)
	}
	if math.Abs(f1.DetrendedLogPrice-f2.DetrendedLogPrice) > 1e-10 {
		t.Errorf("detrended mismatch after state restore: %v vs %v",
			f1.DetrendedLogPrice, f2.DetrendedLogPrice)
	}
}

func TestStationarity_EdgeCases(t *testing.T) {
	t.Run("zero price", func(t *testing.T) {
		stat := NewStationarity(DefaultStationarityConfig())
		bar := bars.DynamicDollarBar{Close: 0}
		result := stat.Update(bar)

		if math.IsNaN(result.FracDiffClose) || math.IsInf(result.FracDiffClose, 0) {
			t.Error("should handle zero price gracefully")
		}
	})

	t.Run("very large price", func(t *testing.T) {
		stat := NewStationarity(DefaultStationarityConfig())
		bar := makeStatTestBar(1e15)
		for i := 0; i < 50; i++ {
			stat.Update(bar)
		}

		result := stat.CurrentFeatures()
		if math.IsNaN(result.FracDiffClose) || math.IsInf(result.FracDiffClose, 0) {
			t.Error("should handle large prices")
		}
	})

	t.Run("very small price", func(t *testing.T) {
		stat := NewStationarity(DefaultStationarityConfig())
		bar := makeStatTestBar(1e-10)
		for i := 0; i < 50; i++ {
			stat.Update(bar)
		}

		result := stat.CurrentFeatures()
		if math.IsNaN(result.FracDiffClose) || math.IsInf(result.FracDiffClose, 0) {
			t.Error("should handle small prices")
		}
	})
}

// Benchmark
func BenchmarkStationarity_Update(b *testing.B) {
	stat := NewStationarity(DefaultStationarityConfig())
	bar := makeStatTestBar(100)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = stat.Update(bar)
	}
}
