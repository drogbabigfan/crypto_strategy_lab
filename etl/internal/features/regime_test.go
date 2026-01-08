package features

import (
	"math"
	"math/rand"
	"testing"

	"dl-rl-btc-etl/internal/bars"
)

func makeRegimeTestBar(high, low, close float64) bars.DynamicDollarBar {
	return bars.DynamicDollarBar{
		Open:        close * 0.99,
		High:        high,
		Low:         low,
		Close:       close,
		Volume:      1000,
		DollarValue: close * 1000,
		Duration:    60,
		TickCount:   100,
	}
}

func TestRegime_GarmanKlassVolatility(t *testing.T) {
	t.Run("basic calculation", func(t *testing.T) {
		regime := NewRegime(RegimeConfig{
			ParkinsonWindow: 5,
			EntropyWindow:   5,
			EntropyBins:     10,
			VolHalflife:     10,
			EntropyHalflife: 10,
		})

		// Feed bars with known High/Low range
		for i := 0; i < 10; i++ {
			bar := makeRegimeTestBar(110, 90, 100) // 20% range
			_ = regime.Update(bar)
		}

		result := regime.CurrentFeatures()
		// GK vol should be positive for volatile bars
		if result.GarmanKlassVol <= 0 {
			t.Errorf("expected positive garman_klass_vol, got %v", result.GarmanKlassVol)
		}
	})

	t.Run("zero volatility", func(t *testing.T) {
		regime := NewRegime(RegimeConfig{
			ParkinsonWindow: 5,
			EntropyWindow:   5,
			EntropyBins:     10,
			VolHalflife:     10,
			EntropyHalflife: 10,
		})

		// Feed bars with no range (High == Low, Open == Close)
		for i := 0; i < 10; i++ {
			bar := bars.DynamicDollarBar{
				Open:        100,
				High:        100,
				Low:         100,
				Close:       100,
				Volume:      1000,
				DollarValue: 100000,
				Duration:    60,
			}
			_ = regime.Update(bar)
		}

		result := regime.CurrentFeatures()
		// Should be very close to zero
		if result.GarmanKlassVol > 0.001 {
			t.Errorf("expected ~0 garman_klass_vol for flat bars, got %v", result.GarmanKlassVol)
		}
	})

	t.Run("garman_klass formula", func(t *testing.T) {
		regime := NewRegime(RegimeConfig{
			ParkinsonWindow: 1,
			EntropyWindow:   5,
			EntropyBins:     10,
			VolHalflife:     10,
			EntropyHalflife: 10,
		})

		// Single bar: GK = sqrt(0.5*ln(H/L)^2 - (2ln2-1)*ln(C/O)^2)
		open := 99.0
		high := 110.0
		low := 90.0
		close := 100.0
		bar := bars.DynamicDollarBar{
			Open:        open,
			High:        high,
			Low:         low,
			Close:       close,
			Volume:      1000,
			DollarValue: 100000,
			Duration:    60,
		}
		_ = regime.Update(bar)

		result := regime.CurrentFeatures()
		logHL := math.Log(high / low)
		logCO := math.Log(close / open)
		gkTerm := 0.5*logHL*logHL - (2*math.Ln2-1)*logCO*logCO
		expected := math.Sqrt(gkTerm)

		if math.Abs(result.GarmanKlassVol-expected) > 1e-6 {
			t.Errorf("expected GK vol %v, got %v", expected, result.GarmanKlassVol)
		}
	})
}

func TestRegime_ShannonEntropy(t *testing.T) {
	t.Run("random returns high entropy", func(t *testing.T) {
		regime := NewRegime(RegimeConfig{
			ParkinsonWindow: 5,
			EntropyWindow:   50,
			EntropyBins:     10,
			VolHalflife:     10,
			EntropyHalflife: 10,
		})

		rng := rand.New(rand.NewSource(42))

		// Feed random price movements
		price := 100.0
		for i := 0; i < 100; i++ {
			change := rng.NormFloat64() * 5
			price += change
			bar := makeRegimeTestBar(price*1.01, price*0.99, price)
			_ = regime.Update(bar)
		}

		result := regime.CurrentFeatures()
		// Random returns should have high entropy (close to 1.0 when normalized)
		if result.ShannonEntropy < 0.5 {
			t.Errorf("expected high entropy for random returns, got %v", result.ShannonEntropy)
		}
	})

	t.Run("trending returns low entropy", func(t *testing.T) {
		regime := NewRegime(RegimeConfig{
			ParkinsonWindow: 5,
			EntropyWindow:   50,
			EntropyBins:     10,
			VolHalflife:     10,
			EntropyHalflife: 10,
		})

		// Strong uptrend: consistent positive returns
		price := 100.0
		for i := 0; i < 100; i++ {
			price *= 1.01 // 1% increase each bar
			bar := makeRegimeTestBar(price*1.005, price*0.995, price)
			_ = regime.Update(bar)
		}

		result := regime.CurrentFeatures()
		// Consistent returns should have lower entropy
		// Note: may still be moderate due to binning
		if result.ShannonEntropy > 0.9 {
			t.Errorf("expected lower entropy for trending, got %v", result.ShannonEntropy)
		}
	})
}

func TestRegime_VolRatio(t *testing.T) {
	regime := NewRegime(RegimeConfig{
		ParkinsonWindow: 10,
		EntropyWindow:   10,
		EntropyBins:     10,
		VolHalflife:     10,
		EntropyHalflife: 10,
	})

	// Feed bars with varying prices to generate non-zero returns
	for i := 0; i < 20; i++ {
		// Vary the close price to generate returns
		closePrice := 100.0 + float64(i%5)*2 // 100, 102, 104, 106, 108, 100, ...
		bar := makeRegimeTestBar(closePrice*1.1, closePrice*0.9, closePrice)
		_ = regime.Update(bar)
	}

	result := regime.CurrentFeatures()
	// VolRatio = GarmanKlass / Realized, should be positive
	if result.VolRatio <= 0 {
		t.Errorf("expected positive vol_ratio, got %v (gk=%v, realized=%v)",
			result.VolRatio, result.GarmanKlassVol, result.RealizedVol)
	}
}

func TestRegime_ZScores(t *testing.T) {
	t.Run("vol_zscore after warmup", func(t *testing.T) {
		regime := NewRegime(RegimeConfig{
			ParkinsonWindow: 10,
			EntropyWindow:   10,
			EntropyBins:     10,
			VolHalflife:     20, // Need ~40 samples to prime
			EntropyHalflife: 20,
		})

		// Warmup with consistent volatility
		for i := 0; i < 100; i++ {
			bar := makeRegimeTestBar(105, 95, 100)
			_ = regime.Update(bar)
		}

		result := regime.CurrentFeatures()
		// After many similar bars, zscore should be close to 0
		if math.Abs(result.VolZScore) > 2 {
			t.Errorf("expected vol_zscore near 0 for stable vol, got %v", result.VolZScore)
		}
	})

	t.Run("vol_zscore spike detection", func(t *testing.T) {
		regime := NewRegime(RegimeConfig{
			ParkinsonWindow: 10,
			EntropyWindow:   10,
			EntropyBins:     10,
			VolHalflife:     20,
			EntropyHalflife: 20,
		})

		// Warmup with low volatility
		for i := 0; i < 100; i++ {
			bar := makeRegimeTestBar(101, 99, 100)
			_ = regime.Update(bar)
		}

		// Sudden high volatility bar
		spikeBar := makeRegimeTestBar(120, 80, 100) // 40% range vs 2% before
		_ = regime.Update(spikeBar)

		result := regime.CurrentFeatures()
		// Z-score should be significantly positive
		if result.VolZScore < 1.0 {
			t.Errorf("expected high vol_zscore for spike, got %v", result.VolZScore)
		}
	})
}

func TestRegime_IsPrimed(t *testing.T) {
	regime := NewRegime(RegimeConfig{
		ParkinsonWindow: 10,
		EntropyWindow:   10,
		EntropyBins:     10,
		VolHalflife:     50, // Need ~100 samples
		EntropyHalflife: 50,
	})

	// Initially not primed
	if regime.IsPrimed() {
		t.Error("should not be primed initially")
	}

	// Feed some bars (not enough)
	for i := 0; i < 50; i++ {
		bar := makeRegimeTestBar(110, 90, 100)
		_ = regime.Update(bar)
	}

	if regime.IsPrimed() {
		t.Error("should not be primed with only 50 samples")
	}

	// Feed more bars
	for i := 0; i < 100; i++ {
		bar := makeRegimeTestBar(110, 90, 100)
		_ = regime.Update(bar)
	}

	if !regime.IsPrimed() {
		t.Error("should be primed after 150 samples")
	}
}

func TestRegime_Reset(t *testing.T) {
	regime := NewRegime(RegimeConfig{
		ParkinsonWindow: 10,
		EntropyWindow:   10,
		EntropyBins:     10,
		VolHalflife:     10,
		EntropyHalflife: 10,
	})

	// Feed some bars
	for i := 0; i < 50; i++ {
		bar := makeRegimeTestBar(110, 90, 100)
		_ = regime.Update(bar)
	}

	regime.Reset()

	if regime.IsPrimed() {
		t.Error("should not be primed after reset")
	}

	result := regime.CurrentFeatures()
	if result.GarmanKlassVol != 0 {
		t.Error("features should be zero after reset")
	}
}

func TestRegime_StateSerialize(t *testing.T) {
	regime1 := NewRegime(RegimeConfig{
		ParkinsonWindow: 10,
		EntropyWindow:   10,
		EntropyBins:     10,
		VolHalflife:     20,
		EntropyHalflife: 20,
	})

	// Feed some bars
	for i := 0; i < 50; i++ {
		bar := makeRegimeTestBar(110, 90, 100)
		_ = regime1.Update(bar)
	}

	// Save state
	state := regime1.State()

	// Create new instance and load state
	regime2 := NewRegime(RegimeConfig{
		ParkinsonWindow: 10,
		EntropyWindow:   10,
		EntropyBins:     10,
		VolHalflife:     20,
		EntropyHalflife: 20,
	})
	regime2.LoadState(state)

	// Both should produce same results for same input
	bar := makeRegimeTestBar(115, 85, 100)
	f1 := regime1.Update(bar)
	f2 := regime2.Update(bar)

	if math.Abs(f1.GarmanKlassVol-f2.GarmanKlassVol) > 1e-10 {
		t.Errorf("garman_klass mismatch after state restore")
	}
	if math.Abs(f1.VolZScore-f2.VolZScore) > 1e-10 {
		t.Errorf("vol_zscore mismatch after state restore")
	}
}

func TestRegime_EdgeCases(t *testing.T) {
	t.Run("zero price", func(t *testing.T) {
		regime := NewRegime(DefaultRegimeConfig())
		bar := bars.DynamicDollarBar{
			High:  0,
			Low:   0,
			Close: 0,
		}
		result := regime.Update(bar)

		// Should not produce NaN or Inf
		if math.IsNaN(result.GarmanKlassVol) || math.IsInf(result.GarmanKlassVol, 0) {
			t.Error("should handle zero price gracefully")
		}
	})

	t.Run("negative log input", func(t *testing.T) {
		regime := NewRegime(DefaultRegimeConfig())
		// Low > High (invalid but should handle)
		bar := makeRegimeTestBar(90, 110, 100) // Swapped high/low
		result := regime.Update(bar)

		if math.IsNaN(result.GarmanKlassVol) {
			t.Error("should handle swapped high/low")
		}
	})
}

// Benchmark
func BenchmarkRegime_Update(b *testing.B) {
	regime := NewRegime(DefaultRegimeConfig())
	bar := makeRegimeTestBar(110, 90, 100)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = regime.Update(bar)
	}
}
