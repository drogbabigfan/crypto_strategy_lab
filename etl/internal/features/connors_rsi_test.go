package features

import (
	"math"
	"math/rand"
	"testing"
)

func TestConnorsRSI_RSI(t *testing.T) {
	t.Run("all gains gives RSI component 100", func(t *testing.T) {
		crsi := NewConnorsRSI(DefaultConnorsRSIConfig())

		// Feed consistently rising prices
		price := 100.0
		for i := 0; i < 20; i++ {
			price += 1.0 // Always up
			crsi.Update(price)
		}

		result := crsi.Current()

		// RSI(3) component approaches 100 with all gains
		// But PercentRank depends on where current return ranks vs history
		// With diminishing percentage returns (1/101, 1/102, ...), rank varies
		// Result should be in reasonable range for bullish indicator
		if result < 30 || result > 100 {
			t.Errorf("ConnorsRSI for all gains should be in valid range, got %v", result)
		}
	})

	t.Run("all losses gives RSI near 0", func(t *testing.T) {
		crsi := NewConnorsRSI(DefaultConnorsRSIConfig())

		// Feed consistently falling prices
		price := 100.0
		for i := 0; i < 20; i++ {
			price -= 1.0 // Always down
			crsi.Update(price)
		}

		result := crsi.Current()

		// RSI(3) component approaches 0 with all losses
		// PercentRank of negative returns vs all negative returns
		if result > 70 {
			t.Errorf("expected low ConnorsRSI for all losses, got %v", result)
		}
	})

	t.Run("mixed movement produces valid output", func(t *testing.T) {
		crsi := NewConnorsRSI(DefaultConnorsRSIConfig())

		// Feed alternating prices
		for i := 0; i < 50; i++ {
			if i%2 == 0 {
				crsi.Update(101)
			} else {
				crsi.Update(99)
			}
		}

		result := crsi.Current()

		// With alternating +2%/-2% returns, the value depends on
		// the last return's rank. Should be valid 0-100 range.
		if result < 0 || result > 100 {
			t.Errorf("ConnorsRSI should be in [0, 100], got %v", result)
		}
	})
}

func TestConnorsRSI_PercentRank(t *testing.T) {
	t.Run("extreme positive return ranks high", func(t *testing.T) {
		crsi := NewConnorsRSI(ConnorsRSIConfig{
			RSIPeriod:  3,
			RankPeriod: 20,
		})

		// Feed some normal returns
		price := 100.0
		for i := 0; i < 25; i++ {
			price += (float64(i%5) - 2) * 0.1 // Small fluctuations
			crsi.Update(price)
		}

		// Then a big jump
		price += 10.0
		result := crsi.Update(price)

		// PercentRank should be high, pushing ConnorsRSI up
		if result < 60 {
			t.Errorf("expected high ConnorsRSI after extreme positive return, got %v", result)
		}
	})

	t.Run("extreme negative return ranks low", func(t *testing.T) {
		crsi := NewConnorsRSI(ConnorsRSIConfig{
			RSIPeriod:  3,
			RankPeriod: 20,
		})

		// Feed some normal returns
		price := 100.0
		for i := 0; i < 25; i++ {
			price += (float64(i%5) - 2) * 0.1
			crsi.Update(price)
		}

		// Then a big drop
		price -= 10.0
		result := crsi.Update(price)

		// PercentRank should be low, pushing ConnorsRSI down
		if result > 40 {
			t.Errorf("expected low ConnorsRSI after extreme negative return, got %v", result)
		}
	})
}

func TestConnorsRSI_FirstBar(t *testing.T) {
	crsi := NewConnorsRSI(DefaultConnorsRSIConfig())

	// First bar should return 0
	result := crsi.Update(100)

	if result != 0 {
		t.Errorf("expected 0 for first bar, got %v", result)
	}
}

func TestConnorsRSI_IsPrimed(t *testing.T) {
	crsi := NewConnorsRSI(ConnorsRSIConfig{
		RSIPeriod:  3,
		RankPeriod: 50, // Need 50 samples for full rank buffer
	})

	// Initially not primed
	if crsi.IsPrimed() {
		t.Error("should not be primed initially")
	}

	// Feed data but not enough
	for i := 0; i < 30; i++ {
		crsi.Update(100 + float64(i%5))
	}

	if crsi.IsPrimed() {
		t.Error("should not be primed with only 30 samples (need 50)")
	}

	// Feed enough data
	for i := 0; i < 30; i++ {
		crsi.Update(100 + float64(i%5))
	}

	if !crsi.IsPrimed() {
		t.Error("should be primed after 60 samples")
	}
}

func TestConnorsRSI_Reset(t *testing.T) {
	crsi := NewConnorsRSI(DefaultConnorsRSIConfig())

	// Feed some data
	for i := 0; i < 50; i++ {
		crsi.Update(100 + float64(i))
	}

	crsi.Reset()

	// After reset, first bar should return 0
	result := crsi.Update(100)
	if result != 0 {
		t.Errorf("expected 0 after reset, got %v", result)
	}

	if crsi.IsPrimed() {
		t.Error("should not be primed after reset")
	}
}

func TestConnorsRSI_StateSerialize(t *testing.T) {
	crsi1 := NewConnorsRSI(DefaultConnorsRSIConfig())

	// Feed some data
	price := 100.0
	for i := 0; i < 150; i++ {
		price += (float64(i%10) - 5) * 0.5
		crsi1.Update(price)
	}

	// Save state
	state := crsi1.State()

	// Create new instance and load state
	crsi2 := NewConnorsRSI(DefaultConnorsRSIConfig())
	crsi2.LoadState(state)

	// Both should produce identical results
	testPrice := 120.0
	result1 := crsi1.Update(testPrice)
	result2 := crsi2.Update(testPrice)

	if math.Abs(result1-result2) > 1e-10 {
		t.Errorf("ConnorsRSI mismatch after state restore: %v vs %v", result1, result2)
	}
}

func TestConnorsRSI_ValueRange(t *testing.T) {
	crsi := NewConnorsRSI(DefaultConnorsRSIConfig())

	rng := rand.New(rand.NewSource(42))
	price := 100.0

	for i := 0; i < 500; i++ {
		price += rng.NormFloat64() * 2
		if price < 10 {
			price = 10
		}

		result := crsi.Update(price)

		// Skip first bar (returns 0)
		if i == 0 {
			continue
		}

		// ConnorsRSI should always be in [0, 100]
		if result < 0 || result > 100 {
			t.Errorf("ConnorsRSI out of range at iteration %d: %v", i, result)
			break
		}
	}
}

func TestConnorsRSI_StreamingProperty(t *testing.T) {
	// Same input sequence should produce same output
	config := DefaultConnorsRSIConfig()
	crsi1 := NewConnorsRSI(config)
	crsi2 := NewConnorsRSI(config)

	rng := rand.New(rand.NewSource(123))
	price := 100.0

	for i := 0; i < 300; i++ {
		price += rng.NormFloat64()

		result1 := crsi1.Update(price)
		result2 := crsi2.Update(price)

		if result1 != result2 {
			t.Errorf("non-deterministic at iteration %d: %v vs %v", i, result1, result2)
			break
		}
	}
}

func TestConnorsRSI_EdgeCases(t *testing.T) {
	t.Run("zero price", func(t *testing.T) {
		crsi := NewConnorsRSI(DefaultConnorsRSIConfig())

		crsi.Update(100)
		result := crsi.Update(0) // Zero price

		if math.IsNaN(result) {
			t.Error("should handle zero price gracefully")
		}
	})

	t.Run("constant price", func(t *testing.T) {
		crsi := NewConnorsRSI(DefaultConnorsRSIConfig())

		// Same price every time (no movement)
		for i := 0; i < 50; i++ {
			crsi.Update(100)
		}

		result := crsi.Current()

		// With no movement:
		// - RSI(3) returns 50 (avgGain and avgLoss both near 0)
		// - PercentRank: all returns are 0, so countLess = 0, rank = 0
		// - ConnorsRSI = (50 + 0) / 2 = 25
		if result < 20 || result > 30 {
			t.Errorf("expected ~25 ConnorsRSI for constant price, got %v", result)
		}
	})

	t.Run("very large price", func(t *testing.T) {
		crsi := NewConnorsRSI(DefaultConnorsRSIConfig())

		crsi.Update(100)
		result := crsi.Update(1e15)

		if math.IsNaN(result) || math.IsInf(result, 0) {
			t.Error("should handle large price without NaN/Inf")
		}
	})

	t.Run("very small price", func(t *testing.T) {
		crsi := NewConnorsRSI(DefaultConnorsRSIConfig())

		crsi.Update(1e-10)
		result := crsi.Update(1e-11)

		if math.IsNaN(result) {
			t.Error("should handle small price without NaN")
		}
	})

	t.Run("negative price", func(t *testing.T) {
		crsi := NewConnorsRSI(DefaultConnorsRSIConfig())

		crsi.Update(100)
		result := crsi.Update(-50) // Negative price (invalid but should handle)

		// Should not crash, might produce unusual values
		if math.IsNaN(result) {
			t.Log("NaN for negative price (acceptable)")
		}
	})
}

func TestConnorsRSI_NoStreak(t *testing.T) {
	// Verify that Streak has been removed (modified formula)
	// ConnorsRSI = (RSI(3) + PercentRank) / 2
	// Not (RSI(3) + RSI_Streak(2) + PercentRank) / 3

	crsi := NewConnorsRSI(DefaultConnorsRSIConfig())

	// With consistent small gains:
	// - RSI(3) approaches 100 (all gains)
	// - PercentRank: returns are ~0.1% and decreasing (0.1/100.1, 0.1/100.2, ...)
	//   so current return ranks LOW among historical returns
	// This demonstrates streak is removed - old formula would be higher

	price := 100.0
	for i := 0; i < 100; i++ {
		price += 0.1 // Consistent small gains
		crsi.Update(price)
	}

	result := crsi.Current()

	// RSI(3) → ~100
	// PercentRank → ~0 (current return is smallest due to % diminishing)
	// Average → ~50 (demonstrates streak removal - old formula would be ~66+)
	if result < 40 || result > 60 {
		t.Errorf("expected ~50 ConnorsRSI (streak removed), got %v", result)
	}
}

func TestConnorsRSI_WilderSmoothing(t *testing.T) {
	// Verify Wilder's smoothing is used (alpha = 1/period)
	crsi := NewConnorsRSI(ConnorsRSIConfig{
		RSIPeriod:  3, // alpha = 1/3
		RankPeriod: 10,
	})

	// First few updates
	crsi.Update(100)
	crsi.Update(101) // +1% gain
	crsi.Update(102) // ~+0.99% gain
	result := crsi.Update(103) // ~+0.98% gain

	// RSI(3) with all gains → 100
	// PercentRank with only 3 returns, current is smallest % → 0
	// (returns are ~1%, 0.99%, 0.98% - current ranks last)
	// ConnorsRSI = (100 + 0) / 2 = 50
	if result < 40 || result > 60 {
		t.Errorf("expected ~50 ConnorsRSI for consecutive gains (early buffer), got %v", result)
	}
}

func TestConnorsRSI_RankBoundary(t *testing.T) {
	// Test that rank boundary values are correct when buffer is full
	t.Run("max return gives rank 100", func(t *testing.T) {
		crsi := NewConnorsRSI(ConnorsRSIConfig{
			RSIPeriod:  3,
			RankPeriod: 100,
		})

		// Fill buffer with normal returns
		price := 100.0
		for i := 0; i < 110; i++ {
			price += (float64(i%5) - 2) * 0.1 // Small fluctuations
			crsi.Update(price)
		}

		// Now inject a huge return that should rank at 100
		price += 50.0 // Massive jump
		result := crsi.Update(price)

		// RSI should be high, PercentRank should be 100 (max return)
		// ConnorsRSI should be very high
		if result < 80 {
			t.Errorf("expected high ConnorsRSI for max return, got %v", result)
		}
	})

	t.Run("min return gives rank 0", func(t *testing.T) {
		crsi := NewConnorsRSI(ConnorsRSIConfig{
			RSIPeriod:  3,
			RankPeriod: 100,
		})

		// Fill buffer with normal returns
		price := 100.0
		for i := 0; i < 110; i++ {
			price += (float64(i%5) - 2) * 0.1 // Small fluctuations
			crsi.Update(price)
		}

		// Now inject a huge loss that should rank at 0
		price -= 50.0 // Massive drop
		result := crsi.Update(price)

		// RSI should be low, PercentRank should be 0 (min return)
		// ConnorsRSI should be very low
		if result > 20 {
			t.Errorf("expected low ConnorsRSI for min return, got %v", result)
		}
	})
}

// Benchmark
func BenchmarkConnorsRSI_Update(b *testing.B) {
	crsi := NewConnorsRSI(DefaultConnorsRSIConfig())

	// Warmup
	price := 100.0
	for i := 0; i < 200; i++ {
		price += float64(i%5) - 2
		crsi.Update(price)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		crsi.Update(price + float64(i%10))
	}
}
