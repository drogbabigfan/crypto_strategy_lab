package features

import (
	"math"
	"testing"

	"dl-rl-btc-etl/internal/bars"
)

func makeTestBar(close, volume, dollarValue, duration float64, buyDollarVol, sellDollarVol float64, tickCount int64) bars.DynamicDollarBar {
	return bars.DynamicDollarBar{
		Open:          close * 0.99,
		High:          close * 1.01,
		Low:           close * 0.98,
		Close:         close,
		Volume:        volume,
		DollarValue:   dollarValue,
		Duration:      duration,
		TickCount:     tickCount,
		BuyDollarVol:  buyDollarVol,
		SellDollarVol: sellDollarVol,
		NetImbalance:  buyDollarVol - sellDollarVol,
	}
}

func TestL1_LogTransforms(t *testing.T) {
	l1 := NewL1()

	t.Run("log_volume", func(t *testing.T) {
		bar := makeTestBar(100, 1000, 100000, 60, 60000, 40000, 500)
		result := l1.Compute(bar)

		expected := math.Log1p(1000)
		if math.Abs(result.LogVolume-expected) > 1e-10 {
			t.Errorf("expected log_volume %v, got %v", expected, result.LogVolume)
		}
	})

	t.Run("log_tick_count", func(t *testing.T) {
		bar := makeTestBar(100, 1000, 100000, 60, 60000, 40000, 500)
		result := l1.Compute(bar)

		expected := math.Log1p(500)
		if math.Abs(result.LogTickCount-expected) > 1e-10 {
			t.Errorf("expected log_tick_count %v, got %v", expected, result.LogTickCount)
		}
	})

	t.Run("log_duration", func(t *testing.T) {
		bar := makeTestBar(100, 1000, 100000, 60, 60000, 40000, 500)
		result := l1.Compute(bar)

		expected := math.Log1p(60)
		if math.Abs(result.LogDuration-expected) > 1e-10 {
			t.Errorf("expected log_duration %v, got %v", expected, result.LogDuration)
		}
	})

	t.Run("log_trade_intensity", func(t *testing.T) {
		bar := makeTestBar(100, 1000, 100000, 60, 60000, 40000, 500)
		result := l1.Compute(bar)

		// trade_intensity = dollar_value / duration
		intensity := 100000.0 / 60.0
		expected := math.Log1p(intensity)
		if math.Abs(result.LogTradeIntensity-expected) > 1e-10 {
			t.Errorf("expected log_trade_intensity %v, got %v", expected, result.LogTradeIntensity)
		}
	})
}

func TestL1_VWAPDeviation(t *testing.T) {
	l1 := NewL1()

	t.Run("positive deviation", func(t *testing.T) {
		// VWAP = dollar_value / volume = 100000 / 1000 = 100
		// Close = 105, deviation = (105 - 100) / 105 = 0.0476
		bar := makeTestBar(105, 1000, 100000, 60, 60000, 40000, 500)
		result := l1.Compute(bar)

		vwap := 100000.0 / 1000.0
		expected := (105 - vwap) / (105 + Epsilon)
		if math.Abs(result.VWAPDeviation-expected) > 1e-10 {
			t.Errorf("expected vwap_deviation %v, got %v", expected, result.VWAPDeviation)
		}
	})

	t.Run("negative deviation", func(t *testing.T) {
		// VWAP = 100, Close = 95
		bar := makeTestBar(95, 1000, 100000, 60, 60000, 40000, 500)
		result := l1.Compute(bar)

		if result.VWAPDeviation >= 0 {
			t.Errorf("expected negative vwap_deviation, got %v", result.VWAPDeviation)
		}
	})

	t.Run("zero volume", func(t *testing.T) {
		bar := makeTestBar(100, 0, 0, 60, 0, 0, 0)
		result := l1.Compute(bar)

		// Should handle gracefully (no division by zero)
		if math.IsNaN(result.VWAPDeviation) || math.IsInf(result.VWAPDeviation, 0) {
			t.Errorf("vwap_deviation should not be NaN or Inf for zero volume")
		}
	})
}

func TestL1_VolumeImbalance(t *testing.T) {
	l1 := NewL1()

	t.Run("buy dominant", func(t *testing.T) {
		// Buy = 80000, Sell = 20000, Net = 60000, Total = 100000
		// Imbalance = 60000 / 100000 = 0.6
		bar := makeTestBar(100, 1000, 100000, 60, 80000, 20000, 500)
		result := l1.Compute(bar)

		expected := 60000.0 / (100000.0 + Epsilon)
		if math.Abs(result.VolumeImbalance-expected) > 1e-10 {
			t.Errorf("expected volume_imbalance %v, got %v", expected, result.VolumeImbalance)
		}
	})

	t.Run("sell dominant", func(t *testing.T) {
		// Buy = 20000, Sell = 80000, Net = -60000
		bar := makeTestBar(100, 1000, 100000, 60, 20000, 80000, 500)
		result := l1.Compute(bar)

		if result.VolumeImbalance >= 0 {
			t.Errorf("expected negative volume_imbalance, got %v", result.VolumeImbalance)
		}
	})

	t.Run("balanced", func(t *testing.T) {
		bar := makeTestBar(100, 1000, 100000, 60, 50000, 50000, 500)
		result := l1.Compute(bar)

		if math.Abs(result.VolumeImbalance) > 1e-10 {
			t.Errorf("expected volume_imbalance ~0, got %v", result.VolumeImbalance)
		}
	})
}

func TestL1_BarRange(t *testing.T) {
	l1 := NewL1()

	t.Run("normal range", func(t *testing.T) {
		bar := bars.DynamicDollarBar{
			Open:        100,
			High:        110,
			Low:         90,
			Close:       105,
			Volume:      1000,
			DollarValue: 100000,
			Duration:    60,
		}
		result := l1.Compute(bar)

		// Range = (High - Low) / Close = (110 - 90) / 105
		expected := 20.0 / (105 + Epsilon)
		if math.Abs(result.BarRange-expected) > 1e-10 {
			t.Errorf("expected bar_range %v, got %v", expected, result.BarRange)
		}
	})

	t.Run("zero range (doji)", func(t *testing.T) {
		bar := bars.DynamicDollarBar{
			Open:        100,
			High:        100,
			Low:         100,
			Close:       100,
			Volume:      1000,
			DollarValue: 100000,
			Duration:    60,
		}
		result := l1.Compute(bar)

		if math.Abs(result.BarRange) > 1e-10 {
			t.Errorf("expected bar_range ~0 for doji, got %v", result.BarRange)
		}
	})
}

func TestL1_BarBody(t *testing.T) {
	l1 := NewL1()

	t.Run("bullish bar", func(t *testing.T) {
		bar := bars.DynamicDollarBar{
			Open:        100,
			High:        110,
			Low:         95,
			Close:       108,
			Volume:      1000,
			DollarValue: 100000,
			Duration:    60,
		}
		result := l1.Compute(bar)

		// Body = (Close - Open) / Close = (108 - 100) / 108
		expected := 8.0 / (108 + Epsilon)
		if math.Abs(result.BarBody-expected) > 1e-10 {
			t.Errorf("expected bar_body %v, got %v", expected, result.BarBody)
		}
		if result.BarBody <= 0 {
			t.Error("expected positive bar_body for bullish bar")
		}
	})

	t.Run("bearish bar", func(t *testing.T) {
		bar := bars.DynamicDollarBar{
			Open:        108,
			High:        110,
			Low:         95,
			Close:       100,
			Volume:      1000,
			DollarValue: 100000,
			Duration:    60,
		}
		result := l1.Compute(bar)

		if result.BarBody >= 0 {
			t.Error("expected negative bar_body for bearish bar")
		}
	})

	t.Run("doji", func(t *testing.T) {
		bar := bars.DynamicDollarBar{
			Open:        100,
			High:        105,
			Low:         95,
			Close:       100,
			Volume:      1000,
			DollarValue: 100000,
			Duration:    60,
		}
		result := l1.Compute(bar)

		if math.Abs(result.BarBody) > 1e-10 {
			t.Errorf("expected bar_body ~0 for doji, got %v", result.BarBody)
		}
	})
}

func TestL1_EdgeCases(t *testing.T) {
	l1 := NewL1()

	t.Run("zero duration", func(t *testing.T) {
		bar := makeTestBar(100, 1000, 100000, 0, 50000, 50000, 500)
		result := l1.Compute(bar)

		// Should handle gracefully
		if math.IsNaN(result.LogTradeIntensity) || math.IsInf(result.LogTradeIntensity, 0) {
			t.Error("LogTradeIntensity should handle zero duration")
		}
	})

	t.Run("zero everything", func(t *testing.T) {
		bar := bars.DynamicDollarBar{}
		result := l1.Compute(bar)

		// Should not panic or produce NaN/Inf
		if math.IsNaN(result.LogVolume) {
			t.Error("should handle zero values gracefully")
		}
	})

	t.Run("very large values", func(t *testing.T) {
		bar := makeTestBar(1e10, 1e15, 1e25, 1e6, 5e24, 5e24, 1e12)
		result := l1.Compute(bar)

		if math.IsNaN(result.LogVolume) || math.IsInf(result.LogVolume, 0) {
			t.Error("should handle large values")
		}
	})
}

// Benchmark
func BenchmarkL1_Compute(b *testing.B) {
	l1 := NewL1()
	bar := makeTestBar(100, 1000, 100000, 60, 60000, 40000, 500)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = l1.Compute(bar)
	}
}
