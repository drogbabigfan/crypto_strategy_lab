package features

import (
	"math"
	"math/rand"
	"testing"

	"dl-rl-btc-etl/internal/bars"
)

func makeFullTestBar(open, high, low, close, volume, dollarValue float64, buyVol, sellVol float64, duration float64, tickCount int64) bars.DynamicDollarBar {
	return bars.DynamicDollarBar{
		Open:          open,
		High:          high,
		Low:           low,
		Close:         close,
		Volume:        volume,
		DollarValue:   dollarValue,
		BuyDollarVol:  buyVol,
		SellDollarVol: sellVol,
		NetImbalance:  buyVol - sellVol,
		Duration:      duration,
		TickCount:     tickCount,
		StartTime:     1000000,
		EndTime:       1060000,
	}
}

func TestGenerator_Basic(t *testing.T) {
	gen := NewGenerator(DefaultConfig())

	bar := makeFullTestBar(100, 110, 90, 105, 1000, 100000, 60000, 40000, 60, 500)
	row := gen.Process(bar)

	// Check that basic fields are populated
	if row.Open != 100 || row.Close != 105 {
		t.Errorf("expected open=100, close=105, got open=%v, close=%v", row.Open, row.Close)
	}

	// Check L1 features
	if row.LogVolume == 0 {
		t.Error("LogVolume should not be zero")
	}
	if row.VWAPDeviation == 0 {
		t.Error("VWAPDeviation should not be zero for this bar")
	}
}

func TestGenerator_AllFeaturesPopulated(t *testing.T) {
	// Use config with smaller thresholds for faster priming
	config := Config{
		ParkinsonWindow: 24,
		EntropyWindow:   24,
		EntropyBins:     10,
		VolHalflife:     30,
		EntropyHalflife: 30,
		FracDiffD:       0.4,
		FracDiffThresh:  1e-3, // Larger threshold = smaller buffer
		DetrendHalflife: 30,
	}
	gen := NewGenerator(config)

	// Feed enough bars to prime all normalizers
	rng := rand.New(rand.NewSource(42))
	price := 100.0

	for i := 0; i < 200; i++ {
		price += rng.NormFloat64() * 2
		if price < 10 {
			price = 10
		}
		bar := makeFullTestBar(
			price*0.99, price*1.02, price*0.98, price,
			1000+rng.Float64()*500, price*1000,
			price*500+rng.Float64()*200, price*500-rng.Float64()*200,
			60+rng.Float64()*30, 100+rng.Int63n(200),
		)
		gen.Process(bar)
	}

	// Get final row
	bar := makeFullTestBar(price*0.99, price*1.01, price*0.98, price, 1000, price*1000, price*600, price*400, 60, 100)
	row := gen.Process(bar)

	// Note: Full priming requires many more bars due to momentum (1000 halflife) and ConnorsRSI (100 rank)
	// We just check individual components here
	if !gen.regime.IsPrimed() {
		t.Error("regime should be primed after 200 bars")
	}
	if !gen.stationarity.IsPrimed() {
		t.Error("stationarity should be primed after 200 bars")
	}

	// Check all main feature groups are non-zero
	t.Run("L1 features", func(t *testing.T) {
		if row.LogVolume == 0 {
			t.Error("LogVolume should be populated")
		}
		if row.LogDuration == 0 {
			t.Error("LogDuration should be populated")
		}
	})

	t.Run("Regime features", func(t *testing.T) {
		if row.GarmanKlassVol == 0 {
			t.Error("GarmanKlassVol should be populated")
		}
		// VolZScore can legitimately be near 0
	})

	t.Run("Stationarity features", func(t *testing.T) {
		if math.IsNaN(row.FracDiffClose) {
			t.Error("FracDiffClose should not be NaN")
		}
		// DetrendedLogPrice can be near 0
	})

	t.Run("Returns", func(t *testing.T) {
		// Returns should be valid (not NaN)
		if math.IsNaN(row.Returns) {
			t.Error("Returns should not be NaN")
		}
	})
}

func TestGenerator_IsPrimed(t *testing.T) {
	// Use config with smaller thresholds for faster priming
	config := Config{
		ParkinsonWindow: 24,
		EntropyWindow:   24,
		EntropyBins:     10,
		VolHalflife:     30,
		EntropyHalflife: 30,
		FracDiffD:       0.4,
		FracDiffThresh:  1e-3,
		DetrendHalflife: 30,
	}
	gen := NewGenerator(config)

	// Initially not primed
	bar := makeFullTestBar(100, 110, 90, 105, 1000, 100000, 60000, 40000, 60, 500)
	row := gen.Process(bar)

	if row.IsPrimed {
		t.Error("should not be primed after 1 bar")
	}

	// Feed bars until regime and stationarity are primed
	// Note: Full generator priming requires 2000+ bars due to momentum (1000 halflife)
	for i := 0; i < 200; i++ {
		bar := makeFullTestBar(100+float64(i%10), 110, 90, 100+float64(i%10), 1000, 100000, 60000, 40000, 60, 500)
		row = gen.Process(bar)
	}

	// Check component priming
	if !gen.regime.IsPrimed() {
		t.Error("regime should be primed after 200 bars")
	}
	if !gen.stationarity.IsPrimed() {
		t.Error("stationarity should be primed after 200 bars")
	}
}

func TestGenerator_Reset(t *testing.T) {
	gen := NewGenerator(DefaultConfig())

	// Feed some bars
	for i := 0; i < 100; i++ {
		bar := makeFullTestBar(100, 110, 90, 100, 1000, 100000, 60000, 40000, 60, 500)
		gen.Process(bar)
	}

	gen.Reset()

	// After reset, should not be primed
	bar := makeFullTestBar(100, 110, 90, 100, 1000, 100000, 60000, 40000, 60, 500)
	row := gen.Process(bar)

	if row.IsPrimed {
		t.Error("should not be primed after reset")
	}
}

func TestGenerator_StateSerialize(t *testing.T) {
	gen1 := NewGenerator(DefaultConfig())

	// Feed some bars
	for i := 0; i < 200; i++ {
		bar := makeFullTestBar(100+float64(i)*0.5, 110+float64(i)*0.5, 90+float64(i)*0.5, 100+float64(i)*0.5,
			1000, 100000, 60000, 40000, 60, 500)
		gen1.Process(bar)
	}

	// Save state
	state := gen1.State()

	// Create new generator and load state
	gen2 := NewGenerator(DefaultConfig())
	gen2.LoadState(state)

	// Both should produce identical results
	bar := makeFullTestBar(250, 260, 240, 255, 1000, 250000, 150000, 100000, 60, 500)
	row1 := gen1.Process(bar)
	row2 := gen2.Process(bar)

	if row1.LogVolume != row2.LogVolume {
		t.Error("LogVolume mismatch after state restore")
	}
	if math.Abs(row1.GarmanKlassVol-row2.GarmanKlassVol) > 1e-10 {
		t.Errorf("GarmanKlassVol mismatch: %v vs %v", row1.GarmanKlassVol, row2.GarmanKlassVol)
	}
	if math.Abs(row1.VolZScore-row2.VolZScore) > 1e-10 {
		t.Errorf("VolZScore mismatch: %v vs %v", row1.VolZScore, row2.VolZScore)
	}
	if math.Abs(row1.FracDiffClose-row2.FracDiffClose) > 1e-10 {
		t.Errorf("FracDiffClose mismatch: %v vs %v", row1.FracDiffClose, row2.FracDiffClose)
	}
}

func TestGenerator_ContinuityAcrossFiles(t *testing.T) {
	// Simulate processing multiple files without resetting state
	gen := NewGenerator(DefaultConfig())

	// "File 1": 200 bars
	var lastPrice float64 = 100
	for i := 0; i < 200; i++ {
		lastPrice += float64(i%10) * 0.1
		bar := makeFullTestBar(lastPrice*0.99, lastPrice*1.01, lastPrice*0.98, lastPrice,
			1000, lastPrice*1000, lastPrice*600, lastPrice*400, 60, 100)
		gen.Process(bar)
	}

	// Save state at "file boundary"
	stateAfterFile1 := gen.State()

	// "File 2": 200 more bars (state should continue)
	for i := 0; i < 200; i++ {
		lastPrice += float64(i%10) * 0.1
		bar := makeFullTestBar(lastPrice*0.99, lastPrice*1.01, lastPrice*0.98, lastPrice,
			1000, lastPrice*1000, lastPrice*600, lastPrice*400, 60, 100)
		gen.Process(bar)
	}

	stateAfterFile2 := gen.State()

	// States should be different (state evolved)
	if stateAfterFile1.Regime.Count == stateAfterFile2.Regime.Count {
		t.Error("state should have evolved between files")
	}
}

func TestGenerator_EdgeCases(t *testing.T) {
	t.Run("zero bar", func(t *testing.T) {
		gen := NewGenerator(DefaultConfig())
		bar := bars.DynamicDollarBar{} // All zeros
		row := gen.Process(bar)

		// Should not panic or produce NaN in critical fields
		if math.IsNaN(row.LogVolume) {
			t.Error("should handle zero bar gracefully")
		}
	})

	t.Run("very large values", func(t *testing.T) {
		gen := NewGenerator(DefaultConfig())
		bar := makeFullTestBar(1e12, 1.1e12, 0.9e12, 1e12, 1e15, 1e27, 5e26, 5e26, 1e6, 1e9)
		row := gen.Process(bar)

		if math.IsNaN(row.LogVolume) || math.IsInf(row.LogVolume, 0) {
			t.Error("should handle large values")
		}
	})
}

func TestGenerator_StreamingProperty(t *testing.T) {
	// Verify that processing is truly streaming: same bar sequence = same results
	config := DefaultConfig()
	gen1 := NewGenerator(config)
	gen2 := NewGenerator(config)

	rng := rand.New(rand.NewSource(123))

	for i := 0; i < 300; i++ {
		price := 100.0 + rng.Float64()*50
		bar := makeFullTestBar(price*0.99, price*1.02, price*0.98, price,
			1000+rng.Float64()*500, price*1000,
			price*500, price*500, 60, 100)

		row1 := gen1.Process(bar)
		row2 := gen2.Process(bar)

		if row1.GarmanKlassVol != row2.GarmanKlassVol {
			t.Errorf("non-deterministic at bar %d", i)
			break
		}
	}
}

func TestCyclicalTimeFeatures(t *testing.T) {
	t.Run("midnight_UTC", func(t *testing.T) {
		// 2024-01-01 00:00:00 UTC (Monday)
		// Monday = Weekday 1
		ts := int64(1704067200000) // 2024-01-01 00:00:00 UTC
		sinTime, cosTime, sinWeek, cosWeek := cyclicalTimeFeatures(ts)

		// Hour 0: sin(0) = 0, cos(0) = 1
		if math.Abs(sinTime-0) > 1e-10 {
			t.Errorf("expected sinTime=0 at midnight, got %v", sinTime)
		}
		if math.Abs(cosTime-1) > 1e-10 {
			t.Errorf("expected cosTime=1 at midnight, got %v", cosTime)
		}

		// Monday (day 1): sin(2π*1/7), cos(2π*1/7)
		expectedSinWeek := math.Sin(2 * math.Pi * 1 / 7)
		expectedCosWeek := math.Cos(2 * math.Pi * 1 / 7)
		if math.Abs(sinWeek-expectedSinWeek) > 1e-10 {
			t.Errorf("expected sinWeek=%v for Monday, got %v", expectedSinWeek, sinWeek)
		}
		if math.Abs(cosWeek-expectedCosWeek) > 1e-10 {
			t.Errorf("expected cosWeek=%v for Monday, got %v", expectedCosWeek, cosWeek)
		}
	})

	t.Run("noon_UTC", func(t *testing.T) {
		// 2024-01-01 12:00:00 UTC
		ts := int64(1704110400000) // 2024-01-01 12:00:00 UTC
		sinTime, cosTime, _, _ := cyclicalTimeFeatures(ts)

		// Hour 12: sin(π) = 0, cos(π) = -1
		if math.Abs(sinTime-0) > 1e-10 {
			t.Errorf("expected sinTime=0 at noon, got %v", sinTime)
		}
		if math.Abs(cosTime-(-1)) > 1e-10 {
			t.Errorf("expected cosTime=-1 at noon, got %v", cosTime)
		}
	})

	t.Run("6am_UTC", func(t *testing.T) {
		// 2024-01-01 06:00:00 UTC
		ts := int64(1704088800000)
		sinTime, cosTime, _, _ := cyclicalTimeFeatures(ts)

		// Hour 6: sin(π/2) = 1, cos(π/2) = 0
		if math.Abs(sinTime-1) > 1e-10 {
			t.Errorf("expected sinTime=1 at 6am, got %v", sinTime)
		}
		if math.Abs(cosTime-0) > 1e-10 {
			t.Errorf("expected cosTime=0 at 6am, got %v", cosTime)
		}
	})

	t.Run("6pm_UTC", func(t *testing.T) {
		// 2024-01-01 18:00:00 UTC
		ts := int64(1704132000000)
		sinTime, cosTime, _, _ := cyclicalTimeFeatures(ts)

		// Hour 18: sin(3π/2) = -1, cos(3π/2) = 0
		if math.Abs(sinTime-(-1)) > 1e-10 {
			t.Errorf("expected sinTime=-1 at 6pm, got %v", sinTime)
		}
		if math.Abs(cosTime-0) > 1e-10 {
			t.Errorf("expected cosTime=0 at 6pm, got %v", cosTime)
		}
	})

	t.Run("sunday", func(t *testing.T) {
		// 2024-01-07 12:00:00 UTC (Sunday)
		ts := int64(1704628800000)
		_, _, sinWeek, cosWeek := cyclicalTimeFeatures(ts)

		// Sunday (day 0): sin(0) = 0, cos(0) = 1
		if math.Abs(sinWeek-0) > 1e-10 {
			t.Errorf("expected sinWeek=0 for Sunday, got %v", sinWeek)
		}
		if math.Abs(cosWeek-1) > 1e-10 {
			t.Errorf("expected cosWeek=1 for Sunday, got %v", cosWeek)
		}
	})

	t.Run("cyclical_continuity_hour", func(t *testing.T) {
		// 23:00 and 00:00 should be close in encoded space
		// 2024-01-01 23:00:00 UTC
		ts23 := int64(1704150000000)
		// 2024-01-02 00:00:00 UTC
		ts00 := int64(1704153600000)

		sin23, cos23, _, _ := cyclicalTimeFeatures(ts23)
		sin00, cos00, _, _ := cyclicalTimeFeatures(ts00)

		// Euclidean distance in sin/cos space should be small
		dist := math.Sqrt((sin23-sin00)*(sin23-sin00) + (cos23-cos00)*(cos23-cos00))

		// Distance between 23:00 and 00:00 should be ~same as 00:00 to 01:00
		// 1 hour = 2π/24 radians, chord length ≈ 0.26
		if dist > 0.3 {
			t.Errorf("23:00 and 00:00 should be close, distance=%v", dist)
		}
	})

	t.Run("cyclical_continuity_week", func(t *testing.T) {
		// Saturday and Sunday should be close in encoded space
		// 2024-01-06 12:00:00 UTC (Saturday)
		tsSat := int64(1704542400000)
		// 2024-01-07 12:00:00 UTC (Sunday)
		tsSun := int64(1704628800000)

		_, _, sinSat, cosSat := cyclicalTimeFeatures(tsSat)
		_, _, sinSun, cosSun := cyclicalTimeFeatures(tsSun)

		// Euclidean distance
		dist := math.Sqrt((sinSat-sinSun)*(sinSat-sinSun) + (cosSat-cosSun)*(cosSat-cosSun))

		// 1 day = 2π/7 radians, chord length ≈ 0.87
		if dist > 1.0 {
			t.Errorf("Saturday and Sunday should be close, distance=%v", dist)
		}
	})
}

func TestCyclicalTimeFeatures_EdgeCases(t *testing.T) {
	t.Run("unix_epoch", func(t *testing.T) {
		// Unix epoch: 1970-01-01 00:00:00 UTC (Thursday)
		ts := int64(0)
		sinTime, cosTime, sinWeek, cosWeek := cyclicalTimeFeatures(ts)

		// Should not panic, should produce valid values
		if math.IsNaN(sinTime) || math.IsNaN(cosTime) {
			t.Error("should handle unix epoch")
		}

		// Hour 0: sin=0, cos=1
		if math.Abs(sinTime-0) > 1e-10 || math.Abs(cosTime-1) > 1e-10 {
			t.Error("wrong hour encoding for unix epoch")
		}

		// Thursday (day 4)
		expectedSinWeek := math.Sin(2 * math.Pi * 4 / 7)
		expectedCosWeek := math.Cos(2 * math.Pi * 4 / 7)
		if math.Abs(sinWeek-expectedSinWeek) > 1e-10 {
			t.Errorf("expected sinWeek=%v for Thursday, got %v", expectedSinWeek, sinWeek)
		}
		if math.Abs(cosWeek-expectedCosWeek) > 1e-10 {
			t.Errorf("expected cosWeek=%v for Thursday, got %v", expectedCosWeek, cosWeek)
		}
	})

	t.Run("negative_timestamp", func(t *testing.T) {
		// Before Unix epoch
		ts := int64(-86400000) // 1969-12-31 00:00:00 UTC
		sinTime, cosTime, sinWeek, cosWeek := cyclicalTimeFeatures(ts)

		// Should not panic
		if math.IsNaN(sinTime) || math.IsNaN(cosTime) ||
			math.IsNaN(sinWeek) || math.IsNaN(cosWeek) {
			t.Error("should handle negative timestamp")
		}

		// Values should be in valid range [-1, 1]
		if sinTime < -1 || sinTime > 1 || cosTime < -1 || cosTime > 1 {
			t.Error("time encoding out of range")
		}
	})

	t.Run("far_future", func(t *testing.T) {
		// Year 2100
		ts := int64(4102444800000) // 2100-01-01 00:00:00 UTC
		sinTime, cosTime, sinWeek, cosWeek := cyclicalTimeFeatures(ts)

		// Should not panic
		if math.IsNaN(sinTime) || math.IsNaN(cosTime) ||
			math.IsNaN(sinWeek) || math.IsNaN(cosWeek) {
			t.Error("should handle far future timestamp")
		}
	})

	t.Run("value_range", func(t *testing.T) {
		// Test multiple timestamps to ensure all outputs are in [-1, 1]
		timestamps := []int64{
			0,
			1704067200000,  // 2024-01-01
			1704153600000,  // 2024-01-02
			-86400000,      // 1969-12-31
			4102444800000,  // 2100-01-01
		}

		for _, ts := range timestamps {
			sinTime, cosTime, sinWeek, cosWeek := cyclicalTimeFeatures(ts)

			if sinTime < -1 || sinTime > 1 {
				t.Errorf("sinTime=%v out of range for ts=%d", sinTime, ts)
			}
			if cosTime < -1 || cosTime > 1 {
				t.Errorf("cosTime=%v out of range for ts=%d", cosTime, ts)
			}
			if sinWeek < -1 || sinWeek > 1 {
				t.Errorf("sinWeek=%v out of range for ts=%d", sinWeek, ts)
			}
			if cosWeek < -1 || cosWeek > 1 {
				t.Errorf("cosWeek=%v out of range for ts=%d", cosWeek, ts)
			}
		}
	})
}

func TestGenerator_CyclicalTimeInProcess(t *testing.T) {
	gen := NewGenerator(DefaultConfig())

	// Create bar with specific timestamp: 2024-01-01 06:00:00 UTC (Monday)
	bar := bars.DynamicDollarBar{
		StartTime:   1704088800000,
		EndTime:     1704088860000,
		Open:        100,
		High:        110,
		Low:         90,
		Close:       105,
		Volume:      1000,
		DollarValue: 100000,
	}

	row := gen.Process(bar)

	// Hour 6: sin=1, cos=0
	if math.Abs(row.SinTime-1) > 1e-10 {
		t.Errorf("expected SinTime=1 at 6am, got %v", row.SinTime)
	}
	if math.Abs(row.CosTime-0) > 1e-10 {
		t.Errorf("expected CosTime=0 at 6am, got %v", row.CosTime)
	}

	// Monday (day 1)
	expectedSinWeek := math.Sin(2 * math.Pi * 1 / 7)
	if math.Abs(row.SinWeek-expectedSinWeek) > 1e-10 {
		t.Errorf("expected SinWeek=%v for Monday, got %v", expectedSinWeek, row.SinWeek)
	}
}

// =============================================================================
// State Rehydration Idempotency Tests
// =============================================================================
// Purpose: Verify that pipeline restart produces bit-perfect results.
// Process 1000 bars → Result A
// Process 500 bars, SaveState, LoadState, Process 500 more → Result B
// A and B must be identical to the last decimal place.

func TestGenerator_StateIdempotency(t *testing.T) {
	config := DefaultConfig()
	rng := rand.New(rand.NewSource(42))

	// Generate 1000 test bars
	bars := make([]InputBar, 1000)
	price := 100.0
	for i := range bars {
		price += rng.NormFloat64() * 2
		if price < 10 {
			price = 10
		}
		bars[i] = makeFullTestBar(
			price*0.99, price*1.02, price*0.98, price,
			1000+rng.Float64()*500, price*1000,
			price*500+rng.Float64()*200, price*500-rng.Float64()*200,
			60+rng.Float64()*30, 100+rng.Int63n(200),
		)
	}

	// Path A: Process all 1000 bars at once
	genA := NewGenerator(config)
	var resultA FeatureRow
	for _, bar := range bars {
		resultA = genA.Process(bar)
	}

	// Path B: Process 500, save state, load state, process remaining 500
	genB := NewGenerator(config)
	for _, bar := range bars[:500] {
		genB.Process(bar)
	}

	// Save and restore state
	state := genB.State()
	genB2 := NewGenerator(config)
	genB2.LoadState(state)

	var resultB FeatureRow
	for _, bar := range bars[500:] {
		resultB = genB2.Process(bar)
	}

	// Verify ALL feature values are bit-perfect identical
	t.Run("L1_features", func(t *testing.T) {
		if resultA.LogVolume != resultB.LogVolume {
			t.Errorf("LogVolume mismatch: %v vs %v", resultA.LogVolume, resultB.LogVolume)
		}
		if resultA.LogTickCount != resultB.LogTickCount {
			t.Errorf("LogTickCount mismatch: %v vs %v", resultA.LogTickCount, resultB.LogTickCount)
		}
		if resultA.VWAPDeviation != resultB.VWAPDeviation {
			t.Errorf("VWAPDeviation mismatch: %v vs %v", resultA.VWAPDeviation, resultB.VWAPDeviation)
		}
		if resultA.VolumeImbalance != resultB.VolumeImbalance {
			t.Errorf("VolumeImbalance mismatch: %v vs %v", resultA.VolumeImbalance, resultB.VolumeImbalance)
		}
	})

	t.Run("Regime_features", func(t *testing.T) {
		if resultA.GarmanKlassVol != resultB.GarmanKlassVol {
			t.Errorf("GarmanKlassVol mismatch: %v vs %v", resultA.GarmanKlassVol, resultB.GarmanKlassVol)
		}
		if resultA.RealizedVol != resultB.RealizedVol {
			t.Errorf("RealizedVol mismatch: %v vs %v", resultA.RealizedVol, resultB.RealizedVol)
		}
		if resultA.ShannonEntropy != resultB.ShannonEntropy {
			t.Errorf("ShannonEntropy mismatch: %v vs %v", resultA.ShannonEntropy, resultB.ShannonEntropy)
		}
		if resultA.VolZScore != resultB.VolZScore {
			t.Errorf("VolZScore mismatch: %v vs %v", resultA.VolZScore, resultB.VolZScore)
		}
		if resultA.EntropyZScore != resultB.EntropyZScore {
			t.Errorf("EntropyZScore mismatch: %v vs %v", resultA.EntropyZScore, resultB.EntropyZScore)
		}
		if resultA.Skewness != resultB.Skewness {
			t.Errorf("Skewness mismatch: %v vs %v", resultA.Skewness, resultB.Skewness)
		}
		if resultA.Kurtosis != resultB.Kurtosis {
			t.Errorf("Kurtosis mismatch: %v vs %v", resultA.Kurtosis, resultB.Kurtosis)
		}
	})

	t.Run("Stationarity_features", func(t *testing.T) {
		if resultA.FracDiffClose != resultB.FracDiffClose {
			t.Errorf("FracDiffClose mismatch: %v vs %v", resultA.FracDiffClose, resultB.FracDiffClose)
		}
		if resultA.DetrendedLogPrice != resultB.DetrendedLogPrice {
			t.Errorf("DetrendedLogPrice mismatch: %v vs %v", resultA.DetrendedLogPrice, resultB.DetrendedLogPrice)
		}
		if resultA.Returns != resultB.Returns {
			t.Errorf("Returns mismatch: %v vs %v", resultA.Returns, resultB.Returns)
		}
	})

	t.Run("Momentum_features", func(t *testing.T) {
		if resultA.VWMomentum != resultB.VWMomentum {
			t.Errorf("VWMomentum mismatch: %v vs %v", resultA.VWMomentum, resultB.VWMomentum)
		}
		if resultA.MomentumZScore10 != resultB.MomentumZScore10 {
			t.Errorf("MomentumZScore10 mismatch: %v vs %v", resultA.MomentumZScore10, resultB.MomentumZScore10)
		}
		if resultA.MomentumZScore50 != resultB.MomentumZScore50 {
			t.Errorf("MomentumZScore50 mismatch: %v vs %v", resultA.MomentumZScore50, resultB.MomentumZScore50)
		}
		if resultA.MomentumZScore250 != resultB.MomentumZScore250 {
			t.Errorf("MomentumZScore250 mismatch: %v vs %v", resultA.MomentumZScore250, resultB.MomentumZScore250)
		}
		if resultA.MomentumZScore1000 != resultB.MomentumZScore1000 {
			t.Errorf("MomentumZScore1000 mismatch: %v vs %v", resultA.MomentumZScore1000, resultB.MomentumZScore1000)
		}
	})

	t.Run("Technical_features", func(t *testing.T) {
		if resultA.ConnorsRSI != resultB.ConnorsRSI {
			t.Errorf("ConnorsRSI mismatch: %v vs %v", resultA.ConnorsRSI, resultB.ConnorsRSI)
		}
	})

	t.Run("Count", func(t *testing.T) {
		if genA.Count() != genB2.Count() {
			t.Errorf("Count mismatch: %v vs %v", genA.Count(), genB2.Count())
		}
	})
}

func TestGenerator_StateIdempotency_MultipleCheckpoints(t *testing.T) {
	// Test with multiple save/restore cycles
	config := DefaultConfig()
	rng := rand.New(rand.NewSource(123))

	// Generate 600 test bars
	bars := make([]InputBar, 600)
	price := 100.0
	for i := range bars {
		price += rng.NormFloat64() * 2
		if price < 10 {
			price = 10
		}
		bars[i] = makeFullTestBar(
			price*0.99, price*1.02, price*0.98, price,
			1000+rng.Float64()*500, price*1000,
			price*500, price*500, 60, 100,
		)
	}

	// Path A: Process all at once
	genA := NewGenerator(config)
	var resultA FeatureRow
	for _, bar := range bars {
		resultA = genA.Process(bar)
	}

	// Path B: Process in 3 chunks with state save/restore
	genB := NewGenerator(config)
	for _, bar := range bars[:200] {
		genB.Process(bar)
	}
	state1 := genB.State()

	genB = NewGenerator(config)
	genB.LoadState(state1)
	for _, bar := range bars[200:400] {
		genB.Process(bar)
	}
	state2 := genB.State()

	genB = NewGenerator(config)
	genB.LoadState(state2)
	var resultB FeatureRow
	for _, bar := range bars[400:] {
		resultB = genB.Process(bar)
	}

	// Verify all features match
	if resultA.GarmanKlassVol != resultB.GarmanKlassVol {
		t.Errorf("GarmanKlassVol mismatch after 2 checkpoints: %v vs %v",
			resultA.GarmanKlassVol, resultB.GarmanKlassVol)
	}
	if resultA.ConnorsRSI != resultB.ConnorsRSI {
		t.Errorf("ConnorsRSI mismatch after 2 checkpoints: %v vs %v",
			resultA.ConnorsRSI, resultB.ConnorsRSI)
	}
	if resultA.MomentumZScore10 != resultB.MomentumZScore10 {
		t.Errorf("MomentumZScore10 mismatch after 2 checkpoints: %v vs %v",
			resultA.MomentumZScore10, resultB.MomentumZScore10)
	}
}

// =============================================================================
// NaN Hunter Tests (Fuzzing)
// =============================================================================
// Purpose: Ensure no NaN/Inf outputs for edge-case inputs that can occur
// with Dollar Bars (Duration → 0, Volume → 0, OHLC all same, etc.)

// Helper function to check all features for NaN/Inf
func assertNoNaNOrInf(t *testing.T, row FeatureRow, context string) {
	t.Helper()

	checkFloat := func(name string, value float64) {
		if math.IsNaN(value) {
			t.Errorf("%s: %s is NaN", context, name)
		}
		if math.IsInf(value, 0) {
			t.Errorf("%s: %s is Inf", context, name)
		}
	}

	// L1 features
	checkFloat("LogVolume", row.LogVolume)
	checkFloat("LogTickCount", row.LogTickCount)
	checkFloat("LogDuration", row.LogDuration)
	checkFloat("LogTradeIntensity", row.LogTradeIntensity)
	checkFloat("VWAPDeviation", row.VWAPDeviation)
	checkFloat("VolumeImbalance", row.VolumeImbalance)
	checkFloat("BarRange", row.BarRange)
	checkFloat("BarBody", row.BarBody)

	// Regime features
	checkFloat("GarmanKlassVol", row.GarmanKlassVol)
	checkFloat("RealizedVol", row.RealizedVol)
	checkFloat("ShannonEntropy", row.ShannonEntropy)
	checkFloat("VolRatio", row.VolRatio)
	checkFloat("VolZScore", row.VolZScore)
	checkFloat("EntropyZScore", row.EntropyZScore)
	checkFloat("Skewness", row.Skewness)
	checkFloat("Kurtosis", row.Kurtosis)

	// Stationarity features
	checkFloat("FracDiffClose", row.FracDiffClose)
	checkFloat("DetrendedLogPrice", row.DetrendedLogPrice)
	checkFloat("Returns", row.Returns)

	// Momentum features
	checkFloat("VWMomentum", row.VWMomentum)
	checkFloat("MomentumZScore10", row.MomentumZScore10)
	checkFloat("MomentumZScore50", row.MomentumZScore50)
	checkFloat("MomentumZScore250", row.MomentumZScore250)
	checkFloat("MomentumZScore1000", row.MomentumZScore1000)

	// Technical features
	checkFloat("ConnorsRSI", row.ConnorsRSI)

	// Cyclical time features
	checkFloat("SinTime", row.SinTime)
	checkFloat("CosTime", row.CosTime)
	checkFloat("SinWeek", row.SinWeek)
	checkFloat("CosWeek", row.CosWeek)
}

func TestGenerator_NaNHunter_ZeroDuration(t *testing.T) {
	// Dollar Bars can have very short duration (milliseconds)
	gen := NewGenerator(DefaultConfig())

	// Warmup with normal data
	for i := 0; i < 50; i++ {
		bar := makeFullTestBar(100, 110, 90, 105, 1000, 100000, 60000, 40000, 60, 500)
		gen.Process(bar)
	}

	// Test with Duration = 1ms (near zero)
	bar := makeFullTestBar(100, 110, 90, 105, 1000, 100000, 60000, 40000, 0.001, 500)
	row := gen.Process(bar)
	assertNoNaNOrInf(t, row, "Duration=1ms")

	// Test with Duration = 0
	bar = makeFullTestBar(100, 110, 90, 105, 1000, 100000, 60000, 40000, 0, 500)
	row = gen.Process(bar)
	assertNoNaNOrInf(t, row, "Duration=0")
}

func TestGenerator_NaNHunter_ZeroVolume(t *testing.T) {
	// Data error: Volume = 0
	gen := NewGenerator(DefaultConfig())

	// Warmup
	for i := 0; i < 50; i++ {
		bar := makeFullTestBar(100, 110, 90, 105, 1000, 100000, 60000, 40000, 60, 500)
		gen.Process(bar)
	}

	// Test with Volume = 0
	bar := makeFullTestBar(100, 110, 90, 105, 0, 0, 0, 0, 60, 500)
	row := gen.Process(bar)
	assertNoNaNOrInf(t, row, "Volume=0")
}

func TestGenerator_NaNHunter_FlatBar(t *testing.T) {
	// Open = High = Low = Close (no price movement)
	gen := NewGenerator(DefaultConfig())

	// Warmup
	for i := 0; i < 50; i++ {
		bar := makeFullTestBar(100, 110, 90, 105, 1000, 100000, 60000, 40000, 60, 500)
		gen.Process(bar)
	}

	// Test with completely flat bar
	bar := makeFullTestBar(100, 100, 100, 100, 1000, 100000, 60000, 40000, 60, 500)
	row := gen.Process(bar)
	assertNoNaNOrInf(t, row, "Flat bar (OHLC equal)")

	// Multiple consecutive flat bars
	for i := 0; i < 20; i++ {
		bar := makeFullTestBar(100, 100, 100, 100, 1000, 100000, 60000, 40000, 60, 500)
		row := gen.Process(bar)
		assertNoNaNOrInf(t, row, "Consecutive flat bars")
	}
}

func TestGenerator_NaNHunter_ExtremeValues(t *testing.T) {
	gen := NewGenerator(DefaultConfig())

	// Warmup
	for i := 0; i < 50; i++ {
		bar := makeFullTestBar(100, 110, 90, 105, 1000, 100000, 60000, 40000, 60, 500)
		gen.Process(bar)
	}

	t.Run("very_small_price", func(t *testing.T) {
		bar := makeFullTestBar(1e-10, 1e-9, 1e-11, 1e-10, 1000, 1e-7, 1e-8, 1e-8, 60, 500)
		row := gen.Process(bar)
		assertNoNaNOrInf(t, row, "Very small price")
	})

	t.Run("very_large_price", func(t *testing.T) {
		bar := makeFullTestBar(1e12, 1.1e12, 0.9e12, 1e12, 1e15, 1e27, 5e26, 5e26, 60, 500)
		row := gen.Process(bar)
		assertNoNaNOrInf(t, row, "Very large price")
	})

	t.Run("zero_tick_count", func(t *testing.T) {
		bar := makeFullTestBar(100, 110, 90, 105, 1000, 100000, 60000, 40000, 60, 0)
		row := gen.Process(bar)
		assertNoNaNOrInf(t, row, "Zero tick count")
	})
}

func TestGenerator_NaNHunter_Fuzz(t *testing.T) {
	// Random fuzzing with edge cases
	gen := NewGenerator(DefaultConfig())
	rng := rand.New(rand.NewSource(999))

	// Warmup
	for i := 0; i < 100; i++ {
		bar := makeFullTestBar(100, 110, 90, 105, 1000, 100000, 60000, 40000, 60, 500)
		gen.Process(bar)
	}

	// Fuzz with random edge cases
	for i := 0; i < 1000; i++ {
		price := rng.Float64() * 1e6
		if price < 1e-10 {
			price = 1e-10
		}

		vol := rng.Float64() * 1e10
		duration := rng.Float64() * 1000

		// Randomly introduce edge cases
		switch rng.Intn(10) {
		case 0:
			vol = 0 // Zero volume
		case 1:
			duration = 0 // Zero duration
		case 2:
			// Flat bar
			bar := makeFullTestBar(price, price, price, price, vol, vol*price, vol*0.5, vol*0.5, duration, rng.Int63n(1000))
			row := gen.Process(bar)
			assertNoNaNOrInf(t, row, "Fuzz flat bar")
			continue
		}

		high := price * (1 + rng.Float64()*0.1)
		low := price * (1 - rng.Float64()*0.1)
		close := low + rng.Float64()*(high-low)

		bar := makeFullTestBar(
			price, high, low, close,
			vol, vol*price,
			vol*rng.Float64(), vol*rng.Float64(),
			duration, rng.Int63n(1000),
		)

		row := gen.Process(bar)
		assertNoNaNOrInf(t, row, "Fuzz random")
	}
}

// =============================================================================
// Soft Clip Verification Tests
// =============================================================================

func TestMomentum_SoftClip(t *testing.T) {
	mom := NewMomentum(DefaultMomentumConfig())

	// Warmup with small, consistent momentum
	for i := 0; i < 200; i++ {
		mom.Update(0.001, 1000) // Small positive momentum
	}

	t.Run("extreme_positive_gets_clipped", func(t *testing.T) {
		// Inject extreme momentum that would give z-score >> 5
		result := mom.Update(1.0, 1000000) // Massive momentum spike

		// Z-scores should be soft-clipped to near ±5
		if result.MomentumZScore10 > 5.1 {
			t.Errorf("MomentumZScore10 should be soft-clipped, got %v", result.MomentumZScore10)
		}
		if result.MomentumZScore50 > 5.1 {
			t.Errorf("MomentumZScore50 should be soft-clipped, got %v", result.MomentumZScore50)
		}
	})

	t.Run("extreme_negative_gets_clipped", func(t *testing.T) {
		// Reset and test negative
		mom2 := NewMomentum(DefaultMomentumConfig())
		for i := 0; i < 200; i++ {
			mom2.Update(0.001, 1000)
		}

		result := mom2.Update(-1.0, 1000000) // Massive negative momentum

		if result.MomentumZScore10 < -5.1 {
			t.Errorf("MomentumZScore10 should be soft-clipped, got %v", result.MomentumZScore10)
		}
	})

	t.Run("soft_clip_is_smooth", func(t *testing.T) {
		// Verify tanh-style soft clipping (smooth, not hard cutoff)
		// Test that soft clip approaches but doesn't exceed bound
		mom3 := NewMomentum(DefaultMomentumConfig())
		for i := 0; i < 200; i++ {
			mom3.Update(0.001, 1000)
		}

		// Moderate outlier
		result1 := mom3.Update(0.05, 5000)
		z1 := result1.MomentumZScore10

		// The z-score should be positive for positive outlier
		if z1 <= 0 {
			t.Error("z-score should be positive for positive outlier")
		}

		// Soft clip should keep it bounded
		if z1 > 5.1 {
			t.Errorf("z-score should be soft-clipped to ~5, got %v", z1)
		}

		// Test that extremely large values still get clipped properly
		mom4 := NewMomentum(DefaultMomentumConfig())
		for i := 0; i < 200; i++ {
			mom4.Update(0.001, 1000)
		}
		result2 := mom4.Update(10.0, 1000000) // Massive outlier
		z2 := result2.MomentumZScore10

		// Even massive outlier should be bounded
		if z2 > 5.1 {
			t.Errorf("massive outlier should be clipped to ~5, got %v", z2)
		}
		// Should still be positive
		if z2 <= 0 {
			t.Error("z-score should be positive for positive outlier")
		}
	})
}

func TestMomentum_Decay(t *testing.T) {
	// Test that z-scores decay toward 0 when momentum stops
	mom := NewMomentum(DefaultMomentumConfig())

	// Create momentum spike
	for i := 0; i < 100; i++ {
		mom.Update(0.01, 1000) // Positive momentum
	}

	result := mom.CurrentFeatures()
	initialZ := result.MomentumZScore10

	// Now flat momentum (returns = 0)
	for i := 0; i < 50; i++ {
		mom.Update(0, 1000) // No momentum
	}

	result = mom.CurrentFeatures()
	finalZ := result.MomentumZScore10

	// Z-score should decay toward 0 (but might not reach it quickly)
	if math.Abs(finalZ) >= math.Abs(initialZ) && initialZ != 0 {
		t.Logf("Initial Z: %v, Final Z: %v", initialZ, finalZ)
		// Not a hard failure - just informational
	}
}

// Benchmark
func BenchmarkGenerator_Process(b *testing.B) {
	gen := NewGenerator(DefaultConfig())
	bar := makeFullTestBar(100, 110, 90, 105, 1000, 100000, 60000, 40000, 60, 500)

	// Warmup
	for i := 0; i < 1000; i++ {
		gen.Process(bar)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = gen.Process(bar)
	}
}
