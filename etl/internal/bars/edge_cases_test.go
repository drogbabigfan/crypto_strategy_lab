package bars

import (
	"math"
	"testing"
	"time"
)

// ============================================================================
// RISK 1: Boundary Conditions for Dollar Threshold
// ============================================================================

func TestExactlyThresholdAmount(t *testing.T) {
	g := setupGeneratorSkipWarmup(1_000_000) // $1M threshold

	baseTime := time.Date(2020, 1, 15, 0, 0, 0, 0, time.UTC).UnixMilli()

	// Trade exactly at threshold
	bar := g.ProcessTrade(50000, 20.0, 1_000_000, baseTime, true)

	if bar == nil {
		t.Fatal("Bar should be generated when exactly at threshold")
	}

	if bar.DollarValue != 1_000_000 {
		t.Errorf("Expected dollar value 1000000, got %f", bar.DollarValue)
	}
}

func TestSlightlyBelowThreshold(t *testing.T) {
	g := setupGeneratorSkipWarmup(1_000_000)

	baseTime := time.Date(2020, 1, 15, 0, 0, 0, 0, time.UTC).UnixMilli()

	// Trade just below threshold
	bar := g.ProcessTrade(50000, 19.99998, 999_999, baseTime, true)

	if bar != nil {
		t.Error("Bar should NOT be generated when below threshold")
	}

	// Verify accumulator state
	pending := g.GetPendingBar()
	if pending == nil {
		t.Fatal("Should have pending bar")
	}
	if pending.DollarValue != 999_999 {
		t.Errorf("Expected pending dollar value 999999, got %f", pending.DollarValue)
	}
}

func TestSingleTradeExceedsThreshold(t *testing.T) {
	g := setupGeneratorSkipWarmup(1_000_000) // $1M threshold

	baseTime := time.Date(2020, 1, 15, 0, 0, 0, 0, time.UTC).UnixMilli()

	// Single massive trade (whale order)
	bar := g.ProcessTrade(50000, 100.0, 5_000_000, baseTime, true) // $5M trade

	if bar == nil {
		t.Fatal("Bar should be generated for massive single trade")
	}

	if bar.TickCount != 1 {
		t.Errorf("Expected 1 tick, got %d", bar.TickCount)
	}

	if bar.DollarValue != 5_000_000 {
		t.Errorf("Expected $5M, got %f", bar.DollarValue)
	}
}

// ============================================================================
// RISK 2: Zero/Invalid Values
// ============================================================================

func TestZeroQuantityTrade(t *testing.T) {
	g := setupGeneratorSkipWarmup(1_000_000)

	baseTime := time.Date(2020, 1, 15, 0, 0, 0, 0, time.UTC).UnixMilli()

	// Zero quantity trade (should still accumulate)
	bar := g.ProcessTrade(50000, 0, 0, baseTime, true)

	if bar != nil {
		t.Error("Zero trade should not generate bar")
	}

	pending := g.GetPendingBar()
	if pending == nil {
		t.Fatal("Should have pending bar even with zero trade")
	}

	if pending.TickCount != 1 {
		t.Errorf("Zero trade should still count as a tick")
	}
}

func TestZeroPriceTrade(t *testing.T) {
	g := setupGeneratorSkipWarmup(1_000_000)

	baseTime := time.Date(2020, 1, 15, 0, 0, 0, 0, time.UTC).UnixMilli()

	// Zero price (invalid but should handle gracefully)
	g.ProcessTrade(0, 1.0, 0, baseTime, true)

	pending := g.GetPendingBar()
	if pending == nil {
		t.Fatal("Should have pending bar")
	}

	// OHLC should all be zero
	if pending.Open != 0 || pending.High != 0 || pending.Low != 0 || pending.Close != 0 {
		t.Error("Zero price should set all OHLC to zero")
	}
}

func TestNegativeValues(t *testing.T) {
	// This shouldn't happen in real data, but test defensive handling
	g := setupGeneratorSkipWarmup(1_000_000)

	baseTime := time.Date(2020, 1, 15, 0, 0, 0, 0, time.UTC).UnixMilli()

	// Negative price (invalid data)
	g.ProcessTrade(-50000, 1.0, -50000, baseTime, true)

	pending := g.GetPendingBar()
	if pending == nil {
		t.Fatal("Should have pending bar")
	}

	// Note: Current implementation doesn't validate - this documents behavior
	if pending.Open != -50000 {
		t.Logf("Negative price handling: Open=%f", pending.Open)
	}
}

// ============================================================================
// RISK 3: Timestamp Edge Cases
// ============================================================================

func TestOutOfOrderTimestamps(t *testing.T) {
	g := setupGeneratorSkipWarmup(100_000)

	baseTime := time.Date(2020, 1, 15, 12, 0, 0, 0, time.UTC).UnixMilli()

	// Process trades out of order
	g.ProcessTrade(50000, 0.5, 25000, baseTime, true)         // t=0
	g.ProcessTrade(50100, 0.5, 25050, baseTime-1000, true)    // t=-1s (earlier!)
	g.ProcessTrade(50200, 0.5, 25100, baseTime+1000, true)    // t=+1s
	bar := g.ProcessTrade(50300, 0.5, 25150, baseTime+2000, true) // t=+2s

	if bar == nil {
		t.Fatal("Bar should be generated")
	}

	// EndTime should be the last processed, not the max
	// This documents current behavior - may need fixing if order matters
	if bar.EndTime != baseTime+2000 {
		t.Logf("Out-of-order EndTime handling: EndTime=%d, expected=%d",
			bar.EndTime, baseTime+2000)
	}

	// Duration calculation may be incorrect with out-of-order data
	// Currently: EndTime - StartTime, not max - min
	t.Logf("Duration with out-of-order: %.3fs", bar.Duration)
}

func TestMidnightUTCBoundary(t *testing.T) {
	g := setupGeneratorSkipWarmup(100_000)

	// Just before midnight UTC
	beforeMidnight := time.Date(2020, 1, 15, 23, 59, 59, 999_000_000, time.UTC).UnixMilli()
	// Just after midnight UTC
	afterMidnight := time.Date(2020, 1, 16, 0, 0, 0, 1_000_000, time.UTC).UnixMilli()

	g.currentDayStart = getDayStartUTC(beforeMidnight)
	initialThreshold := g.currentThreshold

	// Trade before midnight
	g.ProcessTrade(50000, 0.5, 25000, beforeMidnight, true)

	// Trade after midnight (should trigger day change)
	g.ProcessTrade(50100, 0.5, 25050, afterMidnight, true)

	// Day should have changed
	expectedNewDayStart := getDayStartUTC(afterMidnight)
	if g.currentDayStart != expectedNewDayStart {
		t.Errorf("Day start should update: expected %d, got %d",
			expectedNewDayStart, g.currentDayStart)
	}

	// Threshold should have recalculated
	if g.currentThreshold == initialThreshold && g.currentDayVolume != 0 {
		t.Log("Note: Threshold unchanged after day change (may be expected)")
	}
}

func TestFutureTimestamp(t *testing.T) {
	g := setupGeneratorSkipWarmup(100_000)

	// Far future timestamp (year 2050)
	futureTime := time.Date(2050, 1, 1, 0, 0, 0, 0, time.UTC).UnixMilli()

	// Should still process without panic
	bar := g.ProcessTrade(50000, 2.0, 100_000, futureTime, true)

	if bar == nil {
		t.Fatal("Should generate bar even with future timestamp")
	}
}

func TestVeryOldTimestamp(t *testing.T) {
	// Test 1: Fresh state (currentDayStart=0) should accept any timestamp
	t.Run("FreshStateAcceptsOldTimestamp", func(t *testing.T) {
		g := NewGenerator()
		g.isWarmup = false
		g.currentThreshold = 100_000
		// currentDayStart is 0 (fresh state)

		// Before Bitcoin existed (2008)
		oldTime := time.Date(2008, 1, 1, 0, 0, 0, 0, time.UTC).UnixMilli()

		// Should process without error
		bar, err := g.ProcessTradeWithIDSafe(1, 50000, 2.0, 100_000, oldTime, true)

		if err != nil {
			t.Fatalf("Fresh state should accept old timestamp: %v", err)
		}
		if bar == nil {
			t.Fatal("Should generate bar even with old timestamp")
		}
	})

	// Test 2: State with matching old timestamp should work
	t.Run("MatchingStateAcceptsOldTimestamp", func(t *testing.T) {
		g := NewGenerator()
		g.isWarmup = false
		g.currentThreshold = 100_000
		// Set state to 2008 (matching the trade timestamp)
		g.currentDayStart = time.Date(2008, 1, 1, 0, 0, 0, 0, time.UTC).UnixMilli()

		oldTime := time.Date(2008, 1, 1, 12, 0, 0, 0, time.UTC).UnixMilli()

		bar, err := g.ProcessTradeWithIDSafe(1, 50000, 2.0, 100_000, oldTime, true)

		if err != nil {
			t.Fatalf("Matching state should accept timestamp: %v", err)
		}
		if bar == nil {
			t.Fatal("Should generate bar")
		}
	})

	// Test 3: Future state should reject old timestamp (Fail-Fast)
	t.Run("FutureStateRejectsOldTimestamp", func(t *testing.T) {
		g := setupGeneratorSkipWarmup(100_000) // Sets currentDayStart to 2020

		// Before Bitcoin existed (2008) - 12 years gap
		oldTime := time.Date(2008, 1, 1, 0, 0, 0, 0, time.UTC).UnixMilli()

		// Should be detected as state corruption
		_, err := g.ProcessTradeWithIDSafe(1, 50000, 2.0, 100_000, oldTime, true)

		if err == nil {
			t.Fatal("Future state should reject old timestamp (state corruption)")
		}
	})
}

// ============================================================================
// RISK 4: Warmup Period Edge Cases
// ============================================================================

func TestWarmupWithDataGaps(t *testing.T) {
	g := NewGenerator()

	baseTime := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC).UnixMilli()

	// Process only 7 days - each day needs at least one trade
	for day := 0; day < 7; day++ {
		dayStart := baseTime + int64(day)*MillisPerDay
		g.ProcessTrade(50000, 1.0, 100_000_000, dayStart, true) // $100M
	}

	// Day 7's volume is still being accumulated (not yet saved to buffer)
	// Buffer contains days 0-6's volumes
	if g.dailyVolumeCount != 6 {
		t.Logf("Days in buffer after 7 days: %d (expected 6 - current day not saved yet)", g.dailyVolumeCount)
	}

	if !g.IsWarmup() {
		t.Error("Should still be in warmup with only 7 days")
	}

	// Skip to day 20 (13 days gap) - this triggers day change
	// Day 7's volume is saved when day 20 starts
	day20Start := baseTime + int64(20)*MillisPerDay
	g.ProcessTrade(50000, 1.0, 100_000_000, day20Start, true)

	// Now buffer contains days 0-6 + day 7 (saved on day change) = 7 days
	// Day 20's volume is being accumulated (not saved)
	if g.dailyVolumeCount != 7 {
		t.Errorf("Expected 7 days of data (gap days don't count), got %d", g.dailyVolumeCount)
	}

	// Add one more day to confirm gap handling
	day21Start := baseTime + int64(21)*MillisPerDay
	g.ProcessTrade(50000, 1.0, 100_000_000, day21Start, true)

	// Now day 20's volume is saved = 8 days total
	if g.dailyVolumeCount != 8 {
		t.Errorf("Expected 8 days after day 21 starts, got %d", g.dailyVolumeCount)
	}
}

func TestWarmupExactly14Days(t *testing.T) {
	g := NewGenerator()

	baseTime := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC).UnixMilli()

	// Process exactly 14 days
	for day := 0; day < WarmupDays; day++ {
		dayStart := baseTime + int64(day)*MillisPerDay
		g.ProcessTrade(50000, 1.0, 100_000_000, dayStart, true)
	}

	// Still in warmup until first trade of day 15
	if !g.IsWarmup() {
		t.Error("Should still be in warmup after 14 days until day change")
	}

	// First trade of day 15 should exit warmup
	day15Start := baseTime + int64(WarmupDays)*MillisPerDay
	g.ProcessTrade(50000, 1.0, 1000, day15Start, true)

	if g.IsWarmup() {
		t.Error("Should exit warmup after day 15 starts")
	}
}

func TestWarmupWithZeroVolumeDays(t *testing.T) {
	g := NewGenerator()

	// Manually set up state with zero volume days
	for i := 0; i < WarmupDays; i++ {
		g.dailyVolumes[i] = 0 // Zero volume days
	}
	g.dailyVolumeCount = WarmupDays
	g.isWarmup = false
	g.initializeEMA() // EMA initialized to 0 (SMA of zeros)

	// Threshold calculation with zero volume
	threshold := g.calculateThreshold()

	if threshold != 0 {
		t.Errorf("Threshold with zero volume should be 0, got %f", threshold)
	}

	// EMA should also be 0
	if g.GetEMAVolume() != 0 {
		t.Errorf("EMA with zero volume should be 0, got %f", g.GetEMAVolume())
	}
}

// ============================================================================
// RISK 5: Crash Recovery Edge Cases
// ============================================================================

func TestResumeWithHigherTradeID(t *testing.T) {
	g := NewGenerator()

	// Set last processed to very high ID
	g.lastProcessedTradeID = 9_999_999_999

	// Try to process lower ID trades (should all be skipped)
	for i := int64(1); i <= 100; i++ {
		if !g.ShouldSkipTrade(i) {
			t.Errorf("Trade ID %d should be skipped", i)
		}
	}

	// Next valid trade
	if g.ShouldSkipTrade(10_000_000_000) {
		t.Error("Trade ID 10000000000 should NOT be skipped")
	}
}

func TestTradeIDOverflow(t *testing.T) {
	g := NewGenerator()

	// Set to near max int64
	g.lastProcessedTradeID = math.MaxInt64 - 10

	// Trade IDs GREATER than lastProcessedTradeID should NOT be skipped (they're new)
	// Trade IDs LESS OR EQUAL should be skipped (already processed)

	// MaxInt64 - 5 > MaxInt64 - 10, so this is a NEW trade (should NOT be skipped)
	if g.ShouldSkipTrade(math.MaxInt64 - 5) {
		t.Error("Trade ID greater than lastProcessed should NOT be skipped")
	}

	// MaxInt64 - 15 < MaxInt64 - 10, so this is an OLD trade (should be skipped)
	if !g.ShouldSkipTrade(math.MaxInt64 - 15) {
		t.Error("Trade ID less than lastProcessed should be skipped")
	}

	// Test at exactly MaxInt64
	g.lastProcessedTradeID = math.MaxInt64

	// Any trade <= MaxInt64 should be skipped
	if !g.ShouldSkipTrade(math.MaxInt64) {
		t.Error("Trade at MaxInt64 should be skipped when lastProcessed is MaxInt64")
	}

	if !g.ShouldSkipTrade(math.MaxInt64 - 1) {
		t.Error("Trade below MaxInt64 should be skipped when lastProcessed is MaxInt64")
	}

	// Note: There's no valid trade ID > MaxInt64, so this is the boundary
	t.Log("Successfully handled MaxInt64 boundary for trade IDs")
}

func TestResumeWithCorruptedAccumulator(t *testing.T) {
	// Simulate corrupted state with invalid accumulator values
	state := GeneratorState{
		Accumulator: BarAccumulator{
			StartTimestamp: 0,    // Invalid
			EndTimestamp:   -100, // Invalid
			Open:           0,
			High:           -1, // Invalid
			Low:            math.MaxFloat64,
			Close:          0,
			TickCount:      -5, // Invalid
		},
		IsWarmup:             false,
		CurrentThreshold:     1_000_000,
		LastProcessedTradeID: 1000,
	}

	g := NewGenerator()
	g.ImportState(state)

	// Should not panic when processing new trade
	baseTime := time.Date(2020, 1, 15, 0, 0, 0, 0, time.UTC).UnixMilli()
	g.currentDayStart = getDayStartUTC(baseTime)

	// Process a normal trade - behavior with corrupted state
	bar := g.ProcessTrade(50000, 20.0, 1_000_000, baseTime, true)

	// Document current behavior
	if bar != nil {
		t.Logf("Bar generated despite corrupted state: TickCount=%d", bar.TickCount)
	}
}

// ============================================================================
// RISK 6: Circular Buffer Edge Cases
// ============================================================================

func TestCircularBufferWrapAround(t *testing.T) {
	g := NewGenerator()

	baseTime := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC).UnixMilli()

	// Process 30 days (buffer wraps around twice for 14-day buffer)
	for day := 0; day < 30; day++ {
		dayStart := baseTime + int64(day)*MillisPerDay
		volume := float64((day + 1) * 100_000_000) // Increasing volume
		g.ProcessTrade(50000, 1.0, volume, dayStart, true)

		// Force day change
		if day < 29 {
			nextDay := dayStart + MillisPerDay
			g.ProcessTrade(50000, 0.001, 1, nextDay, true)
		}
	}

	// After 30 days, warmup should be complete and EMA should be set
	if g.IsWarmup() {
		t.Error("Should not be in warmup after 30 days")
	}

	// EMA should be positive (tracking volume over time)
	if g.GetEMAVolume() <= 0 {
		t.Error("EMA should be positive after 30 days")
	}

	// Threshold should be positive
	threshold := g.GetCurrentThreshold()
	if threshold <= 0 {
		t.Error("Threshold should be positive after 30 days")
	}
}

// ============================================================================
// RISK 7: Imbalance Calculation Edge Cases
// ============================================================================

func TestAllBuyTrades(t *testing.T) {
	g := setupGeneratorSkipWarmup(100_000)

	baseTime := time.Date(2020, 1, 15, 0, 0, 0, 0, time.UTC).UnixMilli()

	// All trades are buys
	var bar *DynamicDollarBar
	for i := 0; i < 100; i++ {
		bar = g.ProcessTrade(50000, 0.02, 1000, baseTime+int64(i*1000), true) // All buys
		if bar != nil {
			break
		}
	}

	if bar == nil {
		t.Fatal("Bar should be generated")
	}

	if bar.SellDollarVol != 0 {
		t.Errorf("Expected zero sell volume, got %f", bar.SellDollarVol)
	}

	if bar.NetImbalance != bar.BuyDollarVol {
		t.Errorf("NetImbalance should equal BuyDollarVol when no sells")
	}
}

func TestAllSellTrades(t *testing.T) {
	g := setupGeneratorSkipWarmup(100_000)

	baseTime := time.Date(2020, 1, 15, 0, 0, 0, 0, time.UTC).UnixMilli()

	// All trades are sells
	var bar *DynamicDollarBar
	for i := 0; i < 100; i++ {
		bar = g.ProcessTrade(50000, 0.02, 1000, baseTime+int64(i*1000), false) // All sells
		if bar != nil {
			break
		}
	}

	if bar == nil {
		t.Fatal("Bar should be generated")
	}

	if bar.BuyDollarVol != 0 {
		t.Errorf("Expected zero buy volume, got %f", bar.BuyDollarVol)
	}

	if bar.NetImbalance != -bar.SellDollarVol {
		t.Errorf("NetImbalance should be negative of SellDollarVol when no buys")
	}
}

// ============================================================================
// RISK 8: OHLC Edge Cases
// ============================================================================

func TestOHLCWithSingleTrade(t *testing.T) {
	g := setupGeneratorSkipWarmup(100_000)

	baseTime := time.Date(2020, 1, 15, 0, 0, 0, 0, time.UTC).UnixMilli()

	// Single trade that exceeds threshold
	bar := g.ProcessTrade(50000, 2.0, 100_000, baseTime, true)

	if bar == nil {
		t.Fatal("Bar should be generated")
	}

	// All OHLC should be the same for single trade
	if bar.Open != bar.High || bar.High != bar.Low || bar.Low != bar.Close {
		t.Errorf("Single trade OHLC should all be equal: O=%f H=%f L=%f C=%f",
			bar.Open, bar.High, bar.Low, bar.Close)
	}
}

func TestOHLCDecreasingPrice(t *testing.T) {
	g := setupGeneratorSkipWarmup(100_000)

	baseTime := time.Date(2020, 1, 15, 0, 0, 0, 0, time.UTC).UnixMilli()

	// Decreasing prices
	prices := []float64{50000, 49500, 49000, 48500, 48000}
	var bar *DynamicDollarBar

	for i, price := range prices {
		bar = g.ProcessTrade(price, 0.4, 20000, baseTime+int64(i*1000), true)
		if bar != nil {
			break
		}
	}

	if bar == nil {
		t.Fatal("Bar should be generated")
	}

	if bar.Open != 50000 {
		t.Errorf("Open should be first price 50000, got %f", bar.Open)
	}
	if bar.High != 50000 {
		t.Errorf("High should be 50000, got %f", bar.High)
	}
	if bar.Low != bar.Close {
		t.Errorf("Low should equal Close in decreasing trend, Low=%f Close=%f", bar.Low, bar.Close)
	}
}

func TestOHLCVolatilePrices(t *testing.T) {
	g := setupGeneratorSkipWarmup(100_000)

	baseTime := time.Date(2020, 1, 15, 0, 0, 0, 0, time.UTC).UnixMilli()

	// Volatile prices: up, down, up, down
	prices := []float64{50000, 51000, 49000, 52000, 48000}
	var bar *DynamicDollarBar

	for i, price := range prices {
		bar = g.ProcessTrade(price, 0.4, 20000, baseTime+int64(i*1000), true)
		if bar != nil {
			break
		}
	}

	if bar == nil {
		t.Fatal("Bar should be generated")
	}

	if bar.High != 52000 {
		t.Errorf("High should be max 52000, got %f", bar.High)
	}
	if bar.Low != 48000 {
		t.Errorf("Low should be min 48000, got %f", bar.Low)
	}
}

// ============================================================================
// RISK 9: Flush Edge Cases
// ============================================================================

func TestFlushWithNoTrades(t *testing.T) {
	g := setupGeneratorSkipWarmup(100_000)

	// Flush without any trades
	bar := g.Flush()

	if bar != nil {
		t.Error("Flush with no trades should return nil")
	}
}

func TestFlushDuringWarmup(t *testing.T) {
	g := NewGenerator() // In warmup mode

	baseTime := time.Date(2020, 1, 15, 0, 0, 0, 0, time.UTC).UnixMilli()

	// Add some trades during warmup
	g.ProcessTrade(50000, 1.0, 50000, baseTime, true)

	// Flush during warmup
	bar := g.Flush()

	if bar != nil {
		t.Error("Flush during warmup should return nil")
	}
}

func TestMultipleFlushCalls(t *testing.T) {
	g := setupGeneratorSkipWarmup(100_000)

	baseTime := time.Date(2020, 1, 15, 0, 0, 0, 0, time.UTC).UnixMilli()

	// Add some trades (below threshold)
	g.ProcessTrade(50000, 1.0, 50000, baseTime, true)

	// First flush should return bar
	bar1 := g.Flush()
	if bar1 == nil {
		t.Fatal("First flush should return bar")
	}

	// Second flush should return nil (accumulator reset)
	bar2 := g.Flush()
	if bar2 != nil {
		t.Error("Second flush should return nil")
	}
}

// ============================================================================
// RISK 10: Duration Edge Cases
// ============================================================================

func TestVeryLongDuration(t *testing.T) {
	acc := BarAccumulator{}

	// First trade
	startTime := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC).UnixMilli()
	acc.AddTrade(50000, 1.0, 50000, startTime, true)

	// Trade 1 year later (unrealistic but test handling)
	endTime := time.Date(2021, 1, 1, 0, 0, 0, 0, time.UTC).UnixMilli()
	acc.AddTrade(60000, 1.0, 60000, endTime, true)

	bar := acc.Finalize(110000)

	// Duration should be about 365 days in seconds
	expectedDuration := float64(endTime-startTime) / 1000.0
	if bar.Duration != expectedDuration {
		t.Errorf("Expected duration %f, got %f", expectedDuration, bar.Duration)
	}

	// Verify it's approximately 1 year (365.25 days * 24 * 60 * 60)
	oneYear := 365.25 * 24 * 60 * 60
	if math.Abs(bar.Duration-oneYear) > 86400 { // Allow 1 day tolerance
		t.Errorf("Duration should be approximately 1 year (%f), got %f", oneYear, bar.Duration)
	}
}

func TestNegativeDuration(t *testing.T) {
	acc := BarAccumulator{}

	// Trades with reversed timestamps (shouldn't happen but test handling)
	acc.AddTrade(50000, 1.0, 50000, 1600000060000, true) // Later timestamp first
	acc.AddTrade(50100, 1.0, 50100, 1600000000000, true) // Earlier timestamp

	bar := acc.Finalize(100100)

	// Duration calculation: EndTime - StartTime
	// EndTime is updated to latest processed (1600000000000)
	// StartTime is set at first trade (1600000060000)
	// This would give negative duration

	// With MinDuration clipping, should be at least MinDuration
	if bar.Duration < MinDuration {
		t.Errorf("Duration should be at least MinDuration, got %f", bar.Duration)
	}
}

// ============================================================================
// Helper Functions
// ============================================================================

func setupGeneratorSkipWarmup(threshold float64) *Generator {
	g := NewGenerator()

	// Set up to skip warmup
	avgDailyVolume := threshold * float64(DefaultTargetBarsPerDay)
	for i := 0; i < WarmupDays; i++ {
		g.dailyVolumes[i] = avgDailyVolume
	}
	g.dailyVolumeCount = WarmupDays
	g.isWarmup = false
	g.initializeEMA() // Initialize EMA with SMA of warmup data
	g.currentThreshold = g.calculateThreshold()

	// Set current day
	baseTime := time.Date(2020, 1, 15, 0, 0, 0, 0, time.UTC).UnixMilli()
	g.currentDayStart = getDayStartUTC(baseTime)

	return g
}
