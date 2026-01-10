package bars

import (
	"testing"
	"time"
)

func TestNewGenerator(t *testing.T) {
	g := NewGenerator()

	if !g.IsWarmup() {
		t.Error("New generator should be in warmup mode")
	}

	if g.GetWarmupDaysRemaining() != WarmupDays {
		t.Errorf("Expected %d warmup days remaining, got %d", WarmupDays, g.GetWarmupDaysRemaining())
	}

	if g.GetCurrentThreshold() != 0 {
		t.Errorf("Initial threshold should be 0, got %f", g.GetCurrentThreshold())
	}
}

func TestWarmupPeriod(t *testing.T) {
	g := NewGenerator()

	// Process trades for 14 days
	baseTime := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC).UnixMilli()
	dailyVolume := 1_000_000_000.0 // $1B per day

	for day := 0; day < WarmupDays; day++ {
		dayStart := baseTime + int64(day)*MillisPerDay

		// Simulate trades throughout the day
		for i := 0; i < 100; i++ {
			ts := dayStart + int64(i*60000) // Every minute
			bar := g.ProcessTrade(50000, 0.2, dailyVolume/100, ts, true)

			// During warmup, no bars should be generated
			if bar != nil {
				t.Errorf("Day %d: bar generated during warmup", day)
			}
		}

		if day < WarmupDays-1 && !g.IsWarmup() {
			t.Errorf("Day %d: warmup should not be complete yet", day)
		}
	}

	// Process first trade of day 15 (should trigger warmup completion)
	day15Start := baseTime + int64(WarmupDays)*MillisPerDay
	g.ProcessTrade(50000, 0.2, 10000, day15Start, true)

	if g.IsWarmup() {
		t.Error("After 14 days, warmup should be complete")
	}

	if g.GetWarmupDaysRemaining() != 0 {
		t.Errorf("Expected 0 warmup days remaining, got %d", g.GetWarmupDaysRemaining())
	}

	// Threshold should now be calculated
	expectedThreshold := dailyVolume / float64(DefaultTargetBarsPerDay)
	threshold := g.GetCurrentThreshold()

	// Allow some tolerance for accumulated volume variations
	if threshold < expectedThreshold*0.5 || threshold > expectedThreshold*1.5 {
		t.Errorf("Threshold %f is not in expected range around %f", threshold, expectedThreshold)
	}
}

func TestBarGeneration(t *testing.T) {
	g := NewGenerator()

	// Skip warmup by setting up state directly
	for i := 0; i < WarmupDays; i++ {
		g.dailyVolumes[i] = 1_000_000_000 // $1B per day
	}
	g.dailyVolumeCount = WarmupDays
	g.isWarmup = false
	g.initializeEMA() // Initialize EMA with SMA of warmup data
	g.currentThreshold = g.calculateThreshold()

	baseTime := time.Date(2020, 1, 15, 0, 0, 0, 0, time.UTC).UnixMilli()
	g.currentDayStart = getDayStartUTC(baseTime)

	// Expected threshold: $1B / 50 = $20M
	threshold := g.GetCurrentThreshold()
	t.Logf("Threshold: $%.2f", threshold)

	// Process trades until bar is generated
	var generatedBar *DynamicDollarBar
	tradeCount := 0

	for i := 0; i < 10000; i++ {
		ts := baseTime + int64(i*1000)
		quoteQty := 100000.0 // $100k per trade

		bar := g.ProcessTrade(50000, 2.0, quoteQty, ts, i%2 == 0)
		tradeCount++

		if bar != nil {
			generatedBar = bar
			break
		}
	}

	if generatedBar == nil {
		t.Fatal("No bar was generated")
	}

	t.Logf("Bar generated after %d trades", tradeCount)
	t.Logf("Bar: DollarValue=%.2f, Duration=%.2fs, TickCount=%d",
		generatedBar.DollarValue, generatedBar.Duration, generatedBar.TickCount)

	// Verify bar fields
	if generatedBar.DollarValue < threshold {
		t.Errorf("Bar dollar value %f is less than threshold %f", generatedBar.DollarValue, threshold)
	}

	if generatedBar.Duration <= 0 {
		t.Error("Bar duration should be positive")
	}

	if generatedBar.TickCount != int64(tradeCount) {
		t.Errorf("Expected %d ticks, got %d", tradeCount, generatedBar.TickCount)
	}

	if generatedBar.EndTime <= generatedBar.StartTime {
		t.Error("EndTime should be greater than StartTime")
	}
}

func TestImbalanceTracking(t *testing.T) {
	g := NewGenerator()

	// Setup to skip warmup
	for i := 0; i < WarmupDays; i++ {
		g.dailyVolumes[i] = 100_000_000
	}
	g.dailyVolumeCount = WarmupDays
	g.isWarmup = false
	g.initializeEMA()
	g.currentThreshold = g.calculateThreshold() // ~$2M threshold (100M / 50)

	baseTime := time.Date(2020, 1, 15, 0, 0, 0, 0, time.UTC).UnixMilli()
	g.currentDayStart = getDayStartUTC(baseTime)

	// Process buy-heavy trades
	var bar *DynamicDollarBar
	for i := 0; i < 1000; i++ {
		ts := baseTime + int64(i*1000)
		isBuy := i < 700 // 70% buy

		bar = g.ProcessTrade(50000, 0.1, 5000, ts, isBuy)
		if bar != nil {
			break
		}
	}

	if bar == nil {
		t.Fatal("No bar generated")
	}

	if bar.BuyDollarVol <= bar.SellDollarVol {
		t.Error("Buy volume should be greater than sell volume")
	}

	if bar.NetImbalance <= 0 {
		t.Error("Net imbalance should be positive (buy-heavy)")
	}

	expectedImbalance := bar.BuyDollarVol - bar.SellDollarVol
	if bar.NetImbalance != expectedImbalance {
		t.Errorf("NetImbalance mismatch: expected %f, got %f", expectedImbalance, bar.NetImbalance)
	}
}

func TestStateExportImport(t *testing.T) {
	g1 := NewGenerator()

	// Setup state
	for i := 0; i < 10; i++ {
		g1.dailyVolumes[i] = float64((i + 1) * 1_000_000_000)
	}
	g1.dailyVolumeHead = 10
	g1.dailyVolumeCount = 10
	g1.emaVolume = 5_500_000_000 // EMA value
	g1.currentDayStart = 1600000000000
	g1.currentDayVolume = 500_000_000
	g1.currentThreshold = 10_000_000
	g1.isWarmup = false

	// Add some accumulator state
	g1.acc.AddTrade(50000, 1.0, 50000, 1600000001000, true)

	// Export state
	state := g1.ExportState()

	// Import to new generator
	g2 := NewGenerator()
	g2.ImportState(state)

	// Verify state matches
	if g2.dailyVolumeCount != g1.dailyVolumeCount {
		t.Errorf("DailyVolumeCount mismatch: %d vs %d", g2.dailyVolumeCount, g1.dailyVolumeCount)
	}

	if g2.emaVolume != g1.emaVolume {
		t.Errorf("EMAVolume mismatch: %f vs %f", g2.emaVolume, g1.emaVolume)
	}

	if g2.currentThreshold != g1.currentThreshold {
		t.Errorf("Threshold mismatch: %f vs %f", g2.currentThreshold, g1.currentThreshold)
	}

	if g2.isWarmup != g1.isWarmup {
		t.Errorf("Warmup mismatch: %v vs %v", g2.isWarmup, g1.isWarmup)
	}

	if g2.acc.TickCount != g1.acc.TickCount {
		t.Errorf("Accumulator tick count mismatch: %d vs %d", g2.acc.TickCount, g1.acc.TickCount)
	}

	for i := 0; i < WarmupDays; i++ {
		if g2.dailyVolumes[i] != g1.dailyVolumes[i] {
			t.Errorf("DailyVolume[%d] mismatch: %f vs %f", i, g2.dailyVolumes[i], g1.dailyVolumes[i])
		}
	}
}

func TestFlush(t *testing.T) {
	g := NewGenerator()

	// During warmup, flush should return nil
	g.acc.AddTrade(50000, 1.0, 50000, 1600000000000, true)
	bar := g.Flush()
	if bar != nil {
		t.Error("Flush during warmup should return nil")
	}

	// After warmup, flush should return the pending bar
	g.isWarmup = false
	g.currentThreshold = 1_000_000

	g.acc.Reset()
	g.acc.AddTrade(50000, 1.0, 50000, 1600000000000, true)
	g.acc.AddTrade(50100, 1.0, 50100, 1600000001000, false)

	bar = g.Flush()
	if bar == nil {
		t.Error("Flush after warmup should return pending bar")
	}

	if bar.TickCount != 2 {
		t.Errorf("Expected 2 ticks, got %d", bar.TickCount)
	}
}

func TestDayChangeThresholdUpdate(t *testing.T) {
	g := NewGenerator()

	// Setup with 14 days of varying volume
	volumes := []float64{
		1_000_000_000, 1_100_000_000, 900_000_000, 1_200_000_000,
		800_000_000, 1_500_000_000, 1_300_000_000, 1_400_000_000,
		1_000_000_000, 1_100_000_000, 1_200_000_000, 1_300_000_000,
		1_400_000_000, 1_500_000_000,
	}

	for i := 0; i < WarmupDays; i++ {
		g.dailyVolumes[i] = volumes[i]
	}
	g.dailyVolumeHead = 0
	g.dailyVolumeCount = WarmupDays
	g.isWarmup = false
	g.initializeEMA() // Initialize EMA with SMA

	// Set initial day
	day1 := time.Date(2020, 1, 15, 0, 0, 0, 0, time.UTC).UnixMilli()
	g.currentDayStart = getDayStartUTC(day1)
	g.currentThreshold = g.calculateThreshold()

	threshold1 := g.GetCurrentThreshold()
	ema1 := g.GetEMAVolume()
	t.Logf("Day 1 threshold: $%.2f, EMA: $%.2f", threshold1, ema1)

	// Simulate trading and day change
	g.currentDayVolume = 2_000_000_000 // $2B volume today (higher than average)

	// Process trade on next day (triggers day change and EMA update)
	day2 := day1 + MillisPerDay
	g.ProcessTrade(50000, 0.1, 1000, day2, true)

	threshold2 := g.GetCurrentThreshold()
	ema2 := g.GetEMAVolume()
	t.Logf("Day 2 threshold: $%.2f, EMA: $%.2f", threshold2, ema2)

	// EMA should have increased (new volume $2B > previous EMA ~$1.2B)
	if ema2 <= ema1 {
		t.Errorf("EMA should have increased: before=%.2f, after=%.2f", ema1, ema2)
	}

	// Threshold should have increased accordingly
	if threshold2 <= threshold1 {
		t.Errorf("Threshold should have increased: before=%.2f, after=%.2f", threshold1, threshold2)
	}
}

func TestDurationCalculation(t *testing.T) {
	acc := BarAccumulator{}

	// First trade
	acc.AddTrade(50000, 1.0, 50000, 1600000000000, true)

	// More trades over 60 seconds
	acc.AddTrade(50100, 1.0, 50100, 1600000030000, true) // +30s
	acc.AddTrade(50200, 1.0, 50200, 1600000060000, true) // +60s

	bar := acc.Finalize(150000)

	expectedDuration := 60.0 // 60 seconds
	if bar.Duration != expectedDuration {
		t.Errorf("Expected duration %.2f, got %.2f", expectedDuration, bar.Duration)
	}

	if bar.EndTime != 1600000060000 {
		t.Errorf("Expected EndTime 1600000060000, got %d", bar.EndTime)
	}
}

func TestThresholdCalculation(t *testing.T) {
	g := NewGenerator()

	// Add 14 days of $1B volume each
	for i := 0; i < WarmupDays; i++ {
		g.dailyVolumes[i] = 1_000_000_000
	}
	g.dailyVolumeCount = WarmupDays

	// Initialize EMA with SMA (all same values, so EMA = SMA = $1B)
	g.initializeEMA()

	threshold := g.calculateThreshold()
	expectedThreshold := 1_000_000_000.0 / float64(DefaultTargetBarsPerDay) // $1B / 50 = $20M

	if threshold != expectedThreshold {
		t.Errorf("Expected threshold %.2f, got %.2f", expectedThreshold, threshold)
	}

	// Verify EMA is correctly initialized
	if g.GetEMAVolume() != 1_000_000_000 {
		t.Errorf("Expected EMA $1B, got %.2f", g.GetEMAVolume())
	}
}

func TestEMAUpdate(t *testing.T) {
	g := NewGenerator()

	// Initialize with $1B EMA
	g.emaVolume = 1_000_000_000

	// Update with $2B daily volume
	g.updateEMA(2_000_000_000)

	// Expected: EMA_new = 0.133 * 2B + 0.867 * 1B = 0.266B + 0.867B = 1.133B
	expectedEMA := EMAAlpha*2_000_000_000 + (1-EMAAlpha)*1_000_000_000

	// Use tolerance for floating point comparison
	tolerance := 0.01
	diff := g.emaVolume - expectedEMA
	if diff < 0 {
		diff = -diff
	}
	if diff > tolerance {
		t.Errorf("Expected EMA %.2f, got %.2f (diff: %f)", expectedEMA, g.emaVolume, diff)
	}

	prevEMA := g.emaVolume

	// Update with $500M (lower volume)
	g.updateEMA(500_000_000)

	// EMA should decrease
	if g.emaVolume >= prevEMA {
		t.Errorf("EMA should have decreased after low volume day: before=%.2f, after=%.2f", prevEMA, g.emaVolume)
	}
}

func TestFlushDay(t *testing.T) {
	g := NewGenerator()

	// Simulate accumulated daily volume
	g.currentDayVolume = 500_000_000 // $500M accumulated

	// Flush day
	g.FlushDay()

	if g.currentDayVolume != 0 {
		t.Error("Current day volume should be reset after FlushDay")
	}

	if g.dailyVolumeCount != 1 {
		t.Errorf("Expected 1 day in buffer, got %d", g.dailyVolumeCount)
	}

	if g.dailyVolumes[0] != 500_000_000 {
		t.Errorf("Expected daily volume 500000000, got %f", g.dailyVolumes[0])
	}
}

func TestCrashRecoveryWithTradeID(t *testing.T) {
	g := NewGenerator()

	// Setup to skip warmup
	for i := 0; i < WarmupDays; i++ {
		g.dailyVolumes[i] = 100_000_000
	}
	g.dailyVolumeCount = WarmupDays
	g.isWarmup = false
	g.initializeEMA()
	g.currentThreshold = g.calculateThreshold()

	baseTime := time.Date(2020, 1, 15, 0, 0, 0, 0, time.UTC).UnixMilli()
	g.currentDayStart = getDayStartUTC(baseTime)

	// Process trades with IDs
	tradeID := int64(1000)
	for i := 0; i < 100; i++ {
		g.ProcessTradeWithID(tradeID+int64(i), 50000, 0.1, 5000, baseTime+int64(i*1000), true)
	}

	// Verify last processed trade ID is updated
	if g.GetLastProcessedTradeID() != tradeID+99 {
		t.Errorf("Expected last trade ID %d, got %d", tradeID+99, g.GetLastProcessedTradeID())
	}

	// Export state (simulating crash save)
	state := g.ExportState()

	// Create new generator and restore state (simulating restart)
	g2 := NewGenerator()
	g2.ImportState(state)

	// Verify that duplicate trades are skipped
	for i := 0; i < 100; i++ {
		if !g2.ShouldSkipTrade(tradeID + int64(i)) {
			t.Errorf("Trade ID %d should be skipped (already processed)", tradeID+int64(i))
		}
	}

	// Verify that new trades are NOT skipped
	newTradeID := tradeID + 100
	if g2.ShouldSkipTrade(newTradeID) {
		t.Errorf("Trade ID %d should NOT be skipped (new trade)", newTradeID)
	}

	// Process new trades and verify they're counted
	accBefore := g2.acc.TickCount
	g2.ProcessTradeWithID(newTradeID, 50000, 0.1, 5000, baseTime+100000, true)
	accAfter := g2.acc.TickCount

	if accAfter != accBefore+1 {
		t.Errorf("New trade should be counted: before=%d, after=%d", accBefore, accAfter)
	}
}

func TestShouldSkipTrade(t *testing.T) {
	g := NewGenerator()

	// Trade ID 0 should never be skipped (special case for legacy data)
	if g.ShouldSkipTrade(0) {
		t.Error("Trade ID 0 should not be skipped")
	}

	// Set last processed trade ID
	g.lastProcessedTradeID = 1000

	// Trades <= 1000 should be skipped
	if !g.ShouldSkipTrade(1) {
		t.Error("Trade ID 1 should be skipped")
	}
	if !g.ShouldSkipTrade(1000) {
		t.Error("Trade ID 1000 should be skipped")
	}

	// Trades > 1000 should NOT be skipped
	if g.ShouldSkipTrade(1001) {
		t.Error("Trade ID 1001 should NOT be skipped")
	}
}

func TestMinDurationClipping(t *testing.T) {
	acc := BarAccumulator{}

	// Single trade (same timestamp for start and end)
	acc.AddTrade(50000, 1.0, 50000, 1600000000000, true)

	bar := acc.Finalize(50000)

	// Duration should be clipped to MinDuration, not 0
	if bar.Duration != MinDuration {
		t.Errorf("Expected minimum duration %.3f, got %.3f", MinDuration, bar.Duration)
	}

	// Reset and test with very short duration
	acc.Reset()
	acc.AddTrade(50000, 1.0, 50000, 1600000000000, true)
	acc.AddTrade(50100, 1.0, 50100, 1600000000000, true) // Same millisecond

	bar = acc.Finalize(100100)

	if bar.Duration != MinDuration {
		t.Errorf("Expected minimum duration %.3f for same-millisecond trades, got %.3f", MinDuration, bar.Duration)
	}
}
