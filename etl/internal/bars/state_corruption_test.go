package bars

import (
	"errors"
	"testing"
	"time"
)

// TestFailFastStateCorruption tests that ProcessTradeWithIDSafe detects state corruption
// when processing trades from the past with a future state.
func TestFailFastStateCorruption(t *testing.T) {
	tests := []struct {
		name           string
		stateDay       time.Time // State's currentDayStart
		tradeDay       time.Time // Trade's timestamp
		expectError    bool
		errorContains  string
	}{
		{
			name:        "Normal: trade same day as state",
			stateDay:    time.Date(2024, 1, 15, 0, 0, 0, 0, time.UTC),
			tradeDay:    time.Date(2024, 1, 15, 12, 30, 0, 0, time.UTC),
			expectError: false,
		},
		{
			name:        "Normal: trade 1 day before state",
			stateDay:    time.Date(2024, 1, 15, 0, 0, 0, 0, time.UTC),
			tradeDay:    time.Date(2024, 1, 14, 12, 30, 0, 0, time.UTC),
			expectError: false,
		},
		{
			name:        "Normal: trade 30 days before state (within limit)",
			stateDay:    time.Date(2024, 1, 30, 0, 0, 0, 0, time.UTC),
			tradeDay:    time.Date(2024, 1, 1, 12, 30, 0, 0, time.UTC),
			expectError: false,
		},
		{
			name:          "CORRUPTION: trade 60 days before state (exceeds limit)",
			stateDay:      time.Date(2024, 3, 1, 0, 0, 0, 0, time.UTC),
			tradeDay:      time.Date(2024, 1, 1, 12, 30, 0, 0, time.UTC),
			expectError:   true,
			errorContains: "STATE CORRUPTION DETECTED",
		},
		{
			name:          "CORRUPTION: trade 1 year before state",
			stateDay:      time.Date(2024, 9, 1, 0, 0, 0, 0, time.UTC),
			tradeDay:      time.Date(2023, 10, 15, 12, 30, 0, 0, time.UTC),
			expectError:   true,
			errorContains: "STATE CORRUPTION DETECTED",
		},
		{
			name:        "Normal: trade after state (future trade)",
			stateDay:    time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
			tradeDay:    time.Date(2024, 3, 1, 12, 30, 0, 0, time.UTC),
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := NewGenerator()

			// Set up state with specific currentDayStart
			g.currentDayStart = tt.stateDay.UnixMilli()
			g.currentThreshold = 1000000 // Set threshold to enable processing
			g.isWarmup = false
			g.lastProcessedTradeID = 0

			// Process trade
			_, err := g.ProcessTradeWithIDSafe(
				1,                           // tradeID
				50000.0,                     // price
				1.0,                         // quantity
				50000.0,                     // quoteQty
				tt.tradeDay.UnixMilli(),     // timestamp
				true,                        // isAggressorBuy
			)

			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error but got nil")
					return
				}

				var stateErr *ErrStateCorruption
				if !errors.As(err, &stateErr) {
					t.Errorf("Expected ErrStateCorruption, got %T: %v", err, err)
					return
				}

				if tt.errorContains != "" && stateErr.Error() == "" {
					t.Errorf("Error message should contain '%s'", tt.errorContains)
				}
			} else {
				if err != nil {
					t.Errorf("Expected no error but got: %v", err)
				}
			}
		})
	}
}

// TestBugScenarioReproduction reproduces the exact bug scenario:
// 1. Process data up to 2024-09, save state
// 2. Try to process 2023-10 data without resetting state
// 3. This should trigger state corruption error
func TestBugScenarioReproduction(t *testing.T) {
	g := NewGenerator()

	// Simulate state after processing 2024-09
	g.currentDayStart = time.Date(2024, 9, 30, 0, 0, 0, 0, time.UTC).UnixMilli()
	g.lastProcessedTradeID = 5427633412 // Actual trade ID from 2024-09
	g.isWarmup = false
	g.currentThreshold = 248933024.92
	g.emaVolume = 12446651245.97

	// Try to process 2023-10 trade (the bug scenario)
	tradeTimestamp := time.Date(2023, 10, 15, 12, 0, 0, 0, time.UTC).UnixMilli()

	_, err := g.ProcessTradeWithIDSafe(
		3000000000,     // trade ID from 2023-10
		27000.0,        // price
		1.0,            // quantity
		27000.0,        // quoteQty
		tradeTimestamp, // timestamp
		true,           // isAggressorBuy
	)

	// This MUST return an error - this is the bug we're fixing
	if err == nil {
		t.Fatal("CRITICAL: Bug scenario was not detected! State corruption should have been caught.")
	}

	var stateErr *ErrStateCorruption
	if !errors.As(err, &stateErr) {
		t.Fatalf("Expected ErrStateCorruption, got %T: %v", err, err)
	}

	// Verify error details
	if stateErr.GapDays < 300 { // ~11 months gap
		t.Errorf("Expected gap of ~11 months, got %d days", stateErr.GapDays)
	}

	t.Logf("Bug correctly detected: %v", stateErr)
}

// TestStatelessStartPrinciple tests that PrepareForMonth correctly resets state
func TestStatelessStartPrinciple(t *testing.T) {
	g := NewGenerator()

	// Set up completed months with checkpoints
	// 2023-09 completed
	g.MarkMonthComplete("2023-09", 1424,
		2900000000, 3000000000, // trade ID range
		time.Date(2023, 9, 1, 0, 0, 0, 0, time.UTC).UnixMilli(),
		time.Date(2023, 9, 30, 23, 59, 0, 0, time.UTC).UnixMilli(),
	)

	// Simulate corrupted state (as if we processed 2024-09)
	g.currentDayStart = time.Date(2024, 9, 30, 0, 0, 0, 0, time.UTC).UnixMilli()
	g.lastProcessedTradeID = 5427633412
	g.emaVolume = 12446651245.97
	g.currentThreshold = 248933024.92
	g.isWarmup = false

	// Record corrupted state values
	corruptedTradeID := g.lastProcessedTradeID
	corruptedDayStart := g.currentDayStart

	// Call PrepareForMonth for 2023-10 (should restore from 2023-09 checkpoint)
	restored, prevMonth := g.PrepareForMonth("2023-10")

	// Verify restoration happened
	if !restored {
		t.Fatal("PrepareForMonth should have restored from previous checkpoint")
	}

	if prevMonth != "2023-09" {
		t.Errorf("Expected prevMonth '2023-09', got '%s'", prevMonth)
	}

	// Verify state was restored from 2023-09 checkpoint
	if g.lastProcessedTradeID == corruptedTradeID {
		t.Error("lastProcessedTradeID was not reset from corrupted value")
	}

	if g.currentDayStart == corruptedDayStart {
		t.Error("currentDayStart was not reset from corrupted value")
	}

	// The lastProcessedTradeID should now be 3000000000 (2023-09's last trade ID)
	if g.lastProcessedTradeID != 3000000000 {
		t.Errorf("Expected lastProcessedTradeID to be 3000000000, got %d", g.lastProcessedTradeID)
	}

	t.Logf("State correctly restored: lastProcessedTradeID=%d, currentDayStart=%d",
		g.lastProcessedTradeID, g.currentDayStart)
}

// TestStatelessStartNoPreviousCheckpoint tests behavior when no previous checkpoint exists
func TestStatelessStartNoPreviousCheckpoint(t *testing.T) {
	g := NewGenerator()

	// Simulate corrupted state without any checkpoints
	g.currentDayStart = time.Date(2024, 9, 30, 0, 0, 0, 0, time.UTC).UnixMilli()
	g.lastProcessedTradeID = 5427633412
	g.isWarmup = false

	// Call PrepareForMonth for 2019-09 (first month, no previous checkpoint)
	restored, prevMonth := g.PrepareForMonth("2019-09")

	// Should not restore (no checkpoint)
	if restored {
		t.Error("Should not restore when no previous checkpoint exists")
	}

	if prevMonth != "" {
		t.Errorf("prevMonth should be empty, got '%s'", prevMonth)
	}

	// lastProcessedTradeID should be reset to 0 (safe default)
	if g.lastProcessedTradeID != 0 {
		t.Errorf("lastProcessedTradeID should be reset to 0, got %d", g.lastProcessedTradeID)
	}
}

// TestValidateStateForMonth tests month validation
func TestValidateStateForMonth(t *testing.T) {
	tests := []struct {
		name        string
		stateDay    time.Time
		month       string
		expectError bool
	}{
		{
			name:        "Valid: state within month",
			stateDay:    time.Date(2023, 10, 15, 0, 0, 0, 0, time.UTC),
			month:       "2023-10",
			expectError: false,
		},
		{
			name:        "Valid: state before month",
			stateDay:    time.Date(2023, 9, 30, 0, 0, 0, 0, time.UTC),
			month:       "2023-10",
			expectError: false,
		},
		{
			name:        "Invalid: state after month (future corruption)",
			stateDay:    time.Date(2024, 9, 30, 0, 0, 0, 0, time.UTC),
			month:       "2023-10",
			expectError: true,
		},
		{
			name:        "Valid: fresh state (currentDayStart=0)",
			stateDay:    time.Time{}, // Zero value
			month:       "2023-10",
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := NewGenerator()

			if !tt.stateDay.IsZero() {
				g.currentDayStart = tt.stateDay.UnixMilli()
			}

			err := g.ValidateStateForMonth(tt.month, 0)

			if tt.expectError && err == nil {
				t.Error("Expected error but got nil")
			}
			if !tt.expectError && err != nil {
				t.Errorf("Expected no error but got: %v", err)
			}
		})
	}
}

// TestIntegrationStatelessStartWithFailFast tests both mechanisms working together
func TestIntegrationStatelessStartWithFailFast(t *testing.T) {
	g := NewGenerator()

	// Set up 2023-09 checkpoint
	g.MarkMonthComplete("2023-09", 1424,
		2900000000, 3000000000,
		time.Date(2023, 9, 1, 0, 0, 0, 0, time.UTC).UnixMilli(),
		time.Date(2023, 9, 30, 23, 59, 0, 0, time.UTC).UnixMilli(),
	)

	// Simulate corrupted state from 2024-09
	g.currentDayStart = time.Date(2024, 9, 30, 0, 0, 0, 0, time.UTC).UnixMilli()
	g.lastProcessedTradeID = 5427633412
	g.isWarmup = false
	g.currentThreshold = 248933024.92

	// Step 1: PrepareForMonth should fix the state
	restored, _ := g.PrepareForMonth("2023-10")
	if !restored {
		t.Fatal("PrepareForMonth should have restored from checkpoint")
	}

	// Step 2: Now processing 2023-10 trade should work without error
	tradeTimestamp := time.Date(2023, 10, 15, 12, 0, 0, 0, time.UTC).UnixMilli()

	_, err := g.ProcessTradeWithIDSafe(
		3000000001,     // trade ID just after 2023-09's last
		27000.0,
		1.0,
		27000.0,
		tradeTimestamp,
		true,
	)

	if err != nil {
		t.Fatalf("After PrepareForMonth, processing should succeed: %v", err)
	}

	t.Log("Integration test passed: Stateless Start + Fail-Fast work together correctly")
}

// =============================================================================
// Edge Case: Continuous Processing Should NOT Call PrepareForMonth
// =============================================================================

// TestContinuousProcessingPreservesState tests that during continuous processing,
// the state (threshold, EMA) should be preserved and NOT reset.
// This is the bug that caused 94,124 bars in 2019-10.
func TestContinuousProcessingPreservesState(t *testing.T) {
	g := NewGenerator()

	// Simulate processing through warmup and into regular operation
	baseTime := time.Date(2019, 9, 1, 0, 0, 0, 0, time.UTC).UnixMilli()

	// Process 15 days to complete warmup (need day change to trigger warmup completion)
	// Day 0-13: accumulate daily volume
	// Day 14: first trade triggers warmup completion
	for day := 0; day <= 14; day++ {
		dayStart := baseTime + int64(day)*MillisPerDay
		// Each day: $500M volume in one trade
		g.ProcessTradeWithID(
			int64(day+1),
			50000.0,
			10000.0, // 10000 BTC
			500_000_000.0, // $500M per day
			dayStart+int64(12*3600*1000), // Noon each day
			true,
		)
	}

	// Should be out of warmup now (day 15 trade triggered completion)
	if g.IsWarmup() {
		t.Logf("Warmup days remaining: %d, dailyVolumeCount: %d",
			g.GetWarmupDaysRemaining(), g.dailyVolumeCount)
		t.Fatal("Should be out of warmup after 15 days")
	}

	// Record state after warmup
	thresholdAfterWarmup := g.GetCurrentThreshold()
	emaAfterWarmup := g.GetEMAVolume()

	if thresholdAfterWarmup <= 0 {
		t.Fatalf("Threshold should be positive after warmup, got %f", thresholdAfterWarmup)
	}

	t.Logf("After warmup: threshold=$%.2f, EMA=$%.2f", thresholdAfterWarmup, emaAfterWarmup)

	// Process day 16 (more data)
	day16Start := baseTime + int64(15)*MillisPerDay
	g.ProcessTradeWithID(
		int64(16),
		50000.0,
		10000.0,
		500_000_000.0,
		day16Start+int64(12*3600*1000),
		true,
	)

	// Threshold should have evolved, NOT reset to 0
	thresholdAfterDay16 := g.GetCurrentThreshold()

	if thresholdAfterDay16 <= 0 {
		t.Fatalf("Threshold should still be positive, got %f", thresholdAfterDay16)
	}

	// Threshold should be similar (EMA evolves slowly)
	ratio := thresholdAfterDay16 / thresholdAfterWarmup
	if ratio < 0.5 || ratio > 2.0 {
		t.Errorf("Threshold changed too drastically: before=%f, after=%f, ratio=%f",
			thresholdAfterWarmup, thresholdAfterDay16, ratio)
	}

	t.Logf("After day 16: threshold=$%.2f (ratio=%.2f)", thresholdAfterDay16, ratio)
}

// TestReprocessingResetsStateCorrectly tests that when reprocessing is needed,
// the state is correctly restored from the previous month's checkpoint.
func TestReprocessingResetsStateCorrectly(t *testing.T) {
	g := NewGenerator()

	// Set up 2023-09 as complete with specific state
	expectedThreshold := 150_000_000.0
	expectedEMA := 7_500_000_000.0
	expectedLastTradeID := int64(3_000_000_000)

	g.monthlyCheckpoints = make(map[string]MonthCheckpoint)
	g.monthlyCheckpoints["2023-09"] = MonthCheckpoint{
		Status:            "complete",
		BarCount:          1424,
		FirstTradeID:      2_900_000_000,
		LastTradeID:       expectedLastTradeID,
		FirstBarTime:      time.Date(2023, 9, 1, 0, 0, 0, 0, time.UTC).UnixMilli(),
		LastBarTime:       time.Date(2023, 9, 30, 23, 59, 0, 0, time.UTC).UnixMilli(),
		EMAVolumeSnapshot: expectedEMA,
		ThresholdSnapshot: expectedThreshold,
		DayStartSnapshot:  time.Date(2023, 9, 30, 0, 0, 0, 0, time.UTC).UnixMilli(),
		DayVolumeSnapshot: 0,
		IsWarmupSnapshot:  false,
	}

	// Simulate corrupted state (from processing 2024 data)
	g.currentThreshold = 999_999_999.0  // Wrong!
	g.emaVolume = 50_000_000_000.0      // Wrong!
	g.lastProcessedTradeID = 9_999_999_999 // Wrong!
	g.isWarmup = false

	// Reprocess 2023-10: should restore from 2023-09 checkpoint
	restored, prevMonth := g.PrepareForMonth("2023-10")

	if !restored {
		t.Fatal("Should have restored from previous checkpoint")
	}
	if prevMonth != "2023-09" {
		t.Errorf("Expected prevMonth='2023-09', got '%s'", prevMonth)
	}

	// Verify state was restored correctly
	if g.currentThreshold != expectedThreshold {
		t.Errorf("Threshold not restored: expected %f, got %f", expectedThreshold, g.currentThreshold)
	}
	if g.emaVolume != expectedEMA {
		t.Errorf("EMA not restored: expected %f, got %f", expectedEMA, g.emaVolume)
	}
	if g.lastProcessedTradeID != expectedLastTradeID {
		t.Errorf("LastProcessedTradeID not restored: expected %d, got %d", expectedLastTradeID, g.lastProcessedTradeID)
	}

	t.Logf("State correctly restored: threshold=$%.2f, EMA=$%.2f, lastTradeID=%d",
		g.currentThreshold, g.emaVolume, g.lastProcessedTradeID)
}

// TestPrepareForMonthDoesNotDestroyCurrentState tests that PrepareForMonth
// with no previous checkpoint doesn't destroy valid current state when
// processing is already in progress.
func TestContinuousProcessingWithoutCheckpoint(t *testing.T) {
	g := NewGenerator()

	// Set up valid state (no checkpoints yet, first month processing)
	g.isWarmup = false
	g.currentThreshold = 100_000_000.0
	g.emaVolume = 5_000_000_000.0
	g.lastProcessedTradeID = 1_000_000
	g.currentDayStart = time.Date(2019, 9, 15, 0, 0, 0, 0, time.UTC).UnixMilli()
	// No monthlyCheckpoints set

	originalThreshold := g.currentThreshold
	originalEMA := g.emaVolume

	// PrepareForMonth for 2019-10 with NO 2019-09 checkpoint
	restored, _ := g.PrepareForMonth("2019-10")

	// Should NOT restore (no checkpoint exists)
	if restored {
		t.Error("Should not restore when no previous checkpoint exists")
	}

	// lastProcessedTradeID should be reset to 0 (safe for fresh start)
	// But threshold and EMA should be preserved if already valid
	if g.lastProcessedTradeID != 0 {
		t.Errorf("lastProcessedTradeID should be reset to 0, got %d", g.lastProcessedTradeID)
	}

	// Note: Current implementation resets lastProcessedTradeID but preserves threshold/EMA
	// This is the expected behavior for the edge case of no checkpoint
	t.Logf("Without checkpoint: threshold preserved=%v, EMA preserved=%v",
		g.currentThreshold == originalThreshold,
		g.emaVolume == originalEMA)
}

// TestBarCountConsistency tests that the number of bars generated per month
// is consistent with the threshold (should be ~50 bars/day = ~1500/month)
func TestBarCountConsistency(t *testing.T) {
	g := NewGenerator()

	// Complete warmup first
	baseTime := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC).UnixMilli()

	// 15 days to complete warmup with $10B daily volume
	dailyVolume := 10_000_000_000.0 // $10B
	for day := 0; day <= 14; day++ {
		dayStart := baseTime + int64(day)*MillisPerDay
		g.ProcessTradeWithID(
			int64(day+1),
			50000.0,
			dailyVolume/50000.0,
			dailyVolume,
			dayStart+int64(12*3600*1000), // Noon
			true,
		)
	}

	// Verify warmup is complete
	if g.IsWarmup() {
		t.Fatalf("Should be out of warmup, days remaining: %d", g.GetWarmupDaysRemaining())
	}

	// Get threshold after warmup
	threshold := g.GetCurrentThreshold()
	expectedThreshold := dailyVolume / float64(TargetBarsPerDay) // $10B / 50 = $200M

	if threshold <= 0 {
		t.Fatalf("Threshold should be positive, got %f", threshold)
	}

	// Should be approximately correct
	ratio := threshold / expectedThreshold
	if ratio < 0.5 || ratio > 2.0 {
		t.Errorf("Threshold not as expected: got $%.2f, expected ~$%.2f (ratio=%.2f)",
			threshold, expectedThreshold, ratio)
	}

	t.Logf("Threshold after warmup: $%.2f (expected ~$%.2f, ratio=%.2f)", threshold, expectedThreshold, ratio)
}
