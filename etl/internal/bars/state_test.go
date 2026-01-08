package bars

import (
	"encoding/json"
	"os"
	"testing"
)

func TestSaveAndLoadState(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "bars_state_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create state with all fields
	original := GeneratorState{
		Accumulator: BarAccumulator{
			StartTimestamp: 1234567890,
			EndTimestamp:   1234567900,
			Open:           50000,
			High:           51000,
			Low:            49000,
			Close:          50500,
			Volume:         10.5,
			DollarValue:    525000,
			TickCount:      150,
			BuyDollarVol:   300000,
			SellDollarVol:  225000,
		},
		DailyVolumes:         [WarmupDays]float64{1e9, 2e9, 1.5e9, 1.8e9, 1.2e9, 1.1e9, 1.3e9, 1.4e9, 1.6e9, 1.7e9, 1.9e9, 2.1e9, 2.0e9, 1.8e9},
		DailyVolumeHead:      5,
		DailyVolumeCount:     14,
		EMAVolume:            1_650_000_000, // EMA of daily volumes
		CurrentDayStart:      1600000000000,
		CurrentDayVolume:     500_000_000,
		CurrentThreshold:     10_000_000,
		IsWarmup:             false,
		LastProcessedTradeID: 123456789, // Critical for crash recovery
	}

	// Save
	if err := SaveState(tmpDir, original); err != nil {
		t.Fatalf("SaveState failed: %v", err)
	}

	// Load
	loaded, err := LoadState(tmpDir)
	if err != nil {
		t.Fatalf("LoadState failed: %v", err)
	}
	if loaded == nil {
		t.Fatal("LoadState returned nil")
	}

	// Verify all fields
	if loaded.Accumulator.Open != original.Accumulator.Open {
		t.Errorf("Accumulator.Open mismatch")
	}
	if loaded.Accumulator.DollarValue != original.Accumulator.DollarValue {
		t.Errorf("Accumulator.DollarValue mismatch")
	}
	if loaded.Accumulator.EndTimestamp != original.Accumulator.EndTimestamp {
		t.Errorf("Accumulator.EndTimestamp mismatch")
	}
	if loaded.CurrentThreshold != original.CurrentThreshold {
		t.Errorf("CurrentThreshold mismatch")
	}
	if loaded.DailyVolumeCount != original.DailyVolumeCount {
		t.Errorf("DailyVolumeCount mismatch")
	}
	if loaded.EMAVolume != original.EMAVolume {
		t.Errorf("EMAVolume mismatch: %f vs %f", loaded.EMAVolume, original.EMAVolume)
	}
	if loaded.IsWarmup != original.IsWarmup {
		t.Errorf("IsWarmup mismatch")
	}
	if loaded.LastProcessedTradeID != original.LastProcessedTradeID {
		t.Errorf("LastProcessedTradeID mismatch: %d vs %d", loaded.LastProcessedTradeID, original.LastProcessedTradeID)
	}
	for i := 0; i < WarmupDays; i++ {
		if loaded.DailyVolumes[i] != original.DailyVolumes[i] {
			t.Errorf("DailyVolumes[%d] mismatch: %f vs %f", i, loaded.DailyVolumes[i], original.DailyVolumes[i])
		}
	}
}

func TestLoadStateNonExistent(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "bars_state_test_empty")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Load from empty directory
	state, err := LoadState(tmpDir)
	if err != nil {
		t.Fatalf("LoadState should not error for non-existent file: %v", err)
	}
	if state != nil {
		t.Error("LoadState should return nil for non-existent file")
	}
}

func TestDeleteState(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "bars_state_test_delete")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Save state
	state := GeneratorState{
		CurrentThreshold: 100,
		IsWarmup:         true,
	}
	if err := SaveState(tmpDir, state); err != nil {
		t.Fatalf("SaveState failed: %v", err)
	}

	// Verify it exists
	loaded, _ := LoadState(tmpDir)
	if loaded == nil {
		t.Fatal("State should exist after save")
	}

	// Delete
	if err := DeleteState(tmpDir); err != nil {
		t.Fatalf("DeleteState failed: %v", err)
	}

	// Verify it's gone
	loaded, _ = LoadState(tmpDir)
	if loaded != nil {
		t.Error("State should be nil after delete")
	}
}

func TestStateValidation(t *testing.T) {
	// Test valid state passes validation
	validState := GeneratorState{
		DailyVolumeCount:     WarmupDays,
		DailyVolumeHead:      0,
		EMAVolume:            1_000_000_000,
		CurrentThreshold:     1_000_000,
		CurrentDayVolume:     500_000,
		LastProcessedTradeID: 12345,
		IsWarmup:             false,
	}
	for i := 0; i < WarmupDays; i++ {
		validState.DailyVolumes[i] = float64(i+1) * 1e9
	}

	if err := validState.Validate(); err != nil {
		t.Errorf("Valid state should pass validation: %v", err)
	}

	// Test invalid states
	testCases := []struct {
		name    string
		mutate  func(*GeneratorState)
		wantErr string
	}{
		{
			name:    "negative DailyVolumeCount",
			mutate:  func(s *GeneratorState) { s.DailyVolumeCount = -1 },
			wantErr: "DailyVolumeCount out of range",
		},
		{
			name:    "DailyVolumeCount too high",
			mutate:  func(s *GeneratorState) { s.DailyVolumeCount = WarmupDays + 1 },
			wantErr: "DailyVolumeCount out of range",
		},
		{
			name:    "negative DailyVolumeHead",
			mutate:  func(s *GeneratorState) { s.DailyVolumeHead = -1 },
			wantErr: "DailyVolumeHead out of range",
		},
		{
			name:    "DailyVolumeHead too high",
			mutate:  func(s *GeneratorState) { s.DailyVolumeHead = WarmupDays },
			wantErr: "DailyVolumeHead out of range",
		},
		{
			name:    "negative CurrentThreshold",
			mutate:  func(s *GeneratorState) { s.CurrentThreshold = -1 },
			wantErr: "CurrentThreshold is negative",
		},
		{
			name:    "negative Accumulator TickCount",
			mutate:  func(s *GeneratorState) { s.Accumulator.TickCount = -5 },
			wantErr: "Accumulator.TickCount is negative",
		},
		{
			name:    "High < Low in Accumulator",
			mutate:  func(s *GeneratorState) { s.Accumulator.TickCount = 10; s.Accumulator.High = 100; s.Accumulator.Low = 200 },
			wantErr: "High < Low",
		},
		{
			name:    "negative LastProcessedTradeID",
			mutate:  func(s *GeneratorState) { s.LastProcessedTradeID = -1 },
			wantErr: "LastProcessedTradeID is negative",
		},
		{
			name:    "negative EMAVolume",
			mutate:  func(s *GeneratorState) { s.EMAVolume = -1 },
			wantErr: "EMAVolume is negative",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Copy valid state
			state := validState
			state.Accumulator = validState.Accumulator

			// Apply mutation
			tc.mutate(&state)

			err := state.Validate()
			if err == nil {
				t.Errorf("Expected validation error for %s", tc.name)
				return
			}

			if !containsStr(err.Error(), tc.wantErr) {
				t.Errorf("Expected error containing '%s', got: %v", tc.wantErr, err)
			}
		})
	}
}

func containsStr(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestLoadCorruptedStateIsRejected(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "bars_state_corrupted")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create corrupted state with invalid values
	corruptedState := GeneratorState{
		DailyVolumeCount:     -999, // Invalid
		CurrentThreshold:     -1,   // Invalid
		LastProcessedTradeID: -1,   // Invalid
	}

	// Save corrupted state directly (bypassing validation)
	statePath := tmpDir + "/generator_state.json"
	data, _ := json.Marshal(corruptedState)
	os.WriteFile(statePath, data, 0644)

	// Load should detect corruption and return error
	_, err = LoadState(tmpDir)
	if err == nil {
		t.Fatal("Should reject corrupted state")
	}

	t.Logf("Corrupted state correctly rejected: %v", err)
}

func TestStateRoundTrip(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "bars_state_roundtrip")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create a generator and modify its state
	g := NewGenerator()

	// Simulate some processing
	for i := 0; i < WarmupDays; i++ {
		g.dailyVolumes[i] = float64((i + 1) * 1_000_000_000)
	}
	g.dailyVolumeCount = WarmupDays
	g.dailyVolumeHead = 0
	g.isWarmup = false
	g.initializeEMA() // Initialize EMA with SMA of daily volumes
	g.currentThreshold = g.calculateThreshold()
	g.currentDayStart = 1600000000000
	g.currentDayVolume = 250_000_000

	// Add some accumulator state
	g.acc.AddTrade(50000, 1.0, 50000, 1600000001000, true)
	g.acc.AddTrade(50100, 0.5, 25050, 1600000002000, false)

	// Set last processed trade ID for crash recovery
	g.lastProcessedTradeID = 987654321

	// Export and save
	state := g.ExportState()
	if err := SaveState(tmpDir, state); err != nil {
		t.Fatalf("SaveState failed: %v", err)
	}

	// Load and import to new generator
	loaded, err := LoadState(tmpDir)
	if err != nil {
		t.Fatalf("LoadState failed: %v", err)
	}

	g2 := NewGenerator()
	g2.ImportState(*loaded)

	// Verify state matches
	if g2.isWarmup != g.isWarmup {
		t.Errorf("Warmup mismatch: %v vs %v", g2.isWarmup, g.isWarmup)
	}
	if g2.emaVolume != g.emaVolume {
		t.Errorf("EMAVolume mismatch: %f vs %f", g2.emaVolume, g.emaVolume)
	}
	if g2.currentThreshold != g.currentThreshold {
		t.Errorf("Threshold mismatch: %f vs %f", g2.currentThreshold, g.currentThreshold)
	}
	if g2.acc.TickCount != g.acc.TickCount {
		t.Errorf("TickCount mismatch: %d vs %d", g2.acc.TickCount, g.acc.TickCount)
	}
	if g2.acc.DollarValue != g.acc.DollarValue {
		t.Errorf("DollarValue mismatch: %f vs %f", g2.acc.DollarValue, g.acc.DollarValue)
	}
	if g2.lastProcessedTradeID != g.lastProcessedTradeID {
		t.Errorf("LastProcessedTradeID mismatch: %d vs %d", g2.lastProcessedTradeID, g.lastProcessedTradeID)
	}
}
